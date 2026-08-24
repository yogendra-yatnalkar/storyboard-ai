import assert from "node:assert/strict";
import { mkdtemp, readFile, rm } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { after, describe, it } from "node:test";

import { runPipeline } from "./pipeline";
import {
  FixtureImageProvider,
  FixtureLlmProvider,
  FixtureSegmentationProvider,
  FixtureTtsProvider,
} from "./providers/fixtures";
import { DeepSeekProvider } from "./providers/deepseek";
import { buildImagePrompt, polygonArea } from "./providers/media";
import type { GenerationRequest } from "./types";

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../../..");
const fixtureOptions = { repoRoot: REPO_ROOT };

const request: GenerationRequest = {
  input: "The boy who cried wolf",
  aspect: "portrait",
  tier: "cheap",
  language: "en",
  sceneCount: 2,
  segmentation: true,
};

const tempDirs: string[] = [];
async function tempDir() {
  const dir = await mkdtemp(join(tmpdir(), "pipeline-test-"));
  tempDirs.push(dir);
  return dir;
}

after(async () => {
  await Promise.all(tempDirs.map((dir) => rm(dir, { recursive: true, force: true })));
});

describe("runPipeline (fixture providers)", () => {
  it("produces a fixture the renderer can consume", async () => {
    const outDir = await tempDir();
    const providers = {
      llm: new FixtureLlmProvider(),
      image: new FixtureImageProvider(fixtureOptions),
      tts: new FixtureTtsProvider(fixtureOptions),
      segmentation: new FixtureSegmentationProvider(fixtureOptions),
    };

    const result = await runPipeline({ request, providers, outDir });
    const fixture = JSON.parse(await readFile(result.fixturePath, "utf8"));

    assert.equal(fixture.scenes.length, 2);
    assert.equal(fixture.captionStyle, "word", "vertical output should use word captions");
    for (const scene of fixture.scenes) {
      // Relative so the fixture survives being moved between machines.
      assert.ok(scene.image.startsWith("./"), "asset paths must be relative");
      assert.ok(scene.text.length > 0, "every scene needs narration to speak");
    }
  });

  it("measures a real per-video cost", async () => {
    const outDir = await tempDir();
    const result = await runPipeline({
      request: { ...request, sceneCount: 4 },
      providers: {
        llm: new FixtureLlmProvider(),
        image: new FixtureImageProvider(fixtureOptions),
        tts: new FixtureTtsProvider(fixtureOptions),
        segmentation: new FixtureSegmentationProvider(fixtureOptions),
      },
      outDir,
    });

    // Sanity-check the estimate in docs/cost-model.md rather than restating it:
    // a four-scene cheap-tier video should land near $0.15 before render.
    assert.ok(result.cost.total > 0.1 && result.cost.total < 0.25, `got ${result.cost.total}`);
    assert.ok(result.cost.byOperation.image > result.cost.byOperation.storyboard);
  });

  it("skips segmentation when the project turns it off", async () => {
    const outDir = await tempDir();
    const result = await runPipeline({
      request: { ...request, segmentation: false },
      providers: {
        llm: new FixtureLlmProvider(),
        image: new FixtureImageProvider(fixtureOptions),
        tts: new FixtureTtsProvider(fixtureOptions),
        segmentation: new FixtureSegmentationProvider(fixtureOptions),
      },
      outDir,
    });

    assert.ok(result.scenes.every((scene) => scene.mask === null));
    assert.equal(result.cost.byOperation.segmentation, undefined);
  });

  it("keeps scenes in order despite running them concurrently", async () => {
    const outDir = await tempDir();
    const result = await runPipeline({
      request: { ...request, sceneCount: 4 },
      providers: {
        llm: new FixtureLlmProvider(),
        image: new FixtureImageProvider(fixtureOptions),
        tts: new FixtureTtsProvider(fixtureOptions),
        segmentation: null,
      },
      outDir,
    });

    assert.deepEqual(
      result.scenes.map((scene) => scene.index),
      [0, 1, 2, 3],
    );
  });
});

describe("DeepSeekProvider", () => {
  const chatResponse = (content: string) => ({
    ok: true,
    status: 200,
    json: async () => ({
      choices: [{ message: { content } }],
      usage: { prompt_tokens: 500, completion_tokens: 800 },
    }),
    text: async () => "",
  });

  it("parses a storyboard and prices the call", async () => {
    const provider = new DeepSeekProvider({
      apiKey: "test",
      fetchImpl: (async () =>
        chatResponse(
          JSON.stringify({
            title: "A Tale",
            scenes: [{ imagePrompt: "a hill", narration: "Once upon a time." }],
          }),
        )) as unknown as typeof fetch,
    });

    const result = await provider.storyboard(request);

    assert.equal(result.value.title, "A Tale");
    assert.equal(result.value.scenes[0].index, 0);
    // 500 in + 800 out at DeepSeek list prices is a fraction of a cent.
    assert.ok(result.usage[0].cost > 0 && result.usage[0].cost < 0.002);
  });

  it("drops scenes missing narration or a prompt", async () => {
    const provider = new DeepSeekProvider({
      apiKey: "test",
      fetchImpl: (async () =>
        chatResponse(
          JSON.stringify({
            scenes: [
              { imagePrompt: "a hill", narration: "Good." },
              { imagePrompt: "no narration" },
            ],
          }),
        )) as unknown as typeof fetch,
    });

    const result = await provider.storyboard(request);
    assert.equal(result.value.scenes.length, 1);
  });

  it("retries a rate limit and then succeeds", async () => {
    let calls = 0;
    const provider = new DeepSeekProvider({
      apiKey: "test",
      maxRetries: 3,
      fetchImpl: (async () => {
        calls++;
        if (calls < 3) return { ok: false, status: 429, text: async () => "rate limited" };
        return chatResponse(
          JSON.stringify({ scenes: [{ imagePrompt: "x", narration: "y" }] }),
        );
      }) as unknown as typeof fetch,
    });

    const result = await provider.storyboard(request);
    assert.equal(calls, 3);
    assert.equal(result.value.scenes.length, 1);
  });

  it("fails loudly when the model returns nothing usable", async () => {
    const provider = new DeepSeekProvider({
      apiKey: "test",
      maxRetries: 1,
      fetchImpl: (async () => chatResponse(JSON.stringify({ scenes: [] }))) as unknown as typeof fetch,
    });

    await assert.rejects(() => provider.storyboard(request), /no usable scenes/);
  });
});

describe("image prompting", () => {
  it("steers the model toward what the renderer can threshold", () => {
    const prompt = buildImagePrompt("a wolf on a hill");

    assert.match(prompt, /a wolf on a hill/);
    // The renderer turns the image into line art, so these matter more than style.
    assert.match(prompt, /clear outlines|clean illustration/i);
    assert.match(prompt, /no text/i);
  });
});

describe("polygonArea", () => {
  it("measures a square", () => {
    assert.equal(
      polygonArea([
        [0, 0],
        [10, 0],
        [10, 10],
        [0, 10],
      ]),
      100,
    );
  });

  it("ignores winding direction", () => {
    const clockwise: [number, number][] = [
      [0, 0],
      [0, 10],
      [10, 10],
      [10, 0],
    ];
    assert.equal(polygonArea(clockwise), 100);
  });
});
