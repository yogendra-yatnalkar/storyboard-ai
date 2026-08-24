/**
 * Fixture providers: the whole pipeline, offline.
 *
 * These are not throwaway test doubles. They are how the pipeline runs in CI and
 * on a laptop with no API keys, and how the renderer gets exercised end to end
 * without spending money on every commit. They reuse the repo's own Shepherd Boy
 * assets, so a fixture run produces a real video rather than placeholder frames.
 *
 * They report the same UsageRecords as the paid providers, with the same prices,
 * so a fixture run still tells you what the video WOULD have cost.
 */

import { copyFile, mkdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

import type {
  ImageProvider,
  LlmProvider,
  ProviderResult,
  SegmentationProvider,
  TtsProvider,
} from "./index";
import { MEDIA_PRICING } from "./media";
import { DEEPSEEK_PRICING } from "./deepseek";
import type {
  GeneratedAudio,
  GeneratedImage,
  GeneratedMasks,
  GenerationRequest,
  Scene,
  Storyboard,
} from "../types";

export interface FixtureOptions {
  /** Repository root, used to find the Shepherd Boy assets. */
  repoRoot: string;
}

const STORY_SCENES = [
  {
    imagePrompt:
      "A bored shepherd boy on a green hillside with sheep grazing, clear outlines, plain background",
    narration:
      'Once, there was a boy who became bored when he watched over the village sheep grazing on the hillside. To entertain himself, he sang out, "Wolf! Wolf! The wolf is chasing the sheep! When the villagers heard the cry, they ran up the hill to drive the wolf away."',
  },
  {
    imagePrompt:
      "Angry villagers confronting the shepherd boy on the hillside, clear outlines, plain background",
    narration:
      'But when they arrived, they saw no wolf. The boy was amused when he saw their angry faces. "Don\'t scream wolf when there is no wolf, boy!" the villagers warned. They angrily went back down the hill. Later, the shepherd boy cried out once again, "Wolf! Wolf! The wolf is chasing the sheep!" To his amusement, the villagers came running up the hill to scare the wolf away. As they saw there was no wolf, they said strictly, "Save your frightened cry for when there really is a wolf! Don\'t cry \'wolf\' when there is no wolf!" But the boy grinned at their words while they walked, grumbling down the hill once more.',
  },
  {
    imagePrompt: "A real wolf approaching frightened sheep at dusk, clear outlines, plain background",
    narration:
      "The last time, a real wolf came and wanted to eat his sheep. The boy said help, help, but the people did not listen to him. They thought he was lying again. They did not help him and he lost his sheep.",
  },
  {
    imagePrompt:
      "An old man speaking kindly to the shepherd boy at sunset, clear outlines, plain background",
    narration:
      "The story ends with an old man telling the boy that lying is bad and people will not trust you if you lie. The Moral of the story is - Lying breaks trust, and no one believes a liar - even when they tell the truth.",
  },
];

/** Deterministic storyboard, priced as though DeepSeek had produced it. */
export class FixtureLlmProvider implements LlmProvider {
  readonly name = "fixture-llm";

  async storyboard(request: GenerationRequest): Promise<ProviderResult<Storyboard>> {
    const scenes: Scene[] = STORY_SCENES.slice(0, request.sceneCount).map((scene, index) => ({
      index,
      ...scene,
    }));

    // Token counts from a representative live call, so cost reporting stays
    // meaningful offline.
    const promptTokens = 420 + Math.ceil(request.input.length / 4);
    const completionTokens = scenes.reduce(
      (sum, scene) => sum + Math.ceil((scene.narration.length + scene.imagePrompt.length) / 4),
      0,
    );

    return {
      value: { title: "The Shepherd Boy and the Wolf", scenes },
      usage: [
        {
          provider: this.name,
          operation: "storyboard",
          cost:
            (promptTokens / 1e6) * DEEPSEEK_PRICING.inputPerMillion +
            (completionTokens / 1e6) * DEEPSEEK_PRICING.outputPerMillion,
          quantity: promptTokens + completionTokens,
          unit: "tokens",
        },
      ],
    };
  }
}

export class FixtureImageProvider implements ImageProvider {
  readonly name = "fixture-image";

  constructor(private readonly options: FixtureOptions) {}

  async generate(
    scene: Scene,
    options: { aspect: GenerationRequest["aspect"]; tier: GenerationRequest["tier"]; outDir: string },
  ): Promise<ProviderResult<GeneratedImage>> {
    const source = resolve(
      this.options.repoRoot,
      `generate-whiteboard-animated-videos/images/${scene.index + 1}.png`,
    );
    if (!existsSync(source)) {
      throw new Error(`fixture image missing for scene ${scene.index + 1}: ${source}`);
    }

    await mkdir(options.outDir, { recursive: true });
    const path = join(options.outDir, `scene-${scene.index + 1}.png`);
    await copyFile(source, path);

    return {
      value: { sceneIndex: scene.index, path, width: 1024, height: 1024 },
      usage: [
        {
          provider: this.name,
          operation: "image",
          cost: MEDIA_PRICING.image[options.tier],
          quantity: 1,
          unit: "images",
        },
      ],
    };
  }
}

export class FixtureTtsProvider implements TtsProvider {
  readonly name = "fixture-tts";
  /** gTTS returns no timings, which is exactly why the Whisper path exists. */
  readonly providesWordTimings = false;

  constructor(private readonly options: FixtureOptions) {}

  async speak(
    scene: Scene,
    options: { voice?: string; language: string; outDir: string },
  ): Promise<ProviderResult<GeneratedAudio>> {
    const source = resolve(this.options.repoRoot, `tts/saved_audios/${scene.index + 1}.mp3`);
    if (!existsSync(source)) {
      throw new Error(`fixture audio missing for scene ${scene.index + 1}: ${source}`);
    }

    await mkdir(options.outDir, { recursive: true });
    const path = join(options.outDir, `scene-${scene.index + 1}.mp3`);
    await copyFile(source, path);

    const { parseFile } = await import("music-metadata");
    const durationSeconds = (await parseFile(path)).format.duration ?? 0;

    return {
      value: { sceneIndex: scene.index, path, durationSeconds },
      usage: [
        {
          provider: this.name,
          operation: "tts",
          cost: (scene.narration.length / 1000) * MEDIA_PRICING.ttsPerThousandChars.cheap,
          quantity: scene.narration.length,
          unit: "chars",
        },
      ],
    };
  }
}

export class FixtureSegmentationProvider implements SegmentationProvider {
  readonly name = "fixture-segmentation";

  constructor(private readonly options: FixtureOptions) {}

  async segment(
    image: GeneratedImage,
    options: { outDir: string },
  ): Promise<ProviderResult<GeneratedMasks>> {
    const source = resolve(
      this.options.repoRoot,
      `generate-whiteboard-animated-videos/images/${image.sceneIndex + 1}.json`,
    );
    if (!existsSync(source)) {
      return { value: { sceneIndex: image.sceneIndex, path: null }, usage: [] };
    }

    await mkdir(options.outDir, { recursive: true });
    const path = join(options.outDir, `scene-${image.sceneIndex + 1}.json`);
    // Re-serialised rather than copied: the committed files embed a base64 copy
    // of the source image, which would bloat every generated project.
    const parsed = JSON.parse(await readFile(source, "utf8")) as Record<string, unknown>;
    delete parsed.imageData;
    await copyFileJson(path, parsed);

    return {
      value: { sceneIndex: image.sceneIndex, path },
      usage: [
        {
          provider: this.name,
          operation: "segmentation",
          cost: MEDIA_PRICING.segmentationPerImage,
          quantity: 1,
          unit: "images",
        },
      ],
    };
  }
}

async function copyFileJson(path: string, value: unknown) {
  const { writeFile } = await import("node:fs/promises");
  await writeFile(path, JSON.stringify(value));
}
