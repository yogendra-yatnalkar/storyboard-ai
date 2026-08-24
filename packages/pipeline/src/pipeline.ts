/**
 * Generation pipeline: story text in, a render-ready fixture out.
 *
 * The output is deliberately the same fixture shape `buildVideo.ts` already
 * consumes, so generation and rendering stay decoupled: the renderer does not
 * know which providers produced its inputs, and the pipeline does not know how
 * the video will be drawn.
 *
 * Scene work runs concurrently because image, speech and segmentation for scene
 * 3 have nothing to do with scene 1 - on a five-scene video that is the
 * difference between a serial minute and a parallel fifteen seconds.
 */

import { mkdir, writeFile } from "node:fs/promises";
import { join, relative } from "node:path";

import type { ProviderSet } from "./providers";
import { summariseCost, type CostBreakdown } from "./cost";
import type { GenerationRequest, Storyboard, UsageRecord } from "./types";

export interface PipelineOptions {
  request: GenerationRequest;
  providers: ProviderSet;
  /** Directory for generated assets and the fixture. */
  outDir: string;
  onProgress?: (stage: string, detail?: string) => void;
}

export interface PipelineResult {
  storyboard: Storyboard;
  /** Path to the fixture buildVideo.ts consumes. */
  fixturePath: string;
  cost: CostBreakdown;
  scenes: {
    index: number;
    image: string;
    mask: string | null;
    audio: string;
    durationSeconds: number;
    narration: string;
  }[];
}

export async function runPipeline(options: PipelineOptions): Promise<PipelineResult> {
  const { request, providers, outDir, onProgress = () => {} } = options;
  const usage: UsageRecord[] = [];

  await mkdir(outDir, { recursive: true });

  // --- 1. storyboard --------------------------------------------------------
  onProgress("storyboard", providers.llm.name);
  const storyboardResult = await providers.llm.storyboard(request);
  usage.push(...storyboardResult.usage);
  const storyboard = storyboardResult.value;
  onProgress("storyboard", `${storyboard.scenes.length} scenes`);

  // --- 2. per-scene assets --------------------------------------------------
  const scenes = await Promise.all(
    storyboard.scenes.map(async (scene) => {
      // Image first: segmentation needs it, speech does not, so speech overlaps.
      const [imageResult, audioResult] = await Promise.all([
        providers.image.generate(scene, {
          aspect: request.aspect,
          tier: request.tier,
          outDir,
        }),
        providers.tts.speak(scene, {
          voice: request.voice,
          language: request.language,
          outDir,
        }),
      ]);
      onProgress("scene", `${scene.index + 1}: image + narration`);

      let maskPath: string | null = null;
      if (request.segmentation && providers.segmentation) {
        const maskResult = await providers.segmentation.segment(imageResult.value, { outDir });
        usage.push(...maskResult.usage);
        maskPath = maskResult.value.path;
        onProgress("scene", `${scene.index + 1}: masks`);
      }

      usage.push(...imageResult.usage, ...audioResult.usage);

      return {
        index: scene.index,
        image: imageResult.value.path,
        mask: maskPath,
        audio: audioResult.value.path,
        durationSeconds: audioResult.value.durationSeconds,
        narration: scene.narration,
        wordTimings: audioResult.value.wordTimings,
      };
    }),
  );

  scenes.sort((a, b) => a.index - b.index);

  // --- 3. fixture -----------------------------------------------------------
  // Paths are relative so the fixture stays portable between machines and the
  // render container.
  const fixture = {
    title: storyboard.title,
    language: request.language,
    // Vertical is short-form, where word highlighting is the convention.
    captionStyle: request.aspect === "portrait" ? "word" : "sentence",
    watermark: false,
    music: null,
    brand: {},
    scenes: scenes.map((scene) => ({
      image: `./${relative(outDir, scene.image)}`,
      mask: scene.mask ? `./${relative(outDir, scene.mask)}` : null,
      audio: `./${relative(outDir, scene.audio)}`,
      text: scene.narration,
      // Carried through when the TTS supplied them, so the renderer can skip
      // transcription entirely.
      wordTimings: scene.wordTimings ?? null,
    })),
  };

  const fixturePath = join(outDir, "fixture.json");
  await writeFile(fixturePath, JSON.stringify(fixture, null, 2));

  const totalNarration = scenes.reduce((sum, scene) => sum + scene.durationSeconds, 0);
  onProgress("done", `${totalNarration.toFixed(1)}s of narration`);

  return {
    storyboard,
    fixturePath,
    cost: summariseCost(usage),
    scenes: scenes.map(({ wordTimings, ...rest }) => rest),
  };
}
