/**
 * Provider interfaces.
 *
 * Every external model sits behind one of these, for three reasons that all
 * showed up in planning:
 *
 *  - the cheap-to-premium upsell has to be a config change, not a code path;
 *  - DeepSeek is the default LLM and its API is China-hosted, so EU/UK data
 *    residency may force a swap without touching the pipeline;
 *  - the pipeline must run offline in CI, which the fixture providers give us.
 *
 * Each method reports UsageRecords so per-video cost is measured rather than
 * estimated - the margin analysis in docs/cost-model.md depends on real numbers
 * once keys exist.
 */

import type {
  GeneratedAudio,
  GeneratedImage,
  GeneratedMasks,
  GenerationRequest,
  Scene,
  Storyboard,
  UsageRecord,
} from "../types";

export interface ProviderResult<T> {
  value: T;
  usage: UsageRecord[];
}

export interface LlmProvider {
  readonly name: string;
  /** Split a story into scenes with image prompts and narration. */
  storyboard(request: GenerationRequest): Promise<ProviderResult<Storyboard>>;
}

export interface ImageProvider {
  readonly name: string;
  generate(
    scene: Scene,
    options: { aspect: GenerationRequest["aspect"]; tier: GenerationRequest["tier"]; outDir: string },
  ): Promise<ProviderResult<GeneratedImage>>;
}

export interface TtsProvider {
  readonly name: string;
  /** Whether this provider returns word timings, which saves a transcription pass. */
  readonly providesWordTimings: boolean;
  speak(
    scene: Scene,
    options: { voice?: string; language: string; outDir: string },
  ): Promise<ProviderResult<GeneratedAudio>>;
}

export interface SegmentationProvider {
  readonly name: string;
  segment(
    image: GeneratedImage,
    options: { outDir: string },
  ): Promise<ProviderResult<GeneratedMasks>>;
}

export interface ProviderSet {
  llm: LlmProvider;
  image: ImageProvider;
  tts: TtsProvider;
  segmentation: SegmentationProvider | null;
}

/** Aspect ratio to pixel dimensions for image generation. */
export const IMAGE_SIZES = {
  square: { width: 1024, height: 1024 },
  landscape: { width: 1344, height: 768 },
  portrait: { width: 768, height: 1344 },
} as const;
