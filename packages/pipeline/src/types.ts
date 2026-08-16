/** Domain types shared across the generation pipeline. */

export type AspectRatio = "square" | "landscape" | "portrait";

export type QualityTier = "cheap" | "premium";

export interface GenerationRequest {
  /** The story or topic the creator typed. */
  input: string;
  aspect: AspectRatio;
  tier: QualityTier;
  language: string;
  /** Target scene count; the LLM may return fewer if the story is short. */
  sceneCount: number;
  /** Object masking is the differentiator, but it is also ~25% of cheap-tier COGS. */
  segmentation: boolean;
  voice?: string;
}

/** One beat of the story: what is shown, what is said. */
export interface Scene {
  index: number;
  /** Prompt handed to the image model. */
  imagePrompt: string;
  /** Narration text, read aloud and captioned. */
  narration: string;
}

export interface Storyboard {
  title: string;
  scenes: Scene[];
}

export interface GeneratedImage {
  sceneIndex: number;
  /** Absolute path on disk. */
  path: string;
  width: number;
  height: number;
}

export interface GeneratedAudio {
  sceneIndex: number;
  path: string;
  durationSeconds: number;
  /** Word timings, when the provider returns them. */
  wordTimings?: { text: string; offsetMs: number; durationMs: number }[];
}

export interface GeneratedMasks {
  sceneIndex: number;
  /** labelme/AnyLabeling-shaped JSON path, or null when segmentation is off. */
  path: string | null;
}

/** What every provider call reports so cost can be attributed per video. */
export interface UsageRecord {
  provider: string;
  operation: string;
  /** USD. */
  cost: number;
  quantity?: number;
  unit?: string;
}
