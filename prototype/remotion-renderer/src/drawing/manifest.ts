/** Manifest shapes shared by the build scripts and the compositions. */

import type { Tile } from "./tilePath";

/**
 * Defaults mirror RenderConfig in prototype/reference/render.py.
 *
 * `width`/`height` are the parity fixture's dimensions, not a constraint - the
 * product generates artwork at the project's aspect ratio and overrides them.
 * Both must be multiples of every `splitLen` in use; see assertGridFits().
 */
export const CONFIG = {
  frameRate: 25,
  width: 1020,
  height: 1020,
  splitLen: 10,
  objectSkipRate: 8,
  backgroundSplitLen: 20,
  backgroundSkipRate: 14,
  endHoldSeconds: 3,
  blackPixelThreshold: 10,
  noMaskSplitLen: 15,
  noMaskSkipRate: 8,
} as const;

/** Width and height are per-project; everything else is a tuning constant. */
export type RenderConfig = Omit<typeof CONFIG, "width" | "height"> & {
  width: number;
  height: number;
};

export const ASPECT_PRESETS = {
  square: { width: 1020, height: 1020 },
  landscape: { width: 1920, height: 1080 },
  portrait: { width: 1080, height: 1920 },
} as const;

export type AspectPreset = keyof typeof ASPECT_PRESETS;

export interface SegmentManifest {
  label: string;
  kind: "object" | "background" | "full";
  splitLen: number;
  skipRate: number;
  tiles: Tile[];
  frameCount: number;
  /** Frame index, within the scene, where this segment starts. */
  startFrame: number;
}

export interface PathManifest {
  image: string;
  mask: string | null;
  config: RenderConfig;
  width: number;
  height: number;
  durationInFrames: number;
  /** Narration length this scene was timed against, if any. */
  narrationSeconds: number | null;
  segments: SegmentManifest[];
}

// --- video (multi-scene) ------------------------------------------------------

export type CaptionStyle = "word" | "sentence" | "none";

export interface CaptionWord {
  text: string;
  startMs: number;
  endMs: number;
}

export interface SceneManifest {
  /** Folder under public/precomputed holding this scene's path.json and layers. */
  name: string;
  /** Audio file, relative to public/. */
  audio: string | null;
  captions: CaptionWord[];
  durationInFrames: number;
  /** Frame index, within the whole video, where this scene starts. */
  startFrame: number;
}

export interface BrandKit {
  /** Logo image relative to public/, drawn as a corner mark. */
  logo?: string;
  accentColor?: string;
  captionFont?: string;
  /** Paper colour behind the drawing. */
  background?: string;
}

export interface VideoManifest {
  title: string;
  width: number;
  height: number;
  fps: number;
  captionStyle: CaptionStyle;
  /** Free-tier renders carry a watermark. */
  watermark: boolean;
  /** Background music relative to public/, ducked under narration. */
  music: string | null;
  musicGainDb: number;
  brand: BrandKit;
  language: string;
  durationInFrames: number;
  scenes: SceneManifest[];
}

/**
 * The tile grid slices the image with reshape/exact division, so a dimension that
 * is not a multiple of splitLen fails obscurely in numpy and writes out of bounds
 * in TypeScript. Round render dimensions with this before using them.
 */
export function roundToGrid(value: number, splitLens: number[]): number {
  const step = splitLens.reduce((a, b) => (a * b) / gcd(a, b), 1);
  return Math.max(step, Math.floor(value / step) * step);
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

export function assertGridFits(width: number, height: number, splitLens: number[]): void {
  for (const splitLen of splitLens) {
    if (width % splitLen !== 0 || height % splitLen !== 0) {
      throw new Error(
        `render size ${width}x${height} is not a multiple of splitLen ${splitLen}; ` +
          `use roundToGrid() when choosing dimensions`,
      );
    }
  }
}
