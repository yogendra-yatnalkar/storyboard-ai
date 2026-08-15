/** Shape of the precomputed drawing manifest, shared by the scripts and the composition. */

import type { Tile } from "./tilePath";

/** Mirrors RenderConfig in prototype/reference/render.py. */
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

export type RenderConfig = typeof CONFIG;

export interface SegmentManifest {
  label: string;
  kind: "object" | "background" | "full";
  splitLen: number;
  skipRate: number;
  tiles: Tile[];
  frameCount: number;
  /** Frame index, within the whole composition, where this segment starts. */
  startFrame: number;
}

export interface PathManifest {
  image: string;
  mask: string | null;
  config: RenderConfig;
  width: number;
  height: number;
  durationInFrames: number;
  segments: SegmentManifest[];
}
