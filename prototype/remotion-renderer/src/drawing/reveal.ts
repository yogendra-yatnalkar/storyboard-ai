/**
 * The reveal step: frame index -> pixels.
 *
 * This is deliberately a pure function with no memory of previous frames. That is
 * the whole point of the precompute/reveal split - it is what lets the Remotion
 * Player scrub instantly, lets Lambda render disjoint frame ranges in parallel,
 * and lets a re-render skip straight to the frame it needs.
 *
 * The same function backs both the React composition and the parity harness, so
 * what gets measured against the Python reference is exactly what gets rendered.
 */

import type { PathManifest, SegmentManifest } from "./manifest";

export interface Sprite {
  data: Uint8Array;
  width: number;
  height: number;
}

export interface RevealAssets {
  /** Canvas state entering each segment, RGBA, one per segment. */
  cumulative: Uint8Array[];
  /** Each segment's masked threshold, grayscale, one per segment. */
  ink: Uint8Array[];
  /** Resized colour image, RGBA - shown during the end hold. */
  source: Uint8Array;
  hand: Sprite;
}

export type FrameState =
  | { kind: "hold" }
  | {
      kind: "drawing";
      segmentIndex: number;
      segment: SegmentManifest;
      tilesDrawn: number;
      handTileIndex: number;
    };

/**
 * Map an absolute frame index onto its drawing state.
 *
 * Frame f inside a segment corresponds to the reference's counter
 * (f + 1) * skipRate: tiles [0, counter) are on the canvas and the hand rests on
 * tile counter - 1.
 */
export function resolveFrame(manifest: PathManifest, frame: number): FrameState {
  for (let index = 0; index < manifest.segments.length; index++) {
    const segment = manifest.segments[index];
    const frameInSegment = frame - segment.startFrame;
    if (frameInSegment >= 0 && frameInSegment < segment.frameCount) {
      const tilesDrawn = (frameInSegment + 1) * segment.skipRate;
      return {
        kind: "drawing",
        segmentIndex: index,
        segment,
        tilesDrawn,
        handTileIndex: tilesDrawn - 1,
      };
    }
  }
  return { kind: "hold" };
}

/** Paint tiles [0, tilesDrawn) of a segment onto an RGBA canvas. */
function paintTiles(
  out: Uint8ClampedArray,
  ink: Uint8Array,
  segment: SegmentManifest,
  tilesDrawn: number,
  width: number,
) {
  const { splitLen, tiles } = segment;
  const limit = Math.min(tilesDrawn, tiles.length);

  for (let i = 0; i < limit; i++) {
    const [row, col] = tiles[i];
    const yStart = row * splitLen;
    const xStart = col * splitLen;
    for (let y = yStart; y < yStart + splitLen; y++) {
      let pixel = y * width + xStart;
      for (let x = 0; x < splitLen; x++, pixel++) {
        const value = ink[pixel];
        out[pixel * 4] = value;
        out[pixel * 4 + 1] = value;
        out[pixel * 4 + 2] = value;
      }
    }
  }
}

/**
 * Composite the hand with its top-left corner at (x, y), clipped at the canvas
 * edges. The mask is binary, so this is a straight copy of opaque pixels.
 */
function paintHand(
  out: Uint8ClampedArray,
  hand: Sprite,
  x: number,
  y: number,
  width: number,
  height: number,
) {
  const cropWidth = Math.min(hand.width, width - x);
  const cropHeight = Math.min(hand.height, height - y);

  for (let row = 0; row < cropHeight; row++) {
    for (let col = 0; col < cropWidth; col++) {
      const src = (row * hand.width + col) * 4;
      if (hand.data[src + 3] === 0) continue;
      const dst = ((y + row) * width + (x + col)) * 4;
      out[dst] = hand.data[src];
      out[dst + 1] = hand.data[src + 1];
      out[dst + 2] = hand.data[src + 2];
    }
  }
}

/** Render a single frame as RGBA pixels. */
export function renderFrame(
  manifest: PathManifest,
  assets: RevealAssets,
  frame: number,
): Uint8ClampedArray {
  const { width, height } = manifest;
  const state = resolveFrame(manifest, frame);

  if (state.kind === "hold") {
    return new Uint8ClampedArray(assets.source);
  }

  const out = new Uint8ClampedArray(assets.cumulative[state.segmentIndex]);
  paintTiles(out, assets.ink[state.segmentIndex], state.segment, state.tilesDrawn, width);

  const handTile = state.segment.tiles[state.handTileIndex];
  if (handTile) {
    const [row, col] = handTile;
    const splitLen = state.segment.splitLen;
    paintHand(
      out,
      assets.hand,
      col * splitLen + Math.floor(splitLen / 2),
      row * splitLen + Math.floor(splitLen / 2),
      width,
      height,
    );
  }

  return out;
}
