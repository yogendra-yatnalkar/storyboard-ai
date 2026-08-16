/**
 * Narration-driven timing.
 *
 * The original pipeline fixed the drawing speed and then reconciled audio to
 * video by hand, with per-scene `apad=pad_dur=1.19` corrections typed into
 * ffmpeg-command-notes.txt. This inverts that: narration duration is the input,
 * and the drawing speed is solved for so the hand finishes as the voice does.
 *
 * The reference draws the background faster than objects (skip rates 14 vs 8).
 * That ratio is preserved as a weight rather than a constant, so a scene keeps
 * its "objects deliberate, background brisk" character at any narration length.
 */

/** Relative drawing speed per segment kind - higher means faster. */
export const SPEED_WEIGHTS = {
  object: 1,
  background: 14 / 8,
  full: 1,
} as const;

export type SegmentKind = keyof typeof SPEED_WEIGHTS;

export interface TimingBounds {
  /** Never emit a frame per tile faster than this (1 = every tile is a frame). */
  minSkipRate: number;
  /**
   * Upper bound on tiles-per-frame. Without it, a dense image with short
   * narration strobes: hundreds of tiles appearing between consecutive frames
   * reads as a jump cut rather than as drawing.
   */
  maxSkipRate: number;
}

export const DEFAULT_BOUNDS: TimingBounds = { minSkipRate: 1, maxSkipRate: 30 };

export interface SegmentTiming {
  kind: SegmentKind;
  tileCount: number;
  skipRate: number;
  frameCount: number;
}

export interface SceneTiming {
  segments: SegmentTiming[];
  /** Frames spent drawing. */
  drawingFrames: number;
  /** Frames holding the finished colour image. */
  holdFrames: number;
  durationInFrames: number;
  /** True when the drawing could not fit the narration and the scene was extended. */
  clamped: boolean;
}

/**
 * Solve for the drawing speed that makes a scene last as long as its narration.
 *
 * Frames for a segment are `tiles / skipRate`, and `skipRate = base * weight`, so
 * the total is `(1 / base) * sum(tiles / weight)` - which inverts directly.
 *
 * Pass `narrationSeconds: null` to keep the reference's fixed rates, which is how
 * the parity fixtures stay bit-identical.
 */
export function computeSceneTiming(
  segments: { kind: SegmentKind; tileCount: number; fixedSkipRate: number }[],
  options: {
    fps: number;
    endHoldSeconds: number;
    narrationSeconds: number | null;
    bounds?: TimingBounds;
  },
): SceneTiming {
  const { fps, endHoldSeconds, narrationSeconds } = options;
  const bounds = options.bounds ?? DEFAULT_BOUNDS;
  const endHoldFrames = Math.round(endHoldSeconds * fps);

  let rates: number[];

  if (narrationSeconds === null) {
    rates = segments.map((segment) => segment.fixedSkipRate);
  } else {
    const targetFrames = Math.round(narrationSeconds * fps);
    const drawingBudget = Math.max(1, targetFrames - endHoldFrames);
    const weightedTiles = segments.reduce(
      (sum, segment) => sum + segment.tileCount / SPEED_WEIGHTS[segment.kind],
      0,
    );

    const base = weightedTiles / drawingBudget;
    // Deliberately NOT rounded to an integer. At the low rates a long narration
    // produces (2.25 tiles/frame, say), rounding to 2 overshoots the target by
    // seconds. `reveal.ts` rounds tile counts per frame instead, so a fractional
    // rate is exact in aggregate and identical to the old behaviour whenever the
    // rate happens to be whole - which is what keeps the parity fixtures valid.
    rates = segments.map((segment) =>
      Math.min(
        bounds.maxSkipRate,
        Math.max(bounds.minSkipRate, base * SPEED_WEIGHTS[segment.kind]),
      ),
    );
  }

  const timings: SegmentTiming[] = segments.map((segment, i) => ({
    kind: segment.kind,
    tileCount: segment.tileCount,
    skipRate: rates[i],
    // Matches the reference: a frame every `skipRate` tiles, trailing tiles
    // drawn but not given a frame of their own.
    frameCount: Math.floor(segment.tileCount / rates[i]),
  }));

  const drawingFrames = timings.reduce((sum, t) => sum + t.frameCount, 0);

  if (narrationSeconds === null) {
    return {
      segments: timings,
      drawingFrames,
      holdFrames: endHoldFrames,
      durationInFrames: drawingFrames + endHoldFrames,
      clamped: false,
    };
  }

  // The scene never ends before its narration: rounding slack and any clamping
  // extend the hold rather than cutting the voice off.
  const targetFrames = Math.round(narrationSeconds * fps);
  const durationInFrames = Math.max(targetFrames, drawingFrames + endHoldFrames);

  return {
    segments: timings,
    drawingFrames,
    holdFrames: durationInFrames - drawingFrames,
    durationInFrames,
    clamped: drawingFrames + endHoldFrames > targetFrames,
  };
}
