import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { computeSceneTiming, DEFAULT_BOUNDS, SPEED_WEIGHTS } from "./timing";

const FPS = 25;
const END_HOLD = 3;

/** The nine segments of the parity fixture (images/2.png with 2.json masks). */
const PARITY_SEGMENTS = [
  { kind: "object" as const, tileCount: 251, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 920, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 170, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 164, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 102, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 715, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 272, fixedSkipRate: 8 },
  { kind: "object" as const, tileCount: 239, fixedSkipRate: 8 },
  { kind: "background" as const, tileCount: 648, fixedSkipRate: 14 },
];

describe("computeSceneTiming", () => {
  it("reproduces the reference's frame count when no narration is supplied", () => {
    const timing = computeSceneTiming(PARITY_SEGMENTS, {
      fps: FPS,
      endHoldSeconds: END_HOLD,
      narrationSeconds: null,
    });

    // 397 drawing frames + 75 end hold is what the Python renderer emits, and
    // the parity harness compares against exactly this.
    assert.equal(timing.drawingFrames, 397);
    assert.equal(timing.durationInFrames, 472);
    assert.equal(timing.clamped, false);
    assert.deepEqual(
      timing.segments.map((s) => s.skipRate),
      [8, 8, 8, 8, 8, 8, 8, 8, 14],
    );
  });

  it("lands the drawing on the narration", () => {
    for (const narrationSeconds of [10, 20.208, 46.824, 60]) {
      const timing = computeSceneTiming(PARITY_SEGMENTS, {
        fps: FPS,
        endHoldSeconds: END_HOLD,
        narrationSeconds,
      });

      const target = Math.round(narrationSeconds * FPS);
      assert.ok(
        timing.durationInFrames >= target,
        `scene must not end before its narration (${timing.durationInFrames} < ${target})`,
      );
      // Integer skip rates cannot hit the target exactly; the slack lands in the
      // hold, and must stay within a second or so of the end-hold budget.
      const overshoot = timing.durationInFrames - target;
      assert.ok(
        overshoot <= END_HOLD * FPS,
        `overshoot ${overshoot} exceeded the end-hold budget at ${narrationSeconds}s`,
      );
    }
  });

  it("keeps the background faster than objects", () => {
    const timing = computeSceneTiming(PARITY_SEGMENTS, {
      fps: FPS,
      endHoldSeconds: END_HOLD,
      narrationSeconds: 30,
    });

    const object = timing.segments.find((s) => s.kind === "object")!;
    const background = timing.segments.find((s) => s.kind === "background")!;
    assert.ok(
      background.skipRate > object.skipRate,
      "the background pass should still cover ground faster than objects",
    );
    const ratio = background.skipRate / object.skipRate;
    assert.ok(
      Math.abs(ratio - SPEED_WEIGHTS.background) < 0.35,
      `expected roughly the reference's 14:8 pacing, got ${ratio.toFixed(2)}`,
    );
  });

  it("clamps rather than strobing when narration is far too short", () => {
    const timing = computeSceneTiming(PARITY_SEGMENTS, {
      fps: FPS,
      endHoldSeconds: END_HOLD,
      narrationSeconds: 1,
    });

    assert.equal(timing.clamped, true);
    for (const segment of timing.segments) {
      assert.ok(
        segment.skipRate <= DEFAULT_BOUNDS.maxSkipRate,
        "skip rate must respect the upper bound",
      );
    }
  });

  it("never emits a skip rate below one", () => {
    const timing = computeSceneTiming([{ kind: "full", tileCount: 12, fixedSkipRate: 8 }], {
      fps: FPS,
      endHoldSeconds: 0,
      narrationSeconds: 600,
    });

    assert.equal(timing.segments[0].skipRate, 1);
    // A long narration over a tiny drawing becomes hold time, not a stalled hand.
    assert.equal(timing.durationInFrames, 600 * FPS);
    assert.ok(timing.holdFrames > timing.drawingFrames);
  });
});
