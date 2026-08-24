import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  computeTilePath,
  maskThreshold,
  revealStateForFrame,
  segmentFrameCount,
} from "./tilePath";

/** Build a threshold bitmap with ink at the given tile coordinates. */
function bitmapWithInkTiles(
  tiles: [number, number][],
  { width, height, splitLen }: { width: number; height: number; splitLen: number },
) {
  const data = new Uint8Array(width * height).fill(255);
  for (const [row, col] of tiles) {
    // One ink pixel is enough - the detector counts pixels below the threshold.
    data[(row * splitLen) * width + col * splitLen] = 0;
  }
  return data;
}

const GRID = { width: 40, height: 40, splitLen: 10 };

describe("computeTilePath", () => {
  it("starts at the top-left ink tile (numpy row-major seeding)", () => {
    const bitmap = bitmapWithInkTiles(
      [
        [2, 3],
        [0, 1],
        [1, 0],
      ],
      GRID,
    );
    const path = computeTilePath(bitmap, null, GRID);
    assert.deepEqual(path[0], [0, 1]);
  });

  it("leaves the final candidate undrawn", () => {
    // The reference loops while more than one candidate remains, so a segment
    // always ends one tile short. Preserving this keeps parity.
    const bitmap = bitmapWithInkTiles(
      [
        [0, 0],
        [0, 1],
        [0, 2],
      ],
      GRID,
    );
    assert.equal(computeTilePath(bitmap, null, GRID).length, 2);
  });

  it("walks to the nearest remaining tile", () => {
    const bitmap = bitmapWithInkTiles(
      [
        [0, 0],
        [0, 1],
        [3, 3],
      ],
      GRID,
    );
    const path = computeTilePath(bitmap, null, GRID);
    // From (0,0) the neighbour at (0,1) is nearer than the far corner.
    assert.deepEqual(path[0], [0, 0]);
    assert.deepEqual(path[1], [0, 1]);
  });

  it("is deterministic", () => {
    const bitmap = bitmapWithInkTiles(
      [
        [0, 0],
        [1, 2],
        [2, 1],
        [3, 3],
        [1, 1],
      ],
      GRID,
    );
    assert.deepEqual(
      computeTilePath(bitmap, null, GRID),
      computeTilePath(bitmap, null, GRID),
    );
  });

  it("returns nothing for a blank image", () => {
    assert.deepEqual(computeTilePath(new Uint8Array(1600).fill(255), null, GRID), []);
  });

  it("only walks tiles inside the mask", () => {
    const bitmap = bitmapWithInkTiles(
      [
        [0, 0],
        [0, 1],
        [3, 3],
      ],
      GRID,
    );
    // Mask covering only the top-left quadrant.
    const mask = new Uint8Array(GRID.width * GRID.height).fill(0);
    for (let y = 0; y < 20; y++) {
      for (let x = 0; x < 20; x++) mask[y * GRID.width + x] = 255;
    }

    const path = computeTilePath(bitmap, mask, GRID);
    for (const [row, col] of path) {
      assert.ok(row < 2 && col < 2, `tile ${row},${col} lies outside the mask`);
    }
  });
});

describe("maskThreshold", () => {
  it("blanks everything outside the object to paper white", () => {
    const threshold = new Uint8Array([0, 0, 0, 0]);
    const mask = new Uint8Array([255, 0, 255, 0]);
    assert.deepEqual(Array.from(maskThreshold(threshold, mask)), [0, 255, 0, 255]);
  });

  it("returns the input untouched when there is no mask", () => {
    const threshold = new Uint8Array([1, 2, 3]);
    assert.equal(maskThreshold(threshold, null), threshold);
  });
});

describe("frame mapping", () => {
  it("emits one frame per skipRate tiles, discarding the remainder", () => {
    assert.equal(segmentFrameCount(920, 8), 115);
    assert.equal(segmentFrameCount(7, 8), 0);
  });

  it("puts the hand on the last tile drawn", () => {
    // Frame 0 corresponds to the reference's counter === skipRate.
    assert.deepEqual(revealStateForFrame(0, 8), { tilesDrawn: 8, handTileIndex: 7 });
    assert.deepEqual(revealStateForFrame(3, 8), { tilesDrawn: 32, handTileIndex: 31 });
  });
});
