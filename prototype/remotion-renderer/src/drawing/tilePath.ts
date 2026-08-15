/**
 * Port of the tile traversal from the Python reference renderer.
 *
 * This is the half of the algorithm that must match byte for byte. Three details
 * are load-bearing and easy to "clean up" into something that produces a different
 * animation:
 *
 *  1. Candidates are seeded in row-major order (numpy's `np.where` ordering), so the
 *     walk always starts at the top-left ink tile.
 *  2. A visited tile is removed by overwriting it with the array's LAST element and
 *     shrinking the array - not by splicing. That reshuffles the remaining
 *     candidates and changes which tile is nearest on later iterations.
 *  3. The loop runs while more than one candidate remains, so the final tile of
 *     every segment is intentionally never drawn.
 */

export type Tile = [row: number, col: number];

export interface TilePathOptions {
  width: number;
  height: number;
  splitLen: number;
  blackPixelThreshold?: number;
}

/**
 * Apply an object mask to a threshold bitmap: everything outside the object is
 * blanked to white so only the object's ink is left to draw.
 */
export function maskThreshold(
  threshold: Uint8Array,
  objectMask: Uint8Array | null,
): Uint8Array {
  if (!objectMask) return threshold;

  const masked = new Uint8Array(threshold);
  for (let i = 0; i < masked.length; i++) {
    if (objectMask[i] === 0) masked[i] = 255;
  }
  return masked;
}

/** Count ink pixels per tile, returned as a [rows x cols] row-major grid. */
function inkPerTile(
  threshold: Uint8Array,
  { width, height, splitLen, blackPixelThreshold = 10 }: TilePathOptions,
) {
  const rows = Math.ceil(height / splitLen);
  const cols = Math.ceil(width / splitLen);
  const counts = new Int32Array(rows * cols);

  for (let y = 0; y < height; y++) {
    const row = Math.floor(y / splitLen);
    const rowOffset = row * cols;
    const pixelOffset = y * width;
    for (let x = 0; x < width; x++) {
      if (threshold[pixelOffset + x] < blackPixelThreshold) {
        counts[rowOffset + Math.floor(x / splitLen)]++;
      }
    }
  }

  return { counts, rows, cols };
}

/**
 * Greedy nearest-neighbour walk over the ink tiles, so the drawing hand travels
 * smoothly instead of jumping around the canvas.
 */
export function computeTilePath(
  threshold: Uint8Array,
  objectMask: Uint8Array | null,
  options: TilePathOptions,
): Tile[] {
  const masked = maskThreshold(threshold, objectMask);
  const { counts, rows, cols } = inkPerTile(masked, options);

  // Row-major seeding, matching numpy's np.where ordering.
  const candidates: Tile[] = [];
  for (let row = 0; row < rows; row++) {
    for (let col = 0; col < cols; col++) {
      if (counts[row * cols + col] > 0) candidates.push([row, col]);
    }
  }
  if (candidates.length === 0) return [];

  const visitOrder: Tile[] = [];
  let selectedIndex = 0;

  while (candidates.length > 1) {
    const selected = candidates[selectedIndex];
    visitOrder.push([selected[0], selected[1]]);

    // Swap-with-last removal (see note 2 above).
    candidates[selectedIndex] = candidates[candidates.length - 1];
    candidates.pop();

    // argmin over squared distance - monotonic in the reference's sqrt, and
    // numpy's argmin keeps the FIRST minimum, so the comparison stays strict.
    let bestIndex = 0;
    let bestDistance = Infinity;
    for (let i = 0; i < candidates.length; i++) {
      const dr = candidates[i][0] - selected[0];
      const dc = candidates[i][1] - selected[1];
      const distance = dr * dr + dc * dc;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = i;
      }
    }
    selectedIndex = bestIndex;
  }

  return visitOrder;
}

/**
 * Frames emitted for a segment. The reference writes a frame every `skipRate`
 * tiles, so a segment's trailing tiles beyond the last multiple are drawn but
 * never land on their own frame.
 */
export function segmentFrameCount(tileCount: number, skipRate: number): number {
  return Math.floor(tileCount / skipRate);
}

/**
 * Map a frame index within a segment back to the drawing state it should show.
 *
 * Frame f corresponds to the reference's counter (f + 1) * skipRate: tiles
 * [0, counter) are on the canvas and the hand rests on tile counter - 1.
 */
export function revealStateForFrame(frameInSegment: number, skipRate: number) {
  const tilesDrawn = (frameInSegment + 1) * skipRate;
  return { tilesDrawn, handTileIndex: tilesDrawn - 1 };
}
