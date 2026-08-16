/**
 * Precompute step: image (+ optional polygon masks) -> drawing path + render assets.
 *
 * This is the half of the port that makes Remotion viable at all. The Python
 * renderer accumulates state - `drawn_frame` is mutated in a loop and frames fall
 * out as a side effect - so frame N is only reachable by computing frames 0..N-1.
 * Remotion (and Lambda, and a scrubbable Player) needs frame N to be a pure
 * function of N.
 *
 * So the expensive, order-dependent work happens once, here, and is baked into:
 *
 *   path.json          the tile visit order per segment, plus frame timings
 *   source.png         the resized colour image (final reveal + end hold)
 *   seg-<i>-ink.png    that segment's masked threshold - what its tiles paint
 *   cum-<i>.png        the exact canvas state after segment <i> finishes
 *   hand.png           the drawing hand, cropped, with a real alpha channel
 *
 * With `cum-<i>` as a starting point, any frame in segment i+1 is one full-image
 * blit plus a bounded number of tile blits. No history required.
 *
 * Usage:
 *   tsx scripts/precompute.ts --image ../../path/2.png --mask ../../path/2.json --name scene-2
 *   tsx scripts/precompute.ts --image in.png --name scene --narration 20.2   # narration-timed
 */

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

import {
  buildObjectMasks,
  openCvReady,
  preprocessImage,
  readPng,
  writeGrayPng,
  writeRgbaPng,
  type FitMode,
  type LabelmeFile,
} from "../src/drawing/preprocess";
import {
  computeTilePath,
  maskThreshold,
  type Tile,
} from "../src/drawing/tilePath";
import {
  CONFIG,
  assertGridFits,
  type PathManifest,
  type SegmentManifest,
} from "../src/drawing/manifest";
import { computeSceneTiming, type SegmentKind } from "../src/drawing/timing";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../../..");

export const DEFAULT_HAND = `${REPO_ROOT}/generate-whiteboard-animated-videos/images/drawing-hand.png`;
export const DEFAULT_HAND_MASK = `${REPO_ROOT}/generate-whiteboard-animated-videos/images/hand-mask.png`;

export interface PrecomputeOptions {
  image: string;
  mask?: string | null;
  name: string;
  width?: number;
  height?: number;
  /**
   * Narration length in seconds. When supplied, drawing speed is solved for so
   * the hand finishes with the voice. Omit to keep the reference's fixed skip
   * rates - which is what the parity fixtures rely on.
   */
  narrationSeconds?: number | null;
  hand?: string;
  handMask?: string;
  outDir?: string;
  quiet?: boolean;
  /**
   * How to fit source artwork into the render size. Generated artwork already
   * matches the project ratio ("stretch"); legacy square art in a non-square
   * project must be padded or the drawing is visibly distorted.
   */
  fit?: FitMode;
}

/** Crop the hand to its mask's bounding box and turn the mask into an alpha channel. */
function buildHandSprite(handPath: string, handMaskPath: string) {
  const hand = readPng(handPath);
  const mask = readPng(handMaskPath);

  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  for (let y = 0; y < mask.height; y++) {
    for (let x = 0; x < mask.width; x++) {
      if (mask.data[(y * mask.width + x) * 4] === 255) {
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
      }
    }
  }

  // Exclusive upper bound, matching the reference's numpy slice.
  const width = maxX - minX;
  const height = maxY - minY;
  const sprite = new Uint8Array(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const src = ((y + minY) * hand.width + (x + minX)) * 4;
      const maskSrc = ((y + minY) * mask.width + (x + minX)) * 4;
      const dst = (y * width + x) * 4;
      const inside = mask.data[maskSrc] === 255;
      sprite[dst] = inside ? hand.data[src] : 0;
      sprite[dst + 1] = inside ? hand.data[src + 1] : 0;
      sprite[dst + 2] = inside ? hand.data[src + 2] : 0;
      sprite[dst + 3] = inside ? 255 : 0;
    }
  }

  return { sprite, width, height };
}

interface Pass {
  label: string;
  kind: SegmentKind;
  splitLen: number;
  fixedSkipRate: number;
  tiles: Tile[];
  mask: Uint8Array | null;
}

export async function precomputeScene(options: PrecomputeOptions): Promise<PathManifest> {
  const {
    image,
    mask = null,
    name,
    width = CONFIG.width,
    height = CONFIG.height,
    narrationSeconds = null,
    hand = DEFAULT_HAND,
    handMask = DEFAULT_HAND_MASK,
    quiet = false,
    fit = "stretch",
  } = options;

  await openCvReady();

  assertGridFits(width, height, [
    CONFIG.splitLen,
    CONFIG.backgroundSplitLen,
    CONFIG.noMaskSplitLen,
  ]);

  const outDir = options.outDir ?? resolve(HERE, `../public/precomputed/${name}`);
  mkdirSync(outDir, { recursive: true });

  const png = readPng(image);
  const pre = preprocessImage(png, width, height, fit);

  // --- passes ---------------------------------------------------------------
  const passes: Pass[] = [];

  if (mask) {
    const file = JSON.parse(readFileSync(mask, "utf8")) as LabelmeFile;
    const { objects, background } = buildObjectMasks(
      file,
      pre.sourceWidth,
      pre.sourceHeight,
      width,
      height,
      fit,
    );

    for (const object of objects) {
      passes.push({
        label: object.label,
        kind: "object",
        splitLen: CONFIG.splitLen,
        fixedSkipRate: CONFIG.objectSkipRate,
        tiles: computeTilePath(pre.threshold, object.mask, {
          width,
          height,
          splitLen: CONFIG.splitLen,
          blackPixelThreshold: CONFIG.blackPixelThreshold,
        }),
        mask: object.mask,
      });
    }

    passes.push({
      label: "background",
      kind: "background",
      splitLen: CONFIG.backgroundSplitLen,
      fixedSkipRate: CONFIG.backgroundSkipRate,
      tiles: computeTilePath(pre.threshold, background, {
        width,
        height,
        splitLen: CONFIG.backgroundSplitLen,
        blackPixelThreshold: CONFIG.blackPixelThreshold,
      }),
      mask: background,
    });
  } else {
    passes.push({
      label: "full",
      kind: "full",
      splitLen: CONFIG.noMaskSplitLen,
      fixedSkipRate: CONFIG.noMaskSkipRate,
      tiles: computeTilePath(pre.threshold, null, {
        width,
        height,
        splitLen: CONFIG.noMaskSplitLen,
        blackPixelThreshold: CONFIG.blackPixelThreshold,
      }),
      mask: null,
    });
  }

  // --- timing ---------------------------------------------------------------
  const timing = computeSceneTiming(
    passes.map((pass) => ({
      kind: pass.kind,
      tileCount: pass.tiles.length,
      fixedSkipRate: pass.fixedSkipRate,
    })),
    {
      fps: CONFIG.frameRate,
      endHoldSeconds: CONFIG.endHoldSeconds,
      narrationSeconds,
    },
  );

  // --- assets ---------------------------------------------------------------
  // Replay the passes to capture the true canvas state at each segment boundary.
  // This cannot be shortcut to "white + every finished mask": a tile straddling an
  // object edge paints white outside that object, so a later pass can white out
  // part of an earlier object that is never repainted. The reference does that,
  // so parity requires reproducing it rather than tidying it away.
  const canvas = new Uint8Array(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    canvas[i * 4] = 255;
    canvas[i * 4 + 1] = 255;
    canvas[i * 4 + 2] = 255;
    canvas[i * 4 + 3] = 255;
  }

  const segments: SegmentManifest[] = [];
  let startFrame = 0;

  passes.forEach((pass, index) => {
    const ink = maskThreshold(pre.threshold, pass.mask);
    writeGrayPng(`${outDir}/seg-${index}-ink.png`, ink, width, height);

    // Canvas state entering this segment is what the composition starts from.
    writeRgbaPng(`${outDir}/cum-${index}.png`, canvas, width, height);

    const { splitLen } = pass;
    for (const [row, col] of pass.tiles) {
      for (let y = row * splitLen; y < (row + 1) * splitLen; y++) {
        for (let x = col * splitLen; x < (col + 1) * splitLen; x++) {
          const pixel = y * width + x;
          const value = ink[pixel];
          canvas[pixel * 4] = value;
          canvas[pixel * 4 + 1] = value;
          canvas[pixel * 4 + 2] = value;
        }
      }
    }

    if (pass.mask === null) {
      canvas.set(pre.source);
    } else {
      for (let i = 0; i < width * height; i++) {
        if (pass.mask[i] === 255) {
          canvas[i * 4] = pre.source[i * 4];
          canvas[i * 4 + 1] = pre.source[i * 4 + 1];
          canvas[i * 4 + 2] = pre.source[i * 4 + 2];
        }
      }
    }

    segments.push({
      label: pass.label,
      kind: pass.kind,
      splitLen: pass.splitLen,
      skipRate: timing.segments[index].skipRate,
      tiles: pass.tiles,
      frameCount: timing.segments[index].frameCount,
      startFrame,
    });
    startFrame += timing.segments[index].frameCount;
  });

  writeRgbaPng(`${outDir}/source.png`, pre.source, width, height);
  writeGrayPng(`${outDir}/threshold.png`, pre.threshold, width, height);

  const sprite = buildHandSprite(hand, handMask);
  writeRgbaPng(`${outDir}/hand.png`, sprite.sprite, sprite.width, sprite.height);

  const manifest: PathManifest = {
    image,
    mask,
    config: { ...CONFIG, width, height },
    width,
    height,
    durationInFrames: timing.durationInFrames,
    narrationSeconds,
    segments,
  };
  writeFileSync(`${outDir}/path.json`, JSON.stringify(manifest));

  if (!quiet) {
    const totalTiles = segments.reduce((sum, s) => sum + s.tiles.length, 0);
    console.log(
      `${name}: ${width}x${height}  segments ${segments.length}  tiles ${totalTiles}  ` +
        `frames ${manifest.durationInFrames}` +
        (narrationSeconds ? `  narration ${narrationSeconds.toFixed(2)}s` : "  fixed rates"),
    );
    for (const segment of segments) {
      console.log(
        `  ${segment.label.padEnd(12)} ${segment.kind.padEnd(10)} ` +
          `tiles=${String(segment.tiles.length).padEnd(6)} skip=${String(segment.skipRate).padEnd(3)} frames=${segment.frameCount}`,
      );
    }
    if (timing.clamped) {
      console.log("  note: drawing could not fit the narration; scene extended");
    }
  }

  return manifest;
}

// --- CLI ---------------------------------------------------------------------

function parseArgs() {
  const args = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
  }
  const image = args.get("image");
  if (!image) throw new Error("--image is required");
  return {
    image: resolve(process.cwd(), image),
    mask: args.get("mask") ? resolve(process.cwd(), args.get("mask")!) : null,
    name: args.get("name") ?? "scene",
    width: args.get("width") ? Number(args.get("width")) : CONFIG.width,
    height: args.get("height") ? Number(args.get("height")) : CONFIG.height,
    narrationSeconds: args.get("narration") ? Number(args.get("narration")) : null,
  };
}

const isCli = process.argv[1] && import.meta.url.endsWith(process.argv[1].split("/").pop()!);
if (isCli) {
  precomputeScene(parseArgs())
    .then((manifest) => {
      console.log(`wrote assets for ${manifest.segments.length} segments`);
    })
    .catch((error) => {
      console.error(error);
      process.exit(1);
    });
}
