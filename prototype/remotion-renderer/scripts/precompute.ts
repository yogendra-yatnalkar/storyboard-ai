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
 * blit plus a bounded number of 10x10 tile blits. No history required.
 *
 * Usage:
 *   tsx scripts/precompute.ts --image ../../path/2.png --mask ../../path/2.json --name scene-2
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
  type LabelmeFile,
} from "../src/drawing/preprocess";
import {
  computeTilePath,
  maskThreshold,
  segmentFrameCount,
} from "../src/drawing/tilePath";
import {
  CONFIG,
  type PathManifest,
  type SegmentManifest,
} from "../src/drawing/manifest";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../../..");

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
    hand: resolve(
      process.cwd(),
      args.get("hand") ??
        `${REPO_ROOT}/generate-whiteboard-animated-videos/images/drawing-hand.png`,
    ),
    handMask: resolve(
      process.cwd(),
      args.get("hand-mask") ??
        `${REPO_ROOT}/generate-whiteboard-animated-videos/images/hand-mask.png`,
    ),
  };
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

async function main() {
  const { image, mask, name, hand, handMask } = parseArgs();
  await openCvReady();

  const outDir = resolve(HERE, `../public/precomputed/${name}`);
  mkdirSync(outDir, { recursive: true });

  const png = readPng(image);
  const pre = preprocessImage(png, CONFIG.width, CONFIG.height);
  const { width, height } = pre;

  // --- segments -------------------------------------------------------------
  type Pass = { segment: Omit<SegmentManifest, "startFrame">; mask: Uint8Array | null };
  const passes: Pass[] = [];

  if (mask) {
    const file = JSON.parse(readFileSync(mask, "utf8")) as LabelmeFile;
    const { objects, background } = buildObjectMasks(
      file,
      pre.sourceWidth,
      pre.sourceHeight,
      width,
      height,
    );

    for (const object of objects) {
      const tiles = computeTilePath(pre.threshold, object.mask, {
        width,
        height,
        splitLen: CONFIG.splitLen,
        blackPixelThreshold: CONFIG.blackPixelThreshold,
      });
      passes.push({
        segment: {
          label: object.label,
          kind: "object",
          splitLen: CONFIG.splitLen,
          skipRate: CONFIG.objectSkipRate,
          tiles,
          frameCount: segmentFrameCount(tiles.length, CONFIG.objectSkipRate),
        },
        mask: object.mask,
      });
    }

    const backgroundTiles = computeTilePath(pre.threshold, background, {
      width,
      height,
      splitLen: CONFIG.backgroundSplitLen,
      blackPixelThreshold: CONFIG.blackPixelThreshold,
    });
    passes.push({
      segment: {
        label: "background",
        kind: "background",
        splitLen: CONFIG.backgroundSplitLen,
        skipRate: CONFIG.backgroundSkipRate,
        tiles: backgroundTiles,
        frameCount: segmentFrameCount(backgroundTiles.length, CONFIG.backgroundSkipRate),
      },
      mask: background,
    });
  } else {
    const tiles = computeTilePath(pre.threshold, null, {
      width,
      height,
      splitLen: CONFIG.noMaskSplitLen,
      blackPixelThreshold: CONFIG.blackPixelThreshold,
    });
    passes.push({
      segment: {
        label: "full",
        kind: "full",
        splitLen: CONFIG.noMaskSplitLen,
        skipRate: CONFIG.noMaskSkipRate,
        tiles,
        frameCount: segmentFrameCount(tiles.length, CONFIG.noMaskSkipRate),
      },
      mask: null,
    });
  }

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

    const { splitLen } = pass.segment;
    for (const [row, col] of pass.segment.tiles) {
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

    segments.push({ ...pass.segment, startFrame });
    startFrame += pass.segment.frameCount;
  });

  writeRgbaPng(`${outDir}/source.png`, pre.source, width, height);
  writeGrayPng(`${outDir}/threshold.png`, pre.threshold, width, height);

  const sprite = buildHandSprite(hand, handMask);
  writeRgbaPng(`${outDir}/hand.png`, sprite.sprite, sprite.width, sprite.height);

  const manifest: PathManifest = {
    image,
    mask,
    config: CONFIG,
    width,
    height,
    durationInFrames: startFrame + CONFIG.frameRate * CONFIG.endHoldSeconds,
    segments,
  };
  writeFileSync(`${outDir}/path.json`, JSON.stringify(manifest));

  const totalTiles = segments.reduce((sum, s) => sum + s.tiles.length, 0);
  console.log(`segments: ${segments.length}  tiles: ${totalTiles}  frames: ${manifest.durationInFrames}`);
  for (const segment of segments) {
    console.log(
      `  ${segment.label.padEnd(12)} ${segment.kind.padEnd(10)} tiles=${String(segment.tiles.length).padEnd(6)} frames=${segment.frameCount}`,
    );
  }
  console.log(`wrote assets -> ${outDir}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
