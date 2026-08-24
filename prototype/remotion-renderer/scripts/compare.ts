/**
 * Parity harness: does the TypeScript port reproduce the Python reference?
 *
 * Two independent checks, because they fail for different reasons:
 *
 *  1. PATH parity - element-wise equality of the tile visit order. This is the
 *     brittle half (numpy ordering, swap-with-last removal, argmin tie-breaks)
 *     and a mismatch here means the animation is simply a different drawing.
 *  2. PIXEL parity - SSIM and mean absolute error on sampled frames. This catches
 *     compositing mistakes the path check cannot see: wrong hand offset, tiles
 *     sourced from the unmasked threshold, a missed colour reveal.
 *  3. BROWSER parity - the same metrics against frames Remotion actually rendered
 *     through Chromium. Checks 1 and 2 exercise the shared reveal function in
 *     Node; only this one proves the React/canvas layer does not resample,
 *     colour-shift or misalign what that function produced.
 *
 * Usage:
 *   tsx scripts/compare.ts --name scene-2 \
 *     --python-path ../out/path-python.json --python-frames ../out/frames-python \
 *     [--remotion-frames ../out/frames-remotion]
 */

import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

import { renderFrame, type RevealAssets } from "../src/drawing/reveal";
import type { PathManifest } from "../src/drawing/manifest";

const HERE = dirname(fileURLToPath(import.meta.url));

const SSIM_PASS_THRESHOLD = 0.97;

function parseArgs() {
  const args = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
  }
  const name = args.get("name") ?? "scene-2";
  return {
    name,
    assetDir: resolve(HERE, `../public/precomputed/${name}`),
    pythonPath: resolve(process.cwd(), args.get("python-path") ?? `${HERE}/../../out/path-python.json`),
    pythonFrames: resolve(process.cwd(), args.get("python-frames") ?? `${HERE}/../../out/frames-python`),
    remotionFrames: resolve(process.cwd(), args.get("remotion-frames") ?? `${HERE}/../../out/frames-remotion`),
    outDir: resolve(process.cwd(), args.get("out") ?? `${HERE}/../../out/parity`),
  };
}

function readRgba(path: string) {
  const png = PNG.sync.read(readFileSync(path));
  return { data: new Uint8Array(png.data), width: png.width, height: png.height };
}

function readGray(path: string) {
  const png = PNG.sync.read(readFileSync(path));
  const gray = new Uint8Array(png.width * png.height);
  for (let i = 0; i < gray.length; i++) gray[i] = png.data[i * 4];
  return gray;
}

// --- metrics ----------------------------------------------------------------

/** Rec.601 luma, matching how the reference's BGR frames are perceived. */
function toLuma(rgba: Uint8Array | Uint8ClampedArray, pixels: number) {
  const luma = new Float64Array(pixels);
  for (let i = 0; i < pixels; i++) {
    luma[i] =
      0.299 * rgba[i * 4] + 0.587 * rgba[i * 4 + 1] + 0.114 * rgba[i * 4 + 2];
  }
  return luma;
}

/**
 * Global SSIM over 8x8 windows (the standard formulation with C1/C2 stabilisers).
 * Windowed rather than whole-image because these frames are mostly flat white -
 * a global statistic would report near-perfect similarity even for a badly
 * misplaced hand.
 */
function ssim(a: Float64Array, b: Float64Array, width: number, height: number) {
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const window = 8;
  let total = 0;
  let windows = 0;

  for (let y = 0; y + window <= height; y += window) {
    for (let x = 0; x + window <= width; x += window) {
      let sumA = 0, sumB = 0, sumAA = 0, sumBB = 0, sumAB = 0;
      for (let wy = 0; wy < window; wy++) {
        for (let wx = 0; wx < window; wx++) {
          const i = (y + wy) * width + (x + wx);
          sumA += a[i];
          sumB += b[i];
          sumAA += a[i] * a[i];
          sumBB += b[i] * b[i];
          sumAB += a[i] * b[i];
        }
      }
      const n = window * window;
      const meanA = sumA / n;
      const meanB = sumB / n;
      const varA = sumAA / n - meanA * meanA;
      const varB = sumBB / n - meanB * meanB;
      const covAB = sumAB / n - meanA * meanB;

      total +=
        ((2 * meanA * meanB + C1) * (2 * covAB + C2)) /
        ((meanA * meanA + meanB * meanB + C1) * (varA + varB + C2));
      windows++;
    }
  }

  return windows === 0 ? 1 : total / windows;
}

function meanAbsoluteError(
  a: Uint8Array | Uint8ClampedArray,
  b: Uint8Array | Uint8ClampedArray,
  pixels: number,
) {
  let sum = 0;
  let differing = 0;
  for (let i = 0; i < pixels; i++) {
    let pixelDiff = 0;
    for (let c = 0; c < 3; c++) pixelDiff += Math.abs(a[i * 4 + c] - b[i * 4 + c]);
    if (pixelDiff > 0) differing++;
    sum += pixelDiff / 3;
  }
  return { mae: sum / pixels, differingPixels: differing };
}

// --- output -----------------------------------------------------------------

function writeSideBySide(
  path: string,
  left: Uint8Array | Uint8ClampedArray,
  right: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  scale = 3,
) {
  const outWidth = Math.floor((width * 2) / scale);
  const outHeight = Math.floor(height / scale);
  const png = new PNG({ width: outWidth, height: outHeight });

  for (let y = 0; y < outHeight; y++) {
    for (let x = 0; x < outWidth; x++) {
      const half = x < outWidth / 2;
      const source = half ? left : right;
      const sourceX = Math.floor((half ? x : x - outWidth / 2) * scale);
      const sourceY = Math.floor(y * scale);
      const src = (sourceY * width + sourceX) * 4;
      const dst = (y * outWidth + x) * 4;
      png.data[dst] = source[src];
      png.data[dst + 1] = source[src + 1];
      png.data[dst + 2] = source[src + 2];
      png.data[dst + 3] = 255;
    }
  }
  writeFileSync(path, PNG.sync.write(png));
}

// --- main -------------------------------------------------------------------

function main() {
  const { name, assetDir, pythonPath, pythonFrames, remotionFrames, outDir } = parseArgs();
  mkdirSync(outDir, { recursive: true });

  const manifest: PathManifest = JSON.parse(
    readFileSync(`${assetDir}/path.json`, "utf8"),
  );

  console.log(`\n=== parity report: ${name} ===\n`);

  // --- 1. path parity -------------------------------------------------------
  let pathPass = true;
  if (existsSync(pythonPath)) {
    const python = JSON.parse(readFileSync(pythonPath, "utf8"));
    const pySegments = python.segments as {
      label: string;
      tiles: [number, number][];
    }[];

    if (pySegments.length !== manifest.segments.length) {
      pathPass = false;
      console.log(
        `segment count MISMATCH: python=${pySegments.length} typescript=${manifest.segments.length}`,
      );
    }

    console.log("PATH PARITY (tile visit order, element-wise)");
    for (let i = 0; i < Math.min(pySegments.length, manifest.segments.length); i++) {
      const py = pySegments[i].tiles;
      const ts = manifest.segments[i].tiles;
      let identical = py.length === ts.length;
      let firstDivergence = -1;
      for (let j = 0; identical && j < py.length; j++) {
        if (py[j][0] !== ts[j][0] || py[j][1] !== ts[j][1]) {
          identical = false;
          firstDivergence = j;
        }
      }
      pathPass &&= identical;
      const label = manifest.segments[i].label.padEnd(12);
      console.log(
        `  ${label} python=${String(py.length).padEnd(5)} ts=${String(ts.length).padEnd(5)} ${identical ? "identical" : `DIVERGES at tile ${firstDivergence}`}`,
      );
    }
  } else {
    console.log(`PATH PARITY skipped - no reference at ${pythonPath}`);
  }

  // --- 2. pixel parity ------------------------------------------------------
  const source = readRgba(`${assetDir}/source.png`);
  const hand = readRgba(`${assetDir}/hand.png`);
  const assets: RevealAssets = {
    cumulative: manifest.segments.map((_, i) => readRgba(`${assetDir}/cum-${i}.png`).data),
    ink: manifest.segments.map((_, i) => readGray(`${assetDir}/seg-${i}-ink.png`)),
    source: source.data,
    hand: { data: hand.data, width: hand.width, height: hand.height },
  };

  const framesMeta = existsSync(`${pythonFrames}/frames.json`)
    ? JSON.parse(readFileSync(`${pythonFrames}/frames.json`, "utf8"))
    : null;

  let pixelPass = true;
  let pixelChecked = 0;
  const rows: string[] = [];

  if (framesMeta) {
    console.log(`\nDURATION  python=${framesMeta.total}  typescript=${manifest.durationInFrames}`);
    if (framesMeta.total !== manifest.durationInFrames) pixelPass = false;

    console.log("\nPIXEL PARITY (sampled frames)");
    console.log("  frame     SSIM      MAE     differing px");

    const pixels = manifest.width * manifest.height;
    for (const frame of framesMeta.sampled as number[]) {
      const file = `${pythonFrames}/frame-${String(frame).padStart(5, "0")}.png`;
      if (!existsSync(file)) continue;

      const reference = readRgba(file);
      const ours = renderFrame(manifest, assets, frame);

      const score = ssim(
        toLuma(reference.data, pixels),
        toLuma(ours, pixels),
        manifest.width,
        manifest.height,
      );
      const { mae, differingPixels } = meanAbsoluteError(reference.data, ours, pixels);
      const ok = score >= SSIM_PASS_THRESHOLD;
      pixelPass &&= ok;
      pixelChecked++;

      console.log(
        `  ${String(frame).padStart(5)}  ${score.toFixed(5)}  ${mae.toFixed(4)}  ${String(differingPixels).padStart(9)}  ${ok ? "" : "FAIL"}`,
      );
      rows.push(`${frame},${score.toFixed(6)},${mae.toFixed(6)},${differingPixels}`);

      writeSideBySide(
        `${outDir}/sidebyside-${String(frame).padStart(5, "0")}.png`,
        reference.data,
        ours,
        manifest.width,
        manifest.height,
      );
    }
    writeFileSync(`${outDir}/metrics.csv`, `frame,ssim,mae,differing_pixels\n${rows.join("\n")}\n`);
  } else {
    console.log(`\nPIXEL PARITY skipped - no reference frames at ${pythonFrames}`);
  }

  // --- 3. browser parity ----------------------------------------------------
  // Remotion writes an image sequence as element-000.png, element-001.png, ...
  let browserPass = true;
  let browserChecked = 0;

  if (framesMeta && existsSync(remotionFrames)) {
    console.log("\nBROWSER PARITY (python vs frames Remotion rendered in Chromium)");
    console.log("  frame     SSIM      MAE     differing px");

    const pixels = manifest.width * manifest.height;
    for (const frame of framesMeta.sampled as number[]) {
      const pythonFile = `${pythonFrames}/frame-${String(frame).padStart(5, "0")}.png`;
      const remotionFile = `${remotionFrames}/element-${String(frame).padStart(3, "0")}.png`;
      if (!existsSync(pythonFile) || !existsSync(remotionFile)) continue;

      const reference = readRgba(pythonFile);
      const rendered = readRgba(remotionFile);

      const score = ssim(
        toLuma(reference.data, pixels),
        toLuma(rendered.data, pixels),
        manifest.width,
        manifest.height,
      );
      const { mae, differingPixels } = meanAbsoluteError(
        reference.data,
        rendered.data,
        pixels,
      );
      const ok = score >= SSIM_PASS_THRESHOLD;
      browserPass &&= ok;
      browserChecked++;

      console.log(
        `  ${String(frame).padStart(5)}  ${score.toFixed(5)}  ${mae.toFixed(4)}  ${String(differingPixels).padStart(9)}  ${ok ? "" : "FAIL"}`,
      );
    }
    if (browserChecked === 0) {
      console.log("  no matching frame pairs found");
      browserPass = false;
    }
  } else {
    console.log(`\nBROWSER PARITY skipped - no Remotion frames at ${remotionFrames}`);
  }

  console.log(`\ncontact sheet -> ${outDir}`);
  const verdict = (checked: number, pass: boolean) =>
    checked === 0 ? "SKIPPED" : pass ? "PASS" : "FAIL";

  console.log(`\nPATH    PARITY: ${pathPass ? "PASS" : "FAIL"}`);
  console.log(
    `PIXEL   PARITY: ${verdict(pixelChecked, pixelPass)} (threshold SSIM >= ${SSIM_PASS_THRESHOLD})`,
  );
  console.log(`BROWSER PARITY: ${verdict(browserChecked, browserPass)}\n`);

  if (!pathPass || (pixelChecked > 0 && !pixelPass) || (browserChecked > 0 && !browserPass)) {
    process.exit(1);
  }
}

main();
