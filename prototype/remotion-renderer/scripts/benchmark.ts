/**
 * How fast is a single frame, and does seeking cost more than playing?
 *
 * The product decision this backs is real-time scrubbable preview. That only
 * works if an arbitrary frame is cheap to produce - if random seeks were slower
 * than sequential playback it would mean state is leaking between frames, and the
 * Player would stutter whenever the user dragged the timeline.
 *
 * Usage: tsx scripts/benchmark.ts --name scene-2
 */

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { PNG } from "pngjs";

import { renderFrame, type RevealAssets } from "../src/drawing/reveal";
import type { PathManifest } from "../src/drawing/manifest";

const HERE = dirname(fileURLToPath(import.meta.url));

function readRgba(path: string) {
  const png = PNG.sync.read(readFileSync(path));
  return new Uint8Array(png.data);
}

function readGray(path: string) {
  const png = PNG.sync.read(readFileSync(path));
  const gray = new Uint8Array(png.width * png.height);
  for (let i = 0; i < gray.length; i++) gray[i] = png.data[i * 4];
  return gray;
}

function summarise(label: string, timings: number[], fps: number) {
  const sorted = [...timings].sort((a, b) => a - b);
  const mean = timings.reduce((s, t) => s + t, 0) / timings.length;
  const p95 = sorted[Math.floor(sorted.length * 0.95)];
  const budget = 1000 / fps;
  console.log(
    `  ${label.padEnd(22)} mean ${mean.toFixed(2)}ms  p95 ${p95.toFixed(2)}ms  max ${sorted[sorted.length - 1].toFixed(2)}ms  ` +
      `(${(budget / mean).toFixed(1)}x realtime at ${fps}fps)`,
  );
  return mean;
}

function main() {
  const nameArg = process.argv.indexOf("--name");
  const name = nameArg > -1 ? process.argv[nameArg + 1] : "scene-2";
  const assetDir = resolve(HERE, `../public/precomputed/${name}`);

  const manifest: PathManifest = JSON.parse(
    readFileSync(`${assetDir}/path.json`, "utf8"),
  );
  const hand = PNG.sync.read(readFileSync(`${assetDir}/hand.png`));
  const assets: RevealAssets = {
    cumulative: manifest.segments.map((_, i) => readRgba(`${assetDir}/cum-${i}.png`)),
    ink: manifest.segments.map((_, i) => readGray(`${assetDir}/seg-${i}-ink.png`)),
    source: readRgba(`${assetDir}/source.png`),
    hand: { data: new Uint8Array(hand.data), width: hand.width, height: hand.height },
  };

  const fps = manifest.config.frameRate;
  const total = manifest.durationInFrames;

  // Warm up so JIT compilation is not counted.
  for (let i = 0; i < 20; i++) renderFrame(manifest, assets, i);

  const sequential: number[] = [];
  for (let frame = 0; frame < total; frame++) {
    const start = performance.now();
    renderFrame(manifest, assets, frame);
    sequential.push(performance.now() - start);
  }

  // Deterministic pseudo-random seeks - a stand-in for a user dragging the
  // playhead, which is the access pattern the Player actually produces.
  const seeks: number[] = [];
  let cursor = 7;
  for (let i = 0; i < total; i++) {
    cursor = (cursor * 1103515245 + 12345) % 2147483647;
    const frame = cursor % total;
    const start = performance.now();
    renderFrame(manifest, assets, frame);
    seeks.push(performance.now() - start);
  }

  console.log(`\n=== frame cost: ${name} (${manifest.width}x${manifest.height}, ${total} frames) ===\n`);
  const sequentialMean = summarise("sequential playback", sequential, fps);
  const seekMean = summarise("random seeks", seeks, fps);

  const ratio = seekMean / sequentialMean;
  console.log(
    `\n  random-seek cost vs sequential: ${ratio.toFixed(2)}x ` +
      `(~1.0 confirms frames are independent - no accumulated state)\n`,
  );
}

main();
