/**
 * Build a complete video from a scene list.
 *
 * Reads a fixture describing scenes (image, mask, audio, narration text), probes
 * each audio file, precomputes each scene timed against its narration, sources
 * captions, and writes a VideoManifest the Remotion composition renders directly.
 *
 * This is the whole of ffmpeg-command-notes.txt as one deterministic step. There
 * is no ffprobe call, no apad, and no concat list - scene offsets are just the
 * running sum of scene durations.
 *
 * Usage:
 *   tsx scripts/buildVideo.ts --fixture ../fixtures/shepherd-boy.json --name story
 *   tsx scripts/buildVideo.ts --fixture ... --name story-portrait --aspect portrait
 */

import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { parseFile } from "music-metadata";

import { precomputeScene } from "./precompute";
import {
  ASPECT_PRESETS,
  CONFIG,
  roundToGrid,
  type AspectPreset,
  type CaptionStyle,
  type CaptionWord,
  type SceneManifest,
  type VideoManifest,
} from "../src/drawing/manifest";
import { loadWhisperWords, splitByScenes } from "../src/captions/whisperTranscript";
import { ProportionalCaptionSource } from "../src/captions/whisperTranscript";

const HERE = dirname(fileURLToPath(import.meta.url));
const PUBLIC = resolve(HERE, "../public");

interface FixtureScene {
  image: string;
  mask?: string | null;
  audio?: string | null;
  text?: string;
}

interface Fixture {
  title: string;
  language?: string;
  captionStyle?: CaptionStyle;
  watermark?: boolean;
  music?: string | null;
  musicGainDb?: number;
  brand?: VideoManifest["brand"];
  /** A single Whisper transcript covering every scene's narration, concatenated. */
  combinedTranscript?: string;
  scenes: FixtureScene[];
}

function parseArgs() {
  const args = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
  }
  const fixture = args.get("fixture");
  if (!fixture) throw new Error("--fixture is required");
  return {
    fixture: resolve(process.cwd(), fixture),
    name: args.get("name") ?? "story",
    aspect: (args.get("aspect") ?? "square") as AspectPreset,
    captionStyle: args.get("captions") as CaptionStyle | undefined,
    watermark: args.get("watermark") === "true",
  };
}

/** Copy narration into public/ so staticFile() can serve it to the browser. */
function stageAudio(source: string, name: string, index: number): string {
  const target = `audio/${name}-${index}${basename(source).slice(basename(source).lastIndexOf("."))}`;
  mkdirSync(resolve(PUBLIC, "audio"), { recursive: true });
  copyFileSync(source, resolve(PUBLIC, target));
  return target;
}

async function main() {
  const options = parseArgs();
  const fixtureDir = dirname(options.fixture);
  const fixture = JSON.parse(readFileSync(options.fixture, "utf8")) as Fixture;

  const preset = ASPECT_PRESETS[options.aspect];
  if (!preset) throw new Error(`unknown aspect "${options.aspect}"`);

  // Every dimension must divide by each splitLen or the tile grid cannot slice it.
  const splitLens = [CONFIG.splitLen, CONFIG.backgroundSplitLen, CONFIG.noMaskSplitLen];
  const width = roundToGrid(preset.width, splitLens);
  const height = roundToGrid(preset.height, splitLens);
  if (width !== preset.width || height !== preset.height) {
    console.log(
      `note: ${options.aspect} ${preset.width}x${preset.height} rounded to ${width}x${height} to fit the tile grid`,
    );
  }

  const resolvePath = (path: string) => resolve(fixtureDir, path);

  // Legacy square artwork has to be padded into a non-square project or the
  // drawing is visibly stretched. Generated artwork will already match the
  // project ratio, at which point "stretch" is a no-op and stays the default.
  const fit = width === height ? "stretch" : "pad";

  // --- audio durations ------------------------------------------------------
  const durations = await Promise.all(
    fixture.scenes.map(async (scene) =>
      scene.audio ? (await parseFile(resolvePath(scene.audio))).format.duration ?? null : null,
    ),
  );

  // --- captions -------------------------------------------------------------
  // The committed transcript covers all scenes as one recording, so it is split
  // back per scene by aligning against the narration text rather than by guessing
  // at the padding that produced the concatenation.
  let perSceneCaptions: CaptionWord[][] = fixture.scenes.map(() => []);

  if (fixture.combinedTranscript) {
    const words = loadWhisperWords(resolvePath(fixture.combinedTranscript));
    perSceneCaptions = splitByScenes(
      words,
      fixture.scenes.map((scene) => scene.text ?? ""),
    );
    console.log(
      `captions: split ${words.length} transcribed words across ${fixture.scenes.length} scenes ` +
        `(${perSceneCaptions.map((c) => c.length).join("/")})`,
    );
  } else {
    const fallback = new ProportionalCaptionSource();
    perSceneCaptions = await Promise.all(
      fixture.scenes.map((scene, i) =>
        fallback.load({
          audioPath: scene.audio ? resolvePath(scene.audio) : "",
          text: scene.text,
          durationSeconds: durations[i] ?? 0,
        }),
      ),
    );
    console.log("captions: no transcript supplied, using proportional timings");
  }

  // --- scenes ---------------------------------------------------------------
  const scenes: SceneManifest[] = [];
  let startFrame = 0;

  for (let i = 0; i < fixture.scenes.length; i++) {
    const scene = fixture.scenes[i];
    const sceneName = `${options.name}-scene-${i + 1}`;

    const manifest = await precomputeScene({
      image: resolvePath(scene.image),
      mask: scene.mask ? resolvePath(scene.mask) : null,
      name: sceneName,
      width,
      height,
      narrationSeconds: durations[i],
      fit,
    });

    scenes.push({
      name: sceneName,
      audio: scene.audio ? stageAudio(resolvePath(scene.audio), options.name, i + 1) : null,
      captions: perSceneCaptions[i] ?? [],
      durationInFrames: manifest.durationInFrames,
      startFrame,
    });
    startFrame += manifest.durationInFrames;
  }

  // --- manifest -------------------------------------------------------------
  const video: VideoManifest = {
    title: fixture.title,
    width,
    height,
    fps: CONFIG.frameRate,
    captionStyle:
      options.captionStyle ??
      fixture.captionStyle ??
      // Vertical is short-form, where word highlighting is the convention;
      // landscape explainers read better as calm phrase blocks.
      (height > width ? "word" : "sentence"),
    watermark: options.watermark || Boolean(fixture.watermark),
    music: fixture.music ?? null,
    musicGainDb: fixture.musicGainDb ?? -18,
    brand: fixture.brand ?? {},
    language: fixture.language ?? "en",
    durationInFrames: startFrame,
    scenes,
  };

  const outDir = resolve(PUBLIC, `precomputed/${options.name}`);
  mkdirSync(outDir, { recursive: true });
  writeFileSync(`${outDir}/video.json`, JSON.stringify(video));

  console.log(`\n${video.title}  ${width}x${height}  captions=${video.captionStyle}`);
  for (let i = 0; i < scenes.length; i++) {
    const seconds = (scenes[i].durationInFrames / CONFIG.frameRate).toFixed(2);
    const narration = durations[i] ? `${durations[i]!.toFixed(2)}s narration` : "no audio";
    console.log(
      `  scene ${i + 1}: ${String(scenes[i].durationInFrames).padStart(4)} frames (${seconds}s), ` +
        `${narration}, ${scenes[i].captions.length} caption words`,
    );
  }
  console.log(
    `  total: ${video.durationInFrames} frames (${(video.durationInFrames / CONFIG.frameRate).toFixed(2)}s)`,
  );
  console.log(`wrote ${outDir}/video.json`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
