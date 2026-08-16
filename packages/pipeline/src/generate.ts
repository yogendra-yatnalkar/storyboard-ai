/**
 * CLI: generate a video project from story text.
 *
 * Runs with fixture providers by default so it works with no API keys - that is
 * how CI exercises the whole path. Supply keys via environment variables to use
 * the real providers.
 *
 *   tsx src/generate.ts --input "a story about..." --out ../../prototype/out/generated
 *   PROVIDERS=live DEEPSEEK_API_KEY=... IMAGE_API_KEY=... TTS_API_KEY=... tsx src/generate.ts ...
 */

import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { dirname } from "node:path";

import { runPipeline } from "./pipeline";
import { evaluateTier, renderCost } from "./cost";
import { DeepSeekProvider } from "./providers/deepseek";
import {
  HostedImageProvider,
  HostedSegmentationProvider,
  HostedTtsProvider,
} from "./providers/media";
import {
  FixtureImageProvider,
  FixtureLlmProvider,
  FixtureSegmentationProvider,
  FixtureTtsProvider,
} from "./providers/fixtures";
import type { ProviderSet } from "./providers";
import type { GenerationRequest } from "./types";

const HERE = dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = resolve(HERE, "../../..");

function parseArgs() {
  const args = new Map<string, string>();
  for (let i = 2; i < process.argv.length; i += 2) {
    args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);
  }
  return {
    input: args.get("input") ?? "The boy who cried wolf",
    out: resolve(process.cwd(), args.get("out") ?? "./generated"),
    aspect: (args.get("aspect") ?? "portrait") as GenerationRequest["aspect"],
    tier: (args.get("tier") ?? "cheap") as GenerationRequest["tier"],
    language: args.get("language") ?? "en",
    scenes: Number(args.get("scenes") ?? 4),
    segmentation: args.get("segmentation") !== "false",
  };
}

export function buildProviders(mode: string, tier: GenerationRequest["tier"]): ProviderSet {
  if (mode !== "live") {
    const options = { repoRoot: REPO_ROOT };
    return {
      llm: new FixtureLlmProvider(),
      image: new FixtureImageProvider(options),
      tts: new FixtureTtsProvider(options),
      segmentation: new FixtureSegmentationProvider(options),
    };
  }

  const require = (name: string) => {
    const value = process.env[name];
    if (!value) throw new Error(`${name} is required when PROVIDERS=live`);
    return value;
  };

  return {
    llm: new DeepSeekProvider({ apiKey: require("DEEPSEEK_API_KEY") }),
    image: new HostedImageProvider({ apiKey: require("IMAGE_API_KEY"), tier }),
    tts: new HostedTtsProvider({ apiKey: require("TTS_API_KEY"), tier }),
    segmentation: process.env.SEGMENTATION_API_KEY
      ? new HostedSegmentationProvider({ apiKey: process.env.SEGMENTATION_API_KEY })
      : null,
  };
}

async function main() {
  const options = parseArgs();
  const mode = process.env.PROVIDERS ?? "fixture";

  const request: GenerationRequest = {
    input: options.input,
    aspect: options.aspect,
    tier: options.tier,
    language: options.language,
    sceneCount: options.scenes,
    segmentation: options.segmentation,
  };

  console.log(`providers: ${mode}   tier: ${request.tier}   aspect: ${request.aspect}`);

  const result = await runPipeline({
    request,
    providers: buildProviders(mode, request.tier),
    outDir: options.out,
    onProgress: (stage, detail) => console.log(`  ${stage}${detail ? `: ${detail}` : ""}`),
  });

  const narrationSeconds = result.scenes.reduce((sum, s) => sum + s.durationSeconds, 0);
  const render = renderCost(narrationSeconds);
  const total = result.cost.total + render;

  console.log(`\n${result.storyboard.title}`);
  for (const scene of result.scenes) {
    console.log(
      `  scene ${scene.index + 1}: ${scene.durationSeconds.toFixed(2)}s narration, ` +
        `masks ${scene.mask ? "yes" : "no"}`,
    );
  }

  console.log("\ncost breakdown (USD)");
  for (const [operation, cost] of Object.entries(result.cost.byOperation)) {
    console.log(`  ${operation.padEnd(14)} ${cost.toFixed(4)}`);
  }
  console.log(`  ${"render".padEnd(14)} ${render.toFixed(4)}`);
  console.log(`  ${"TOTAL".padEnd(14)} ${total.toFixed(4)}`);

  // The check that caught Studio being underwater at its planned quota.
  const tiers = [
    { name: "Pro", tier: request.tier, price: 19, quota: 30, feeRate: 0.05 },
    { name: "Studio", tier: request.tier, price: 49, quota: 50, feeRate: 0.05 },
  ];
  console.log("\nplan margin at full quota use");
  for (const plan of tiers) {
    const margin = evaluateTier(
      { tier: plan.tier, price: plan.price, quota: plan.quota, feeRate: plan.feeRate },
      total,
    );
    console.log(
      `  ${plan.name.padEnd(7)} $${plan.price}/${plan.quota} videos  ` +
        `COGS $${margin.cogsAtFullUse.toFixed(2)}  margin ${(margin.marginAtFullUse * 100).toFixed(0)}%  ` +
        `${margin.profitable ? "ok" : "LOSS-MAKING"}`,
    );
  }

  console.log(`\nfixture -> ${result.fixturePath}`);
  console.log(
    "render it with:\n" +
      `  cd prototype/remotion-renderer && npx tsx scripts/buildVideo.ts --fixture ${result.fixturePath} --name generated --aspect ${request.aspect}`,
  );
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
