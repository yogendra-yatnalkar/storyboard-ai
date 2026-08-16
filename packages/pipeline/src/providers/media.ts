/**
 * Image, speech and segmentation providers.
 *
 * All three are tier-aware: the cheap path is what free and Pro renders use, and
 * the premium path is the Studio upsell. The tier is a constructor argument
 * rather than a separate class so switching costs nothing at the call site.
 *
 * Pricing constants live next to the code that incurs them so docs/cost-model.md
 * can be checked against reality instead of being a separate guess.
 */

import { writeFile } from "node:fs/promises";
import { join } from "node:path";

import { IMAGE_SIZES, type ImageProvider, type ProviderResult, type SegmentationProvider, type TtsProvider } from "./index";
import type {
  GeneratedAudio,
  GeneratedImage,
  GeneratedMasks,
  GenerationRequest,
  Scene,
} from "../types";

/** USD per unit, at list price. */
export const MEDIA_PRICING = {
  image: { cheap: 0.02, premium: 0.05 },
  ttsPerThousandChars: { cheap: 0.016, premium: 0.22 },
  segmentationPerImage: 0.01,
};

// --- images -------------------------------------------------------------------

export interface HostedImageOptions {
  apiKey: string;
  tier: GenerationRequest["tier"];
  /** Model slug, e.g. a Fal or Replicate endpoint. */
  model?: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
}

/**
 * Hosted diffusion over an OpenAI-shaped HTTP endpoint (Fal, Replicate and
 * friends all accept a JSON body and return image URLs).
 */
export class HostedImageProvider implements ImageProvider {
  readonly name: string;
  private readonly fetchImpl: typeof fetch;

  constructor(private readonly options: HostedImageOptions) {
    this.name = `hosted-image:${options.tier}`;
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async generate(
    scene: Scene,
    options: { aspect: GenerationRequest["aspect"]; tier: GenerationRequest["tier"]; outDir: string },
  ): Promise<ProviderResult<GeneratedImage>> {
    const size = IMAGE_SIZES[options.aspect];

    const response = await this.fetchImpl(
      this.options.baseUrl ?? "https://fal.run/fal-ai/fast-sdxl",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Key ${this.options.apiKey}`,
        },
        body: JSON.stringify({
          prompt: buildImagePrompt(scene.imagePrompt),
          negative_prompt: NEGATIVE_PROMPT,
          image_size: { width: size.width, height: size.height },
          num_images: 1,
        }),
      },
    );

    if (!response.ok) {
      throw new Error(`image generation failed ${response.status}: ${await response.text()}`);
    }

    const body = (await response.json()) as { images: { url: string }[] };
    const imageResponse = await this.fetchImpl(body.images[0].url);
    const path = join(options.outDir, `scene-${scene.index + 1}.png`);
    await writeFile(path, Buffer.from(await imageResponse.arrayBuffer()));

    return {
      value: { sceneIndex: scene.index, path, width: size.width, height: size.height },
      usage: [
        {
          provider: this.name,
          operation: "image",
          cost: MEDIA_PRICING.image[options.tier],
          quantity: 1,
          unit: "images",
        },
      ],
    };
  }
}

/**
 * The renderer thresholds the image into line art, so what matters is clean
 * edges and separable subjects - not photographic realism. Steering the prompt
 * here is far cheaper than compensating in the renderer.
 */
export function buildImagePrompt(prompt: string): string {
  return `${prompt}. Clean illustration, clear outlines, distinct separated subjects, plain white background, high contrast, no text.`;
}

const NEGATIVE_PROMPT =
  "text, watermark, signature, blurry, low contrast, busy background, gradient background, photorealistic noise";

// --- speech -------------------------------------------------------------------

export interface HostedTtsOptions {
  apiKey: string;
  tier: GenerationRequest["tier"];
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  /** Set when the provider returns word boundaries, which skips transcription. */
  providesWordTimings?: boolean;
}

export class HostedTtsProvider implements TtsProvider {
  readonly name: string;
  readonly providesWordTimings: boolean;
  private readonly fetchImpl: typeof fetch;

  constructor(private readonly options: HostedTtsOptions) {
    this.name = `hosted-tts:${options.tier}`;
    this.providesWordTimings = options.providesWordTimings ?? options.tier === "premium";
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async speak(
    scene: Scene,
    options: { voice?: string; language: string; outDir: string },
  ): Promise<ProviderResult<GeneratedAudio>> {
    const response = await this.fetchImpl(this.options.baseUrl ?? "https://api.tts.example/v1/speak", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        authorization: `Bearer ${this.options.apiKey}`,
      },
      body: JSON.stringify({
        text: scene.narration,
        voice: options.voice,
        language: options.language,
        include_word_timings: this.providesWordTimings,
      }),
    });

    if (!response.ok) {
      throw new Error(`tts failed ${response.status}: ${await response.text()}`);
    }

    const body = (await response.json()) as {
      audioUrl: string;
      durationSeconds: number;
      words?: { text: string; offsetMs: number; durationMs: number }[];
    };

    const audioResponse = await this.fetchImpl(body.audioUrl);
    const path = join(options.outDir, `scene-${scene.index + 1}.mp3`);
    await writeFile(path, Buffer.from(await audioResponse.arrayBuffer()));

    return {
      value: {
        sceneIndex: scene.index,
        path,
        durationSeconds: body.durationSeconds,
        wordTimings: body.words,
      },
      usage: [
        {
          provider: this.name,
          operation: "tts",
          cost:
            (scene.narration.length / 1000) *
            MEDIA_PRICING.ttsPerThousandChars[this.options.tier],
          quantity: scene.narration.length,
          unit: "chars",
        },
      ],
    };
  }
}

// --- segmentation -------------------------------------------------------------

export interface HostedSegmentationOptions {
  apiKey: string;
  baseUrl?: string;
  fetchImpl?: typeof fetch;
  /** Masks smaller than this share of the frame are dropped as noise. */
  minAreaFraction?: number;
  maxObjects?: number;
}

/**
 * Automatic object masking (SAM-shaped).
 *
 * This is the differentiator - drawing object by object rather than sweeping the
 * whole frame - and also ~25% of cheap-tier COGS, so it is optional per project
 * and the pipeline runs without it.
 */
export class HostedSegmentationProvider implements SegmentationProvider {
  readonly name = "hosted-segmentation";
  private readonly fetchImpl: typeof fetch;

  constructor(private readonly options: HostedSegmentationOptions) {
    this.fetchImpl = options.fetchImpl ?? fetch;
  }

  async segment(
    image: GeneratedImage,
    options: { outDir: string },
  ): Promise<ProviderResult<GeneratedMasks>> {
    const response = await this.fetchImpl(
      this.options.baseUrl ?? "https://api.segmentation.example/v1/segment",
      {
        method: "POST",
        headers: {
          "content-type": "application/json",
          authorization: `Bearer ${this.options.apiKey}`,
        },
        body: JSON.stringify({ imagePath: image.path }),
      },
    );

    if (!response.ok) {
      throw new Error(`segmentation failed ${response.status}`);
    }

    const body = (await response.json()) as {
      masks: { label?: string; polygon: [number, number][] }[];
    };

    const minArea = (this.options.minAreaFraction ?? 0.005) * image.width * image.height;
    const shapes = body.masks
      .filter((mask) => polygonArea(mask.polygon) >= minArea)
      // Largest first: the renderer draws in file order, and drawing the biggest
      // subject first reads more like a person sketching than starting on detail.
      .sort((a, b) => polygonArea(b.polygon) - polygonArea(a.polygon))
      .slice(0, this.options.maxObjects ?? 8)
      .map((mask, index) => ({ label: mask.label ?? String(index + 1), points: mask.polygon }));

    // Written in the labelme/AnyLabeling shape the renderer already consumes.
    const path = join(options.outDir, `scene-${image.sceneIndex + 1}.json`);
    await writeFile(
      path,
      JSON.stringify({
        version: "5.0.0",
        shapes,
        imageWidth: image.width,
        imageHeight: image.height,
      }),
    );

    return {
      value: { sceneIndex: image.sceneIndex, path },
      usage: [
        {
          provider: this.name,
          operation: "segmentation",
          cost: MEDIA_PRICING.segmentationPerImage,
          quantity: 1,
          unit: "images",
        },
      ],
    };
  }
}

/** Shoelace formula. */
export function polygonArea(points: [number, number][]): number {
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const [x1, y1] = points[i];
    const [x2, y2] = points[(i + 1) % points.length];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area) / 2;
}
