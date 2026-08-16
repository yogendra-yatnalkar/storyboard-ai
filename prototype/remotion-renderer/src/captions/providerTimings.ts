/**
 * Caption timings supplied by the TTS provider.
 *
 * Most neural TTS APIs (Azure, Google, ElevenLabs, OpenAI) return word or
 * phoneme boundaries alongside the audio, which is both free and exact - no
 * transcription pass, no drift. This is the preferred source once a provider is
 * chosen in Phase 2; the Whisper path stays as the universal fallback for
 * providers that return nothing (gTTS, the current free tier).
 *
 * The adapter is here now so the renderer already depends on the interface
 * rather than on Whisper specifically.
 */

import type { CaptionRequest, CaptionSource, CaptionWord } from "./types";

/** Normalised shape a TTS adapter is expected to produce. */
export interface ProviderWordBoundary {
  text: string;
  /** Offset from the start of the audio. */
  offsetMs: number;
  durationMs: number;
}

export class ProviderTimingSource implements CaptionSource {
  readonly name = "provider";

  constructor(
    private readonly fetchBoundaries: (
      request: CaptionRequest,
    ) => Promise<ProviderWordBoundary[]>,
  ) {}

  async load(request: CaptionRequest): Promise<CaptionWord[]> {
    const boundaries = await this.fetchBoundaries(request);
    return boundaries.map((boundary) => ({
      text: boundary.text,
      startMs: boundary.offsetMs,
      endMs: boundary.offsetMs + boundary.durationMs,
    }));
  }
}

/**
 * Try sources in order, taking the first that returns usable timings.
 * Lets a project prefer provider timings while still rendering captions when the
 * provider has none.
 */
export class FallbackCaptionSource implements CaptionSource {
  readonly name = "fallback";

  constructor(private readonly sources: CaptionSource[]) {}

  async load(request: CaptionRequest): Promise<CaptionWord[]> {
    for (const source of this.sources) {
      try {
        const words = await source.load(request);
        if (words.length > 0) return words;
      } catch {
        // A source that cannot answer is not an error - that is what the chain is for.
      }
    }
    return [];
  }
}
