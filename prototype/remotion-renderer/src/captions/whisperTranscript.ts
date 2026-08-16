/**
 * Caption timings from a Whisper transcript.
 *
 * Whisper run with word timestamps emits `segments[].words[]` with `text`,
 * `start` and `end` - the format already committed at
 * `adding-subtitles/transcript.json`. That file covers the whole four-scene
 * narration as one concatenated recording, so it also needs splitting back into
 * scenes; `splitByScenes` does that by aligning the transcript against the known
 * narration text rather than guessing at the ffmpeg padding that produced it.
 *
 * Whisper itself is deliberately NOT a dependency here: torch plus a model is
 * ~2 GB, and the pipeline only needs the JSON. See docs/testing.md for the
 * command that regenerates it.
 */

import { readFileSync } from "node:fs";

import type { CaptionRequest, CaptionSource, CaptionWord } from "./types";

interface WhisperWord {
  text?: string;
  word?: string;
  start: number;
  end: number;
}

interface WhisperSegment {
  start: number;
  end: number;
  text: string;
  words?: WhisperWord[];
}

interface WhisperTranscript {
  segments: WhisperSegment[];
}

const normalise = (value: string) => value.toLowerCase().replace(/[^a-z0-9']/g, "");

/** Flatten a transcript to a word list, falling back to segments when needed. */
export function loadWhisperWords(path: string): CaptionWord[] {
  const transcript = JSON.parse(readFileSync(path, "utf8")) as WhisperTranscript;
  const words: CaptionWord[] = [];

  for (const segment of transcript.segments) {
    if (segment.words?.length) {
      for (const word of segment.words) {
        const text = (word.text ?? word.word ?? "").trim();
        if (text) {
          words.push({ text, startMs: word.start * 1000, endMs: word.end * 1000 });
        }
      }
    } else {
      // Segment-level fallback: spread the segment's words across its span.
      const tokens = segment.text.trim().split(/\s+/).filter(Boolean);
      const step = (segment.end - segment.start) / Math.max(tokens.length, 1);
      tokens.forEach((token, i) => {
        words.push({
          text: token,
          startMs: (segment.start + i * step) * 1000,
          endMs: (segment.start + (i + 1) * step) * 1000,
        });
      });
    }
  }

  return words;
}

/**
 * Split one transcript across several scenes using the narration text.
 *
 * A two-pointer alignment with a small resync window: the transcript and the
 * source text are near-identical (the audio was synthesised from that text), so
 * mismatches are occasional insertions or dropped words rather than divergence.
 * Times are rebased so each scene starts near zero, since each scene now gets its
 * own audio track.
 */
export function splitByScenes(
  words: CaptionWord[],
  sceneTexts: string[],
  options: { leadInMs?: number } = {},
): CaptionWord[][] {
  const leadInMs = options.leadInMs ?? 150;
  const transcriptTokens = words.map((word) => normalise(word.text));
  const result: CaptionWord[][] = [];

  let cursor = 0;

  for (const sceneText of sceneTexts) {
    const expected = sceneText
      .split(/\s+/)
      .map(normalise)
      .filter(Boolean);

    const start = cursor;
    let expectedIndex = 0;

    while (expectedIndex < expected.length && cursor < words.length) {
      if (transcriptTokens[cursor] === expected[expectedIndex]) {
        cursor++;
        expectedIndex++;
        continue;
      }
      // Resync: an inserted transcript word, or a word the transcript dropped.
      const ahead = transcriptTokens
        .slice(cursor, cursor + 3)
        .indexOf(expected[expectedIndex]);
      if (ahead > 0) cursor += ahead;
      else expectedIndex++;
    }

    const sceneWords = words.slice(start, cursor);
    if (sceneWords.length === 0) {
      result.push([]);
      continue;
    }

    const offset = sceneWords[0].startMs - leadInMs;
    result.push(
      sceneWords.map((word) => ({
        text: word.text,
        startMs: Math.max(0, word.startMs - offset),
        endMs: Math.max(0, word.endMs - offset),
      })),
    );
  }

  return result;
}

/** Reads a per-scene Whisper JSON sitting next to the audio. */
export class WhisperTranscriptSource implements CaptionSource {
  readonly name = "whisper";

  constructor(private readonly transcriptFor: (request: CaptionRequest) => string) {}

  async load(request: CaptionRequest): Promise<CaptionWord[]> {
    return loadWhisperWords(this.transcriptFor(request));
  }
}

/**
 * Last-resort source: spread the narration text evenly across the audio.
 * Costs nothing and needs no provider, but drifts audibly on longer scenes - use
 * only when neither provider timings nor a transcript are available.
 */
export class ProportionalCaptionSource implements CaptionSource {
  readonly name = "proportional";

  async load(request: CaptionRequest): Promise<CaptionWord[]> {
    const tokens = (request.text ?? "").split(/\s+/).filter(Boolean);
    if (tokens.length === 0) return [];

    const totalMs = request.durationSeconds * 1000;
    // Weight by word length so long words hold the screen longer than short ones.
    const weights = tokens.map((token) => Math.max(1, token.length));
    const totalWeight = weights.reduce((a, b) => a + b, 0);

    let elapsed = 0;
    return tokens.map((token, i) => {
      const span = (weights[i] / totalWeight) * totalMs;
      const word = { text: token, startMs: elapsed, endMs: elapsed + span };
      elapsed += span;
      return word;
    });
  }
}
