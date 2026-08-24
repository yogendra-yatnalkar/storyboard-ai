/**
 * Caption sourcing.
 *
 * Timings come from different places depending on the voice provider: most neural
 * TTS APIs return word timings with the audio (free and exact), while gTTS - the
 * current free path - returns none, so those need transcribing. Both sit behind
 * this interface so the renderer never knows which it got.
 */

import type { CaptionWord } from "../drawing/manifest";

export type { CaptionWord };

export interface CaptionRequest {
  /** Absolute path to the narration audio. */
  audioPath: string;
  /** The narration text, when known - lets a source align rather than transcribe. */
  text?: string;
  /** Audio duration in seconds, used for validation and proportional fallbacks. */
  durationSeconds: number;
}

export interface CaptionSource {
  readonly name: string;
  load(request: CaptionRequest): Promise<CaptionWord[]>;
}

/** Group a flat word list into caption lines of at most `maxChars`. */
export function groupIntoLines(
  words: CaptionWord[],
  maxChars = 42,
): { words: CaptionWord[]; text: string; startMs: number; endMs: number }[] {
  const lines: { words: CaptionWord[]; text: string; startMs: number; endMs: number }[] = [];
  let current: CaptionWord[] = [];

  const flush = () => {
    if (current.length === 0) return;
    lines.push({
      words: current,
      text: current.map((w) => w.text).join(" "),
      startMs: current[0].startMs,
      endMs: current[current.length - 1].endMs,
    });
    current = [];
  };

  for (const word of words) {
    const candidate = [...current, word].map((w) => w.text).join(" ");
    if (current.length > 0 && candidate.length > maxChars) flush();
    current.push(word);
    // Break on sentence-ending punctuation so lines follow the narration's phrasing.
    if (/[.!?]$/.test(word.text)) flush();
  }
  flush();

  return lines;
}

/** Words active at a given time, plus which one is currently being spoken. */
export function wordAt(words: CaptionWord[], timeMs: number): number {
  for (let i = 0; i < words.length; i++) {
    if (timeMs >= words[i].startMs && timeMs < words[i].endMs) return i;
  }
  return -1;
}
