import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { groupIntoLines, wordAt, type CaptionWord } from "./types";
import { ProportionalCaptionSource, splitByScenes } from "./whisperTranscript";

const words = (texts: string[], msPerWord = 400): CaptionWord[] =>
  texts.map((text, i) => ({
    text,
    startMs: i * msPerWord,
    endMs: (i + 1) * msPerWord,
  }));

describe("splitByScenes", () => {
  it("assigns each transcribed word to the scene whose text it belongs to", () => {
    const transcript = words(["once", "there", "was", "a", "boy", "the", "end", "came"]);
    const [first, second] = splitByScenes(transcript, ["Once there was a boy", "The end came"]);

    assert.deepEqual(
      first.map((w) => w.text),
      ["once", "there", "was", "a", "boy"],
    );
    assert.deepEqual(
      second.map((w) => w.text),
      ["the", "end", "came"],
    );
  });

  it("rebases each scene to its own timeline", () => {
    const transcript = words(["a", "b", "c", "d"]);
    const [, second] = splitByScenes(transcript, ["a b", "c d"], { leadInMs: 100 });

    // Scene two starts at 800ms in the combined recording but gets its own audio
    // file, so its captions must start near zero.
    assert.equal(second[0].startMs, 100);
    assert.ok(second[1].startMs < 600);
  });

  it("resyncs when the transcript inserts a word the source text lacks", () => {
    const transcript = words(["once", "uh", "there", "was", "the", "end"]);
    const [first, second] = splitByScenes(transcript, ["Once there was", "The end"]);

    assert.deepEqual(
      first.map((w) => w.text),
      ["once", "uh", "there", "was"],
    );
    assert.deepEqual(
      second.map((w) => w.text),
      ["the", "end"],
    );
  });

  it("ignores punctuation and case when matching", () => {
    const transcript = words(["\"Wolf!", "Wolf!\"", "he", "cried"]);
    const [only] = splitByScenes(transcript, ["wolf wolf he cried"]);
    assert.equal(only.length, 4);
  });

  it("returns an empty list for a scene with no matching words", () => {
    const result = splitByScenes(words(["a", "b"]), ["a b", "totally different text"]);
    assert.equal(result[1].length, 0);
  });
});

describe("groupIntoLines", () => {
  it("wraps rather than letting a line run off frame", () => {
    const long = words("the quick brown fox jumps over the lazy dog again and again".split(" "));
    const lines = groupIntoLines(long, 20);

    assert.ok(lines.length > 1, "a long narration must wrap");
    for (const line of lines) {
      assert.ok(
        line.text.length <= 20 + 12,
        `line "${line.text}" is too long to fit the caption box`,
      );
    }
  });

  it("breaks on sentence endings so lines follow the phrasing", () => {
    const lines = groupIntoLines(words(["Hello.", "A", "new", "thought"]), 80);
    assert.equal(lines[0].text, "Hello.");
    assert.equal(lines[1].text, "A new thought");
  });

  it("carries the span of its words", () => {
    const [line] = groupIntoLines(words(["one", "two"]), 80);
    assert.equal(line.startMs, 0);
    assert.equal(line.endMs, 800);
  });
});

describe("wordAt", () => {
  const list = words(["a", "b", "c"]);

  it("finds the word being spoken", () => {
    assert.equal(wordAt(list, 0), 0);
    assert.equal(wordAt(list, 450), 1);
    assert.equal(wordAt(list, 1199), 2);
  });

  it("returns -1 outside the narration", () => {
    assert.equal(wordAt(list, 5000), -1);
  });
});

describe("ProportionalCaptionSource", () => {
  it("spreads words across the audio without exceeding its duration", async () => {
    const source = new ProportionalCaptionSource();
    const result = await source.load({
      audioPath: "",
      text: "one two three four five",
      durationSeconds: 5,
    });

    assert.equal(result.length, 5);
    assert.equal(result[0].startMs, 0);
    assert.ok(Math.abs(result[4].endMs - 5000) < 1);
  });

  it("returns nothing when there is no text to time", async () => {
    const result = await new ProportionalCaptionSource().load({
      audioPath: "",
      durationSeconds: 5,
    });
    assert.deepEqual(result, []);
  });
});
