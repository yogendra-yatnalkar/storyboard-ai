/**
 * Caption renderers.
 *
 * Two styles, because the two output formats have different conventions:
 * word-by-word highlighting is what short-form vertical video uses (and it is
 * what keeps muted autoplay legible), while landscape explainers read better with
 * calm phrase blocks.
 *
 * Both wrap. The pipeline this replaces used cv2.putText with no wrapping at all,
 * so any long line simply ran off the frame.
 */

import { useCurrentFrame, useVideoConfig } from "remotion";

import type { BrandKit, CaptionStyle, CaptionWord } from "../drawing/manifest";
import { groupIntoLines, wordAt } from "./types";

export interface CaptionsProps {
  words: CaptionWord[];
  style: CaptionStyle;
  brand?: BrandKit;
}

const SAFE_AREA_BOTTOM = 0.12;

export const Captions: React.FC<CaptionsProps> = ({ words, style, brand }) => {
  const frame = useCurrentFrame();
  const { fps, width, height } = useVideoConfig();

  if (style === "none" || words.length === 0) return null;

  const timeMs = (frame / fps) * 1000;
  // Scale with the frame's short edge so one style works at every resolution.
  const unit = Math.min(width, height);
  const accent = brand?.accentColor ?? "#ffd60a";
  const fontFamily = brand?.captionFont ?? "Inter, Helvetica, Arial, sans-serif";

  if (style === "word") {
    const active = wordAt(words, timeMs);
    if (active < 0) return null;

    // A short window around the spoken word: enough context to read, not so much
    // that the eye has to search for the highlight.
    const from = Math.max(0, active - 2);
    const window = words.slice(from, from + 5);

    return (
      <div
        style={{
          position: "absolute",
          left: 0,
          right: 0,
          bottom: height * SAFE_AREA_BOTTOM,
          display: "flex",
          flexWrap: "wrap",
          justifyContent: "center",
          gap: unit * 0.018,
          padding: `0 ${unit * 0.08}px`,
          fontFamily,
          fontSize: unit * 0.062,
          fontWeight: 800,
          lineHeight: 1.2,
          textAlign: "center",
        }}
      >
        {window.map((word, i) => {
          const isActive = from + i === active;
          return (
            <span
              key={`${word.startMs}-${i}`}
              style={{
                color: isActive ? "#111111" : "#ffffff",
                backgroundColor: isActive ? accent : "rgba(17,17,17,0.82)",
                padding: `${unit * 0.008}px ${unit * 0.018}px`,
                borderRadius: unit * 0.012,
                // Keeps white text legible over pale paper before the box paints.
                textShadow: isActive ? "none" : "0 2px 6px rgba(0,0,0,0.45)",
              }}
            >
              {word.text}
            </span>
          );
        })}
      </div>
    );
  }

  // Sentence blocks: one phrase at a time, wrapped, no yellow box.
  const lines = groupIntoLines(words);
  const line = lines.find((l) => timeMs >= l.startMs && timeMs < l.endMs);
  if (!line) return null;

  return (
    <div
      style={{
        position: "absolute",
        left: 0,
        right: 0,
        bottom: height * SAFE_AREA_BOTTOM * 0.6,
        display: "flex",
        justifyContent: "center",
        padding: `0 ${unit * 0.09}px`,
      }}
    >
      <span
        style={{
          fontFamily,
          fontSize: unit * 0.042,
          fontWeight: 600,
          lineHeight: 1.35,
          color: "#ffffff",
          backgroundColor: "rgba(17,17,17,0.84)",
          padding: `${unit * 0.016}px ${unit * 0.028}px`,
          borderRadius: unit * 0.014,
          textAlign: "center",
          maxWidth: "100%",
        }}
      >
        {line.text}
      </span>
    </div>
  );
};
