/**
 * Free-tier watermark and brand mark.
 *
 * Deliberately corner-anchored and semi-transparent rather than centred: it has
 * to be present enough that removing it is worth paying for, without making the
 * free output useless to look at. The same component renders a paying customer's
 * logo, so there is one placement rule rather than two.
 */

import { Img, staticFile, useVideoConfig } from "remotion";

import type { BrandKit } from "./drawing/manifest";

export interface WatermarkProps {
  /** Show the product watermark (free tier). */
  watermark: boolean;
  brand?: BrandKit;
  label?: string;
}

export const Watermark: React.FC<WatermarkProps> = ({
  watermark,
  brand,
  label = "storyboard-ai",
}) => {
  const { width, height } = useVideoConfig();
  const unit = Math.min(width, height);

  if (!watermark && !brand?.logo) return null;

  return (
    <div
      style={{
        position: "absolute",
        top: unit * 0.035,
        right: unit * 0.035,
        display: "flex",
        alignItems: "center",
        gap: unit * 0.014,
      }}
    >
      {brand?.logo ? (
        <Img
          src={staticFile(brand.logo)}
          style={{ height: unit * 0.06, objectFit: "contain" }}
        />
      ) : null}

      {watermark ? (
        <span
          style={{
            fontFamily: brand?.captionFont ?? "Inter, Helvetica, Arial, sans-serif",
            fontSize: unit * 0.026,
            fontWeight: 700,
            letterSpacing: unit * 0.0006,
            color: "rgba(17,17,17,0.55)",
            backgroundColor: "rgba(255,255,255,0.6)",
            padding: `${unit * 0.008}px ${unit * 0.016}px`,
            borderRadius: unit * 0.008,
          }}
        >
          {label}
        </span>
      ) : null}
    </div>
  );
};
