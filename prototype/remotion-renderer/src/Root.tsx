/**
 * Composition registry.
 *
 * Duration and fps come from the precomputed manifest via calculateMetadata, so
 * the timeline length is derived from the drawing itself rather than hardcoded -
 * the same mechanism that will later let narration length drive the skip rate
 * instead of the ffmpeg `apad` guesswork in the current pipeline.
 *
 * The square composition is the parity target. The 16:9 and 9:16 entries prove
 * the aspect-ratio switch the product needs; the square artwork is letterboxed
 * for now, and Phase 1 replaces that with a proper per-ratio scene layout.
 */

import { Composition } from "remotion";

import { DrawingScene, fetchManifest, type DrawingSceneProps } from "./DrawingScene";

const SCENE = "scene-2";

const calculateMetadata = async ({ props }: { props: DrawingSceneProps }) => {
  const manifest = await fetchManifest(props.name);
  return {
    durationInFrames: manifest.durationInFrames,
    fps: manifest.config.frameRate,
  };
};

export const RemotionRoot: React.FC = () => {
  return (
    <>
      {/* Parity target: identical dimensions to the Python reference render. */}
      <Composition
        id="Whiteboard-Square"
        component={DrawingScene}
        width={1020}
        height={1020}
        fps={25}
        durationInFrames={1}
        defaultProps={{ name: SCENE }}
        calculateMetadata={calculateMetadata}
      />
      <Composition
        id="Whiteboard-Landscape"
        component={DrawingScene}
        width={1920}
        height={1080}
        fps={25}
        durationInFrames={1}
        defaultProps={{ name: SCENE }}
        calculateMetadata={calculateMetadata}
      />
      <Composition
        id="Whiteboard-Portrait"
        component={DrawingScene}
        width={1080}
        height={1920}
        fps={25}
        durationInFrames={1}
        defaultProps={{ name: SCENE }}
        calculateMetadata={calculateMetadata}
      />
    </>
  );
};
