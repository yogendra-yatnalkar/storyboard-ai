/**
 * Composition registry.
 *
 * Dimensions and duration come from the built manifests rather than being
 * hardcoded, so a project's aspect ratio is a build-time choice and the timeline
 * length is derived from the drawing and its narration.
 *
 * `Whiteboard-Square` is the parity target and must stay pinned to the reference
 * renderer's dimensions and fixed skip rates - the parity harness compares
 * against it, and it is the regression net for every change here.
 */

import { Composition } from "remotion";

import { DrawingScene, fetchManifest, type DrawingSceneProps } from "./DrawingScene";
import { StoryVideo, fetchVideoManifest, type VideoProps } from "./Video";

const PARITY_SCENE = "scene-2";
const STORY = "story";

const sceneMetadata = async ({ props }: { props: DrawingSceneProps }) => {
  const manifest = await fetchManifest(props.name);
  return {
    width: manifest.width,
    height: manifest.height,
    durationInFrames: manifest.durationInFrames,
    fps: manifest.config.frameRate,
  };
};

const videoMetadata = async ({ props }: { props: VideoProps }) => {
  const manifest = await fetchVideoManifest(props.video);
  return {
    width: manifest.width,
    height: manifest.height,
    durationInFrames: manifest.durationInFrames,
    fps: manifest.fps,
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
        defaultProps={{ name: PARITY_SCENE }}
        calculateMetadata={sceneMetadata}
      />

      {/* Single scene, dimensions taken from whatever was precomputed. */}
      <Composition
        id="Scene"
        component={DrawingScene}
        width={1020}
        height={1020}
        fps={25}
        durationInFrames={1}
        defaultProps={{ name: PARITY_SCENE }}
        calculateMetadata={sceneMetadata}
      />

      {/* The full narrated, captioned video. */}
      <Composition
        id="Story"
        component={StoryVideo}
        width={1020}
        height={1020}
        fps={25}
        durationInFrames={1}
        defaultProps={{ video: STORY }}
        calculateMetadata={videoMetadata}
      />

      <Composition
        id="Story-Landscape"
        component={StoryVideo}
        width={1920}
        height={1080}
        fps={25}
        durationInFrames={1}
        defaultProps={{ video: "story-landscape" }}
        calculateMetadata={videoMetadata}
      />

      <Composition
        id="Story-Portrait"
        component={StoryVideo}
        width={1080}
        height={1920}
        fps={25}
        durationInFrames={1}
        defaultProps={{ video: "story-portrait" }}
        calculateMetadata={videoMetadata}
      />
    </>
  );
};
