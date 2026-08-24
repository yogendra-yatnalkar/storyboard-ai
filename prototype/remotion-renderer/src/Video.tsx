/**
 * Multi-scene video composition.
 *
 * This is what replaces ffmpeg-command-notes.txt in its entirety: no ffprobe
 * duration checks, no per-scene `apad=pad_dur=1.19`, no concat list. Scene
 * offsets come from the manifest, which was built by solving drawing speed
 * against narration length, so audio and video line up by construction.
 *
 * Everything stays a pure function of frame, so Remotion Lambda can still render
 * disjoint frame ranges in parallel across the whole video.
 */

import { AbsoluteFill, Audio, Sequence, staticFile } from "remotion";
import { useEffect, useState } from "react";
import { continueRender, delayRender } from "remotion";

import { DrawingScene } from "./DrawingScene";
import { Captions } from "./captions/Captions";
import { Watermark } from "./Watermark";
import type { VideoManifest } from "./drawing/manifest";

export interface VideoProps {
  /** Folder under public/precomputed holding video.json. */
  video: string;
  [key: string]: unknown;
}

export async function fetchVideoManifest(video: string): Promise<VideoManifest> {
  const response = await fetch(staticFile(`precomputed/${video}/video.json`));
  return (await response.json()) as VideoManifest;
}

export const StoryVideo: React.FC<VideoProps> = ({ video }) => {
  const [manifest, setManifest] = useState<VideoManifest | null>(null);
  const [handle] = useState(() => delayRender(`loading video manifest ${video}`));

  useEffect(() => {
    let cancelled = false;
    fetchVideoManifest(video)
      .then((loaded) => {
        if (cancelled) return;
        setManifest(loaded);
        continueRender(handle);
      })
      .catch((error) => {
        console.error(error);
        continueRender(handle);
      });
    return () => {
      cancelled = true;
    };
  }, [video, handle]);

  if (!manifest) return <AbsoluteFill style={{ backgroundColor: "white" }} />;

  const background = manifest.brand?.background ?? "white";

  return (
    <AbsoluteFill style={{ backgroundColor: background }}>
      {manifest.scenes.map((scene) => (
        <Sequence
          key={scene.name}
          from={scene.startFrame}
          durationInFrames={scene.durationInFrames}
          name={scene.name}
        >
          <DrawingScene name={scene.name} background={background} />

          {scene.audio ? <Audio src={staticFile(scene.audio)} /> : null}

          <Captions
            words={scene.captions}
            style={manifest.captionStyle}
            brand={manifest.brand}
          />
        </Sequence>
      ))}

      {/*
        Music spans the whole video rather than per scene, so it does not restart
        at every cut. The gain is a fixed duck under narration; Phase 5 replaces
        it with sidechain-style envelope following.
      */}
      {manifest.music ? (
        <Audio
          src={staticFile(manifest.music)}
          volume={dbToGain(manifest.musicGainDb)}
          loop
        />
      ) : null}

      <Watermark watermark={manifest.watermark} brand={manifest.brand} />
    </AbsoluteFill>
  );
};

function dbToGain(db: number): number {
  return Math.pow(10, db / 20);
}
