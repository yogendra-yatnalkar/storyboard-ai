/**
 * Remotion composition for the whiteboard drawing.
 *
 * The component is a thin shell: all the drawing logic lives in `renderFrame`,
 * the same pure function the parity harness measures against the Python
 * reference. Rendering through it - rather than reimplementing the reveal with
 * canvas drawImage calls - means the video Remotion produces is exactly what the
 * parity numbers describe, with no second implementation to drift.
 *
 * Assets are decoded once and cached; each frame is then an array copy plus a
 * bounded number of tile writes, with no dependency on any other frame.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AbsoluteFill,
  continueRender,
  delayRender,
  staticFile,
  useCurrentFrame,
} from "remotion";

import type { PathManifest } from "./drawing/manifest";
import { renderFrame, type RevealAssets } from "./drawing/reveal";

export interface DrawingSceneProps {
  /** Folder name under public/precomputed. */
  name: string;
  /** Paper colour shown in the letterbox area for non-square compositions. */
  background?: string;
  [key: string]: unknown;
}

/**
 * The manifest is fetched here rather than threaded through props: it carries
 * every tile coordinate (3,481 for the sample scene) and Remotion serialises
 * props into the render, so passing it would bloat every invocation for no gain.
 */
export async function fetchManifest(name: string): Promise<PathManifest> {
  const response = await fetch(staticFile(`precomputed/${name}/path.json`));
  return (await response.json()) as PathManifest;
}

async function decodeToPixels(url: string) {
  const response = await fetch(url);
  const bitmap = await createImageBitmap(await response.blob());

  // Read the dimensions BEFORE close(): closing an ImageBitmap resets width and
  // height to 0, which silently produces zero-sized assets rather than an error.
  const width = bitmap.width;
  const height = bitmap.height;

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const context = canvas.getContext("2d", { willReadFrequently: true })!;
  context.drawImage(bitmap, 0, 0);
  const { data } = context.getImageData(0, 0, width, height);
  bitmap.close();

  if (width === 0 || height === 0) {
    throw new Error(`decoded a zero-sized image from ${url}`);
  }
  return { data: new Uint8Array(data), width, height };
}

async function loadAssets(
  name: string,
  manifest: PathManifest,
): Promise<RevealAssets> {
  const base = `precomputed/${name}`;
  const [source, hand, ...rest] = await Promise.all([
    decodeToPixels(staticFile(`${base}/source.png`)),
    decodeToPixels(staticFile(`${base}/hand.png`)),
    ...manifest.segments.flatMap((_, i) => [
      decodeToPixels(staticFile(`${base}/cum-${i}.png`)),
      decodeToPixels(staticFile(`${base}/seg-${i}-ink.png`)),
    ]),
  ]);

  const cumulative: Uint8Array[] = [];
  const ink: Uint8Array[] = [];
  manifest.segments.forEach((_, i) => {
    cumulative.push(rest[i * 2].data);
    // The ink layers are written as grayscale PNGs, so any channel will do.
    const gray = rest[i * 2 + 1];
    const packed = new Uint8Array(gray.width * gray.height);
    for (let p = 0; p < packed.length; p++) packed[p] = gray.data[p * 4];
    ink.push(packed);
  });

  return {
    cumulative,
    ink,
    source: source.data,
    hand: { data: hand.data, width: hand.width, height: hand.height },
  };
}

interface Loaded {
  manifest: PathManifest;
  assets: RevealAssets;
}

export const DrawingScene: React.FC<DrawingSceneProps> = ({
  name,
  background = "white",
}) => {
  const frame = useCurrentFrame();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [loaded, setLoaded] = useState<Loaded | null>(null);
  const [handle] = useState(() => delayRender(`decoding assets for ${name}`));

  useEffect(() => {
    let cancelled = false;
    fetchManifest(name)
      .then(async (manifest) => ({ manifest, assets: await loadAssets(name, manifest) }))
      .then((result) => {
        if (cancelled) return;
        setLoaded(result);
        continueRender(handle);
      })
      .catch((error) => {
        console.error(error);
        continueRender(handle);
      });
    return () => {
      cancelled = true;
    };
  }, [name, handle]);

  const paint = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !loaded) return;
    const context = canvas.getContext("2d");
    if (!context) return;

    const { manifest, assets } = loaded;
    const pixels = renderFrame(manifest, assets, frame);
    context.putImageData(new ImageData(pixels, manifest.width, manifest.height), 0, 0);
  }, [loaded, frame]);

  useEffect(paint, [paint]);

  return (
    <AbsoluteFill
      style={{
        backgroundColor: background,
        justifyContent: "center",
        alignItems: "center",
      }}
    >
      <canvas
        ref={canvasRef}
        width={loaded?.manifest.width ?? 1020}
        height={loaded?.manifest.height ?? 1020}
        // Fit the square drawing into whatever aspect ratio the composition uses.
        style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
      />
    </AbsoluteFill>
  );
};
