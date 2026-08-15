/**
 * Image preprocessing for the whiteboard renderer, backed by OpenCV compiled to
 * WebAssembly.
 *
 * The reference implementation is OpenCV/Python, so the port uses the same OpenCV
 * primitives rather than reimplementing adaptive thresholding or bilinear resize
 * in JavaScript. Hand-rolled equivalents drift by a pixel here and there, and a
 * single flipped pixel can add or remove an entire tile from the drawing path.
 *
 * Two ordering details are carried over from the reference and matter for parity:
 *  - the colour image is resized first, then converted to grayscale;
 *  - polygon masks are rasterised at the SOURCE resolution and then resized, so
 *    the bilinear edges soften and only pixels that stay exactly 255 count as
 *    part of the object.
 */

import cvModule from "@techstark/opencv-js";
import { PNG } from "pngjs";
import { readFileSync, writeFileSync } from "node:fs";

/**
 * The package's default export is a thenable that resolves to the real module
 * once the WASM runtime has initialised - importing it does not give you an
 * OpenCV namespace, and touching `cv.Mat` before awaiting silently yields
 * undefined. Everything below goes through `requireCv()` so that mistake fails
 * loudly instead of producing an empty render.
 */
type OpenCv = typeof cvModule;

let runtime: OpenCv | undefined;

export interface RgbaImage {
  data: Uint8ClampedArray;
  width: number;
  height: number;
}

/** Must be awaited once before any other export in this module is used. */
export async function openCvReady(): Promise<OpenCv> {
  if (!runtime) {
    runtime = await (cvModule as unknown as PromiseLike<OpenCv>);
  }
  return runtime;
}

function requireCv(): OpenCv {
  if (!runtime) {
    throw new Error("openCvReady() must be awaited before using OpenCV helpers");
  }
  return runtime;
}

export function readPng(path: string): RgbaImage {
  const png = PNG.sync.read(readFileSync(path));
  return {
    data: new Uint8ClampedArray(png.data),
    width: png.width,
    height: png.height,
  };
}

export function writeGrayPng(
  path: string,
  gray: Uint8Array,
  width: number,
  height: number,
): void {
  const png = new PNG({ width, height });
  for (let i = 0; i < width * height; i++) {
    png.data[i * 4] = gray[i];
    png.data[i * 4 + 1] = gray[i];
    png.data[i * 4 + 2] = gray[i];
    png.data[i * 4 + 3] = 255;
  }
  writeFileSync(path, PNG.sync.write(png));
}

export function writeRgbaPng(
  path: string,
  rgba: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): void {
  const png = new PNG({ width, height });
  png.data.set(Buffer.from(rgba.buffer, rgba.byteOffset, rgba.length));
  writeFileSync(path, PNG.sync.write(png));
}

export interface PreprocessResult {
  /** Resized source image as RGBA, used for the colour reveal. */
  source: Uint8Array;
  /** Adaptive-threshold bitmap: 0 = ink, 255 = paper. */
  threshold: Uint8Array;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
}

/**
 * Resize, grayscale and adaptively threshold the source image.
 *
 * NOTE: the reference computes a CLAHE-equalised image and then thresholds the
 * plain grayscale anyway, so CLAHE is a no-op there. It is omitted here rather
 * than replicated, since applying it would change the output.
 */
export function preprocessImage(
  image: RgbaImage,
  width: number,
  height: number,
): PreprocessResult {
  const cv = requireCv();
  const rgba = cv.matFromImageData({
    data: image.data,
    width: image.width,
    height: image.height,
  } as ImageData);

  const rgb = new cv.Mat();
  cv.cvtColor(rgba, rgb, cv.COLOR_RGBA2RGB);

  const resized = new cv.Mat();
  cv.resize(rgb, resized, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);

  const gray = new cv.Mat();
  cv.cvtColor(resized, gray, cv.COLOR_RGB2GRAY);

  const threshold = new cv.Mat();
  cv.adaptiveThreshold(
    gray,
    threshold,
    255,
    cv.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv.THRESH_BINARY,
    15,
    10,
  );

  const resizedRgba = new cv.Mat();
  cv.cvtColor(resized, resizedRgba, cv.COLOR_RGB2RGBA);

  const result: PreprocessResult = {
    source: Uint8Array.from(resizedRgba.data),
    threshold: Uint8Array.from(threshold.data),
    width,
    height,
    sourceWidth: image.width,
    sourceHeight: image.height,
  };

  for (const mat of [rgba, rgb, resized, gray, threshold, resizedRgba]) mat.delete();
  return result;
}

export interface LabelmeShape {
  label?: string;
  points: [number, number][];
}

export interface LabelmeFile {
  shapes: LabelmeShape[];
  imageWidth?: number;
  imageHeight?: number;
}

export interface ObjectMask {
  label: string;
  /** 255 inside the object, 0 outside, at render resolution. */
  mask: Uint8Array;
}

/**
 * Rasterise labelme/AnyLabeling polygons into per-object masks, plus the leftover
 * background mask that the final drawing pass uses.
 */
export function buildObjectMasks(
  file: LabelmeFile,
  sourceWidth: number,
  sourceHeight: number,
  width: number,
  height: number,
): { objects: ObjectMask[]; background: Uint8Array } {
  const cv = requireCv();
  const background = new Uint8Array(width * height).fill(255);
  const objects: ObjectMask[] = [];

  file.shapes.forEach((shape, index) => {
    const mat = cv.Mat.zeros(sourceHeight, sourceWidth, cv.CV_8UC1);

    // fillPoly takes a MatVector of Nx1 CV_32SC2 point matrices.
    const points = cv.matFromArray(
      shape.points.length,
      1,
      cv.CV_32SC2,
      shape.points.flatMap(([x, y]) => [Math.trunc(x), Math.trunc(y)]),
    );
    const contours = new cv.MatVector();
    contours.push_back(points);
    cv.fillPoly(mat, contours, new cv.Scalar(255));

    const resized = new cv.Mat();
    cv.resize(mat, resized, new cv.Size(width, height), 0, 0, cv.INTER_LINEAR);

    const mask = Uint8Array.from(resized.data);
    for (let i = 0; i < mask.length; i++) {
      if (mask[i] === 255) background[i] = 0;
    }
    objects.push({ label: String(shape.label ?? index), mask });

    mat.delete();
    points.delete();
    contours.delete();
    resized.delete();
  });

  return { objects, background };
}
