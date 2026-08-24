"""Reference implementation of the whiteboard drawing algorithm.

This is a faithful, path-free extraction of
``generate-whiteboard-animated-videos/draw-whiteboard-animations.py``. It exists to
produce the ground truth that the Remotion/TypeScript port is diffed against, so it
deliberately preserves the original's behaviour - including its quirks:

* Thresholding runs on the raw grayscale image; the CLAHE result is computed and
  discarded by the original, so it is not applied here either.
* A visited tile is removed from the candidate array by overwriting it with the
  array's last element rather than splicing it out. That reorders the remaining
  candidates and therefore changes the traversal, so it must be replicated exactly.
* The traversal loop runs while more than one candidate remains, which means the
  final tile of every segment is never drawn.

The work is split into two halves so the port can be verified in isolation:

    compute_segments()  -> the drawing path (pure, deterministic, serialisable)
    render_segments()   -> pixels (compositing only)

Usage:
    python render.py --image path/to.png [--mask path/to.json] \
        [--dump-path out/path.json] [--video out/render.mp4]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, field

import cv2
import numpy as np


# --- configuration -----------------------------------------------------------


@dataclass
class RenderConfig:
    """Defaults mirror the ``AllVariables`` block in the original script."""

    frame_rate: int = 25
    resize_wd: int = 1020
    resize_ht: int = 1020
    split_len: int = 10
    object_skip_rate: int = 8
    bg_object_skip_rate: int = 14
    end_gray_img_duration_in_sec: int = 3
    black_pixel_threshold: int = 10

    # Applied only when no mask file is supplied, matching the original's
    # else-branch which overrides both values before drawing.
    no_mask_split_len: int = 15
    no_mask_skip_rate: int = 8

    # The background pass uses a coarser grid so it covers ground faster.
    background_split_len: int = 20


@dataclass
class Segment:
    """One drawing pass: a single masked object, or the background, or the whole image."""

    label: str
    kind: str  # "object" | "background" | "full"
    split_len: int
    skip_rate: int
    tiles: list[list[int]] = field(default_factory=list)  # [row, col] in visit order

    @property
    def frame_count(self) -> int:
        return len(self.tiles) // self.skip_rate


# --- preprocessing -----------------------------------------------------------


def preprocess_image(img_path: str, config: RenderConfig):
    """Resize, grayscale and adaptively threshold the source image."""
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"could not read image: {img_path}")

    src_ht, src_wd = img.shape[0], img.shape[1]
    img = cv2.resize(img, (config.resize_wd, config.resize_ht))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # NOTE: the original computes CLAHE here and then thresholds img_gray anyway.
    # Kept as a no-op so the threshold bitmap matches byte for byte.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    _unused_clahe = clahe.apply(img_gray)

    img_thresh = cv2.adaptiveThreshold(
        img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 10
    )
    return img, img_gray, img_thresh, (src_ht, src_wd)


def preprocess_hand(hand_path: str, hand_mask_path: str):
    """Crop the hand to its mask bounding box and knock out its background."""
    hand = cv2.imread(hand_path)
    hand_mask = cv2.imread(hand_mask_path, cv2.IMREAD_GRAYSCALE)
    if hand is None or hand_mask is None:
        raise FileNotFoundError("could not read hand image or hand mask")

    indices = np.where(hand_mask == 255)
    top_left = (int(np.min(indices[1])), int(np.min(indices[0])))
    bottom_right = (int(np.max(indices[1])), int(np.max(indices[0])))

    hand = hand[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask = hand_mask[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]]
    hand_mask_inv = (255 - hand_mask) / 255
    hand_mask = hand_mask / 255

    hand[np.where(hand_mask == 0)] = [0, 0, 0]
    return hand, hand_mask_inv


def load_object_masks(mask_path: str, src_shape, config: RenderConfig):
    """Rasterise the AnyLabeling/labelme polygons into per-object masks.

    Returns the per-object masks in file order plus the leftover background mask.
    """
    with open(mask_path) as handle:
        data = json.load(handle)

    src_ht, src_wd = src_shape
    background_mask = np.zeros((config.resize_ht, config.resize_wd), dtype=np.uint8) + 255
    object_masks = []

    for index, shape in enumerate(data["shapes"]):
        mask = np.zeros((src_ht, src_wd), dtype=np.uint8)
        points = np.expand_dims(np.array(shape["points"], dtype=np.int32), axis=0)
        cv2.fillPoly(mask, points, 255)
        mask = cv2.resize(mask, (config.resize_wd, config.resize_ht))

        background_mask[np.where(mask == 255)] = 0
        object_masks.append((str(shape.get("label", index)), mask))

    return object_masks, background_mask


# --- path computation --------------------------------------------------------


def _euclidean(candidates: np.ndarray, point: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((candidates - point) ** 2, axis=1))


def compute_tile_path(
    img_thresh: np.ndarray,
    object_mask: np.ndarray | None,
    split_len: int,
    config: RenderConfig,
) -> list[list[int]]:
    """Return the tile visit order for one drawing pass.

    Tiles are ``split_len`` squares containing at least one ink pixel. The walk is
    greedy nearest-neighbour so the hand travels smoothly instead of jumping.
    """
    thresh = img_thresh.copy()
    if object_mask is not None:
        # Everything outside the object is blanked so only the object has ink.
        thresh[np.where(object_mask == 0)] = 255

    n_cuts_vertical = int(math.ceil(config.resize_ht / split_len))
    n_cuts_horizontal = int(math.ceil(config.resize_wd / split_len))

    # Equivalent to the original's nested np.split, but without the copies.
    grid = thresh.reshape(n_cuts_vertical, split_len, n_cuts_horizontal, split_len)
    ink_per_tile = (grid < config.black_pixel_threshold).sum(axis=(1, 3))

    # np.where yields row-major order, which seeds the walk at the top-left tile.
    candidates = np.array(np.where(ink_per_tile > 0)).T
    if len(candidates) == 0:
        return []

    visit_order: list[list[int]] = []
    selected_ind = 0

    while len(candidates) > 1:
        selected = candidates[selected_ind].copy()
        visit_order.append([int(selected[0]), int(selected[1])])

        # Swap-with-last removal. This is NOT equivalent to np.delete: it moves the
        # final candidate into the vacated slot and so changes the traversal order.
        candidates[selected_ind] = candidates[-1]
        candidates = candidates[:-1]

        selected_ind = int(np.argmin(_euclidean(candidates, selected)))

    return visit_order


def compute_segments(
    img_thresh: np.ndarray,
    mask_path: str | None,
    src_shape,
    config: RenderConfig,
) -> list[tuple[Segment, np.ndarray | None]]:
    """Build every drawing pass for an image, in the order they are drawn.

    Each segment is paired with its own mask. Masks are carried positionally
    rather than by label because labels repeat - image 2 has five shapes all
    labelled "4" - so a label-keyed lookup would silently collapse them.
    """
    if mask_path is None:
        segment = Segment(
            label="full",
            kind="full",
            split_len=config.no_mask_split_len,
            skip_rate=config.no_mask_skip_rate,
            tiles=compute_tile_path(img_thresh, None, config.no_mask_split_len, config),
        )
        return [(segment, None)]

    object_masks, background_mask = load_object_masks(mask_path, src_shape, config)

    paired: list[tuple[Segment, np.ndarray | None]] = [
        (
            Segment(
                label=label,
                kind="object",
                split_len=config.split_len,
                skip_rate=config.object_skip_rate,
                tiles=compute_tile_path(img_thresh, mask, config.split_len, config),
            ),
            mask,
        )
        for label, mask in object_masks
    ]

    paired.append(
        (
            Segment(
                label="background",
                kind="background",
                split_len=config.background_split_len,
                skip_rate=config.bg_object_skip_rate,
                tiles=compute_tile_path(
                    img_thresh, background_mask, config.background_split_len, config
                ),
            ),
            background_mask,
        )
    )

    return paired


# --- compositing -------------------------------------------------------------


def composite_hand(canvas, hand, hand_mask_inv, x, y):
    """Alpha-composite the hand onto the canvas with its top-left corner at (x, y)."""
    hand_ht, hand_wd = hand.shape[0], hand.shape[1]
    crop_ht = min(hand_ht, canvas.shape[0] - y)
    crop_wd = min(hand_wd, canvas.shape[1] - x)

    hand_crop = hand[:crop_ht, :crop_wd]
    mask_crop = hand_mask_inv[:crop_ht, :crop_wd]

    region = canvas[y : y + crop_ht, x : x + crop_wd]
    for channel in range(3):
        region[:, :, channel] = region[:, :, channel] * mask_crop
    canvas[y : y + crop_ht, x : x + crop_wd] = region + hand_crop
    return canvas


def render_segments(
    paired,
    img,
    img_thresh,
    hand,
    hand_mask_inv,
    config: RenderConfig,
    video_path: str | None,
    frame_callback=None,
):
    """Replay the computed paths into video frames.

    Frames are emitted every ``skip_rate`` tiles, matching the original's counter.
    Tiles are copied from the segment's OWN masked threshold, so a tile straddling
    an object boundary paints white outside that object - including over a region
    an earlier pass had already revealed in colour.
    """
    writer = None
    if video_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        fourcc = getattr(cv2, "VideoWriter_fourcc", None)
        fourcc = fourcc(*"mp4v") if fourcc else cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            video_path, fourcc, config.frame_rate, (config.resize_wd, config.resize_ht)
        )

    canvas = np.zeros(img.shape, np.uint8) + np.array([255, 255, 255], np.uint8)
    frames_written = 0

    def emit(frame):
        nonlocal frames_written
        if writer is not None:
            writer.write(frame)
        if frame_callback is not None:
            frame_callback(frames_written, frame)
        frames_written += 1

    for segment, mask in paired:
        seg_thresh = img_thresh.copy()
        if mask is not None:
            seg_thresh[np.where(mask == 0)] = 255

        for counter, (row, col) in enumerate(segment.tiles, start=1):
            v_start = row * segment.split_len
            h_start = col * segment.split_len
            v_end = v_start + segment.split_len
            h_end = h_start + segment.split_len

            canvas[v_start:v_end, h_start:h_end] = seg_thresh[
                v_start:v_end, h_start:h_end
            ][:, :, None]

            if counter % segment.skip_rate == 0:
                emit(
                    composite_hand(
                        canvas.copy(),
                        hand.copy(),
                        hand_mask_inv.copy(),
                        h_start + segment.split_len // 2,
                        v_start + segment.split_len // 2,
                    )
                )

        # A finished pass is revealed in full colour before the next one starts.
        if mask is None:
            canvas[:, :, :] = img
        else:
            region = np.where(mask == 255)
            canvas[region] = img[region]

    for _ in range(config.frame_rate * config.end_gray_img_duration_in_sec):
        emit(img.copy())

    if writer is not None:
        writer.release()

    return frames_written


# --- entry point -------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--mask", default=None, help="labelme/AnyLabeling JSON; omit to draw the whole image")
    parser.add_argument("--hand", default="generate-whiteboard-animated-videos/images/drawing-hand.png")
    parser.add_argument("--hand-mask", default="generate-whiteboard-animated-videos/images/hand-mask.png")
    parser.add_argument("--dump-path", default=None, help="write the tile path as JSON")
    parser.add_argument("--video", default=None, help="write an mp4 render")
    parser.add_argument("--dump-frames", default=None, help="directory for sampled PNG frames")
    parser.add_argument("--frame-samples", type=int, default=12, help="how many frames to sample")
    parser.add_argument("--width", type=int, default=1020)
    parser.add_argument("--height", type=int, default=1020)
    args = parser.parse_args()

    config = RenderConfig(resize_wd=args.width, resize_ht=args.height)
    img, _, img_thresh, src_shape = preprocess_image(args.image, config)
    paired = compute_segments(img_thresh, args.mask, src_shape, config)
    segments = [segment for segment, _ in paired]

    total_tiles = sum(len(s.tiles) for s in segments)
    total_frames = sum(s.frame_count for s in segments)
    print(f"segments: {len(segments)}  tiles: {total_tiles}  drawing frames: {total_frames}")
    for segment in segments:
        print(f"  {segment.label:<12} {segment.kind:<10} tiles={len(segment.tiles):<6} frames={segment.frame_count}")

    if args.dump_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.dump_path)), exist_ok=True)
        payload = {
            "image": args.image,
            "mask": args.mask,
            "config": config.__dict__,
            "segments": [
                {
                    "label": s.label,
                    "kind": s.kind,
                    "splitLen": s.split_len,
                    "skipRate": s.skip_rate,
                    "tiles": s.tiles,
                }
                for s in segments
            ],
        }
        with open(args.dump_path, "w") as handle:
            json.dump(payload, handle)
        print(f"wrote path -> {args.dump_path}")

    if args.video or args.dump_frames:
        hand, hand_mask_inv = preprocess_hand(args.hand, args.hand_mask)

        callback = None
        sampled = []
        if args.dump_frames:
            os.makedirs(args.dump_frames, exist_ok=True)
            duration = total_frames + config.frame_rate * config.end_gray_img_duration_in_sec
            # Evenly spaced across the whole video so every segment is represented.
            sampled = sorted({
                round(i * (duration - 1) / max(args.frame_samples - 1, 1))
                for i in range(args.frame_samples)
            })
            wanted = set(sampled)

            def callback(index, frame):
                if index in wanted:
                    cv2.imwrite(os.path.join(args.dump_frames, f"frame-{index:05d}.png"), frame)

        written = render_segments(
            paired, img, img_thresh, hand, hand_mask_inv, config, args.video, callback,
        )
        if args.video:
            print(f"wrote video -> {args.video} ({written} frames)")
        if args.dump_frames:
            with open(os.path.join(args.dump_frames, "frames.json"), "w") as handle:
                json.dump({"total": written, "sampled": sampled}, handle)
            print(f"wrote {len(sampled)} sampled frames -> {args.dump_frames}")


if __name__ == "__main__":
    main()
