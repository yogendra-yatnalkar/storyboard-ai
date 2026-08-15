"""Stack two renders side by side into a single comparison video.

Used to put the Python reference and the Remotion port next to each other, since
parity metrics can pass while the animation still feels wrong - the eyeball test
is the real gate.

Usage:
    python make_comparison.py --left a.mp4 --right b.mp4 --out compare.mp4 \
        [--left-label "python"] [--right-label "remotion"]
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def label_frame(frame, text, height):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 46), (245, 245, 245), -1)
    cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--left-label", default="python reference")
    parser.add_argument("--right-label", default="remotion port")
    parser.add_argument("--scale", type=float, default=0.5)
    args = parser.parse_args()

    left = cv2.VideoCapture(args.left)
    right = cv2.VideoCapture(args.right)
    if not left.isOpened() or not right.isOpened():
        raise SystemExit("could not open one of the input videos")

    fps = left.get(cv2.CAP_PROP_FPS) or 25
    width = int(left.get(cv2.CAP_PROP_FRAME_WIDTH) * args.scale)
    height = int(left.get(cv2.CAP_PROP_FRAME_HEIGHT) * args.scale)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    fourcc = getattr(cv2, "VideoWriter_fourcc", None)
    fourcc = fourcc(*"mp4v") if fourcc else cv2.VideoWriter.fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, fps, (width * 2 + 8, height))

    written = 0
    max_delta = 0.0
    while True:
        ok_left, frame_left = left.read()
        ok_right, frame_right = right.read()
        if not ok_left or not ok_right:
            break

        frame_left = cv2.resize(frame_left, (width, height))
        frame_right = cv2.resize(frame_right, (width, height))

        # Track the worst per-frame difference so the summary is not purely visual.
        delta = float(np.mean(cv2.absdiff(frame_left, frame_right)))
        max_delta = max(max_delta, delta)

        canvas = np.zeros((height, width * 2 + 8, 3), np.uint8) + 255
        canvas[:, :width] = label_frame(frame_left, args.left_label, height)
        canvas[:, width + 8 :] = label_frame(frame_right, args.right_label, height)
        writer.write(canvas)
        written += 1

    left.release()
    right.release()
    writer.release()

    print(f"wrote {written} frames -> {args.out}")
    # Both inputs are lossy-encoded, so a small non-zero delta here is codec noise,
    # not a parity failure. The authoritative check is scripts/compare.ts.
    print(f"worst per-frame mean abs difference (post-encode): {max_delta:.3f}/255")


if __name__ == "__main__":
    main()
