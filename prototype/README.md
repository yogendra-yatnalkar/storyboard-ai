# Phase 0 prototype: Python reference → Remotion

Proves the whiteboard drawing algorithm ports to Remotion without losing the hand-drawn look, and
measures whether frames are cheap and independent enough for a scrubbable browser preview.

## Result

| Check | Result |
|---|---|
| Tile visit order, element-wise vs Python | **identical** - 3,481 tiles across 9 segments |
| Sampled frames, reveal function vs Python | **0 differing pixels** (12 frames) |
| Sampled frames, Remotion/Chromium vs Python | **0 differing pixels** (12 frames) |
| Frame cost | 1.38 ms mean, ~29x realtime at 25 fps |
| Random seek vs sequential | 1.08x - frames carry no accumulated state |

The port is pixel-identical, not merely close. Preview and Lambda-style distributed rendering are
both viable. See [`../docs/architecture.md`](../docs/architecture.md) for the design and the
parity constraints that make it work.

## Layout

```
reference/render.py            faithful extraction of the original algorithm,
                               with --dump-path for ground-truth tile order
reference/make_comparison.py   stacks two renders into one side-by-side video
remotion-renderer/
  src/drawing/preprocess.ts    OpenCV WASM: resize, threshold, mask rasterisation
  src/drawing/tilePath.ts      the traversal port (parity-critical)
  src/drawing/reveal.ts        frame → pixels, pure
  src/DrawingScene.tsx         Remotion composition, a thin shell over reveal.ts
  scripts/precompute.ts        image → path.json + PNG layers
  scripts/compare.ts           parity harness; exits non-zero on regression
  scripts/benchmark.ts         frame cost and seek independence
```

## Running it

From the repository root:

```bash
pip install -r requirements.txt
cd prototype/remotion-renderer && npm install && cd ../..
```

Remotion needs a Chromium with the old headless mode - a full `chrome` binary will not launch.
Point it at a headless shell if one is already installed, otherwise Remotion downloads its own:

```bash
export REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell
```

**1. Render the Python reference** (ground truth: tile path, sampled frames, video)

```bash
python prototype/reference/render.py \
  --image generate-whiteboard-animated-videos/images/2.png \
  --mask  generate-whiteboard-animated-videos/images/2.json \
  --dump-path   prototype/out/path-python.json \
  --dump-frames prototype/out/frames-python \
  --video       prototype/out/render-python.mp4
```

**2. Precompute the drawing path and render assets**

```bash
cd prototype/remotion-renderer
npx tsx scripts/precompute.ts \
  --image ../../generate-whiteboard-animated-videos/images/2.png \
  --mask  ../../generate-whiteboard-animated-videos/images/2.json \
  --name scene-2
```

**3. Render through Remotion** (image sequence for lossless comparison, plus an MP4)

```bash
OUT=$(git rev-parse --show-toplevel)/prototype/out
npx remotion render Whiteboard-Square "$OUT/frames-remotion" --sequence --image-format=png
npx remotion render Whiteboard-Square "$OUT/render-remotion.mp4"
```

**4. Check parity and frame cost**

```bash
npx tsx scripts/compare.ts   --name scene-2
npx tsx scripts/benchmark.ts --name scene-2
```

**5. Watch them side by side** - metrics can pass while the animation still feels wrong, so this
is the real gate:

```bash
cd ../..
python prototype/reference/make_comparison.py \
  --left  prototype/out/render-python.mp4 \
  --right prototype/out/render-remotion.mp4 \
  --out   prototype/out/comparison-python-vs-remotion.mp4
```

**Scrub the timeline** to confirm the preview architecture interactively:

```bash
cd prototype/remotion-renderer && npx remotion studio
```

`.github/workflows/render-prototype.yml` runs all of the above on manual dispatch and uploads the
videos and parity report as artifacts.

## Deciding the masking question

The product decision left open in planning was whether to run server-side segmentation at all.
Render both paths on the same image and judge the output:

```bash
python prototype/reference/render.py \
  --image generate-whiteboard-animated-videos/images/2.png \
  --video prototype/out/render-python-nomask.mp4          # no --mask

python prototype/reference/make_comparison.py \
  --left  prototype/out/render-python.mp4 \
  --right prototype/out/render-python-nomask.mp4 \
  --out   prototype/out/comparison-masked-vs-unmasked.mp4 \
  --left-label "with SAM masks" --right-label "no masks"
```

|  | With masks | No masks |
|---|---|---|
| Segments | 9 | 1 |
| Tiles | 3,481 | 2,034 |
| Duration | 472 frames (18.9s) | 329 frames (13.2s) |
| Added cost per scene | ~$0.01-0.05 | none |

Masking is about a third of cheap-tier COGS - see [`../docs/cost-model.md`](../docs/cost-model.md).

## Notes

- `prototype/out/` and `public/precomputed/` are generated and git-ignored; step 2 rebuilds them.
- Remotion's intermediate frame format is set to PNG in `remotion.config.ts`. JPEG costs about
  8/255 mean error per frame on black strokes over white paper - enough to swamp a parity
  comparison.
- A small residual difference between the two **encoded MP4s** is expected: they come from
  different encoders (OpenCV `mp4v` vs Remotion's h264). `scripts/compare.ts` compares lossless
  frames and is the authoritative check.
