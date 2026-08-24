# Architecture

## The core idea: precompute, then reveal

The Python renderer **accumulates state**. `drawn_frame` is mutated inside a loop and frames fall
out as a side effect (`draw-whiteboard-animations.py:184-229`), so frame *N* is only reachable by
computing frames 0..*N*-1.

Remotion needs the opposite: frame *N* must be a pure function of *N*. Splitting the algorithm in
two satisfies that, and the same split is what makes three separate product requirements possible:

```
  precompute (once per image)              reveal (per frame, pure)
  ───────────────────────────              ─────────────────────────
  resize → grayscale → threshold           cum-<i>.png  (canvas entering segment i)
  rasterise polygon masks                    + tiles [0, (f+1) * skipRate)
  per segment: grid → ink tiles →            + hand at the last drawn tile
    nearest-neighbour walk                 ────────────────────────────
  ↓                                        frame pixels
  path.json + PNG layers
```

| Requirement | Why the split is needed |
|---|---|
| Scrubbable preview | The Player seeks to any frame without replaying history |
| Remotion Lambda | Workers render disjoint frame ranges in parallel |
| Cheap re-render | Change the hand or aspect ratio without recomputing the path |

Measured: 1.38 ms per frame, and random seeks cost 1.08x sequential frames - frames are genuinely
independent.

A useful side effect: total duration is known before rendering
(`ceil(tiles / skipRate) + endHold * fps`). Narration length can drive `skipRate` so audio and
video align by construction, replacing the hand-tuned `apad=pad_dur=1.19` corrections in
`ffmpeg-command-notes.txt`.

## Parity constraints

These reference behaviours are load-bearing. Each is easy to "clean up" into something that
produces a different animation.

- **CLAHE is a no-op.** `preprocess_image` computes `cl1` and then thresholds `img_gray` anyway.
  Applying CLAHE changes the output, so the port omits it too.
- **Swap-with-last removal.** A visited tile is removed by overwriting it with the array's last
  element (`draw-whiteboard-animations.py:215-216`), not by splicing. That reshuffles the
  remaining candidates and changes which tile is nearest on later iterations.
- **Row-major seeding.** `np.where` returns row-major order, so the walk starts at the top-left
  ink tile.
- **Loop stops at one candidate.** The final tile of every segment is never drawn.
- **Tiles come from the segment's own masked threshold.** A tile straddling an object edge paints
  white outside that object - which can white out part of an earlier object that is never
  repainted. The reference does this, so parity requires it.
- **Masks are positional, not label-keyed.** Image 2 has five shapes all labelled `4`; a
  label-keyed lookup silently collapses them.
- **OpenCV, not a reimplementation.** The port uses OpenCV compiled to WebAssembly
  (`@techstark/opencv-js`) for resize and adaptive threshold. Hand-rolled equivalents drift by a
  pixel, and one flipped pixel can add or remove a whole tile from the path.

## Production system

```
  Next.js (Vercel)
      │  auth, projects, storyboard editor, Remotion Player preview
      ▼
  Supabase ── Postgres + row-level security, auth, job records
      │
      ▼
  Job orchestrator
      ├── LLM adapter        DeepSeek → scenes, image prompts, narration
      ├── Image adapter      hosted SDXL (premium model on the top tier)
      ├── Segmentation       SAM 2, if the masked path wins the comparison
      ├── TTS adapter        neural default, ElevenLabs on the top tier
      └── Precompute         path.json + PNG layers → R2
      ▼
  Remotion Lambda ── renders frame ranges in parallel
      ▼
  Cloudflare R2 ── rendered MP4, zero egress
```

Every external model sits behind a narrow adapter interface, so the cheap-to-premium upsell is a
config change rather than a code path, and a provider swap (DeepSeek's China-hosted API being the
most likely candidate) does not touch the pipeline.

### Why not GitHub Actions for production rendering

The prototype workflow renders on Actions because it needs no infrastructure. It does not
generalise:

1. GitHub's acceptable-use policy restricts Actions to compute related to the repository's own
   development; a commercial render farm is the disallowed case.
2. Concurrency is capped per account (20 jobs on Free/Pro), so a modest spike queues user renders
   with no autoscaling.
3. Cold start is 30-60s before Remotion begins - most of the latency budget for a 60-second short.

Remotion Lambda scales to zero, has predictable per-render cost, and splits a video across workers
by frame range, which the precompute/reveal split already supports.

## Repository layout

```
prototype/
  reference/render.py            faithful extraction of the original algorithm
  reference/make_comparison.py   side-by-side comparison video
  remotion-renderer/
    src/drawing/preprocess.ts    OpenCV WASM: resize, threshold, mask rasterisation
    src/drawing/tilePath.ts      the traversal port (the parity-critical half)
    src/drawing/reveal.ts        frame → pixels, pure
    src/DrawingScene.tsx         Remotion composition (a thin shell over reveal.ts)
    scripts/precompute.ts        image → path.json + PNG layers
    scripts/compare.ts           parity harness, exits non-zero on regression
    scripts/benchmark.ts         frame cost and seek independence
```

The composition renders through `reveal.ts` rather than reimplementing the reveal with canvas
`drawImage` calls. That is slightly slower than it could be, and deliberate: what the parity
harness measures is exactly what Remotion renders, with no second implementation to drift.
