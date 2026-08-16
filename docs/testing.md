# Testing

Three layers, each catching a different class of failure. Every one of them runs
without API keys.

| Layer | What it protects | Command | Runtime |
|---|---|---|---|
| Unit tests | Logic: timing, caption alignment, tile walk, pricing, webhooks | `npm test` | ~3s |
| Parity harness | The rendered animation still matches the original renderer | `npx tsx scripts/compare.ts` | ~20s |
| End-to-end | Generation and rendering actually produce a video | see below | ~15min |

## Layer 1 — unit tests

68 tests across two packages, using Node's built-in test runner (no framework).

```bash
cd prototype/remotion-renderer && npm test    # 27 tests
cd packages/pipeline          && npm test     # 41 tests
```

**What they cover, and why those things:**

- `src/drawing/timing.test.ts` — narration-driven drawing speed. Asserts the
  no-narration case still produces the reference's exact 472 frames, that a
  timed scene never ends before its audio, and that the background stays faster
  than objects. *This suite caught a real defect*: rounding the skip rate to an
  integer overshot long narrations by up to six seconds, which is why rates are
  now fractional.
- `src/drawing/tilePath.test.ts` — the parity-critical traversal. Pins the three
  behaviours that are easy to "clean up" into a different animation: row-major
  seeding, swap-with-last removal, and the loop stopping one tile early.
- `src/captions/captions.test.ts` — transcript-to-scene alignment, including
  resync when the transcript inserts a word, punctuation-insensitive matching,
  and line wrapping (the old `cv2.putText` code never wrapped).
- `packages/pipeline/src/cost.test.ts` — the margin maths, including an explicit
  test that Studio at $49/100 premium videos is loss-making and that the revised
  50-video quota is not.
- `packages/pipeline/src/pipeline.test.ts` — the generation pipeline against
  fixture providers, plus DeepSeek retry/parse behaviour driven by a stubbed
  `fetch`.
- `packages/pipeline/src/billing/billing.test.ts` — quota gating, plan
  constraint clamping, constant-time webhook signature checks, and webhook
  idempotency including out-of-order delivery.

## Layer 2 — parity harness

The renderer was ported from Python to Remotion. The harness is what proves the
port did not change the animation, and it is the regression net for every later
change.

```bash
cd prototype/remotion-renderer
npx tsx scripts/compare.ts --name scene-2
```

Three independent checks, because they fail for different reasons:

1. **PATH** — element-wise equality of the tile visit order against the Python
   reference. A mismatch means the drawing is simply different.
2. **PIXEL** — SSIM and mean absolute error on sampled frames, comparing the
   shared reveal function against frames the Python renderer wrote.
3. **BROWSER** — the same metrics against frames Remotion actually rendered in
   Chromium. Checks 1 and 2 exercise Node; only this one covers the React and
   canvas layer.

Current result is **0 differing pixels** on all three, not merely above
threshold. That third check earned its place immediately — it caught
`ImageBitmap.width` being read after `close()` (which zeroes it), so every ink
layer was empty and the hand never drew, while the Node-side checks looked
perfect.

Regenerate the ground truth if the reference implementation changes:

```bash
python prototype/reference/render.py \
  --image generate-whiteboard-animated-videos/images/2.png \
  --mask  generate-whiteboard-animated-videos/images/2.json \
  --dump-path prototype/out/path-python.json \
  --dump-frames prototype/out/frames-python \
  --video prototype/out/render-python.mp4

export REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell
npx remotion render Whiteboard-Square "$PWD/../out/frames-remotion" \
  --sequence --image-format=png
```

**Non-square parity.** The same harness validates non-square rendering, using
the Python reference as the oracle at 1920x1080:

```bash
python prototype/reference/render.py --image prototype/fixtures/synthetic-nonsquare.png \
  --width 1920 --height 1080 --dump-path prototype/out/path-python-nonsquare.json
npx tsx scripts/precompute.ts --image ../fixtures/synthetic-nonsquare.png \
  --name synthetic-nonsquare --width 1920 --height 1080
npx tsx scripts/compare.ts --name synthetic-nonsquare \
  --python-path ../out/path-python-nonsquare.json
```

## Layer 3 — end-to-end

Runs the whole product path with no credentials, using fixture providers and the
repository's own assets.

```bash
# 1. generate: story text -> scenes, images, narration, masks, cost report
cd packages/pipeline
npx tsx src/generate.ts --input "The boy who cried wolf" \
  --out ../../prototype/out/generated --aspect portrait

# 2. build: solve drawing speed against narration, source captions
cd ../../prototype/remotion-renderer
npx tsx scripts/buildVideo.ts \
  --fixture ../out/generated/fixture.json --name generated --aspect portrait

# 3. render
export REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell
npx remotion render Story "$PWD/../out/generated.mp4"
```

Step 1 reports measured per-video cost. The last fixture run produced **$0.1899**
for a four-scene video, which independently confirms the ~$0.20 estimate in
[cost-model.md](cost-model.md).

**Assertions worth making by eye on the output:**

- Every scene's duration is within a frame of its narration (`buildVideo` prints
  both).
- Captions highlight the word being spoken.
- Long caption lines wrap instead of running off frame.
- The video has an audio track: `node -e "…music-metadata…"` or any player.

## Performance

```bash
cd prototype/remotion-renderer && npx tsx scripts/benchmark.ts --name scene-2
```

Asserts the property the scrubbable preview depends on: a random seek must cost
about the same as a sequential frame. If that ratio climbs, state has leaked
between frames and both the Player and Lambda's frame-range splitting break.

Current: **1.38 ms/frame** (~29x realtime at 25 fps), random seeks **1.08x**
sequential.

## CI

`.github/workflows/render-prototype.yml` runs the parity harness and benchmark on
manual dispatch and uploads the comparison video and metrics as artifacts.
`compare.ts` exits non-zero on any parity regression, so a drift fails the job.

Actions is prototype-only infrastructure — see
[architecture.md](architecture.md#why-not-github-actions-for-production-rendering).

## What is not covered

Stated plainly, because these are the gaps a reader would otherwise assume are
handled:

- **No live-provider tests.** DeepSeek, image, TTS and segmentation adapters are
  exercised against stubbed `fetch` only. Their real request and response shapes
  are unverified until keys exist.
- **No browser tests for the web app.** It builds and typechecks, and the
  screenshots in `docs/screenshots/` were captured from the running app, but
  there are no interaction tests.
- **No Supabase integration tests.** The schema and RLS policies in
  `supabase/migrations/` have not been applied to a live project.
- **No load or concurrency testing** of the render queue.
- **Caption timing accuracy** is checked structurally (word counts, ordering,
  bounds) but its alignment to the audio is judged by ear, not asserted.
