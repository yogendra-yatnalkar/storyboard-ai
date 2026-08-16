# Status: what is built, what is stubbed, what is not started

An honest accounting per phase. "Verified" means something automated proves it;
"unverified" means the code exists and typechecks but has never run against the
real thing.

## Summary

| Phase | State | Verified by |
|---|---|---|
| 0 — Remotion port | **Complete** | Parity harness: 0 differing pixels, three ways |
| 1 — Renderer productization | **Complete** | 27 unit tests, parity intact, full video renders |
| 2 — Pipeline services | **Runs offline; live providers unverified** | 41 unit tests, fixture end-to-end run |
| 3 — Application | **UI builds and renders; not connected to a database** | `next build`, screenshots |
| 4 — Billing | **Logic complete and tested; no provider account** | 20 billing tests |
| 5 — v1 extras | **Partially built** | Watermark and brand render; music and i18n are wiring only |

Nothing here has been deployed, and no external API has been called with a real
key.

---

## Phase 0 — Remotion port · complete

The drawing algorithm was ported from Python/OpenCV to Remotion/canvas.

- Tile visit order identical to the original across 3,481 tiles and 9 segments.
- 0 differing pixels on 12 sampled frames — including frames Remotion rendered
  through Chromium, not just the shared function in Node.
- 1.38 ms/frame; random seeks cost 1.08x a sequential frame, confirming frames
  carry no accumulated state.

Also fixed three bugs in the original scripts: `VideoWriter` given width for
height, `getTextSize` called on `None` between subtitle segments, and TTS files
written 0-indexed while every downstream command expected 1-indexed.

## Phase 1 — Renderer productization · complete

`ffmpeg-command-notes.txt` is fully replaced. No `ffprobe`, no `apad`, no concat
list.

- **Narration-driven timing.** Drawing speed is solved against audio length;
  every scene lands within one frame of its narration.
- **Multi-scene video** with per-scene audio, rendered as one composition.
- **Captions** behind a source interface. Whisper JSON is the first
  implementation, split per scene by aligning against the narration text. All
  248 transcribed words allocate cleanly across the four scenes.
- **Two caption styles** — word highlighting for vertical, phrase blocks for
  landscape. Both wrap.
- **Aspect ratios** per project, validated against the tile grid; legacy square
  art is padded rather than stretched.
- **Watermark and brand mark.**

Rendered output: `prototype/out/story-square.mp4` and `story-portrait.mp4`
(97.64s, AAC audio). Screenshots in `docs/screenshots/01`–`05`.

**Not done:** Whisper is not run automatically — the committed transcript is used
as a fixture. `torch` plus a model is ~2 GB and Phase 2's TTS will supply word
timings directly for most providers.

## Phase 2 — Pipeline services · runs offline, live path unverified

`packages/pipeline` takes story text and produces render-ready scenes.

**Working and tested:**
- Provider interfaces for LLM, image, TTS and segmentation.
- Fixture providers that run the entire pipeline with no keys, reusing the
  repo's own assets — this is how CI exercises the path.
- Orchestration with per-scene concurrency.
- Cost accounting. A fixture run reports **$0.1899** per four-scene video,
  independently confirming the ~$0.20 estimate in the cost model.
- DeepSeek adapter: prompt, JSON parsing, pricing, and retry on 429/5xx, all
  tested against a stubbed `fetch`.

**Unverified — code exists, never called for real:**
- The HTTP request and response shapes for image generation, TTS and
  segmentation are written against typical vendor APIs but not against a
  specific one. Expect to adjust them when a vendor is chosen.
- `HostedTtsProvider` points at a placeholder base URL.
- No live DeepSeek call has been made.

## Phase 3 — Application · UI only

`apps/web` is a Next.js 15 app: projects list, new-project form, project detail
with storyboard and cost breakdown, and pricing. It builds clean and typechecks.
Screenshots in `docs/screenshots/10`–`13`.

`supabase/migrations/0001_initial_schema.sql` defines accounts, projects, render
jobs and webhook idempotency, with RLS on every table and an atomic
`claim_render_slot` function (a read-then-write quota check races — two
concurrent renders both read 29 against a quota of 30 and both proceed).

**Not done:**
- The app reads `lib/demoData.ts`, not Supabase. There is no auth, no data
  fetching, no mutation.
- The migration has never been applied to a live project.
- No R2 client, no signed URLs, no upload path.
- No job queue. Nothing connects the app to the pipeline or the renderer.

## Phase 4 — Billing · logic complete, unconnected

`packages/pipeline/src/billing` implements plans, quota gating and webhook
handling, with 20 tests.

Design decisions worth noting, both from the cost model:
- **Studio is 50 videos, not 100.** At $49/100 premium videos a fully-using
  subscriber costs ~$54 against ~$46.55 net of fees.
- **Free is 3 videos lifetime, not monthly**, at 720p, unmasked, watermarked,
  behind email verification. A renewing monthly allowance is ~$0.45 per free
  user per month forever.

Webhook handling verifies signatures in constant time, is idempotent, and
ignores deliveries older than current state.

**Not done:** no Paddle or Lemon Squeezy account, no checkout, no webhook
endpoint mounted, no plan changes ever applied to a real account.

## Phase 5 — v1 extras · partial

- **Brand kit** — logo, accent colour, caption font and paper colour flow
  through the manifest and render. Working.
- **Watermark** — working, and gated by plan in the billing logic.
- **Music** — the manifest carries a track and gain, and the composition mixes
  it at a fixed level. Sidechain-style ducking under narration is not
  implemented, and no licensed track library has been sourced.
- **Multi-language** — the language flows through the request, manifest and
  fixture, but nothing translates the script, and no non-English voice has been
  tested.
- **Publishing to YouTube/TikTok** — not started. Each needs OAuth and platform
  app review.

---

## The open product question

The masking decision was left to the prototype and the comparison now exists:
`prototype/out/comparison-masked-vs-unmasked.mp4`.

| | With SAM masks | Whole-image draw |
|---|---|---|
| Segments | 9 | 1 |
| Tiles | 3,481 | 2,034 |
| Duration | 472 frames | 329 frames |
| Cost per scene | ~$0.01–0.05 | none |

Masking is ~25% of cheap-tier COGS. The current default keeps it on for paid
plans and off for free, which is implemented in `billing/plans.ts` — but the
final call is a judgement about how the output looks, and that is still yours.
