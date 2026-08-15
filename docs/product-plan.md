# storyboard-ai as a SaaS product

## What exists today

A personal research pipeline that produced one video, *The Shepherd Boy and the Wolf*. Seven
stages, four of them performed by hand: ChatGPT for the scene summary, Stable Diffusion run
elsewhere, SAM masking in the AnyLabeling desktop app, and ffmpeg commands typed manually with
hardcoded `apad=pad_dur=1.19` sync corrections.

The asset worth building a product on is the drawing algorithm in
`generate-whiteboard-animated-videos/draw-whiteboard-animations.py`. It splits a thresholded
image into tiles, keeps the ones containing ink, and walks them in nearest-neighbour order so the
hand travels smoothly. Given SAM masks it draws **one object at a time, background last**, which
is what makes the output read as hand-drawn. Commodity whiteboard tools mostly do a naive
edge-reveal.

## Product definition

**For** solo content creators publishing to YouTube, TikTok and Reels
**who** want whiteboard-style story and explainer videos without learning animation software,
**storyboard-ai** takes story text and returns a finished narrated video,
**unlike** template-based tools, by generating the artwork and drawing it object by object.

### Decisions

| Area | Decision |
|---|---|
| Audience | Solo content creators |
| Product | Fully automatic: text in, video out |
| Renderer | Remotion (TypeScript/canvas) |
| Preview | Real-time scrubbable Remotion Player |
| Monetization | Subscription with render quota, free watermarked tier |
| Payments | Paddle or Lemon Squeezy (merchant of record) |
| Image generation | Cheap hosted SDXL now, premium model as an upsell |
| TTS | Cheap neural default, ElevenLabs on the top tier |
| LLM | DeepSeek, behind a provider adapter |
| Aspect ratio | 16:9 and 9:16, chosen per project |
| Stack | Next.js on Vercel + Supabase + Cloudflare R2 |
| Render infra | Remotion Lambda (GitHub Actions for the prototype only) |
| Masking | **Decided by the prototype comparison - see below** |
| Policy | User owns output, 30-day post-cancellation deletion |
| Cadence | Nights and weekends; every phase independently shippable |

## Phase 0 - complete

The Remotion port was the single largest unknown, and it is now settled. See
[`prototype/README.md`](../prototype/README.md) for how to reproduce.

**Result: the port is pixel-identical to the Python reference.** All 3,481 tiles across 9
segments visit in the same order, and all 12 sampled frames match with zero differing pixels -
including frames rendered through Chromium by Remotion itself, not just the shared reveal
function in Node.

Frame cost is **1.38 ms** (about 29x realtime at 25 fps), and a random seek costs 1.08x a
sequential frame, confirming frames carry no accumulated state. Real-time scrubbable preview is
therefore viable, and Remotion Lambda can split a render across workers by frame range.

Also delivered: three bug fixes in the original scripts, a `requirements.txt`, a parity harness
that fails CI on regression, and a manual-dispatch GitHub Actions workflow.

### The open masking question

Phase 0 renders both paths on the same image so the decision comes from output, not argument.
Run `prototype/out/comparison-masked-vs-unmasked.mp4` and judge:

| | With SAM masks | Whole-image draw |
|---|---|---|
| Segments | 9 | 1 |
| Tiles | 3,481 | 2,034 |
| Duration | 472 frames (18.9s) | 329 frames (13.2s) |
| Added cost per scene | ~$0.01-0.05 segmentation | none |
| Look | objects complete one at a time | uniform sweep across the image |

Masking costs roughly 33% of per-video COGS on the cheap tier. If the unmasked version is close
enough on generated art, dropping it is the single largest margin improvement available.

## Phases 1-5

Each phase is independently shippable, sized for nights-and-weekends work. If the scope needs
cutting, cut from the end.

**Phase 1 - Renderer productization.** Both aspect ratios with proper per-ratio layout rather
than letterboxing; the masking decision applied; Remotion Player preview in the browser; audio and
caption compositing (this replaces the whole of `ffmpeg-command-notes.txt`); watermark overlay for
the free tier. Narration length drives the skip rate, so audio and video stay in sync by
construction rather than by hand-tuned `apad` padding.

**Phase 2 - Pipeline services.** Provider adapters for LLM, images, TTS and segmentation, each
behind a narrow interface so the cheap-to-premium upsell is a config change. DeepSeek is the
default LLM; the adapter exists because its API is China-hosted (a GDPR disclosure for EU users)
and its rate limits are less predictable than the incumbents.

**Phase 3 - Application.** Next.js on Vercel; Supabase for auth, Postgres and row-level security;
Cloudflare R2 for rendered video (zero egress fees, which matters as soon as users start sharing);
project CRUD, storyboard view, job queue, render status.

**Phase 4 - Billing.** Paddle or Lemon Squeezy checkout, quota enforcement, tier gating. A
merchant of record handles global VAT/GST registration and remittance, which is the largest
operational burden of selling subscriptions worldwide from a small entity.

**Phase 5 - v1 extras, in ascending cost order.** Brand kit (logo, colours, custom hand,
intro/outro) → background music with ducking → multi-language output → direct publishing to
YouTube/TikTok. Publishing is last because each integration needs OAuth and platform app review.

## Risks

**Scope.** All four v1 extras on top of the Remotion rewrite is roughly the difference between a
6-week and a 4-month build at full time, and considerably longer at nights and weekends. The phase
order above is the cut list.

**Free tier economics.** Three renders per month costs about $0.45 per free user per month with no
revenue attached. See [`cost-model.md`](cost-model.md) for the recommended change.

**Premium tier margin.** Studio at $49 for 100 premium videos is loss-making as specified. Also
in `cost-model.md`.

**Generated-art consistency.** Character appearance drifting between scenes is the most common
complaint about automatic story-video tools. Not yet addressed; worth a spike before Phase 3.

## Open items

- Long-term repository home: the prototype lives in `prototype/` here, decided after Phase 0.
- DeepSeek data residency disclosure for EU/UK users.
- Product name and domain (working name: storyboard-ai).
- A properly licensed royalty-free music library, needed before Phase 5.
