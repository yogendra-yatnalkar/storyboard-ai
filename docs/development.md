# Development

## Layout

```
generate-whiteboard-animated-videos/   original Python renderer (the reference)
tts/  text-processing/  adding-subtitles/   original pipeline stages
prototype/
  reference/           faithful, path-free extraction of the original algorithm
  remotion-renderer/   the Remotion port + build scripts + parity harness
  fixtures/            scene lists and test images
packages/pipeline/     generation: providers, orchestration, cost, billing
apps/web/              Next.js app
supabase/migrations/   database schema and RLS policies
docs/                  product plan, architecture, cost model, testing
```

## Prerequisites

- Node 22+
- Python 3.11+ (only for the reference renderer and parity ground truth)
- A Chromium with the **old headless mode** for Remotion. A modern `chrome`
  binary will not launch; use `chrome-headless-shell`:

  ```bash
  export REMOTION_BROWSER_EXECUTABLE=/path/to/chrome-headless-shell
  ```

  Remotion downloads its own if this is unset.

## Setup

```bash
pip install -r requirements.txt
(cd prototype/remotion-renderer && npm install)
(cd packages/pipeline          && npm install)
(cd apps/web                   && npm install)
```

## Common tasks

```bash
# unit tests
(cd prototype/remotion-renderer && npm test)
(cd packages/pipeline && npm test)

# parity - run this after ANY renderer change
(cd prototype/remotion-renderer && npx tsx scripts/compare.ts --name scene-2)

# generate a project offline (no API keys)
(cd packages/pipeline && npx tsx src/generate.ts --input "a story" --out ../../prototype/out/generated)

# build + render a video
cd prototype/remotion-renderer
npx tsx scripts/buildVideo.ts --fixture ../fixtures/shepherd-boy.json --name story --aspect portrait
npx remotion render Story-Portrait "$PWD/../out/story-portrait.mp4"

# scrub the timeline
npx remotion studio

# web app
(cd apps/web && npm run dev)     # http://localhost:3100
```

## Environment variables

Everything runs with fixture providers by default. Set `PROVIDERS=live` to use
real ones:

| Variable | Used for | Required |
|---|---|---|
| `PROVIDERS` | `fixture` (default) or `live` | no |
| `DEEPSEEK_API_KEY` | Storyboard generation | when live |
| `IMAGE_API_KEY` | Illustration generation | when live |
| `TTS_API_KEY` | Narration | when live |
| `SEGMENTATION_API_KEY` | Object masking; omitted disables it | no |
| `REMOTION_BROWSER_EXECUTABLE` | Headless shell path | if not auto-downloading |

Supabase, R2 and Paddle credentials are not yet wired to the app — see
[status.md](status.md) for exactly what is and is not connected.

## Conventions

**The parity harness is the contract.** The renderer's job is to reproduce the
original animation exactly. Any change to `tilePath.ts`, `preprocess.ts`,
`reveal.ts` or the reference renderer must leave `compare.ts` reporting 0
differing pixels. If a change is *meant* to alter the output, regenerate the
ground truth deliberately and say so in the commit.

**Three behaviours in the algorithm are load-bearing** and documented in
[architecture.md](architecture.md#parity-constraints). They look like bugs and
are not safe to tidy: the unused CLAHE, the swap-with-last removal, and the loop
stopping one tile early.

**Providers stay behind their interfaces.** The cheap-to-premium upsell, and any
provider swap, should be a config change. Nothing outside `src/providers/` should
name a vendor.

**Precompute is where expensive work goes.** The composition must stay a pure
function of frame — that is what makes scrubbing, Lambda's frame-range splitting
and cheap re-renders all work. If you find yourself wanting per-frame state,
move it into `precompute.ts` instead.

## Adding a provider

1. Implement the interface in `packages/pipeline/src/providers/index.ts`.
2. Report `UsageRecord`s with real prices so cost stays measured, not estimated.
3. Add it to `buildProviders()` in `src/generate.ts`.
4. Test against a stubbed `fetch` — see the DeepSeek tests for the pattern.

## Adding a caption source

Implement `CaptionSource` in `src/captions/types.ts`. Providers that return word
timings should use `ProviderTimingSource`; anything else can chain through
`FallbackCaptionSource` to the Whisper or proportional path.
