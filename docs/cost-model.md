# Cost model

All figures are estimates at list prices as of the analysis, for a **5-scene, ~60-second video**.
Provider pricing moves; re-check before committing to a pricing page. Treat these as the shape of
the economics, not a quote.

## Per-video COGS

### Cheap tier (free and Pro plans)

| Item | Basis | Cost |
|---|---|---|
| LLM - scene split, image prompts, narration | DeepSeek, ~2k in / 1k out | $0.002 |
| Images | 5 x hosted SDXL @ ~$0.02 | $0.100 |
| Segmentation | 5 x SAM @ ~$0.01 | $0.050 |
| TTS | ~900 chars, neural @ ~$16/1M chars | $0.015 |
| Render | Remotion Lambda, 1080p 60s | $0.030 |
| Storage + egress | R2, zero egress | ~$0.001 |
| **Total** | | **~$0.20** |

**Without segmentation: ~$0.15.** Masking is about a third of cheap-tier COGS, which is why the
masked-vs-unmasked comparison in Phase 0 is worth judging carefully.

### Premium tier (Studio plan)

| Item | Basis | Cost |
|---|---|---|
| LLM | as above | $0.002 |
| Images | 5 x premium model @ ~$0.05 | $0.250 |
| Segmentation | as above | $0.050 |
| TTS | ~900 chars, ElevenLabs @ ~$0.22/1k chars | $0.200 |
| Render | as above | $0.030 |
| **Total** | | **~$0.54** |

## Tier analysis

Net revenue is after ~5% merchant-of-record fees (Paddle/Lemon Squeezy).

| Tier | Price | Quota | Net revenue | COGS at full use | Margin |
|---|---|---|---|---|---|
| Free | $0 | 3/month, watermarked | $0 | $0.45 | **-$0.45/user** |
| Pro | $19 | 30 videos | $18.05 | $6.00 | **67%** |
| Studio (as specified) | $49 | ~100 premium | $46.55 | $54.00 | **-16%** |

**Pro is healthy.** The other two need changes.

### Problem 1: Studio is loss-making

At $49 for 100 premium videos, a fully-utilising customer costs $54 against $46.55 net revenue.
Most customers will not hit the cap, so this may not show up in aggregate at first - it shows up
as your best customers being your least profitable, which is the worst shape for a churn-sensitive
creator business.

Options, in order of preference:

1. **$49 for 40 videos** - ~54% margin, and 40 premium videos a month is generous for a solo
   creator.
2. **$79 for 100 videos** - ~27% margin; thin, and a big jump from $19.
3. **$49 for 50 videos** - ~42% margin; the most defensible middle.

### Problem 2: the free tier is an open cost tap

Three renders per month is ~$0.45 per free user per month, forever, with no revenue. A thousand
signups is ~$450/month, and it does not decay - dormant accounts keep their monthly allowance.

Recommended, in combination:

- Make it **3 videos lifetime**, not monthly. Trial usage is the goal; a renewing free allowance
  makes the free tier a product.
- Require **email verification** before the first render.
- Keep the **storyboard preview unlimited** on free. Preview costs LLM plus images (~$0.10) and no
  render or TTS, and it is enough to show the product's value.
- Free renders use the **unmasked path at 720p**, cutting free-tier COGS to roughly $0.11.

### Recommended pricing

| Tier | Price | Quota | COGS at full use | Margin |
|---|---|---|---|---|
| Free | $0 | 3 lifetime, watermarked 720p | $0.33 once | one-off acquisition cost |
| Pro | $19 | 30 videos, 1080p, no watermark | $6.00 | 67% |
| Studio | $49 | 50 videos, premium art and voice | $27.00 | 42% |

## What moves the numbers most

1. **Dropping segmentation** - 25% of cheap-tier COGS, and Phase 0 gives you the output to judge
   it on.
2. **Image count per video** - images are half of cheap-tier COGS. Four scenes instead of five is
   a 20% saving; reusing a scene image across two narration beats is more.
3. **Free-tier design** - the difference between "3 lifetime, unmasked, 720p" and "3 monthly,
   full quality" is roughly 4x on customer acquisition cost.
4. **TTS tier** - ElevenLabs is 13x the neural default and the single largest premium-tier line
   item after images.

Render cost is not on this list. At ~$0.03 per video it is under 15% of COGS, so optimizing the
renderer further buys less than any of the above.

## Fixed costs

Roughly $50-100/month at low volume: Vercel Pro, Supabase Pro, R2 storage, domain, plus a small
allowance for provider minimums. Break-even is on the order of 5 Pro subscribers, so the model
works early - the risk is free-tier volume, not fixed cost.
