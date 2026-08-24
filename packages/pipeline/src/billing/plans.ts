/**
 * Plans and quota enforcement.
 *
 * The quotas here are the revised ones from docs/cost-model.md, not the ones
 * originally sketched. Two changes matter:
 *
 *  - Studio is 50 videos, not 100. At $49 for 100 premium videos a fully-using
 *    subscriber costs ~$54 against ~$46.55 net of merchant-of-record fees, which
 *    makes your heaviest users your least profitable.
 *  - Free is 3 videos LIFETIME, not monthly, and renders unmasked at 720p. A
 *    renewing monthly allowance is ~$0.45 per free user per month forever;
 *    lifetime turns it into a one-off acquisition cost.
 *
 * Storyboard preview stays unlimited on free: it costs LLM plus images and no
 * render or TTS, and it is enough to show the product's value.
 */

export type PlanId = "free" | "pro" | "studio";

export type QuotaPeriod = "lifetime" | "month";

export interface Plan {
  id: PlanId;
  name: string;
  /** USD per month. */
  price: number;
  videoQuota: number;
  quotaPeriod: QuotaPeriod;
  maxHeight: number;
  watermark: boolean;
  segmentation: boolean;
  tier: "cheap" | "premium";
  /** Unlimited storyboard previews cost no render and no speech. */
  unlimitedPreviews: boolean;
}

export const PLANS: Record<PlanId, Plan> = {
  free: {
    id: "free",
    name: "Free",
    price: 0,
    videoQuota: 3,
    quotaPeriod: "lifetime",
    maxHeight: 720,
    watermark: true,
    segmentation: false,
    tier: "cheap",
    unlimitedPreviews: true,
  },
  pro: {
    id: "pro",
    name: "Pro",
    price: 19,
    videoQuota: 30,
    quotaPeriod: "month",
    maxHeight: 1080,
    watermark: false,
    segmentation: true,
    tier: "cheap",
    unlimitedPreviews: true,
  },
  studio: {
    id: "studio",
    name: "Studio",
    price: 49,
    videoQuota: 50,
    quotaPeriod: "month",
    maxHeight: 1080,
    watermark: false,
    segmentation: true,
    tier: "premium",
    unlimitedPreviews: true,
  },
};

export type SubscriptionStatus = "active" | "past_due" | "cancelled" | "none";

export interface Account {
  id: string;
  planId: PlanId;
  status: SubscriptionStatus;
  /** Videos rendered in the current period (or ever, for lifetime quotas). */
  rendersUsed: number;
  /** When the current billing period started; null for free accounts. */
  periodStart: Date | null;
  emailVerified: boolean;
}

export type RenderDenialReason =
  | "quota_exhausted"
  | "email_unverified"
  | "subscription_inactive";

export interface RenderDecision {
  allowed: boolean;
  reason?: RenderDenialReason;
  remaining: number;
  /** Constraints the render must honour for this account's plan. */
  constraints: {
    maxHeight: number;
    watermark: boolean;
    segmentation: boolean;
    tier: "cheap" | "premium";
  };
}

/**
 * Can this account render right now, and under what constraints?
 *
 * Deliberately returns constraints alongside the verdict: the render job needs
 * to know the watermark and resolution rules, and deriving them twice is how
 * free-tier renders quietly go out unwatermarked.
 */
export function canRender(account: Account): RenderDecision {
  const plan = PLANS[account.planId];
  const remaining = Math.max(0, plan.videoQuota - account.rendersUsed);
  const constraints = {
    maxHeight: plan.maxHeight,
    watermark: plan.watermark,
    segmentation: plan.segmentation,
    tier: plan.tier,
  };

  // A paid plan that has lapsed drops to free rules rather than to nothing, so a
  // failed card does not look like data loss to the customer.
  if (plan.price > 0 && account.status !== "active") {
    return {
      allowed: false,
      reason: "subscription_inactive",
      remaining,
      constraints: {
        maxHeight: PLANS.free.maxHeight,
        watermark: true,
        segmentation: false,
        tier: "cheap",
      },
    };
  }

  // Verification gate exists because free renders cost real money and a
  // throwaway signup loop is the cheapest way to burn it.
  if (plan.price === 0 && !account.emailVerified) {
    return { allowed: false, reason: "email_unverified", remaining, constraints };
  }

  if (remaining <= 0) {
    return { allowed: false, reason: "quota_exhausted", remaining: 0, constraints };
  }

  return { allowed: true, remaining, constraints };
}

/** Does this account's usage counter reset, and when? */
export function nextQuotaReset(account: Account, now: Date): Date | null {
  const plan = PLANS[account.planId];
  if (plan.quotaPeriod === "lifetime") return null;

  const start = account.periodStart ?? now;
  const reset = new Date(start);
  reset.setUTCMonth(reset.getUTCMonth() + 1);
  return reset;
}

export function shouldResetQuota(account: Account, now: Date): boolean {
  const reset = nextQuotaReset(account, now);
  return reset !== null && now >= reset;
}

/** Clamp a requested render to what the plan permits. */
export function applyPlanConstraints(
  decision: RenderDecision,
  requested: { height: number; segmentation: boolean },
): { height: number; segmentation: boolean; watermark: boolean } {
  return {
    height: Math.min(requested.height, decision.constraints.maxHeight),
    segmentation: requested.segmentation && decision.constraints.segmentation,
    watermark: decision.constraints.watermark,
  };
}
