/**
 * Per-video cost accounting.
 *
 * The margin analysis in docs/cost-model.md is only as good as its inputs, so
 * every provider call reports what it spent and this rolls it up. Once real keys
 * exist these become measurements rather than estimates, and the tier quotas can
 * be set from data.
 */

import type { QualityTier, UsageRecord } from "./types";

export interface CostBreakdown {
  total: number;
  byProvider: Record<string, number>;
  byOperation: Record<string, number>;
  records: UsageRecord[];
}

export function summariseCost(records: UsageRecord[]): CostBreakdown {
  const byProvider: Record<string, number> = {};
  const byOperation: Record<string, number> = {};
  let total = 0;

  for (const record of records) {
    total += record.cost;
    byProvider[record.provider] = (byProvider[record.provider] ?? 0) + record.cost;
    byOperation[record.operation] = (byOperation[record.operation] ?? 0) + record.cost;
  }

  return { total, byProvider, byOperation, records };
}

/** Render cost is billed by the render backend, not by a generation provider. */
export const RENDER_COST_PER_MINUTE = 0.03;

export function renderCost(durationSeconds: number): number {
  return (durationSeconds / 60) * RENDER_COST_PER_MINUTE;
}

export interface TierEconomics {
  tier: QualityTier;
  price: number;
  quota: number;
  /** Merchant-of-record fee share, e.g. 0.05 for Paddle. */
  feeRate: number;
}

export interface MarginResult {
  netRevenue: number;
  cogsAtFullUse: number;
  marginAtFullUse: number;
  /** Videos a subscriber can render before the plan stops being profitable. */
  breakEvenVideos: number;
  profitable: boolean;
}

/**
 * Does a plan survive a subscriber who uses their whole quota?
 *
 * This is the check that caught Studio at $49 for 100 premium videos being
 * loss-making: $54 of COGS against $46.55 net revenue, which shows up not as an
 * aggregate loss at first but as your heaviest users being your least profitable.
 */
export function evaluateTier(
  economics: TierEconomics,
  costPerVideo: number,
): MarginResult {
  const netRevenue = economics.price * (1 - economics.feeRate);
  const cogsAtFullUse = costPerVideo * economics.quota;

  return {
    netRevenue,
    cogsAtFullUse,
    marginAtFullUse: netRevenue === 0 ? 0 : (netRevenue - cogsAtFullUse) / netRevenue,
    breakEvenVideos: costPerVideo === 0 ? Infinity : netRevenue / costPerVideo,
    profitable: cogsAtFullUse <= netRevenue,
  };
}
