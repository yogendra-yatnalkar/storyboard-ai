import assert from "node:assert/strict";
import { describe, it } from "node:test";

import { evaluateTier, renderCost, summariseCost } from "./cost";
import type { UsageRecord } from "./types";

const records: UsageRecord[] = [
  { provider: "deepseek", operation: "storyboard", cost: 0.002 },
  { provider: "fal", operation: "image", cost: 0.02 },
  { provider: "fal", operation: "image", cost: 0.02 },
  { provider: "azure", operation: "tts", cost: 0.015 },
];

describe("summariseCost", () => {
  it("rolls up by provider and operation", () => {
    const summary = summariseCost(records);

    assert.ok(Math.abs(summary.total - 0.057) < 1e-9);
    assert.ok(Math.abs(summary.byProvider.fal - 0.04) < 1e-9);
    assert.ok(Math.abs(summary.byOperation.image - 0.04) < 1e-9);
  });

  it("handles an empty run", () => {
    const summary = summariseCost([]);
    assert.equal(summary.total, 0);
    assert.deepEqual(summary.byProvider, {});
  });
});

describe("renderCost", () => {
  it("bills by duration", () => {
    assert.ok(Math.abs(renderCost(60) - 0.03) < 1e-9);
    assert.ok(Math.abs(renderCost(30) - 0.015) < 1e-9);
  });
});

describe("evaluateTier", () => {
  it("flags the planned Studio tier as loss-making", () => {
    // $49 for 100 premium videos at ~$0.54 each: the case that prompted the
    // pricing change in docs/cost-model.md.
    const result = evaluateTier(
      { tier: "premium", price: 49, quota: 100, feeRate: 0.05 },
      0.54,
    );

    assert.equal(result.profitable, false);
    assert.ok(result.cogsAtFullUse > result.netRevenue);
    assert.ok(result.breakEvenVideos < 100);
  });

  it("accepts the revised Studio quota", () => {
    const result = evaluateTier(
      { tier: "premium", price: 49, quota: 50, feeRate: 0.05 },
      0.54,
    );

    assert.equal(result.profitable, true);
    assert.ok(result.marginAtFullUse > 0.4, "revised quota should clear 40% margin");
  });

  it("confirms Pro is healthy at the cheap-tier cost", () => {
    const result = evaluateTier({ tier: "cheap", price: 19, quota: 30, feeRate: 0.05 }, 0.2);

    assert.equal(result.profitable, true);
    assert.ok(result.marginAtFullUse > 0.6);
  });

  it("reports how many videos a plan survives", () => {
    const result = evaluateTier({ tier: "cheap", price: 20, quota: 1000, feeRate: 0 }, 0.5);
    assert.equal(result.breakEvenVideos, 40);
  });

  it("treats a free plan as pure acquisition cost", () => {
    const result = evaluateTier({ tier: "cheap", price: 0, quota: 3, feeRate: 0 }, 0.15);

    assert.equal(result.netRevenue, 0);
    assert.equal(result.profitable, false);
    // The number that makes the free tier a real line item: 1,000 signups is
    // ~$450/month if the allowance renews.
    assert.ok(Math.abs(result.cogsAtFullUse - 0.45) < 1e-9);
  });
});
