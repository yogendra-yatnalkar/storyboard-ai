import assert from "node:assert/strict";
import { createHmac } from "node:crypto";
import { describe, it } from "node:test";

import {
  applyPlanConstraints,
  canRender,
  nextQuotaReset,
  PLANS,
  shouldResetQuota,
  type Account,
} from "./plans";
import {
  interpretEvent,
  processEvent,
  verifySignature,
  type ProcessedEventStore,
  type WebhookEvent,
} from "./webhooks";

const account = (overrides: Partial<Account> = {}): Account => ({
  id: "acct_1",
  planId: "pro",
  status: "active",
  rendersUsed: 0,
  periodStart: new Date("2026-01-01T00:00:00Z"),
  emailVerified: true,
  ...overrides,
});

describe("canRender", () => {
  it("allows a paid account inside its quota", () => {
    const decision = canRender(account({ rendersUsed: 10 }));
    assert.equal(decision.allowed, true);
    assert.equal(decision.remaining, 20);
    assert.equal(decision.constraints.watermark, false);
  });

  it("stops a paid account at its quota", () => {
    const decision = canRender(account({ rendersUsed: 30 }));
    assert.equal(decision.allowed, false);
    assert.equal(decision.reason, "quota_exhausted");
  });

  it("requires a verified email on free, since free renders cost real money", () => {
    const decision = canRender(
      account({ planId: "free", periodStart: null, emailVerified: false }),
    );
    assert.equal(decision.allowed, false);
    assert.equal(decision.reason, "email_unverified");
  });

  it("watermarks and downscales free renders", () => {
    const decision = canRender(account({ planId: "free", periodStart: null }));
    assert.equal(decision.allowed, true);
    assert.equal(decision.constraints.watermark, true);
    assert.equal(decision.constraints.maxHeight, 720);
    // Segmentation off on free is the single biggest free-tier cost saving.
    assert.equal(decision.constraints.segmentation, false);
  });

  it("drops a lapsed subscription to free rules rather than to nothing", () => {
    const decision = canRender(account({ status: "past_due" }));
    assert.equal(decision.allowed, false);
    assert.equal(decision.reason, "subscription_inactive");
    assert.equal(decision.constraints.watermark, true);
  });

  it("gives Studio premium generation", () => {
    const decision = canRender(account({ planId: "studio" }));
    assert.equal(decision.constraints.tier, "premium");
    assert.equal(decision.remaining, 50);
  });
});

describe("plan economics", () => {
  it("keeps the free allowance lifetime, not monthly", () => {
    // A renewing free allowance is ~$0.45/user/month forever; lifetime makes it
    // a one-off acquisition cost.
    assert.equal(PLANS.free.quotaPeriod, "lifetime");
  });

  it("caps Studio at the quota that stays profitable", () => {
    assert.equal(PLANS.studio.videoQuota, 50);
  });
});

describe("quota periods", () => {
  it("never resets a lifetime allowance", () => {
    const free = account({ planId: "free", periodStart: null });
    assert.equal(nextQuotaReset(free, new Date("2027-01-01T00:00:00Z")), null);
    assert.equal(shouldResetQuota(free, new Date("2030-01-01T00:00:00Z")), false);
  });

  it("rolls a monthly allowance a month after the period start", () => {
    const pro = account();
    assert.equal(
      nextQuotaReset(pro, new Date("2026-01-15T00:00:00Z"))!.toISOString(),
      "2026-02-01T00:00:00.000Z",
    );
    assert.equal(shouldResetQuota(pro, new Date("2026-01-20T00:00:00Z")), false);
    assert.equal(shouldResetQuota(pro, new Date("2026-02-02T00:00:00Z")), true);
  });
});

describe("applyPlanConstraints", () => {
  it("clamps a free render to 720p, unmasked and watermarked", () => {
    const decision = canRender(account({ planId: "free", periodStart: null }));
    const applied = applyPlanConstraints(decision, { height: 1920, segmentation: true });

    assert.equal(applied.height, 720);
    assert.equal(applied.segmentation, false);
    assert.equal(applied.watermark, true);
  });

  it("leaves a Pro render alone", () => {
    const applied = applyPlanConstraints(canRender(account()), {
      height: 1080,
      segmentation: true,
    });
    assert.deepEqual(applied, { height: 1080, segmentation: true, watermark: false });
  });
});

describe("verifySignature", () => {
  const secret = "whsec_test";
  const body = JSON.stringify({ hello: "world" });
  const valid = createHmac("sha256", secret).update(body).digest("hex");

  it("accepts a genuine signature", () => {
    assert.equal(verifySignature(body, valid, secret), true);
  });

  it("rejects a forged one", () => {
    assert.equal(verifySignature(body, "0".repeat(valid.length), secret), false);
  });

  it("rejects a tampered body", () => {
    assert.equal(verifySignature(JSON.stringify({ hello: "evil" }), valid, secret), false);
  });

  it("rejects a wrong-length signature without throwing", () => {
    assert.equal(verifySignature(body, "abc", secret), false);
  });
});

describe("interpretEvent", () => {
  const base = { id: "evt_1", accountId: "acct_1", occurredAt: "2026-01-01T00:00:00Z" };

  it("does not cancel on a failed payment", () => {
    // The provider retries; revoking immediately loses customers who would pay.
    const change = interpretEvent({ ...base, type: "payment.failed" } as WebhookEvent);
    assert.equal(change.status, "past_due");
    assert.equal(change.resetUsage, false);
  });

  it("resets the allowance on a new period", () => {
    const change = interpretEvent({ ...base, type: "payment.succeeded" } as WebhookEvent);
    assert.equal(change.status, "active");
    assert.equal(change.resetUsage, true);
  });

  it("returns a cancelled account to the free plan", () => {
    const change = interpretEvent({ ...base, type: "subscription.cancelled" } as WebhookEvent);
    assert.equal(change.planId, "free");
    assert.equal(change.status, "cancelled");
  });
});

describe("processEvent", () => {
  const makeStore = (): ProcessedEventStore & { seen: Set<string> } => {
    const seen = new Set<string>();
    return {
      seen,
      async has(id) {
        return seen.has(id);
      },
      async record(id) {
        seen.add(id);
      },
    };
  };

  const event: WebhookEvent = {
    id: "evt_1",
    type: "payment.succeeded",
    accountId: "acct_1",
    occurredAt: "2026-02-01T00:00:00Z",
  };

  it("applies an event once", async () => {
    const store = makeStore();
    assert.notEqual(await processEvent(event, store, null), null);
    // Providers retry; the second delivery must be a no-op, not a double reset.
    assert.equal(await processEvent(event, store, null), null);
  });

  it("ignores a delivery older than the state it would overwrite", async () => {
    const store = makeStore();
    const stale: WebhookEvent = {
      ...event,
      id: "evt_old",
      type: "payment.failed",
      occurredAt: "2026-01-01T00:00:00Z",
    };

    const result = await processEvent(stale, store, new Date("2026-02-01T00:00:00Z"));
    assert.equal(result, null, "a late failure must not undo a newer success");
    assert.equal(store.seen.has("evt_old"), true, "still recorded so retries stop");
  });

  it("applies a newer delivery", async () => {
    const store = makeStore();
    const result = await processEvent(event, store, new Date("2026-01-01T00:00:00Z"));
    assert.equal(result?.status, "active");
  });
});
