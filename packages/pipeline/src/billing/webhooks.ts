/**
 * Merchant-of-record webhook handling (Paddle / Lemon Squeezy shaped).
 *
 * A merchant of record was chosen so global VAT/GST registration and remittance
 * are someone else's problem - the trade is that subscription state lives on
 * their side and arrives here as webhooks, so this has to be defensive:
 *
 *  - signatures are verified before anything is trusted;
 *  - events are idempotent, because providers retry and will deliver twice;
 *  - out-of-order delivery is expected, so an older event never overwrites a
 *    newer one.
 */

import { createHmac, timingSafeEqual } from "node:crypto";

import type { PlanId, SubscriptionStatus } from "./plans";

export type WebhookEventType =
  | "subscription.created"
  | "subscription.updated"
  | "subscription.cancelled"
  | "payment.failed"
  | "payment.succeeded";

export interface WebhookEvent {
  id: string;
  type: WebhookEventType;
  accountId: string;
  planId?: PlanId;
  occurredAt: string;
}

export interface SubscriptionChange {
  accountId: string;
  planId?: PlanId;
  status: SubscriptionStatus;
  /** Reset the usage counter - a new paid period has begun. */
  resetUsage: boolean;
  occurredAt: Date;
}

/**
 * Constant-time signature check.
 *
 * Using `===` here would leak the expected signature a byte at a time, which is
 * the standard way webhook endpoints get forged.
 */
export function verifySignature(
  rawBody: string,
  signature: string,
  secret: string,
): boolean {
  const expected = createHmac("sha256", secret).update(rawBody).digest("hex");
  const given = Buffer.from(signature, "utf8");
  const want = Buffer.from(expected, "utf8");
  if (given.length !== want.length) return false;
  return timingSafeEqual(given, want);
}

/** Translate a provider event into the state change it implies. */
export function interpretEvent(event: WebhookEvent): SubscriptionChange {
  const occurredAt = new Date(event.occurredAt);

  switch (event.type) {
    case "subscription.created":
      return {
        accountId: event.accountId,
        planId: event.planId,
        status: "active",
        resetUsage: true,
        occurredAt,
      };

    case "subscription.updated":
      // An upgrade mid-period resets the allowance; the customer just paid for a
      // new quota and withholding it reads as a bug.
      return {
        accountId: event.accountId,
        planId: event.planId,
        status: "active",
        resetUsage: true,
        occurredAt,
      };

    case "payment.succeeded":
      return {
        accountId: event.accountId,
        status: "active",
        resetUsage: true,
        occurredAt,
      };

    case "payment.failed":
      // Not cancelled: the provider will retry, and revoking immediately on a
      // temporary card failure loses customers who would have paid.
      return {
        accountId: event.accountId,
        status: "past_due",
        resetUsage: false,
        occurredAt,
      };

    case "subscription.cancelled":
      return {
        accountId: event.accountId,
        planId: "free",
        status: "cancelled",
        resetUsage: false,
        occurredAt,
      };
  }
}

export interface ProcessedEventStore {
  has(eventId: string): Promise<boolean>;
  record(eventId: string): Promise<void>;
}

/**
 * Apply an event exactly once, ignoring replays and stale deliveries.
 *
 * Returns null when the event should not change anything, so the caller can
 * still return 200 and stop the provider retrying.
 */
export async function processEvent(
  event: WebhookEvent,
  store: ProcessedEventStore,
  currentStateUpdatedAt: Date | null,
): Promise<SubscriptionChange | null> {
  if (await store.has(event.id)) return null;

  const change = interpretEvent(event);

  // Webhooks arrive out of order often enough that this matters: a delayed
  // "payment.failed" must not undo a newer "payment.succeeded".
  if (currentStateUpdatedAt && change.occurredAt < currentStateUpdatedAt) {
    await store.record(event.id);
    return null;
  }

  await store.record(event.id);
  return change;
}
