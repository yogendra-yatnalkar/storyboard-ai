/**
 * Pricing page.
 *
 * The quotas shown are the revised ones from docs/cost-model.md: Studio is 50
 * videos rather than 100 (at 100 a fully-using subscriber costs more than they
 * pay), and Free is a lifetime allowance rather than a monthly one.
 */

const PLANS = [
  {
    name: "Free",
    price: "$0",
    cadence: "",
    featured: false,
    quota: "3 videos, ever",
    features: [
      "720p with a watermark",
      "Unlimited storyboard previews",
      "Standard voice",
      "Whole-image drawing",
    ],
  },
  {
    name: "Pro",
    price: "$19",
    cadence: "/month",
    featured: true,
    quota: "30 videos a month",
    features: [
      "1080p, no watermark",
      "Object-by-object drawing",
      "9:16 and 16:9",
      "Standard voice",
    ],
  },
  {
    name: "Studio",
    price: "$49",
    cadence: "/month",
    featured: false,
    quota: "50 videos a month",
    features: [
      "Everything in Pro",
      "Premium illustration model",
      "Premium voices",
      "Brand kit and music",
    ],
  },
];

export default function PricingPage() {
  return (
    <>
      <h1>Pricing</h1>
      <p className="lede">Every plan renders the same hand-drawn animation. Quotas differ.</p>

      <div className="plans">
        {PLANS.map((plan) => (
          <div className={`plan${plan.featured ? " featured" : ""}`} key={plan.name}>
            <strong>{plan.name}</strong>
            <div className="price">
              {plan.price}
              <small>{plan.cadence}</small>
            </div>
            <div className="meta">{plan.quota}</div>
            <ul>
              {plan.features.map((feature) => (
                <li key={feature}>{feature}</li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <p className="note">
        Billed through a merchant of record, so VAT and GST are handled at checkout. Cancel any
        time; projects stay available for 30 days afterwards.
      </p>
    </>
  );
}
