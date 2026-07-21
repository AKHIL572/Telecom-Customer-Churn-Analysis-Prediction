# Dashboard Insights & Recommendations

Companion document to `Power_BI/Churn_Details.pbix`. Every number below is
verified directly against `Dataset/churn_dataset.csv` (see
`docs/data_dictionary.md` and `docs/executive_summary.md` for the full
verification trail). Use this alongside the dashboard PDF export for anyone
who won't open the `.pbix` file directly.

---

## Page 1: Executive Overview

**What it shows:** Total Customers (7,043), Total Churned (1,869), Overall
Churn Rate (26.5%), Monthly Revenue at Risk ($139,130.85), plus four charts:
Churn Count by Status, Churn Rate by Contract Type, Customer Base by Tenure
& Churn, Revenue at Risk by Payment Method.

**Insights (add as text boxes next to each chart in Power BI Desktop):**

> **Churn Rate by Contract Type:** Month-to-month customers churn at 42.71%,
> versus 11.27% for one-year and just 2.83% for two-year contracts.
> Recommendation: prioritize contract-upgrade incentives for month-to-month
> customers — this is the single largest lever in the dataset for reducing
> churn.

> **Revenue at Risk by Payment Method:** Electronic check customers account
> for $84,288.75 of the $139,130.85 total revenue at risk — 60.6% of all
> at-risk revenue concentrated in one payment method, despite being one of
> four options. Recommendation: investigate this segment first (see Page 2)
> — it's the highest-concentration, likely lowest-effort target for an
> initial retention campaign.

---

## Page 2: Retention Analysis

**What it shows:** Impact of Tech Support on Churn, Impact of Online
Security on Churn, Churn Rate Trend by Tenure Group, and a "High-Risk
Customers for Retention" table.

**Insights (add as text boxes next to each chart):**

> **Impact of Tech Support:** customers without Tech Support churn at
> 41.64%, versus 15.17% for those who have it — a 2.7x difference.
> Recommendation: proactively offer Tech Support add-ons to flagged
> high-risk customers who currently lack it.

> **Impact of Online Security:** customers without Online Security churn at
> 41.77%, versus 14.61% for those who have it — nearly identical pattern to
> Tech Support. Recommendation: bundle both add-ons into a single retention
> offer rather than treating them separately, since they show the same
> effect and likely the same underlying customer segment.

> **Churn Rate Trend by Tenure Group:** churn drops steadily from 47.44%
> (0-1yr) to 28.71% (1-2yr) to 20.39% (2-4yr) to 9.51% (4+yr). Recommendation:
> the first 12 months is the highest-risk window — consider a structured
> onboarding/check-in program specifically for new customers, since this is
> where the largest single drop in risk would come from.

> **High-Risk Customers for Retention (table):** this list should not be
> treated as a single undifferentiated target. Recommendation: prioritize
> within this list using the two strongest levers above — Electronic check
> payment method and missing Tech Support/Online Security — rather than
> contacting every listed customer with the same generic offer.

---

## Cross-Page Recommendation Summary

Ranked by estimated impact (revenue concentration x actionability), not
just churn rate alone:

1. **Electronic check customers** — 60.6% of all revenue at risk in one
   segment. Highest-priority, most concentrated target.
2. **Month-to-month contract customers** — the single strongest churn
   driver (Cramér's V = 0.41, see `2_EDA.ipynb`). Contract-upgrade offers
   here have the broadest reach.
3. **New customers (0-12 months tenure)** — highest churn rate segment
   (47.44%); a proactive onboarding program targets risk at its source
   rather than after it's already elevated.
4. **Customers missing Tech Support / Online Security** — a clear,
   low-cost add-on offer with a well-supported 2.7x churn difference.

**What this document does not yet include:** a costed estimate of what each
recommendation is worth in retained revenue vs. the cost of the offer
itself. That quantification happens in `docs/executive_summary.md` Section 7
once finalized — see that document for the dollar-level business case
before running any of these campaigns.