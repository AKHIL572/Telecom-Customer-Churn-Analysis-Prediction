# Executive Summary — Telecom Customer Churn Analysis

## 1. Business Problem
Telecom providers lose significant recurring revenue every month to customer
churn. In this dataset, **1,869 of 7,043 customers (26.5%)** have churned,
representing an estimated **$139,130 in monthly recurring revenue at risk**.

Acquiring a new customer typically costs far more than retaining an existing
one. Without a way to identify *which* customers are likely to leave and
*why*, retention efforts are reactive, unfocused, and expensive — offers get
sent broadly instead of to the customers who actually need them.

## 2. Objective
Build a system that identifies customers at high risk of churning **before**
they leave, so the Retention team can prioritize outreach and tailor offers
to the specific reasons a customer is at risk — rather than applying
blanket discounts across the entire customer base.

## 3. Stakeholder
**Primary stakeholder:** Head of Customer Retention / Customer Success team.
**Secondary stakeholders:** Finance (revenue-at-risk reporting), Marketing
(designing targeted retention offers).

## 4. Decision This Project Supports
> "Which customers should the Retention team contact this month, with what
> kind of offer, to reduce churn at the lowest possible cost?"

This is a **prioritization and targeting** problem, not just a prediction
problem — the output needs to be something a retention team can act on
directly, not just a probability score.

## 5. Success Metric — Why Recall Is Prioritized Over Precision
Two types of mistakes are possible:
- **False Negative** (missed churner): the company loses that customer's
  full future revenue — the expensive mistake.
- **False Positive** (flagged a loyal customer): the company spends a
  retention offer (e.g., a discount) on someone who wasn't going to leave
  anyway — a cheap mistake by comparison.

Because the cost of a **miss** is far higher than the cost of a **false
alarm**, this project deliberately optimizes for **recall** over precision.
The final model (Random Forest, threshold = 0.25) catches ~96% of actual
churners, at the cost of a higher false-positive rate (~62% of flagged
customers are not true churners). This trade-off is intentional and is
revisited with actual cost estimates in Section 7.

## 6. Headline Findings
*(Sourced from EDA — Notebook 2 and Power BI dashboard)*
- **Contract type is the strongest churn driver identified:** month-to-month
  customers churn at **43%**, versus roughly **3%** for two-year contract
  customers.
- **Tenure matters:** churn rate is **47.68%** for customers in their first
  year, dropping to **28.71%** (1–2 yrs), **20.39%** (2–4 yrs), and
  **9.51%** (4+ yrs) — a clear, steady decline as tenure increases.
- Customers without **Tech Support** churn at **41.64%**, versus **15.17%**
  for those who have it.
- Customers without **Online Security** churn at **41.77%**, versus
  **14.61%** for those who have it.
- **Electronic check** users churn at **45.29%** — the highest of any
  payment method (Bank transfer 16.71%, Credit card 15.24%, Mailed check
  19.11%) — and account for **$84,288.75 of the $139,130.85 total monthly
  revenue at risk (60.6%)**, despite being one of four payment methods.
  This is the single most concentrated, actionable finding in the dataset.
- **Senior citizens churn at 41.68%**, nearly double the rate of
  non-seniors (23.61%) — a real, verified pattern, though not yet
  visualized in the EDA notebook (see `data_dictionary.md` / Phase 3
  follow-up).
- A **high-risk segment** was identified: customers with month-to-month
  contracts, fiber optic internet, and under 12 months tenure — this
  combination shows compounding risk beyond any single factor alone.

## 7. Recommended Actions
*(Placeholder — finalized with dollar-quantified impact in Phase 7/8, once
the model and dashboard are locked. Do not treat these as final numbers.)*
- Target month-to-month, low-tenure, fiber-optic customers first — the
  highest-density risk segment.
- Offer contract-upgrade incentives (e.g., discounted annual pricing) to
  month-to-month customers, given the steep churn drop-off at longer
  contract lengths.
- **Prioritize Electronic check customers first** — they represent 60.6%
  of all revenue at risk in a single segment, making this the highest
  concentration/lowest-effort target for a first retention campaign.
  Investigate whether this is a proxy for a broader customer segment
  (e.g., lower engagement, different demographic) or a friction issue
  with the payment method itself (e.g., encourage migration to automatic
  payment methods, which show 15–17% churn vs. 45%).
- Proactively offer Tech Support / Online Security add-ons to at-risk
  customers who currently lack them — both show churn rates roughly
  2.5–2.9x higher than customers who have these services.
- Flag senior citizens as a distinct at-risk segment (41.68% vs. 23.61%
  churn) for tailored outreach, pending further EDA to understand drivers
  (see Section 8 / Phase 3 follow-up).

## 8. Project Scope & Boundaries
- This analysis is based on a **single static snapshot** of customer data
  (no time-series history) — it cannot detect *changes* in a customer's
  risk over time or seasonal churn patterns.
- The model predicts **likelihood of churn**, not the *reason* a specific
  customer might churn — reason codes/interpretability are addressed
  separately (see `model_card.md`, Phase 5).
- Financial estimates (e.g., revenue at risk) assume churned customers'
  last known `MonthlyCharges` as a proxy for lost recurring revenue; it
  does not account for acquisition cost, offer cost, or customer lifetime
  value.

---
*This document sets the objective and success criteria for all downstream
work in this project (EDA, feature engineering, modeling, deployment, and
reporting). Any change to the recall/precision priority in Section 5 should
be reflected back through the modeling phase.*