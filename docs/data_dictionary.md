# Data Dictionary — Telecom Customer Churn Dataset

**Source file:** `Dataset/churn_dataset.csv`
**Shape:** 7,043 rows × 21 columns
**Grain:** One row = one customer
**Verified against:** raw CSV, directly (see Section 3 for exact value checks)

## 1. Column Reference

| Column | Type | Values / Range | Description |
|---|---|---|---|
| `customerID` | string | 7,043 unique values (e.g. `0002-ORFBO`) | Unique customer identifier. Dropped before modeling — no predictive value, pure identifier. |
| `gender` | categorical | `Female`, `Male` | Customer's gender. |
| `SeniorCitizen` | binary (int) | `0`, `1` | Whether customer is a senior citizen (1 = yes). Stored as int, not Yes/No like other binary fields — inconsistent encoding vs. `Partner`/`Dependents`, worth noting for anyone writing encoding logic. |
| `Partner` | categorical | `Yes`, `No` | Whether customer has a partner. |
| `Dependents` | categorical | `Yes`, `No` | Whether customer has dependents. |
| `tenure` | numeric (int) | 0–72 (months), 73 unique values | Number of months the customer has stayed with the company. `0` = brand-new customer (billed for less than one month). |
| `PhoneService` | categorical | `Yes`, `No` | Whether customer has phone service. |
| `MultipleLines` | categorical | `No phone service`, `No`, `Yes` | Whether customer has multiple phone lines. `No phone service` is a valid third category, not a missing value — it's structurally dependent on `PhoneService = No`. |
| `InternetService` | categorical | `DSL`, `Fiber optic`, `No` | Customer's internet service type. |
| `OnlineSecurity` | categorical | `Yes`, `No`, `No internet service` | Same pattern as `MultipleLines` — third category is structural, dependent on `InternetService = No`. |
| `OnlineBackup` | categorical | `Yes`, `No`, `No internet service` | Same structural pattern. |
| `DeviceProtection` | categorical | `Yes`, `No`, `No internet service` | Same structural pattern. |
| `TechSupport` | categorical | `Yes`, `No`, `No internet service` | Same structural pattern. |
| `StreamingTV` | categorical | `Yes`, `No`, `No internet service` | Same structural pattern. |
| `StreamingMovies` | categorical | `Yes`, `No`, `No internet service` | Same structural pattern. |
| `Contract` | categorical | `Month-to-month`, `One year`, `Two year` | Contract term. Strongest single churn driver identified (see `executive_summary.md`). |
| `PaperlessBilling` | categorical | `Yes`, `No` | Whether customer uses paperless billing. |
| `PaymentMethod` | categorical | `Electronic check`, `Mailed check`, `Bank transfer (automatic)`, `Credit card (automatic)` | Customer's payment method. Electronic check shows the highest churn rate and revenue concentration (see `executive_summary.md`). |
| `MonthlyCharges` | numeric (float) | 18.25 – 118.75, 1,585 unique values | Current monthly charge amount. Clean, no missing values, correctly typed as float in the raw file. |
| `TotalCharges` | **string** (not numeric as-is) | mostly numeric strings; 11 rows contain a literal blank `" "` | Cumulative charges billed to the customer. **Data quality issue:** stored as a string/object column in the raw CSV, not a number — must be coerced with `pd.to_numeric(..., errors='coerce')` before use. |
| `Churn` | categorical (target) | `Yes`, `No` | **Target variable.** Whether the customer left within the last month. Mapped to `1`/`0` before modeling. Class distribution: 73.46% No / 26.54% Yes — moderately imbalanced. |

## 2. Data Quality Issues Found (verified directly against raw CSV)

1. **`TotalCharges` is typed as a string, not a number**, in the raw file —
   confirmed via `df.dtypes`. This is a common trap: `df.isnull().sum()`
   reports **zero** missing values for this column, because the 11
   problem rows aren't `NaN` — they are the literal string `" "` (a single
   space). A naive missing-value check will miss this entirely.
2. **All 11 blank `TotalCharges` rows have `tenure = 0`.** This is not
   random — it's logically consistent: a customer who just signed up
   hasn't been billed a cumulative total yet. This should be documented
   as a *known, explainable* data quality issue, not treated as a random
   defect.
   Affected customer IDs: `4472-LVYGI`, `3115-CZMZD`, `5709-LVOEQ`,
   `4367-NUYAO`, `1371-DWPAZ`, `7644-OMVMY`, `3213-VVOLG`, `2520-SGTTA`,
   `2923-ARZLG`, `4075-WKNIU`, `2775-SEFEE`.
3. **`SeniorCitizen` is encoded as `0`/`1` (int)** while every other binary
   field in the dataset (`Partner`, `Dependents`, `PhoneService`, etc.) is
   encoded as `Yes`/`No` (string). Inconsistent encoding across
   conceptually similar columns — worth normalizing for readability in
   EDA, even though it doesn't affect model training (both get
   encoded/scaled anyway).
4. **No true duplicate rows** and **no missing values in any other
   column** — confirmed via `df.isnull().sum()` and `df.duplicated().sum()`
   directly on the raw file.
5. **Several columns (`MultipleLines`, `OnlineSecurity`, `OnlineBackup`,
   `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`)
   share a structural three-way category** (`Yes` / `No` / `No <service>`)
   that reflects a dependency on `PhoneService` or `InternetService`, not
   a data quality problem. Treating `"No internet service"` as an ordinary
   missing value would be incorrect.

## 3. Recommended Cleaning Steps (feeds Phase 4 — Feature Engineering)
- Coerce `TotalCharges` to numeric with `pd.to_numeric(errors='coerce')`.
- For the resulting 11 nulls: since all have `tenure = 0`, fill with `0`
  (not median) — this reflects the actual business reality (no charges
  yet), and is more defensible than an arbitrary median substitution.
- Drop `customerID` before modeling (identifier, zero predictive signal).
- Map `Churn` to binary (`Yes` → 1, `No` → 0) for modeling.
- Leave the three-way categorical columns as-is; they carry real
  information and should be one-hot encoded normally.

---
*All values in this document were computed directly from `churn_dataset.csv`,
not estimated or recalled from prior notebook output. Re-run the
verification script (Appendix, project repo) if the source file changes.*