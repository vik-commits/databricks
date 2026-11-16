# Retail Churn Demo Dataset (2024–2025)

Synthetic data for a customer-churn classification demo on Databricks. Seed 42, reproducible with `generate_churn_data.py`.

## Files

| File | Rows | Load into |
|---|---|---|
| customers.csv | 2,010 (2,000 + 10 dupes) | Bronze |
| stores.csv | 12 (clean) | Bronze |
| transactions.csv | ~78.8k | Bronze |
| engagement.csv | ~43.2k (1 row per customer per month) | Bronze |
| churn_answer_key.csv | 2,000 | **Validation only — never join into features** |

### customers.csv
Customer_Id, Customer_name, Customer_address, Email, Phone_number, Membership_tier (Bronze/Silver/Gold), Signup_date (2021-01 to 2025-06), Home_store_id, Email_opt_in

### stores.csv
Store_id, Store_location, Store_Region (Northeast/Southeast/Midwest/Southwest/West), Store_open_date

### transactions.csv
Transaction_Id, Customer_id (blank = walk-in, ~15%), Store_id, Transaction_date (2024-01-01 to 2025-12-31), Number_of_products_Sold (negative for returns), Sale_Amount ($5–$500), Transaction_Status (Sold/Returned)

### engagement.csv
Customer_Id, Month (YYYY-MM), Emails_sent (8 if opted in, else 0), Emails_opened, App_logins

### churn_answer_key.csv
Customer_Id, Persona, True_churn_date, Churn_label_Q4_2024, Churn_label_Q4_2025

## Churn definition (used for the labels)
For a snapshot year Y:
- **Active:** at least one *Sold* transaction between Apr 1 and Sep 30 of Y
- **Churned = 1:** active, with **no** *Sold* transaction between Oct 1 and Dec 31 of Y
- Not active → no label (blank)

Two snapshots are provided:

| Snapshot | Active customers | Churn rate |
|---|---|---|
| Q4 2024 | ~1,620 | ~12% |
| Q4 2025 | ~1,590 | ~19% |

**Suggested setup:** build features as of Sep 30 2024 → train on the Q4 2024 labels → build features as of Sep 30 2025 → test on Q4 2025 (an out-of-time validation).

## Planted signal (what the model should find)
| Persona | Share | Behavior |
|---|---|---|
| loyal | ~43% | Steady visits, holiday bump in Nov/Dec |
| decline_churn | ~22% | Visits taper off 3–6 months before churn, return rate rises, engagement falls ~2 months *ahead* of purchases |
| sudden_churn | ~15% | Stops with no warning (noise) |
| dip_recover | ~10% | Temporary 3–5 month slump, then recovers (false-alarm noise) |
| occasional | ~9% | Low, irregular activity |

Churn is skewed toward the **Bronze** tier and **Southwest** stores (the story: a competitor opened there).

**Baseline check:** with simple RFM, trend and engagement features, training on 2024 and testing on 2025 gives about **AUC 0.86–0.87**. Precision in the top 10% is ~0.83, against a 0.19 base rate (about 4.4× lift).

## Injected data-quality issues (for the Silver layer)
- customers: 10 exact duplicates, 20 case or whitespace variants of the tier, 20 malformed emails, 15 null phones
- transactions: ~1% exact duplicates (~780), ~0.5% nulls each in date, product count and amount
- engagement: ~0.5% exact duplicates

Stores and the answer key are clean.
