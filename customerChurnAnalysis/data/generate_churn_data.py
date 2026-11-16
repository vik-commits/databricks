"""
Synthetic retail churn dataset generator (2024-2025).

Produces:
  customers.csv, stores.csv, transactions.csv, engagement.csv  -> load into Bronze
  churn_answer_key.csv                                          -> validation ONLY, never a feature

Planted signal: each customer gets a hidden persona. Churners taper off
(fewer visits, more returns, engagement dropping ~2 months earlier) before a
churn date, then stop. Churn is skewed to Bronze tier and the Southwest region
("competitor opened mid-2024"). Noise: sudden churners with no warning, and
loyal customers who dip then recover.

Usage:  python generate_churn_data.py [output_dir]
"""
import sys
import os
import numpy as np
import pandas as pd
from faker import Faker

# ---------------- knobs ----------------
SEED = 42
N_CUSTOMERS = 2000
START = pd.Timestamp("2024-01-01")
END = pd.Timestamp("2025-12-31")
WALKIN_SHARE = 0.15          # share of transactions with blank Customer_id
DQ = dict(cust_dupes=10, tier_case=20, bad_email=20, null_phone=15,
          txn_dupe_frac=0.01, txn_null_frac=0.005, eng_dupe_frac=0.005)
# ---------------------------------------

rng = np.random.default_rng(SEED)
fake = Faker("en_US")
Faker.seed(SEED)
OUT = sys.argv[1] if len(sys.argv) > 1 else "output"
os.makedirs(OUT, exist_ok=True)

MONTHS = pd.date_range(START, END, freq="MS")
# seasonality: holiday bump Nov/Dec, small summer bump
SEASON = {1: 0.85, 2: 0.85, 3: 0.95, 4: 1.0, 5: 1.0, 6: 1.05, 7: 1.1,
          8: 1.05, 9: 0.95, 10: 1.0, 11: 1.3, 12: 1.5}

# ---------------- stores ----------------
store_rows = [
    ("Boston", "Northeast"), ("New York", "Northeast"), ("Philadelphia", "Northeast"),
    ("Atlanta", "Southeast"), ("Miami", "Southeast"),
    ("Chicago", "Midwest"), ("Minneapolis", "Midwest"),
    ("Phoenix", "Southwest"), ("Albuquerque", "Southwest"), ("Tucson", "Southwest"),
    ("Seattle", "West"), ("Denver", "West"),
]
stores = pd.DataFrame({
    "Store_id": [f"S{i+1:03d}" for i in range(len(store_rows))],
    "Store_location": [r[0] for r in store_rows],
    "Store_Region": [r[1] for r in store_rows],
    "Store_open_date": ["2018-03-01"] * len(store_rows),
})
store_ids = stores["Store_id"].to_numpy()
store_region = dict(zip(stores.Store_id, stores.Store_Region))

# ---------------- customers ----------------
n = N_CUSTOMERS
cust_ids = np.array([f"CUST-{i+1:05d}" for i in range(n)])
early = rng.random(n) < 0.7
signup = np.where(
    early,
    pd.to_datetime(rng.integers(pd.Timestamp("2021-01-01").value // 10**9,
                                pd.Timestamp("2023-12-31").value // 10**9, n), unit="s"),
    pd.to_datetime(rng.integers(pd.Timestamp("2024-01-01").value // 10**9,
                                pd.Timestamp("2025-06-30").value // 10**9, n), unit="s"),
)
signup = pd.to_datetime(signup).normalize()
tier = rng.choice(["Bronze", "Silver", "Gold"], n, p=[0.5, 0.32, 0.18])
home_store = rng.choice(store_ids, n)
opt_in = rng.random(n) < 0.7

# hidden persona
p_churn = np.full(n, 0.30)
p_churn *= np.select([tier == "Bronze", tier == "Gold"], [1.35, 0.55], 1.0)
p_churn *= np.where(np.vectorize(store_region.get)(home_store) == "Southwest", 1.7, 1.0)
p_churn = np.clip(p_churn, 0, 0.85)
is_churner = rng.random(n) < p_churn
occasional = rng.random(n) < 0.15
persona = np.where(is_churner,
                   np.where(rng.random(n) < 0.4, "sudden_churn", "decline_churn"),
                   np.where(occasional, "occasional",
                            np.where(rng.random(n) < 0.2, "dip_recover", "loyal")))

# churn dates
earliest = np.maximum(signup + pd.Timedelta(days=120), pd.Timestamp("2024-03-01"))
churn_date = []
for i in range(n):
    if is_churner[i] and earliest[i] < END:
        lo, hi = earliest[i].value, END.value
        churn_date.append(pd.Timestamp(rng.integers(lo, hi)).normalize())
    else:
        churn_date.append(pd.NaT)
churn_date = pd.DatetimeIndex(churn_date)
# churners whose window didn't fit become loyal
persona = np.where(is_churner & churn_date.isna(), "loyal", persona)

base_rate = np.select([tier == "Bronze", tier == "Silver"], [1.2, 2.0], 3.2)
base_rate = base_rate * rng.lognormal(0, 0.35, n)
base_rate = np.where(persona == "occasional", base_rate * 0.3, base_rate)
decline_len = rng.integers(3, 7, n)                     # months of taper
dip_start = pd.to_datetime(rng.integers(START.value, pd.Timestamp("2025-06-01").value, n)).normalize()
dip_len = rng.integers(3, 6, n)
open_rate = rng.beta(4, 6, n)
login_rate = rng.gamma(2, 1.5, n) * np.select([tier == "Gold", tier == "Silver"], [1.5, 1.1], 0.8)


def months_between(a, b):
    return (a.year - b.year) * 12 + (a.month - b.month) + (a.day - b.day) / 30.0


def activity_mult(i, t):
    """Purchase-activity multiplier at time t for customer i (1 = normal, 0 = gone)."""
    p = persona[i]
    if p in ("decline_churn", "sudden_churn"):
        cd = churn_date[i]
        if t >= cd:
            return 0.0
        if p == "decline_churn":
            m_left = months_between(cd, t)
            if m_left < decline_len[i]:
                return 0.4 + 0.6 * (m_left / decline_len[i])
        return 1.0
    if p == "dip_recover":
        into = months_between(t, dip_start[i])
        if 0 <= into < dip_len[i]:
            return 0.35
    return 1.0


# ---------------- transactions + engagement ----------------
txn = []
eng = []
for i in range(n):
    first = max(signup[i], START)
    for ms in MONTHS:
        me = ms + pd.offsets.MonthEnd(0)
        if me < first:
            continue
        mid = ms + pd.Timedelta(days=14)
        mult = activity_mult(i, mid)
        # ---- engagement leads purchases by ~2 months
        e_mult = activity_mult(i, mid + pd.DateOffset(months=2))
        if persona[i] in ("decline_churn", "sudden_churn") and mid >= churn_date[i]:
            e_mult = 0.0
        if persona[i] == "sudden_churn" and mid < churn_date[i]:
            e_mult = 1.0  # no warning
        sent = 8 if opt_in[i] else 0
        opened = rng.binomial(sent, min(1.0, open_rate[i] * (0.45 + 0.55 * e_mult))) if sent else 0
        logins = rng.poisson(login_rate[i] * (0.4 + 0.6 * e_mult))
        eng.append((cust_ids[i], ms.strftime("%Y-%m"), sent, opened, logins))

        if mult == 0:
            continue
        k = rng.poisson(base_rate[i] * SEASON[ms.month] * mult)
        lo = max(ms, signup[i])
        hi = me
        if persona[i] in ("decline_churn", "sudden_churn"):
            hi = min(hi, churn_date[i] - pd.Timedelta(days=1))
        if hi < lo or k == 0:
            continue
        days = rng.integers(lo.value, hi.value + 1, k)
        declining = mult < 1.0 and persona[i] == "decline_churn"
        p_ret = 0.05 + (0.18 * (1 - mult) if declining else 0) + (0.10 if persona[i] == "dip_recover" and mult < 1 else 0)
        for d in days:
            st = home_store[i] if rng.random() < 0.85 else rng.choice(store_ids)
            qty = int(np.clip(rng.poisson(3 if tier[i] != "Gold" else 4) + 1, 1, 10))
            amt = float(np.clip(qty * rng.lognormal(3.1, 0.5), 5, 500))
            status = "Returned" if rng.random() < p_ret else "Sold"
            txn.append((cust_ids[i], st, pd.Timestamp(d).normalize(),
                        -qty if status == "Returned" else qty, round(amt, 2), status))

# walk-ins
n_walk = int(len(txn) * WALKIN_SHARE / (1 - WALKIN_SHARE))
wdays = rng.integers(START.value, END.value, n_walk)
for d in wdays:
    qty = int(np.clip(rng.poisson(2) + 1, 1, 10))
    amt = float(np.clip(qty * rng.lognormal(3.0, 0.5), 5, 500))
    status = "Returned" if rng.random() < 0.06 else "Sold"
    txn.append(("", rng.choice(store_ids), pd.Timestamp(d).normalize(),
                -qty if status == "Returned" else qty, round(amt, 2), status))

tx = pd.DataFrame(txn, columns=["Customer_id", "Store_id", "Transaction_date",
                                "Number_of_products_Sold", "Sale_Amount", "Transaction_Status"])
tx = tx.sort_values("Transaction_date", kind="stable").reset_index(drop=True)
tx.insert(0, "Transaction_Id", [f"TXN-{i+1:07d}" for i in range(len(tx))])
tx["Transaction_date"] = tx["Transaction_date"].dt.strftime("%Y-%m-%d")

engagement = pd.DataFrame(eng, columns=["Customer_Id", "Month", "Emails_sent",
                                        "Emails_opened", "App_logins"])

customers = pd.DataFrame({
    "Customer_Id": cust_ids,
    "Customer_name": [fake.name() for _ in range(n)],
    "Customer_address": [fake.address().replace("\n", ", ") for _ in range(n)],
    "Email": [fake.unique.email() for _ in range(n)],
    "Phone_number": [fake.numerify("(###) ###-####") for _ in range(n)],
    "Membership_tier": tier,
    "Signup_date": signup.strftime("%Y-%m-%d"),
    "Home_store_id": home_store,
    "Email_opt_in": opt_in,
})

# ---------------- answer key (computed from CLEAN data) ----------------
def label(year):
    t = tx[(tx.Customer_id != "") & (tx.Transaction_Status == "Sold")].copy()
    t["d"] = pd.to_datetime(t.Transaction_date)
    active = set(t[(t.d >= f"{year}-04-01") & (t.d <= f"{year}-09-30")].Customer_id)
    q4 = set(t[(t.d >= f"{year}-10-01") & (t.d <= f"{year}-12-31")].Customer_id)
    return pd.Series([(0 if c in q4 else 1) if c in active else np.nan for c in cust_ids])

key = pd.DataFrame({
    "Customer_Id": cust_ids,
    "Persona": persona,
    "True_churn_date": churn_date.strftime("%Y-%m-%d"),
    "Churn_label_Q4_2024": label(2024).astype("Int64"),
    "Churn_label_Q4_2025": label(2025).astype("Int64"),
})
key["True_churn_date"] = key["True_churn_date"].replace("NaT", "")

# ---------------- inject light data-quality errors ----------------
c = customers.copy()
idx = rng.choice(n, DQ["tier_case"], replace=False)
c.loc[idx, "Membership_tier"] = [rng.choice([t.lower(), t.upper(), f" {t} "]) for t in c.loc[idx, "Membership_tier"]]
idx = rng.choice(n, DQ["bad_email"], replace=False)
def break_email(e):
    u, d = e.split("@")
    return rng.choice([u + d, f"{u}@", f"{u}@@{d}", f" {e} ", f"{u}@{d.split('.')[0]}"])
c.loc[idx, "Email"] = [break_email(e) for e in c.loc[idx, "Email"]]
idx = rng.choice(n, DQ["null_phone"], replace=False)
c.loc[idx, "Phone_number"] = np.nan
c = pd.concat([c, c.sample(DQ["cust_dupes"], random_state=SEED)], ignore_index=True)

t = tx.copy()
for col in ["Transaction_date", "Number_of_products_Sold", "Sale_Amount"]:
    idx = rng.choice(len(t), int(len(t) * DQ["txn_null_frac"]), replace=False)
    t.loc[idx, col] = np.nan
t["Number_of_products_Sold"] = t["Number_of_products_Sold"].astype("Int64")
t = pd.concat([t, t.sample(frac=DQ["txn_dupe_frac"], random_state=SEED)], ignore_index=True)

e = pd.concat([engagement, engagement.sample(frac=DQ["eng_dupe_frac"], random_state=SEED)], ignore_index=True)

# ---------------- write ----------------
c.to_csv(f"{OUT}/customers.csv", index=False)
stores.to_csv(f"{OUT}/stores.csv", index=False)
t.to_csv(f"{OUT}/transactions.csv", index=False)
e.to_csv(f"{OUT}/engagement.csv", index=False)
key.to_csv(f"{OUT}/churn_answer_key.csv", index=False)
tx.to_csv(f"{OUT}/_clean_transactions.csv", index=False)       # for validation only
engagement.to_csv(f"{OUT}/_clean_engagement.csv", index=False)

print("customers", len(c), "| stores", len(stores), "| transactions", len(t), "| engagement", len(e))
print(key.Persona.value_counts().to_string())
for y in (2024, 2025):
    s = key[f"Churn_label_Q4_{y}"]
    print(f"Q4 {y}: active={s.notna().sum()}, churn rate={s.mean():.1%}")
