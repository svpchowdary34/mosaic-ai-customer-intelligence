# ==============================================================================
# MOSAIC AI — Customer Intelligence Studio | ONE Google Colab cell
# Every customer is a pattern. Find the ones that matter.
# ==============================================================================
import subprocess, sys, os, re, json, zipfile, html, hashlib, warnings
warnings.filterwarnings("ignore")

def _deps():
    try:
        from importlib.metadata import version
        if int(version("gradio").split(".")[0]) not in (4, 5): raise RuntimeError("gradio")
        for p in ("plotly", "scikit-learn", "scipy", "requests", "pandas", "numpy"): version(p)
    except Exception:
        print("Installing dependencies (about a minute)...")
        try: subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio>=4.44,<6", "plotly", "scikit-learn>=1.2", "scipy", "requests", "pandas", "numpy"]); print("Dependencies installed.")
        except Exception as e: print("Install warning:", type(e).__name__, "- check internet access and re-run.")
_deps()
try:
    import numpy as np, pandas as pd, plotly.graph_objects as go, gradio as gr, requests
    from plotly.subplots import make_subplots
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.mixture import GaussianMixture
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score, adjusted_rand_score
except Exception as e:
    raise SystemExit("Libraries could not be imported (" + type(e).__name__ + "). Restart the runtime and re-run this cell.")

C = dict(bg="#151A28", sf="#22293A", li="#B8F36B", la="#D6B9FF", co="#F2B6A0", tx="#F5F5F0", mu="#A9B1C5")
SEGC = [C["li"], C["la"], C["co"], "#8FD3FF", "#FFE08A", "#9AE6C4", "#FF9FB2", "#C7C7FF", "#E0E0E0", "#FFB347"]
G = {}; TESTS = []; PREV = {}; _SM = {}
BLANK = ["", "nan", "none", "n/a", "na", "-", "null"]
esc = lambda x: html.escape(str(x))
norm = lambda s: re.sub(r"\W+", "_", str(s).strip().lower()).strip("_")
nul = lambda v: v is None or (not isinstance(v, str) and bool(pd.isna(v)))
money = lambda v: "—" if nul(v) else ("-$" + format(abs(v), ",.0f") if v < 0 else "$" + format(v, ",.0f"))
pc0 = lambda v: "—" if nul(v) else format(v, ".0%")
pc1 = lambda v: "—" if nul(v) else format(v, ".1%")
fx = lambda v, d=2: "—" if nul(v) else format(v, "." + str(d) + "f")
fx0 = lambda v: fx(v, 0)
nf = lambda v: "—" if nul(v) else format(v, ",.0f")
x2 = lambda v: "—" if nul(v) else format(v, ".2f") + "x"
segcol = lambda s: SEGC[(int(str(s)[1:]) - 1) % len(SEGC)]
def mcol(v): return "—" if nul(v) else "<span style='color:%s'>%s</span>" % (C["li"] if v >= 0 else C["co"], money(v))
def note(t, k="info"):
    col = {"info": C["la"], "ok": C["li"], "warn": C["co"], "err": "#ff8a8a"}[k]
    return "<div class='note' style='border-left:3px solid %s'>%s</div>" % (col, t)
def card(lab, val, desc, col=C["li"]):
    return "<div class='kc'><div class='kl'>%s</div><div class='kv' style='color:%s'>%s</div><div class='kd'>%s</div></div>" % (lab, col, val, desc)
H = lambda t: "<div class='sec'>%s</div>" % t
def tbl(d, cols, fmt=None):
    fmt = fmt or {}; h = "<div class='tw'><table><tr>" + "".join("<th>%s</th>" % n for _, n in cols) + "</tr>"
    for _, r in d.iterrows():
        h += "<tr>" + "".join("<td>%s</td>" % (fmt[k](r[k]) if k in fmt else ("—" if nul(r[k]) else esc(r[k]))) for k, _n in cols) + "</tr>"
    return h + "</table></div>"
def sty(f, h=380, t=""):
    f.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color=C["tx"]), height=h, title=dict(text=t, x=.02),
                    margin=dict(l=50, r=20, t=55, b=50), legend=dict(orientation="h", y=-.25), colorway=SEGC)
    return f
def empty(msg):
    f = go.Figure(); f.add_annotation(text=msg, showarrow=False, font=dict(color=C["mu"], size=14)); f.update_xaxes(visible=False); f.update_yaxes(visible=False); return sty(f, 300)
NOP = lambda n: tuple(empty("No data") for _ in range(n))
LAB = dict(recency_days="recency (days)", frequency="order frequency", monetary="net spend", discount_rate="discount usage", category_diversity="category diversity", tenure_days="tenure",
           visit_rate_30d="site visits", email_opens_30d="email opens", email_ctr="email click-through", ad_clicks_30d="ad clicks", cart_abandon_rate="cart abandonment",
           conversion_rate="visit-to-order conversion", days_since_last_interaction="days since last interaction", return_rate="return rate")

# ------------------------------ FICTIONAL synthetic data ------------------------------
ARCH = [("loyal", .13, 1.1, 95, .08, .05, 4.0, .5, .25, .25, 0.0, 1.0, .8), ("promo", .17, .6, 48, .75, .09, 2.5, .55, .3, .3, 0.0, 1.0, 1.2), ("abandoner", .12, .12, 60, .25, .08, 5.0, .4, .2, .7, 0.0, 1.0, 1.5),
        ("browser", .14, .03, 40, .2, .06, 4.5, .3, .1, .35, 0.0, 1.0, 1.2), ("lapsed", .15, .7, 70, .3, .07, 1.0, .25, .08, .2, 0.0, .4, .4), ("quiet", .17, .45, 62, .1, .04, .7, .15, .05, .1, 0.0, 1.0, .15),
        ("new", .12, .5, 55, .35, .06, 2.5, .45, .2, .35, .65, 1.0, .8)]
AGES = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGEP = dict(promo=[.3, .35, .2, .1, .05], new=[.3, .35, .2, .1, .05], loyal=[.05, .15, .3, .3, .2], lapsed=[.1, .2, .25, .25, .2], abandoner=[.15, .3, .25, .2, .1], browser=[.2, .3, .25, .15, .1], quiet=[.1, .2, .25, .25, .2])
def make_synth(n=1800, seed=7):
    r = np.random.default_rng(seed); T0 = pd.Timestamp("2025-10-01"); D = 365.0
    CATS = ["Apparel", "Home", "Beauty", "Electronics", "Outdoor", "Grocery"]; ACQ = ["Organic Search", "Paid Social", "Email", "Referral", "Direct"]
    w = np.array([a[1] for a in ARCH]); w = w / w.sum(); tx, it, cu = [], [], []; tid = 0
    for i in range(n):
        cid = "C%05d" % (10000 + i); ai = int(r.choice(len(ARCH), p=w))
        if r.random() < .15: ai = int(r.integers(len(ARCH)))
        nm, _w, opm, aov, dp, rp, vpm, op, cp, cartp, s0, s1, adpm = ARCH[ai]
        lo, hi = s0 * D, s1 * D; mo = max((hi - lo) / 30.4, .2); pref = r.dirichlet(np.ones(len(CATS)) * .6); acq = str(r.choice(ACQ))
        days = np.sort(r.uniform(lo, hi, int(r.poisson(opm * r.lognormal(0, .35) * mo))))
        for dd in days:
            tid += 1; cat = CATS[int(r.choice(len(CATS), p=pref))]; q = int(r.integers(1, 4)); price = round(aov / 2 * r.lognormal(0, .25), 2); gross = round(q * price, 2)
            used = r.random() < dp; disc = round(gross * r.uniform(.1, .3), 2) if used else 0.0
            tx.append((cid, "T%06d" % tid, dd, "P-%s-%d" % (cat[:3].upper(), int(r.integers(1, 40))), cat, q, price, round(gross - disc, 2), disc, int(used), int(r.random() < rp), acq))
        if r.random() > .12:
            ilo, ihi = lo, min(D, hi + .15 * D); imo = max((ihi - ilo) / 30.4, .2); nv = int(r.poisson(vpm * r.lognormal(0, .3) * imo)); nop = int(r.poisson(vpm * op * imo))
            for kind, cnt in (("website_visit", nv), ("email_open", nop), ("email_click", int(r.binomial(nop, cp))), ("ad_click", int(r.poisson(adpm * imo))), ("cart_add", int(r.binomial(nv, min(.95, cartp * .5))))):
                if cnt <= 0: continue
                chn = ["email"] * cnt if kind.startswith("email") else (list(r.choice(["social", "search", "display"], cnt)) if kind == "ad_click" else list(r.choice(["direct", "search", "social", "email"], cnt)))
                cmp_ = ["CMP%02d" % x for x in r.integers(1, 13, cnt)] if kind in ("email_open", "email_click", "ad_click") else [None] * cnt
                dur = list(r.lognormal(5.2, .6, cnt).astype(int)) if kind == "website_visit" else [None] * cnt
                it.extend(zip([cid] * cnt, r.uniform(ilo, ihi, cnt), [kind] * cnt, chn, cmp_, list(r.choice(CATS, cnt, p=pref)), ["S%d_%d" % (i, s) for s in r.integers(0, max(nv, 1), cnt)], dur))
            for dd in days:
                if r.random() < .85: it.append((cid, dd + r.uniform(0, .02), "purchase_completed", "direct", None, None, "S%d_%d" % (i, int(r.integers(0, 99))), None))
        ap = np.array(AGEP[nm]); su = (T0 + pd.Timedelta(days=float(r.uniform(.55 * D, .65 * D) if nm == "new" else -r.uniform(30, 1500)))).strftime("%Y-%m-%d")
        cu.append((cid, str(r.choice(AGES, p=ap / ap.sum())), str(r.choice(["North", "South", "East", "West"])), su, str(r.choice(["Gold", "Silver", "Bronze"], p=[.5, .3, .2] if nm == "loyal" else [.1, .25, .65])), "opted_in" if r.random() < .86 else "opted_out"))
    TX = pd.DataFrame(tx, columns=["customer_id", "transaction_id", "d", "product_id", "product_category", "quantity", "unit_price", "transaction_value", "discount_amount", "discount_used", "return_flag", "acquisition_channel"])
    TX.insert(2, "transaction_date", (T0 + pd.to_timedelta(TX.pop("d"), unit="D")).dt.strftime("%Y-%m-%d"))
    IT = pd.DataFrame(it, columns=["customer_id", "d", "event_type", "interaction_channel", "campaign_id", "product_category", "session_id", "session_duration_seconds"])
    IT.insert(1, "event_timestamp", (T0 + pd.to_timedelta(IT.pop("d"), unit="D")).dt.strftime("%Y-%m-%d %H:%M:%S")); IT = IT.sort_values("event_timestamp").reset_index(drop=True)
    TX = pd.concat([TX, TX.sample(min(8, len(TX)), random_state=1)], ignore_index=True); TX.loc[TX.sample(frac=.01, random_state=2).index, "product_category"] = np.nan
    IT = pd.concat([IT, IT.sample(min(40, len(IT)), random_state=1)], ignore_index=True)
    return TX, IT, pd.DataFrame(cu, columns=["customer_id", "age_band", "region", "signup_date", "loyalty_status", "consent_status"])
SAMPLE_TX, SAMPLE_IT, SAMPLE_CU = make_synth()

# ------------------------------ ingestion & validation ------------------------------
CID = ["customerid", "cust_id", "user_id", "client_id", "contact_id", "customer"]
SYN_TX = {"customer_id": CID, "transaction_id": ["order_id", "txn_id", "invoice_id", "order_number"], "transaction_date": ["date", "order_date", "purchase_date", "invoice_date"], "product_id": ["sku", "item_id"],
          "product_category": ["category", "product_type", "department"], "quantity": ["qty", "units"], "unit_price": ["price"], "transaction_value": ["amount", "revenue", "order_value", "sales", "total", "line_total"],
          "discount_amount": ["discount", "discount_value"], "discount_used": ["promo_used", "coupon_used", "has_discount"], "return_flag": ["returned", "is_return", "refund_flag"],
          "acquisition_channel": ["acq_channel", "source", "channel"], "status": ["order_status", "transaction_status"]}
FL = dict(email_opened="open", email_clicked="click", ad_clicked="ad", website_visit="visit", cart_added="cart", purchase_completed="purchase"); KINDS = ["visit", "open", "click", "ad", "cart", "purchase"]
SYN_IT = dict({"customer_id": CID, "event_timestamp": ["timestamp", "event_time", "datetime", "date"], "event_type": ["event", "event_name", "action"], "interaction_channel": ["channel", "medium"], "campaign_id": ["campaign"],
               "product_category": ["category"], "session_id": ["session"], "session_duration_seconds": ["session_duration", "duration"]}, **{k: [] for k in FL})
SYN_CU = {"customer_id": CID, "age_band": ["age_group", "age"], "region": ["state", "country", "city"], "signup_date": ["join_date", "registration_date"], "loyalty_status": ["loyalty_tier", "tier"], "consent_status": ["consent", "opt_in"]}
CONSENT_OK = ["opted_in", "opt_in", "yes", "true", "1", "granted", "y", "subscribed"]
def clean_str(s):
    t = s.astype(str).str.strip(); return t.where(~(s.isna() | t.str.lower().isin(BLANK)), np.nan)
def to_num(s):
    if pd.api.types.is_numeric_dtype(s): return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,%₹€£\s]", "", regex=True), errors="coerce")
def to_bool(s):
    if s.dtype == bool: return s
    return s.astype(str).str.strip().str.lower().isin(["1", "1.0", "true", "yes", "y", "t", "returned", "return", "refund"])
def to_dt(s):
    try: return pd.to_datetime(s, errors="coerce", format="mixed")
    except Exception: return pd.to_datetime(s, errors="coerce")
def read_csv_any(file):
    path = file if isinstance(file, str) else getattr(file, "name", file)
    try: df = pd.read_csv(path)
    except Exception: df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    if df.empty: raise ValueError("The file contains no rows.")
    if len(df) > 1500000: raise ValueError("File too large (limit 1,500,000 rows).")
    return df
def parse_map(text):
    ov = {}
    for p in re.split(r"[,;\n]", text or ""):
        if "=" in p: k, v = p.split("=", 1); ov[norm(k)] = norm(v)
    return ov
def map_cols(df, syn, ov):
    df = df.copy(); df.columns = [norm(c) for c in df.columns]; ren = {}
    for k, al in syn.items():
        c = ov[k] if ov.get(k) in df.columns else next((a for a in [k] + al if a in df.columns and a not in ren), None)
        if c is None or c in ren: continue
        if c != k and k in df.columns: df = df.drop(columns=[k])
        ren[c] = k
    df = df.rename(columns=ren); return df.loc[:, ~df.columns.duplicated()]
def pseudo(s, salt):
    mp = {x: "C-" + hashlib.sha256((salt + str(x)).encode()).hexdigest()[:10] for x in s.dropna().unique()}; return s.map(mp)
def ev_kind(x):
    s = re.sub(r"[^a-z0-9]+", "_", str(x).lower()).strip("_")
    if "cart" in s: return "cart"
    if re.search(r"purchase|order|checkout_complete", s): return "purchase"
    if re.search(r"(^|_)ads?(_|$)", s): return "ad"
    if "open" in s: return "open"
    if "click" in s: return "click"
    if re.search(r"visit|view|session|page|browse", s): return "visit"
    return None

def clean_tx(raw, ov, W, dq):
    d = map_cols(raw, SYN_TX, ov)
    if not ({"customer_id", "transaction_date"} <= set(d.columns) and ("transaction_value" in d.columns or {"quantity", "unit_price"} <= set(d.columns))):
        raise ValueError("Transactions need customer_id, transaction_date and transaction_value (or quantity and unit_price). Columns found: " + ", ".join(map(str, raw.columns)) + ". Use the mapping box, e.g. customer_id=cust, transaction_value=amount.")
    dq["tx_in"] = len(d); d["cid"] = d["customer_id"].astype(str).str.strip(); d = d[~d["cid"].str.lower().isin(BLANK)]
    if "status" in d.columns:
        cx = d["status"].astype(str).str.lower().str.contains("cancel|void|fail"); dq["cancelled"] = int(cx.sum())
        if cx.any(): W.append("%d cancelled/void transactions excluded." % int(cx.sum())); d = d[~cx]
    sub = ["cid", "transaction_id", "product_id"] if {"transaction_id", "product_id"} <= set(d.columns) else None
    n_ = len(d); d = d.drop_duplicates(subset=sub); dq["dup_tx"] = n_ - len(d)
    if dq["dup_tx"]: W.append("%d duplicate transaction rows removed." % dq["dup_tx"])
    d["date"] = to_dt(d["transaction_date"]); nb = int(d["date"].isna().sum()); dq["bad_dates"] = nb
    if nb: W.append("%d transactions with unparseable dates removed." % nb); d = d[d["date"].notna()]
    if "transaction_value" in d.columns: val = to_num(d["transaction_value"])
    else: val = to_num(d["quantity"]) * to_num(d["unit_price"]) - (to_num(d["discount_amount"]).fillna(0) if "discount_amount" in d.columns else 0)
    d["value"] = val; nm_ = int(val.isna().sum()); dq["missing_value"] = nm_
    if nm_: W.append("%d transactions with a missing value removed (never treated as zero revenue)." % nm_); d = d[d["value"].notna()]
    if "transaction_id" in d.columns: d["tid"] = d["transaction_id"].astype(str).str.strip()
    else: d["tid"] = ["R%d" % i for i in range(len(d))]; W.append("No transaction_id: each row is treated as its own order.")
    d["cat"] = clean_str(d["product_category"]) if "product_category" in d.columns else np.nan
    d["disc_amt"] = to_num(d["discount_amount"]) if "discount_amount" in d.columns else np.nan
    if "discount_used" in d.columns: d["disc_used"] = to_bool(d["discount_used"]).astype(float)
    elif "discount_amount" in d.columns: d["disc_used"] = (d["disc_amt"] > 0).astype(float).where(d["disc_amt"].notna())
    else: d["disc_used"] = np.nan; W.append("No discount fields: discount dependence is unavailable.")
    if "return_flag" in d.columns: d["ret"] = to_bool(d["return_flag"])
    else: d["ret"] = False; W.append("No return_flag: only negative values are treated as refunds.")
    d["acq"] = clean_str(d["acquisition_channel"]) if "acquisition_channel" in d.columns else np.nan
    d["net"] = np.where(d["ret"] & (d["value"] > 0), 0.0, d["value"]); dq["returns"] = float((d["ret"] & (d["value"] > 0)).mean()) if len(d) else 0.0
    if d.empty: raise ValueError("No valid transactions remain after validation.")
    return d[["cid", "tid", "date", "cat", "value", "disc_amt", "disc_used", "ret", "acq", "net"]].reset_index(drop=True)

def clean_it(raw, ov, W, dq):
    d = map_cols(raw, SYN_IT, ov); fl = [c for c in FL if c in d.columns]
    if not ({"customer_id", "event_timestamp"} <= set(d.columns)) or not ("event_type" in d.columns or fl):
        raise ValueError("Interactions need customer_id, event_timestamp and event_type (or flag columns such as email_opened). Columns found: " + ", ".join(map(str, raw.columns)) + ".")
    dq["it_in"] = len(d); d["cid"] = d["customer_id"].astype(str).str.strip(); d = d[~d["cid"].str.lower().isin(BLANK)]
    d["ts"] = to_dt(d["event_timestamp"]); nb = int(d["ts"].isna().sum()); dq["bad_ts"] = nb
    if nb: W.append("%d interaction events with unparseable timestamps removed." % nb); d = d[d["ts"].notna()]
    kind = d["event_type"].map(ev_kind) if "event_type" in d.columns else pd.Series(None, index=d.index, dtype=object)
    for k in KINDS: d["k_" + k] = (kind == k)
    for col, k in FL.items():
        if col in d.columns: d["k_" + k] = d["k_" + k] | to_bool(d[col])
    anyk = d[["k_" + k for k in KINDS]].any(axis=1); nu = int((~anyk).sum())
    if nu: W.append("%d interaction events with unrecognised types were ignored." % nu); d = d[anyk]
    for k in KINDS: d["k_" + k] = d["k_" + k].astype(float)
    d["chan"] = clean_str(d["interaction_channel"]) if "interaction_channel" in d.columns else np.nan
    sub = [c for c in ["cid", "ts", "session_id", "event_type"] if c in d.columns] + ["k_" + k for k in KINDS]
    n_ = len(d); d = d.drop_duplicates(subset=sub); dq["dup_it"] = n_ - len(d)
    if dq["dup_it"]: W.append("%d duplicate interaction events removed." % dq["dup_it"])
    if d.empty: raise ValueError("No valid interaction events remain after validation.")
    return d[["cid", "ts", "chan"] + ["k_" + k for k in KINDS]].reset_index(drop=True)

def clean_cu(raw, ov, W):
    d = map_cols(raw, SYN_CU, ov)
    if "customer_id" not in d.columns: raise ValueError("The customer file needs a customer_id column.")
    d["cid"] = d["customer_id"].astype(str).str.strip(); d = d.drop_duplicates("cid")
    for k in ["age_band", "region", "loyalty_status", "consent_status"]: d[k] = clean_str(d[k]) if k in d.columns else np.nan
    return d[["cid", "age_band", "region", "loyalty_status", "consent_status"]].reset_index(drop=True)

# ------------------------------ feature engineering ------------------------------
CAND = ["recency_days", "frequency", "monetary", "discount_rate", "category_diversity", "tenure_days", "visit_rate_30d", "email_opens_30d", "email_ctr", "ad_clicks_30d", "cart_abandon_rate", "conversion_rate", "days_since_last_interaction", "return_rate"]
LOGF = {"frequency", "monetary", "visit_rate_30d", "email_opens_30d", "ad_clicks_30d"}
def build_features(TX, IT, CU, adate, act, W):
    n_ex = int((TX["date"] >= adate).sum()); TX = TX[TX["date"] < adate]
    if IT is not None: IT = IT[IT["ts"] < adate]
    if TX.empty: raise ValueError("No transactions before the analysis date.")
    first = TX["date"].min()
    if IT is not None and len(IT): first = min(first, IT["ts"].min())
    obs = max((adate - first).days, 1); months = obs / 30.4
    ids = pd.Index(sorted(set(TX["cid"]) | (set(IT["cid"]) if IT is not None else set())))
    F = pd.DataFrame(index=ids); S = TX[(TX["value"] > 0) & (~TX["ret"])]; S1 = S.drop_duplicates(["cid", "tid"]).sort_values(["cid", "date"]).copy()
    F["frequency"] = S.groupby("cid")["tid"].nunique().reindex(ids).fillna(0); F["monetary"] = TX.groupby("cid")["net"].sum().reindex(ids).fillna(0)
    F["gross"] = S.groupby("cid")["value"].sum().reindex(ids); lp = S.groupby("cid")["date"].max().reindex(ids)
    F["recency_days"] = (adate - lp).dt.days.astype(float).fillna(float(obs))
    F["discount_rate"] = S.groupby("cid")["disc_used"].mean().reindex(ids); F["discount_amt"] = S.groupby("cid")["disc_amt"].sum(min_count=1).reindex(ids)
    sr = TX[TX["value"] > 0].groupby("cid").size().reindex(ids); rr = TX[TX["ret"] & (TX["value"] > 0)].groupby("cid").size().reindex(ids).fillna(0); F["return_rate"] = rr / sr
    F["category_diversity"] = S.groupby("cid")["cat"].nunique().reindex(ids).where(F["frequency"] > 0) if S["cat"].notna().any() else np.nan
    S1["gap"] = S1.groupby("cid")["date"].diff().dt.days; F["purchase_interval"] = S1.groupby("cid")["gap"].mean().reindex(ids)
    F["acq_channel"] = TX.sort_values("date").dropna(subset=["acq"]).groupby("cid")["acq"].first().reindex(ids)
    f0 = TX.groupby("cid")["date"].min().reindex(ids)
    if IT is not None and len(IT): f0 = pd.concat([f0.rename("a"), IT.groupby("cid")["ts"].min().reindex(ids).rename("b")], axis=1).min(axis=1)
    F["tenure_days"] = (adate - f0).dt.days.astype(float)
    for c in ["visit_rate_30d", "email_opens_30d", "email_ctr", "ad_clicks_30d", "cart_abandon_rate", "conversion_rate", "days_since_last_interaction"]: F[c] = np.nan
    F["tracked"] = False
    if IT is not None and len(IT):
        K = ["k_" + k for k in KINDS]; gi = IT.groupby("cid"); a = gi[K].sum().reindex(ids); F["tracked"] = a["k_visit"].notna().values
        F["visit_rate_30d"] = a["k_visit"] / months; F["email_opens_30d"] = a["k_open"] / months; F["ad_clicks_30d"] = a["k_ad"] / months
        F["email_ctr"] = (a["k_click"] / a["k_open"].where(a["k_open"] > 0)).clip(upper=1); cart = a["k_cart"]
        F["cart_abandon_rate"] = (cart - np.minimum(cart, F["frequency"])) / cart.where(cart > 0); F["conversion_rate"] = np.minimum(F["frequency"], a["k_visit"]) / a["k_visit"].where(a["k_visit"] > 0)
        F["days_since_last_interaction"] = (adate - gi["ts"].max().reindex(ids)).dt.days.astype(float)
        if IT["chan"].notna().any():
            sh = IT.groupby(["cid", "chan"]).size().unstack(fill_value=0).reindex(ids); sh = sh.div(sh.sum(axis=1).replace(0, np.nan), axis=0)
            for c in sh.columns: F["ch_" + str(c)] = sh[c]
    for k in ["age_band", "region", "loyalty_status", "consent_status"]: F[k] = np.nan
    if CU is not None:
        c = CU.drop_duplicates("cid").set_index("cid").reindex(ids)
        for k in ["age_band", "region", "loyalty_status", "consent_status"]: F[k] = c[k].values
    if F["consent_status"].notna().any(): F["eligible"] = F["consent_status"].astype(str).str.lower().isin(CONSENT_OK).values
    else: F["eligible"] = True
    if n_ex: W.append("%d transactions on or after the analysis date were excluded." % n_ex)
    W.append("Customers with no interaction records are treated as UNTRACKED (interaction features missing and imputed for clustering), not as zero engagement. Customers with no purchase have recency capped at the %d-day observation window." % obs)
    return F

def select_feats(F):
    used, excl = [], []
    for c in CAND:
        if F[c].notna().sum() == 0: excl.append((c, "not available in the data"))
        elif F[c].notna().mean() < .6: excl.append((c, "%.0f%% missing" % (100 * (1 - F[c].notna().mean()))))
        elif F[c].nunique(dropna=True) < 2: excl.append((c, "constant"))
        else: used.append(c)
    return used, excl
def make_Z(F, used):
    X = F[used].astype(float).copy(); imp = {}
    for c in used:
        if c in LOGF: X[c] = np.log1p(X[c].clip(lower=0))
        X[c] = X[c].clip(X[c].quantile(.01), X[c].quantile(.99)); imp[c] = float(X[c].isna().mean()); X[c] = X[c].fillna(X[c].median())
    keep = [c for c in used if X[c].std() > 1e-9]; drop = [c for c in used if c not in keep]
    if len(keep) < 2: raise ValueError("Fewer than 2 informative behavioural features after cleaning: customers look identical on the available data.")
    return pd.DataFrame(StandardScaler().fit_transform(X[keep]), index=X.index, columns=keep), imp, keep, drop

# ------------------------------ segmentation engine ------------------------------
def fit_lab(a, Z, k, seed=42, ninit=10):
    if a == "K-Means": return KMeans(k, n_init=ninit, random_state=seed).fit(Z).labels_
    if a == "Agglomerative": return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Z)
    return GaussianMixture(k, covariance_type="diag", n_init=2, reg_covar=1e-4, random_state=seed).fit(Z).predict(Z)
def stability(a, Z, k, lab, reps=3):
    r = np.random.default_rng(0); out = []
    for _ in range(reps):
        idx = np.sort(r.choice(len(Z), int(.85 * len(Z)), replace=False))
        try: out.append(adjusted_rand_score(lab[idx], fit_lab(a, Z[idx], k, seed=int(r.integers(1000000)), ninit=3)))
        except Exception: pass
    return float(np.mean(out)) if out else np.nan
def evaluate(Z, algos, ks):
    n = len(Z); si = np.random.default_rng(1).choice(n, min(n, 2500), replace=False); rows, fails = [], []
    for a in algos:
        if a == "Agglomerative" and n > 6000: fails.append((a, "-", "skipped: ward linkage needs too much memory above 6,000 customers")); continue
        for k in ks:
            try:
                lab = fit_lab(a, Z, k); cnt = np.bincount(lab, minlength=k)
                if (cnt == 0).any() or len(np.unique(lab)) < 2: raise ValueError("degenerate clusters (empty or single)")
                ls = lab[si]; sil = silhouette_score(Z[si], ls) if len(np.unique(ls)) > 1 else 0.0; db = davies_bouldin_score(Z, lab); st = stability(a, Z, k, lab); ms = cnt.min() / n
                sc = .30 * np.clip(sil / .5, 0, 1) + .25 * np.clip(0 if np.isnan(st) else st, 0, 1) + .15 * np.clip(1 - (db - .5) / 2.5, 0, 1) + .15 * np.clip(ms / .05, 0, 1) + .15 * np.clip(1 - (k - 3) / 6, .3, 1)
                if cnt.min() < 10: sc *= .5
                rows.append(dict(algo=a, k=k, sil=sil, db=db, stab=st, minshare=ms, score=float(sc), lab=lab))
            except Exception as e: fails.append((a, k, str(e)[:70]))
    return rows, fails
def run_seg(Z, algos, kmin, kmax, kfix):
    n = len(Z)
    if n < 60: raise ValueError("Only %d customers: at least 60 are needed for meaningful behavioural segments." % n)
    ks = [int(kfix)] if kfix and int(kfix) >= 2 else list(range(max(2, int(kmin)), min(int(kmax), max(2, n // 20)) + 1))
    if not ks: raise ValueError("No valid cluster counts to test. Check the k range.")
    rows, fails = evaluate(Z.values, algos or ["K-Means"], ks)
    if not rows: raise ValueError("No clustering method produced a valid solution (" + "; ".join("%s k=%s: %s" % f for f in fails[:4]) + ").")
    return dict(rows=rows, fails=fails, best=max(rows, key=lambda r: r["score"]))
def order_labels(lab, F):
    d = pd.DataFrame({"m": F["monetary"].values, "f": F["frequency"].values, "l": lab}).groupby("l")[["m", "f"]].median().sort_values(["m", "f"], ascending=False)
    mp = {old: "S%d" % (i + 1) for i, old in enumerate(d.index)}; return np.array([mp[x] for x in lab])
def assign_info(Z, seg):
    ids = sorted(seg.unique(), key=lambda s: int(s[1:])); Zv = Z.values; cen = np.array([Zv[(seg == s).values].mean(0) for s in ids])
    D = np.linalg.norm(Zv[:, None, :] - cen[None, :, :], axis=2); own = np.array([ids.index(s) for s in seg]); ar = np.arange(len(D)); d1 = D[ar, own]; D2 = D.copy(); D2[ar, own] = np.inf
    return ids, (D2.min(1) - d1) / (D2.min(1) + d1 + 1e-9), np.array(ids)[D2.argmin(1)]
def reasons(Z, seg, Zc):
    out = pd.Series("", index=Z.index)
    for s in Zc.index:
        m = (seg == s).values; parts = []
        for f in Zc.loc[s].abs().sort_values(ascending=False).index[:3]:
            sg = 1 if Zc.loc[s, f] > 0 else -1
            parts.append(pd.Series(["%s average %s (z=%+.1f)" % ("above" if sg > 0 else "below", LAB.get(f, f), v) if np.sign(v) == sg else "" for v in Z.loc[m, f].values], index=Z.index[m]))
        j = pd.concat(parts, axis=1).apply(lambda r: "; ".join(x for x in r if x), axis=1); out[m] = j.replace("", "weak fit to the segment profile").values
    return out
def prof_row(f, act, tot):
    n = len(f); orders = f["frequency"].sum(); rev = f["monetary"].sum(); b = f["frequency"] > 0
    return dict(n=n, rev=rev, rev_share=rev / tot if tot > 0 else np.nan, avg_val=rev / n if n else np.nan, aov=f["gross"].sum() / orders if orders > 0 else np.nan, buy_share=b.mean(), repeat=(f["frequency"] >= 2).mean(),
                active=(b & (f["recency_days"] <= act)).mean(), buy90=(b & (f["recency_days"] <= 90)).mean(), recency=f.loc[b, "recency_days"].median() if b.any() else np.nan, freq=f["frequency"].mean(), interval=f["purchase_interval"].median(),
                discount=f["discount_rate"].mean(), visit=f["visit_rate_30d"].mean(), opens=f["email_opens_30d"].mean(), ctr=f["email_ctr"].mean(), ads=f["ad_clicks_30d"].mean(), cart=f["cart_abandon_rate"].mean(),
                conv=f["conversion_rate"].mean(), ret=f["return_rate"].mean(), div=f["category_diversity"].mean(), tenure=f["tenure_days"].mean(), tracked=f["tracked"].mean(), eligible=f["eligible"].mean(), disc_amt=f["discount_amt"].sum(min_count=1))
def build_profile(F, seg, Z, TX, act):
    ids = sorted(seg.unique(), key=lambda s: int(s[1:])); tot = F["monetary"].sum(); rows = []
    try:
        si = np.random.default_rng(3).choice(len(Z), min(len(Z), 3000), replace=False); sil = pd.Series(silhouette_samples(Z.values[si], seg.values[si])).groupby(seg.values[si]).mean()
    except Exception: sil = pd.Series(dtype=float)
    for s in ids:
        d = prof_row(F[seg == s], act, tot); d.update(id=s, share=len(F[seg == s]) / len(F), sil=sil.get(s, np.nan)); rows.append(d)
    PR = pd.DataFrame(rows).set_index("id", drop=False); PR.index.name = None; CATP = pd.DataFrame(); CHP = pd.DataFrame()
    if TX["cat"].notna().any():
        t = TX[(TX["net"] > 0) & TX["cat"].notna()].assign(seg=lambda x: x["cid"].map(seg)); pv = t.groupby(["seg", "cat"])["net"].sum().unstack(fill_value=0); CATP = pv.div(pv.sum(axis=1), axis=0).reindex(ids)
    chc = [c for c in F.columns if c.startswith("ch_")]
    if chc: CHP = F[chc].groupby(seg).mean().reindex(ids)
    OV = prof_row(F, act, tot); OV["share"] = 1.0; return PR, CATP, CHP, OV

# ------------------------------ interpretation (deterministic) ------------------------------
def name_seg(p, z):
    g = lambda c: float(z.get(c, 0))
    if p["buy_share"] < .35: return "Browsers with Low Purchase Conversion" if g("visit_rate_30d") > 0 else "Inactive Non-Buyers"
    if g("recency_days") > .6 or p["active"] < .25: return "Lapsed Former Regulars" if p["freq"] >= 2 else "Lapsed One-Time Buyers"
    if g("monetary") > .6 and g("frequency") > .3: return "High-Value Repeat Buyers"
    if g("discount_rate") > .6: return "Promotion-Responsive Shoppers"
    if g("cart_abandon_rate") > .5 and g("visit_rate_30d") > 0: return "High-Intent Cart Abandoners"
    if g("visit_rate_30d") < -.3 and g("email_opens_30d") < -.3 and g("frequency") > -.2: return "Quiet but Consistent Customers"
    if g("tenure_days") < -.5: return "Newer Developing Customers"
    if g("monetary") < -.4: return "Light Occasional Buyers"
    return "Core Mid-Value Shoppers"
def hyps_of(p, z):
    g = lambda c: float(z.get(c, 0)); h = []
    if g("discount_rate") > .6: h.append("May be sensitive to price or promotions; test offers against a no-offer control.")
    if g("cart_abandon_rate") > .5: h.append("May be meeting friction or comparing options at checkout; investigate the checkout path.")
    if g("recency_days") > .6 and p["buy_share"] >= .35: h.append("May have lapsed or switched provider; reactivation tests can show whether interest remains.")
    if g("visit_rate_30d") > .5 and g("conversion_rate") < 0: h.append("May be browsing without clear purchase intent, or facing price or product-fit barriers.")
    if g("monetary") > .6: h.append("May value recognition and convenience over price; test non-discount perks.")
    if g("email_ctr") > .6: h.append("May respond well to email content; test personalised email.")
    if g("return_rate") > .6: h.append("May have fit or expectation issues with products; review return reasons.")
    h.append("Customer motivations are not observed in this data; every statement above is a hypothesis to test."); return h
def narratives(PR, Zc, act):
    seen, NARR = {}, {}; avgv = PR["rev"].sum() / PR["n"].sum()
    for s in PR.index:
        p = PR.loc[s]; z = Zc.loc[s]; nm = name_seg(p, z); seen[nm] = seen.get(nm, 0) + 1; nm = nm if seen[nm] == 1 else nm + " " + "I" * seen[nm]
        top = z.reindex(z.abs().sort_values(ascending=False).index)[:4]; diffs = ["%s %s (%+.1f sd)" % ("higher" if v > 0 else "lower", LAB.get(c, c), v) for c, v in top.items()]
        ev = ["%s customers (%s of the base)" % (nf(p["n"]), pc1(p["share"])), "%s of observed net revenue; net value per customer %s (%s the average customer)" % (pc1(p["rev_share"]), money(p["avg_val"]), x2(p["avg_val"] / avgv if avgv > 0 else np.nan)),
              "%s have purchased; %s repeat-purchase rate; %s active in the last %d days" % (pc0(p["buy_share"]), pc0(p["repeat"]), pc0(p["active"]), act)]
        for lab, key, f_ in [("discount used on %s of orders", "discount", pc0), ("average email click-through %s", "ctr", pc0), ("cart abandonment %s", "cart", pc0), ("visit-to-order conversion %s", "conv", pc0), ("return rate %s", "ret", pc1)]:
            if pd.notna(p[key]): ev.append(lab % f_(p[key]))
        unc = []
        if p["tracked"] < .8: unc.append("Interaction tracking covers only %s of this segment; engagement metrics rely on imputed values for the rest." % pc0(p["tracked"]))
        if p["n"] < 100: unc.append("Small segment (%s customers): estimates are noisy." % nf(p["n"]))
        if pd.notna(p["sil"]) and p["sil"] < .1: unc.append("Weak separation from neighbouring segments (silhouette %s)." % fx(p["sil"]))
        unc.append("Segments describe behaviour in the observed window, not fixed traits; they may shift after retraining.")
        sc = (1 if p["n"] >= 100 else 0) + (1 if float(z.abs().max()) >= 1 else 0) + (1 if pd.notna(p["sil"]) and p["sil"] > .15 else 0)
        NARR[s] = dict(name=nm, summary="%s customers (%s) producing %s of net revenue; distinguished by %s." % (nf(p["n"]), pc1(p["share"]), pc1(p["rev_share"]), ", ".join(diffs[:2])), profile=diffs, evidence=ev,
                       hyps=hyps_of(p, z), unc=unc, strength=["Low", "Low", "Moderate", "High"][sc], src="deterministic (rule-based)")
    return NARR

def n_per_arm(p0, rel=.2):
    p0 = float(np.clip(p0, .02, .9)); p1 = min(p0 * (1 + rel), .99); return int(np.ceil((1.96 + .84) ** 2 * (p0 * (1 - p0) + p1 * (1 - p1)) / (p1 - p0) ** 2))
PLAY_COLS = ["Segment", "Name", "Objective", "Campaign concept", "Message angle", "Sample copy", "Channels", "Offer strategy", "No-discount alternative", "Timing / trigger", "Primary KPI", "Guardrail KPI", "Test design", "Risks & assumptions"]
def playbooks(PR, Zc, NARR, F, CATP, CHP):
    rows, evs = [], {}
    for s in PR.index:
        p = PR.loc[s]; z = Zc.loc[s]; g = lambda c: float(z.get(c, 0)); nm = NARR[s]["name"]
        lapsed = (g("recency_days") > .6 or p["active"] < .25) and p["buy_share"] >= .35; hv = g("monetary") > .6 and g("frequency") > .3
        if p["buy_share"] < .35: obj, con, ang, off, alt, kp, gk = "Conversion (first purchase)", "Guided first-purchase journey built on browsed categories", "Help me choose: reviews, fit and comparison", "Free shipping or a sample before any price cut; discount only after a test", "Buying guide, reviews and size/fit help", "First-purchase conversion rate", "Margin per order and unsubscribe rate"
        elif lapsed: obj, con, ang, off, alt, kp, gk = "Reactivation", "Win-back series timed to each customer's usual purchase interval", "We saved what you liked; here is what is new", "Time-limited incentive only for customers past 1.5x their usual interval, tested against a no-offer control", "New-arrival or replenishment reminder with no incentive", "Reactivation rate (order within 30 days)", "Contribution margin per recipient and opt-out rate"
        elif hv: obj, con, ang, off, alt, kp, gk = "Retention and loyalty", "Early access and loyalty benefits with complementary-product recommendations", "Recognition and exclusivity", "Non-monetary perks first (early access, free returns); no blanket discount", "Early access and personalised recommendations", "Repeat-purchase rate and net spend per customer", "Discount cost per order"
        elif g("discount_rate") > .6: obj, con, ang, off, alt, kp, gk = "Margin-aware repeat purchase", "Threshold-based offer test (for example free shipping above a basket size) against no offer", "Value for money", "Threshold incentive with a margin floor and frequency cap", "Bundle suggestions and loyalty points with no price cut", "Contribution margin per recipient", "Discount rate and cannibalisation (orders from customers who would buy anyway)"
        elif g("cart_abandon_rate") > .5: obj, con, ang, off, alt, kp, gk = "Conversion (recover intent)", "Contextual cart reminders plus a checkout-friction audit", "Your items are waiting; easy checkout", "Reminder first; incentive only on a second reminder inside a test", "Stock or delivery information and simpler checkout", "Cart-to-order recovery rate", "Margin per order and complaint rate"
        else: obj, con, ang, off, alt, kp, gk = "Repeat purchase and growth", "Relevance-led recommendations from category preference", "Relevant picks, low frequency", "No discount by default", "Replenishment or recommendation messages", "Repeat-purchase rate", "Unsubscribe rate"
        tc = CATP.loc[s].sort_values(ascending=False).index[0] if len(CATP) and s in CATP.index and CATP.loc[s].notna().any() else None
        copy = "Hi, new in %s picked with you in mind. Take a look when you are ready." % tc if tc else "Hi, here is a short selection picked with you in mind. Take a look when you are ready."
        ch = []
        if pd.notna(p["ctr"]) and p["ctr"] >= .15: ch.append("Email (observed click-through %s)" % pc0(p["ctr"]))
        if len(CHP) and s in CHP.index and CHP.loc[s].notna().any():
            t = CHP.loc[s].dropna().sort_values(ascending=False); ch.append("Top tracked channel: %s (%s of interactions)" % (t.index[0][3:], pc0(t.iloc[0])))
        if not ch: ch.append("Email or owned channels (engagement by channel is not tracked for this segment)")
        tm = ("About %.0f days after the last order (1.5x the segment's median purchase interval of %.0f days)" % (1.5 * p["interval"], p["interval"])) if lapsed and pd.notna(p["interval"]) else ("Within 1-4 hours and again at 24 hours after a cart addition" if obj.startswith("Conversion (recover") else ("After 60-90 days without an order (assumption: purchase interval not measurable)" if lapsed else "Between expected orders; cap at one message per week"))
        base = max(p["buy90"] if pd.notna(p["buy90"]) else .05, .02); npa = n_per_arm(base); el = int(F[F["seg"] == s]["eligible"].sum())
        extra = (" This segment has only %s eligible customers, so it may be too small to detect that lift." % nf(el)) if el < 2 * npa else ""
        test = "Randomise eligible customers into treatment and a 10-20%% holdout for 2-4 weeks; read out %s. About %s customers per arm are needed to detect a 20%% relative lift on a %s baseline (proxy: share ordering in the last 90 days; alpha 0.05, power 0.8).%s" % (kp, nf(npa), pc0(base), extra)
        risks = ["Segment is a statistical pattern, not a verified customer type", "Assumes consent for the chosen channel; use opted-in customers only", "Recommendations are hypotheses until validated by an experiment"]
        if p["tracked"] < .8: risks.append("Interaction tracking covers only %s of the segment" % pc0(p["tracked"]))
        if p["n"] < 100: risks.append("Small segment")
        rows.append([s, nm, obj, con, ang, copy, "; ".join(ch) + " · consented customers only", off, alt, tm, kp, gk, test, "; ".join(risks)]); evs[s] = NARR[s]["evidence"]
    return pd.DataFrame(rows, columns=PLAY_COLS), evs

# ------------------------------ core pipeline ------------------------------
def core(txr, itr, cur, ov, adate_txt, act, algos, kmin, kmax, kfix, pseud=False, salt="s"):
    W, dq = [], {}; TX = clean_tx(txr, ov, W, dq); IT = clean_it(itr, ov, W, dq) if itr is not None else None; CU = clean_cu(cur, ov, W) if cur is not None else None
    if pseud:
        for t in (TX, IT, CU):
            if t is not None: t["cid"] = pseudo(t["cid"], salt)
    last = TX["date"].max()
    if IT is not None: last = max(last, IT["ts"].max())
    ad = pd.to_datetime(adate_txt, errors="coerce") if (adate_txt or "").strip() else pd.NaT
    if (adate_txt or "").strip() and pd.isna(ad): W.append("Analysis date not understood; the day after the last record is used.")
    adate = ad if pd.notna(ad) else last.normalize() + pd.Timedelta(days=1)
    F = build_features(TX, IT, CU, adate, act, W); used, excl = select_feats(F); Z, imp, used, dropped = make_Z(F, used); excl += [(c, "constant after outlier clipping") for c in dropped]
    M = run_seg(Z, algos, kmin, kmax, kfix); best = M["best"]; seg = pd.Series(order_labels(best["lab"], F), index=F.index)
    ids, margin, second = assign_info(Z, seg); Zc = Z.groupby(seg).mean().loc[ids]; F["seg"] = seg; F["margin"] = margin; F["second"] = second; F["reasons"] = reasons(Z, seg, Zc)
    PR, CATP, CHP, OV = build_profile(F, seg, Z, TX, act); NARR = narratives(PR, Zc, act); PLAY, PLAYEV = playbooks(PR, Zc, NARR, F, CATP, CHP); PR["name"] = [NARR[s]["name"] for s in PR.index]
    return dict(W=W, dq=dq, TX=TX, IT=IT, CU=CU, F=F, Z=Z, used=used, excl=excl, imp=imp, M=M, seg=seg, Zc=Zc, PR=PR, OV=OV, CATP=CATP, CHP=CHP, NARR=NARR, PLAY=PLAY, PLAYEV=PLAYEV, adate=adate, act=act)

def compare_prev(prev, Zc, PR):
    try:
        if not prev or list(prev["Zc"].columns) != list(Zc.columns): return None
        A, B = prev["Zc"].values, Zc.values; D = np.linalg.norm(B[:, None, :] - A[None, :, :], axis=2); r_, c_ = linear_sum_assignment(D)
        return pd.DataFrame([dict(new=Zc.index[i], old=prev["Zc"].index[j], dist=D[i, j], new_n=int(PR.loc[Zc.index[i], "n"]), old_n=prev["n"].get(prev["Zc"].index[j])) for i, j in zip(r_, c_)])
    except Exception: return None
def run_all(txf, itf, cuf, mapping, adate_txt, act, algos, kmin, kmax, kfix, pseud_on):
    prev = dict(PREV); G.clear()
    try:
        demo = txf is None
        if demo: txr, itr, cur = SAMPLE_TX.copy(), SAMPLE_IT.copy(), SAMPLE_CU.copy()
        else: txr = read_csv_any(txf); itr = read_csv_any(itf) if itf else None; cur = read_csv_any(cuf) if cuf else None
        salt = os.urandom(6).hex(); do_p = bool(pseud_on) and not demo
        R = core(txr, itr, cur, parse_map(mapping), adate_txt, int(act or 90), list(algos or ["K-Means"]), kmin or 3, kmax or 7, kfix or 0, do_p, salt)
        G.update(R); G.update(demo=demo, pseud=do_p, salt=salt, raw_n=len(txr), cnt=R["F"].groupby("seg").size().to_dict(), elig=R["F"].groupby("seg")["eligible"].sum().to_dict(), prev_cmp=compare_prev(prev, R["Zc"], R["PR"]))
        PREV.clear(); PREV.update(Zc=R["Zc"], n=R["PR"]["n"].to_dict()); b = R["M"]["best"]
        return note(("Fictional demo data analysed. " if demo else "Your data analysed. ") + "%s customers · %d segments · model: %s (k=%d, silhouette %s)." % (nf(len(R["F"])), len(R["PR"]), b["algo"], b["k"], fx(b["sil"])), "ok")
    except Exception as e:
        G.clear(); return note("Could not complete the analysis: " + esc(e), "err")

# ------------------------------ dashboard ------------------------------
def refresh():
    if "F" not in G: return tuple(gr.update() for _ in range(7))
    F, TX, PR = G["F"], G["TX"], G["PR"]; ch = [(s + " · " + G["NARR"][s]["name"], s) for s in PR.index]; ids = F.sort_values("monetary", ascending=False).index[:300].tolist()
    return (gr.update(choices=ch, value=[]), gr.update(choices=sorted(TX["cat"].dropna().unique().tolist()), value=[]), gr.update(choices=sorted(F["acq_channel"].dropna().unique().tolist()), value=[]),
            gr.update(choices=sorted(G["IT"]["chan"].dropna().unique().tolist()) if G["IT"] is not None else [], value=[]), gr.update(choices=ch, value=ch[0][1]), gr.update(choices=ch, value=[c[1] for c in ch[:2]]), gr.update(choices=ids, value=ids[0] if ids else None))

def render_dash(segs, d1, d2, cats, achs, ichs):
    if "F" not in G: return (note("Load data first.", "info"),) + NOP(6)
    F, TX, IT, PR, Zc = G["F"], G["TX"], G["IT"], G["PR"], G["Zc"]; act = G["act"]; seg = F["seg"]; cm = pd.Series(True, index=F.index)
    if segs: cm &= seg.isin(segs)
    if achs: cm &= F["acq_channel"].isin(achs)
    Fs = F[cm]
    if Fs.empty: return (note("No customers match the filters.", "warn"),) + NOP(6)
    tm = TX["cid"].isin(Fs.index); D1 = pd.to_datetime(d1, errors="coerce") if d1 else pd.NaT; D2 = pd.to_datetime(d2, errors="coerce") if d2 else pd.NaT
    if pd.notna(D1): tm &= TX["date"] >= D1
    if pd.notna(D2): tm &= TX["date"] <= D2
    if cats: tm &= TX["cat"].isin(cats)
    T = TX[tm]; S = T[(T["value"] > 0) & (~T["ret"])]; ords = S.groupby("cid")["tid"].nunique(); rev = T["net"].sum(); n = len(Fs); rep = (ords >= 2).sum() / len(ords) if len(ords) else np.nan
    am = (Fs["frequency"] > 0) & (Fs["recency_days"] <= act); an = int(am.sum())
    h = ("<div class='kg'>" + card("Customers analysed", nf(n), "after filters") + card("Segments discovered", str(Fs["seg"].nunique()), "behavioural, from clustering", C["la"]) + card("Observed net revenue", money(rev), "after returns, filtered transactions") +
         card("Average customer value", money(rev / n), "net revenue per customer") + card("Repeat-purchase rate", pc0(rep), "buyers with 2+ orders (filtered)") + card("Active customers", pc0(an / n), "ordered in the last %d days (full period)" % act, C["la"]) + "</div>")
    if G["demo"]: h = note("<b>Illustrative synthetic data.</b> Fictional customers for demonstration; not real results.", "warn") + h
    if cats or pd.notna(D1) or pd.notna(D2): h += note("Revenue and repeat metrics use the filtered transactions. Segments and behavioural features are computed on the full observation period.", "info")
    st = PR.copy(); st["revs"] = T.assign(s=T["cid"].map(seg)).groupby("s")["net"].sum().reindex(st.index).fillna(0); st["cn"] = Fs.groupby("seg").size().reindex(st.index).fillna(0)
    h += H("Segments") + tbl(st, [("id", "ID"), ("name", "Segment name (descriptive label)"), ("cn", "Customers"), ("revs", "Net revenue"), ("avg_val", "Net value / customer"), ("buy_share", "Purchased"), ("repeat", "Repeat rate")], dict(cn=nf, revs=money, avg_val=money, buy_share=pc0, repeat=pc0))
    ids = list(PR.index); cols = [segcol(s) for s in ids]
    f1 = sty(go.Figure(go.Bar(x=ids, y=st["cn"], marker_color=cols, text=[pc0(v / n) for v in st["cn"]], textposition="outside", customdata=st["name"], hovertemplate="%{x} · %{customdata}<br>%{y} customers<extra></extra>")), 340, "Segment size")
    rs = st["revs"] / st["revs"].sum() if st["revs"].sum() > 0 else st["revs"]; f2 = go.Figure([go.Bar(x=ids, y=rs, marker_color=cols, name="Share of net revenue"), go.Scatter(x=ids, y=st["cn"] / n, mode="markers", marker=dict(size=11, color=C["tx"], symbol="diamond"), name="Share of customers")])
    f2 = sty(f2, 340, "Revenue contribution vs customer share"); f2.update_yaxes(tickformat=".0%")
    bf = Fs[Fs["frequency"] > 0]; samp = bf.sample(min(1500, len(bf)), random_state=1) if len(bf) else bf; f3 = go.Figure()
    for s in ids:
        d = samp[samp["seg"] == s]
        if len(d): f3.add_trace(go.Scattergl(x=d["recency_days"], y=d["frequency"] + np.random.default_rng(1).uniform(-.15, .15, len(d)), mode="markers", name=s, marker=dict(size=5, color=segcol(s), opacity=.6)))
    f3 = sty(f3, 380, "Recency vs purchase frequency (buyers; frequency jittered)"); f3.update_xaxes(title="Days since last purchase"); f3.update_yaxes(title="Orders")
    f4 = empty("No interaction data")
    if IT is not None:
        v = IT[IT["cid"].isin(Fs.index)]
        if ichs: v = v[v["chan"].isin(ichs)]
        vc = v.groupby("cid")["k_visit"].sum().reindex(Fs.index).fillna(0); mo = max((G["adate"] - TX["date"].min()).days, 1) / 30.4; od = ords.reindex(Fs.index).fillna(0); tr = Fs["tracked"]
        g = pd.DataFrame({"v": vc[tr], "o": od[tr], "s": Fs.loc[tr, "seg"]}).groupby("s").agg(v=("v", "sum"), o=("o", "sum"), n=("v", "size")).reindex(ids).dropna(); g["rate"] = g["v"] / g["n"] / mo; g["conv"] = np.minimum(g["o"], g["v"]) / g["v"].replace(0, np.nan)
        if len(g):
            f4 = go.Figure(go.Scatter(x=g["rate"], y=g["conv"], mode="markers+text", text=g.index, textposition="top center", marker=dict(size=14 + 40 * g["n"] / g["n"].max(), color=[segcol(s) for s in g.index], opacity=.85), hovertemplate="%{text}<br>%{x:.1f} visits / 30d<br>conversion %{y:.0%}<extra></extra>"))
            f4 = sty(f4, 380, "Engagement vs conversion (tracked customers)"); f4.update_xaxes(title="Site visits per customer per 30 days"); f4.update_yaxes(title="Orders / visits", tickformat=".0%")
    f5 = go.Figure(go.Heatmap(z=Zc.values.T, x=list(Zc.index), y=[LAB.get(c, c) for c in Zc.columns], colorscale=[[0, C["co"]], [.5, C["bg"]], [1, C["li"]]], zmid=0, colorbar=dict(title="z"), hovertemplate="%{x}<br>%{y}: %{z:+.2f} sd<extra></extra>")); f5 = sty(f5, 460, "Behavioural feature comparison across segments (standardised)")
    a = Fs.assign(a=am).groupby("seg")["a"].agg(["sum", "count"]).reindex(ids).dropna(); f6 = go.Figure([go.Bar(x=a.index, y=a["sum"], name="Active", marker_color=C["li"]), go.Bar(x=a.index, y=a["count"] - a["sum"], name="Inactive", marker_color=C["co"])]); f6.update_layout(barmode="stack"); f6 = sty(f6, 340, "Active vs inactive customers (%d-day definition)" % act)
    return h, f1, f2, f3, f4, f5, f6

# ------------------------------ segment explorer ------------------------------
METRICS = [("n", "Customers", nf), ("share", "Share of customers", pc1), ("rev_share", "Net revenue share", pc1), ("avg_val", "Net value per customer", money), ("aov", "Average order value", money), ("buy_share", "Customers who purchased", pc0),
           ("repeat", "Repeat-purchase rate", pc0), ("active", "Active in window", pc0), ("recency", "Median recency of buyers (days)", fx0), ("freq", "Average orders", fx), ("interval", "Median purchase interval (days)", fx0),
           ("discount", "Discount usage", pc0), ("visit", "Site visits / 30 days", fx), ("ctr", "Email click-through", pc0), ("cart", "Cart abandonment", pc0), ("conv", "Visit-to-order conversion", pc0), ("ret", "Return rate", pc1), ("tracked", "Interaction tracking coverage", pc0)]
def render_explorer(sid):
    if "F" not in G or not sid or sid not in G["PR"].index: return (note("Load data and pick a segment.", "info"),) + NOP(3)
    F, PR, Zc, N, TX = G["F"], G["PR"], G["Zc"], G["NARR"][sid], G["TX"]; p = PR.loc[sid]; ov = G["OV"]; z = Zc.loc[sid]
    h = "<div class='kg'>" + card("Segment", sid + " · " + esc(N["name"]), "descriptive label, derived after reviewing the statistics", segcol(sid)) + card("Customers", nf(p["n"]), pc1(p["share"]) + " of the base") + card("Net revenue", money(p["rev"]), pc1(p["rev_share"]) + " of the total") + \
        card("Average order value", money(p["aov"]), "net value per customer %s" % money(p["avg_val"])) + card("Repeat-purchase rate", pc0(p["repeat"]), "all customers: %s" % pc0(ov["repeat"]), C["la"]) + card("Evidence strength", N["strength"], "size, distinctness and separation", C["la"]) + "</div>"
    h += "<div class='ai'><b>AI-style interpretation (%s)</b><br>%s<br><br><b>Behavioural profile:</b> %s<br><b>Evidence:</b><ul>%s</ul><b>Hypotheses (not facts):</b><ul>%s</ul><b>Uncertainties:</b><ul>%s</ul></div>" % (esc(N["src"]), esc(N["summary"]), esc("; ".join(N["profile"])), "".join("<li>%s</li>" % esc(x) for x in N["evidence"]), "".join("<li>%s</li>" % esc(x) for x in N["hyps"]), "".join("<li>%s</li>" % esc(x) for x in N["unc"]))
    rows = pd.DataFrame([dict(m=lab, a=f(p[k]), b=f(ov[k])) for k, lab, f in METRICS]); h += H("Segment vs all customers") + tbl(rows, [("m", "Metric"), ("a", sid), ("b", "All customers")], dict(a=str, b=str))
    oth = Zc.drop(sid); nn = (((oth - z) ** 2).sum(axis=1) ** .5).idxmin() if len(oth) else None
    if nn:
        dd = z - Zc.loc[nn]; df_ = dd.reindex(dd.abs().sort_values(ascending=False).index)[:5]
        h += H("Closest other segment: %s · %s" % (nn, esc(G["NARR"][nn]["name"]))) + tbl(pd.DataFrame(dict(f=[LAB.get(c, c) for c in df_.index], d=df_.values)), [("f", "Feature"), ("d", "Difference (sd, %s minus %s)" % (sid, nn))], dict(d=lambda v: "%+.2f" % v))
    zs = z.reindex(z.abs().sort_values(ascending=False).index)[:10].iloc[::-1]; f1 = sty(go.Figure(go.Bar(y=[LAB.get(c, c) for c in zs.index], x=zs.values, orientation="h", marker_color=[C["li"] if v > 0 else C["co"] for v in zs.values])), 380, "Distinguishing features vs all customers (sd)")
    f2 = make_subplots(rows=1, cols=3, subplot_titles=["Recency (days)", "Orders", "Net spend"]); m = (F["seg"] == sid).values
    for j, (c, cl) in enumerate([("recency_days", None), ("frequency", None), ("monetary", 99)], 1):
        a_, b_ = F.loc[m, c].values, F[c].values
        if cl: cap = np.nanpercentile(b_, cl); a_, b_ = np.clip(a_, 0, cap), np.clip(b_, 0, cap)
        f2.add_trace(go.Histogram(x=b_, histnorm="probability", nbinsx=20, marker_color=C["mu"], opacity=.55, name="All", showlegend=j == 1), 1, j); f2.add_trace(go.Histogram(x=a_, histnorm="probability", nbinsx=20, marker_color=segcol(sid), opacity=.75, name=sid, showlegend=j == 1), 1, j)
    f2.update_layout(barmode="overlay"); f2 = sty(f2, 340, "Distributions: segment vs all customers")
    f3 = empty("No product-category data")
    if len(G["CATP"]):
        t = TX[(TX["net"] > 0) & TX["cat"].notna()].groupby("cat")["net"].sum(); t = t / t.sum(); cp = G["CATP"].loc[sid].reindex(t.index).fillna(0)
        f3 = go.Figure([go.Bar(x=t.index, y=t.values, name="All customers", marker_color=C["mu"]), go.Bar(x=t.index, y=cp.values, name=sid, marker_color=segcol(sid))]); f3 = sty(f3, 340, "Share of net spend by product category"); f3.update_layout(barmode="group"); f3.update_yaxes(tickformat=".0%")
    return h, f1, f2, f3
def render_compare(sids):
    if "F" not in G or not sids or len(sids) < 2: return note("Select two or more segments to compare.", "info"), empty("Select segments")
    PR, Zc = G["PR"], G["Zc"]; d = pd.DataFrame({"m": [lab for _, lab, _ in METRICS]})
    for s in sids: d[s] = [f(PR.loc[s, k]) for k, _, f in METRICS]
    h = tbl(d, [("m", "Metric")] + [(s, s + " · " + G["NARR"][s]["name"]) for s in sids], {s: str for s in sids}); top = Zc.loc[sids].var().sort_values(ascending=False).index[:8]
    fig = go.Figure([go.Bar(x=[LAB.get(c, c) for c in top], y=Zc.loc[s, top].values, name=s, marker_color=segcol(s)) for s in sids]); fig.update_layout(barmode="group"); return h, sty(fig, 380, "Features that differ most among the selected segments (sd)")
EXQ = ["Why was this segment identified?", "What makes it different from the other segments?", "Which behaviours have the strongest differences?", "What data would help us understand it better?"]
def explain_seg(sid, q):
    if "F" not in G or sid not in G["PR"].index: return "Load data and select a segment."
    N, p, z = G["NARR"][sid], G["PR"].loc[sid], G["Zc"].loc[sid]; top = z.reindex(z.abs().sort_values(ascending=False).index)[:5]; ts = "; ".join("%s %+.1f sd" % (LAB.get(c, c), v) for c, v in top.items()); b = G["M"]["best"]
    if q == EXQ[0]: return "**%s** was found by %s clustering (k=%d) on standardised behavioural features. Its customers sit close together on: %s. Evidence strength: %s. This is a statistical grouping, not an observed customer type." % (N["name"], b["algo"], b["k"], ts, N["strength"])
    if q == EXQ[1]: return "Versus the other segments, %s differs most in: %s. Net value per customer is %s (all customers: %s)." % (sid, ts, money(p["avg_val"]), money(G["OV"]["avg_val"]))
    if q == EXQ[2]: return "Strongest differences (standard deviations from the all-customer average): %s." % ts
    miss = [lab for lab, k in [("consent status", "eligible"), ("interaction tracking", "tracked")] if pd.notna(p[k]) and p[k] < .9]
    return "More data would sharpen this segment: campaign-response history (for lift), margin by product, reasons for returns and cancellations, survey or support data on motivations%s. Today, motivations are hypotheses only." % ("; also better coverage of " + " and ".join(miss) if miss else "")

# ------------------------------ LLM (optional) ------------------------------
def get_key(key, provider, env=True):
    k = (key or "").strip()
    if k or not env: return k or None
    nm = "ANTHROPIC_API_KEY" if provider == "Anthropic" else "OPENAI_API_KEY"; k = os.environ.get(nm)
    if not k:
        try:
            from google.colab import userdata; k = userdata.get(nm)
        except Exception: k = None
    return k
def llm(system, user, key, model, provider, base):
    k = get_key(key, provider)
    if not k: return None
    if provider == "Anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", headers={"x-api-key": k, "anthropic-version": "2023-06-01", "content-type": "application/json"}, json={"model": (model or "claude-sonnet-5-5").strip(), "max_tokens": 1500, "system": system, "messages": [{"role": "user", "content": user}]}, timeout=90)
        r.raise_for_status(); return "".join(b.get("text", "") for b in r.json().get("content", []))
    r = requests.post((base or "https://api.openai.com/v1").rstrip("/") + "/chat/completions", headers={"Authorization": "Bearer " + k, "content-type": "application/json"}, json={"model": (model or "gpt-4o-mini").strip(), "max_tokens": 1500, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}, timeout=90)
    r.raise_for_status(); return r.json()["choices"][0]["message"]["content"]
def check_numbers(ans, ctx):
    cn = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", ctx.replace(",", ""))]; bad = []
    for s in re.findall(r"\d+(?:\.\d+)?", ans.replace(",", "")):
        v = float(s)
        if v <= 10 or 1990 <= v <= 2100: continue
        if not any(abs(v - c) <= max(.06, .015 * abs(c)) for c in cn): bad.append(s)
    return sorted(set(bad))
def refine_llm(provider, model, base, key):
    if "PR" not in G: return note("Load data first.", "info")
    if not get_key(key, provider): return note("No API key set: the deterministic, rule-based interpretation stays in use (fully functional).", "info")
    stats = [dict(id=s, size=int(p["n"]), customer_share=round(float(p["share"]), 3), revenue_share=round(float(p["rev_share"]), 3) if pd.notna(p["rev_share"]) else None, current_name=G["NARR"][s]["name"], top_differences=G["NARR"][s]["profile"],
                  metrics={k: (round(float(p[k]), 3) if pd.notna(p[k]) else None) for k in ["buy_share", "repeat", "active", "aov", "discount", "ctr", "cart", "conv", "ret"]}) for s, p in G["PR"].iterrows()]
    ctx = json.dumps(stats)
    try:
        txt = llm('You label customer segments. Use ONLY the supplied statistics. Never invent demographics, motivations or numbers. Return ONLY JSON: {"segments":[{"id":"S1","name":"...","summary":"...","hypotheses":["..."],"uncertainties":["..."]}]}. Hypotheses must be phrased as hypotheses.', "SEGMENT STATISTICS:\n" + ctx, key, model, provider, base)
        t = (txt or "").replace("`" * 3 + "json", "").replace("`" * 3, "").strip(); data = json.loads(t[t.index("{"): t.rindex("}") + 1]); segs = data["segments"]; names = [x["name"].strip() for x in segs]
        assert {x["id"] for x in segs} == set(G["PR"].index) and len(set(names)) == len(names) and all(3 <= len(n_) <= 60 for n_ in names), "ids/names invalid"
        for x in segs:
            assert isinstance(x["summary"], str) and len(x["summary"]) < 500 and all(isinstance(y, str) for y in x["hypotheses"] + x["uncertainties"]), "bad field types"
            assert not check_numbers(x["summary"], ctx), "summary contains numbers not in the statistics"
        for x in segs:
            N = G["NARR"][x["id"]]; N.update(name=x["name"].strip(), summary=x["summary"], hyps=x["hypotheses"][:5] + ["Customer motivations are not observed in this data."], unc=x["uncertainties"][:4] + N["unc"][-1:], src="LLM-assisted, validated against computed statistics"); G["PR"].loc[x["id"], "name"] = x["name"].strip()
        G["PLAY"]["Name"] = [G["NARR"][s]["name"] for s in G["PLAY"]["Segment"]]
        return note("LLM narratives validated and applied. Only aggregated segment statistics were sent (no customer-level data). Segment membership is still computed by the clustering model, not the LLM.", "ok")
    except Exception as e: return note("LLM output was rejected (%s). The deterministic interpretation stays in use." % esc(type(e).__name__ + ": " + str(e)[:80]), "warn")

# ------------------------------ playbooks ------------------------------
def render_playbooks():
    if "PLAY" not in G: return note("Load data first.", "info"), pd.DataFrame()
    P = G["PLAY"].copy(); h = note("Playbooks are generated from each segment's observed statistics. The table below is <b>editable</b> before export. The rationale separates observed facts, hypotheses and proposed experiments.", "info")
    for s in P["Segment"]:
        r = P[P["Segment"] == s].iloc[0]; h += "<div class='tw'><table><tr><th colspan='2'>%s · %s · objective: %s</th></tr><tr><td style='width:160px'>Observed facts</td><td>%s</td></tr><tr><td>Strategic hypotheses</td><td>%s</td></tr><tr><td>Proposed experiment</td><td>%s</td></tr></table></div>" % (
            s, esc(r["Name"]), esc(r["Objective"]), esc("; ".join(G["PLAYEV"][s])), esc(" ".join(G["NARR"][s]["hyps"][:3])), esc(r["Test design"]))
    return h, P
def playbook_md(df):
    d = pd.DataFrame(df); out = ["# Marketing playbooks (MOSAIC AI)", "", "_Hypotheses to test with controlled experiments; not validated results._", ""]
    for _, r in d.iterrows(): out += ["## %s · %s" % (r.get("Segment", ""), r.get("Name", ""))] + ["- **%s:** %s" % (c, r[c]) for c in PLAY_COLS[2:] if c in d.columns] + [""]
    return "\n".join(out)
def export_pb(df):
    d = pd.DataFrame(df); d.to_csv("mosaic_playbooks_edited.csv", index=False); open("mosaic_playbooks_edited.md", "w", encoding="utf-8").write(playbook_md(d)); return ["mosaic_playbooks_edited.csv", "mosaic_playbooks_edited.md"]

# ------------------------------ scenario simulator ------------------------------
SCN_COLS = ["Segment", "Name", "Eligible customers", "Budget ($; 0 = no cap)", "Target %", "Cost / contact ($)", "Response %", "Conversion %", "AOV ($)", "Discount %", "Margin %"]
def sc_vec(cu, tgt, cost, resp, conv, aov, disc, mar, bud, mult=1.0):
    cu, tgt, cost, resp, conv, aov, disc, mar, bud = [np.asarray(x, float) for x in (cu, tgt, cost, resp, conv, aov, disc, mar, bud)]
    cap = np.where((bud > 0) & (cost > 0), bud / np.where(cost > 0, cost, 1.0), np.inf); contacts = np.minimum(cu * tgt, cap); orders = contacts * np.minimum(1.0, resp * mult) * np.minimum(1.0, conv)
    rev = orders * aov * (1 - disc); contrib = orders * aov * (mar - disc); spend = contacts * cost; return dict(contacts=contacts, spend=spend, orders=orders, rev=rev, contrib=contrib, net=contrib - spend)
def validate_scn(d):
    nc = SCN_COLS[3:]; d[nc] = d[nc].apply(pd.to_numeric, errors="coerce")
    if (d[nc] < 0).any().any(): raise ValueError("Negative assumptions are not allowed.")
    for c in ["Target %", "Response %", "Conversion %", "Discount %", "Margin %"]:
        if (d[c] > 100).any(): raise ValueError("%s must be between 0 and 100." % c)
    return d
def scn_defaults(margin=45.0, cost=.5, target=50.0):
    PR, Zc = G["PR"], G["Zc"]; rows = []
    for s in PR.index:
        p = PR.loc[s]; disc = 10.0 if float(Zc.loc[s].get("discount_rate", 0)) > .6 else 0.0
        rows.append([s, G["NARR"][s]["name"], int(G["elig"][s]), 0.0, target, cost, round(float(np.clip(p["ctr"], .02, .25)) * 100, 1) if pd.notna(p["ctr"]) else 5.0, round(float(np.clip(p["conv"], .05, .5)) * 100, 1) if pd.notna(p["conv"]) else 15.0, None if pd.isna(p["aov"]) else round(float(p["aov"]), 2), disc, margin])
    return pd.DataFrame(rows, columns=SCN_COLS)
def scn_reset(margin, cost, target):
    if "PR" not in G: return (pd.DataFrame(columns=SCN_COLS),) + tuple(gr.update() for _ in range(6))
    d = scn_defaults(float(margin or 45), float(cost if cost is not None else .5), float(target or 50)); w = d["Eligible customers"].astype(float); wa = lambda c: float((d[c].astype(float) * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    return d, round(float(target or 50), 1), round(wa("Response %"), 1), round(wa("Conversion %"), 1), round(float(G["OV"]["aov"]) if pd.notna(G["OV"]["aov"]) else 0, 2), 10.0, round(float(margin or 45), 1)
def scn_core(df, bl, match, lowm, highm, uplift, cons):
    d = pd.DataFrame(df).copy()
    if d.shape[1] != len(SCN_COLS): raise ValueError("The scenario table was altered; press 'Reset to data-derived defaults'.")
    d.columns = SCN_COLS; d = validate_scn(d); d["elig"] = d["Segment"].map(G["elig"] if cons else G["cnt"]).astype(float)
    if d["elig"].isna().any(): raise ValueError("Unknown segment ID in the scenario table.")
    need = SCN_COLS[4:]; ok = d[need].notna().all(axis=1).values; a = lambda c: d[c].values.astype(float) / 100 if "%" in c else d[c].values.astype(float); res = {}
    for nm, m in (("Low", lowm), ("Base", 1.0), ("High", highm)):
        df_ = sc_vec(d["elig"].values, a("Target %"), a("Cost / contact ($)"), a("Response %") * (1 + uplift / 100), a("Conversion %"), a("AOV ($)"), a("Discount %"), a("Margin %"), a("Budget ($; 0 = no cap)"), m)
        tot_e = float(d["elig"].sum()); ct = float(np.nansum(df_["contacts"][ok])); bt = min(1.0, ct / tot_e) if match and tot_e > 0 else min(1.0, bl["t"] / 100)
        b_ = sc_vec([tot_e], [bt], [bl["cost"]], [bl["resp"] / 100], [bl["conv"] / 100], [bl["aov"]], [bl["disc"] / 100], [bl["mar"] / 100], [0.0], m)
        res[nm] = dict(diff={k: float(np.nansum(v[ok])) for k, v in df_.items()}, blanket={k: float(v[0]) for k, v in b_.items()}, rows=df_)
    return d, ok, res
def render_scn(df, bt, br, bc, ba, bd, bm, bk, match, lowm, highm, uplift, cons):
    if "PR" not in G: return note("Load data first.", "info"), empty("No data"), None
    try:
        bl = dict(t=float(bt or 0), resp=float(br or 0), conv=float(bc or 0), aov=float(ba or 0), disc=float(bd or 0), mar=float(bm or 0), cost=float(bk or 0))
        if min(bl.values()) < 0 or max(bl["t"], bl["resp"], bl["conv"], bl["disc"], bl["mar"]) > 100: raise ValueError("Blanket assumptions must be non-negative and percentages must be at most 100.")
        d, ok, res = scn_core(df, bl, bool(match), float(lowm), float(highm), float(uplift or 0), bool(cons))
    except Exception as e: return note("Invalid scenario input: " + esc(e), "err"), empty("Invalid input"), None
    b, a = res["Base"]["blanket"], res["Base"]["diff"]
    h = note("<b>Illustrative projection, not a forecast.</b> Defaults are seeded from observed engagement proxies (email click-through, visit-to-order conversion), <b>not measured campaign response</b>. Controlled experiments (holdout or A/B tests) are required to establish incremental lift.", "warn")
    h += note("<b>Formulas.</b> contacts = eligible customers x target share (capped by budget / cost per contact); orders = contacts x min(1, response x multiplier x (1 + relevance uplift)) x conversion; revenue = orders x AOV x (1 - discount); contribution = orders x AOV x (margin - discount); net = contribution - contacts x cost. "
              "Each customer belongs to one segment and is contacted at most once; the blanket campaign contacts " + ("the same number of customers as the differentiated plan" if match else "the share you set") + ". Cannibalisation and offer-driven response changes are <b>not</b> modelled.", "info")
    if not ok.all(): h += note("%d segment row(s) have missing inputs (for example no AOV because nobody purchased) and are excluded from the totals." % int((~ok).sum()), "warn")
    if bl["mar"] <= bl["disc"] or (d["Margin %"] <= d["Discount %"]).any(): h += note("Some discounts equal or exceed the margin: contribution per order is zero or negative there.", "warn")
    h += "<div class='kg'>" + card("Differentiated: net contribution", money(a["net"]), "spend %s · %s orders" % (money(a["spend"]), nf(a["orders"])), C["li"] if a["net"] >= 0 else C["co"]) + card("Blanket: net contribution", money(b["net"]), "spend %s · %s orders" % (money(b["spend"]), nf(b["orders"])), C["li"] if b["net"] >= 0 else C["co"]) + \
         card("Difference (base)", money(a["net"] - b["net"]), "differentiated minus blanket", C["la"]) + card("Contacted customers", nf(a["contacts"]), "no customer counted twice") + "</div>"
    rows = [dict(sc=nm, ds=r["diff"]["spend"], dr=r["diff"]["rev"], dn=r["diff"]["net"], bs=r["blanket"]["spend"], br=r["blanket"]["rev"], bn=r["blanket"]["net"]) for nm, r in res.items()]
    h += H("Low / base / high (user-defined multipliers on response rate)") + tbl(pd.DataFrame(rows), [("sc", "Scenario"), ("ds", "Diff. spend"), ("dr", "Diff. revenue"), ("dn", "Diff. net contribution"), ("bs", "Blanket spend"), ("br", "Blanket revenue"), ("bn", "Blanket net contribution")], dict(ds=money, dr=money, dn=mcol, bs=money, br=money, bn=mcol))
    r0 = res["Base"]["rows"]; seg = pd.DataFrame(dict(s=d["Segment"].values, n=d["Name"].values, contacts=r0["contacts"], spend=r0["spend"], orders=r0["orders"], rev=r0["rev"], net=r0["net"]))
    h += H("Base scenario by segment (differentiated)") + tbl(seg, [("s", "Segment"), ("n", "Name"), ("contacts", "Contacts"), ("spend", "Spend"), ("orders", "Orders"), ("rev", "Revenue"), ("net", "Net contribution")], dict(contacts=nf, spend=money, orders=nf, rev=money, net=mcol))
    fig = go.Figure([go.Bar(x=list(res), y=[r["diff"]["net"] for r in res.values()], name="Differentiated", marker_color=C["li"]), go.Bar(x=list(res), y=[r["blanket"]["net"] for r in res.values()], name="Blanket", marker_color=C["la"])]); fig.update_layout(barmode="group"); fig = sty(fig, 340, "Net contribution after campaign costs (illustrative)")
    out = pd.concat([seg.assign(kind="differentiated_base"), pd.DataFrame(rows).rename(columns={"sc": "s"}).assign(kind="totals")], ignore_index=True); path = "mosaic_scenario_results.csv"; out.to_csv(path, index=False); return h, fig, path
def scn_default_result():
    d, t, r_, c_, a_, d_, m_ = scn_reset(45, .5, 50); return scn_core(d, dict(t=t, resp=r_, conv=c_, aov=a_, disc=d_, mar=m_, cost=.5), True, .6, 1.4, 0.0, True)

# ------------------------------ quality & methodology ------------------------------
def eta2(y, g):
    y = np.asarray(y, float); g = pd.Series(g).astype(str).values; m = ~np.isnan(y); y, g = y[m], g[m]
    if len(y) < 30 or np.var(y) == 0: return np.nan
    mu = y.mean(); return sum(len(y[g == k]) * (y[g == k].mean() - mu) ** 2 for k in np.unique(g)) / ((y - mu) ** 2).sum()
def audit(F, col):
    if F[col].notna().sum() < 100 or F[col].nunique() > 12: return ""
    ct = pd.crosstab(F["seg"], F[col]); sh = ct.div(ct.sum(axis=1), axis=0); ratio = sh / F[col].value_counts(normalize=True)
    f = lambda v, n_: ("<span style='color:%s'>%.2fx</span>" % (C["co"], v) if (v > 1.5 or v < .67) and n_ >= 30 else "%.2fx" % v)
    h = "<div class='tw'><table><tr><th>Segment</th>" + "".join("<th>%s</th>" % esc(c) for c in ratio.columns) + "</tr>"
    for s in ratio.index: h += "<tr><td>%s</td>" % s + "".join("<td>%s</td>" % f(ratio.loc[s, c], ct.loc[s, c]) for c in ratio.columns) + "</tr>"
    return H("Composition audit: %s (share within segment ÷ share overall; coral = over/under-represented, n≥30)" % col) + h + "</table></div>"
def render_quality():
    if "F" not in G: return note("Load data first.", "info"), ""
    F, M, dq = G["F"], G["M"], G["dq"]; b = M["best"]; h = note("Customers analysed: <b>%s</b> (transaction rows read: %s). " % (nf(len(F)), nf(G["raw_n"])) + ("<b>Illustrative synthetic data.</b>" if G["demo"] else "Uploaded data; never written into the project ZIP.") + (" IDs are pseudonymised (salted SHA-256)." if G["pseud"] else ""), "ok")
    h += "".join(note(esc(w), "warn") for w in G["W"])
    cov = [("Duplicate transactions removed", nf(dq.get("dup_tx"))), ("Duplicate interaction events removed", nf(dq.get("dup_it"))), ("Invalid transaction dates removed", nf(dq.get("bad_dates"))), ("Transactions missing a value (removed)", nf(dq.get("missing_value"))), ("Returned order lines", pc1(dq.get("returns"))),
           ("Customers with interaction tracking", pc0(F["tracked"].mean())), ("Customers eligible for contact (consent)", pc0(F["eligible"].mean()))]; h += H("Data quality report") + tbl(pd.DataFrame(cov, columns=["a", "b"]), [("a", "Check"), ("b", "Result")])
    ms = pd.DataFrame([dict(f=LAB.get(c, c), miss=1 - F[c].notna().mean(), used="used" if c in G["used"] else "excluded") for c in CAND]); h += H("Feature availability and missingness (missing tracking is never treated as zero activity)") + tbl(ms, [("f", "Candidate feature"), ("miss", "Missing"), ("used", "Status")], dict(miss=pc0))
    h += note("Never used for clustering: customer_id, transaction_id and other identifiers; demographics (age_band, region, loyalty_status) only describe and audit segments. Excluded features: " + (esc("; ".join("%s (%s)" % (LAB.get(c, c), w) for c, w in G["excl"])) or "none") + ". Median imputation applied to: " + (esc(", ".join("%s %s" % (LAB.get(c, c), pc0(v)) for c, v in G["imp"].items() if v > 0)) or "nothing") + ".", "info")
    rows = pd.DataFrame([{k: v for k, v in r.items() if k != "lab"} for r in sorted(M["rows"], key=lambda r: -r["score"])]); rows["sel"] = np.where((rows["algo"] == b["algo"]) & (rows["k"] == b["k"]), "SELECTED", "")
    h += H("Model comparison (composite score; silhouette alone does not decide)") + tbl(rows, [("algo", "Algorithm"), ("k", "k"), ("sil", "Silhouette"), ("db", "Davies-Bouldin (lower is better)"), ("stab", "Stability (ARI, 85% subsamples)"), ("minshare", "Smallest segment"), ("score", "Composite score"), ("sel", "")], dict(sil=fx, db=fx, stab=fx, minshare=pc1, score=fx))
    h += note("Composite score = 0.30 silhouette + 0.25 stability + 0.15 Davies-Bouldin + 0.15 minimum segment size (target ≥5%) + 0.15 interpretability (fewer segments score higher); halved if any segment has under 10 customers. Segment IDs are ordered by median net spend, so S1 is the highest-value segment in this run and IDs are not stable across retrains.", "info")
    if M["fails"]: h += note("Combinations that failed or were skipped: " + esc("; ".join("%s k=%s (%s)" % f for f in M["fails"][:6])), "warn")
    sens = rows[rows["algo"] == b["algo"]].sort_values("k"); h += H("Sensitivity to the number of clusters (%s)" % b["algo"]) + tbl(sens, [("k", "k"), ("sil", "Silhouette"), ("stab", "Stability"), ("minshare", "Smallest segment"), ("score", "Composite score")], dict(sil=fx, stab=fx, minshare=pc1, score=fx))
    sz = G["PR"]["n"]; w_ = []
    if sz.max() / sz.min() > 10: w_.append("Segment sizes are very imbalanced (largest/smallest = %.0fx)." % (sz.max() / sz.min()))
    if pd.notna(b["stab"]) and b["stab"] < .6: w_.append("Stability is low (ARI %.2f): segments may change materially on resampling." % b["stab"])
    if b["sil"] < .15: w_.append("Overall separation is weak (silhouette %.2f): treat segments as soft patterns rather than distinct groups." % b["sil"])
    h += "".join(note(esc(x), "warn") for x in w_) or note("No imbalance, instability or weak-separation flags at the selected model.", "ok")
    dem = [c for c in ["age_band", "region", "loyalty_status"] if F[c].notna().sum() >= 100]
    if dem:
        lm = np.log1p(F["monetary"].clip(lower=0)); e = [dict(g="Behavioural segments (k=%d)" % len(G["PR"]), a=eta2(lm, F["seg"]), b=eta2(F["frequency"], F["seg"]))] + [dict(g="Demographic only: " + c, a=eta2(lm, F[c]), b=eta2(F["frequency"], F[c])) for c in dem]
        h += H("Behavioural segmentation vs demographic-only baseline") + tbl(pd.DataFrame(e), [("g", "Grouping"), ("a", "Share of net-spend variance explained"), ("b", "Share of order-frequency variance explained")], dict(a=pc1, b=pc1)) + note("Spend and frequency are inputs to the clustering, so this comparison favours behavioural segments by construction. It shows that demographics alone describe behaviour poorly, not that the segments are 'correct'.", "warn")
        for c in dem: h += audit(F, c)
        h += note("Demographics are used for auditing only. Do not use them to exclude customers or to infer sensitive traits. Groups flagged coral deserve human review before offers differ between them.", "warn")
    else: h += note("No usable demographic fields: composition audit and demographic baseline are unavailable.", "info")
    h += note("<b>Targeting risks.</b> Contact only customers who have given consent (%s of customers are eligible here). Avoid offers that systematically disadvantage groups and pressure tactics; respect opt-outs and contact-frequency caps." % pc0(F["eligible"].mean()), "warn")
    pc = G.get("prev_cmp")
    if pc is not None and len(pc): h += H("Comparison with the previous run in this session (nearest-profile matching)") + tbl(pc, [("new", "New segment"), ("old", "Closest previous segment"), ("dist", "Profile distance (sd)"), ("new_n", "New size"), ("old_n", "Previous size")], dict(dist=fx, new_n=nf, old_n=nf))
    h += H("Segment-refresh workflow") + note("1) Re-run on new data with the same settings. 2) Compare new profiles with the previous run (table above appears after a second run). 3) Treat IDs and names as labels for this run only; a customer's segment can change when behaviour or the model changes. 4) Re-validate playbooks with holdout tests before reusing them.", "info")
    return h, H("Self-test results (run at start-up)") + "".join(note(("PASS " if ok else "FAIL ") + esc(g + ": " + n) + " — " + esc(d), "ok" if ok else "err") for g, n, ok, d in TESTS)

# ------------------------------ lookup ------------------------------
def render_lookup(cid):
    if "F" not in G or not cid: return note("Select or type a customer ID.", "info")
    cid = str(cid).strip(); key = ("C-" + hashlib.sha256((G["salt"] + cid).encode()).hexdigest()[:10]) if G["pseud"] and not cid.startswith("C-") else cid; F = G["F"]
    if key not in F.index: return note("Customer ID not found%s." % (" (IDs are pseudonymised in this session)" if G["pseud"] else ""), "warn")
    r = F.loc[key]; s = r["seg"]; N = G["NARR"][s]; P = G["PLAY"][G["PLAY"]["Segment"] == s].iloc[0]
    m = [("Recency (days since last order)", fx0(r["recency_days"])), ("Orders", nf(r["frequency"])), ("Net spend", money(r["monetary"])), ("Average order value", money(r["gross"] / r["frequency"]) if r["frequency"] > 0 else "—"), ("Discount usage", pc0(r["discount_rate"])),
         ("Site visits / 30 days", fx(r["visit_rate_30d"])), ("Email click-through", pc0(r["email_ctr"])), ("Cart abandonment", pc0(r["cart_abandon_rate"])), ("Interaction tracking", "yes" if r["tracked"] else "no (engagement unknown, not zero)"), ("Contact consent", "eligible" if r["eligible"] else "not eligible")]
    h = "<div class='kg'>" + card("Assigned segment", s + " · " + esc(N["name"]), "behavioural grouping", segcol(s)) + card("Assignment margin", fx(r["margin"]), "distance-based, not a probability (0 = on the boundary with %s)" % r["second"], C["la"]) + "</div>"
    h += note("<b>Why this segment:</b> " + esc(r["reasons"]) + ". This describes observed behaviour; it does not establish this individual's motivations.", "info") + tbl(pd.DataFrame(m, columns=["a", "b"]), [("a", "Metric"), ("b", "Value")], dict(b=str))
    return h + H("Segment playbook (for the segment, not a personalised promise)") + note("<b>%s</b>: %s. Channels: %s. Offer: %s. No-discount alternative: %s." % (esc(P["Objective"]), esc(P["Campaign concept"]), esc(P["Channels"]), esc(P["Offer strategy"]), esc(P["No-discount alternative"])), "info")

# ------------------------------ AI strategist (tools) ------------------------------
def find_segs(q):
    ql = q.lower(); out = [s for s in G["PR"].index if re.search(r"\b%s\b" % s.lower(), ql)]; stop = {"customers", "shoppers", "buyers", "customer", "with", "low", "high", "value"}
    for s in G["PR"].index:
        w = [x for x in re.findall(r"[a-z]{4,}", G["NARR"][s]["name"].lower()) if x not in stop]
        if s not in out and any(x in ql for x in w): out.append(s)
    return out[:3]
def t_overview():
    PR, F, OV = G["PR"], G["F"], G["OV"]; return "%s customers in %d behavioural segments; observed net revenue %s; average customer value %s; repeat-purchase rate %s; %s active in the last %d days. Model: %s (k=%d)." % (nf(len(F)), len(PR), money(PR["rev"].sum()), money(OV["avg_val"]), pc0(OV["repeat"]), pc0(OV["active"]), G["act"], G["M"]["best"]["algo"], G["M"]["best"]["k"])
def t_value():
    d = G["PR"].sort_values("avg_val", ascending=False); return "Segments by net value per customer: " + "; ".join("%s %s: %s per customer, %s of revenue from %s of customers" % (s, G["NARR"][s]["name"], money(r["avg_val"]), pc0(r["rev_share"]), pc0(r["share"])) for s, r in d.iterrows())
def t_features(s):
    z = G["Zc"].loc[s]; top = z.reindex(z.abs().sort_values(ascending=False).index)[:5]; return "%s %s differs most on: %s (standard deviations from the all-customer average)." % (s, G["NARR"][s]["name"], "; ".join("%s %+.1f" % (LAB.get(c, c), v) for c, v in top.items()))
def t_compare(a, b):
    PR = G["PR"]; ks = ["avg_val", "aov", "repeat", "active", "discount", "ctr", "conv"]; return "%s vs %s: " % (a, b) + "; ".join("%s %s vs %s" % (lab, f(PR.loc[a, k]), f(PR.loc[b, k])) for k, lab, f in METRICS if k in ks) + ". " + t_features(a) + " " + t_features(b)
def t_discount():
    d = G["PR"].sort_values("discount", ascending=False); return "Discount reliance (share of orders with a discount): " + "; ".join("%s %s %s (discount value %s)" % (s, G["NARR"][s]["name"], pc0(r["discount"]), money(r["disc_amt"])) for s, r in d.iterrows() if pd.notna(r["discount"]))
def t_reduce(pct):
    PR = G["PR"]; tot = PR["disc_amt"].sum(min_count=1)
    if pd.isna(tot): return "Discount amounts are not in the data, so the effect of reducing discounts cannot be computed."
    sv = tot * pct / 100; ord_ = PR["freq"].mul(PR["n"]).sum(); avg = G["OV"]["aov"]; mar = .45; cpo = avg * mar if pd.notna(avg) else np.nan
    return "Observed discount spend is %s. A %.0f%% reduction would save about %s directly. The effect on order volume is unknown from observational data: at an assumed %.0f%% margin and average order value %s, the saving is offset if about %s orders (%s of all orders) are lost. Test with a holdout before changing discounts broadly." % (money(tot), pct, money(sv), mar * 100, money(avg), nf(sv / cpo) if pd.notna(cpo) and cpo > 0 else "—", pc1(sv / cpo / ord_) if pd.notna(cpo) and cpo > 0 and ord_ > 0 else "—")
def t_engaged_weak():
    PR, Zc = G["PR"], G["Zc"]; m = [s for s in PR.index if float(Zc.loc[s].get("visit_rate_30d", 0)) > 0 and float(Zc.loc[s].get("conversion_rate", 0)) < 0]
    return "Higher-than-average visits but below-average conversion: " + ("; ".join("%s %s (visits %s/30d, conversion %s)" % (s, G["NARR"][s]["name"], fx(PR.loc[s, "visit"]), pc0(PR.loc[s, "conv"])) for s in m) if m else "no segment shows this pattern (or interaction data is missing).")
def t_inactive():
    PR = G["PR"].sort_values("active"); s = PR.index[0]; P = G["PLAY"][G["PLAY"]["Segment"] == s].iloc[0]; return "Least active segment: %s %s (%s active, median recency of buyers %s days). Suggested test: %s; timing: %s; offer: %s; no-discount alternative: %s." % (s, G["NARR"][s]["name"], pc0(PR.loc[s, "active"]), fx0(PR.loc[s, "recency"]), P["Campaign concept"], P["Timing / trigger"], P["Offer strategy"], P["No-discount alternative"])
def t_next():
    d, ok, res = scn_default_result(); r = res["Base"]["rows"]; seg = pd.DataFrame(dict(s=d["Segment"].values, c=r["contacts"], n=r["net"])); seg["pc"] = seg["n"] / seg["c"].replace(0, np.nan); seg = seg.dropna().sort_values("pc", ascending=False)
    return ("Using illustrative, data-seeded assumptions (not measured response), the segments with the highest modelled net contribution per contact are: " + "; ".join("%s %s (%s per contact)" % (r_.s, G["NARR"][r_.s]["name"], money(r_.pc)) for r_ in seg.head(3).itertuples()) + ". Validate with a holdout test before committing budget.") if len(seg) else "Not enough inputs to rank segments."
def t_playbook(s):
    P = G["PLAY"][G["PLAY"]["Segment"] == s].iloc[0]; return "%s %s playbook: objective %s; concept %s; channels %s; offer %s; no-discount alternative %s; timing %s; KPI %s (guardrail %s); test: %s" % (s, G["NARR"][s]["name"], P["Objective"], P["Campaign concept"], P["Channels"], P["Offer strategy"], P["No-discount alternative"], P["Timing / trigger"], P["Primary KPI"], P["Guardrail KPI"], P["Test design"])
def t_limits():
    return "Limitations: segments are statistical patterns, not verified customer types, and may shift after retraining; interaction tracking covers %s of customers; revenue is observed in-window and is not lifetime value; scenarios use illustrative assumptions and need controlled experiments; demographics are used for auditing only. Data warnings: %s" % (pc0(G["F"]["tracked"].mean()), " | ".join(G["W"][:4]))
def ask(q, provider="Anthropic", model="", base="", key="", use_llm=True):
    if "PR" not in G: return "Load data first."
    q = (q or "").strip() or "Give me an overview and tell me which segment should receive the next campaign."; ql = q.lower(); ss = find_segs(q); parts = []
    m = re.search(r"(\d+)\s*%", ql)
    if re.search(r"next campaign|which segment should|prioriti", ql): parts.append(("scenario_calculation", t_next()))
    if re.search(r"valuable|value|worth", ql): parts.append(("revenue_analysis", t_value()))
    if len(ss) >= 2 and re.search(r"differ|compare|versus| vs |why", ql): parts.append(("segment_comparison", t_compare(ss[0], ss[1])))
    elif len(ss) == 1 and re.search(r"why|differ|feature|behav", ql): parts.append(("feature_differences", t_features(ss[0])))
    if re.search(r"discount|promo", ql): parts.append(("discount_analysis", t_reduce(float(m.group(1)) if m else 20.0) if re.search(r"reduc|cut|lower|change", ql) else t_discount()))
    if re.search(r"engag.*(weak|low|poor)|weak.*conver|high engagement|browse", ql): parts.append(("channel_interaction_analysis", t_engaged_weak()))
    if re.search(r"inactive|lapsed|dormant|win-back|reactivat", ql): parts.append(("playbook_retrieval", t_inactive()))
    if len(ss) == 1 and re.search(r"playbook|strategy|campaign|test", ql): parts.append(("playbook_retrieval", t_playbook(ss[0])))
    if re.search(r"limit|caveat|bias|risk|reliab", ql): parts.append(("limitations", t_limits()))
    if "report" in ql: parts.append(("report", "Use 'Generate executive report' in the Quality & Methodology tab."))
    if not parts: parts = [("overall_summary", t_overview()), ("revenue_analysis", t_value())]
    ctx = "\n".join("[%s] %s" % (n, t) for n, t in parts); base_ans = "### Answer (deterministic, computed by tools)\n" + "\n".join("- **%s**: %s" % (n, t) for n, t in parts) + "\n\n*Tools used: " + ", ".join(n for n, _ in parts) + ". Segments were created by the clustering model, not by an LLM.*"
    if not use_llm: return base_ans
    if not get_key(key, provider): return base_ans + "\n\n*No API key set: the transparent rule-based assistant answered. Add a key (session only) for natural-language synthesis.*"
    try:
        ans = llm("You are a marketing analytics assistant. Answer ONLY from the TOOL RESULTS. Never invent numbers, motivations or segments. Label explanations as hypotheses. Be concise. Segments come from a clustering model, not from you.", "TOOL RESULTS:\n" + ctx + "\n\nQUESTION: " + q, key, model, provider, base); bad = check_numbers(ans, ctx)
        return "### LLM-written answer (grounded in tool results; only aggregated statistics were sent)\n" + ans + (("\n\n⚠️ Numbers not found in tool results: " + ", ".join(bad) + ". Treat as unverified.") if bad else "\n\nNumeric check passed: all figures match tool results.") + "\n\n---\n" + base_ans
    except Exception as e: return base_ans + "\n\n*LLM call failed (%s); the deterministic answer is shown.*" % type(e).__name__

# ------------------------------ exports, report ------------------------------
def customers_df(F, NARR):
    return pd.DataFrame({"customer_id": F.index, "segment_id": F["seg"].values, "segment_name": F["seg"].map(lambda s: NARR[s]["name"]).values, "assignment_margin": F["margin"].round(3).values, "nearest_other_segment": F["second"].values,
                         "reasons": F["reasons"].values, "recency_days": F["recency_days"].values, "orders": F["frequency"].values, "net_spend": F["monetary"].round(2).values})
def export_all(pb):
    if "F" not in G: return None, note("Load data first.", "info")
    out = []; customers_df(G["F"], G["NARR"]).to_csv("mosaic_customer_segments.csv", index=False); out.append("mosaic_customer_segments.csv")
    PRj = G["PR"].copy(); PRj["narrative"] = [json.dumps(G["NARR"][s]) for s in PRj.index]; PRj.drop(columns=["id"]).to_csv("mosaic_segment_profiles.csv"); out.append("mosaic_segment_profiles.csv")
    js = [dict(id=s, **{k: (None if pd.isna(v) else v) for k, v in G["PR"].loc[s].items() if k != "id"}, narrative=G["NARR"][s]) for s in G["PR"].index]
    open("mosaic_segment_profiles.json", "w").write(json.dumps(js, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o), indent=1)); out.append("mosaic_segment_profiles.json")
    df = pd.DataFrame(pb) if pb is not None and len(pd.DataFrame(pb)) else G["PLAY"]; df.to_csv("mosaic_playbooks.csv", index=False); open("mosaic_playbooks.md", "w", encoding="utf-8").write(playbook_md(df)); out += ["mosaic_playbooks.csv", "mosaic_playbooks.md"]
    try:
        d, ok, res = scn_default_result(); pd.DataFrame([dict(scenario=k, **{"diff_" + a: b for a, b in v["diff"].items()}, **{"blanket_" + a: b for a, b in v["blanket"].items()}) for k, v in res.items()]).to_csv("mosaic_scenario_defaults.csv", index=False); out.append("mosaic_scenario_defaults.csv")
    except Exception: pass
    return out, note("Exported %d files from the current dataset and model." % len(out), "ok")
def make_report(pb):
    if "PR" not in G: return None, note("Load data first.", "info")
    PR, NARR, M = G["PR"], G["NARR"], G["M"]["best"]; h = lambda t: "<h2>" + t + "</h2>"; p = []
    p.append("<h1>MOSAIC AI — Executive Report</h1><p class='m'>Every customer is a pattern. Find the ones that matter.</p>" + ("<p class='w'>Illustrative synthetic data: fictional customers; not real results.</p>" if G["demo"] else ""))
    p.append(h("Executive summary") + "<p>%s</p><p>%s</p>" % (esc(t_overview()), esc(t_value())))
    p.append(h("Dataset scope and limitations") + "<ul>" + "".join("<li>%s</li>" % esc(w) for w in G["W"]) + "</ul><p>%s</p>" % esc(t_limits()))
    p.append(h("Segmentation methodology") + "<p>Behavioural features (%s) were log-transformed where skewed, winsorised at the 1st/99th percentile and standardised. K-Means, Agglomerative and Gaussian Mixture models were compared across candidate k using silhouette, Davies-Bouldin, bootstrap stability, minimum segment size and interpretability. Selected: %s with k=%d (silhouette %s, stability %s). Demographics were used for auditing only.</p>" % (esc(", ".join(LAB.get(c, c) for c in G["used"])), M["algo"], M["k"], fx(M["sil"]), fx(M["stab"])))
    p.append(h("Segments: size, value and engagement") + tbl(PR, [("id", "ID"), ("name", "Segment"), ("n", "Customers"), ("share", "Share"), ("rev_share", "Revenue share"), ("avg_val", "Net value / customer"), ("repeat", "Repeat rate"), ("active", "Active"), ("discount", "Discount usage"), ("ctr", "Email CTR")], dict(n=nf, share=pc1, rev_share=pc1, avg_val=money, repeat=pc0, active=pc0, discount=pc0, ctr=pc0)))
    p.append(h("Most important behavioural differences") + "<ul>" + "".join("<li><b>%s %s:</b> %s</li>" % (s, esc(NARR[s]["name"]), esc(NARR[s]["summary"])) for s in PR.index) + "</ul>")
    pbd = pd.DataFrame(pb) if pb is not None and len(pd.DataFrame(pb)) else G["PLAY"]; p.append(h("Recommended marketing playbooks (hypotheses)") + tbl(pbd, [("Segment", "ID"), ("Name", "Segment"), ("Objective", "Objective"), ("Campaign concept", "Concept"), ("Channels", "Channels"), ("Offer strategy", "Offer"), ("Primary KPI", "KPI")]))
    p.append(h("Proposed experiments") + "<ul>" + "".join("<li><b>%s:</b> %s</li>" % (r["Segment"], esc(r["Test design"])) for _, r in pbd.iterrows()) + "</ul>")
    try:
        d, ok, res = scn_default_result(); a, b = res["Base"]["diff"], res["Base"]["blanket"]; p.append(h("Scenario results (illustrative projections, not forecasts)") + "<p>Default assumptions are seeded from observed engagement proxies. Base case: differentiated plan net contribution %s vs blanket %s at the same contact volume (%s contacts). Cannibalisation is not modelled; controlled experiments are required to establish incremental lift.</p>" % (money(a["net"]), money(b["net"]), nf(a["contacts"])))
    except Exception: pass
    p.append(h("Data-quality, fairness and measurement cautions") + "<ul><li>Segments are statistical patterns and may shift on retraining.</li><li>Demographics are for auditing only; do not use them to exclude customers.</li><li>Missing tracking is not zero activity.</li><li>Contact only consented customers.</li></ul>")
    f = go.Figure([go.Bar(x=list(PR.index), y=PR["share"], name="Customers", marker_color=C["la"]), go.Bar(x=list(PR.index), y=PR["rev_share"], name="Net revenue", marker_color=C["li"])]); f.update_layout(barmode="group", title="Customer share vs revenue share", height=340); p.append(f.to_html(full_html=False, include_plotlyjs="cdn"))
    css = "body{font-family:Inter,Arial,sans-serif;background:#151A28;color:#F5F5F0;max-width:1000px;margin:30px auto;padding:0 18px}h1{color:#B8F36B}h2{color:#D6B9FF;border-left:3px solid #B8F36B;padding-left:10px;margin-top:28px}.m{color:#A9B1C5}.w{background:#F2B6A022;border-left:3px solid #F2B6A0;padding:8px 12px}table{border-collapse:collapse;width:100%;background:#22293A;margin:8px 0}th{background:#2c3550;color:#B8F36B;text-align:left;padding:8px}td{padding:7px 8px;border-top:1px solid #ffffff14}.tw{overflow-x:auto}"
    path = "mosaic_executive_summary.html"; open(path, "w", encoding="utf-8").write("<html><head><meta charset='utf-8'><title>MOSAIC AI Report</title><style>" + css + "</style></head><body>" + "".join(p) + "</body></html>"); return path, note("Executive report generated (aggregated figures only).", "ok")

# ------------------------------ self-tests ------------------------------
def _mini():
    return pd.DataFrame(dict(customer_id=["a", "a", "b", "b", "c", "c"], transaction_id=["t1", "t2", "t3", "t3", "t4", "t5"], transaction_date=["2026-01-10", "2026-02-10", "2026-01-20", "2026-01-20", "2026-03-01", "2026-03-05"], product_id=["p1", "p2", "p1", "p1", "p3", "p3"],
                             product_category=["X", "Y", "X", "X", "Y", "Y"], quantity=[1] * 6, unit_price=[100, 50, 80, 80, 40, 40], transaction_value=[100, 50, 80, 80, 40, 40], discount_amount=[0, 10, 0, 0, 0, 0], discount_used=[0, 1, 0, 0, 0, 0], return_flag=[0, 0, 0, 0, 1, 0]))
def _small(with_it=True):
    if with_it not in _SM:
        tx, it, cu = make_synth(400, seed=3); _SM[with_it] = core(tx, it if with_it else None, cu, {}, "", 90, ["K-Means"], 3, 4, 0)
    return _SM[with_it]
def run_selftests(verbose=True):
    T = []; saved = dict(G); ad = pd.Timestamp("2026-04-01")
    def chk(g, name, fn):
        try: ok, det = fn(); T.append((g, name, bool(ok), det))
        except Exception as e: T.append((g, name, False, type(e).__name__ + ": " + str(e)[:90]))
    def feat():
        W, dq = [], {}; TX = clean_tx(_mini(), {}, W, dq); return build_features(TX, None, None, ad, 90, []), dq
    def t_agg(): F, dq = feat(); return list(F["frequency"]) == [2, 1, 1] and list(F["monetary"]) == [150, 80, 40], "orders %s, net spend %s" % (list(F["frequency"]), list(F["monetary"]))
    def t_rec(): F, dq = feat(); return list(F["recency_days"]) == [50, 71, 27], "recency vs fixed date 2026-04-01: %s" % list(F["recency_days"])
    def t_dup(): F, dq = feat(); return dq["dup_tx"] == 1, "duplicate transaction lines removed: %d" % dq["dup_tx"]
    def t_ret(): F, dq = feat(); return F.loc["c", "monetary"] == 40 and F.loc["c", "frequency"] == 1 and abs(F.loc["c", "return_rate"] - .5) < 1e-9, "returned line adds no revenue or order; return rate 50%"
    def t_cols():
        try: clean_tx(pd.DataFrame({"x": [1]}), {}, [], {}); return False, "no error raised"
        except ValueError as e: return "Transactions need" in str(e), "clear error for missing columns"
    def t_miss():
        d = _mini(); d.loc[0, "transaction_value"] = np.nan; W, dq = [], {}; TX = clean_tx(d, {}, W, dq); return dq["missing_value"] == 1 and len(TX) == 4, "row with missing value removed, not counted as zero"
    def t_scale(): Z = _small()["Z"]; return bool((Z.mean().abs() < 1e-6).all() and ((Z.std(ddof=0) - 1).abs() < 1e-6).all()), "all %d clustering features have mean 0, sd 1" % Z.shape[1]
    def t_ids(): R = _small(); bad = [c for c in R["used"] if c not in CAND or "id" in c.split("_")]; return not bad and "customer_id" not in R["Z"].columns, "clustered on %d behavioural features, no identifiers" % len(R["used"])
    def t_track(): R = _small(False); return R["F"]["visit_rate_30d"].isna().all() and "visit_rate_30d" not in R["used"], "no interaction file: engagement is missing (not zero) and excluded"
    def t_clu(): R = _small(); F = R["F"]; return F["seg"].nunique() >= 2 and R["PR"]["n"].sum() == len(F), "%d segments cover all %d customers (%s k=%d)" % (F["seg"].nunique(), len(F), R["M"]["best"]["algo"], R["M"]["best"]["k"])
    def t_small():
        try: run_seg(pd.DataFrame(np.random.default_rng(0).normal(size=(20, 3))), ["K-Means"], 3, 4, 0); return False, "no error"
        except ValueError as e: return "at least 60" in str(e), "tiny dataset rejected with a clear message"
    def t_ident():
        try: make_Z(pd.DataFrame({"frequency": [1.0] * 100, "monetary": [50.0] * 100, "recency_days": [10.0] * 100}), ["frequency", "monetary", "recency_days"]); return False, "no error"
        except ValueError as e: return "identical" in str(e), "identical customers handled gracefully"
    def t_prof(): PR = _small()["PR"]; return abs(PR["rev_share"].sum() - 1) < 1e-6 and abs(PR["share"].sum() - 1) < 1e-9, "revenue shares sum to %.4f, customer shares to %.4f" % (PR["rev_share"].sum(), PR["share"].sum())
    def t_strat(): R = _small(False); bad = [s for s, ev in R["PLAYEV"].items() if any("click-through" in x for x in ev)]; return not bad and R["PLAY"]["Channels"].str.contains("not tracked").any(), "no engagement claims made where interaction data is missing"
    def t_fall():
        R = _small(); G.clear(); G.update(R); G.update(demo=True, act=90, pseud=False, salt="s", cnt=R["F"].groupby("seg").size().to_dict(), elig=R["F"].groupby("seg")["eligible"].sum().to_dict())
        a = ask("what are the limitations?", use_llm=False); k = get_key("", "Anthropic", env=False); G.clear(); return k is None and "deterministic" in a, "no key: deterministic tool-grounded answer is returned"
    def t_arith(): r = sc_vec([1000], [.5], [1], [.1], [.2], [100], [.1], [.5], [0]); return abs(r["orders"][0] - 10) < 1e-9 and abs(r["rev"][0] - 900) < 1e-9 and abs(r["contrib"][0] - 400) < 1e-9 and abs(r["net"][0] + 100) < 1e-9, "orders 10, revenue 900, contribution 400, net -100"
    def t_inv():
        d = pd.DataFrame([["S1", "x", 10, 0, 50, -1, 5, 10, 50, 0, 40]], columns=SCN_COLS)
        try: validate_scn(d); return False, "no error"
        except ValueError: return True, "negative assumption rejected"
    def t_nodbl(): r = sc_vec([100, 50], [1, 1], [1, 1], [.1, .1], [.2, .2], [10, 10], [0, 0], [.5, .5], [0, 20]); return r["contacts"].sum() <= 150 and abs(r["contacts"][1] - 20) < 1e-9, "contacts never exceed eligible customers; budget cap limits spend to $20"
    def t_exp(): R = _small(); d = customers_df(R["F"], R["NARR"]); return len(d) == len(R["F"]) and {"customer_id", "segment_id", "segment_name", "assignment_margin", "reasons"} <= set(d.columns) and d["customer_id"].is_unique, "export has %d rows and the expected schema" % len(d)
    for g_, n_, f_ in [("data", "transaction aggregation", t_agg), ("data", "recency against a fixed analysis date", t_rec), ("data", "duplicate transactions", t_dup), ("data", "returns handling", t_ret), ("data", "missing columns", t_cols), ("data", "missing values", t_miss),
                       ("features", "scaling", t_scale), ("features", "identifiers excluded", t_ids), ("features", "missing tracking is not zero", t_track), ("segmentation", "clustering on valid data", t_clu), ("segmentation", "small dataset", t_small),
                       ("segmentation", "identical records", t_ident), ("segmentation", "segment profile calculations", t_prof), ("strategy", "strategy uses only available evidence", t_strat), ("strategy", "LLM fallback without a key", t_fall),
                       ("scenarios", "arithmetic", t_arith), ("scenarios", "invalid assumptions", t_inv), ("scenarios", "no double counting", t_nodbl), ("scenarios", "export schema", t_exp)]: chk(g_, n_, f_)
    G.clear(); G.update(saved)
    if verbose:
        for g_, n_, ok, d in T: print(("PASS " if ok else "FAIL ") + g_ + ": " + n_ + " :: " + d)
    return T

# ------------------------------ project export ------------------------------
METH = """### Methodology
- **Data:** transactions (required), interactions and customer attributes (optional). Duplicates removed; invalid dates and missing values reported; cancelled rows excluded. A returned line contributes no revenue and no order.
- **Features:** recency (against an explicit analysis date), frequency (distinct orders), net spend, discount usage, category diversity, tenure, visit rate, email opens and click-through, ad clicks, cart abandonment (carts not matched by an order, an approximation), visit-to-order conversion, interaction recency, return rate. Missing tracking stays missing and is median-imputed only for clustering. Skewed features are log-transformed, winsorised at 1%/99% and standardised.
- **Exclusions:** identifiers are never features. Demographics describe and audit segments only.
- **Models:** K-Means, Agglomerative (ward) and Gaussian Mixture across a configurable k range. Composite score = 0.30 silhouette + 0.25 bootstrap stability (ARI on 85% subsamples) + 0.15 Davies-Bouldin + 0.15 minimum segment size + 0.15 interpretability.
- **IDs and names:** segment IDs (S1, S2...) are ordered by median net spend within a run. Names are deterministic descriptive labels derived after reviewing the statistics; an optional LLM may rewrite narratives from aggregated statistics only, validated before use. The LLM never assigns customers to segments.
- **Scenarios:** contacts = eligible x target (budget capped); orders = contacts x response x conversion; revenue = orders x AOV x (1 - discount); contribution = orders x AOV x (margin - discount); net = contribution - contacts x cost. Illustrative; needs controlled experiments.
- **Refresh:** re-run on new data, match new profiles to old by nearest centroid, treat IDs as run-specific, re-validate playbooks with holdouts.
"""
DICT = "# Data dictionary\n**Transactions** (sample_data.csv): customer_id, transaction_id, transaction_date, product_id, product_category, quantity, unit_price, transaction_value (after discount), discount_amount, discount_used (0/1), return_flag (0/1), acquisition_channel.\n**Interactions** (sample_interactions.csv): customer_id, event_timestamp, event_type (website_visit, email_open, email_click, ad_click, cart_add, purchase_completed), interaction_channel, campaign_id, product_category, session_id, session_duration_seconds.\n**Customers** (sample_customers.csv): customer_id, age_band, region, signup_date, loyalty_status, consent_status.\nAll sample data is synthetic and fictional. Column names can be remapped in the app (field=column).\n"
LIMITS = "# Limitations\nSegments are statistical patterns, not verified customer types; they can change after retraining. Revenue is observed within the window and is not lifetime value. Interaction tracking may be incomplete. Scenario outputs are illustrative and do not model cannibalisation. Demographic audits are partial and not a full fairness review. Incremental lift requires controlled experiments.\n"
LICENSE = "MIT License\n\nCopyright (c) 2026 MOSAIC AI contributors\n\nPermission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the \"Software\"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:\n\nThe above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.\n\nTHE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.\n"
README = """# MOSAIC AI — Customer Intelligence Studio
Every customer is a pattern. Find the ones that matter.

## Overview and business problem
Broad demographic segments send similar messages to customers who behave very differently. MOSAIC AI discovers behavioural segments from transaction and interaction data, explains them with measurable evidence, proposes differentiated playbooks and lets marketers compare hypothetical campaign outcomes. Bundled data is **fictional synthetic data**.

## Features
Schema mapping and validation, RFM plus engagement features, K-Means / Agglomerative / Gaussian Mixture comparison with stability checks, deterministic interpretation with optional validated LLM narratives, segment explorer and comparison, editable playbooks, campaign scenario simulator, customer lookup, tool-grounded AI strategist, quality and bias audit, exports and executive report.

## Architecture and data flow
@@@mermaid
flowchart LR
  A[CSV uploads or synthetic data] --> B[Schema mapping + validation]
  B --> C[Customer feature matrix]
  C --> D[Scaling + clustering comparison]
  D --> E[Segment profiles + narratives]
  E --> F[Playbook engine]
  F --> H[Scenario simulator]
  E --> I[AI strategist tools]
  F --> I
  H --> I
  I --> J[Gradio UI, exports, report]
@@@

## Technology stack
Python, Gradio, Plotly, scikit-learn, SciPy, pandas, NumPy.

## Installation and local launch
@@@
pip install -r requirements.txt
python app.py
@@@
Google Colab: paste `app.py` into one cell and run it.

## Data, upload and mapping
See `docs/data_dictionary.md`. Upload transactions (required), interactions and customers (optional). If auto-detection fails, enter mappings such as `customer_id=cust, transaction_value=amount`.

## Methodology
See `docs/methodology.md`. Segment IDs are ordered by net spend per run and are not stable across retrains.

## LLM configuration
Optional. Choose Anthropic or an OpenAI-compatible endpoint and enter a key in the app (session only), or set `ANTHROPIC_API_KEY` / `OPENAI_API_KEY`. Without a key a deterministic, transparent fallback is used. Only aggregated statistics are sent; the LLM never assigns segments.

## Tests
@@@
pip install pytest
pytest tests
@@@

## Screenshots
Add your own to `screenshots/` (see its README). None are included.

## Privacy, limitations and responsible AI
Use pseudonymous IDs (the app can hash IDs with a session salt). Never commit real customer data or API keys. Demographics are for auditing only. See `docs/limitations.md`.

## Deployment
Run locally, in Colab, or deploy `app.py` as a Hugging Face Space (SDK: Gradio) with your own credentials.

## Future improvements
Campaign-response data for calibrated simulations, uplift modelling, scheduled retraining with profile matching.
""".replace("@@@", "`" * 3)
TFILE = "from app import run_selftests\n\n\ndef test_%s():\n    res = [r for r in run_selftests(verbose=False) if r[0] == '%s']\n    assert res and all(r[2] for r in res), res\n"
def _src():
    try: return open(__file__, encoding="utf-8").read()
    except Exception: pass
    try:
        for s in list(get_ipython().user_ns.get("In") or [])[::-1]:
            if "MOSAIC AI — Customer Intelligence Studio" in s and "def main" in s: return s
    except Exception: pass
    return None
def build_project():
    root = "mosaic-ai-customer-intelligence"
    for d in ["tests", "docs", "screenshots"]: os.makedirs(os.path.join(root, d), exist_ok=True)
    src = _src()
    if src: open(root + "/app.py", "w", encoding="utf-8").write(src)
    SAMPLE_TX.to_csv(root + "/sample_data.csv", index=False); SAMPLE_IT.to_csv(root + "/sample_interactions.csv", index=False); SAMPLE_CU.to_csv(root + "/sample_customers.csv", index=False)
    for fn, tx in [("requirements.txt", "gradio>=4.44,<6\nplotly\nscikit-learn>=1.2\nscipy\nrequests\npandas\nnumpy\n"), ("README.md", README), ("LICENSE", LICENSE), (".gitignore", ".env\n*.key\nsecrets*\n__pycache__/\n.ipynb_checkpoints/\n*.zip\nmosaic_*.csv\nmosaic_*.json\nmosaic_*.md\nmosaic_*.html\n"),
                   ("docs/methodology.md", METH), ("docs/data_dictionary.md", DICT), ("docs/limitations.md", LIMITS), ("screenshots/README.md", "# Screenshots\nAdd your own screenshots of the running app here and reference them from the main README. No screenshots are included.\n"),
                   ("tests/test_data_validation.py", TFILE % ("data_validation", "data")), ("tests/test_feature_engineering.py", TFILE % ("feature_engineering", "features")), ("tests/test_segmentation.py", TFILE % ("segmentation", "segmentation")),
                   ("tests/test_strategy.py", TFILE % ("strategy", "strategy")), ("tests/test_scenarios.py", TFILE % ("scenarios", "scenarios"))]: open(os.path.join(root, fn), "w", encoding="utf-8").write(tx)
    z = "mosaic_ai_project.zip"
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, _, fs in os.walk(root):
            for f in fs: zf.write(os.path.join(dp, f))
    return z, bool(src)

# ------------------------------ UI ------------------------------
CSS = """
.gradio-container{background:#151A28!important;color:#F5F5F0!important;font-family:Inter,system-ui,-apple-system,sans-serif!important;max-width:1320px!important}
.hdr{display:flex;justify-content:space-between;align-items:flex-end;flex-wrap:wrap;gap:14px;padding:28px 30px;border-radius:14px;background:#22293A;border:1px solid #ffffff1a;border-left:4px solid #B8F36B;margin-bottom:12px}
.brand{display:flex;gap:16px;align-items:center}.logo{width:52px;height:52px;border-radius:10px;border:2px solid #B8F36B;color:#B8F36B;font-weight:800;font-size:22px;display:flex;align-items:center;justify-content:center}
.hdr h1{margin:0;font-size:30px;letter-spacing:3px;color:#F5F5F0;font-weight:700}.hdr h1 span{color:#B8F36B}.hdr .sub{color:#D6B9FF;font-size:13px;letter-spacing:2px;text-transform:uppercase;margin-top:2px}.hdr p{margin:6px 0 0;color:#A9B1C5;font-size:15px;font-style:italic}
.badges span{display:inline-block;margin:3px;padding:4px 11px;border-radius:6px;background:#ffffff0d;border:1px solid #ffffff1a;color:#A9B1C5;font-size:12px}
.kg{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin:10px 0}
.kc{background:#22293A;border:1px solid #ffffff14;border-radius:10px;padding:14px 16px}
.kl{color:#A9B1C5;font-size:11px;text-transform:uppercase;letter-spacing:1.2px}.kv{font-size:24px;font-weight:700;margin:6px 0 4px}.kd{color:#A9B1C5;font-size:12px}
.note{background:#22293A;padding:10px 14px;border-radius:8px;margin:8px 0;color:#F5F5F0;font-size:14px;line-height:1.5}
.ai{background:#22293A;border:1px solid #D6B9FF55;border-left:3px solid #D6B9FF;border-radius:8px;padding:14px 16px;margin:10px 0;font-size:14px;line-height:1.55}.ai ul{margin:4px 0 4px 18px}
.sec{font-size:15px;font-weight:700;color:#D6B9FF;margin:18px 0 6px;text-transform:uppercase;letter-spacing:1px}
.tw{overflow-x:auto;border-radius:8px;border:1px solid #ffffff14;margin:8px 0}table{border-collapse:collapse;width:100%;background:#22293A;font-size:13px}
th{background:#2c3550;color:#B8F36B;text-align:left;padding:9px 10px;font-weight:600}td{padding:8px 10px;border-top:1px solid #ffffff10;color:#F5F5F0}tr:hover td{background:#B8F36B0d}
button.primary{background:#B8F36B!important;color:#151A28!important;border:none!important;font-weight:700!important}
@media(max-width:640px){.hdr h1{font-size:22px}.kv{font-size:20px}.hdr{padding:18px}}
"""
HEADER = ("<div class='hdr'><div class='brand'><div class='logo'>MA</div><div><h1>MOSAIC <span>AI</span></h1><div class='sub'>Customer Intelligence Studio</div><p>Every customer is a pattern. Find the ones that matter.</p></div></div>"
          "<div class='badges'><span>Segments are patterns, not facts</span><span>Scenarios are illustrative</span><span>Core analytics need no API key</span></div></div>")

def build_ui(zip_path, files):
    with gr.Blocks(css=CSS, title="MOSAIC AI", theme=gr.themes.Base(primary_hue="lime", neutral_hue="slate")) as demo:
        gr.HTML(HEADER)
        with gr.Accordion("Data and settings (demo data is FICTIONAL and illustrative)", open=True):
            with gr.Row():
                up_tx = gr.File(label="Transactions CSV (leave empty for the demo)", file_types=[".csv"])
                up_it = gr.File(label="Interactions CSV (optional)", file_types=[".csv"])
                up_cu = gr.File(label="Customers CSV (optional)", file_types=[".csv"])
            with gr.Row():
                mp = gr.Textbox(label="Column mapping (only if auto-detect fails)", placeholder="customer_id=cust, transaction_value=amount, event_type=action", lines=2)
                ad = gr.Textbox(label="Analysis date (YYYY-MM-DD; blank = day after the last record)")
                act = gr.Slider(30, 365, 90, step=15, label="Active customer = ordered within (days)")
            with gr.Row():
                algos = gr.CheckboxGroup(["K-Means", "Agglomerative", "Gaussian Mixture"], value=["K-Means", "Agglomerative", "Gaussian Mixture"], label="Clustering algorithms to compare")
                kmin = gr.Slider(2, 10, 3, step=1, label="Min clusters")
                kmax = gr.Slider(2, 10, 7, step=1, label="Max clusters")
                kfix = gr.Number(0, label="Fixed k (0 = choose automatically)")
                pse = gr.Checkbox(True, label="Pseudonymise customer IDs (uploads)")
            gr.File(value=files, label="Sample files (synthetic): transactions, interactions, customers", file_count="multiple", interactive=False)
            run_btn = gr.Button("Analyse and segment", variant="primary")
            status = gr.HTML(note("Loading the fictional demo data and training segment models (about 30-60 seconds)...", "info"))
        with gr.Tabs():
            with gr.Tab("Overview"):
                with gr.Row():
                    fl_seg = gr.Dropdown([], multiselect=True, label="Segment")
                    fl_cat = gr.Dropdown([], multiselect=True, label="Product category")
                    fl_ach = gr.Dropdown([], multiselect=True, label="Acquisition channel")
                    fl_ich = gr.Dropdown([], multiselect=True, label="Interaction channel")
                with gr.Row():
                    fl_d1 = gr.Textbox(label="From date (YYYY-MM-DD)")
                    fl_d2 = gr.Textbox(label="To date (YYYY-MM-DD)")
                    fl_btn = gr.Button("Apply filters", variant="primary")
                d_html = gr.HTML()
                with gr.Row():
                    d1 = gr.Plot()
                    d2 = gr.Plot()
                with gr.Row():
                    d3 = gr.Plot()
                    d4 = gr.Plot()
                with gr.Row():
                    d5 = gr.Plot()
                    d6 = gr.Plot()
            with gr.Tab("Segment Explorer"):
                with gr.Row():
                    ex_seg = gr.Dropdown([], label="Segment")
                    ex_q = gr.Dropdown(EXQ, value=EXQ[0], label="Ask about this segment")
                    ex_ask = gr.Button("Explain", variant="primary")
                ex_ans = gr.Markdown()
                ex_html = gr.HTML()
                with gr.Row():
                    ex_f1 = gr.Plot()
                    ex_f2 = gr.Plot()
                ex_f3 = gr.Plot()
                gr.Markdown("#### Compare segments side by side")
                cmp_seg = gr.CheckboxGroup([], label="Segments to compare")
                cmp_html = gr.HTML()
                cmp_fig = gr.Plot()
            with gr.Tab("Marketing Playbooks"):
                pb_html = gr.HTML()
                pb_df = gr.Dataframe(interactive=True, wrap=True, label="Editable playbooks")
                pb_btn = gr.Button("Export edited playbooks (Markdown + CSV)", variant="primary")
                pb_file = gr.File(label="Playbook files", file_count="multiple", interactive=False)
            with gr.Tab("Campaign Simulator"):
                gr.Markdown("Compare a blanket campaign with a differentiated plan. **All outputs are illustrative projections.** Defaults are seeded from observed engagement proxies, not measured response. Controlled experiments are required to establish incremental lift.")
                with gr.Row():
                    sc_mar = gr.Number(45, label="Default margin % (used on reset)")
                    sc_cost = gr.Number(.5, label="Default cost per contact ($)")
                    sc_tgt = gr.Number(50, label="Default target %")
                    sc_rst = gr.Button("Reset to data-derived defaults")
                scn_tbl = gr.Dataframe(value=pd.DataFrame(columns=SCN_COLS), headers=SCN_COLS, interactive=True, wrap=True, label="Differentiated plan: editable assumptions per segment")
                with gr.Row():
                    bl_t = gr.Number(50, label="Blanket target %")
                    bl_r = gr.Number(5, label="Blanket response %")
                    bl_c = gr.Number(15, label="Blanket conversion %")
                    bl_a = gr.Number(60, label="Blanket AOV ($)")
                with gr.Row():
                    bl_d = gr.Number(10, label="Blanket discount %")
                    bl_m = gr.Number(45, label="Blanket margin %")
                    bl_k = gr.Number(.5, label="Blanket cost per contact ($)")
                    bl_match = gr.Checkbox(True, label="Match blanket contacts to the differentiated plan")
                with gr.Row():
                    lowm = gr.Slider(.2, 1, .6, step=.05, label="Low scenario: response multiplier")
                    highm = gr.Slider(1, 3, 1.4, step=.05, label="High scenario: response multiplier")
                    upl = gr.Slider(0, 50, 0, step=1, label="Assumed relevance uplift for tailored messaging (%)")
                    cons = gr.Checkbox(True, label="Only contact consented customers")
                sc_btn = gr.Button("Run simulation", variant="primary")
                sc_html = gr.HTML()
                sc_fig = gr.Plot()
                sc_file = gr.File(label="Scenario results (CSV)", interactive=False)
            with gr.Tab("AI Strategist"):
                gr.Markdown("Questions are answered by analytical tools over the current segments. Without an API key a transparent rule-based assistant answers; with a key an LLM phrases the answer and its numbers are checked against the tool results. Only aggregated statistics are sent, never customer-level data.")
                ag_q = gr.Textbox(label="Ask a question", placeholder="Which segment should receive the next campaign? / What changes if we reduce discount spending by 20%? / Which segments have high engagement but weak conversion?")
                with gr.Row():
                    prov = gr.Radio(["Anthropic", "OpenAI-compatible"], value="Anthropic", label="LLM provider (optional)")
                    mdl = gr.Textbox(label="Model", value="claude-sonnet-5-5")
                    base = gr.Textbox(label="Base URL (OpenAI-compatible only)", placeholder="https://api.openai.com/v1")
                    key = gr.Textbox(label="API key (optional, session only, never stored)", type="password")
                with gr.Row():
                    ag_btn = gr.Button("Ask", variant="primary")
                    llm_btn = gr.Button("Refine segment narratives with LLM (aggregated stats only)")
                llm_status = gr.HTML()
                ag_out = gr.Markdown()
            with gr.Tab("Customer Lookup"):
                lk_id = gr.Dropdown([], label="Customer ID (top customers listed; you can type any ID)", allow_custom_value=True)
                lk_html = gr.HTML()
            with gr.Tab("Quality & Methodology"):
                q_html = gr.HTML()
                t_html = gr.HTML()
                gr.Markdown(METH)
                gr.Markdown("### Export and publish\nDownload the ZIP, unzip it, create an empty GitHub repository and upload the contents (Add file, then Upload files). Never upload real customer data or API keys.")
                gr.File(value=zip_path, label="Download GitHub-ready project (ZIP)", interactive=False)
                with gr.Row():
                    rep_btn = gr.Button("Generate executive report", variant="primary")
                    exp_btn = gr.Button("Export CSV / JSON / Markdown files", variant="primary")
                rep_msg = gr.HTML()
                rep_file = gr.File(label="Executive summary (HTML)", interactive=False)
                exp_msg = gr.HTML()
                exp_files = gr.File(label="Exports", file_count="multiple", interactive=False)
        DASH_IN = [fl_seg, fl_d1, fl_d2, fl_cat, fl_ach, fl_ich]; DASH_OUT = [d_html, d1, d2, d3, d4, d5, d6]
        REF_OUT = [fl_seg, fl_cat, fl_ach, fl_ich, ex_seg, cmp_seg, lk_id]; EX_OUT = [ex_html, ex_f1, ex_f2, ex_f3]
        SCN_IN = [scn_tbl, bl_t, bl_r, bl_c, bl_a, bl_d, bl_m, bl_k, bl_match, lowm, highm, upl, cons]; SCN_OUT = [sc_html, sc_fig, sc_file]
        RST_IN = [sc_mar, sc_cost, sc_tgt]; RST_OUT = [scn_tbl, bl_t, bl_r, bl_c, bl_a, bl_d, bl_m]
        RUN_IN = [up_tx, up_it, up_cu, mp, ad, act, algos, kmin, kmax, kfix, pse]
        def chain(ev):
            e = ev.then(refresh, None, REF_OUT).then(scn_reset, RST_IN, RST_OUT).then(render_dash, DASH_IN, DASH_OUT)
            e = e.then(render_explorer, ex_seg, EX_OUT).then(render_compare, cmp_seg, [cmp_html, cmp_fig]).then(render_playbooks, None, [pb_html, pb_df])
            return e.then(render_scn, SCN_IN, SCN_OUT).then(render_quality, None, [q_html, t_html])
        chain(run_btn.click(run_all, RUN_IN, [status], show_progress="full"))
        chain(demo.load(run_all, RUN_IN, [status]))
        fl_btn.click(render_dash, DASH_IN, DASH_OUT)
        ex_ask.click(explain_seg, [ex_seg, ex_q], ex_ans)
        ex_seg.change(render_explorer, ex_seg, EX_OUT)
        cmp_seg.change(render_compare, cmp_seg, [cmp_html, cmp_fig])
        pb_btn.click(export_pb, pb_df, pb_file)
        sc_rst.click(scn_reset, RST_IN, RST_OUT).then(render_scn, SCN_IN, SCN_OUT)
        sc_btn.click(render_scn, SCN_IN, SCN_OUT, show_progress="full")
        ag_btn.click(ask, [ag_q, prov, mdl, base, key], ag_out, show_progress="full")
        ag_q.submit(ask, [ag_q, prov, mdl, base, key], ag_out)
        llm_btn.click(refine_llm, [prov, mdl, base, key], llm_status).then(render_explorer, ex_seg, EX_OUT).then(render_playbooks, None, [pb_html, pb_df])
        lk_id.change(render_lookup, lk_id, lk_html)
        rep_btn.click(make_report, pb_df, [rep_file, rep_msg])
        exp_btn.click(export_all, pb_df, [exp_files, exp_msg])
    return demo

def main():
    zp, saved = build_project()
    for n_, d_ in [("mosaic_sample_transactions.csv", SAMPLE_TX), ("mosaic_sample_interactions.csv", SAMPLE_IT), ("mosaic_sample_customers.csv", SAMPLE_CU)]: d_.to_csv(n_, index=False)
    print("=" * 78); print("MOSAIC AI starting. Demo data is FICTIONAL and illustrative. Core analytics need no API key."); print("Running self-tests (about 20-40 seconds):")
    TESTS[:] = run_selftests(verbose=True)
    demo = build_ui(zp, ["mosaic_sample_transactions.csv", "mosaic_sample_interactions.csv", "mosaic_sample_customers.csv"])
    print("Project ZIP:", zp, "| standalone app.py included:", "yes" if saved else "NO (create app.py yourself by pasting this cell into it)")
    print("The demo analyses automatically (about 30-60 seconds); tabs fill in when it finishes. Keys: use Colab Secrets (ANTHROPIC_API_KEY / OPENAI_API_KEY). The public Gradio link is open to anyone: do not share it with real data loaded.")
    print("=" * 78); demo.queue().launch(debug=False, show_error=True)

if __name__ == "__main__" or "google.colab" in sys.modules:
    main()