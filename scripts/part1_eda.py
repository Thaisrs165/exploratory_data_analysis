"""Part 1 - EDA stages 0-3 for UCI Bank Marketing (bank-full.csv).

Computes every number used in docs/projects/eda/part1_draft.md, saves the
figures and machine-readable outputs, and renders the draft from
scripts/templates/part1_draft_template.md. No model is trained.

Usage (from the repository root, after scripts/part1_download_data.py):
    python scripts/part1_eda.py

Outputs:
    docs/projects/eda/part1_figures/   numbered PNG figures
    docs/projects/eda/part1_outputs/   CSV / JSON tables, split indices, feature lists
    docs/projects/eda/part1_draft.md   rendered draft (stages 0-3)
"""

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "bank-full.csv"
DOCS = ROOT / "docs" / "projects" / "eda"
FIG_DIR = DOCS / "part1_figures"
OUT_DIR = DOCS / "part1_outputs"
TEMPLATE = ROOT / "scripts" / "templates" / "part1_draft_template.md"
DRAFT = DOCS / "part1_draft.md"

TARGET = "y"
LEAKAGE = ["duration"]
EXPECTED_NUM = ["age", "balance", "day", "duration", "campaign", "pdays", "previous"]
EXPECTED_CAT = ["job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "poutcome"]
MONTHS = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]

# Validated categorical slots 1-2 (dataviz reference palette); ink and grid tokens.
C_NO, C_YES = "#2a78d6", "#eb6834"
C_MAIN, C_MUTED = "#2a78d6", "#a3a29c"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DIVERGING = LinearSegmentedColormap.from_list("div", [C_YES, "#f1f0ec", C_NO])
Y_PALETTE = {"no": C_NO, "yes": C_YES}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.titlesize": 12, "figure.titleweight": "bold",
})

N = {}       # scalar values for the template
T = {}       # markdown tables for the template
FIGS = []    # (number, file, title)


# ----------------------------------------------------------------- helpers
def pct(x, d=2):
    return f"{100 * x:.{d}f}%"


def fmt_int(x):
    return f"{int(x):,}"


def md_table(df, index=False, floatfmt=".2f"):
    return df.to_markdown(index=index, floatfmt=floatfmt)


def save_fig(fig, slug, title):
    n = len(FIGS) + 1
    name = f"fig{n:02d}_{slug}.png"
    fig.suptitle(f"Figure {n}. {title}", x=0.01, ha="left")
    fig.savefig(FIG_DIR / name, facecolor="white")
    plt.close(fig)
    FIGS.append((n, name, title))
    N[f"fig_{slug}"] = n
    N[f"fig_{slug}_md"] = f"![Figure {n}. {title}](part1_figures/{name})"


def wilson(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return centre - half, centre + half


def iqr_bounds(s):
    q1, q3 = s.quantile([0.25, 0.75])
    return q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)


def write_json(obj, name):
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


# ----------------------------------------------------------------- load
def load():
    df = pd.read_csv(DATA, sep=";")
    log = json.loads((DATA.parent / "part1_download_log.json").read_text())
    N["sha256"] = log["files"]["bank-full.csv"]["sha256"]
    N["downloaded_at"] = log["downloaded_at_utc"]
    N["file_bytes"] = fmt_int(log["files"]["bank-full.csv"]["bytes"])
    return df


# ----------------------------------------------------------------- stage 0/1A
def schema(df):
    num = df.select_dtypes(include="number").columns.tolist()
    cat = [c for c in df.columns if c not in num and c != TARGET]
    assert num == EXPECTED_NUM, num
    assert cat == EXPECTED_CAT, cat
    assert set(df[TARGET].unique()) == {"no", "yes"}
    N.update(rows=fmt_int(df.shape[0]), cols=df.shape[1], n_num=len(num), n_cat=len(cat),
             n_inputs=len(num) + len(cat))
    num_ret = [c for c in num if c not in LEAKAGE]
    cat_ret = [c for c in cat if c not in LEAKAGE]
    N.update(n_num_ret=len(num_ret), n_cat_ret=len(cat_ret), n_pred_ret=len(num_ret) + len(cat_ret),
             cols_after=len(num_ret) + len(cat_ret) + 1,
             num_ret_list=", ".join(f"`{c}`" for c in num_ret),
             cat_ret_list=", ".join(f"`{c}`" for c in cat_ret))

    meaning = {
        "age": ("Client age", "years"),
        "job": ("Type of job", "-"),
        "marital": ("Marital status (`divorced` = divorced or widowed)", "-"),
        "education": ("Education level", "-"),
        "default": ("Has credit in default?", "-"),
        "balance": ("Average yearly balance", "euros"),
        "housing": ("Has housing loan?", "-"),
        "loan": ("Has personal loan?", "-"),
        "contact": ("Contact communication type (last contact, current campaign)", "-"),
        "day": ("Day of month of the last contact", "day (1-31)"),
        "month": ("Month of year of the last contact", "-"),
        "duration": ("Duration of the last contact", "seconds"),
        "campaign": ("Contacts performed during this campaign for this client, incl. the last one", "count"),
        "pdays": ("Days since the client was last contacted in a previous campaign; -1 = not previously contacted", "days"),
        "previous": ("Contacts performed before this campaign for this client", "count"),
        "poutcome": ("Outcome of the previous marketing campaign", "-"),
        "y": ("Target: has the client subscribed to a term deposit?", "-"),
    }
    rows = []
    for c in df.columns:
        role = "target" if c == TARGET else ("numerical" if c in num else "categorical")
        status = "excluded (leakage)" if c in LEAKAGE else ("target" if c == TARGET else "retained")
        if c in num:
            values = f"{df[c].min():,} to {df[c].max():,}"
        else:
            values = ", ".join(f"`{v}`" for v in sorted(df[c].unique(), key=lambda v: MONTHS.index(v) if c == "month" else v))
        rows.append({"column": f"`{c}`", "meaning": meaning[c][0], "role": role,
                     "pandas dtype": str(df[c].dtype), "unit": meaning[c][1],
                     "observed values / range": values, "Part 1 status": status})
    dd = pd.DataFrame(rows)
    dd.assign(column=dd["column"].str.strip("`")).to_csv(OUT_DIR / "data_dictionary.csv", index=False)
    T["data_dictionary"] = md_table(dd)
    return num, cat, num_ret, cat_ret


# ----------------------------------------------------------------- stage 1B
def quality(df, num, cat):
    n = len(df)
    # Missing values: true NaN and the literal "unknown" category.
    miss = pd.DataFrame({
        "column": df.columns,
        "NaN count": [int(df[c].isna().sum()) for c in df.columns],
        "\"unknown\" count": [int((df[c] == "unknown").sum()) if c in cat else 0 for c in df.columns],
    })
    miss["NaN %"] = miss["NaN count"] / n * 100
    miss["\"unknown\" %"] = miss["\"unknown\" count"] / n * 100
    miss = miss[["column", "NaN count", "NaN %", "\"unknown\" count", "\"unknown\" %"]]
    miss.to_csv(OUT_DIR / "missing_and_unknown.csv", index=False)
    T["missing"] = md_table(miss.assign(column=miss["column"].map(lambda c: f"`{c}`")))
    N["total_nan"] = int(df.isna().sum().sum())
    unk = miss[miss["\"unknown\" count"] > 0]
    N["unknown_cols"] = ", ".join(f"`{c}`" for c in unk["column"])
    for _, r in unk.iterrows():
        N[f"unk_{r['column']}"] = f"{fmt_int(r.iloc[3])} ({r.iloc[4]:.2f}%)"
    N["rows_any_unknown"] = fmt_int((df[cat] == "unknown").any(axis=1).sum())
    N["rows_any_unknown_pct"] = pct((df[cat] == "unknown").any(axis=1).mean())
    excl_pout = [c for c in cat if c != "poutcome"]
    N["rows_any_unknown_ex_pout"] = fmt_int((df[excl_pout] == "unknown").any(axis=1).sum())
    N["rows_any_unknown_ex_pout_pct"] = pct((df[excl_pout] == "unknown").any(axis=1).mean())

    # Duplicates.
    N["dup_full"] = int(df.duplicated().sum())
    pred = df.drop(columns=LEAKAGE)
    N["dup_no_dur"] = int(pred.duplicated().sum())
    N["dup_no_dur_rows_involved"] = int(pred.duplicated(keep=False).sum())
    feat_cols = [c for c in pred.columns if c != TARGET]
    grp = pred[pred.duplicated(subset=feat_cols, keep=False)].groupby(feat_cols)[TARGET].nunique()
    N["dup_feat_groups"] = len(grp)
    N["dup_feat_conflicts"] = int((grp > 1).sum())
    N["dup_feat_only"] = int(pred.duplicated(subset=feat_cols).sum())

    # pdays sentinel.
    sent = df["pdays"] == -1
    prev0 = df["previous"] == 0
    N["pdays_neg1"] = fmt_int(sent.sum())
    N["pdays_neg1_pct"] = pct(sent.mean())
    N["pdays_other_neg"] = int((df["pdays"] < -1).sum())
    N["pdays_zero"] = int((df["pdays"] == 0).sum())
    N["pdays_pos_min"] = int(df.loc[~sent, "pdays"].min())
    N["pdays_pos_max"] = int(df.loc[~sent, "pdays"].max())
    N["prev0"] = fmt_int(prev0.sum())
    N["sent_prev_mismatch"] = int((sent != prev0).sum())
    ct = pd.crosstab(df["pdays"].eq(-1).map({True: "pdays = -1", False: "pdays >= 1"}),
                     df["poutcome"])
    ct.to_csv(OUT_DIR / "pdays_sentinel_vs_poutcome.csv")
    T["pdays_poutcome"] = md_table(ct.reset_index().rename(columns={"pdays": "pdays group"}), floatfmt=".0f")
    N["pout_unknown_contacted"] = int(((df["poutcome"] == "unknown") & ~sent).sum())
    N["pout_known_not_contacted"] = int(((df["poutcome"] != "unknown") & sent).sum())
    N["sent_rate_y"] = pct((df.loc[sent, TARGET] == "yes").mean())
    N["nonsent_rate_y"] = pct((df.loc[~sent, TARGET] == "yes").mean())
    N["pdays_mean_raw"] = f"{df['pdays'].mean():.2f}"
    N["pdays_mean_contacted"] = f"{df.loc[~sent, 'pdays'].mean():.2f}"

    # Impossible / inconsistent / implausible checks.
    days_in_month = {"feb": 29, "apr": 30, "jun": 30, "sep": 30, "nov": 30}
    checks = [
        ("age < 18 or > 100", int(((df.age < 18) | (df.age > 100)).sum()), "impossible for a bank client contact list", "none"),
        ("age > 90", int((df.age > 90).sum()), "plausible but extreme", "keep"),
        ("day outside 1-31", int(((df.day < 1) | (df.day > 31)).sum()), "impossible", "none"),
        ("day exceeds days in its month (e.g. 31 apr, 30 feb)", int((df.day > df.month.map(days_in_month).fillna(31)).sum()), "impossible calendar date", "none"),
        ("campaign < 1 (must include the last contact)", int((df.campaign < 1).sum()), "impossible by definition", "none"),
        ("previous < 0", int((df.previous < 0).sum()), "impossible", "none"),
        ("pdays < -1, or pdays = 0", int(((df.pdays < -1) | (df.pdays == 0)).sum()), "impossible / undefined code", "none"),
        ("pdays = -1 while previous > 0 (or the reverse)", N["sent_prev_mismatch"], "inconsistent", "none"),
        ("poutcome = unknown while previous > 0", N["pout_unknown_contacted"], "inconsistent: a previous contact exists but its outcome is unrecorded", "keep; treat as category"),
        ("poutcome known while previous = 0", N["pout_known_not_contacted"], "inconsistent", "none"),
        ("balance < 0", int((df.balance < 0).sum()), "valid: overdrawn account (balance is not bounded at 0)", "keep"),
        ("duration = 0 s", int((df.duration == 0).sum()), "consistent with no conversation; all have y = no", "moot: duration excluded"),
        ("job = student and age > 40", int(((df.job == "student") & (df.age > 40)).sum()), "implausible but possible", "keep"),
        ("job = retired and age < 40", int(((df.job == "retired") & (df.age < 40)).sum()), "implausible but possible (early/disability retirement)", "keep"),
    ]
    chk = pd.DataFrame(checks, columns=["check", "rows", "interpretation", "action"])
    chk.to_csv(OUT_DIR / "consistency_checks.csv", index=False)
    T["checks"] = md_table(chk)
    N["neg_balance"] = fmt_int((df.balance < 0).sum())
    N["neg_balance_pct"] = pct((df.balance < 0).mean())
    N["age_gt90"] = int((df.age > 90).sum())
    N["dur0"] = int((df.duration == 0).sum())
    N["dur0_all_no"] = "yes" if (df.loc[df.duration == 0, TARGET] == "no").all() else "no"

    # Identifiers and constants.
    card = pd.DataFrame({"column": df.columns,
                         "unique values": [df[c].nunique() for c in df.columns]})
    card["unique / rows"] = card["unique values"] / n
    N["max_unique_ratio_col"] = card.loc[card["unique / rows"].idxmax(), "column"]
    N["max_unique_ratio"] = f"{card['unique / rows'].max():.3f}"
    N["max_unique_values"] = fmt_int(card["unique values"].max())
    N["n_constant"] = int((card["unique values"] <= 1).sum())
    top_share = {c: df[c].value_counts(normalize=True).iloc[0] for c in df.columns}
    N["most_dominant_col"] = max(top_share, key=top_share.get)
    N["most_dominant_share"] = pct(max(top_share.values()))
    N["default_yes"] = fmt_int((df["default"] == "yes").sum())
    N["default_yes_pct"] = pct((df["default"] == "yes").mean())

    # Leakage: duration.
    yb = (df[TARGET] == "yes").astype(int)
    N["dur_spearman_y"] = f"{df['duration'].corr(yb, method='spearman'):.3f}"
    bins = [-1, 0, 60, 120, 300, 600, df["duration"].max()]
    labels = ["0", "1-60", "61-120", "121-300", "301-600", f"601-{df['duration'].max()}"]
    dur = df.groupby(pd.cut(df["duration"], bins=bins, labels=labels), observed=False)[TARGET] \
        .agg(rows="size", subscribed=lambda s: int((s == "yes").sum()))
    dur["subscription rate %"] = dur["subscribed"] / dur["rows"] * 100
    dur.index.name = "duration (s)"
    dur.to_csv(OUT_DIR / "duration_leakage_by_bin.csv")
    T["duration"] = md_table(dur.reset_index())
    N["other_max_abs_spearman_y"] = max(
        (abs(df[c].corr(yb, method="spearman")), c) for c in num if c not in LEAKAGE)
    N["other_max_abs_spearman_y_val"] = f"{N['other_max_abs_spearman_y'][0]:.3f}"
    N["other_max_abs_spearman_y_col"] = N.pop("other_max_abs_spearman_y")[1]

    # Leakage-adjacent: row order is chronological (May 2008 - Nov 2010).
    dec = pd.qcut(np.arange(n), 10, labels=[f"D{i}" for i in range(1, 11)])
    od = pd.DataFrame({
        "row decile (file order)": dec,
        "yes": yb.values,
        "contact unknown": (df["contact"] == "unknown").values,
        "poutcome unknown": (df["poutcome"] == "unknown").values,
    }).groupby("row decile (file order)", observed=False).agg(
        rows=("yes", "size"), **{"subscription rate %": ("yes", "mean"),
                                  "contact = unknown %": ("contact unknown", "mean"),
                                  "poutcome = unknown %": ("poutcome unknown", "mean")})
    for c in od.columns[1:]:
        od[c] = od[c] * 100
    od.to_csv(OUT_DIR / "row_order_drift_by_decile.csv")
    T["row_order"] = md_table(od.reset_index(), floatfmt=".1f")
    N["order_rate_first"] = f"{od['subscription rate %'].iloc[0]:.1f}%"
    N["order_rate_last"] = f"{od['subscription rate %'].iloc[-1]:.1f}%"
    first_known = int(np.argmax((df["contact"] != "unknown").values))
    N["contact_unknown_first_rows"] = fmt_int(first_known)
    N["contact_unknown_in_first_30pct"] = pct(
        (df["contact"].iloc[: int(0.3 * n)] == "unknown").sum() / (df["contact"] == "unknown").sum())
    N["contact_unknown_rate_y"] = pct((df.loc[df.contact == "unknown", TARGET] == "yes").mean())
    N["contact_known_rate_y"] = pct((df.loc[df.contact != "unknown", TARGET] == "yes").mean())


# ----------------------------------------------------------------- stage 1C/1D
def target_and_split(df, num_ret, cat_ret):
    vc = df[TARGET].value_counts()
    tgt = pd.DataFrame({"class": vc.index, "count": vc.values, "percent": vc.values / len(df) * 100})
    tgt.to_csv(OUT_DIR / "target_distribution.csv", index=False)
    T["target"] = md_table(tgt)
    N["n_no"], N["n_yes"] = fmt_int(vc["no"]), fmt_int(vc["yes"])
    N["pct_no"], N["pct_yes"] = pct(vc["no"] / len(df)), pct(vc["yes"] / len(df))
    N["imbalance_ratio"] = f"{vc['no'] / vc['yes']:.2f}"
    N["base_rate"] = pct(vc["yes"] / len(df))
    N["majority_acc"] = pct(vc["no"] / len(df))

    features = num_ret + cat_ret
    X = df[features]
    y = (df[TARGET] == "yes").astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=RANDOM_STATE)
    split = pd.Series("train", index=df.index)
    split.loc[X_te.index] = "test"
    pd.DataFrame({"row_id": df.index, "split": split.values}).to_csv(
        OUT_DIR / "train_test_split_indices.csv", index=False)

    rows = []
    for name, yy in [("full", y), ("train", y_tr), ("test", y_te)]:
        rows.append({"set": name, "rows": len(yy), "share of full %": len(yy) / len(df) * 100,
                     "y = no": int((yy == 0).sum()), "y = yes": int((yy == 1).sum()),
                     "y = yes %": yy.mean() * 100})
    sp = pd.DataFrame(rows)
    sp.to_csv(OUT_DIR / "train_test_split_summary.csv", index=False)
    T["split"] = md_table(sp, floatfmt=".2f")
    N.update(n_train=fmt_int(len(y_tr)), n_test=fmt_int(len(y_te)),
             train_yes=fmt_int(y_tr.sum()), test_yes=fmt_int(y_te.sum()),
             train_no=fmt_int((y_tr == 0).sum()), test_no=fmt_int((y_te == 0).sum()),
             train_yes_pct=pct(y_tr.mean()), test_yes_pct=pct(y_te.mean()))

    split_spec = {
        "function": "sklearn.model_selection.train_test_split",
        "test_size": TEST_SIZE, "train_size": 1 - TEST_SIZE,
        "stratify": "y (encoded yes=1, no=0)", "random_state": RANDOM_STATE, "shuffle": True,
        "input_file": "data/raw/bank-full.csv", "input_sha256": N["sha256"],
        "row_id": "0-based row position in bank-full.csv (pandas default index after read_csv(sep=';'))",
        "indices_file": "docs/projects/eda/part1_outputs/train_test_split_indices.csv",
        "X_columns_order": features,
        "sizes": {r["set"]: {"rows": r["rows"], "no": r["y = no"], "yes": r["y = yes"]} for r in rows},
        "reproduce": ("X = df[numerical + categorical]; y = (df['y'] == 'yes').astype(int); "
                      "train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)"),
    }
    write_json(split_spec, "train_test_split.json")
    write_json({
        "target": TARGET, "target_encoding": {"no": 0, "yes": 1},
        "excluded_columns": {"duration": "leakage: known only during/after the call; UCI says discard for realistic prediction"},
        "numerical": num_ret, "categorical": cat_ret,
        "categorical_levels": {c: sorted(df[c].unique().tolist()) for c in cat_ret},
        "month_order": MONTHS,
        "sentinels": {"pdays": {"value": -1, "meaning": "client not previously contacted",
                                "equivalent_to": "previous == 0"}},
        "unknown_is_category_in": [c for c in cat_ret if (df[c] == "unknown").any()],
    }, "feature_lists.json")

    train = df.loc[X_tr.index].copy()
    N["eda_set"] = "training set"
    return train


# ----------------------------------------------------------------- stage 2A
def univariate_numeric(train, num_ret):
    rows = []
    for c in num_ret:
        s = train[c]
        lo, hi = iqr_bounds(s)
        rows.append({"feature": c, "mean": s.mean(), "median": s.median(), "std": s.std(),
                     "min": s.min(), "Q1": s.quantile(0.25), "Q3": s.quantile(0.75), "max": s.max(),
                     "skew": s.skew(), "mode": s.mode().iloc[0], "mode share %": (s == s.mode().iloc[0]).mean() * 100,
                     "IQR outliers": int(((s < lo) | (s > hi)).sum()),
                     "IQR outliers %": ((s < lo) | (s > hi)).mean() * 100})
    contacted = train.loc[train["pdays"] != -1]
    for c in ["pdays", "previous"]:
        s = contacted[c]
        lo, hi = iqr_bounds(s)
        rows.append({"feature": f"{c} (previously contacted only)", "mean": s.mean(), "median": s.median(),
                     "std": s.std(), "min": s.min(), "Q1": s.quantile(0.25), "Q3": s.quantile(0.75),
                     "max": s.max(), "skew": s.skew(), "mode": s.mode().iloc[0],
                     "mode share %": (s == s.mode().iloc[0]).mean() * 100,
                     "IQR outliers": int(((s < lo) | (s > hi)).sum()),
                     "IQR outliers %": ((s < lo) | (s > hi)).mean() * 100})
    st = pd.DataFrame(rows)
    st.to_csv(OUT_DIR / "numerical_summary_train.csv", index=False)
    T["num_stats"] = md_table(st.drop(columns=["mode", "mode share %"]), floatfmt=".2f")
    T["num_modes"] = md_table(st[["feature", "mode", "mode share %", "skew"]], floatfmt=".2f")
    S = st.set_index("feature")
    for c in S.index:
        k = c.split(" ")[0] + ("_c" if "only" in c else "")
        for col, key in [("mean", "mean"), ("median", "med"), ("std", "std"), ("Q1", "q1"), ("Q3", "q3"),
                         ("skew", "skew"), ("min", "min"), ("max", "max"), ("mode share %", "modesh"),
                         ("IQR outliers %", "outp")]:
            N[f"{k}_{key}"] = f"{S.loc[c, col]:,.2f}" if isinstance(S.loc[c, col], float) else f"{S.loc[c, col]:,}"
        N[f"{k}_mode"] = f"{S.loc[c, 'mode']:,}"
        N[f"{k}_out"] = fmt_int(S.loc[c, "IQR outliers"])
    top_pd = contacted["pdays"].value_counts().head(5)
    N["pdays_top"] = ", ".join(f"{d} ({v:,})" for d, v in top_pd.items())
    N["n_contacted_train"] = fmt_int(len(contacted))
    N["contacted_train_pct"] = pct(len(contacted) / len(train))
    N["bal_neg_train_pct"] = pct((train.balance < 0).mean())
    N["bal_zero_train_pct"] = pct((train.balance == 0).mean())
    N["bal_p99"] = fmt_int(train.balance.quantile(0.99))
    N["bal_p01"] = fmt_int(train.balance.quantile(0.01))
    N["camp_le3_pct"] = pct((train.campaign <= 3).mean())
    N["camp_gt10"] = fmt_int((train.campaign > 10).sum())
    top_days = train["day"].value_counts().head(3)
    N["day_top"] = ", ".join(f"{d} ({v / len(train) * 100:.1f}%)" for d, v in top_days.items())
    N["age_25_60_pct"] = pct(train.age.between(25, 60).mean())

    # Figure 1: shape of the continuous-ish features.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    lo, hi = train.balance.quantile([0.01, 0.99])
    specs = [("age", train.age, np.arange(train.age.min(), train.age.max() + 2, 1), "Age (years)"),
             ("balance", train.balance.clip(lo, hi), 60,
              f"Average yearly balance (euros), clipped to P1-P99 [{lo:,.0f}, {hi:,.0f}]"),
             ("day", train.day, np.arange(0.5, 32.5, 1), "Day of month of last contact")]
    for ax, (c, s, bins, xl) in zip(axes, specs):
        ax.hist(s, bins=bins, color=C_MAIN, edgecolor="white", linewidth=0.4)
        ax.axvline(train[c].mean(), color=INK, ls="--", lw=1.2, label=f"mean = {train[c].mean():,.1f}")
        ax.axvline(train[c].median(), color=C_YES, ls="-", lw=1.6, label=f"median = {train[c].median():,.1f}")
        ax.set(xlabel=xl, ylabel="Clients (count)", title=c)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "hist_age_balance_day", "Distributions of age, balance and day (training set)")

    # Figure 2: count-type features with sentinel handled.
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    vc = train.campaign.value_counts().sort_index()
    axes[0].bar(vc.index, vc.values, color=C_MAIN, width=0.8)
    axes[0].set(yscale="log", xlabel="Contacts in current campaign", ylabel="Clients (count, log scale)", title="campaign")
    axes[1].hist(contacted.pdays, bins=60, color=C_MAIN, edgecolor="white", linewidth=0.4,
                 label=f"pdays >= 1 (n = {len(contacted):,})")
    axes[1].set(xlabel="Days since last contact in a previous campaign", ylabel="Clients (count)",
                title="pdays (previously contacted only)")
    axes[1].plot([], [], " ", label=f"pdays = -1 excluded (n = {(train.pdays == -1).sum():,}, "
                                     f"{(train.pdays == -1).mean() * 100:.1f}%)")
    axes[1].legend(loc="upper right", fontsize=8)
    cap = 20
    vp = contacted.previous.clip(upper=cap).value_counts().sort_index()
    axes[2].bar(vp.index, vp.values, color=C_MAIN, width=0.8)
    axes[2].set_xticks([1, 5, 10, 15, cap], ["1", "5", "10", "15", f"{cap}+"])
    axes[2].set(yscale="log", xlabel=f"Contacts before this campaign ({cap}+ = {cap} or more)",
                ylabel="Clients (count, log scale)", title="previous (previously contacted only)")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "counts_campaign_pdays_previous", "Count-type features: campaign, pdays and previous (training set)")

    # Figure 3: boxplots with IQR outlier counts.
    fig, axes = plt.subplots(2, 3, figsize=(13, 5.2))
    for ax, c in zip(axes.flat, num_ret):
        s = train[c]
        ax.boxplot(s, vert=False, widths=0.5, patch_artist=True,
                   boxprops=dict(facecolor="#d6e6f8", edgecolor=C_MAIN),
                   medianprops=dict(color=C_YES, lw=2), whiskerprops=dict(color=C_MAIN),
                   capprops=dict(color=C_MAIN),
                   flierprops=dict(marker="o", ms=2.5, mfc=C_MUTED, mec="none", alpha=0.5))
        if c in ("balance", "campaign", "pdays", "previous"):
            ax.set_xscale("symlog", linthresh=1 if c != "balance" else 100)
        n_out = int(S.loc[c, "IQR outliers"])
        ax.set(yticks=[], title=f"{c}: {n_out:,} IQR outliers ({S.loc[c, 'IQR outliers %']:.1f}%)",
               xlabel=f"{c}" + (" (symlog scale)" if c in ("balance", "campaign", "pdays", "previous") else ""))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "boxplots_numerical", "Boxplots and 1.5 x IQR outliers of retained numerical features (training set)")
    return st


# ----------------------------------------------------------------- stage 2B
def univariate_categorical(train, cat_ret):
    n = len(train)
    freq_rows, card_rows = [], []
    for c in cat_ret:
        vc = train[c].value_counts()
        if c == "month":
            vc = vc.reindex(MONTHS)
        for lvl, cnt in vc.items():
            freq_rows.append({"feature": c, "level": lvl, "count": int(cnt), "percent": cnt / n * 100})
        card_rows.append({"feature": c, "cardinality": len(vc),
                          "most frequent": f"{vc.idxmax()} ({vc.max() / n * 100:.1f}%)",
                          "least frequent": f"{vc.idxmin()} ({vc.min() / n * 100:.2f}%)",
                          "levels < 1%": ", ".join(vc[vc / n < 0.01].index) or "-",
                          "levels < 5%": int((vc / n < 0.05).sum()),
                          "\"unknown\" %": (train[c] == "unknown").mean() * 100})
    fr = pd.DataFrame(freq_rows)
    fr.to_csv(OUT_DIR / "categorical_frequencies_train.csv", index=False)
    cd = pd.DataFrame(card_rows)
    cd.to_csv(OUT_DIR / "categorical_cardinality_train.csv", index=False)
    T["cat_card"] = md_table(cd, floatfmt=".2f")
    blocks = []
    for c in cat_ret:
        sub = fr[fr.feature == c].drop(columns="feature")
        blocks.append(f"**`{c}`** (cardinality {len(sub)})\n\n" + md_table(sub, floatfmt=".2f"))
    T["cat_freq"] = "\n\n".join(blocks)
    N["total_dummies"] = int(cd.cardinality.sum())
    N["rare_levels"] = "; ".join(f"`{r.feature}`: {r['levels < 1%']}" for _, r in cd.iterrows() if r["levels < 1%"] != "-")
    for c in cat_ret:
        vc = train[c].value_counts(normalize=True)
        for lvl, p in vc.items():
            N[f"f_{c}_{lvl}".replace(".", "").replace("-", "_")] = pct(p, 1)
    N["month_mjja_pct"] = pct(train.month.isin(["may", "jun", "jul", "aug"]).mean(), 1)

    # Figure 4: frequency panels.
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, c in zip(axes.flat, cat_ret):
        sub = fr[fr.feature == c]
        if c != "month":
            sub = sub.sort_values("percent")
        else:
            sub = sub.iloc[::-1]
        colors = [C_MUTED if l == "unknown" else C_MAIN for l in sub.level]
        ax.barh(sub.level, sub.percent, color=colors, height=0.7)
        for yv, p in zip(sub.level, sub.percent):
            ax.text(p + 0.5, yv, f"{p:.1f}%", va="center", fontsize=7.5, color=INK2)
        ax.set(xlabel="Share of training clients (%)", ylabel="Level", title=f"{c} (cardinality {len(sub)})",
               xlim=(0, sub.percent.max() * 1.22))
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_MAIN), plt.Rectangle((0, 0), 1, 1, color=C_MUTED)]
    fig.legend(handles, ["observed level", "\"unknown\" (kept as a category)"], loc="lower center",
               ncol=2, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    save_fig(fig, "categorical_frequencies", "Level frequencies of the retained categorical features (training set)")

    # Figure 5: cardinality and "unknown" share.
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    cds = cd.sort_values("cardinality")
    axes[0].barh(cds.feature, cds.cardinality, color=C_MAIN, height=0.6)
    for yv, v in zip(cds.feature, cds.cardinality):
        axes[0].text(v + 0.15, yv, str(v), va="center", fontsize=8, color=INK2)
    axes[0].set(xlabel="Number of distinct levels", ylabel="Feature", title="Cardinality")
    cdu = cd.sort_values("\"unknown\" %")
    axes[1].barh(cdu.feature, cdu["\"unknown\" %"], color=C_MUTED, height=0.6)
    for yv, v in zip(cdu.feature, cdu["\"unknown\" %"]):
        axes[1].text(v + 0.8, yv, f"{v:.1f}%", va="center", fontsize=8, color=INK2)
    axes[1].set(xlabel="Share of training rows equal to \"unknown\" (%)", ylabel="Feature",
                title="\"unknown\" share", xlim=(0, cdu["\"unknown\" %"].max() * 1.15))
    for ax in axes:
        ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    save_fig(fig, "cardinality_unknown", "Cardinality and share of \"unknown\" per categorical feature (training set)")


# ----------------------------------------------------------------- stage 3A
def num_num(train, num_ret):
    pear = train[num_ret].corr(method="pearson")
    spear = train[num_ret].corr(method="spearman")
    pear.to_csv(OUT_DIR / "corr_pearson_train.csv")
    spear.to_csv(OUT_DIR / "corr_spearman_train.csv")
    pairs = []
    for i, a in enumerate(num_ret):
        for b in num_ret[i + 1:]:
            pairs.append({"pair": f"{a} - {b}", "spearman": spear.loc[a, b], "pearson": pear.loc[a, b],
                          "|spearman|": abs(spear.loc[a, b])})
    pr = pd.DataFrame(pairs).sort_values("|spearman|", ascending=False)
    contacted = train[train.pdays != -1]
    pr.to_csv(OUT_DIR / "corr_pairs_train.csv", index=False)
    T["corr_pairs"] = md_table(pr.drop(columns="|spearman|").head(6), floatfmt=".3f")
    top = pr.iloc[0]
    N["top_pair"], N["top_pair_rho"], N["top_pair_r"] = top["pair"], f"{top.spearman:.3f}", f"{top.pearson:.3f}"
    sec = pr.iloc[1]
    N["sec_pair"], N["sec_pair_rho"], N["sec_pair_r"] = sec["pair"], f"{sec.spearman:.3f}", f"{sec.pearson:.3f}"
    N["third_pair"], N["third_pair_rho"] = pr.iloc[2]["pair"], f"{pr.iloc[2].spearman:.3f}"
    others = pr.iloc[1:] if top["pair"] == "pdays - previous" else pr
    N["max_other_abs_rho"] = f"{pr[pr.pair != 'pdays - previous']['|spearman|'].max():.3f}"
    N["contacted_rho_pdays_prev"] = f"{contacted.pdays.corr(contacted.previous, method='spearman'):.3f}"
    N["contacted_r_pdays_prev"] = f"{contacted.pdays.corr(contacted.previous):.3f}"
    sk = train[num_ret].skew()
    N["n_skew_gt1"] = int((sk.abs() > 1).sum())
    N["skew_gt1_list"] = ", ".join(f"`{c}`" for c in sk[sk.abs() > 1].index)

    # Figure 6: Spearman vs Pearson.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, (m, title) in zip(axes, [(spear, "Spearman rho (chosen)"), (pear, "Pearson r (for comparison)")]):
        sns.heatmap(m, ax=ax, vmin=-1, vmax=1, cmap=DIVERGING, annot=True, fmt=".2f", square=True,
                    linewidths=1, linecolor="white", cbar_kws={"label": "Correlation coefficient", "shrink": 0.8},
                    annot_kws={"fontsize": 8})
        ax.set(title=title, xlabel="Feature", ylabel="Feature")
        ax.grid(False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "corr_spearman_pearson", "Correlation among retained numerical features (training set)")

    # Figure 7: pdays vs previous for previously contacted clients.
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    rng = np.random.default_rng(RANDOM_STATE)
    for lvl in ["no", "yes"]:
        sub = contacted[contacted[TARGET] == lvl]
        jitter = rng.uniform(-0.15, 0.15, len(sub))
        ax.scatter(sub.previous + jitter, sub.pdays, s=8, alpha=0.35, color=Y_PALETTE[lvl],
                   edgecolors="none", label=f"y = {lvl} (n = {len(sub):,})")
    ax.set(xscale="log", xlabel="previous: contacts before this campaign (log scale, jittered)",
           ylabel="pdays: days since previous contact",
           title=f"Previously contacted clients only (n = {len(contacted):,}); "
                 f"Spearman rho = {contacted.pdays.corr(contacted.previous, method='spearman'):.2f}")
    ax.legend(loc="upper right", markerscale=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "pdays_vs_previous", "pdays versus previous once the -1 sentinel is removed (training set)")


# ----------------------------------------------------------------- stage 3B
def rate_table(train, c):
    g = train.groupby(c)[TARGET].agg(rows="size", subscribed=lambda s: int((s == "yes").sum()))
    g["rate"] = g.subscribed / g.rows
    lo, hi = wilson(g.subscribed.values, g.rows.values)
    g["ci_low"], g["ci_high"] = lo, hi
    if c == "month":
        g = g.reindex(MONTHS)
    return g


def cat_target(train, cat_ret):
    base = (train[TARGET] == "yes").mean()
    N["train_base_rate"] = pct(base)
    allr = []
    for c in cat_ret:
        g = rate_table(train, c)
        for lvl, r in g.iterrows():
            allr.append({"feature": c, "level": lvl, "rows": int(r.rows), "subscribed": int(r.subscribed),
                         "subscription rate %": r.rate * 100, "95% CI low %": r.ci_low * 100,
                         "95% CI high %": r.ci_high * 100, "lift vs base": r.rate / base})
            N[f"r_{c}_{lvl}".replace(".", "").replace("-", "_")] = pct(r.rate, 1)
    ar = pd.DataFrame(allr)
    ar.to_csv(OUT_DIR / "categorical_target_rates_train.csv", index=False)
    spread = ar.groupby("feature")["subscription rate %"].agg(lambda s: s.max() - s.min()).sort_values(ascending=False)
    sp = spread.reset_index().rename(columns={"subscription rate %": "max - min rate (pp)"})
    T["cat_rate_spread"] = md_table(sp, floatfmt=".1f")
    N["spread_top3"] = ", ".join(f"`{f}` ({v:.1f} pp)" for f, v in spread.head(3).items())
    N["spread_bottom"] = f"`{spread.index[-1]}` ({spread.iloc[-1]:.1f} pp)"

    def panel(ax, c, order=None, horizontal=False):
        g = rate_table(train, c)
        if order is not None:
            g = g.reindex(order)
        x = np.arange(len(g))
        err = np.vstack([(g.rate - g.ci_low) * 100, (g.ci_high - g.rate) * 100])
        if horizontal:
            ax.barh(x, g.rate * 100, xerr=err, color=C_YES, height=0.65, error_kw=dict(ecolor=INK2, lw=1))
            ax.axvline(base * 100, color=INK, ls="--", lw=1, label=f"training base rate {base * 100:.1f}%")
            ax.set_yticks(x, [f"{l} (n={int(n):,})" for l, n in zip(g.index, g.rows)])
            ax.set(xlabel="Subscription rate, y = yes (%)", ylabel=c)
            ax.grid(axis="y", visible=False)
        else:
            ax.bar(x, g.rate * 100, yerr=err, color=C_YES, width=0.65, error_kw=dict(ecolor=INK2, lw=1))
            ax.axhline(base * 100, color=INK, ls="--", lw=1, label=f"training base rate {base * 100:.1f}%")
            ax.set_xticks(x, [f"{l}\nn={int(n):,}" for l, n in zip(g.index, g.rows)], fontsize=8)
            ax.set(ylabel="Subscription rate, y = yes (%)", xlabel=c)
            ax.grid(axis="x", visible=False)
        ax.set_title(c)

    # Figure 8: previous-campaign and contact-related binaries.
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.0), gridspec_kw={"width_ratios": [4, 3, 2, 2]})
    panel(axes[0], "poutcome", ["failure", "other", "success", "unknown"])
    panel(axes[1], "contact", ["cellular", "telephone", "unknown"])
    panel(axes[2], "housing", ["no", "yes"])
    panel(axes[3], "loan", ["no", "yes"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 0.9))
    save_fig(fig, "rate_poutcome_contact_housing_loan",
             "Subscription rate by poutcome, contact, housing and loan, with 95% Wilson CIs (training set)")

    # Figure 9: month (volume and rate as two panels, not a dual axis).
    g = rate_table(train, "month")
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    x = np.arange(12)
    axes[0].bar(x, g.rows, color=C_MAIN, width=0.65, label="clients contacted")
    axes[0].set(ylabel="Clients (count)", title="Contact volume by month")
    axes[0].legend(loc="upper right", fontsize=8)
    err = np.vstack([(g.rate - g.ci_low) * 100, (g.ci_high - g.rate) * 100])
    axes[1].bar(x, g.rate * 100, yerr=err, color=C_YES, width=0.65, error_kw=dict(ecolor=INK2, lw=1),
                label="subscription rate (95% CI)")
    axes[1].axhline(base * 100, color=INK, ls="--", lw=1, label=f"training base rate {base * 100:.1f}%")
    axes[1].set(ylabel="Subscription rate, y = yes (%)", xlabel="Month of last contact", title="Subscription rate by month")
    axes[1].set_xticks(x, MONTHS)
    axes[1].legend(loc="upper left", fontsize=8)
    for ax in axes:
        ax.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "month_volume_rate", "Contact volume and subscription rate by month (training set)")
    N["month_rate_max"] = f"{g.rate.idxmax()} ({g.rate.max() * 100:.1f}%, n = {int(g.rows[g.rate.idxmax()]):,})"
    N["month_rate_min"] = f"{g.rate.idxmin()} ({g.rate.min() * 100:.1f}%, n = {int(g.rows[g.rate.idxmin()]):,})"
    N["month_vol_rate_rho"] = f"{g.rows.corr(g.rate, method='spearman'):.2f}"
    high = g[g.rate > 0.3]
    N["months_rate_gt30"] = ", ".join(f"{m} ({r * 100:.1f}%, n = {int(n):,})" for m, r, n in zip(high.index, high.rate, high.rows))

    # Figure 10: job and education.
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [1.4, 1]})
    jo = rate_table(train, "job").sort_values("rate").index.tolist()
    panel(axes[0], "job", jo, horizontal=True)
    panel(axes[1], "education", ["primary", "secondary", "tertiary", "unknown"])
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=1, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_fig(fig, "rate_job_education", "Subscription rate by job and education, with 95% Wilson CIs (training set)")


# ----------------------------------------------------------------- stage 3C
def group_stats(train, num, cat):
    g = train.groupby([cat, TARGET])[num].describe()[["count", "25%", "50%", "75%"]]
    g.columns = ["rows", "Q1", "median", "Q3"]
    g["IQR"] = g.Q3 - g.Q1
    g["rows"] = g["rows"].astype(int)
    return g.reset_index()


def num_cat(train):
    out = []

    def boxes(ax, num, cat, order, scale=None, legend_loc="upper right"):
        sns.boxplot(data=train, x=cat, y=num, hue=TARGET, order=order, hue_order=["no", "yes"],
                    palette=Y_PALETTE, ax=ax, width=0.7, linewidth=0.9,
                    flierprops=dict(marker="o", markersize=1.8, alpha=0.3, markerfacecolor=C_MUTED,
                                    markeredgecolor="none"))
        if scale == "symlog":
            ax.set_yscale("symlog", linthresh=100)
        elif scale == "log":
            ax.set_yscale("log")
        ax.legend(title="y (subscribed)", loc=legend_loc, fontsize=8, title_fontsize=8)
        ax.grid(axis="x", visible=False)

    # Figure 11: age x job x y.
    jo = train.groupby("job").age.median().sort_values().index.tolist()
    fig, ax = plt.subplots(figsize=(13, 4.6))
    boxes(ax, "age", "job", jo, legend_loc="upper left")
    ax.set(xlabel="Job (ordered by median age)", ylabel="Age (years)")
    ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "box_age_job_y", "Age by job, split by subscription outcome (training set)")
    gs = group_stats(train, "age", "job")
    gs.insert(0, "feature", "age")
    gs.rename(columns={"job": "group"}, inplace=True)
    gs.insert(1, "grouping", "job")
    out.append(gs)

    # Figure 12: balance x education x y.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    boxes(ax, "balance", "education", ["primary", "secondary", "tertiary", "unknown"], "symlog")
    ax.set(xlabel="Education", ylabel="Average yearly balance (euros, symlog scale)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "box_balance_education_y", "Balance by education, split by subscription outcome (training set)")
    gs = group_stats(train, "balance", "education")
    gs.insert(0, "feature", "balance")
    gs.rename(columns={"education": "group"}, inplace=True)
    gs.insert(1, "grouping", "education")
    out.append(gs)

    # Figure 13: campaign x contact x y.
    fig, ax = plt.subplots(figsize=(10, 4.6))
    boxes(ax, "campaign", "contact", ["cellular", "telephone", "unknown"], "log")
    ax.set(xlabel="Contact communication type", ylabel="Contacts in current campaign (log scale)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "box_campaign_contact_y", "Campaign contacts by contact type, split by subscription outcome (training set)")
    gs = group_stats(train, "campaign", "contact")
    gs.insert(0, "feature", "campaign")
    gs.rename(columns={"contact": "group"}, inplace=True)
    gs.insert(1, "grouping", "contact")
    out.append(gs)

    gall = pd.concat(out, ignore_index=True)
    gall.to_csv(OUT_DIR / "grouped_box_stats_train.csv", index=False)
    for f, grp in [("age", "job"), ("balance", "education"), ("campaign", "contact")]:
        sub = gall[(gall.feature == f)].drop(columns=["feature", "grouping"]).rename(columns={"group": grp})
        T[f"box_{f}"] = md_table(sub, floatfmt=".1f")
        for _, r in sub.iterrows():
            key = f"b_{f}_{r[grp]}_{r[TARGET]}".replace(".", "").replace("-", "_")
            N[key + "_med"] = f"{r['median']:,.1f}".rstrip("0").rstrip(".")
            N[key + "_iqr"] = f"{r['IQR']:,.1f}".rstrip("0").rstrip(".")
    # Overall medians by y, for context.
    for f in ["age", "balance", "campaign"]:
        for lvl in ["no", "yes"]:
            N[f"med_{f}_{lvl}"] = f"{train.loc[train[TARGET] == lvl, f].median():,.0f}"
    N["age_ge60_rate"] = pct((train.loc[train.age >= 60, TARGET] == "yes").mean(), 1)
    N["age_ge60_n"] = fmt_int((train.age >= 60).sum())
    N["age_lt25_rate"] = pct((train.loc[train.age < 25, TARGET] == "yes").mean(), 1)
    N["age_lt25_n"] = fmt_int((train.age < 25).sum())
    N["age_30_59_rate"] = pct((train.loc[train.age.between(30, 59), TARGET] == "yes").mean(), 1)
    N["camp1_rate"] = pct((train.loc[train.campaign == 1, TARGET] == "yes").mean(), 1)
    N["camp_gt5_rate"] = pct((train.loc[train.campaign > 5, TARGET] == "yes").mean(), 1)
    N["camp_gt5_n"] = fmt_int((train.campaign > 5).sum())


# ----------------------------------------------------------------- render
def render():
    N["figure_list"] = "\n".join(f"| {n} | `{f}` | {t} |" for n, f, t in FIGS)
    flat = {k: (v if isinstance(v, str) else str(v)) for k, v in N.items()}
    flat.update({f"table_{k}": v for k, v in T.items()})
    write_json({k: v for k, v in flat.items() if not k.startswith("table_") and not k.endswith("_md")},
               "part1_numbers.json")
    if not TEMPLATE.exists():
        print(f"Template not found ({TEMPLATE}); skipped draft rendering.")
        return
    text = TEMPLATE.read_text()
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(flat))
    if missing:
        raise KeyError(f"Template placeholders without computed values: {missing}")
    DRAFT.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: flat[m.group(1)], text))
    print(f"Rendered {DRAFT.relative_to(ROOT)}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load()
    num, cat, num_ret, cat_ret = schema(df)
    quality(df, num, cat)
    train = target_and_split(df, num_ret, cat_ret)
    univariate_numeric(train, num_ret)
    univariate_categorical(train, cat_ret)
    num_num(train, num_ret)
    cat_target(train, cat_ret)
    num_cat(train)
    render()


if __name__ == "__main__":
    main()
