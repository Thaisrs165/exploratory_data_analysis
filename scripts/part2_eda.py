"""Part 2 - EDA stages 4-5 for UCI Bank Marketing (bank-full.csv).

Fits the preprocessing pipeline from code/bank_preprocessing.py on the training
set, checks it, runs PCA / t-SNE / UMAP on the scaled training features, and
renders the final report: stages 0-3 are copied unchanged from
docs/projects/eda/part1_draft.md and stages 4-5 come from
scripts/templates/part2_template.md. No model is trained.

Usage (from the repository root, after scripts/part1_eda.py):
    python scripts/part2_eda.py

Outputs:
    docs/projects/eda/part2_figures/   numbered PNG figures (continue Part 1's numbering)
    docs/projects/eda/part2_outputs/   CSV / JSON tables
    docs/projects/eda/index.md         final report (stages 0-5)
"""

import json
import re
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import bank_preprocessing as bp
# Part 1's plot style (rcParams are set on import), colours and formatting helpers.
from part1_eda import C_MAIN, C_MUTED, C_NO, C_YES, INK, INK2, fmt_int, md_table, pct

RANDOM_STATE = bp.RANDOM_STATE
SAMPLE_SIZE = 5000
PERPLEXITIES = [5, 30, 50]
N_NEIGHBORS = [5, 15, 50]
UMAP_MIN_DIST = 0.1

DOCS = ROOT / "docs" / "projects" / "eda"
P1_DRAFT = DOCS / "part1_draft.md"
P1_OUT = DOCS / "part1_outputs"
FIG_DIR = DOCS / "part2_figures"
OUT_DIR = DOCS / "part2_outputs"
TEMPLATE = ROOT / "scripts" / "templates" / "part2_template.md"
INDEX = DOCS / "index.md"

P1_START, P1_END = "## 0. Proposal", "## Handoff for Part 2"

N = {}       # scalar values for the template
T = {}       # markdown tables for the template
FIGS = []    # (number, file, title)
CHECKS = []  # (what, Part 1 value, recomputed value)


# ----------------------------------------------------------------- helpers
def save_fig(fig, slug, title):
    n = N["p1_n_figs"] + len(FIGS) + 1
    name = f"fig{n:02d}_{slug}.png"
    fig.suptitle(f"Figure {n}. {title}", x=0.01, ha="left")
    fig.savefig(FIG_DIR / name, facecolor="white")
    plt.close(fig)
    FIGS.append((n, name, title))
    N[f"fig_{slug}"] = n
    N[f"fig_{slug}_md"] = f"![Figure {n}. {title}](part2_figures/{name})"


def write_json(obj, name):
    (OUT_DIR / name).write_text(json.dumps(obj, indent=2, default=str) + "\n")


def agree(what, part1, recomputed):
    """Record a Part 1 value next to the value recomputed here; stop on any mismatch."""
    CHECKS.append({"quantity": what, "Part 1": str(part1), "recomputed in Part 2": str(recomputed)})
    if str(part1) != str(recomputed):
        raise AssertionError(f"Part 2 disagrees with Part 1 on {what}: {part1} != {recomputed}")


def scatter_by_target(ax, xy, y, s=5):
    for lvl, name, colour in [(0, "no", C_NO), (1, "yes", C_YES)]:   # minority drawn last, on top
        m = y == lvl
        ax.scatter(xy[m, 0], xy[m, 1], s=s, alpha=0.5 if lvl == 0 else 0.75, color=colour,
                   edgecolors="none", label=f"y = {name} (n = {m.sum():,})", rasterized=True)


# ----------------------------------------------------------------- Part 1 agreement
def load_and_check():
    p1 = json.loads((P1_OUT / "part1_numbers.json").read_text())
    spec = json.loads((P1_OUT / "train_test_split.json").read_text())
    feats = json.loads((P1_OUT / "feature_lists.json").read_text())
    N["p1_n_figs"] = len(list((DOCS / "part1_figures").glob("fig*.png")))
    N["sha256"] = bp.data_sha256()
    agree("bank-full.csv SHA-256 (LF line endings)", spec["input_sha256"], N["sha256"])

    df = pd.read_csv(bp.DATA, sep=";")
    X, y = bp.load_data()
    agree("rows x columns of bank-full.csv", f"{p1['rows']} x {p1['cols']}", f"{fmt_int(df.shape[0])} x {df.shape[1]}")
    agree("retained numerical features", feats["numerical"], bp.NUMERICAL)
    agree("retained categorical features", feats["categorical"], bp.CATEGORICAL)
    agree("excluded columns", list(feats["excluded_columns"]), bp.LEAKAGE)
    agree("NaN values in the file", p1["total_nan"], int(df.isna().sum().sum()))
    agree("minority class (y = yes) share", p1["pct_yes"], pct(y.mean()))
    agree("imbalance ratio no : yes", p1["imbalance_ratio"], f"{(y == 0).sum() / y.sum():.2f}")
    unk = (df[bp.CATEGORICAL] == "unknown").mean()
    agree("column with most \"unknown\"", "poutcome", unk.idxmax())
    agree("poutcome = \"unknown\" (full data)", p1["unk_poutcome"],
          f"{fmt_int((df.poutcome == 'unknown').sum())} ({unk.max() * 100:.2f}%)")
    agree("pdays = -1 rows equal to previous = 0 rows (mismatches)", p1["sent_prev_mismatch"],
          int(((df.pdays == -1) != (df.previous == 0)).sum()))

    X_train, X_test, y_train, y_test = bp.split(X, y)
    idx = pd.read_csv(P1_OUT / "train_test_split_indices.csv")
    agree("train row ids identical to train_test_split_indices.csv", True,
          set(idx.row_id[idx.split == "train"]) == set(X_train.index))
    agree("test row ids identical to train_test_split_indices.csv", True,
          set(idx.row_id[idx.split == "test"]) == set(X_test.index))
    agree("train rows / positives", f"{p1['n_train']} / {p1['train_yes']}", f"{fmt_int(len(y_train))} / {fmt_int(y_train.sum())}")
    agree("test rows / positives", f"{p1['n_test']} / {p1['test_yes']}", f"{fmt_int(len(y_test))} / {fmt_int(y_test.sum())}")

    spear = X_train[bp.NUMERICAL].corr(method="spearman")
    pear = X_train[bp.NUMERICAL].corr(method="pearson")
    off = spear.abs().where(~np.eye(len(spear), dtype=bool)).stack()
    a, b = off.idxmax()
    agree("most correlated numerical pair (train, absolute Spearman)", p1["top_pair"], f"{a} - {b}")
    agree("its Spearman rho", p1["top_pair_rho"], f"{spear.loc[a, b]:.3f}")
    agree("its Pearson r", p1["top_pair_r"], f"{pear.loc[a, b]:.3f}")
    for c in ["balance", "campaign", "previous"]:
        s = X_train[c]
        q1, q3 = s.quantile([0.25, 0.75])
        n_out = int(((s < q1 - 1.5 * (q3 - q1)) | (s > q3 + 1.5 * (q3 - q1))).sum())
        agree(f"{c}: 1.5 x IQR outliers (train)", p1[f"{c}_out"], fmt_int(n_out))
        agree(f"{c}: skew (train)", p1[f"{c}_skew"], f"{s.skew():,.2f}")
    agree("one-hot width (sum of train cardinalities)", p1["total_dummies"],
          int(X_train[bp.CATEGORICAL].nunique().sum()))

    N.update({k: p1[k] for k in ["rows", "cols", "n_num", "n_cat", "n_num_ret", "n_cat_ret", "n_pred_ret",
                                 "pct_yes", "imbalance_ratio", "n_yes", "n_train", "n_test", "train_yes",
                                 "test_yes", "train_yes_pct", "test_yes_pct", "top_pair", "top_pair_rho",
                                 "top_pair_r", "contacted_rho_pdays_prev", "total_dummies", "unk_poutcome",
                                 "unk_contact", "total_nan", "order_rate_first", "order_rate_last",
                                 "balance_out", "balance_outp", "balance_skew", "campaign_out", "campaign_skew",
                                 "previous_skew", "age_skew", "day_skew", "pdays_neg1_pct", "n_contacted_train",
                                 "contacted_train_pct", "pout_unknown_contacted", "majority_acc",
                                 "r_poutcome_success", "r_poutcome_unknown", "r_contact_unknown",
                                 "r_contact_cellular", "month_rate_max", "month_rate_min", "age_ge60_rate",
                                 "age_30_59_rate", "camp1_rate", "camp_gt5_rate", "dup_feat_only",
                                 "default_yes_pct", "rare_levels", "sent_rate_y", "nonsent_rate_y"]})
    T["agreement"] = md_table(pd.DataFrame(CHECKS))
    N["n_checks"] = len(CHECKS)
    pd.DataFrame(CHECKS).to_csv(OUT_DIR / "part1_agreement_checks.csv", index=False)
    return X_train, X_test, y_train, y_test


# ----------------------------------------------------------------- stage 4A / 4C
def preprocess(X_train, X_test, y_train):
    # Checked before any strategy is claimed.
    N["nan_train"], N["nan_test"] = int(X_train.isna().sum().sum()), int(X_test.isna().sum().sum())
    unseen = {c: sorted(set(X_test[c]) - set(X_train[c])) for c in bp.CATEGORICAL}
    N["n_unseen_test_levels"] = sum(len(v) for v in unseen.values())
    unk = pd.DataFrame({
        "feature": [f"`{c}`" for c in bp.CATEGORICAL],
        "NaN (train)": [int(X_train[c].isna().sum()) for c in bp.CATEGORICAL],
        "\"unknown\" rows (train)": [int((X_train[c] == "unknown").sum()) for c in bp.CATEGORICAL],
        "\"unknown\" % (train)": [(X_train[c] == "unknown").mean() * 100 for c in bp.CATEGORICAL],
        "levels (train)": [X_train[c].nunique() for c in bp.CATEGORICAL],
        "test levels unseen in train": [len(unseen[c]) for c in bp.CATEGORICAL],
    })
    unk.to_csv(OUT_DIR / "categorical_checks.csv", index=False)
    T["cat_checks"] = md_table(unk)

    pre = bp.build_preprocessor().fit(X_train)     # training set only
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)                 # transform only
    names = pre.get_feature_names_out()
    Zdf = pd.DataFrame(Z_train, columns=names, index=X_train.index)
    Zte = pd.DataFrame(Z_test, columns=names, index=X_test.index)

    N.update(z_train_shape=f"{Z_train.shape[0]:,} x {Z_train.shape[1]}", z_test_shape=f"{Z_test.shape[0]:,} x {Z_test.shape[1]}",
             n_features_out=Z_train.shape[1], nan_z_train=int(np.isnan(Z_train).sum()), nan_z_test=int(np.isnan(Z_test).sum()),
             inf_z=int(np.isinf(Z_train).sum() + np.isinf(Z_test).sum()),
             n_num_out=len(bp.NUMERICAL), n_flag_out=1, n_onehot_out=int(sum(n.split("_")[0] in bp.CATEGORICAL for n in names)))
    assert N["nan_z_train"] == N["nan_z_test"] == N["inf_z"] == 0
    assert N["n_onehot_out"] + N["n_num_out"] + N["n_flag_out"] == Z_train.shape[1]
    pd.DataFrame({"position": range(len(names)), "feature": names}).to_csv(OUT_DIR / "feature_names.csv", index=False)
    groups = [("standardised numerical", list(names[:2])), ("clipped + transformed numerical", list(names[2:5])),
              ("`pdays` and its indicator", list(names[5:7]))]
    groups += [(f"one-hot `{c}`", [n for n in names if n.startswith(c + "_")]) for c in bp.CATEGORICAL]
    T["feature_names"] = md_table(pd.DataFrame(
        [{"group": g, "columns": len(v), "feature names": ", ".join(f"`{n}`" for n in v)} for g, v in groups]))

    # Outlier strategy: rows changed by clipping.
    bounds, tr, te = bp.clip_bounds(pre), bp.clip_report(pre, X_train), bp.clip_report(pre, X_test)
    rows = []
    for c, (lo, hi) in bounds.items():
        s = X_train[c]
        rows.append({"feature": f"`{c}`", "lower bound (train P1)": lo, "upper bound (train P99)": hi,
                     "train rows below": int((s < lo).sum()), "train rows above": int((s > hi).sum()),
                     "train rows clipped": tr[c], "train %": tr[c] / len(X_train) * 100,
                     "test rows clipped": te[c], "test %": te[c] / len(X_test) * 100})
        N[f"clip_{c}_lo"], N[f"clip_{c}_hi"] = f"{lo:,.2f}", f"{hi:,.2f}"
        N[f"clip_{c}_train"], N[f"clip_{c}_test"] = fmt_int(tr[c]), fmt_int(te[c])
        N[f"clip_{c}_train_pct"] = pct(tr[c] / len(X_train))
    rows.append({"feature": "any of the three", "lower bound (train P1)": np.nan, "upper bound (train P99)": np.nan,
                 "train rows below": np.nan, "train rows above": np.nan,
                 "train rows clipped": tr["any"], "train %": tr["any"] / len(X_train) * 100,
                 "test rows clipped": te["any"], "test %": te["any"] / len(X_test) * 100})
    cl = pd.DataFrame(rows)
    cl.to_csv(OUT_DIR / "clip_report.csv", index=False)
    shown = cl.copy()
    for c, f in [("lower bound (train P1)", "{:,.2f}"), ("upper bound (train P99)", "{:,.2f}"),
                 ("train rows below", "{:,.0f}"), ("train rows above", "{:,.0f}"), ("train rows clipped", "{:,.0f}"),
                 ("train %", "{:.2f}"), ("test rows clipped", "{:,.0f}"), ("test %", "{:.2f}")]:
        shown[c] = cl[c].map(lambda v, f=f: "-" if pd.isna(v) else f.format(v))
    T["clip"] = shown.to_markdown(index=False, colalign=("left",) + ("right",) * 8, disable_numparse=True)
    N.update(clip_any_train=fmt_int(tr["any"]), clip_any_train_pct=pct(tr["any"] / len(X_train)),
             clip_any_test=fmt_int(te["any"]), clip_any_test_pct=pct(te["any"] / len(X_test)),
             clip_any_total=fmt_int(tr["any"] + te["any"]))
    hit = (X_train.balance < bounds["balance"][0]) | (X_train.balance > bounds["balance"][1]) \
        | (X_train.campaign > bounds["campaign"][1]) | (X_train.previous > bounds["previous"][1])
    assert int(hit.sum()) == tr["any"]
    N["clip_rate_y"] = pct(y_train[hit].mean(), 1)
    N["noclip_rate_y"] = pct(y_train[~hit].mean(), 1)

    # Why clipping: Yeo-Johnson on the unclipped balance leaves extreme z-scores.
    from sklearn.preprocessing import PowerTransformer
    raw_yj = PowerTransformer().fit_transform(X_train[["balance"]]).ravel()
    N["bal_yj_noclip_min"], N["bal_yj_noclip_max"] = f"{raw_yj.min():.1f}", f"{raw_yj.max():.1f}"
    N["bal_yj_lambda"] = f"{pre.named_steps['columns'].named_transformers_['signed'].named_steps['yeo_johnson'].lambdas_[0]:.3f}"

    # Before / after per numerical feature (training set).
    def z(s):
        return (s - s.mean()) / s.std(ddof=0)

    treat = {"age": "standardise", "day": "standardise", "balance": "clip P1-P99, Yeo-Johnson, standardise",
             "campaign": "clip P1-P99, log1p, standardise", "previous": "clip P1-P99, log1p, standardise",
             "pdays": "mask -1, standardise on contacted clients, fill 0 + indicator"}
    rows = []
    for c in names[:6]:
        raw, out = X_train[c].astype(float), Zdf[c]
        rows.append({"feature": f"`{c}`", "treatment": treat[c], "raw skew": raw.skew(),
                     "raw min z": z(raw).min(), "raw max z": z(raw).max(),
                     "output mean": out.mean(), "output std": out.std(ddof=0), "output skew": out.skew(),
                     "output min": out.min(), "output max": out.max(),
                     "test output min": Zte[c].min(), "test output max": Zte[c].max()})
        N[f"out_{c}_min"], N[f"out_{c}_max"] = f"{out.min():.2f}", f"{out.max():.2f}"
        N[f"out_{c}_skew"], N[f"out_{c}_std"] = f"{out.skew():.2f}", f"{out.std(ddof=0):.2f}"
        N[f"rawz_{c}_max"], N[f"rawz_{c}_min"] = f"{z(raw).max():.1f}", f"{z(raw).min():.1f}"
    ba = pd.DataFrame(rows)
    ba.to_csv(OUT_DIR / "numerical_before_after_train.csv", index=False)
    T["before_after"] = md_table(ba, floatfmt=".2f")
    N["max_abs_out_train"] = f"{np.abs(Z_train).max():.2f}"
    N["max_abs_out_test"] = f"{np.abs(Z_test).max():.2f}"

    sc = pre.named_steps["columns"].named_transformers_["sentinel"].named_steps["scale"]
    N["pdays_fit_mean"], N["pdays_fit_std"] = f"{sc.mean_[0]:.2f}", f"{sc.scale_[0]:.2f}"
    ind = Zdf["missingindicator_pdays"]
    N["ind_train"], N["ind_train_pct"] = fmt_int(ind.sum()), pct(ind.mean())
    assert (ind == 1).equals(X_train.pdays == -1) and (Zdf.loc[ind == 1, "pdays"] == 0).all()
    N["ind_vs_pout_mismatch_train"] = int((ind != Zdf["poutcome_unknown"]).sum())

    # Figure: raw versus pipeline output for the four features that are more than standardised.
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.2))
    for j, c in enumerate(["balance", "campaign", "previous", "pdays"]):
        raw = X_train[c]
        axes[0, j].hist(raw, bins=60, color=C_MUTED, edgecolor="white", linewidth=0.3)
        axes[0, j].set(yscale="log", title=f"{c}: raw (skew {raw.skew():.2f})", xlabel=f"{c} (raw units)",
                       ylabel="Training clients (count, log scale)")
        if c in bounds:
            for b in bounds[c]:
                axes[0, j].axvline(b, color=C_YES, lw=1.2, ls="--")
            axes[0, j].plot([], [], color=C_YES, lw=1.2, ls="--", label=f"clip bounds (train P1, P99)\n{tr[c]:,} rows clipped")
            axes[0, j].legend(loc="upper right", fontsize=7.5)
        axes[1, j].hist(Zdf[c], bins=60, color=C_MAIN, edgecolor="white", linewidth=0.3)
        axes[1, j].set(yscale="log", title=f"{c}: pipeline output (skew {Zdf[c].skew():.2f})",
                       xlabel=f"{c} (standardised units)", ylabel="Training clients (count, log scale)")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, "raw_vs_preprocessed", "Raw versus preprocessed distributions of the transformed numerical features (training set)")
    return pre, Zdf, Zte


# ----------------------------------------------------------------- stage 4B
def pca_stage(Zdf, y_train, sample):
    pca = PCA(random_state=RANDOM_STATE).fit(Zdf.values)          # all training rows
    evr, cum = pca.explained_variance_ratio_, np.cumsum(pca.explained_variance_ratio_)
    ev = pd.DataFrame({"component": [f"PC{i + 1}" for i in range(len(evr))],
                       "explained variance %": evr * 100, "cumulative %": cum * 100})
    ev.to_csv(OUT_DIR / "pca_explained_variance.csv", index=False)
    T["pca_var"] = md_table(ev.head(10), floatfmt=".2f")
    N.update(pc1=pct(evr[0]), pc2=pct(evr[1]), pc12=pct(cum[1]), pc3=pct(evr[2]), pc10_cum=pct(cum[9]),
             n_pc=len(evr))
    for t in (80, 90, 95, 99):
        N[f"n_pc_{t}"] = int(np.searchsorted(cum, t / 100) + 1)
    N["n_pc_null"] = int((evr < 1e-12).sum())

    # Share of the total variance that each block of columns carries.
    var = Zdf.var(ddof=0)
    num_cols = list(Zdf.columns[:6])
    N["var_share_num"] = pct(var[num_cols].sum() / var.sum(), 1)
    N["var_share_other"] = pct(1 - var[num_cols].sum() / var.sum(), 1)
    N["var_pdays_out"] = f"{var['pdays']:.2f}"

    load = pd.DataFrame(pca.components_[:4].T, index=Zdf.columns, columns=["PC1", "PC2", "PC3", "PC4"])
    load.to_csv(OUT_DIR / "pca_loadings.csv", index_label="feature")
    blocks = []
    for pc in ["PC1", "PC2"]:
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(8)
        blocks.append(pd.DataFrame({f"{pc} feature": [f"`{f}`" for f in top.index], f"{pc} loading": top.values}))
        N[f"{pc.lower()}_top"] = ", ".join(f"`{f}` ({v:+.2f})" for f, v in top.head(5).items())
        N[f"{pc.lower()}_top5_sq"] = pct((top.head(5) ** 2).sum(), 1)
    T["pca_loadings"] = md_table(pd.concat(blocks, axis=1), floatfmt="+.3f")
    for pc in ["PC1", "PC2"]:
        for f in ["previous", "pdays", "missingindicator_pdays", "poutcome_unknown", "poutcome_failure",
                  "age", "day", "balance", "campaign", "contact_unknown", "contact_cellular", "month_may",
                  "housing_yes", "housing_no", "marital_single", "marital_married", "job_retired"]:
            N[f"l_{pc.lower()}_{f}"] = f"{load.loc[f, pc]:+.2f}"

    scores = pca.transform(Zdf.values)
    yv = y_train.values
    for i, pc in enumerate(["pc1", "pc2"]):
        a, b = scores[yv == 1, i], scores[yv == 0, i]
        pooled = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
        N[f"{pc}_d"] = f"{(a.mean() - b.mean()) / pooled:+.2f}"
        N[f"{pc}_mean_yes"], N[f"{pc}_mean_no"] = f"{a.mean():+.2f}", f"{b.mean():+.2f}"

    # Figure: explained variance.
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    x = np.arange(1, len(evr) + 1)
    ax.bar(x, evr * 100, color=C_MUTED, width=0.7, label="individual component")
    ax.plot(x, cum * 100, color=C_MAIN, lw=2, marker="o", ms=3, label="cumulative")
    for t in (80, 95):
        k = N[f"n_pc_{t}"]
        ax.axhline(t, color=INK2, ls=":", lw=0.9)
        ax.annotate(f"{t}% at {k} components", (k, t), xytext=(k + 1.2, t - 7), fontsize=8, color=INK2,
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.annotate(f"PC1 + PC2 = {cum[1] * 100:.1f}%", (2, cum[1] * 100), xytext=(4.5, cum[1] * 100 - 9), fontsize=8,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.7))
    ax.set(xlabel="Principal component (fitted on all training rows)", ylabel="Explained variance (%)",
           xlim=(0, len(evr) + 1), ylim=(0, 104))
    ax.legend(loc="center right")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_fig(fig, "pca_explained_variance", "PCA explained variance of the preprocessed training features")

    # Figure: loadings of PC1 and PC2.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, pc in zip(axes, ["PC1", "PC2"]):
        top = load[pc].reindex(load[pc].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
        ax.barh(top.index, top.values, color=[C_MAIN if v >= 0 else C_YES for v in top.values], height=0.65)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set(xlabel="Loading (weight in the component)", ylabel="Preprocessed feature",
               title=f"{pc}: 10 largest |loadings|")
        ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=C_MAIN), plt.Rectangle((0, 0), 1, 1, color=C_YES)]
    fig.legend(handles, ["positive loading", "negative loading"], loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    save_fig(fig, "pca_loadings", "PCA loadings of PC1 and PC2")
    # Figure: PC1-PC2 projection, coloured by target.
    fig, ax = plt.subplots(figsize=(8, 5.4))
    scatter_by_target(ax, scores[sample][:, :2], yv[sample], s=6)
    ax.set(xlabel=f"PC1 ({evr[0] * 100:.1f}% of variance)", ylabel=f"PC2 ({evr[1] * 100:.1f}% of variance)",
           title=f"PCA fitted on all {len(Zdf):,} training rows; stratified sample of {len(sample):,} shown")
    ax.legend(loc="upper right", markerscale=2.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, "pca_projection", "2D PCA projection of the training set, coloured by target")

    return scores


def manifold_stage(Zdf, X_train, y_train, sample):
    Zs, ys = Zdf.values[sample], y_train.values[sample]
    N.update(sample_n=fmt_int(len(sample)), sample_n_raw=len(sample), sample_yes=fmt_int(ys.sum()), sample_yes_pct=pct(ys.mean()),
             sample_share=pct(len(sample) / len(Zdf), 1),
             perplexities=", ".join(map(str, PERPLEXITIES)), n_neighbors=", ".join(map(str, N_NEIGHBORS)),
             umap_min_dist=UMAP_MIN_DIST)
    emb = {}

    fig, axes = plt.subplots(1, len(PERPLEXITIES), figsize=(15, 5.2))
    for ax, p in zip(axes, PERPLEXITIES):
        ts = TSNE(n_components=2, perplexity=p, init="pca", learning_rate="auto", random_state=RANDOM_STATE)
        emb[f"tsne_{p}"] = ts.fit_transform(Zs)
        N[f"tsne_kl_{p}"] = f"{ts.kl_divergence_:.2f}"
        scatter_by_target(ax, emb[f"tsne_{p}"], ys)
        ax.set(xlabel="t-SNE dimension 1 (arbitrary units)", ylabel="t-SNE dimension 2 (arbitrary units)",
               title=f"perplexity = {p}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, markerscale=2.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_fig(fig, "tsne_perplexity", f"t-SNE of a stratified sample of {len(sample):,} training rows at three perplexities, coloured by target")

    fig, axes = plt.subplots(1, len(N_NEIGHBORS), figsize=(15, 5.2))
    for ax, k in zip(axes, N_NEIGHBORS):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # "n_jobs overridden to 1 by random_state"
            emb[f"umap_{k}"] = umap.UMAP(n_components=2, n_neighbors=k, min_dist=UMAP_MIN_DIST,
                                         random_state=RANDOM_STATE).fit_transform(Zs)
        scatter_by_target(ax, emb[f"umap_{k}"], ys)
        ax.set(xlabel="UMAP dimension 1 (arbitrary units)", ylabel="UMAP dimension 2 (arbitrary units)",
               title=f"n_neighbors = {k}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=2, markerscale=2.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_fig(fig, "umap_n_neighbors", f"UMAP of the same sample of {len(sample):,} training rows at three n_neighbors values, coloured by target")

    # Figure: what the islands are. Same embeddings, coloured by the two features behind the PCA axes.
    Xs = X_train.iloc[sample]
    group = np.where(Xs.pdays.values != -1, "previously contacted",
                     np.where(Xs.contact.values == "unknown", "never contacted, contact = unknown",
                              "never contacted, contact known"))
    palette = {"never contacted, contact known": C_NO, "never contacted, contact = unknown": C_MUTED,
               "previously contacted": C_YES}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for ax, key, label in [(axes[0], f"tsne_{PERPLEXITIES[1]}", f"t-SNE, perplexity = {PERPLEXITIES[1]}"),
                           (axes[1], f"umap_{N_NEIGHBORS[1]}", f"UMAP, n_neighbors = {N_NEIGHBORS[1]}")]:
        for g, colour in palette.items():
            m = group == g
            ax.scatter(emb[key][m, 0], emb[key][m, 1], s=5, alpha=0.6, color=colour, edgecolors="none",
                       label=f"{g} (n = {m.sum():,})", rasterized=True)
        ax.set(xlabel="Embedding dimension 1 (arbitrary units)", ylabel="Embedding dimension 2 (arbitrary units)", title=label)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", ncol=3, markerscale=2.5, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 0.93))
    save_fig(fig, "embeddings_by_contact_history", "The same t-SNE and UMAP embeddings, coloured by contact history instead of target")
    for g in palette:
        m = group == g
        key = "grp_" + re.sub(r"\W+", "_", g).strip("_")
        N[key + "_n"], N[key + "_rate"] = fmt_int(m.sum()), pct(ys[m].mean(), 1)

    out = pd.DataFrame({"row_id": Zdf.index[sample], "y": ys, "group": group})
    for k, v in emb.items():
        out[f"{k}_1"], out[f"{k}_2"] = v[:, 0], v[:, 1]
    out.to_csv(OUT_DIR / "embeddings_sample.csv", index=False, float_format="%.4f")


# ----------------------------------------------------------------- render
def render():
    import matplotlib as mpl
    import sklearn
    N["versions"] = (f"Python {sys.version.split()[0]}, pandas {pd.__version__}, numpy {np.__version__}, "
                     f"scikit-learn {sklearn.__version__}, umap-learn {umap.__version__}, matplotlib {mpl.__version__}")
    N["figure_list"] = "\n".join(f"| {n} | `part2_figures/{f}` | {t} |" for n, f, t in FIGS)
    flat = {k: (v if isinstance(v, str) else str(v)) for k, v in N.items()}
    flat.update({f"table_{k}": v for k, v in T.items()})
    write_json({k: v for k, v in flat.items() if not k.startswith("table_") and not k.endswith("_md")},
               "part2_numbers.json")
    if not TEMPLATE.exists():
        print(f"Template not found ({TEMPLATE}); skipped report rendering.")
        return
    draft = P1_DRAFT.read_text(encoding="utf-8")
    if P1_START not in draft or P1_END not in draft:
        raise ValueError(f"{P1_DRAFT.name} no longer has the expected section markers.")
    flat["part1_body"] = draft[draft.index(P1_START):draft.index(P1_END)].rstrip().rstrip("-").rstrip()
    flat["part1_figure_list"] = "\n".join(
        f"| {m.group(1)} | `part1_figures/{m.group(3)}` | {m.group(2)} |"
        for m in re.finditer(r"!\[Figure (\d+)\. (.+?)\]\(part1_figures/(.+?)\)", flat["part1_body"]))
    text = TEMPLATE.read_text(encoding="utf-8")
    missing = sorted(set(re.findall(r"\{\{(\w+)\}\}", text)) - set(flat))
    if missing:
        raise KeyError(f"Template placeholders without computed values: {missing}")
    INDEX.write_text(re.sub(r"\{\{(\w+)\}\}", lambda m: flat[m.group(1)], text), encoding="utf-8", newline="\n")
    print(f"Rendered {INDEX.relative_to(ROOT)}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = load_and_check()
    pre, Zdf, Zte = preprocess(X_train, X_test, y_train)
    # Reproducible stratified sample of training rows for the 2D plots.
    sample, _ = train_test_split(np.arange(len(Zdf)), train_size=SAMPLE_SIZE, stratify=y_train,
                                 random_state=RANDOM_STATE)
    sample = np.sort(sample)
    pca_stage(Zdf, y_train, sample)
    manifold_stage(Zdf, X_train, y_train, sample)
    render()


if __name__ == "__main__":
    main()
