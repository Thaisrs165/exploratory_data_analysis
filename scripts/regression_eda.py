"""Regression EDA - stages 1-5 for UCI Seoul Bike Sharing Demand.

Computes every number used in docs/projects/eda-regression/index.md and saves
the figures and machine-readable outputs. All numbers quoted in the report are
in outputs/numbers.json. No model is trained.

Usage (from the repository root, after scripts/regression_download_data.py):
    python scripts/regression_eda.py

Outputs:
    docs/projects/eda-regression/figures/   numbered PNG figures
    docs/projects/eda-regression/outputs/   CSV / JSON tables
"""

import json
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
from bike_preprocessing import (  # noqa: E402
    CATEGORICAL, CLIP_UPPER, HEAVY_TAIL, RANDOM_STATE, TARGET, WEEKDAYS,
    build_preprocessor, clip_bounds, clip_report, load_data, read_raw, split,
)

DOCS = ROOT / "docs" / "projects" / "eda-regression"
FIG_DIR = DOCS / "figures"
OUT_DIR = DOCS / "outputs"

NUMERICAL = ["hour", "temperature", "humidity", "wind_speed", "visibility",
             "dew_point", "solar_radiation", "rainfall", "snowfall"]
UNITS = {"hour": "h", "temperature": "°C", "humidity": "%", "wind_speed": "m/s",
         "visibility": "10 m", "dew_point": "°C", "solar_radiation": "MJ/m²",
         "rainfall": "mm", "snowfall": "cm", TARGET: "bikes/h"}
SEASONS = ["Winter", "Spring", "Summer", "Autumn"]

# dataviz reference palette: categorical slots 1-4, blue sequential ramp, ink/grid tokens.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SEASON_COLORS = dict(zip(SEASONS, [C1, C3, C2, C4]))
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list("div", ["#e34948", "#f0efec", C1])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "axes.labelcolor": INK2, "axes.edgecolor": GRID, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "figure.titlesize": 12, "figure.titleweight": "bold",
})

N = {}   # every number quoted in the report


def save_fig(fig, name):
    fig.savefig(FIG_DIR / name)
    plt.close(fig)


def save_csv(df, name, index=True):
    df.to_csv(OUT_DIR / name, index=index, float_format="%.4f")


def r(x, d=4):
    return round(float(x), d)


def box(ax, groups, labels, colors, vert=True):
    bp = ax.boxplot(groups, tick_labels=labels, widths=0.55, patch_artist=True,
                    orientation="vertical" if vert else "horizontal",
                    flierprops=dict(marker="o", markersize=2.5, alpha=0.35, markeredgewidth=0),
                    medianprops=dict(color=INK, linewidth=1.4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
        patch.set_edgecolor(INK2)
    for fl, c in zip(bp["fliers"], colors):
        fl.set_markerfacecolor(c)
    return bp


# ----------------------------------------------------------------- stage 1
def stage1(raw):
    N["rows_raw"], N["cols_raw"] = raw.shape[0], raw.shape[1] - 1   # minus derived day_of_week
    N["date_min"], N["date_max"] = str(raw["date"].min().date()), str(raw["date"].max().date())
    N["n_days"] = int(raw["date"].nunique())
    N["rows_per_day"] = sorted(raw.groupby("date").size().unique().tolist())

    # 1B quality
    src = raw.drop(columns="day_of_week")
    N["nan_total"] = int(src.isna().sum().sum())
    N["duplicate_rows"] = int(src.duplicated().sum())
    N["duplicate_date_hour"] = int(src.duplicated(["date", "hour"]).sum())
    checks = {
        "hour outside 0-23": int((~raw["hour"].between(0, 23)).sum()),
        "humidity outside 0-100": int((~raw["humidity"].between(0, 100)).sum()),
        "humidity == 0": int((raw["humidity"] == 0).sum()),
        "negative wind/visibility/solar/rain/snow": int((raw[["wind_speed", "visibility", "solar_radiation",
                                                               "rainfall", "snowfall"]] < 0).any(axis=1).sum()),
        "dew_point > temperature": int((raw["dew_point"] > raw["temperature"] + 1e-9).sum()),
        "solar_radiation > 0 between 22h and 4h": int(((raw["hour"] >= 22) | (raw["hour"] <= 4))
                                                      .__and__(raw["solar_radiation"] > 0).sum()),
        "snowfall > 0 with temperature > 5 °C": int(((raw["snowfall"] > 0) & (raw["temperature"] > 5)).sum()),
        "season inconsistent with month": int((raw["seasons"] != raw["date"].dt.month.map(
            {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
             6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"})).sum()),
        "visibility at the 2000 (20 km) ceiling": int((raw["visibility"] == 2000).sum()),
        "rented_bike_count == 0": int((raw[TARGET] == 0).sum()),
    }
    save_csv(pd.Series(checks, name="rows").to_frame(), "consistency_checks.csv")
    N["checks"] = checks
    N["humidity_zero_dates"] = sorted(raw.loc[raw["humidity"] == 0, "date"].dt.date.astype(str).unique().tolist())

    # Leakage: functioning_day
    fd = raw.groupby("functioning_day")[TARGET].agg(rows="count", total="sum", mean="mean",
                                                     min="min", max="max")
    save_csv(fd, "functioning_day_vs_target.csv")
    N["nonfunc_rows"] = int(fd.loc["No", "rows"])
    N["nonfunc_pct"] = r(100 * fd.loc["No", "rows"] / len(raw), 2)
    N["nonfunc_max_target"] = int(fd.loc["No", "max"])
    N["zero_target_rows"] = int((raw[TARGET] == 0).sum())
    N["zero_target_in_functioning"] = int(((raw[TARGET] == 0) & (raw["functioning_day"] == "Yes")).sum())
    N["nonfunc_days_full"] = int((raw[raw["functioning_day"] == "No"].groupby("date").size() == 24).sum())
    N["nonfunc_days_any"] = int(raw.loc[raw["functioning_day"] == "No", "date"].nunique())

    # Data dictionary
    dd = pd.DataFrame({
        "column": raw.columns,
        "dtype": [str(t) for t in raw.dtypes],
        "n_unique": [raw[c].nunique() for c in raw.columns],
        "unit": [UNITS.get(c, "") for c in raw.columns],
    })
    save_csv(dd, "data_dictionary.csv", index=False)

    # 1C target on the retained rows (functioning hours) - full retained set
    X, y = load_data()
    N["rows_retained"] = len(X)
    yt = describe_target(y, "all")
    N["target_all"] = yt

    # 1D split
    X_train, X_test, y_train, y_test = split(X, y)
    N["n_train"], N["n_test"] = len(X_train), len(X_test)
    N["test_pct"] = r(100 * len(X_test) / len(X), 2)
    N["days_train"], N["days_test"] = int(X_train["date"].nunique()), int(X_test["date"].nunique())
    N["days_overlap"] = len(set(X_train["date"]) & set(X_test["date"]))
    season_split = pd.DataFrame({
        "train_rows": X_train["seasons"].value_counts(), "test_rows": X_test["seasons"].value_counts()
    }).reindex(SEASONS)
    season_split["train_pct"] = 100 * season_split["train_rows"] / len(X_train)
    season_split["test_pct"] = 100 * season_split["test_rows"] / len(X_test)
    season_split["train_mean_target"] = y_train.groupby(X_train["seasons"]).mean()
    season_split["test_mean_target"] = y_test.groupby(X_test["seasons"]).mean()
    save_csv(season_split, "split_by_season.csv")
    N["target_train"] = describe_target(y_train, "train")
    N["target_test"] = describe_target(y_test, "test")
    pd.DataFrame({"row_id": X.index, "date": X["date"].dt.date,
                  "split": np.where(X.index.isin(X_test.index), "test", "train")}) \
        .to_csv(OUT_DIR / "split_indices.csv", index=False)

    fig_target(y_train)
    return X_train, X_test, y_train, y_test


def describe_target(y, name):
    d = {
        "n": int(len(y)), "mean": r(y.mean(), 2), "median": r(y.median(), 2), "std": r(y.std(), 2),
        "min": r(y.min(), 0), "max": r(y.max(), 0), "q1": r(y.quantile(0.25), 2), "q3": r(y.quantile(0.75), 2),
        "p99": r(y.quantile(0.99), 2),
        "skew": r(y.skew(), 3), "kurtosis": r(y.kurt(), 3),
        "skew_log1p": r(np.log1p(y).skew(), 3), "skew_sqrt": r(np.sqrt(y).skew(), 3),
        "zeros": int((y == 0).sum()),
    }
    return d


def fig_target(y_train):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    specs = [(y_train, "raw (bikes/h)"), (np.sqrt(y_train), "sqrt"), (np.log1p(y_train), "log1p")]
    for ax, (v, lab) in zip(axes, specs):
        ax.hist(v, bins=50, color=C1, edgecolor="white", linewidth=0.5)
        ax.axvline(v.mean(), color=C2, lw=1.6, label=f"mean {v.mean():.2f}")
        ax.axvline(v.median(), color=INK, lw=1.6, ls="--", label=f"median {v.median():.2f}")
        ax.set_title(f"{lab}: skewness {v.skew():.2f}")
        ax.set_xlabel(f"rented_bike_count, {lab}")
        ax.set_ylabel("hours (train)")
        ax.legend(loc="upper right")
    fig.suptitle("Figure 1 - Target distribution on the training set: raw, square root and log1p")
    save_fig(fig, "fig01_target_distribution.png")


# ----------------------------------------------------------------- stage 2
def stage2(X_train, y_train):
    tr = X_train.copy()
    desc = tr[NUMERICAL].describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    desc["skew"] = tr[NUMERICAL].skew()
    desc["kurtosis"] = tr[NUMERICAL].kurt()
    desc["zero_pct"] = 100 * (tr[NUMERICAL] == 0).mean()
    q1, q3 = tr[NUMERICAL].quantile(0.25), tr[NUMERICAL].quantile(0.75)
    iqr = q3 - q1
    desc["iqr_outliers_pct"] = 100 * ((tr[NUMERICAL] < q1 - 1.5 * iqr) | (tr[NUMERICAL] > q3 + 1.5 * iqr)).mean()
    desc["unit"] = [UNITS[c] for c in NUMERICAL]
    save_csv(desc, "numerical_summary_train.csv")
    N["numerical_summary"] = {c: {k: r(desc.loc[c, k], 3) for k in
                                  ["mean", "std", "min", "1%", "50%", "99%", "max", "skew", "zero_pct",
                                   "iqr_outliers_pct"]} for c in NUMERICAL}
    N["visibility_ceiling_pct_train"] = r(100 * (tr["visibility"] == 2000).mean(), 2)
    N["rain_zero_pct_train"] = r(100 * (tr["rainfall"] == 0).mean(), 2)
    N["snow_zero_pct_train"] = r(100 * (tr["snowfall"] == 0).mean(), 2)
    N["solar_zero_pct_train"] = r(100 * (tr["solar_radiation"] == 0).mean(), 2)

    # Figure 2: histograms
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    for ax, c in zip(axes.ravel(), NUMERICAL):
        bins = 24 if c == "hour" else 50
        ax.hist(tr[c], bins=bins, color=C1, edgecolor="white", linewidth=0.4)
        ax.axvline(tr[c].median(), color=INK, lw=1.4, ls="--", label=f"median {tr[c].median():.2f}")
        ax.set_title(f"{c} (skew {tr[c].skew():.2f})")
        ax.set_xlabel(f"{c} ({UNITS[c]})")
        ax.set_ylabel("hours")
        ax.legend(loc="upper right", fontsize=8)
        if c in ("rainfall", "snowfall", "solar_radiation"):
            ax.set_yscale("log")
            ax.set_ylabel("hours (log scale)")
    fig.suptitle("Figure 2 - Distributions of the numerical features (training set)")
    fig.tight_layout()
    save_fig(fig, "fig02_hist_numerical.png")

    # Figure 3: standardized boxplots
    zs = (tr[NUMERICAL] - tr[NUMERICAL].mean()) / tr[NUMERICAL].std()
    fig, ax = plt.subplots(figsize=(11, 4.2))
    box(ax, [zs[c] for c in NUMERICAL], NUMERICAL, [C1] * len(NUMERICAL))
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("z-score (train mean / std)")
    ax.set_xlabel("feature")
    ax.tick_params(axis="x", rotation=20)
    ax.set_title("Figure 3 - Standardized boxplots: tails and outliers by feature (training set)")
    save_fig(fig, "fig03_boxplots_numerical.png")

    # 2B categorical
    raw = read_raw()
    rows = []
    for c in CATEGORICAL + ["functioning_day"]:
        src = raw[c] if c == "functioning_day" else tr[c]
        vc = src.value_counts()
        for lvl, n in vc.items():
            rows.append({"feature": c, "level": lvl, "count": int(n), "pct": 100 * n / len(src),
                         "set": "full data" if c == "functioning_day" else "train"})
    freq = pd.DataFrame(rows)
    save_csv(freq, "categorical_frequencies.csv", index=False)
    card = freq.groupby("feature").agg(levels=("level", "count"), min_pct=("pct", "min"))
    save_csv(card, "categorical_cardinality.csv")
    N["categorical_freq"] = {f: {row.level: [row.count, r(row.pct, 2)] for row in g.itertuples()}
                             for f, g in freq.groupby("feature")}
    N["holiday_days_train"] = int(tr.loc[tr["holiday"] == "Holiday", "date"].nunique())

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), gridspec_kw=dict(width_ratios=[4, 2, 7, 2]))
    order = {"seasons": SEASONS, "holiday": ["No Holiday", "Holiday"], "day_of_week": WEEKDAYS,
             "functioning_day": ["Yes", "No"]}
    for ax, c in zip(axes, ["seasons", "holiday", "day_of_week", "functioning_day"]):
        g = freq[freq["feature"] == c].set_index("level").reindex(order[c])
        bars = ax.bar(g.index, g["pct"], color=C1, width=0.7)
        for b, p in zip(bars, g["pct"]):
            ax.annotate(f"{p:.1f}%", (b.get_x() + b.get_width() / 2, b.get_height()),
                        ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(c + (" (full data)" if c == "functioning_day" else ""))
        ax.set_ylabel("% of rows")
        ax.set_xlabel("level")
        ax.axhline(5, color=C2, lw=1, ls=":", label="5% rare threshold")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="lower left")
    fig.suptitle("Figure 4 - Categorical frequencies (training set; functioning_day on the full data)")
    fig.tight_layout()
    save_fig(fig, "fig04_categorical_frequencies.png")


# ----------------------------------------------------------------- stage 3
def stage3(X_train, y_train):
    tr = X_train.copy()
    tr[TARGET] = y_train.values
    cols = NUMERICAL + [TARGET]
    pear, spear = tr[cols].corr("pearson"), tr[cols].corr("spearman")
    save_csv(pear, "corr_pearson_train.csv")
    save_csv(spear, "corr_spearman_train.csv")
    pairs = []
    for i, a in enumerate(NUMERICAL):
        for b in NUMERICAL[i + 1:]:
            pairs.append({"a": a, "b": b, "pearson": pear.loc[a, b], "spearman": spear.loc[a, b]})
    pairs = pd.DataFrame(pairs).assign(abs_spearman=lambda d: d["spearman"].abs()) \
        .sort_values("abs_spearman", ascending=False)
    save_csv(pairs, "corr_pairs_train.csv", index=False)
    N["top_pairs"] = [{k: (r(v, 3) if isinstance(v, float) else v) for k, v in row.items()}
                      for row in pairs.head(5).drop(columns="abs_spearman").to_dict("records")]
    tgt = pd.DataFrame({"pearson": pear[TARGET].drop(TARGET), "spearman": spear[TARGET].drop(TARGET)}) \
        .sort_values("spearman", key=abs, ascending=False)
    save_csv(tgt, "corr_with_target_train.csv")
    N["corr_target"] = {c: [r(tgt.loc[c, "pearson"], 3), r(tgt.loc[c, "spearman"], 3)] for c in tgt.index}

    # Precipitation acts like a switch: medians with and without rain / snow
    for c in ("rainfall", "snowfall"):
        wet = tr[c] > 0
        N[f"{c}_wet_pct_train"] = r(100 * wet.mean(), 2)
        N[f"median_target_{c}_wet"] = r(tr.loc[wet, TARGET].median(), 1)
        N[f"median_target_{c}_dry"] = r(tr.loc[~wet, TARGET].median(), 1)

    # Is dew_point explained by temperature + humidity?
    A = np.column_stack([np.ones(len(tr)), tr["temperature"], tr["humidity"]])
    coef, *_ = np.linalg.lstsq(A, tr["dew_point"], rcond=None)
    resid = tr["dew_point"] - A @ coef
    N["dew_point_r2_from_temp_humidity"] = r(1 - resid.var() / tr["dew_point"].var(), 4)

    # Figure 5: Spearman and Pearson heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8))
    for ax, (m, name) in zip(axes, [(spear, "Spearman"), (pear, "Pearson")]):
        im = ax.imshow(m.values, cmap=DIVERGING, vmin=-1, vmax=1)
        ax.set_xticks(range(len(cols)), cols, rotation=45, ha="right")
        ax.set_yticks(range(len(cols)), cols)
        ax.grid(False)
        for i in range(len(cols)):
            for j in range(len(cols)):
                v = m.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.6 else INK)
        ax.set_title(f"{name}")
    fig.colorbar(im, ax=axes, shrink=0.8, label="correlation coefficient")
    fig.suptitle("Figure 5 - Correlation matrices of numerical features and target (training set)")
    save_fig(fig, "fig05_correlation.png")

    # Figure 6: scatter plots - redundant pair and the two strongest target drivers
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    sc = axes[0].scatter(tr["temperature"], tr["dew_point"], c=tr["humidity"], cmap=SEQ, s=4, alpha=0.6)
    fig.colorbar(sc, ax=axes[0], label="humidity (%)")
    axes[0].set_xlabel("temperature (°C)")
    axes[0].set_ylabel("dew_point (°C)")
    axes[0].set_title(f"temperature vs dew_point (Spearman {spear.loc['temperature', 'dew_point']:.2f})")
    for ax, c in zip(axes[1:], ["temperature", "hour"]):
        jitter = np.random.default_rng(RANDOM_STATE).uniform(-0.35, 0.35, len(tr)) if c == "hour" else 0
        ax.scatter(tr[c] + jitter, tr[TARGET], s=3, alpha=0.25, color=C1, label="one observed hour")
        bins = tr.groupby(tr[c].round(0) if c == "hour" else pd.cut(tr[c], 30), observed=True)[TARGET].median()
        xs = bins.index.astype(float) if c == "hour" else [iv.mid for iv in bins.index]
        ax.plot(xs, bins.values, color=C2, lw=2, label="binned median")
        ax.set_xlabel(f"{c} ({UNITS[c]})")
        ax.set_ylabel("rented_bike_count (bikes/h)")
        ax.set_title(f"{c} vs target (Spearman {spear.loc[c, TARGET]:.2f})")
        ax.legend(loc="upper left")
    fig.suptitle("Figure 6 - Scatter plots: redundant pair and the two strongest predictors of the target")
    fig.tight_layout()
    save_fig(fig, "fig06_scatter.png")

    # 3B categorical x target
    rows = []
    for c in CATEGORICAL:
        g = tr.groupby(c)[TARGET]
        s = pd.DataFrame({"count": g.size(), "mean": g.mean(), "median": g.median(), "std": g.std(),
                          "q1": g.quantile(0.25), "q3": g.quantile(0.75)})
        s.insert(0, "feature", c)
        rows.append(s)
    cat_t = pd.concat(rows)
    save_csv(cat_t, "categorical_target_stats_train.csv")
    N["cat_target"] = {c: {lvl: [r(cat_t.loc[(cat_t["feature"] == c)].loc[lvl, "mean"], 1),
                                 r(cat_t.loc[(cat_t["feature"] == c)].loc[lvl, "median"], 1)]
                           for lvl in cat_t[cat_t["feature"] == c].index} for c in CATEGORICAL}

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=[4, 2, 7]))
    for ax, c in zip(axes, CATEGORICAL):
        levels = {"seasons": SEASONS, "holiday": ["No Holiday", "Holiday"], "day_of_week": WEEKDAYS}[c]
        colors = [SEASON_COLORS[s] for s in SEASONS] if c == "seasons" else [C1] * len(levels)
        box(ax, [tr.loc[tr[c] == lvl, TARGET] for lvl in levels], levels, colors)
        means = [tr.loc[tr[c] == lvl, TARGET].mean() for lvl in levels]
        ax.scatter(range(1, len(levels) + 1), means, marker="D", s=22, color=INK, zorder=3, label="mean")
        ax.set_title(c)
        ax.set_xlabel(c)
        ax.set_ylabel("rented_bike_count (bikes/h)")
        ax.tick_params(axis="x", rotation=30)
    axes[0].legend(loc="upper left")
    fig.suptitle("Figure 7 - Target distribution by category (training set; line = median, diamond = mean)")
    fig.tight_layout()
    save_fig(fig, "fig07_target_by_category.png")

    # Figure 8: hourly profile, working days vs weekend/holiday
    tr["day_type"] = np.where((tr["day_of_week"].isin(["Sat", "Sun"])) | (tr["holiday"] == "Holiday"),
                              "weekend / holiday", "working day")
    prof = tr.groupby(["day_type", "hour"])[TARGET].agg(["median", lambda s: s.quantile(0.25),
                                                         lambda s: s.quantile(0.75)])
    prof.columns = ["median", "q1", "q3"]
    save_csv(prof, "hourly_profile_by_day_type_train.csv")
    wd, we = prof.loc["working day"], prof.loc["weekend / holiday"]
    N["peak_working"] = {"hour": int(wd["median"].idxmax()), "median": r(wd["median"].max(), 0)}
    N["peak_working_morning"] = {"hour": int(wd.loc[5:10, "median"].idxmax()),
                                 "median": r(wd.loc[5:10, "median"].max(), 0)}
    N["peak_weekend"] = {"hour": int(we["median"].idxmax()), "median": r(we["median"].max(), 0)}
    N["we_8h_median"] = r(we.loc[8, "median"], 0)
    N["day_type_counts"] = tr["day_type"].value_counts().to_dict()
    fig, ax = plt.subplots(figsize=(10, 4))
    for name, color in [("working day", C1), ("weekend / holiday", C2)]:
        p = prof.loc[name]
        ax.plot(p.index, p["median"], color=color, lw=2, label=f"{name} (median)")
        ax.fill_between(p.index, p["q1"], p["q3"], color=color, alpha=0.18, label=f"{name} (IQR)")
    ax.set_xticks(range(0, 24, 2))
    ax.set_xlabel("hour of day")
    ax.set_ylabel("rented_bike_count (bikes/h)")
    ax.legend(loc="upper left")
    ax.set_title("Figure 8 - Hourly demand profile: working days vs weekends and holidays (training set)")
    save_fig(fig, "fig08_hourly_profile_day_type.png")

    # 3C numerical x categorical: weather by season
    feats = ["temperature", "humidity", "solar_radiation", "visibility"]
    rows = []
    for f in feats:
        g = tr.groupby("seasons")[f]
        s = pd.DataFrame({"median": g.median(), "q1": g.quantile(0.25), "q3": g.quantile(0.75),
                          "iqr": g.quantile(0.75) - g.quantile(0.25), "std": g.std()}).reindex(SEASONS)
        s.insert(0, "feature", f)
        rows.append(s)
    gb = pd.concat(rows)
    save_csv(gb, "grouped_box_stats_train.csv")
    N["grouped"] = {f: {s: [r(gb[gb["feature"] == f].loc[s, "median"], 2), r(gb[gb["feature"] == f].loc[s, "iqr"], 2)]
                        for s in SEASONS} for f in feats}
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8))
    for ax, f in zip(axes, feats):
        box(ax, [tr.loc[tr["seasons"] == s, f] for s in SEASONS], SEASONS, [SEASON_COLORS[s] for s in SEASONS])
        ax.set_title(f)
        ax.set_xlabel("season")
        ax.set_ylabel(f"{f} ({UNITS[f]})")
    fig.suptitle("Figure 9 - Weather features by season: location and spread (training set)")
    fig.tight_layout()
    save_fig(fig, "fig09_weather_by_season.png")

    # Temperature effect within season (does season add beyond temperature?)
    fig, ax = plt.subplots(figsize=(10, 4))
    for s in SEASONS:
        sub = tr[tr["seasons"] == s]
        b = sub.groupby(pd.cut(sub["temperature"], np.arange(-20, 42, 3)), observed=True)[TARGET].median()
        n = sub.groupby(pd.cut(sub["temperature"], np.arange(-20, 42, 3)), observed=True)[TARGET].size()
        b = b[n >= 20]
        ax.plot([iv.mid for iv in b.index], b.values, marker="o", ms=4, lw=2, color=SEASON_COLORS[s], label=s)
    ax.set_xlabel("temperature (°C, 3 °C bins with >= 20 hours)")
    ax.set_ylabel("median rented_bike_count (bikes/h)")
    ax.legend(title="season")
    ax.set_title("Figure 10 - Median demand vs temperature, by season (training set)")
    save_fig(fig, "fig10_temperature_by_season_target.png")


# ----------------------------------------------------------------- stage 4
def stage4(X_train, X_test, y_train, y_test):
    pre = build_preprocessor().fit(X_train)
    Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)
    names = list(pre.get_feature_names_out())
    pd.Series(names, name="feature").to_csv(OUT_DIR / "feature_names.csv", index_label="index")
    N["shape_in_train"], N["shape_in_test"] = list(X_train.shape), list(X_test.shape)
    N["shape_out_train"], N["shape_out_test"] = list(Z_train.shape), list(Z_test.shape)
    N["nan_out_train"], N["nan_out_test"] = int(np.isnan(Z_train).sum()), int(np.isnan(Z_test).sum())
    N["clip_bounds"] = clip_bounds(pre)
    N["clip_train"], N["clip_test"] = clip_report(pre, X_train), clip_report(pre, X_test)
    N["clip_train_pct"] = r(100 * N["clip_train"]["any"] / len(X_train), 2)
    N["clip_test_pct"] = r(100 * N["clip_test"]["any"] / len(X_test), 2)
    N["humidity_zero_imputed"] = {"train": int((X_train["humidity"] == 0).sum()),
                                  "test": int((X_test["humidity"] == 0).sum())}
    N["humidity_impute_value"] = r(X_train.loc[X_train["humidity"] > 0, "humidity"].median(), 1)
    N["test_only_categories"] = {c: sorted(set(X_test[c]) - set(X_train[c])) for c in CATEGORICAL}
    N["feature_names"] = names

    Zdf = pd.DataFrame(Z_train, columns=names)
    ba = pd.DataFrame({
        "raw_skew": X_train[["temperature", "humidity", "visibility", "wind_speed", "solar_radiation",
                             "rainfall", "snowfall"]].skew(),
    })
    ba["out_skew"] = [Zdf[c].skew() for c in ba.index]
    ba["out_mean"] = [Zdf[c].mean() for c in ba.index]
    ba["out_std"] = [Zdf[c].std(ddof=0) for c in ba.index]
    save_csv(ba, "numerical_before_after_train.csv")
    N["before_after"] = {c: [r(ba.loc[c, "raw_skew"], 2), r(ba.loc[c, "out_skew"], 2)] for c in ba.index}

    fig, axes = plt.subplots(2, 4, figsize=(15, 6))
    for j, c in enumerate(["wind_speed", "solar_radiation", "rainfall", "snowfall"]):
        axes[0, j].hist(X_train[c], bins=40, color=C4, edgecolor="white", linewidth=0.4)
        axes[0, j].set_title(f"{c} raw (skew {ba.loc[c, 'raw_skew']:.2f})")
        axes[0, j].set_xlabel(f"{c} ({UNITS[c]})")
        axes[1, j].hist(Zdf[c], bins=40, color=C1, edgecolor="white", linewidth=0.4)
        axes[1, j].set_title(f"{c} preprocessed (skew {ba.loc[c, 'out_skew']:.2f})")
        axes[1, j].set_xlabel(f"{c} (standardized)")
        for ax in axes[:, j]:
            ax.set_ylabel("hours (log scale)")
            ax.set_yscale("log")
    fig.suptitle("Figure 11 - Skewed features before (top, orange) and after (bottom, blue) the pipeline")
    fig.tight_layout()
    save_fig(fig, "fig11_raw_vs_preprocessed.png")

    # PCA
    pca = PCA(random_state=RANDOM_STATE).fit(Z_train)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained": evr, "cumulative": cum}) \
        .to_csv(OUT_DIR / "pca_explained_variance.csv", index=False, float_format="%.5f")
    load = pd.DataFrame(pca.components_[:3].T, index=names, columns=["PC1", "PC2", "PC3"])
    save_csv(load, "pca_loadings.csv")
    N["pca_pc1"], N["pca_pc2"], N["pca_pc3"] = r(100 * evr[0], 2), r(100 * evr[1], 2), r(100 * evr[2], 2)
    N["pca_pc12"] = r(100 * cum[1], 2)
    N["pca_n80"], N["pca_n90"], N["pca_n95"] = [int(np.searchsorted(cum, t) + 1) for t in (0.8, 0.9, 0.95)]
    N["pca_rank"] = int(np.sum(evr > 1e-10))
    N["pca_top_loadings"] = {pc: [[n, r(load.loc[n, pc], 2)] for n in load[pc].abs().sort_values(ascending=False)
                                  .head(5).index] for pc in ["PC1", "PC2", "PC3"]}
    P = pca.transform(Z_train)[:, :2]
    N["pc1_corr_log_target"] = r(np.corrcoef(P[:, 0], np.log1p(y_train))[0, 1], 3)
    N["pc2_corr_log_target"] = r(np.corrcoef(P[:, 1], np.log1p(y_train))[0, 1], 3)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    k = np.arange(1, len(evr) + 1)
    axes[0].bar(k, 100 * evr, color=C1, label="per component")
    axes[0].plot(k, 100 * cum, color=C2, marker="o", ms=4, lw=2, label="cumulative")
    for t in (80, 90):
        axes[0].axhline(t, color=INK2, lw=0.8, ls=":")
    axes[0].set_xlabel("principal component")
    axes[0].set_ylabel("explained variance (%)")
    axes[0].set_title("Explained variance")
    axes[0].legend(loc="center right")
    top = load.loc[load[["PC1", "PC2"]].abs().max(axis=1).sort_values(ascending=False).head(12).index, ["PC1", "PC2"]]
    yy = np.arange(len(top))
    axes[1].barh(yy - 0.2, top["PC1"], height=0.4, color=C1, label="PC1")
    axes[1].barh(yy + 0.2, top["PC2"], height=0.4, color=C2, label="PC2")
    axes[1].set_yticks(yy, top.index)
    axes[1].invert_yaxis()
    axes[1].axvline(0, color=INK2, lw=0.8)
    axes[1].set_xlabel("loading")
    axes[1].set_title("Top 12 loadings on PC1 and PC2")
    axes[1].legend(loc="lower right")
    fig.suptitle("Figure 12 - PCA on the 22 preprocessed training features")
    fig.tight_layout()
    save_fig(fig, "fig12_pca_variance_loadings.png")

    # Embeddings (full training set: 6.8k rows is small enough), colored by log1p(target)
    color = np.log1p(y_train.values)
    emb = {"pca": P}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for perp in (30, 50):
            ts = TSNE(n_components=2, perplexity=perp, init="pca", random_state=RANDOM_STATE)
            emb[f"tsne_p{perp}"] = ts.fit_transform(Z_train)
            N[f"tsne_kl_{perp}"] = r(ts.kl_divergence_, 3)
        import umap
        for nn in (15, 50):
            emb[f"umap_n{nn}"] = umap.UMAP(n_neighbors=nn, min_dist=0.1, random_state=RANDOM_STATE) \
                .fit_transform(Z_train)
    out = pd.DataFrame({"row_id": X_train.index, "log1p_target": color, "seasons": X_train["seasons"].values,
                        "hour": X_train["hour"].values})
    for k_, v in emb.items():
        out[f"{k_}_1"], out[f"{k_}_2"] = v[:, 0], v[:, 1]
    out.to_csv(OUT_DIR / "embeddings_train.csv", index=False, float_format="%.4f")

    vmin, vmax = np.quantile(color, [0.02, 0.98])   # most hours sit in a narrow log band

    def panel(ax, xy, title, c, cmap):
        s = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=3, alpha=0.8, linewidths=0, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xlabel("dimension 1")
        ax.set_ylabel("dimension 2")
        return s

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    s = panel(axes[0], P, f"PCA (PC1 {N['pca_pc1']:.1f}%, PC2 {N['pca_pc2']:.1f}%)", color, SEQ)
    axes[0].set_xlabel("PC1")
    axes[0].set_ylabel("PC2")
    for ax, perp in zip(axes[1:], (30, 50)):
        panel(ax, emb[f"tsne_p{perp}"], f"t-SNE perplexity {perp} (KL {N[f'tsne_kl_{perp}']:.2f})", color, SEQ)
    fig.colorbar(s, ax=axes, shrink=0.85, extend="both", label="log1p(rented_bike_count), P2-P98")
    fig.suptitle("Figure 13 - PCA and t-SNE (two perplexities) of the training set, colored by log1p(target)")
    save_fig(fig, "fig13_pca_tsne.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, nn in zip(axes, (15, 50)):
        s = panel(ax, emb[f"umap_n{nn}"], f"UMAP n_neighbors {nn}", color, SEQ)
    fig.colorbar(s, ax=axes, shrink=0.85, extend="both", label="log1p(rented_bike_count), P2-P98")
    fig.suptitle("Figure 14 - UMAP (two n_neighbors) of the training set, colored by log1p(target)")
    save_fig(fig, "fig14_umap.png")

    # Figure 15: what drives the structure - same embeddings colored by season and by hour
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.2), gridspec_kw=dict(hspace=0.35))
    for j, (k_, title) in enumerate([("pca", "PCA"), ("tsne_p30", "t-SNE p=30"), ("umap_n15", "UMAP n=15")]):
        xy = emb[k_]
        for s_ in SEASONS:
            m = X_train["seasons"].values == s_
            axes[0, j].scatter(xy[m, 0], xy[m, 1], s=3, alpha=0.6, color=SEASON_COLORS[s_], label=s_, linewidths=0)
        axes[0, j].set_title(f"{title} - by season")
        sc = axes[1, j].scatter(xy[:, 0], xy[:, 1], c=X_train["hour"].values, cmap="twilight", s=3, alpha=0.7,
                                linewidths=0)
        axes[1, j].set_title(f"{title} - by hour")
        for ax in axes[:, j]:
            ax.set_xlabel("dimension 1")
            ax.set_ylabel("dimension 2")
    axes[0, 0].legend(markerscale=4, loc="best")
    fig.colorbar(sc, ax=axes[1, :], shrink=0.85, label="hour of day (cyclic colormap)")
    fig.suptitle("Figure 15 - The same embeddings colored by season (top) and by hour (bottom)")
    save_fig(fig, "fig15_embeddings_by_season_hour.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_raw()
    X_train, X_test, y_train, y_test = stage1(raw)
    stage2(X_train, y_train)
    stage3(X_train, y_train)
    stage4(X_train, X_test, y_train, y_test)
    import importlib.metadata as md
    N["versions"] = {p: md.version(p) for p in ["pandas", "numpy", "scikit-learn", "umap-learn", "matplotlib"]}
    N["versions"]["python"] = sys.version.split()[0]
    (OUT_DIR / "numbers.json").write_text(json.dumps(N, indent=2, default=str) + "\n")
    print(json.dumps(N, indent=1, default=str))


if __name__ == "__main__":
    main()
