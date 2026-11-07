"""Step 6 - error analysis on the seed=42 test predictions saved by Step 5:
error per subgroup (season, precipitation regime - the EDA's own risk #6,
and holiday), the worst examples with hypotheses, and permutation importance
compared against the EDA's own correlation findings (temperature, hour).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from regression_common import fit_pipeline, load_train_val_test, regression_metrics
from regression_model import MLPRegressor, predict, train_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "regression" / "figures"
OUT_DIR = ROOT / "docs" / "projects" / "regression" / "outputs"

BEST_HIDDEN_SIZES = [64, 32]
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20
N_PERMUTATIONS = 20


def precipitation_regime(row):
    if row["rainfall"] > 0:
        return "rain"
    if row["snowfall"] > 0:
        return "snow"
    return "dry"


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    n_features = Z_train2.shape[1]
    feature_names = pre.get_feature_names_out()

    # retrain seed=42 (the one Figures 2-3 came from) to get a live model for permutation importance
    model, _ = train_model(
        n_features, BEST_HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
        dropout=0.0, batchnorm=False, lr=LR, weight_decay=0.0,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42,
    )
    test_pred = predict(model, Z_test)

    # --- subgroup errors ---
    meta = X_test.reset_index(drop=True).copy()
    meta["precipitation"] = meta.apply(precipitation_regime, axis=1)
    meta["abs_error"] = np.abs(y_test.values - test_pred)
    meta["actual"] = y_test.values
    meta["predicted"] = test_pred

    subgroup_rows = []
    for col in ["seasons", "precipitation", "holiday"]:
        for level, group in meta.groupby(col):
            subgroup_rows.append({
                "group_col": col, "level": level, "n": len(group),
                "mae": float(group["abs_error"].mean()),
            })
    subgroup_df = pd.DataFrame(subgroup_rows).sort_values("mae", ascending=False)
    print("--- Subgroup MAE (sorted, worst first) ---")
    print(subgroup_df.to_string(index=False))
    worst_subgroup = subgroup_df.iloc[0]

    # --- worst 10 examples ---
    worst = meta.sort_values("abs_error", ascending=False).head(10)
    worst_cols = ["seasons", "precipitation", "holiday", "hour", "temperature", "actual", "predicted", "abs_error"]
    print("\n--- 10 worst examples ---")
    print(worst[worst_cols].to_string(index=False))

    # --- permutation importance (manual: shuffle each preprocessed column, measure RMSE increase) ---
    rng = np.random.default_rng(42)
    base_rmse = regression_metrics(y_test, test_pred)["rmse"]
    importances = {}
    for j, name in enumerate(feature_names):
        increases = []
        for _ in range(N_PERMUTATIONS):
            Z_perm = Z_test.copy()
            Z_perm[:, j] = rng.permutation(Z_perm[:, j])
            perm_pred = predict(model, Z_perm)
            perm_rmse = regression_metrics(y_test, perm_pred)["rmse"]
            increases.append(perm_rmse - base_rmse)
        importances[name] = float(np.mean(increases))

    importance_series = pd.Series(importances).sort_values(ascending=False)
    print("\n--- Permutation importance (RMSE increase when shuffled, top 10) ---")
    print(importance_series.head(10).to_string())

    with open(OUT_DIR / "step6_error_analysis.json", "w") as f:
        json.dump({
            "subgroups": subgroup_df.to_dict(orient="records"),
            "worst_examples": worst[worst_cols].to_dict(orient="records"),
            "permutation_importance": importance_series.to_dict(),
            "base_test_rmse": base_rmse,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
