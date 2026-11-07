"""Step 6 - error analysis: error per subgroup (poutcome, month, job - the
EDA's own three strongest categorical drivers, 55.4/44.2/21.6 pp subscription-
rate spread respectively), the worst examples with hypotheses, and permutation
importance compared against those EDA findings.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from classification_common import best_threshold_by_f1, classification_metrics, fit_pipeline, load_train_val_test
from classification_model import MLPClassifier, predict_proba, train_model

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "projects" / "classification" / "outputs"

BEST_HIDDEN_SIZES = [32, 16]
BEST_DROPOUT = 0.2
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15
N_PERMUTATIONS = 10


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    n_features = Z_train2.shape[1]
    feature_names = pre.get_feature_names_out()

    model, _ = train_model(
        n_features, BEST_HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
        dropout=BEST_DROPOUT, batchnorm=False, lr=LR, weight_decay=0.0,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42,
    )
    val_proba = predict_proba(model, Z_val)
    threshold = best_threshold_by_f1(y_val.values, val_proba)
    test_proba = predict_proba(model, Z_test)
    test_pred = (test_proba >= threshold).astype(int)
    print(f"Threshold (max F1 on validation): {threshold:.4f}")

    # --- subgroup errors ---
    meta = X_test.reset_index(drop=True).copy()
    meta["actual"] = y_test.values
    meta["proba"] = test_proba
    meta["pred"] = test_pred
    meta["correct"] = meta["actual"] == meta["pred"]
    meta["error_type"] = np.select(
        [(~meta["correct"]) & (meta["actual"] == 1), (~meta["correct"]) & (meta["actual"] == 0)],
        ["false_negative", "false_positive"], default="correct")

    subgroup_rows = []
    for col in ["poutcome", "month", "job"]:
        for level, group in meta.groupby(col):
            subgroup_rows.append({
                "group_col": col, "level": level, "n": len(group),
                "error_rate": float((~group["correct"]).mean()),
                "actual_positive_rate": float(group["actual"].mean()),
                "recall_1": float(group.loc[group["actual"] == 1, "pred"].mean()) if (group["actual"] == 1).any() else None,
            })
    subgroup_df = pd.DataFrame(subgroup_rows).sort_values("error_rate", ascending=False)
    print("\n--- Subgroup error rate (sorted, worst first, top 15) ---")
    print(subgroup_df.head(15).to_string(index=False))

    # --- worst (most confident wrong) examples ---
    meta["confidence"] = np.where(meta["pred"] == 1, meta["proba"], 1 - meta["proba"])
    worst = meta[~meta["correct"]].sort_values("confidence", ascending=False).head(10)
    worst_cols = ["poutcome", "month", "job", "actual", "pred", "proba", "error_type"]
    print("\n--- 10 most confident wrong predictions ---")
    print(worst[worst_cols].to_string(index=False))

    # --- permutation importance (manual: shuffle each preprocessed column, measure PR-AUC drop) ---
    rng = np.random.default_rng(42)
    base_pr_auc = classification_metrics(y_test, test_proba, threshold)["pr_auc"]
    importances = {}
    for j, name in enumerate(feature_names):
        decreases = []
        for _ in range(N_PERMUTATIONS):
            Z_perm = Z_test.copy()
            Z_perm[:, j] = rng.permutation(Z_perm[:, j])
            perm_proba = predict_proba(model, Z_perm)
            perm_pr_auc = classification_metrics(y_test, perm_proba, threshold)["pr_auc"]
            decreases.append(base_pr_auc - perm_pr_auc)
        importances[name] = float(np.mean(decreases))

    importance_series = pd.Series(importances).sort_values(ascending=False)
    print("\n--- Permutation importance (PR-AUC drop when shuffled, top 12) ---")
    print(importance_series.head(12).to_string())

    with open(OUT_DIR / "step6_error_analysis.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "subgroups": subgroup_df.to_dict(orient="records"),
            "worst_examples": worst[worst_cols].to_dict(orient="records"),
            "permutation_importance": importance_series.to_dict(),
            "base_test_pr_auc": base_pr_auc,
        }, f, indent=2, default=str)


if __name__ == "__main__":
    main()
