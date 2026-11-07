"""Step 5 - the test set, touched exactly once, here. The Step 4 winner is
retrained from 3 different seeds; the classification threshold is chosen on
validation (maximizing F1), BEFORE the test set is looked at, then applied
unchanged to test. Baselines (refit once, deterministic) are evaluated on
test for the same single comparison table. Also produces Figure 2 (ROC) and
Figure 3 (precision-recall curve) from the seed=42 run.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve

from classification_baselines import build_baselines
from classification_common import best_threshold_by_f1, classification_metrics, fit_pipeline, load_train_val_test
from classification_model import predict_proba, train_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "classification" / "figures"
OUT_DIR = ROOT / "docs" / "projects" / "classification" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_HIDDEN_SIZES = [32, 16]
BEST_DROPOUT = 0.2
BEST_BATCHNORM = False
BEST_WEIGHT_DECAY = 0.0
LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15
SEEDS = [42, 43, 44]

SUCCESS_CRITERION = "Network test PR-AUC at least 15% higher (relative) than Logistic Regression's test PR-AUC"


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    n_features = Z_train2.shape[1]

    # --- baselines, refit once; threshold also chosen on validation (max F1), same recipe as the network ---
    baselines = build_baselines()
    baseline_test_metrics = {}
    for name, model in baselines.items():
        model.fit(Z_train2, y_train2)
        val_proba = model.predict_proba(Z_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Z_val)
        # the dummy's score is constant, so an F1-optimal threshold is a degenerate,
        # meaningless search; keep it at the conventional 0.5 instead.
        baseline_threshold = 0.5 if name == "dummy_most_frequent" else best_threshold_by_f1(y_val.values, val_proba)
        test_proba = model.predict_proba(Z_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Z_test)
        baseline_test_metrics[name] = classification_metrics(y_test, test_proba, threshold=baseline_threshold)
        print(f"[baseline:{name}] threshold={baseline_threshold:.4f} test pr_auc={baseline_test_metrics[name]['pr_auc']:.4f} "
              f"roc_auc={baseline_test_metrics[name]['roc_auc']:.4f}")

    # --- network, 3 seeds ---
    seed_metrics = []
    seed_models = {}
    chosen_threshold = None
    for seed in SEEDS:
        model, history = train_model(
            n_features, BEST_HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
            dropout=BEST_DROPOUT, batchnorm=BEST_BATCHNORM, lr=LR, weight_decay=BEST_WEIGHT_DECAY,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
        )
        val_proba = predict_proba(model, Z_val)
        threshold = best_threshold_by_f1(y_val.values, val_proba)
        if seed == 42:
            chosen_threshold = threshold  # the threshold used in the report's headline table

        test_proba = predict_proba(model, Z_test)
        m = classification_metrics(y_test, test_proba, threshold=threshold)
        m["best_epoch"] = history["best_epoch"]
        seed_metrics.append(m)
        seed_models[seed] = (model, test_proba, threshold)
        print(f"[network seed={seed}] threshold={threshold:.4f} test pr_auc={m['pr_auc']:.4f} "
              f"roc_auc={m['roc_auc']:.4f} macro_f1={m['macro_f1']:.4f}")

    mean_metrics = {k: float(np.mean([m[k] for m in seed_metrics]))
                    for k in ["pr_auc", "roc_auc", "macro_f1", "precision_1", "recall_1", "f1_1"]}
    std_metrics = {k: float(np.std([m[k] for m in seed_metrics]))
                   for k in ["pr_auc", "roc_auc", "macro_f1", "precision_1", "recall_1", "f1_1"]}
    print("\nNetwork test metrics, mean (std) over 3 seeds:")
    for k in mean_metrics:
        print(f"  {k}: {mean_metrics[k]:.4f} ({std_metrics[k]:.4f})")

    logreg_pr_auc = baseline_test_metrics["logistic_regression"]["pr_auc"]
    improvement = (mean_metrics["pr_auc"] - logreg_pr_auc) / logreg_pr_auc
    success = improvement >= 0.15
    print(f"\n{SUCCESS_CRITERION}")
    print(f"LogReg test PR-AUC={logreg_pr_auc:.4f}, network mean test PR-AUC={mean_metrics['pr_auc']:.4f}, "
          f"improvement={improvement * 100:.2f}% -> {'MET' if success else 'NOT MET'}")

    rf_pr_auc = baseline_test_metrics["random_forest"]["pr_auc"]
    print(f"(context only, not the success criterion) vs. random forest test PR-AUC={rf_pr_auc:.4f}: "
          f"network is {'better' if mean_metrics['pr_auc'] > rf_pr_auc else 'worse'}")

    # --- Figures 2 & 3, seed=42 run ---
    model_42, test_proba_42, thr_42 = seed_models[42]
    fpr, tpr, _ = roc_curve(y_test, test_proba_42)
    precision, recall, _ = precision_recall_curve(y_test, test_proba_42)

    plt.figure(figsize=(6.5, 6.5))
    plt.plot(fpr, tpr, color="#4C72B0", label=f"Network (AUC={seed_metrics[0]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Chance")
    plt.title("Figure 2 - ROC curve, test set (seed=42)")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_roc_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(6.5, 6.5))
    plt.plot(recall, precision, color="#DD8452", label=f"Network (AP={seed_metrics[0]['pr_auc']:.3f})")
    plt.axhline(y_test.mean(), color="k", linestyle="--", linewidth=1, label=f"Chance (positive rate={y_test.mean():.3f})")
    plt.title("Figure 3 - Precision-recall curve, test set (seed=42)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_pr_curve.png", dpi=150)
    plt.close()

    with open(OUT_DIR / "step5_test_eval.json", "w") as f:
        json.dump({
            "baseline_test_metrics": baseline_test_metrics,
            "seed_metrics": seed_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "chosen_threshold_seed42": chosen_threshold,
            "success_criterion": SUCCESS_CRITERION,
            "logreg_pr_auc": logreg_pr_auc,
            "improvement_pct": improvement * 100,
            "success": success,
        }, f, indent=2)


if __name__ == "__main__":
    main()
