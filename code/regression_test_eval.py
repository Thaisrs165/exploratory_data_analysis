"""Step 5 - the test set, touched exactly once, here. The Step 4 winner is
retrained from 3 different seeds (on train+val refit of the preprocessor is
NOT redone - the pipeline and the inner train/val split stay exactly as in
Steps 1-4; only the network's own weight-init/shuffle seed varies), and its
mean +/- std test metrics are compared against the baselines (refit once,
deterministic) in a single table. Also produces Figures 2 (predicted vs.
actual) and 3 (residuals vs. predicted) from the seed=42 run.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from regression_baselines import build_baselines
from regression_common import fit_pipeline, load_train_val_test, regression_metrics
from regression_model import predict, train_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "regression" / "figures"
OUT_DIR = ROOT / "docs" / "projects" / "regression" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

BEST_HIDDEN_SIZES = [64, 32]
BEST_DROPOUT = 0.0
BEST_BATCHNORM = False
BEST_WEIGHT_DECAY = 0.0
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20
SEEDS = [42, 43, 44]

# Step 0 success criterion, fixed before looking at the test set.
SUCCESS_CRITERION = "Network test MAE at least 15% lower than the Ridge (linear) baseline's test MAE"


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    Z_test = pre.transform(X_test)
    n_features = Z_train2.shape[1]

    # --- baselines, refit once, evaluated on test (same fitted objects style as Step 1) ---
    baselines = build_baselines()
    baseline_test_metrics = {}
    for name, model in baselines.items():
        model.fit(Z_train2, y_train2)
        pred = model.predict(Z_test)
        baseline_test_metrics[name] = regression_metrics(y_test, pred)
        print(f"[baseline:{name}] test: " + ", ".join(f"{k}={v:.4f}" for k, v in baseline_test_metrics[name].items()))

    # --- network, 3 seeds ---
    seed_metrics = []
    seed_models = {}
    for seed in SEEDS:
        model, history = train_model(
            n_features, BEST_HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
            dropout=BEST_DROPOUT, batchnorm=BEST_BATCHNORM, lr=LR, weight_decay=BEST_WEIGHT_DECAY,
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=seed,
        )
        test_pred = predict(model, Z_test)
        m = regression_metrics(y_test, test_pred)
        m["best_epoch"] = history["best_epoch"]
        seed_metrics.append(m)
        seed_models[seed] = (model, test_pred)
        print(f"[network seed={seed}] test: " + ", ".join(f"{k}={v:.4f}" for k, v in m.items()))

    mean_metrics = {k: float(np.mean([m[k] for m in seed_metrics])) for k in ["mae", "rmse", "r2", "mape"]}
    std_metrics = {k: float(np.std([m[k] for m in seed_metrics])) for k in ["mae", "rmse", "r2", "mape"]}
    print("\nNetwork test metrics, mean (std) over 3 seeds:")
    for k in mean_metrics:
        print(f"  {k}: {mean_metrics[k]:.4f} ({std_metrics[k]:.4f})")

    ridge_mae = baseline_test_metrics["ridge"]["mae"]
    improvement = (ridge_mae - mean_metrics["mae"]) / ridge_mae
    success = improvement >= 0.15
    print(f"\n{SUCCESS_CRITERION}")
    print(f"Ridge test MAE={ridge_mae:.4f}, network mean test MAE={mean_metrics['mae']:.4f}, "
          f"improvement={improvement * 100:.2f}% -> {'MET' if success else 'NOT MET'}")

    rf_mae = baseline_test_metrics["random_forest"]["mae"]
    print(f"(for context only, not the success criterion) vs. random forest test MAE={rf_mae:.4f}: "
          f"network is {'better' if mean_metrics['mae'] < rf_mae else 'worse'}")

    # --- Figures 2 & 3, from the seed=42 run ---
    model_42, test_pred_42 = seed_models[42]
    residuals_42 = y_test.values - test_pred_42

    plt.figure(figsize=(6.5, 6.5))
    lims = [0, max(y_test.max(), test_pred_42.max()) * 1.05]
    plt.scatter(y_test, test_pred_42, s=8, alpha=0.3, color="#4C72B0")
    plt.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
    plt.xlim(lims); plt.ylim(lims)
    plt.title("Figure 2 - Predicted vs. actual, test set (seed=42)")
    plt.xlabel("Actual rented_bike_count")
    plt.ylabel("Predicted rented_bike_count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_pred_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 5.5))
    plt.scatter(test_pred_42, residuals_42, s=8, alpha=0.3, color="#DD8452")
    plt.axhline(0, color="black", linewidth=1)
    plt.title("Figure 3 - Residuals vs. predicted, test set (seed=42)")
    plt.xlabel("Predicted rented_bike_count")
    plt.ylabel("Residual (actual - predicted)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_residuals.png", dpi=150)
    plt.close()

    with open(OUT_DIR / "step5_test_eval.json", "w") as f:
        json.dump({
            "baseline_test_metrics": baseline_test_metrics,
            "seed_metrics": seed_metrics,
            "mean_metrics": mean_metrics,
            "std_metrics": std_metrics,
            "success_criterion": SUCCESS_CRITERION,
            "ridge_mae": ridge_mae,
            "improvement_pct": improvement * 100,
            "success": success,
        }, f, indent=2)


if __name__ == "__main__":
    main()
