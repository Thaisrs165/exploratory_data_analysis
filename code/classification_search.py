"""Step 4 - regularization and hyperparameter search: >=2 regularization
techniques (dropout, weight decay, batch normalization) compared against the
Step 2-3 baseline network, plus two architecture widths, all selected on
validation PR-AUC. 8 configurations total."""
import json
from pathlib import Path

from classification_common import classification_metrics, fit_pipeline, load_train_val_test
from classification_model import predict_proba, train_model

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "projects" / "classification" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    {"name": "baseline (no reg.)", "hidden_sizes": [32, 16], "dropout": 0.0, "batchnorm": False, "weight_decay": 0.0},
    {"name": "+ dropout 0.2", "hidden_sizes": [32, 16], "dropout": 0.2, "batchnorm": False, "weight_decay": 0.0},
    {"name": "+ weight decay 1e-4", "hidden_sizes": [32, 16], "dropout": 0.0, "batchnorm": False, "weight_decay": 1e-4},
    {"name": "+ dropout 0.2 + weight decay 1e-4", "hidden_sizes": [32, 16], "dropout": 0.2, "batchnorm": False, "weight_decay": 1e-4},
    {"name": "+ batchnorm", "hidden_sizes": [32, 16], "dropout": 0.0, "batchnorm": True, "weight_decay": 0.0},
    {"name": "+ batchnorm + dropout 0.2", "hidden_sizes": [32, 16], "dropout": 0.2, "batchnorm": True, "weight_decay": 0.0},
    {"name": "wider [64, 32] (no reg.)", "hidden_sizes": [64, 32], "dropout": 0.0, "batchnorm": False, "weight_decay": 0.0},
    {"name": "wider [64, 32] + dropout 0.2 + weight decay 1e-4", "hidden_sizes": [64, 32], "dropout": 0.2, "batchnorm": False, "weight_decay": 1e-4},
]

LR = 1e-3
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 15


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    n_features = Z_train2.shape[1]

    results = []
    for cfg in CONFIGS:
        model, history = train_model(
            n_features, cfg["hidden_sizes"], Z_train2, y_train2.values, Z_val, y_val.values,
            dropout=cfg["dropout"], batchnorm=cfg["batchnorm"], lr=LR, weight_decay=cfg["weight_decay"],
            batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42,
        )
        val_proba = predict_proba(model, Z_val)
        metrics = classification_metrics(y_val, val_proba, threshold=0.5)
        row = {**cfg, "best_epoch": history["best_epoch"], "stopped_epoch": history["stopped_epoch"], **metrics}
        results.append(row)
        print(f"[{cfg['name']}] val_pr_auc={metrics['pr_auc']:.4f} val_roc_auc={metrics['roc_auc']:.4f} "
              f"best_epoch={history['best_epoch']}")

    results.sort(key=lambda r: r["pr_auc"], reverse=True)
    print("\n--- Sorted by validation PR-AUC ---")
    for r in results:
        print(f"{r['name']:<50s} pr_auc={r['pr_auc']:.4f} roc_auc={r['roc_auc']:.4f} macro_f1={r['macro_f1']:.4f}")

    with open(OUT_DIR / "step4_search.json", "w") as f:
        json.dump(results, f, indent=2)

    best = results[0]
    print(f"\nBest configuration: {best['name']}")
    return best


if __name__ == "__main__":
    main()
