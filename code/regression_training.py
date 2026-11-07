"""Steps 2-3 - the chosen network architecture, trained once to show
convergence and the overfitting onset (Figure 1: train/val loss curves)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from regression_common import fit_pipeline, load_train_val_test, regression_metrics
from regression_model import MLPRegressor, predict, train_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "regression" / "figures"
OUT_DIR = ROOT / "docs" / "projects" / "regression" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_SIZES = [32, 16]
LR = 1e-3
BATCH_SIZE = 128
MAX_EPOCHS = 300
PATIENCE = 20


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)
    n_features = Z_train2.shape[1]

    n_params = MLPRegressor(n_features, HIDDEN_SIZES).n_params()
    print(f"Architecture: {n_features} -> {HIDDEN_SIZES} -> 1, {n_params} parameters")
    print(f"Training rows: {Z_train2.shape[0]} (ratio {Z_train2.shape[0] / n_params:.2f} rows/parameter)")

    model, history = train_model(
        n_features, HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
        dropout=0.0, batchnorm=False, lr=LR, weight_decay=0.0,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42,
    )

    val_pred = predict(model, Z_val)
    val_metrics = regression_metrics(y_val, val_pred)
    print(f"Stopped at epoch {history['stopped_epoch']}, best epoch {history['best_epoch']}")
    print(f"Validation metrics at best epoch: " + ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))

    # --- Figure 1: training and validation loss (sqrt-scale MSE) vs epoch ---
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="Train loss (sqrt-scale MSE)")
    plt.plot(epochs, history["val_loss"], label="Validation loss (sqrt-scale MSE)")
    plt.axvline(history["best_epoch"], color="green", linestyle="--", alpha=0.6, label=f"Best epoch ({history['best_epoch']})")
    plt.title("Figure 1 - Training and validation loss vs. epoch (baseline network)")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on sqrt(rented_bike_count), log scale")
    plt.yscale("log")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_loss_curves.png", dpi=150)
    plt.close()

    with open(OUT_DIR / "step2_3_baseline_network.json", "w") as f:
        json.dump({
            "architecture": f"{n_features}->{HIDDEN_SIZES}->1",
            "n_params": n_params,
            "n_train_rows": int(Z_train2.shape[0]),
            "rows_per_param": Z_train2.shape[0] / n_params,
            "best_epoch": history["best_epoch"],
            "stopped_epoch": history["stopped_epoch"],
            "val_metrics": val_metrics,
            "train_loss_at_best": history["train_loss"][history["best_epoch"] - 1],
            "val_loss_at_best": history["val_loss"][history["best_epoch"] - 1],
            "train_loss_final": history["train_loss"][-1],
            "val_loss_final": history["val_loss"][-1],
        }, f, indent=2)


if __name__ == "__main__":
    main()
