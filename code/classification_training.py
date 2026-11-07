"""Steps 2-3 - the chosen network architecture, trained once to show
convergence and the overfitting onset (Figure 1: train/val loss curves)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from classification_common import classification_metrics, fit_pipeline, load_train_val_test
from classification_model import MLPClassifier, predict_proba, train_model

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "projects" / "classification" / "figures"
OUT_DIR = ROOT / "docs" / "projects" / "classification" / "outputs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

HIDDEN_SIZES = [32, 16]
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

    n_params = MLPClassifier(n_features, HIDDEN_SIZES).n_params()
    print(f"Architecture: {n_features} -> {HIDDEN_SIZES} -> 1, {n_params} parameters")
    print(f"Training rows: {Z_train2.shape[0]} (ratio {Z_train2.shape[0] / n_params:.2f} rows/parameter)")

    model, history = train_model(
        n_features, HIDDEN_SIZES, Z_train2, y_train2.values, Z_val, y_val.values,
        dropout=0.0, batchnorm=False, lr=LR, weight_decay=0.0,
        batch_size=BATCH_SIZE, max_epochs=MAX_EPOCHS, patience=PATIENCE, seed=42,
    )
    print(f"pos_weight (n_negative/n_positive in training fold): {history['pos_weight']:.4f}")

    val_proba = predict_proba(model, Z_val)
    val_metrics = classification_metrics(y_val, val_proba, threshold=0.5)
    print(f"Stopped at epoch {history['stopped_epoch']}, best epoch {history['best_epoch']}")
    print("Validation metrics (threshold=0.5): " +
          ", ".join(f"{k}={v:.4f}" for k, v in val_metrics.items() if isinstance(v, float)))

    # --- Figure 1: training and validation loss vs epoch ---
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, ax1 = plt.subplots(figsize=(7, 5))
    ax1.plot(epochs, history["train_loss"], label="Train loss (BCE, pos-weighted)", color="#4C72B0")
    ax1.plot(epochs, history["val_loss"], label="Validation loss (BCE, pos-weighted)", color="#DD8452")
    ax1.axvline(history["best_epoch"], color="green", linestyle="--", alpha=0.6,
                label=f"Best epoch ({history['best_epoch']})")
    ax1.set_title("Figure 1 - Training and validation loss vs. epoch (baseline network)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Pos-weighted BCE loss")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_loss_curves.png", dpi=150)
    plt.close(fig)

    with open(OUT_DIR / "step2_3_baseline_network.json", "w") as f:
        json.dump({
            "architecture": f"{n_features}->{HIDDEN_SIZES}->1",
            "n_params": n_params,
            "n_train_rows": int(Z_train2.shape[0]),
            "rows_per_param": Z_train2.shape[0] / n_params,
            "pos_weight": history["pos_weight"],
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
