"""Step 1 - trivial and linear baselines, fit once on the inner training fold
and reused for both the validation report (here) and the test report (Step 5).
"""
import json
from pathlib import Path

import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from regression_common import fit_pipeline, load_train_val_test, regression_metrics, sqrt_inverse, sqrt_transform

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "projects" / "regression" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_baselines():
    """Dummy (mean), linear (Ridge) and an optional random forest, all trained
    on sqrt(y) via TransformedTargetRegressor - the same target transform the
    network uses, so the comparison in Step 5 is apples-to-apples."""
    return {
        "dummy_mean": TransformedTargetRegressor(
            regressor=DummyRegressor(strategy="mean"), func=sqrt_transform, inverse_func=sqrt_inverse),
        "ridge": TransformedTargetRegressor(
            regressor=Ridge(alpha=1.0, random_state=42), func=sqrt_transform, inverse_func=sqrt_inverse),
        "random_forest": TransformedTargetRegressor(
            regressor=RandomForestRegressor(n_estimators=300, max_depth=12, random_state=42, n_jobs=-1),
            func=sqrt_transform, inverse_func=sqrt_inverse),
    }


def main():
    X_train2, X_val, X_test, y_train2, y_val, y_test, X_train, y_train = load_train_val_test()
    pre = fit_pipeline(X_train2)
    Z_train2 = pre.transform(X_train2)
    Z_val = pre.transform(X_val)

    baselines = build_baselines()
    val_results = {}
    for name, model in baselines.items():
        model.fit(Z_train2, y_train2)
        pred = model.predict(Z_val)
        val_results[name] = regression_metrics(y_val, pred)
        print(f"[{name}] validation: " + ", ".join(f"{k}={v:.4f}" for k, v in val_results[name].items()))

    best_name = min(val_results, key=lambda k: val_results[k]["mae"])
    print(f"\nBest baseline on validation (by MAE): {best_name} -> MAE={val_results[best_name]['mae']:.4f}")

    with open(OUT_DIR / "step1_baselines_val.json", "w") as f:
        json.dump({"val_results": val_results, "best_baseline": best_name}, f, indent=2)


if __name__ == "__main__":
    main()
