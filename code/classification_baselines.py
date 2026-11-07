"""Step 1 - trivial and linear baselines, fit once on the inner training fold
and reused for both the validation report (here) and the test report (Step 5).
"""
import json
from pathlib import Path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from classification_common import classification_metrics, fit_pipeline, load_train_val_test

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "projects" / "classification" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_baselines():
    """Dummy (most frequent), linear (balanced LogisticRegression) and an
    optional balanced random forest - all with class_weight="balanced" so the
    11.7% positive rate is handled the same way (loss/split-criterion
    reweighting) the network handles it via pos_weight."""
    return {
        "dummy_most_frequent": DummyClassifier(strategy="most_frequent"),
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "random_forest": RandomForestClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1),
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
        proba = model.predict_proba(Z_val)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(Z_val)
        val_results[name] = classification_metrics(y_val, proba, threshold=0.5)
        print(f"[{name}] validation: pr_auc={val_results[name]['pr_auc']:.4f} "
              f"roc_auc={val_results[name]['roc_auc']:.4f} macro_f1={val_results[name]['macro_f1']:.4f}")

    best_name = max(val_results, key=lambda k: val_results[k]["pr_auc"])
    print(f"\nBest baseline on validation (by PR-AUC): {best_name} -> PR-AUC={val_results[best_name]['pr_auc']:.4f}")

    with open(OUT_DIR / "step1_baselines_val.json", "w") as f:
        json.dump({"val_results": val_results, "best_baseline": best_name}, f, indent=2)


if __name__ == "__main__":
    main()
