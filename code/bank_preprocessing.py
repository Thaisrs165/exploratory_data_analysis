"""Part 2 - preprocessing pipeline for UCI Bank Marketing (bank-full.csv).

Importable Pipeline + ColumnTransformer that turns the 15 predictors retained
in Part 1 into a dense, scaled matrix for a neural network. Every statistic
(clip bounds, power-transform lambda, means, standard deviations, one-hot
levels) is learned in fit(), so fit on the training set only. No model is trained.

Import (from the repository root):
    import sys; sys.path.insert(0, "code")
    from bank_preprocessing import load_data, split, build_preprocessor
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    pre = build_preprocessor().fit(X_train)
    Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)

Run the checks from the command line:
    python code/bank_preprocessing.py [--out data/processed]
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
TEST_SIZE = 0.20

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "bank-full.csv"
# SHA-256 recorded by Part 1 (data/raw/part1_download_log.json), LF line endings.
DATA_SHA256 = "d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d"

TARGET = "y"
LEAKAGE = ["duration"]
NUMERICAL = ["age", "balance", "day", "campaign", "pdays", "previous"]
CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan",
               "contact", "month", "poutcome"]

# Numerical groups, by the treatment that Part 1 (2A, 3A, 3C) justifies.
STANDARD = ["age", "day"]              # mild skew: standardise only
SIGNED_SKEWED = ["balance"]            # heavy tails on both sides, negative values
COUNT_SKEWED = ["campaign", "previous"]  # heavy-tailed counts
SENTINEL = ["pdays"]                   # -1 = not previously contacted
PDAYS_SENTINEL = -1
CLIP_QUANTILES = (0.01, 0.99)
MISSING_CATEGORY = "unknown"


# ----------------------------------------------------------------- data
def data_sha256(path=DATA):
    """SHA-256 of the file with LF line endings (git may check it out as CRLF on Windows)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_data(path=DATA, verify=True):
    """Return (X, y): the 15 retained predictors and the target encoded yes = 1, no = 0."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/part1_download_data.py")
    if verify and data_sha256(path) != DATA_SHA256:
        raise ValueError(f"{path} does not match the SHA-256 recorded in Part 1.")
    df = pd.read_csv(path, sep=";")
    X = df[NUMERICAL + CATEGORICAL]
    y = (df[TARGET] == "yes").astype(int)
    return X, y


def split(X, y):
    """The Part 1 split (1D): 80/20, stratified on y, random_state=42."""
    return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)


# ----------------------------------------------------------------- transformers
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorise each column to quantile bounds learned in fit()."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(getattr(X, "columns", range(np.shape(X)[1])), dtype=object)
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanquantile(X, self.lower, axis=0)
        self.upper_ = np.nanquantile(X, self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "lower_")
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def affected(self, X):
        """Boolean mask (rows x columns) of the values that transform() changes."""
        check_is_fitted(self, "lower_")
        X = np.asarray(X, dtype=float)
        return (X < self.lower_) | (X > self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


def mask_pdays_sentinel(X):
    """Replace the pdays = -1 code ("not previously contacted") by NaN."""
    X = np.asarray(X, dtype=float)
    return np.where(X == PDAYS_SENTINEL, np.nan, X)


def _median_imputer():
    # Part 1 found 0 NaN, so this changes nothing here; it guards future data.
    return SimpleImputer(strategy="median")


def build_preprocessor():
    """Unfitted preprocessing Pipeline. Fit on X_train only."""
    clip = dict(lower=CLIP_QUANTILES[0], upper=CLIP_QUANTILES[1])
    standard = Pipeline([
        ("impute", _median_imputer()),
        ("scale", StandardScaler()),
    ])
    signed = Pipeline([
        ("impute", _median_imputer()),
        ("clip", QuantileClipper(**clip)),
        ("yeo_johnson", PowerTransformer(method="yeo-johnson", standardize=True)),
    ])
    count = Pipeline([
        ("impute", _median_imputer()),
        ("clip", QuantileClipper(**clip)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    # StandardScaler ignores NaN in fit and keeps it in transform, so pdays is
    # standardised on previously contacted clients only; the sentinel rows then
    # get 0 (the contacted mean) and a missing-indicator column carries the flag.
    sentinel = Pipeline([
        ("mask", FunctionTransformer(mask_pdays_sentinel, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
        ("impute", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True)),
    ])
    categorical = Pipeline([
        # "unknown" is already a level (Part 1, 1B); a NaN would join it.
        ("impute", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("standard", standard, STANDARD),
        ("signed", signed, SIGNED_SKEWED),
        ("count", count, COUNT_SKEWED),
        ("sentinel", sentinel, SENTINEL),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)
    return Pipeline([("columns", columns)])


def clip_report(pre, X):
    """Rows of X changed by the fitted clipping step: per feature, and in total."""
    columns = pre.named_steps["columns"]
    masks = {}
    for group, cols in [("signed", SIGNED_SKEWED), ("count", COUNT_SKEWED)]:
        pipe = columns.named_transformers_[group]
        imputed = pipe.named_steps["impute"].transform(X[cols])
        hit = pipe.named_steps["clip"].affected(imputed)
        masks.update({c: hit[:, i] for i, c in enumerate(cols)})
    report = {c: int(m.sum()) for c, m in masks.items()}
    report["any"] = int(np.logical_or.reduce(list(masks.values())).sum())
    return report


def clip_bounds(pre):
    """Fitted (lower, upper) clip bounds per clipped feature."""
    columns = pre.named_steps["columns"]
    out = {}
    for group, cols in [("signed", SIGNED_SKEWED), ("count", COUNT_SKEWED)]:
        clip = columns.named_transformers_[group].named_steps["clip"]
        out.update({c: (float(clip.lower_[i]), float(clip.upper_[i])) for i, c in enumerate(cols)})
    return out


# ----------------------------------------------------------------- CLI
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for X_train/X_test/y_train/y_test CSVs (not written by default)")
    args = parser.parse_args()

    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    pre = build_preprocessor().fit(X_train)       # training set only
    Z_train = pre.transform(X_train)
    Z_test = pre.transform(X_test)                # transform only, never fitted
    names = pre.get_feature_names_out()

    print(f"raw         train {X_train.shape}  test {X_test.shape}")
    print(f"transformed train {Z_train.shape}  test {Z_test.shape}")
    print(f"NaN         raw train {int(X_train.isna().sum().sum())}, raw test {int(X_test.isna().sum().sum())}, "
          f"transformed train {int(np.isnan(Z_train).sum())}, transformed test {int(np.isnan(Z_test).sum())}")
    print(f"clip bounds (train quantiles {CLIP_QUANTILES}): {clip_bounds(pre)}")
    print(f"rows clipped  train {clip_report(pre, X_train)}")
    print(f"rows clipped  test  {clip_report(pre, X_test)}")
    print(f"feature names ({len(names)}):")
    for i, n in enumerate(names):
        print(f"  {i:2d} {n}")

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(Z_train, columns=names, index=X_train.index).to_csv(args.out / "X_train.csv", index_label="row_id")
        pd.DataFrame(Z_test, columns=names, index=X_test.index).to_csv(args.out / "X_test.csv", index_label="row_id")
        y_train.to_csv(args.out / "y_train.csv", index_label="row_id")
        y_test.to_csv(args.out / "y_test.csv", index_label="row_id")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
