"""Regression EDA - preprocessing pipeline for UCI Seoul Bike Sharing Demand.

Importable Pipeline + ColumnTransformer that turns the retained predictors into
a dense, scaled matrix for a neural network. Every statistic (clip bounds,
means, standard deviations, one-hot levels) is learned in fit(), so fit on the
training set only. The target transform (log1p) is NOT part of this pipeline:
apply it to y when training (e.g. TransformedTargetRegressor). No model is trained.

Import (from the repository root):
    import sys; sys.path.insert(0, "code")
    from bike_preprocessing import load_data, split, build_preprocessor
    X, y = load_data()
    X_train, X_test, y_train, y_test = split(X, y)
    pre = build_preprocessor().fit(X_train)
    Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)

Run the checks from the command line:
    python code/bike_preprocessing.py [--out data/processed]
"""

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

RANDOM_STATE = 42
N_FOLDS = 5          # one fold of 5 = 20% test

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "SeoulBikeData.csv"
# SHA-256 of the file after CRLF -> LF (UCI ships CRLF; the download log records the raw bytes).
DATA_SHA256 = "0ceeae88d0f6486da38bf57a9a58f5f78d17e3ce36bc6836c6bf8e9ac13fa8cc"

# Original UCI headers -> snake_case names used everywhere in the report.
RENAME = {
    "Date": "date",
    "Rented Bike Count": "rented_bike_count",
    "Hour": "hour",
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Visibility (10m)": "visibility",
    "Dew point temperature(°C)": "dew_point",
    "Solar Radiation (MJ/m2)": "solar_radiation",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "seasons",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
}
TARGET = "rented_bike_count"
WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

# Numerical groups, by the treatment that the EDA (2A, 3A) justifies.
CYCLICAL = ["hour"]                                  # 23 h is next to 0 h
STANDARD = ["temperature", "visibility"]              # bounded, no extreme tails
SENSOR_ZERO = ["humidity"]                            # 0% is a sensor fault (1B): treated as missing
SKEWED = ["wind_speed", "solar_radiation"]            # right-skewed, no extreme outliers
HEAVY_TAIL = ["rainfall", "snowfall"]                 # zero-inflated, extreme tails
CATEGORICAL = ["seasons", "holiday", "day_of_week"]
REDUNDANT = ["dew_point"]                             # ~ f(temperature, humidity), 3A
FEATURES = CYCLICAL + STANDARD + SENSOR_ZERO + SKEWED + HEAVY_TAIL + REDUNDANT + CATEGORICAL
CLIP_UPPER = 0.99


# ----------------------------------------------------------------- data
def data_sha256(path=DATA):
    """SHA-256 of the file with LF line endings (git may check it out as CRLF on Windows)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def read_raw(path=DATA, verify=True):
    """All 8,760 rows with snake_case columns, parsed date and the derived day_of_week."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/regression_download_data.py")
    if verify and data_sha256(path) != DATA_SHA256:
        raise ValueError(f"{path} does not match the recorded SHA-256.")
    df = pd.read_csv(path, encoding="latin-1").rename(columns=RENAME)
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
    df["day_of_week"] = pd.Categorical(df["date"].dt.dayofweek.map(dict(enumerate(WEEKDAYS))),
                                       categories=WEEKDAYS).astype(str)
    return df


def load_data(path=DATA, verify=True):
    """Return (X, y) for the functioning hours only.

    Rows with functioning_day == "No" always have 0 rentals (the system was
    closed): they are a deterministic rule, not demand, so they are removed and
    the then-constant column is dropped (EDA 1B). X keeps `date` for the
    grouped split; the preprocessor ignores it.
    """
    df = read_raw(path, verify)
    df = df[df["functioning_day"] == "Yes"].reset_index(drop=True)
    X = df[["date"] + FEATURES]
    y = df[TARGET].astype(float)
    return X, y


def split(X, y):
    """80/20 split grouped by date (all 24 hours of a day on the same side)
    and stratified by season, random_state=42."""
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, X["seasons"], groups=X["date"]))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


# ----------------------------------------------------------------- transformers
class QuantileClipper(BaseEstimator, TransformerMixin):
    """Cap each column at an upper quantile learned in fit()."""

    def __init__(self, upper=0.99):
        self.upper = upper

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(getattr(X, "columns", range(np.shape(X)[1])), dtype=object)
        self.upper_ = np.nanquantile(np.asarray(X, dtype=float), self.upper, axis=0)
        return self

    def transform(self, X):
        check_is_fitted(self, "upper_")
        return np.minimum(np.asarray(X, dtype=float), self.upper_)

    def affected(self, X):
        """Boolean mask (rows x columns) of the values that transform() changes."""
        check_is_fitted(self, "upper_")
        return np.asarray(X, dtype=float) > self.upper_

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_in_ if input_features is None else input_features, dtype=object)


def hour_to_cycle(X):
    """Map hour (0-23) to (sin, cos) on the unit circle."""
    angle = 2 * np.pi * np.asarray(X, dtype=float) / 24.0
    return np.hstack([np.sin(angle), np.cos(angle)])


def _cycle_names(transformer, input_features):
    return np.asarray([f"{f}_{s}" for f in input_features for s in ("sin", "cos")], dtype=object)


def mask_zero(X):
    """Replace physically impossible 0 readings by NaN (imputed next)."""
    X = np.asarray(X, dtype=float)
    return np.where(X == 0, np.nan, X)


def _median_imputer():
    # The EDA found 0 NaN, so outside humidity this changes nothing; it guards future data.
    return SimpleImputer(strategy="median")


def build_preprocessor():
    """Unfitted preprocessing Pipeline. Fit on X_train only."""
    cyclical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("sin_cos", FunctionTransformer(hour_to_cycle, feature_names_out=_cycle_names)),
    ])
    standard = Pipeline([
        ("impute", _median_imputer()),
        ("scale", StandardScaler()),
    ])
    sensor_zero = Pipeline([
        ("mask", FunctionTransformer(mask_zero, feature_names_out="one-to-one")),
        ("impute", _median_imputer()),
        ("scale", StandardScaler()),
    ])
    skewed = Pipeline([
        ("impute", _median_imputer()),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    heavy = Pipeline([
        ("impute", _median_imputer()),
        ("clip", QuantileClipper(upper=CLIP_UPPER)),
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    columns = ColumnTransformer([
        ("cyclical", cyclical, CYCLICAL),
        ("standard", standard, STANDARD),
        ("sensor_zero", sensor_zero, SENSOR_ZERO),
        ("skewed", skewed, SKEWED),
        ("heavy", heavy, HEAVY_TAIL),
        ("categorical", categorical, CATEGORICAL),
    ], remainder="drop", verbose_feature_names_out=False)   # drops date and dew_point
    return Pipeline([("columns", columns)])


def clip_report(pre, X):
    """Rows of X changed by the fitted clipping step: per feature, and in total."""
    pipe = pre.named_steps["columns"].named_transformers_["heavy"]
    hit = pipe.named_steps["clip"].affected(pipe.named_steps["impute"].transform(X[HEAVY_TAIL]))
    report = {c: int(hit[:, i].sum()) for i, c in enumerate(HEAVY_TAIL)}
    report["any"] = int(hit.any(axis=1).sum())
    return report


def clip_bounds(pre):
    """Fitted upper clip bound per clipped feature."""
    clip = pre.named_steps["columns"].named_transformers_["heavy"].named_steps["clip"]
    return {c: float(clip.upper_[i]) for i, c in enumerate(HEAVY_TAIL)}


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
    print(f"humidity == 0 imputed  train {int((X_train['humidity'] == 0).sum())}, "
          f"test {int((X_test['humidity'] == 0).sum())}")
    print(f"clip bounds (train quantile {CLIP_UPPER}): {clip_bounds(pre)}")
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
