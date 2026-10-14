# exploratory_data_analysis

Two EDAs, one per task. Both datasets are approved by the instructor.

| Task | Dataset | Report |
|---|---|---|
| Classification | UCI Bank Marketing (`bank-full.csv`, target `y`) | `docs/projects/eda/index.md` |
| Regression | UCI Seoul Bike Sharing Demand (`SeoulBikeData.csv`, target `rented_bike_count`) | `docs/projects/eda-regression/index.md` |

## Site

The reports are published with MkDocs Material at <https://thaisrs165.github.io/exploratory_data_analysis/>
(GitHub Actions builds it on every push to `main`; see `.github/workflows/main.yaml`). To preview locally:

```
pip install -r requirements-docs.txt
mkdocs serve -o
```

## Run

From the repository root (Python 3.9+):

```
python -m venv .venv
.venv\Scripts\activate            (Windows; on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python code/bank_preprocessing.py      # fit the preprocessing pipeline on the training set and print the checks
python scripts/part1_eda.py            # stages 0-3: part1_figures/, part1_outputs/, part1_draft.md
python scripts/part2_eda.py            # stages 4-5: part2_figures/, part2_outputs/, index.md

# regression EDA
python code/bike_preprocessing.py      # fit the regression pipeline on the training set and print the checks
python scripts/regression_eda.py       # docs/projects/eda-regression/figures/ and outputs/
```

`data/raw/bank-full.csv` and `data/raw/SeoulBikeData.csv` are committed; `python scripts/part1_download_data.py` and
`python scripts/regression_download_data.py` download them again from UCI if needed.

## Use the preprocessing pipeline

```python
import sys; sys.path.insert(0, "code")
from bank_preprocessing import load_data, split, build_preprocessor

X, y = load_data()
X_train, X_test, y_train, y_test = split(X, y)
pre = build_preprocessor().fit(X_train)     # training set only
Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)
```

Regression pipeline (same pattern; the target transform `sqrt(y)` is applied separately when training):

```python
from bike_preprocessing import load_data, split, build_preprocessor
```
