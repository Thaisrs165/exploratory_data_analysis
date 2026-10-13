# exploratory_data_analysis

Classification EDA of the UCI Bank Marketing dataset (`bank-full.csv`, target `y`).
The report is `docs/projects/eda/index.md`. **Instructor approval of the dataset is still pending.**

## Run

From the repository root (Python 3.9+):

```
python -m venv .venv
.venv\Scripts\activate            (Windows; on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt

python code/bank_preprocessing.py      # fit the preprocessing pipeline on the training set and print the checks
python scripts/part1_eda.py            # stages 0-3: part1_figures/, part1_outputs/, part1_draft.md
python scripts/part2_eda.py            # stages 4-5: part2_figures/, part2_outputs/, index.md
```

`data/raw/bank-full.csv` is committed; `python scripts/part1_download_data.py` downloads it again from UCI if needed.

## Use the preprocessing pipeline

```python
import sys; sys.path.insert(0, "code")
from bank_preprocessing import load_data, split, build_preprocessor

X, y = load_data()
X_train, X_test, y_train, y_test = split(X, y)
pre = build_preprocessor().fit(X_train)     # training set only
Z_train, Z_test = pre.transform(X_train), pre.transform(X_test)
```
