# Exploratory Data Analysis — 2026.2

Project for Insper's **Artificial Neural Networks and Deep Learning**, semester 2026.2, in three deliverables on the same team and datasets: an exploratory data analysis (EDA) and preprocessing pipeline for both a **classification** and a **regression** dataset, followed by a neural network built on top of each pipeline.

## Team

<div class="team-grid" markdown>

<figure markdown>
![Ana Helena Caiafa](assets/team/ana-helena-caiafa.jpg){ .team-photo }
<figcaption>Ana Helena Caiafa</figcaption>
</figure>

<figure markdown>
![Thais Rocha](assets/team/thais-rocha.jpg){ .team-photo }
<figcaption>Thais Rocha</figcaption>
</figure>

<figure markdown>
![Esther Caroline Rodrigues](assets/team/esther-caroline-rodrigues.jpg){ .team-photo }
<figcaption>Esther Caroline Rodrigues</figcaption>
</figure>

<figure markdown>
![Carolina Eskenazi](assets/team/carolina-eskenazi.jpg){ .team-photo }
<figcaption>Carolina Eskenazi</figcaption>
</figure>

</div>

## The two analyses

<div class="grid cards" markdown>

-   **[Classification — UCI Bank Marketing](projects/eda/stage-0.md)**

    ---

    Will a client subscribe to a term deposit after a marketing call?

    - 45,211 calls, 16 input features, binary target `y`
    - Imbalanced: 11.70% `yes`
    - `duration` dropped as target leakage
    - Stratified 80/20 split; 51 features after the pipeline

    [:octicons-arrow-right-24: Start reading](projects/eda/stage-0.md) · [Full report on one page](projects/eda/index.md)

-   **[Regression — UCI Seoul Bike Sharing Demand](projects/eda-regression/stage-0.md)**

    ---

    How many public bikes are rented in Seoul in a given hour?

    - 8,760 hours, 13 input columns, target `rented_bike_count`
    - Right-skewed target: train on `sqrt(y)`
    - Closed hours removed as target leakage
    - Split by whole days, stratified by season; 22 features after the pipeline

    [:octicons-arrow-right-24: Start reading](projects/eda-regression/stage-0.md) · [Full report on one page](projects/eda-regression/index.md)

</div>

## The models

Both EDA pipelines, built on in the same way: a from-scratch-configured PyTorch MLP, trivial/linear/random-forest baselines, an 8-configuration regularization search, a test set touched exactly once (3 seeds), and error analysis cross-checked against the EDA's own findings.

<div class="grid cards" markdown>

-   **[Regression model — bike demand](projects/regression/index.md)**

    ---

    A 2-hidden-layer MLP (22 → 64 → 32 → 1) predicting `sqrt(rented_bike_count)` beats the mandatory linear baseline's test MAE by 53%, and matches an optional random-forest baseline at a fraction of the parameters.

    - Honest negative result: dropout, weight decay and batch norm all **hurt** validation performance here
    - Error analysis matches the EDA's own top correlates (temperature, hour)

    [:octicons-arrow-right-24: Read the report](projects/regression/index.md)

-   **[Classification model — term deposit subscription](projects/classification/index.md)**

    ---

    A 2-hidden-layer MLP (51 → 32 → 16 → 1) beats the mandatory Logistic Regression baseline on every metric, but falls just short of its 15% PR-AUC success criterion and trails an optional random-forest baseline.

    - Honest negative result: only a 6.4% PR-AUC gain over Logistic Regression (criterion: ≥ 15%) — here dropout **helps**, the opposite of the regression track
    - Error analysis flags a possible data-collection confound (`day`) the EDA had already warned about

    [:octicons-arrow-right-24: Read the report](projects/classification/index.md)

</div>

Both EDA reports follow the same stages from the assignment. Each stage is its own page (use the tabs at the top and **Next** at the bottom of each page), and each report is also available on a single page:

| Stage | Content |
|---|---|
| 0. Proposal | dataset, target, motivation, first risk |
| 1. Initial inspection | data dictionary, quality and leakage, target, train/test split |
| 2. Univariate analysis | numerical and categorical distributions |
| 3. Bivariate and multivariate analysis | correlations, feature vs target, grouped boxplots |
| 4. Preprocessing | strategies, PCA / t-SNE / UMAP, `Pipeline` + `ColumnTransformer` |
| 5. Synthesis | main findings, risks and plan, 10-item summary table |

## Code

Everything on this site is generated from the code in the [repository](https://github.com/Thaisrs165/exploratory_data_analysis):

| Part | Pipeline (importable) | Analysis / model scripts |
|---|---|---|
| Classification EDA | `code/bank_preprocessing.py` | `scripts/part1_eda.py`, `scripts/part2_eda.py` |
| Regression EDA | `code/bike_preprocessing.py` | `scripts/regression_eda.py` |
| Regression model | `code/regression_common.py`, `code/regression_model.py` | `code/regression_baselines.py`, `code/regression_training.py`, `code/regression_search.py`, `code/regression_test_eval.py`, `code/regression_error_analysis.py` |
| Classification model | `code/classification_common.py`, `code/classification_model.py` | `code/classification_baselines.py`, `code/classification_training.py`, `code/classification_search.py`, `code/classification_test_eval.py`, `code/classification_error_analysis.py` |

## Running locally

``` shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt -r requirements-docs.txt
mkdocs serve -o
```
