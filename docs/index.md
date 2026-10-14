# Exploratory Data Analysis — 2026.2

Project for Insper's **Artificial Neural Networks and Deep Learning**, semester 2026.2: an exploratory data analysis (EDA) and preprocessing pipeline that prepares tabular data for a neural network. We did it for both tasks: a **classification** dataset and a **regression** dataset. No model is trained here; each report ends with a synthesis for whoever builds the model.

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

-   **[Classification — UCI Bank Marketing](projects/eda/index.md)**

    ---

    Will a client subscribe to a term deposit after a marketing call?

    - 45,211 calls, 16 input features, binary target `y`
    - Imbalanced: 11.70% `yes`
    - `duration` dropped as target leakage
    - Stratified 80/20 split; 51 features after the pipeline

-   **[Regression — UCI Seoul Bike Sharing Demand](projects/eda-regression/index.md)**

    ---

    How many public bikes are rented in Seoul in a given hour?

    - 8,760 hours, 13 input columns, target `rented_bike_count`
    - Right-skewed target: train on `sqrt(y)`
    - Closed hours removed as target leakage
    - Split by whole days, stratified by season; 22 features after the pipeline

</div>

Both reports follow the same stages from the assignment:

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

| Part | Pipeline (importable) | Analysis scripts |
|---|---|---|
| Classification | `code/bank_preprocessing.py` | `scripts/part1_eda.py`, `scripts/part2_eda.py` |
| Regression | `code/bike_preprocessing.py` | `scripts/regression_eda.py` |

## Running locally

``` shell
python3 -m venv env
source ./env/bin/activate
python3 -m pip install -r requirements.txt -r requirements-docs.txt
mkdocs serve -o
```
