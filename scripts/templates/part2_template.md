<!--
  GENERATED FILE - do not edit docs/projects/eda/index.md by hand.
  Stages 0-3 are copied unchanged from docs/projects/eda/part1_draft.md (python scripts/part1_eda.py).
  Stages 4-5 come from scripts/templates/part2_template.md.
  Regenerate with: python scripts/part2_eda.py
  Every number below is substituted from values computed on data/raw/bank-full.csv.
-->

# Classification EDA - UCI Bank Marketing

> **Status: PENDING INSTRUCTOR APPROVAL.** The repository contains no record of instructor approval for this dataset. Approval is a prerequisite, so the proposal and the whole EDA below are provisional and must not be read as approved or final until approval is documented.
>
> Structure: stages 0-3 (Part 1) and stages 4-5 (Part 2). No model is trained anywhere in this report.

---

{{part1_body}}

---

## 4. Preprocessing

Everything in this stage is implemented in `code/bank_preprocessing.py` (a scikit-learn `Pipeline` + `ColumnTransformer`) and run by `scripts/part2_eda.py`. The intended model is a neural network, which needs a dense, fully numerical input on comparable scales.

**Consistency with Part 1.** Before anything is fitted, the script recomputes {{n_checks}} quantities from the raw file and stops if any differs from the values saved by Part 1 (`part1_outputs/`). All agree; the full list is in the Appendix. In particular, the split is the one from 1D, row for row: train {{n_train}} rows, test {{n_test}} rows.

**No fitting on test data.** Clip bounds, the Yeo-Johnson parameter, means, standard deviations and one-hot levels are learned by `fit(X_train)`. The test set only goes through `transform`. PCA, t-SNE and UMAP (4B) use training rows only.

### 4A. Strategies

Each choice below starts from a check on the data and from a finding of Part 1.

#### Missing values

**Check.** There are {{nan_train}} `NaN` values in the training predictors and {{nan_test}} in the test predictors (Part 1, 1B: {{total_nan}} in the whole file). So there is **nothing to impute** in this dataset.

**Table 4.1 - Categorical checks before encoding (training set; last column compares with the test set).**

{{table_cat_checks}}

**Strategy.**

- **`"unknown"` stays a category.** Part 1 (1B) showed that it is not missing at random: for `poutcome` it marks clients who were never contacted, for `contact` it marks the early period of the file, and its subscription rate differs ({{r_contact_unknown}} for `contact = unknown` against {{r_contact_cellular}} for `cellular`, Figure 8). Replacing it with the mode would erase that signal, so it is one-hot encoded like any other level.
- **`pdays = -1` is treated as "not applicable", not as a number.** Part 1 (1B, Figure 7) showed that `-1` is a code for "not previously contacted" and coincides exactly with `previous = 0`. The pipeline replaces `-1` by a missing value, standardises `pdays` with the mean and standard deviation of **previously contacted training clients only** ({{pdays_fit_mean}} and {{pdays_fit_std}} days, n = {{n_contacted_train}}), then fills the masked rows with 0 and adds the indicator column `missingindicator_pdays`. The value 0 is the contacted-client mean, so it is neutral, and the indicator carries the "not previously contacted" information that Part 1 asked to preserve. In the training set the indicator is 1 for {{ind_train}} rows ({{ind_train_pct}}).
- **Imputers are safeguards.** The pipeline still contains a median imputer for the numerical features and a constant `"unknown"` imputer for the categorical ones, fitted on the training set. They change **0 values** here. They exist so that `transform` does not fail if future data contains a `NaN`. A `NaN` in a categorical feature would join the existing `"unknown"` level, which is consistent with the decision above.

#### Outliers

**Check.** Part 1 (2A, Figure 3) counted 1.5 x IQR outliers in the training set: `balance` {{balance_out}} ({{balance_outp}}%), `campaign` {{campaign_out}}, and for `pdays`/`previous` an artifact of the sentinel. No value was impossible (1B), so Part 1 removed no rows and recommended caps or transforms fitted on train. Expressed as z-scores of the raw training values, the extremes reach {{rawz_balance_max}} (`balance`), {{rawz_campaign_max}} (`campaign`) and {{rawz_previous_max}} (`previous`) standard deviations. Inputs of that size would dominate the gradients of a neural network.

**Strategy: clip (winsorise), then transform. No row is removed.**

- `balance`, `campaign` and `previous` are clipped to the **1st and 99th percentiles of the training set**. For the two counts the 1st percentile equals the minimum, so only the upper tail is clipped.
- `balance` then goes through a Yeo-Johnson power transform (fitted lambda {{bal_yj_lambda}}). It is used instead of a logarithm because {{clip_balance_lo}} is negative: overdrawn accounts are valid (1B).
- `campaign` and `previous` are counts, so they get `log1p`. This matches Part 1's finding that the signal of `campaign` is in the tail and has diminishing returns (Figure 13).
- **Why clipping is needed, not only a transform.** Yeo-Johnson on the unclipped `balance` still leaves standardised values between {{bal_yj_noclip_min}} and {{bal_yj_noclip_max}}. After clipping the range is {{out_balance_min}} to {{out_balance_max}}.
- `age` and `day` are not clipped. `age` is only mildly skewed ({{age_skew}}) and its extremes are informative: clients aged 60+ subscribe at {{age_ge60_rate}} against {{age_30_59_rate}} for ages 30-59 (Figure 11). `day` has no outliers (Figure 3).
- `pdays` is not clipped. Once the sentinel is masked, it has few outliers among contacted clients (2A) and its standardised range is {{out_pdays_min}} to {{out_pdays_max}}.

**Table 4.2 - Rows changed by clipping.** Bounds are learned on the training set and applied unchanged to the test set.

{{table_clip}}

**{{clip_any_train}} training rows ({{clip_any_train_pct}}) have at least one clipped value**, and {{clip_any_test}} test rows ({{clip_any_test_pct}}). Clipped training rows subscribe at {{clip_rate_y}}, against {{noclip_rate_y}} for the others. Clipping keeps their rank at the top of the scale but removes the distinction among them, which is the cost of this strategy.

#### Categorical encoding

- **One-hot encoding** with `OneHotEncoder(handle_unknown="ignore")`, with no reference level dropped. Part 1 (2B, Figure 5) found low cardinality: at most 12 levels per feature and {{total_dummies}} levels in total. The features are nominal (`month` is cyclical and `education` has an `unknown` level), so integer codes would impose an order that the data does not have.
- **Unseen categories.** The test set contains {{n_unseen_test_levels}} levels that are absent from the training set (Table 4.1). `handle_unknown="ignore"` is still set: a level that appears later is encoded as all zeros for that feature instead of raising an error. This matters most for the rare levels found in Part 1 ({{rare_levels}}).
- No level is dropped because a neural network does not need a reference category, and dropping one would make the all-zeros row of an unseen level indistinguishable from the reference level.
- Rare levels are not grouped. Part 1 suggested grouping only if a model overfits, which cannot be judged before modelling.

#### Scaling

**Choice: standardisation (zero mean, unit variance), fitted on the training set, applied after clipping and transforming.**

- A neural network is trained by gradient descent on weighted sums of its inputs. Features on very different scales (`balance` in thousands of euros, `campaign` in single units) give badly conditioned gradients and make one learning rate unsuitable for all weights. Standard weight initialisation also assumes inputs centred near zero with variance near one.
- **Min-max scaling was not chosen** because the maximum would be set by the tail: the bulk of `balance` would be squeezed into a small part of [0, 1].
- **Robust scaling alone was not chosen** because it centres and scales by median and IQR but leaves the extremes large. Clipping bounds the tails, and after that the mean and standard deviation are no longer dominated by a few points (the concern raised in 2A).
- One-hot columns and the indicator stay as 0/1. They are already bounded and on a scale comparable to standardised features.

**Table 4.3 - Numerical features before and after the pipeline (training set, except the last two columns).** "Raw min z / max z" are the z-scores the raw values would have with plain standardisation.

{{table_before_after}}

{{fig_raw_vs_preprocessed_md}}

**Figure {{fig_raw_vs_preprocessed}} - conclusion.**
- The raw tails (top row) are shortened to a bounded range (bottom row). The largest absolute value in the whole transformed matrix is {{max_abs_out_train}} in train and {{max_abs_out_test}} in test, down from {{rawz_previous_max}} standard deviations for raw `previous`.
- Skew drops from {{balance_skew}} to {{out_balance_skew}} for `balance` and from {{campaign_skew}} to {{out_campaign_skew}} for `campaign`. The bars at both ends of `balance` and at the upper end of the counts are the clipped rows.
- `previous` remains skewed ({{out_previous_skew}}, from {{previous_skew}}). This cannot be removed: {{ind_train_pct}} of the rows are exactly 0, and no monotone transform spreads a point mass.
- `pdays` has standard deviation {{out_pdays_std}} over all rows, not 1, because it is standardised among contacted clients and is 0 for everyone else. The spike at 0 is the never-contacted group.

### 4B. Dimensionality reduction

**Input.** The {{n_features_out}} preprocessed features of the **training set only** (4C). Points are coloured by the target.

**Sample.** PCA is fitted on all {{n_train}} training rows. t-SNE and UMAP are computed on a sample, because t-SNE is slow on the full set and a scatter of 36 thousand points is unreadable. The sample is **{{sample_n}} training rows ({{sample_share}} of train)**, drawn with `train_test_split(train_size={{sample_n_raw}}, stratify=y_train, random_state=42)`. It keeps the class balance: {{sample_yes}} `yes` ({{sample_yes_pct}}). The same sample is used in Figures {{fig_pca_projection}} to {{fig_embeddings_by_contact_history}}, and all methods use `random_state=42`.

#### PCA

**Table 4.4 - Explained variance of the first 10 principal components.**

{{table_pca_var}}

{{fig_pca_explained_variance_md}}

**Figure {{fig_pca_explained_variance}} - conclusion.**
- **PC1 + PC2 explain {{pc12}}** of the variance (PC1 {{pc1}}, PC2 {{pc2}}). A 2D PCA plot therefore hides almost three quarters of the variance.
- Variance is spread over many directions: {{n_pc_80}} components are needed for 80%, {{n_pc_90}} for 90% and {{n_pc_95}} for 95%. There is no low-dimensional linear structure, which agrees with the near-zero correlations among numerical features found in 3A (Figure 6).
- The last {{n_pc_null}} components have zero variance. This is expected: each of the 9 one-hot groups sums to 1, which gives 9 exact linear dependencies.
- The 6 numerical columns carry {{var_share_num}} of the total variance and the 45 binary columns {{var_share_other}}, because a 0/1 column has variance of at most 0.25. PCA therefore weights numerical features more than categorical ones. This is a property of the encoding and not evidence that the numerical features are more important for `y`.

**Table 4.5 - Largest loadings of PC1 and PC2.**

{{table_pca_loadings}}

{{fig_pca_loadings_md}}

**Figure {{fig_pca_loadings}} - conclusion (loadings).**
- **PC1 is a "contact history" axis.** It has `previous` ({{l_pc1_previous}}) at one end and the not-previously-contacted indicator ({{l_pc1_missingindicator_pdays}}) and `poutcome_unknown` ({{l_pc1_poutcome_unknown}}) at the other. These three move together, as Part 1 found (1B; Figure 7). `campaign` ({{l_pc1_campaign}}) and `day` ({{l_pc1_day}}) have the opposite sign to `previous`, consistent with their small negative correlations with `previous` and `pdays` in 3A. The magnitude of `pdays` has almost no weight ({{l_pc1_pdays}}), which matches the finding that `pdays` and `previous` are nearly unrelated once the sentinel is removed (Spearman {{contacted_rho_pdays_prev}}).
- **PC2 is an "age and wealth" axis.** It is led by `age` ({{l_pc2_age}}) and `balance` ({{l_pc2_balance}}), with `housing_no`, `marital_married` on the same side and `housing_yes`, `marital_single` on the other: older clients with higher balances and no housing loan.
- The five largest loadings account for {{pc1_top5_sq}} of PC1 and {{pc2_top5_sq}} of PC2 (sum of squared loadings), so each axis is described by a handful of features. `month` and `job`, which had large subscription-rate spreads in 3B, barely appear, for the variance reason given above.

{{fig_pca_projection_md}}

**Figure {{fig_pca_projection}} - conclusion.**
- The projection shows two clouds along PC1. From the loadings, the right-hand cloud is the previously contacted clients and the large left-hand cloud is the never-contacted majority.
- **The classes overlap in both clouds.** `yes` points are relatively more frequent on the right, but no line separates them. The class means differ by {{pc1_d}} pooled standard deviations on PC1 and by {{pc2_d}} on PC2 (all training rows).
- A linear view of the two highest-variance directions is therefore not enough to separate subscribers. This is expected, since PCA does not use `y` and these two components hold only {{pc12}} of the variance.

#### t-SNE

Perplexities compared: {{perplexities}}. Settings: `init="pca"`, `learning_rate="auto"`, default number of iterations, `random_state=42`.

{{fig_tsne_perplexity_md}}

**Figure {{fig_tsne_perplexity}} - conclusion.**
- **Perplexity 5** breaks the data into many tiny groups of near-identical profiles. Only very local neighbourhoods are kept and no larger structure is visible.
- **Perplexity 30 and 50** show one island that is separated from the main mass and has a visibly higher density of `yes` points. The main mass contains smaller pockets where `yes` is more frequent, but most of it is mixed.
- The island appears at both larger perplexities, so it is not an artifact of a single setting. The layout of the main mass changes between panels and should not be interpreted.

#### UMAP

`n_neighbors` compared: {{n_neighbors}}. Settings: `min_dist={{umap_min_dist}}`, Euclidean metric, `random_state=42`.

{{fig_umap_n_neighbors_md}}

**Figure {{fig_umap_n_neighbors}} - conclusion.**
- All three settings isolate one compact island with a high share of `yes`, the same structure that t-SNE finds.
- With `n_neighbors` = 5 and 15, the remaining points form two further groups. With `n_neighbors` = 50, which gives more weight to global structure, these two groups join into one connected mass while the first island stays apart.
- Inside the two large groups, `yes` and `no` are mixed, with `yes` slightly concentrated along some edges.

{{fig_embeddings_by_contact_history_md}}

**Figure {{fig_embeddings_by_contact_history}} - conclusion.** Colouring the same embeddings by two features identified in Part 1 explains the islands:
- The isolated island is the **previously contacted** clients (n = {{grp_previously_contacted_n}} in the sample, subscription rate {{grp_previously_contacted_rate}}).
- The other two groups are never-contacted clients, split by whether `contact` is `unknown` (n = {{grp_never_contacted_contact_unknown_n}}, rate {{grp_never_contacted_contact_unknown_rate}}) or known (n = {{grp_never_contacted_contact_known_n}}, rate {{grp_never_contacted_contact_known_rate}}).
- The visible structure is therefore **defined by contact history and by the recording period** (1B showed that `contact = unknown` marks the early part of the file), not by the target. The groups differ in subscription rate, but within each group the classes are mixed.

#### What t-SNE and UMAP reveal relative to PCA

- **PCA** gives one interpretable linear axis (contact history) and shows two overlapping clouds. It hides the `contact = unknown` group, because that group differs in binary columns with little variance.
- **t-SNE and UMAP** use local neighbourhoods in all {{n_features_out}} dimensions. They separate the previously contacted clients cleanly, and UMAP also separates the `contact = unknown` group. These groups are defined by categorical features that PCA down-weights.
- **None of the three methods separates `yes` from `no`.** All three agree that subscribers are more frequent among previously contacted clients and are otherwise spread through the majority. The class boundary is not a simple low-dimensional one, so a model will need non-linear combinations of features, and high accuracy on the minority class should not be expected from these inputs alone.

**Interpretation limits.** In t-SNE and UMAP plots, **the size of a cluster, the distance between clusters and the density of points have no direct quantitative interpretation.** The axes have arbitrary units. Both methods preserve neighbourhoods, not distances: a large island is not a large or spread-out group, and two islands that are far apart are not "more different" than two that are close. The layouts also change with perplexity, `n_neighbors` and the random seed. Only structure that persists across settings (here, the previously contacted island) is treated as a finding, and it was confirmed with the original features (Figure {{fig_embeddings_by_contact_history}}), not from the plot alone. These embeddings are diagnostic plots only; they are not used as model inputs.

### 4C. Pipeline

`build_preprocessor()` in `code/bank_preprocessing.py` returns an unfitted `Pipeline` whose single step is this `ColumnTransformer`:

| Branch | Columns | Steps (in order) | Justified by |
|---|---|---|---|
| `standard` | `age`, `day` | median imputer (safeguard) -> `StandardScaler` | mild skew, no clipping needed (2A) |
| `signed` | `balance` | median imputer (safeguard) -> clip to train P1-P99 -> Yeo-Johnson with standardisation | heavy tails on both sides, negative values (2A, Figure 12) |
| `count` | `campaign`, `previous` | median imputer (safeguard) -> clip to train P1-P99 -> `log1p` -> `StandardScaler` | heavy-tailed counts (2A, Figure 13) |
| `sentinel` | `pdays` | mask `-1` -> `StandardScaler` (contacted clients) -> constant imputer 0 with indicator | sentinel analysis (1B, Figure 7) |
| `categorical` | 9 categorical features | constant `"unknown"` imputer (safeguard) -> `OneHotEncoder(handle_unknown="ignore")` | `"unknown"` as category (1B), low cardinality (2B) |

`duration` is not an input: it was excluded as leakage in 1B, and `load_data()` never selects it.

**Usage.** The pipeline is fitted on `X_train` only and the test set is only transformed:

```python
import sys; sys.path.insert(0, "code")
from bank_preprocessing import load_data, split, build_preprocessor

X, y = load_data()                                  # 15 predictors, y: yes = 1, no = 0
X_train, X_test, y_train, y_test = split(X, y)      # the Part 1 split (1D)
pre = build_preprocessor().fit(X_train)             # training set only
Z_train = pre.transform(X_train)
Z_test = pre.transform(X_test)                      # transform only
feature_names = pre.get_feature_names_out()
```

**Verification** (output of `python code/bank_preprocessing.py`, repeated as assertions in `scripts/part2_eda.py`):

| Check | Result |
|---|---|
| `NaN` in raw `X_train` / `X_test` | {{nan_train}} / {{nan_test}} |
| `NaN` after transform, train / test | **{{nan_z_train}} / {{nan_z_test}}** |
| Infinite values after transform | {{inf_z}} |
| Transformed training shape | **{{z_train_shape}}** |
| Transformed test shape | **{{z_test_shape}}** |
| Output columns | {{n_features_out}} = {{n_num_out}} numerical + {{n_flag_out}} indicator + {{n_onehot_out}} one-hot |
| Test levels unseen in train | {{n_unseen_test_levels}} |

**Table 4.6 - Feature names after preprocessing** (in output order; also in `part2_outputs/feature_names.csv`).

{{table_feature_names}}

**Run from a clean clone.** The raw file `data/raw/bank-full.csv` is committed, so no download is needed. If it is missing, `python scripts/part1_download_data.py` fetches it from the UCI archive.

```
python -m venv .venv
.venv\Scripts\activate            (Windows; on Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
python code/bank_preprocessing.py
```

The last command fits the pipeline on the training set and prints the shapes, the `NaN` counts, the clip bounds, the clipped rows and the {{n_features_out}} feature names. Add `--out data/processed` to also write the transformed matrices as CSV.

---

## 5. Synthesis

### 5A. Main findings

| # | Finding | Evidence |
|---|---|---|
| 1 | The data is clean in the usual sense: {{total_nan}} `NaN`, 0 exact duplicate rows, no impossible values. Missing information is coded as `"unknown"`, mostly in `poutcome` (full data: {{unk_poutcome}}) and `contact` ({{unk_contact}}). | 1B missing-value and consistency tables; Figure 5; Table 4.1 |
| 2 | The target is imbalanced: {{pct_yes}} `yes`, ratio {{imbalance_ratio}} : 1. Always predicting `no` gives {{majority_acc}} accuracy. | 1C table; 1D table |
| 3 | `duration` is target leakage and is excluded. It is the only dropped column. | 1B leakage table (duration bins) |
| 4 | The file is chronological and the subscription rate drifts from {{order_rate_first}} to {{order_rate_last}} across row deciles. `contact = unknown` and `month` partly encode the period. | 1B row-decile table; Figures 8 and 9 |
| 5 | `pdays = -1` is a code for "not previously contacted" ({{pdays_neg1_pct}} of rows) and coincides with `previous = 0`. The {{top_pair_rho}} Spearman correlation of `pdays - previous` comes from this shared code; among contacted clients it is {{contacted_rho_pdays_prev}}. | 1B sentinel table; Figures 6 and 7 |
| 6 | Contact history is the strongest signal: `poutcome = success` subscribes at {{r_poutcome_success}} against {{r_poutcome_unknown}} for never-contacted clients. It is also the main axis of variance (PC1) and the group that t-SNE and UMAP isolate. | Figure 8; Figures {{fig_pca_loadings}}, {{fig_tsne_perplexity}}, {{fig_umap_n_neighbors}}, {{fig_embeddings_by_contact_history}} |
| 7 | `month` has a large rate spread, from {{month_rate_min}} to {{month_rate_max}}, with volume and rate moving in opposite directions. | Figure 9; 3B spread table |
| 8 | Numerical effects are non-linear: the age effect is U-shaped ({{age_ge60_rate}} for 60+ against {{age_30_59_rate}} for 30-59) and repeated calls have diminishing returns ({{camp1_rate}} after one contact, {{camp_gt5_rate}} after more than five). | Figures 11 and 13 |
| 9 | `balance`, `campaign` and `previous` are heavy-tailed (skew {{balance_skew}}, {{campaign_skew}}, {{previous_skew}}). Clipping at train P1-P99 changes {{clip_any_train}} training rows ({{clip_any_train_pct}}) and, with a transform and standardisation, bounds every input to within {{max_abs_out_train}} standard deviations. | Figures 1-3; Tables 4.2 and 4.3; Figure {{fig_raw_vs_preprocessed}} |
| 10 | There is no low-dimensional linear structure and no clean class separation: PC1 + PC2 explain {{pc12}}, {{n_pc_90}} components are needed for 90%, and the classes overlap in PCA, t-SNE and UMAP. | Table 4.4; Figures {{fig_pca_explained_variance}}, {{fig_pca_projection}}, {{fig_tsne_perplexity}}, {{fig_umap_n_neighbors}} |
| 11 | The preprocessing pipeline turns 15 predictors into {{n_features_out}} features with no `NaN`: train {{z_train_shape}}, test {{z_test_shape}}. | 4C verification table; Table 4.6 |

### 5B. Modelling risks and handling plan

| # | Risk | Evidence | Handling plan |
|---|---|---|---|
| 1 | **Class imbalance** ({{imbalance_ratio}} : 1). Accuracy is misleading and a network can collapse to predicting `no`. | 1C | Keep stratification in every split and in cross-validation. Report PR-AUC, recall and precision on `yes` and balanced accuracy, with the {{majority_acc}} majority baseline for reference. Use class weights in the loss, computed on the training set. Choose the decision threshold on a validation set, never on the test set. |
| 2 | **Time drift and an optimistic split.** The random stratified split mixes periods, so test performance overstates performance on future campaigns. | 1B row-decile table | State this limitation with every result. As a sensitivity check, also evaluate with a chronological split (train on the first 80% of rows, test on the last 20%) and report the gap. Never use the row index as a feature. |
| 3 | **Period proxies.** `contact = unknown` and `month` may be learned as "which period" instead of a client property. | 1B; Figures 8, 9 and {{fig_embeddings_by_contact_history}} | Train an ablation without `contact` and `month` and compare. If the chronological check (risk 2) shows a large drop, prefer the model without them. |
| 4 | **Leakage.** `duration` would inflate every metric. | 1B | `duration` is never loaded by `load_data()`. Keep all fitted preprocessing inside the pipeline so that cross-validation refits it on each training fold. |
| 5 | **Redundant "never contacted" information.** The indicator, `poutcome_unknown` and `previous = 0` say almost the same thing (they differ in {{ind_vs_pout_mismatch_train}} training rows). | 1B; 3A; Figure {{fig_pca_loadings}} | Harmless for prediction with a neural network, but it makes single weights uninterpretable. Use weight decay, and do not read importance from individual weights. Test dropping the indicator. |
| 6 | **Rare levels and near-constant columns** ({{rare_levels}}; `default = yes` is {{default_yes_pct}}). Their weights are estimated from few rows and can overfit. | 2B; Figures 5 and 9 | Weight decay or dropout plus early stopping on a validation set. If validation shows overfitting, group the rare months and test dropping `default`. `handle_unknown="ignore"` covers levels missing from a fold. |
| 7 | **Clipping removes information** about the {{clip_any_train_pct}} most extreme clients. | Table 4.2 | The clip quantiles are parameters of `QuantileClipper`. Treat them as a hyperparameter (for example P0.5-P99.5, or no clipping) and select on validation data only. |
| 8 | **Weak, non-linear signal.** No projection separates the classes, and the age and campaign effects are non-linear. | Figures 11, 13, {{fig_pca_projection}}, {{fig_tsne_perplexity}}, {{fig_umap_n_neighbors}} | Start from a logistic-regression baseline on the same features, then a small network with hidden layers. Expect modest gains, and compare against the baseline instead of against published results that use `duration`. |
| 9 | **Over-reading the embeddings.** t-SNE and UMAP cluster sizes and distances are not quantitative. | 4B | Use them only as diagnostics. Do not feed them to the model or pick thresholds from them. |
| 10 | **Feature-level duplicates** ({{dup_feat_only}} rows) could sit on both sides of the split. | 1B | Negligible share; kept, as decided in Part 1. If cross-validation is used, note that it does not affect the conclusions. |
| 11 | **Approval is pending.** The dataset has not been approved by the instructor. | Section 0 | Obtain and document the approval before treating this EDA as final. If the dataset is rejected, the scripts are reusable but every number changes. |

### 5C. Summary table

| # | Item | Value |
|---|---|---|
| 1 | Dataset, task and target | UCI Bank Marketing (`bank-full.csv`, older full version). Binary classification. Target `y`: did the client subscribe to a term deposit (`yes` = 1, `no` = 0). **Instructor approval: pending.** |
| 2 | Instances x features (numerical / categorical) | {{rows}} x 16 input features ({{n_num}} numerical / {{n_cat}} categorical) plus the target. After excluding `duration`: {{rows}} x {{n_pred_ret}} ({{n_num_ret}} numerical / {{n_cat_ret}} categorical). |
| 3 | Column with the most missing values and its percentage | No column has `NaN` (0.00% everywhere). Missing information is coded as `"unknown"`: the highest share is `poutcome`, {{unk_poutcome}} of the full data. |
| 4 | Dropped columns and reasons | `duration`: target leakage, known only during or after the call (1B). No other column and no row is dropped. |
| 5 | Minority class percentage | `yes`: {{pct_yes}} ({{n_yes}} of {{rows}}); ratio {{imbalance_ratio}} : 1. |
| 6 | Training and test sizes | Train {{n_train}} rows ({{train_yes}} `yes`, {{train_yes_pct}}); test {{n_test}} rows ({{test_yes}} `yes`, {{test_yes_pct}}). Stratified 80/20, `random_state=42`. |
| 7 | Most correlated numerical pair and correlation | `{{top_pair}}`: Spearman {{top_pair_rho}} (Pearson {{top_pair_r}}), training set. Driven by the shared sentinel; {{contacted_rho_pdays_prev}} among previously contacted clients. |
| 8 | Rows affected by the outlier strategy | Clipping to train P1-P99 of `balance`, `campaign`, `previous`: {{clip_any_train}} training rows ({{clip_any_train_pct}}) and {{clip_any_test}} test rows ({{clip_any_test_pct}}) have at least one clipped value. 0 rows removed. |
| 9 | Variance explained by PC1 + PC2 | {{pc12}} (PC1 {{pc1}} + PC2 {{pc2}}), PCA on the {{n_features_out}} preprocessed training features. |
| 10 | Train / test shapes after preprocessing | Train {{z_train_shape}}; test {{z_test_shape}}. 0 `NaN` in both. |

**Remaining blocker.** Instructor approval of the dataset is not documented in the repository. Everything else in this report was computed and checked from the data.

---

## Appendix. Reproducibility

**Run everything from a clean clone** (from the repository root, after `pip install -r requirements.txt`):

```
python scripts/part1_eda.py            # stages 0-3: part1_figures/, part1_outputs/, part1_draft.md
python code/bank_preprocessing.py      # stage 4C checks only
python scripts/part2_eda.py            # stages 4-5: part2_figures/, part2_outputs/, index.md
```

- `data/raw/bank-full.csv` is committed. `python scripts/part1_download_data.py` downloads it again from UCI if needed.
- Stages 4-5 were last generated with {{versions}}.
- **Data integrity.** The file's SHA-256 is checked on load after normalising line endings to LF (`{{sha256}}`, the value recorded in 1A), because git may check the file out with CRLF line endings on Windows. The parsed data is identical in both cases.
- `random_state=42` is used for the split, the plotting sample, PCA, t-SNE and UMAP. t-SNE and UMAP coordinates can still differ slightly across library versions or machines; the conclusions in 4B rely only on structure that is stable across settings.

**Agreement with Part 1.** Values recomputed by `scripts/part2_eda.py` and compared with `part1_outputs/` (the script stops on any mismatch):

{{table_agreement}}

**Figures.**

| # | File | Title |
|---|---|---|
{{part1_figure_list}}
{{figure_list}}

**Machine-readable outputs.** Part 1: `docs/projects/eda/part1_outputs/` (listed in `part1_draft.md`). Part 2, in `docs/projects/eda/part2_outputs/`:

| File | Content |
|---|---|
| `part2_numbers.json` | every scalar value quoted in stages 4-5 |
| `part1_agreement_checks.csv` | the agreement table above |
| `categorical_checks.csv` | Table 4.1 |
| `clip_report.csv` | Table 4.2 |
| `numerical_before_after_train.csv` | Table 4.3 |
| `pca_explained_variance.csv`, `pca_loadings.csv` | Tables 4.4 and 4.5 (all components; loadings of PC1-PC4) |
| `feature_names.csv` | Table 4.6 |
| `embeddings_sample.csv` | row ids, target and t-SNE / UMAP coordinates of the plotting sample |
