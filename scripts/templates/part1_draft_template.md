<!--
  GENERATED FILE - do not edit docs/projects/eda/part1_draft.md by hand.
  Source template: scripts/templates/part1_draft_template.md
  Regenerate with: python scripts/part1_eda.py
  Every number below is substituted from values computed on data/raw/bank-full.csv.
-->

# Classification EDA - Part 1 draft (stages 0-3)

> **Status: PENDING INSTRUCTOR APPROVAL.** The repository contains no record of instructor approval for this dataset. The proposal and all EDA below are provisional until approval is documented.
>
> Scope: this file is Part 1 only (proposal and stages 0-3). Stages 4-5 and the final `index.md` are owned by the second team member, who will integrate this draft.

---

## 0. Proposal

| Item | Value |
|---|---|
| Dataset | UCI Bank Marketing, older full version: `bank-full.csv` |
| URL | <https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing> |
| Size (original) | **{{rows}} rows x {{cols}} columns** ({{n_inputs}} input features + target) |
| Target | `y`: did the client subscribe to a term deposit? (`yes` / `no`, binary) |
| Features (original) | **{{n_num}} numerical**, **{{n_cat}} categorical** (verified from the data, see 1A) |
| Features (after excluding `duration`) | **{{n_num_ret}} numerical**, **{{n_cat_ret}} categorical** = {{n_pred_ret}} predictors |
| Approval status | **Pending.** No approval evidence exists in the repository. |

**Motivation.** Direct-marketing calls are expensive, and only {{base_rate}} of the contacted clients in this dataset subscribed. A classifier that ranks clients by their chance of subscribing *before* a call is made could help the bank prioritise calls. The dataset is real (a Portuguese bank, May 2008 to November 2010), well documented, and has a mix of numerical and categorical features and a clear binary target, so it fits a classification EDA.

**First risk.** The target is imbalanced ({{pct_yes}} `yes`, ratio {{imbalance_ratio}} : 1), and the strongest single input, `duration`, is target leakage: it is only known after the call. We exclude it (see 1B), so realistic predictive signal will be weaker than published benchmarks that use it. A second risk is that the file is chronological and the subscription rate drifts strongly over time (1B).

---

## 1. Initial inspection

### 1A. Data dictionary

**Source.** UCI Machine Learning Repository, *Bank Marketing* (id 222), created by Paulo Cortez (Univ. Minho) and Sergio Moro (ISCTE-IUL), CC BY 4.0. Citation: S. Moro, R. Laureano and P. Cortez, *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*, ESM'2011, pp. 117-121.

**Download steps** (reproduced by `scripts/part1_download_data.py`):

1. Download `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` (the download link on the UCI page above).
2. The archive contains two nested zips: `bank.zip` (older version) and `bank-additional.zip` (newer version). Only `bank.zip` is used.
3. Extract `bank-full.csv` and `bank-names.txt` from `bank.zip` to `data/raw/`. The file is `;`-separated with quoted strings and a header row.
4. `bank-additional-names.txt` is also extracted, **only as documentation**: it holds UCI's official note on `duration` leakage (quoted in 1B). No `bank-additional` data is used.
5. Recorded: download time {{downloaded_at}} (UTC); `bank-full.csv` is {{file_bytes}} bytes, SHA-256 `{{sha256}}` (see `data/raw/part1_download_log.json`).

**What a row is.** One row is one client contacted in a direct-marketing phone campaign, with the attributes of their **last contact** in the current campaign. The file has {{rows}} rows ordered by date (May 2008 to November 2010). UCI notes that a client often needed more than one contact; `campaign` counts them.

**Schema check.** Read with `pandas.read_csv(sep=";")`: {{rows}} rows x {{cols}} columns. Columns with an integer dtype (`int64`) are the numerical features ({{n_num}}). Columns with an `object` dtype other than `y` are the categorical features ({{n_cat}}). This matches the expected 7 + 9 + target. The script asserts this, so it fails if the file changes.

{{table_data_dictionary}}

Notes: `day` and `month` record the date of the last contact. There is no year column, even though the data spans 2008-2010. `day` is kept numerical, following UCI, but it is a calendar position rather than a quantity.

### 1B. Data quality

**Missing values.** There are {{total_nan}} `NaN` values in the whole file. Missing information is instead coded as the literal level `"unknown"` in {{unknown_cols}}.

{{table_missing}}

- `"unknown"` counts on the full data: `job` {{unk_job}}, `education` {{unk_education}}, `contact` {{unk_contact}}, `poutcome` {{unk_poutcome}}. {{rows_any_unknown}} rows ({{rows_any_unknown_pct}}) have at least one `"unknown"`. Excluding `poutcome`, the figure is {{rows_any_unknown_ex_pout}} rows ({{rows_any_unknown_ex_pout_pct}}). Dropping such rows is therefore not an option.
- **Decision: keep `"unknown"` as its own category.** The data supports this. (i) For `poutcome`, `"unknown"` is structural: it almost exactly marks "not previously contacted" (see the `pdays` section). (ii) For `contact`, `"unknown"` is a recording-practice artifact: the first {{contact_unknown_first_rows}} rows in file (time) order are all `"unknown"`, and {{contact_unknown_in_first_30pct}} of all `contact = "unknown"` rows fall in the first 30% of the file. (iii) `"unknown"` levels have their own subscription rates (for `contact`: {{contact_unknown_rate_y}} vs {{contact_known_rate_y}} when known). This means the values are not missing completely at random, and imputing them with the mode would erase signal.

**Duplicates.** There are {{dup_full}} exact duplicate rows among all {{cols}} columns. After `duration` is excluded, {{dup_no_dur}} rows duplicate an earlier row ({{dup_no_dur_rows_involved}} rows involved, in {{dup_feat_groups}} groups). None of these groups has conflicting labels ({{dup_feat_conflicts}} groups with different `y`). **Decision: keep them.** They differed only by call duration, so they are most likely different contacts with identical profiles rather than copy errors, and they are a negligible share of the data.

**`pdays = -1` sentinel.** UCI documents `-1` as "client was not previously contacted". It is not an impossible value but a code, and the data confirms this:

- {{pdays_neg1}} rows ({{pdays_neg1_pct}}) have `pdays = -1`. Exactly the same rows have `previous = 0` ({{prev0}} rows; {{sent_prev_mismatch}} mismatches).
- No other negative value and no `pdays = 0` exist ({{pdays_other_neg}} and {{pdays_zero}} rows). Real values run from {{pdays_pos_min}} to {{pdays_pos_max}} days.
- Treating `-1` as a number is misleading. The mean of `pdays` is {{pdays_mean_raw}} with the sentinel and {{pdays_mean_contacted}} among previously contacted clients only.
- Cross-tab with `poutcome`:

{{table_pdays_poutcome}}

- Subscription rate: {{sent_rate_y}} for never-contacted clients vs {{nonsent_rate_y}} for previously contacted ones. The sentinel is informative, so the "not previously contacted" information must be preserved in preprocessing (for example, a binary flag plus a `pdays` value that is masked or neutral for those clients).

**Impossible, inconsistent and implausible values** (full data):

{{table_checks}}

- No impossible values were found. The only true inconsistency is {{pout_unknown_contacted}} rows with `previous > 0` but `poutcome = "unknown"`. These are kept, because the category already represents "unrecorded".
- `balance < 0` in {{neg_balance}} rows ({{neg_balance_pct}}) is **not** an error: the average balance of an overdrawn account can be negative.

**Outliers.** These are quantified with the 1.5 x IQR rule on the training set in 2A (Figure 3). **Decision: no outliers are removed in Part 1.** Extreme `balance`, `campaign` and `previous` values are plausible, and removing them would change the population. Part 2 should prefer robust or monotone transforms (see Handoff).

**Identifiers and constants.** There is no ID column. The highest unique-value ratio is `{{max_unique_ratio_col}}` with {{max_unique_values}} distinct values ({{max_unique_ratio}} of rows), which is a measurement, not an identifier. There are {{n_constant}} constant columns. The most dominant level is in `{{most_dominant_col}}` ({{most_dominant_share}} `no`; `default = yes` has only {{default_yes}} rows, {{default_yes_pct}}). It is nearly constant but kept, as Part 2 can decide whether it carries enough signal.

**Leakage investigation.**

1. **`duration` (excluded).** UCI's attribute note (in `bank-additional-names.txt` from the same official archive) says: *"this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model."* The data confirms this:
   - Spearman correlation of `duration` with `y` is {{dur_spearman_y}}. The highest correlation among the other numerical features is {{other_max_abs_spearman_y_val}} (`{{other_max_abs_spearman_y_col}}`).
   - There are {{dur0}} rows with `duration = 0`. All of them have `y = no` (check result: {{dur0_all_no}}), exactly as UCI's note says.
   - The subscription rate rises steadily with call length:

{{table_duration}}

   Our task is to predict subscription **before** calling, so `duration` cannot be a predictor. **It is dropped before the split and is not analysed further.**

2. **Row order (time) - not a column, but a leakage risk.** The file is sorted by date. Splitting the rows into deciles by file position shows strong drift: the subscription rate goes from {{order_rate_first}} in the first decile to {{order_rate_last}} in the last.

{{table_row_order}}

   The row index must **never** be used as a feature. A random stratified split (required here) mixes periods, so test performance will be optimistic about the future. We record this as a limitation; it is not corrected in Part 1.

3. `poutcome`, `pdays` and `previous` describe *previous* campaigns, so they are known before the call. `campaign` includes the current (last) call, but the number of attempts so far is known when deciding on the next call, so it is kept. `contact`, `day` and `month` describe the last contact. They are plausible as scheduling attributes, but their link to the data-collection period (see `contact` above) should be kept in mind.

**Columns dropped and reasons.**

| Column | Reason |
|---|---|
| `duration` | Target leakage: only known during or after the call (UCI note and evidence above). |

No other columns are dropped. **Result: original {{rows}} x {{cols}} ({{n_num}} numerical + {{n_cat}} categorical + `y`); after the exclusion {{rows}} x {{cols_after}} ({{n_num_ret}} numerical + {{n_cat_ret}} categorical + `y`).** No rows are removed.

### 1C. Target

{{table_target}}

The data is imbalanced, with a ratio of **{{imbalance_ratio}} : 1** (`no` : `yes`). Always predicting `no` would reach {{majority_acc}} accuracy. Part 2 should evaluate with metrics that are robust to imbalance (for example PR-AUC, recall/precision on `yes`, balanced accuracy) and keep stratification.

### 1D. Train/test split

- `sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`, with shuffling (the default).
- `X` holds the {{n_pred_ret}} retained predictors (`duration` excluded), and `y` is encoded as `yes = 1`, `no = 0`.
- The split is done **before any preprocessing is fitted**. No imputation, encoding or scaling is fitted in Part 1. Stages 2-3 below use the **training set only**, so nothing is learned from the test set.

{{table_split}}

- Train: {{n_train}} rows ({{train_no}} `no`, {{train_yes}} `yes`, {{train_yes_pct}} positive).
- Test: {{n_test}} rows ({{test_no}} `no`, {{test_yes}} `yes`, {{test_yes_pct}} positive).
- The exact rows are listed in `part1_outputs/train_test_split_indices.csv` (`row_id` = 0-based row position in `bank-full.csv`).

---

## 2. Univariate analysis (training set, n = {{n_train}})

### 2A. Numerical features

There are {{n_num_ret}} retained numerical features: {{num_ret_list}}. Because `pdays` and `previous` are dominated by the sentinel/zero group, the table also shows them for previously contacted clients only (n = {{n_contacted_train}}, {{contacted_train_pct}}).

{{table_num_stats}}

Modes and skewness:

{{table_num_modes}}

{{fig_hist_age_balance_day_md}}

**Figure {{fig_hist_age_balance_day}} - conclusion.**
- **`age`** is unimodal and mildly right-skewed (skew {{age_skew}}). The mean ({{age_mean}}) is above the median ({{age_med}}), and the mode is {{age_mode}}. {{age_25_60_pct}} of clients are 25-60. A thin tail above about 60 holds the IQR outliers ({{age_out}}, {{age_outp}}%), which are real older clients.
- **`balance`** is extremely right-skewed (skew {{balance_skew}}). The mean of {{balance_mean}} is three times the median of {{balance_med}}, and there is a spike at exactly 0 ({{bal_zero_train_pct}} of clients). {{bal_neg_train_pct}} are negative (min {{balance_min}}) and the max is {{balance_max}}. The plot is clipped to P1-P99 [{{bal_p01}}, {{bal_p99}}] for readability.
- **`day`** has no skew ({{day_skew}}) and no outliers, but it is multimodal, with peaks on specific days ({{day_top}}) and uneven counts across the month. This reflects call scheduling rather than a smooth quantity.

{{fig_counts_campaign_pdays_previous_md}}

**Figure {{fig_counts_campaign_pdays_previous}} - conclusion.**
- **`campaign`** is a heavy-tailed count (skew {{campaign_skew}}). The mode is {{campaign_mode}} contact ({{campaign_modesh}}% of clients), {{camp_le3_pct}} received at most 3, and {{camp_gt10}} clients received more than 10 (max {{campaign_max}}).
- **`pdays`** applies only to previously contacted clients ({{contacted_train_pct}}). Its distribution is multimodal (median {{pdays_c_med}}, IQR {{pdays_c_q1}}-{{pdays_c_q3}}). The most frequent values are {{pdays_top}} days, i.e. clients re-contacted about three or six months after the previous campaign. Figure 2 also shows a further cluster just under 400 days.
- **`previous`** among contacted clients has a median of {{previous_c_med}} and a mode of {{previous_c_mode}}, with a very long tail up to {{previous_c_max}} (skew {{previous_c_skew}}).

{{fig_boxplots_numerical_md}}

**Figure {{fig_boxplots_numerical}} - conclusion.** Using the 1.5 x IQR rule:
- `balance` has the most real outliers ({{balance_out}}, {{balance_outp}}%), on both sides. `campaign` has {{campaign_out}} ({{campaign_outp}}%) and `age` {{age_out}} ({{age_outp}}%).
- For `pdays` and `previous`, the IQR collapses to the single sentinel value (Q1 = Q3 = {{pdays_q1}} and {{previous_q1}}). Every previously contacted client ({{pdays_out}}) is therefore flagged as an "outlier". This is an artifact of the sentinel, not evidence of bad data. Within contacted clients, there are only {{pdays_c_out}} outliers for `pdays` and {{previous_c_out}} for `previous`.
- {{n_skew_gt1}} features have |skew| > 1 ({{skew_gt1_list}}). This motivates rank-based correlation in 3A, and transforms (log/Yeo-Johnson, or binning) and/or robust scaling in Part 2.

### 2B. Categorical features

There are {{n_cat_ret}} retained categorical features: {{cat_ret_list}}. Together they have {{total_dummies}} levels (the one-hot width before dropping any reference level).

{{table_cat_card}}

Full frequency tables (training set):

{{table_cat_freq}}

{{fig_categorical_frequencies_md}}

**Figure {{fig_categorical_frequencies}} - conclusion.**
- `job` is dominated by blue-collar ({{f_job_blue_collar}}), management ({{f_job_management}}) and technician ({{f_job_technician}}) clients.
- `marital` is mostly married ({{f_marital_married}}), and `education` mostly secondary ({{f_education_secondary}}).
- `default` is almost constant ({{f_default_no}} `no`).
- `month` is concentrated in May-August ({{month_mjja_pct}} of contacts; May alone is {{f_month_may}}). March, September, October and December are thin ({{f_month_mar}}, {{f_month_sep}}, {{f_month_oct}}, {{f_month_dec}}).
- `poutcome` is {{f_poutcome_unknown}} `"unknown"`, which is the "never contacted" group.

{{fig_cardinality_unknown_md}}

**Figure {{fig_cardinality_unknown}} - conclusion.**
- **No feature has high cardinality.** The maximum is 12 levels (`job`, `month`), so one-hot encoding is feasible.
- **Rare levels (< 1% of training rows):** {{rare_levels}}. March, September and October are also small (just over 1%). Their subscription-rate estimates are noisier (see the wide CIs in Figure {{fig_month_volume_rate}}), so Part 2 may consider grouping them if a model overfits.
- **`"unknown"` is substantial** in `poutcome` and `contact`, small in `education`, and marginal in `job`. It is kept as a category (1B).

---

## 3. Bivariate and multivariate analysis (training set)

### 3A. Numerical x numerical

**Choice: Spearman.** {{n_skew_gt1}} of the {{n_num_ret}} numerical features are strongly skewed ({{skew_gt1_list}}), and `pdays`/`previous` are dominated by a sentinel/zero mass. Pearson assumes linear relationships and is sensitive to extreme values, while Spearman uses ranks: it captures monotone relationships and is robust to outliers. Pearson is shown only for comparison.

{{fig_corr_spearman_pearson_md}}

Top pairs by |Spearman|:

{{table_corr_pairs}}

**Figure {{fig_corr_spearman_pearson}} - conclusion.**
- The only strongly correlated pair is **`{{top_pair}}`**: Spearman {{top_pair_rho}}, compared with Pearson {{top_pair_r}}. Pearson understates it because the relationship is monotone but far from linear (a large sentinel mass plus long tails).
- All other pairs have |rho| <= {{max_other_abs_rho}}. The next strongest are `{{sec_pair}}` ({{sec_pair_rho}}) and `{{third_pair}}` ({{third_pair_rho}}). Apart from the pair above, the numerical features are close to non-redundant.

{{fig_pdays_vs_previous_md}}

**Figure {{fig_pdays_vs_previous}} - conclusion.** The {{top_pair_rho}} correlation is driven almost entirely by the shared sentinel, since `pdays = -1` exactly when `previous = 0` (1B). Among previously contacted clients, Spearman is only {{contacted_rho_pdays_prev}} (Pearson {{contacted_r_pdays_prev}}), so the actual magnitudes are nearly unrelated. **Redundancy:** the two features duplicate the binary "previously contacted" information, not their values. Part 2 should encode that flag once and keep both magnitudes (or test dropping one) instead of treating them as collinear duplicates.

### 3B. Categorical x target

The training base rate is {{train_base_rate}}. Rates for every level of every categorical feature, with 95% Wilson intervals and lift, are in `part1_outputs/categorical_target_rates_train.csv`. The spread between each feature's highest and lowest level rate:

{{table_cat_rate_spread}}

The features with the largest spread are {{spread_top3}}. The smallest is {{spread_bottom}}. The selected figures cover the strongest and most decision-relevant features.

{{fig_rate_poutcome_contact_housing_loan_md}}

**Figure {{fig_rate_poutcome_contact_housing_loan}} - conclusion.**
- **`poutcome`** separates the classes most. Clients whose previous campaign was a `success` subscribe at {{r_poutcome_success}}, against {{r_poutcome_failure}} (`failure`), {{r_poutcome_other}} (`other`) and {{r_poutcome_unknown}} (`unknown` = never contacted).
- **`contact = unknown`** has a much lower rate ({{r_contact_unknown}}) than `cellular` ({{r_contact_cellular}}) and `telephone` ({{r_contact_telephone}}). This matches the early-period artifact found in 1B, so it partly encodes time.
- Clients **without a housing loan** ({{r_housing_no}} vs {{r_housing_yes}}) or **without a personal loan** ({{r_loan_no}} vs {{r_loan_yes}}) subscribe more often.
- The intervals for `poutcome = success`, `contact = unknown`, `housing` and `loan` do not overlap with their contrasts. `cellular` and `telephone` overlap, so the contact channel itself adds little once `unknown` is separated out.

{{fig_month_volume_rate_md}}

**Figure {{fig_month_volume_rate}} - conclusion.** Contact volume and subscription rate move in opposite directions (Spearman correlation across the 12 months: {{month_vol_rate_rho}}).
- The busiest month has the lowest rate: {{month_rate_min}}.
- Low-volume months have rates above 30%: {{months_rate_gt30}}. The maximum is {{month_rate_max}}.
- Volume and rate are drawn as two panels on separate axes, not as a dual-axis chart.
- `month` is strongly predictive, but given the drift found in 1B it likely mixes seasonality with the period of data collection (there is no year column). Its confidence intervals are wide for the thin months.

{{fig_rate_job_education_md}}

**Figure {{fig_rate_job_education}} - conclusion.**
- **`job`:** students ({{r_job_student}}) and retired clients ({{r_job_retired}}) subscribe far above the base rate. Blue-collar clients ({{r_job_blue_collar}}), housemaids ({{r_job_housemaid}}), entrepreneurs ({{r_job_entrepreneur}}) and services workers ({{r_job_services}}) subscribe below it.
- **`education`:** the rate increases from primary ({{r_education_primary}}) to secondary ({{r_education_secondary}}) to tertiary ({{r_education_tertiary}}). `unknown` ({{r_education_unknown}}) has a wide interval, consistent with its small size.
- These features carry moderate, roughly ordinal signal. `job = unknown` ({{r_job_unknown}}) has a very wide interval and is too small to interpret.

### 3C. Numerical x categorical (grouped boxplots, split by `y`)

Each pair below was chosen because 2A/3B suggested a possible interaction, i.e. that the numerical feature's link to `y` might depend on the group. Every boxplot is also split by `y`. Medians and IQRs for every box are in `part1_outputs/grouped_box_stats_train.csv`.

{{fig_box_age_job_y_md}}

{{table_box_age}}

**Figure {{fig_box_age_job_y}} - conclusion.** *Why this pair:* `age` and `job` are strongly linked (students and retirees are at the age extremes), and both have high-rate groups in 3B.
- **Location:** median age ranges from {{b_age_student_no_med}} (students, `no`) to {{b_age_retired_no_med}} (retired, `no`).
- **Within groups:** for most jobs, the `yes` and `no` boxes are within a few years of each other. The clear exception is **retired** clients: those who subscribed are older (median {{b_age_retired_yes_med}} vs {{b_age_retired_no_med}}) and more spread out (IQR {{b_age_retired_yes_iqr}} vs {{b_age_retired_no_iqr}}).
- **Overall:** the age effect is non-linear. Clients aged 60+ subscribe at {{age_ge60_rate}} (n = {{age_ge60_n}}) and those under 25 at {{age_lt25_rate}} (n = {{age_lt25_n}}), against {{age_30_59_rate}} for ages 30-59. The overall medians by class are almost equal ({{med_age_yes}} vs {{med_age_no}}), which is why a linear age effect would be weak. Part 2 should consider age bins or non-linear models.

{{fig_box_balance_education_y_md}}

{{table_box_balance}}

**Figure {{fig_box_balance_education_y}} - conclusion.** *Why this pair:* balance is a likely proxy for the ability to make a deposit, and education is a socioeconomic grouping with an ordinal rate gradient in 3B. The figure checks whether balance still matters within an education level.
- **Location:** within every education level, subscribers have a higher median balance:
  - primary: {{b_balance_primary_yes_med}} vs {{b_balance_primary_no_med}}
  - secondary: {{b_balance_secondary_yes_med}} vs {{b_balance_secondary_no_med}}
  - tertiary: {{b_balance_tertiary_yes_med}} vs {{b_balance_tertiary_no_med}}
- **Spread:** subscribers also have a larger IQR (for example tertiary {{b_balance_tertiary_yes_iqr}} vs {{b_balance_tertiary_no_iqr}}).
- The balance effect is therefore not explained by education alone. Long tails on both sides (a symlog axis is needed) confirm that `balance` needs a robust transform.

{{fig_box_campaign_contact_y_md}}

{{table_box_campaign}}

**Figure {{fig_box_campaign_contact_y}} - conclusion.** *Why this pair:* `campaign` measures call effort, and `contact` is the channel used. This checks whether extra calls relate to subscription in the same way for each channel, and whether the `unknown` (early-period) group behaves differently.
- **Location:** medians are equal ({{b_campaign_cellular_no_med}}) in every box, so location barely differs.
- **Spread:** the difference is in the upper tail. For `cellular` and `telephone`, subscribers have a narrower IQR (cellular {{b_campaign_cellular_yes_iqr}} vs {{b_campaign_cellular_no_iqr}}; telephone {{b_campaign_telephone_yes_iqr}} vs {{b_campaign_telephone_no_iqr}}), while for `unknown` they are equal ({{b_campaign_unknown_yes_iqr}} vs {{b_campaign_unknown_no_iqr}}). The extreme counts belong mostly to non-subscribers.
- **Overall:** the rate is {{camp1_rate}} after a single contact and {{camp_gt5_rate}} after more than 5 contacts (n = {{camp_gt5_n}}). Repeated calls have diminishing returns. The signal is in the tail, so a capped or log-transformed `campaign` is preferable to the raw count.

---

## Handoff for Part 2 (preprocessing, stages 4-5)

**Split - use exactly this.**
- Function: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`, with `X = df[numerical + categorical]` in the column order of `part1_outputs/feature_lists.json`, and `y = (df["y"] == "yes").astype(int)`.
- Input: `data/raw/bank-full.csv` (SHA-256 `{{sha256}}`), read with `pd.read_csv(path, sep=";")`.
- Result: train {{n_train}} rows ({{train_yes}} positive), test {{n_test}} rows ({{test_yes}} positive).
- Row membership: `part1_outputs/train_test_split_indices.csv`. Specification: `part1_outputs/train_test_split.json`.

**Retained features (`duration` excluded).**
- Numerical ({{n_num_ret}}): {{num_ret_list}}
- Categorical ({{n_cat_ret}}): {{cat_ret_list}}
- Target: `y` (`yes` = 1, `no` = 0). The excluded column is `duration` (leakage).

**Findings that justify preprocessing choices** (fit everything on the training set only, inside a pipeline):
1. **No NaNs.** `"unknown"` is a legitimate category in `job`, `education`, `contact` and `poutcome`. Do not impute it, and keep it in one-hot encoding.
2. **`pdays = -1`** is a sentinel identical to `previous = 0` (and to `poutcome = unknown`, except for {{pout_unknown_contacted}} rows). Create a "previously contacted" flag, and neutralise or mask `pdays` for never-contacted clients rather than scaling `-1` as a number.
3. **Skew and outliers:** `balance` (skew {{balance_skew}}, {{balance_outp}}% IQR outliers, negative values present, so use Yeo-Johnson or a signed log rather than a plain log), `campaign` (skew {{campaign_skew}}) and `previous` (skew {{previous_skew}}). No rows were removed. Use robust scaling and/or transforms or caps fitted on train.
4. **Cardinality is low** (max 12, {{total_dummies}} levels in total), so one-hot encoding is fine. Rare levels: {{rare_levels}}. Use `handle_unknown="ignore"`.
5. **`default`** is near-constant ({{default_yes_pct}} `yes`). Keep it or test dropping it.
6. **Imbalance** {{imbalance_ratio}} : 1. Keep stratification in any CV, and use imbalance-aware metrics and/or class weights.
7. **Time drift:** the file is chronological, and the rate goes from {{order_rate_first}} to {{order_rate_last}} across deciles. `month` and `contact = unknown` partly encode the period. Never use the row index as a feature, and state the optimism of a random split as a limitation.
8. **Duplicates:** {{dup_no_dur}} feature-level duplicates after dropping `duration`, with no conflicting labels. They were kept.

**Machine-readable outputs** (`docs/projects/eda/part1_outputs/`):

| File | Content |
|---|---|
| `part1_numbers.json` | every scalar value quoted in this draft |
| `feature_lists.json` | retained numerical/categorical lists, levels, exclusions, sentinel |
| `train_test_split.json`, `train_test_split_indices.csv`, `train_test_split_summary.csv` | split spec, row membership, sizes and class counts |
| `data_dictionary.csv` | 1A table |
| `missing_and_unknown.csv`, `consistency_checks.csv`, `pdays_sentinel_vs_poutcome.csv`, `duration_leakage_by_bin.csv`, `row_order_drift_by_decile.csv` | 1B evidence |
| `target_distribution.csv` | 1C |
| `numerical_summary_train.csv` | 2A statistics, skew, IQR outliers |
| `categorical_frequencies_train.csv`, `categorical_cardinality_train.csv` | 2B |
| `corr_spearman_train.csv`, `corr_pearson_train.csv`, `corr_pairs_train.csv` | 3A |
| `categorical_target_rates_train.csv` | 3B, all levels with Wilson CIs |
| `grouped_box_stats_train.csv` | 3C medians and IQRs |

**Figures** (`docs/projects/eda/part1_figures/`):

| # | File | Title |
|---|---|---|
{{figure_list}}

**Reproduce:** `python scripts/part1_download_data.py && python scripts/part1_eda.py` (Python 3.9, pandas, numpy, matplotlib, seaborn, scikit-learn, tabulate). No model is trained in Part 1.
