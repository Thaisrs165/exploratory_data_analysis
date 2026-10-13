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
| Size (original) | **45,211 rows x 17 columns** (16 input features + target) |
| Target | `y`: did the client subscribe to a term deposit? (`yes` / `no`, binary) |
| Features (original) | **7 numerical**, **9 categorical** (verified from the data, see 1A) |
| Features (after excluding `duration`) | **6 numerical**, **9 categorical** = 15 predictors |
| Approval status | **Pending.** No approval evidence exists in the repository. |

**Motivation.** Direct-marketing calls are expensive, and only 11.70% of the contacted clients in this dataset subscribed. A classifier that ranks clients by their chance of subscribing *before* a call is made could help the bank prioritise calls. The dataset is real (a Portuguese bank, May 2008 to November 2010), well documented, and has a mix of numerical and categorical features and a clear binary target, so it fits a classification EDA.

**First risk.** The target is imbalanced (11.70% `yes`, ratio 7.55 : 1), and the strongest single input, `duration`, is target leakage: it is only known after the call. We exclude it (see 1B), so realistic predictive signal will be weaker than published benchmarks that use it. A second risk is that the file is chronological and the subscription rate drifts strongly over time (1B).

---

## 1. Initial inspection

### 1A. Data dictionary

**Source.** UCI Machine Learning Repository, *Bank Marketing* (id 222), created by Paulo Cortez (Univ. Minho) and Sergio Moro (ISCTE-IUL), CC BY 4.0. Citation: S. Moro, R. Laureano and P. Cortez, *Using Data Mining for Bank Direct Marketing: An Application of the CRISP-DM Methodology*, ESM'2011, pp. 117-121.

**Download steps** (reproduced by `scripts/part1_download_data.py`):

1. Download `https://archive.ics.uci.edu/static/public/222/bank+marketing.zip` (the download link on the UCI page above).
2. The archive contains two nested zips: `bank.zip` (older version) and `bank-additional.zip` (newer version). Only `bank.zip` is used.
3. Extract `bank-full.csv` and `bank-names.txt` from `bank.zip` to `data/raw/`. The file is `;`-separated with quoted strings and a header row.
4. `bank-additional-names.txt` is also extracted, **only as documentation**: it holds UCI's official note on `duration` leakage (quoted in 1B). No `bank-additional` data is used.
5. Recorded: download time 2026-10-13T21:56:18+00:00 (UTC); `bank-full.csv` is 4,610,348 bytes, SHA-256 `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d` (see `data/raw/part1_download_log.json`).

**What a row is.** One row is one client contacted in a direct-marketing phone campaign, with the attributes of their **last contact** in the current campaign. The file has 45,211 rows ordered by date (May 2008 to November 2010). UCI notes that a client often needed more than one contact; `campaign` counts them.

**Schema check.** Read with `pandas.read_csv(sep=";")`: 45,211 rows x 17 columns. Columns with an integer dtype (`int64`) are the numerical features (7). Columns with an `object` dtype other than `y` are the categorical features (9). This matches the expected 7 + 9 + target. The script asserts this, so it fails if the file changes.

| column      | meaning                                                                                        | role        | pandas dtype   | unit       | observed values / range                                                                                                                                      | Part 1 status      |
|:------------|:-----------------------------------------------------------------------------------------------|:------------|:---------------|:-----------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|
| `age`       | Client age                                                                                     | numerical   | int64          | years      | 18 to 95                                                                                                                                                     | retained           |
| `job`       | Type of job                                                                                    | categorical | object         | -          | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` | retained           |
| `marital`   | Marital status (`divorced` = divorced or widowed)                                              | categorical | object         | -          | `divorced`, `married`, `single`                                                                                                                              | retained           |
| `education` | Education level                                                                                | categorical | object         | -          | `primary`, `secondary`, `tertiary`, `unknown`                                                                                                                | retained           |
| `default`   | Has credit in default?                                                                         | categorical | object         | -          | `no`, `yes`                                                                                                                                                  | retained           |
| `balance`   | Average yearly balance                                                                         | numerical   | int64          | euros      | -8,019 to 102,127                                                                                                                                            | retained           |
| `housing`   | Has housing loan?                                                                              | categorical | object         | -          | `no`, `yes`                                                                                                                                                  | retained           |
| `loan`      | Has personal loan?                                                                             | categorical | object         | -          | `no`, `yes`                                                                                                                                                  | retained           |
| `contact`   | Contact communication type (last contact, current campaign)                                    | categorical | object         | -          | `cellular`, `telephone`, `unknown`                                                                                                                           | retained           |
| `day`       | Day of month of the last contact                                                               | numerical   | int64          | day (1-31) | 1 to 31                                                                                                                                                      | retained           |
| `month`     | Month of year of the last contact                                                              | categorical | object         | -          | `jan`, `feb`, `mar`, `apr`, `may`, `jun`, `jul`, `aug`, `sep`, `oct`, `nov`, `dec`                                                                           | retained           |
| `duration`  | Duration of the last contact                                                                   | numerical   | int64          | seconds    | 0 to 4,918                                                                                                                                                   | excluded (leakage) |
| `campaign`  | Contacts performed during this campaign for this client, incl. the last one                    | numerical   | int64          | count      | 1 to 63                                                                                                                                                      | retained           |
| `pdays`     | Days since the client was last contacted in a previous campaign; -1 = not previously contacted | numerical   | int64          | days       | -1 to 871                                                                                                                                                    | retained           |
| `previous`  | Contacts performed before this campaign for this client                                        | numerical   | int64          | count      | 0 to 275                                                                                                                                                     | retained           |
| `poutcome`  | Outcome of the previous marketing campaign                                                     | categorical | object         | -          | `failure`, `other`, `success`, `unknown`                                                                                                                     | retained           |
| `y`         | Target: has the client subscribed to a term deposit?                                           | target      | object         | -          | `no`, `yes`                                                                                                                                                  | target             |

Notes: `day` and `month` record the date of the last contact. There is no year column, even though the data spans 2008-2010. `day` is kept numerical, following UCI, but it is a calendar position rather than a quantity.

### 1B. Data quality

**Missing values.** There are 0 `NaN` values in the whole file. Missing information is instead coded as the literal level `"unknown"` in `job`, `education`, `contact`, `poutcome`.

| column      |   NaN count |   NaN % |   "unknown" count |   "unknown" % |
|:------------|------------:|--------:|------------------:|--------------:|
| `age`       |           0 |    0.00 |                 0 |          0.00 |
| `job`       |           0 |    0.00 |               288 |          0.64 |
| `marital`   |           0 |    0.00 |                 0 |          0.00 |
| `education` |           0 |    0.00 |              1857 |          4.11 |
| `default`   |           0 |    0.00 |                 0 |          0.00 |
| `balance`   |           0 |    0.00 |                 0 |          0.00 |
| `housing`   |           0 |    0.00 |                 0 |          0.00 |
| `loan`      |           0 |    0.00 |                 0 |          0.00 |
| `contact`   |           0 |    0.00 |             13020 |         28.80 |
| `day`       |           0 |    0.00 |                 0 |          0.00 |
| `month`     |           0 |    0.00 |                 0 |          0.00 |
| `duration`  |           0 |    0.00 |                 0 |          0.00 |
| `campaign`  |           0 |    0.00 |                 0 |          0.00 |
| `pdays`     |           0 |    0.00 |                 0 |          0.00 |
| `previous`  |           0 |    0.00 |                 0 |          0.00 |
| `poutcome`  |           0 |    0.00 |             36959 |         81.75 |
| `y`         |           0 |    0.00 |                 0 |          0.00 |

- `"unknown"` counts on the full data: `job` 288 (0.64%), `education` 1,857 (4.11%), `contact` 13,020 (28.80%), `poutcome` 36,959 (81.75%). 37,369 rows (82.65%) have at least one `"unknown"`. Excluding `poutcome`, the figure is 14,304 rows (31.64%). Dropping such rows is therefore not an option.
- **Decision: keep `"unknown"` as its own category.** The data supports this. (i) For `poutcome`, `"unknown"` is structural: it almost exactly marks "not previously contacted" (see the `pdays` section). (ii) For `contact`, `"unknown"` is a recording-practice artifact: the first 12,657 rows in file (time) order are all `"unknown"`, and 97.21% of all `contact = "unknown"` rows fall in the first 30% of the file. (iii) `"unknown"` levels have their own subscription rates (for `contact`: 4.07% vs 14.78% when known). This means the values are not missing completely at random, and imputing them with the mode would erase signal.

**Duplicates.** There are 0 exact duplicate rows among all 17 columns. After `duration` is excluded, 16 rows duplicate an earlier row (31 rows involved, in 15 groups). None of these groups has conflicting labels (0 groups with different `y`). **Decision: keep them.** They differed only by call duration, so they are most likely different contacts with identical profiles rather than copy errors, and they are a negligible share of the data.

**`pdays = -1` sentinel.** UCI documents `-1` as "client was not previously contacted". It is not an impossible value but a code, and the data confirms this:

- 36,954 rows (81.74%) have `pdays = -1`. Exactly the same rows have `previous = 0` (36,954 rows; 0 mismatches).
- No other negative value and no `pdays = 0` exist (0 and 0 rows). Real values run from 1 to 871 days.
- Treating `-1` as a number is misleading. The mean of `pdays` is 40.20 with the sentinel and 224.58 among previously contacted clients only.
- Cross-tab with `poutcome`:

| pdays group   |   failure |   other |   success |   unknown |
|:--------------|----------:|--------:|----------:|----------:|
| pdays = -1    |         0 |       0 |         0 |     36954 |
| pdays >= 1    |      4901 |    1840 |      1511 |         5 |

- Subscription rate: 9.16% for never-contacted clients vs 23.07% for previously contacted ones. The sentinel is informative, so the "not previously contacted" information must be preserved in preprocessing (for example, a binary flag plus a `pdays` value that is masked or neutral for those clients).

**Impossible, inconsistent and implausible values** (full data):

| check                                               |   rows | interpretation                                                        | action                  |
|:----------------------------------------------------|-------:|:----------------------------------------------------------------------|:------------------------|
| age < 18 or > 100                                   |      0 | impossible for a bank client contact list                             | none                    |
| age > 90                                            |      7 | plausible but extreme                                                 | keep                    |
| day outside 1-31                                    |      0 | impossible                                                            | none                    |
| day exceeds days in its month (e.g. 31 apr, 30 feb) |      0 | impossible calendar date                                              | none                    |
| campaign < 1 (must include the last contact)        |      0 | impossible by definition                                              | none                    |
| previous < 0                                        |      0 | impossible                                                            | none                    |
| pdays < -1, or pdays = 0                            |      0 | impossible / undefined code                                           | none                    |
| pdays = -1 while previous > 0 (or the reverse)      |      0 | inconsistent                                                          | none                    |
| poutcome = unknown while previous > 0               |      5 | inconsistent: a previous contact exists but its outcome is unrecorded | keep; treat as category |
| poutcome known while previous = 0                   |      0 | inconsistent                                                          | none                    |
| balance < 0                                         |   3766 | valid: overdrawn account (balance is not bounded at 0)                | keep                    |
| duration = 0 s                                      |      3 | consistent with no conversation; all have y = no                      | moot: duration excluded |
| job = student and age > 40                          |     10 | implausible but possible                                              | keep                    |
| job = retired and age < 40                          |     25 | implausible but possible (early/disability retirement)                | keep                    |

- No impossible values were found. The only true inconsistency is 5 rows with `previous > 0` but `poutcome = "unknown"`. These are kept, because the category already represents "unrecorded".
- `balance < 0` in 3,766 rows (8.33%) is **not** an error: the average balance of an overdrawn account can be negative.

**Outliers.** These are quantified with the 1.5 x IQR rule on the training set in 2A (Figure 3). **Decision: no outliers are removed in Part 1.** Extreme `balance`, `campaign` and `previous` values are plausible, and removing them would change the population. Part 2 should prefer robust or monotone transforms (see Handoff).

**Identifiers and constants.** There is no ID column. The highest unique-value ratio is `balance` with 7,168 distinct values (0.159 of rows), which is a measurement, not an identifier. There are 0 constant columns. The most dominant level is in `default` (98.20% `no`; `default = yes` has only 815 rows, 1.80%). It is nearly constant but kept, as Part 2 can decide whether it carries enough signal.

**Leakage investigation.**

1. **`duration` (excluded).** UCI's attribute note (in `bank-additional-names.txt` from the same official archive) says: *"this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model."* The data confirms this:
   - Spearman correlation of `duration` with `y` is 0.342. The highest correlation among the other numerical features is 0.169 (`previous`).
   - There are 3 rows with `duration = 0`. All of them have `y = no` (check result: yes), exactly as UCI's note says.
   - The subscription rate rises steadily with call length:

| duration (s)   |   rows |   subscribed |   subscription rate % |
|:---------------|-------:|-------------:|----------------------:|
| 0              |      3 |            0 |                  0.00 |
| 1-60           |   4763 |            9 |                  0.19 |
| 61-120         |   9278 |          202 |                  2.18 |
| 121-300        |  18893 |         1620 |                  8.57 |
| 301-600        |   8484 |         1625 |                 19.15 |
| 601-4918       |   3790 |         1833 |                 48.36 |

   Our task is to predict subscription **before** calling, so `duration` cannot be a predictor. **It is dropped before the split and is not analysed further.**

2. **Row order (time) - not a column, but a leakage risk.** The file is sorted by date. Splitting the rows into deciles by file position shows strong drift: the subscription rate goes from 3.0% in the first decile to 47.1% in the last.

| row decile (file order)   |   rows |   subscription rate % |   contact = unknown % |   poutcome = unknown % |
|:--------------------------|-------:|----------------------:|----------------------:|-----------------------:|
| D1                        |   4522 |                   3.0 |                 100.0 |                  100.0 |
| D2                        |   4521 |                   3.7 |                 100.0 |                  100.0 |
| D3                        |   4521 |                   4.8 |                  79.9 |                  100.0 |
| D4                        |   4521 |                   6.4 |                   0.5 |                  100.0 |
| D5                        |   4521 |                   6.1 |                   0.2 |                  100.0 |
| D6                        |   4521 |                   6.2 |                   1.6 |                   83.0 |
| D7                        |   4520 |                  10.5 |                   0.1 |                   71.4 |
| D8                        |   4522 |                  13.1 |                   0.0 |                   61.1 |
| D9                        |   4521 |                  16.1 |                   0.0 |                   60.7 |
| D10                       |   4521 |                  47.1 |                   5.6 |                   41.2 |

   The row index must **never** be used as a feature. A random stratified split (required here) mixes periods, so test performance will be optimistic about the future. We record this as a limitation; it is not corrected in Part 1.

3. `poutcome`, `pdays` and `previous` describe *previous* campaigns, so they are known before the call. `campaign` includes the current (last) call, but the number of attempts so far is known when deciding on the next call, so it is kept. `contact`, `day` and `month` describe the last contact. They are plausible as scheduling attributes, but their link to the data-collection period (see `contact` above) should be kept in mind.

**Columns dropped and reasons.**

| Column | Reason |
|---|---|
| `duration` | Target leakage: only known during or after the call (UCI note and evidence above). |

No other columns are dropped. **Result: original 45,211 x 17 (7 numerical + 9 categorical + `y`); after the exclusion 45,211 x 16 (6 numerical + 9 categorical + `y`).** No rows are removed.

### 1C. Target

| class   |   count |   percent |
|:--------|--------:|----------:|
| no      |   39922 |     88.30 |
| yes     |    5289 |     11.70 |

The data is imbalanced, with a ratio of **7.55 : 1** (`no` : `yes`). Always predicting `no` would reach 88.30% accuracy. Part 2 should evaluate with metrics that are robust to imbalance (for example PR-AUC, recall/precision on `yes`, balanced accuracy) and keep stratification.

### 1D. Train/test split

- `sklearn.model_selection.train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`, with shuffling (the default).
- `X` holds the 15 retained predictors (`duration` excluded), and `y` is encoded as `yes = 1`, `no = 0`.
- The split is done **before any preprocessing is fitted**. No imputation, encoding or scaling is fitted in Part 1. Stages 2-3 below use the **training set only**, so nothing is learned from the test set.

| set   |   rows |   share of full % |   y = no |   y = yes |   y = yes % |
|:------|-------:|------------------:|---------:|----------:|------------:|
| full  |  45211 |            100.00 |    39922 |      5289 |       11.70 |
| train |  36168 |             80.00 |    31937 |      4231 |       11.70 |
| test  |   9043 |             20.00 |     7985 |      1058 |       11.70 |

- Train: 36,168 rows (31,937 `no`, 4,231 `yes`, 11.70% positive).
- Test: 9,043 rows (7,985 `no`, 1,058 `yes`, 11.70% positive).
- The exact rows are listed in `part1_outputs/train_test_split_indices.csv` (`row_id` = 0-based row position in `bank-full.csv`).

---

## 2. Univariate analysis (training set, n = 36,168)

### 2A. Numerical features

There are 6 retained numerical features: `age`, `balance`, `day`, `campaign`, `pdays`, `previous`. Because `pdays` and `previous` are dominated by the sentinel/zero group, the table also shows them for previously contacted clients only (n = 6,584, 18.20%).

| feature                              |    mean |   median |     std |   min |     Q1 |      Q3 |    max |   skew |   IQR outliers |   IQR outliers % |
|:-------------------------------------|--------:|---------:|--------:|------:|-------:|--------:|-------:|-------:|---------------:|-----------------:|
| age                                  |   40.89 |    39.00 |   10.63 |    18 |  33.00 |   48.00 |     95 |   0.69 |            394 |             1.09 |
| balance                              | 1365.49 |   451.00 | 3068.54 | -8019 |  74.00 | 1430.25 | 102127 |   8.34 |           3770 |            10.42 |
| day                                  |   15.82 |    16.00 |    8.33 |     1 |   8.00 |   21.00 |     31 |   0.09 |              0 |             0.00 |
| campaign                             |    2.76 |     2.00 |    3.10 |     1 |   1.00 |    3.00 |     63 |   4.94 |           2465 |             6.82 |
| pdays                                |   40.16 |    -1.00 |  100.16 |    -1 |  -1.00 |   -1.00 |    871 |   2.62 |           6584 |            18.20 |
| previous                             |    0.58 |     0.00 |    2.41 |     0 |   0.00 |    0.00 |    275 |  45.03 |           6584 |            18.20 |
| pdays (previously contacted only)    |  225.09 |   195.00 |  115.33 |     1 | 133.00 |  328.00 |    871 |   0.69 |             40 |             0.61 |
| previous (previously contacted only) |    3.20 |     2.00 |    4.85 |     1 |   1.00 |    4.00 |    275 |  28.55 |            367 |             5.57 |

Modes and skewness:

| feature                              |   mode |   mode share % |   skew |
|:-------------------------------------|-------:|---------------:|-------:|
| age                                  |     32 |           4.65 |   0.69 |
| balance                              |      0 |           7.77 |   8.34 |
| day                                  |     20 |           6.15 |   0.09 |
| campaign                             |      1 |          38.78 |   4.94 |
| pdays                                |     -1 |          81.80 |   2.62 |
| previous                             |      0 |          81.80 |  45.03 |
| pdays (previously contacted only)    |    182 |           1.94 |   0.69 |
| previous (previously contacted only) |      1 |          33.90 |  28.55 |

![Figure 1. Distributions of age, balance and day (training set)](part1_figures/fig01_hist_age_balance_day.png)

**Figure 1 - conclusion.**
- **`age`** is unimodal and mildly right-skewed (skew 0.69). The mean (40.89) is above the median (39.00), and the mode is 32. 95.50% of clients are 25-60. A thin tail above about 60 holds the IQR outliers (394, 1.09%), which are real older clients.
- **`balance`** is extremely right-skewed (skew 8.34). The mean of 1,365.49 is three times the median of 451.00, and there is a spike at exactly 0 (7.77% of clients). 8.30% are negative (min -8,019) and the max is 102,127. The plot is clipped to P1-P99 [-626, 13,161] for readability.
- **`day`** has no skew (0.09) and no outliers, but it is multimodal, with peaks on specific days (20 (6.1%), 18 (5.0%), 21 (4.5%)) and uneven counts across the month. This reflects call scheduling rather than a smooth quantity.

![Figure 2. Count-type features: campaign, pdays and previous (training set)](part1_figures/fig02_counts_campaign_pdays_previous.png)

**Figure 2 - conclusion.**
- **`campaign`** is a heavy-tailed count (skew 4.94). The mode is 1 contact (38.78% of clients), 78.69% received at most 3, and 945 clients received more than 10 (max 63).
- **`pdays`** applies only to previously contacted clients (18.20%). Its distribution is multimodal (median 195.00, IQR 133.00-328.00). The most frequent values are 182 (128), 92 (114), 91 (100), 183 (95), 181 (92) days, i.e. clients re-contacted about three or six months after the previous campaign. Figure 2 also shows a further cluster just under 400 days.
- **`previous`** among contacted clients has a median of 2.00 and a mode of 1, with a very long tail up to 275 (skew 28.55).

![Figure 3. Boxplots and 1.5 x IQR outliers of retained numerical features (training set)](part1_figures/fig03_boxplots_numerical.png)

**Figure 3 - conclusion.** Using the 1.5 x IQR rule:
- `balance` has the most real outliers (3,770, 10.42%), on both sides. `campaign` has 2,465 (6.82%) and `age` 394 (1.09%).
- For `pdays` and `previous`, the IQR collapses to the single sentinel value (Q1 = Q3 = -1.00 and 0.00). Every previously contacted client (6,584) is therefore flagged as an "outlier". This is an artifact of the sentinel, not evidence of bad data. Within contacted clients, there are only 40 outliers for `pdays` and 367 for `previous`.
- 4 features have |skew| > 1 (`balance`, `campaign`, `pdays`, `previous`). This motivates rank-based correlation in 3A, and transforms (log/Yeo-Johnson, or binning) and/or robust scaling in Part 2.

### 2B. Categorical features

There are 9 retained categorical features: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`. Together they have 44 levels (the one-hot width before dropping any reference level).

| feature   |   cardinality | most frequent       | least frequent    | levels < 1%   |   levels < 5% |   "unknown" % |
|:----------|--------------:|:--------------------|:------------------|:--------------|--------------:|--------------:|
| job       |            12 | blue-collar (21.6%) | unknown (0.65%)   | unknown       |             6 |          0.65 |
| marital   |             3 | married (60.2%)     | divorced (11.53%) | -             |             0 |          0.00 |
| education |             4 | secondary (51.3%)   | unknown (4.10%)   | -             |             1 |          4.10 |
| default   |             2 | no (98.2%)          | yes (1.79%)       | -             |             1 |          0.00 |
| housing   |             2 | yes (55.8%)         | no (44.20%)       | -             |             0 |          0.00 |
| loan      |             2 | no (83.9%)          | yes (16.07%)      | -             |             0 |          0.00 |
| contact   |             3 | cellular (64.9%)    | telephone (6.41%) | -             |             0 |         28.72 |
| month     |            12 | may (30.6%)         | dec (0.49%)       | dec           |             5 |          0.00 |
| poutcome  |             4 | unknown (81.8%)     | success (3.33%)   | -             |             2 |         81.81 |

Full frequency tables (training set):

**`job`** (cardinality 12)

| level         |   count |   percent |
|:--------------|--------:|----------:|
| blue-collar   |    7830 |     21.65 |
| management    |    7511 |     20.77 |
| technician    |    6068 |     16.78 |
| admin.        |    4141 |     11.45 |
| services      |    3348 |      9.26 |
| retired       |    1812 |      5.01 |
| self-employed |    1243 |      3.44 |
| entrepreneur  |    1186 |      3.28 |
| unemployed    |    1024 |      2.83 |
| housemaid     |    1013 |      2.80 |
| student       |     758 |      2.10 |
| unknown       |     234 |      0.65 |

**`marital`** (cardinality 3)

| level    |   count |   percent |
|:---------|--------:|----------:|
| married  |   21771 |     60.19 |
| single   |   10227 |     28.28 |
| divorced |    4170 |     11.53 |

**`education`** (cardinality 4)

| level     |   count |   percent |
|:----------|--------:|----------:|
| secondary |   18561 |     51.32 |
| tertiary  |   10594 |     29.29 |
| primary   |    5531 |     15.29 |
| unknown   |    1482 |      4.10 |

**`default`** (cardinality 2)

| level   |   count |   percent |
|:--------|--------:|----------:|
| no      |   35521 |     98.21 |
| yes     |     647 |      1.79 |

**`housing`** (cardinality 2)

| level   |   count |   percent |
|:--------|--------:|----------:|
| yes     |   20182 |     55.80 |
| no      |   15986 |     44.20 |

**`loan`** (cardinality 2)

| level   |   count |   percent |
|:--------|--------:|----------:|
| no      |   30355 |     83.93 |
| yes     |    5813 |     16.07 |

**`contact`** (cardinality 3)

| level     |   count |   percent |
|:----------|--------:|----------:|
| cellular  |   23465 |     64.88 |
| unknown   |   10386 |     28.72 |
| telephone |    2317 |      6.41 |

**`month`** (cardinality 12)

| level   |   count |   percent |
|:--------|--------:|----------:|
| jan     |    1119 |      3.09 |
| feb     |    2078 |      5.75 |
| mar     |     391 |      1.08 |
| apr     |    2394 |      6.62 |
| may     |   11062 |     30.59 |
| jun     |    4237 |     11.71 |
| jul     |    5537 |     15.31 |
| aug     |    4986 |     13.79 |
| sep     |     451 |      1.25 |
| oct     |     592 |      1.64 |
| nov     |    3143 |      8.69 |
| dec     |     178 |      0.49 |

**`poutcome`** (cardinality 4)

| level   |   count |   percent |
|:--------|--------:|----------:|
| unknown |   29589 |     81.81 |
| failure |    3889 |     10.75 |
| other   |    1485 |      4.11 |
| success |    1205 |      3.33 |

![Figure 4. Level frequencies of the retained categorical features (training set)](part1_figures/fig04_categorical_frequencies.png)

**Figure 4 - conclusion.**
- `job` is dominated by blue-collar (21.6%), management (20.8%) and technician (16.8%) clients.
- `marital` is mostly married (60.2%), and `education` mostly secondary (51.3%).
- `default` is almost constant (98.2% `no`).
- `month` is concentrated in May-August (71.4% of contacts; May alone is 30.6%). March, September, October and December are thin (1.1%, 1.2%, 1.6%, 0.5%).
- `poutcome` is 81.8% `"unknown"`, which is the "never contacted" group.

![Figure 5. Cardinality and share of "unknown" per categorical feature (training set)](part1_figures/fig05_cardinality_unknown.png)

**Figure 5 - conclusion.**
- **No feature has high cardinality.** The maximum is 12 levels (`job`, `month`), so one-hot encoding is feasible.
- **Rare levels (< 1% of training rows):** `job`: unknown; `month`: dec. March, September and October are also small (just over 1%). Their subscription-rate estimates are noisier (see the wide CIs in Figure 9), so Part 2 may consider grouping them if a model overfits.
- **`"unknown"` is substantial** in `poutcome` and `contact`, small in `education`, and marginal in `job`. It is kept as a category (1B).

---

## 3. Bivariate and multivariate analysis (training set)

### 3A. Numerical x numerical

**Choice: Spearman.** 4 of the 6 numerical features are strongly skewed (`balance`, `campaign`, `pdays`, `previous`), and `pdays`/`previous` are dominated by a sentinel/zero mass. Pearson assumes linear relationships and is sensitive to extreme values, while Spearman uses ranks: it captures monotone relationships and is robust to outliers. Pearson is shown only for comparison.

![Figure 6. Correlation among retained numerical features (training set)](part1_figures/fig06_corr_spearman_pearson.png)

Top pairs by |Spearman|:

| pair                |   spearman |   pearson |
|:--------------------|-----------:|----------:|
| pdays - previous    |      0.986 |     0.439 |
| day - campaign      |      0.139 |     0.162 |
| campaign - pdays    |     -0.112 |    -0.089 |
| campaign - previous |     -0.108 |    -0.031 |
| age - balance       |      0.096 |     0.096 |
| day - pdays         |     -0.093 |    -0.093 |

**Figure 6 - conclusion.**
- The only strongly correlated pair is **`pdays - previous`**: Spearman 0.986, compared with Pearson 0.439. Pearson understates it because the relationship is monotone but far from linear (a large sentinel mass plus long tails).
- All other pairs have |rho| <= 0.139. The next strongest are `day - campaign` (0.139) and `campaign - pdays` (-0.112). Apart from the pair above, the numerical features are close to non-redundant.

![Figure 7. pdays versus previous once the -1 sentinel is removed (training set)](part1_figures/fig07_pdays_vs_previous.png)

**Figure 7 - conclusion.** The 0.986 correlation is driven almost entirely by the shared sentinel, since `pdays = -1` exactly when `previous = 0` (1B). Among previously contacted clients, Spearman is only -0.099 (Pearson -0.016), so the actual magnitudes are nearly unrelated. **Redundancy:** the two features duplicate the binary "previously contacted" information, not their values. Part 2 should encode that flag once and keep both magnitudes (or test dropping one) instead of treating them as collinear duplicates.

### 3B. Categorical x target

The training base rate is 11.70%. Rates for every level of every categorical feature, with 95% Wilson intervals and lift, are in `part1_outputs/categorical_target_rates_train.csv`. The spread between each feature's highest and lowest level rate:

| feature   |   max - min rate (pp) |
|:----------|----------------------:|
| poutcome  |                  55.4 |
| month     |                  44.2 |
| job       |                  21.6 |
| contact   |                  10.7 |
| housing   |                   9.0 |
| education |                   6.3 |
| default   |                   5.9 |
| loan      |                   5.8 |
| marital   |                   5.0 |

The features with the largest spread are `poutcome` (55.4 pp), `month` (44.2 pp), `job` (21.6 pp). The smallest is `marital` (5.0 pp). The selected figures cover the strongest and most decision-relevant features.

![Figure 8. Subscription rate by poutcome, contact, housing and loan, with 95% Wilson CIs (training set)](part1_figures/fig08_rate_poutcome_contact_housing_loan.png)

**Figure 8 - conclusion.**
- **`poutcome`** separates the classes most. Clients whose previous campaign was a `success` subscribe at 64.6%, against 12.3% (`failure`), 16.5% (`other`) and 9.2% (`unknown` = never contacted).
- **`contact = unknown`** has a much lower rate (4.1%) than `cellular` (14.8%) and `telephone` (13.9%). This matches the early-period artifact found in 1B, so it partly encodes time.
- Clients **without a housing loan** (16.7% vs 7.7%) or **without a personal loan** (12.6% vs 6.8%) subscribe more often.
- The intervals for `poutcome = success`, `contact = unknown`, `housing` and `loan` do not overlap with their contrasts. `cellular` and `telephone` overlap, so the contact channel itself adds little once `unknown` is separated out.

![Figure 9. Contact volume and subscription rate by month (training set)](part1_figures/fig09_month_volume_rate.png)

**Figure 9 - conclusion.** Contact volume and subscription rate move in opposite directions (Spearman correlation across the 12 months: -0.87).
- The busiest month has the lowest rate: may (6.7%, n = 11,062).
- Low-volume months have rates above 30%: mar (50.9%, n = 391), sep (45.2%, n = 451), oct (43.8%, n = 592), dec (47.2%, n = 178). The maximum is mar (50.9%, n = 391).
- Volume and rate are drawn as two panels on separate axes, not as a dual-axis chart.
- `month` is strongly predictive, but given the drift found in 1B it likely mixes seasonality with the period of data collection (there is no year column). Its confidence intervals are wide for the thin months.

![Figure 10. Subscription rate by job and education, with 95% Wilson CIs (training set)](part1_figures/fig10_rate_job_education.png)

**Figure 10 - conclusion.**
- **`job`:** students (28.9%) and retired clients (23.0%) subscribe far above the base rate. Blue-collar clients (7.3%), housemaids (8.4%), entrepreneurs (8.5%) and services workers (8.9%) subscribe below it.
- **`education`:** the rate increases from primary (8.6%) to secondary (10.7%) to tertiary (14.9%). `unknown` (13.2%) has a wide interval, consistent with its small size.
- These features carry moderate, roughly ordinal signal. `job = unknown` (12.0%) has a very wide interval and is too small to interpret.

### 3C. Numerical x categorical (grouped boxplots, split by `y`)

Each pair below was chosen because 2A/3B suggested a possible interaction, i.e. that the numerical feature's link to `y` might depend on the group. Every boxplot is also split by `y`. Medians and IQRs for every box are in `part1_outputs/grouped_box_stats_train.csv`.

![Figure 11. Age by job, split by subscription outcome (training set)](part1_figures/fig11_box_age_job_y.png)

| job           | y   |   rows |   Q1 |   median |   Q3 |   IQR |
|:--------------|:----|-------:|-----:|---------:|-----:|------:|
| admin.        | no  |   3632 | 32.0 |     37.0 | 46.0 |  14.0 |
| admin.        | yes |    509 | 31.0 |     37.0 | 46.0 |  15.0 |
| blue-collar   | no  |   7259 | 33.0 |     39.0 | 47.0 |  14.0 |
| blue-collar   | yes |    571 | 31.0 |     38.0 | 46.0 |  15.0 |
| entrepreneur  | no  |   1085 | 35.0 |     41.0 | 49.0 |  14.0 |
| entrepreneur  | yes |    101 | 35.0 |     41.0 | 50.0 |  15.0 |
| housemaid     | no  |    928 | 38.0 |     46.0 | 54.0 |  16.0 |
| housemaid     | yes |     85 | 38.0 |     48.0 | 58.0 |  20.0 |
| management    | no  |   6470 | 33.0 |     39.0 | 48.0 |  15.0 |
| management    | yes |   1041 | 32.0 |     37.0 | 47.0 |  15.0 |
| retired       | no  |   1396 | 56.0 |     59.0 | 63.0 |   7.0 |
| retired       | yes |    416 | 60.0 |     67.0 | 74.0 |  14.0 |
| self-employed | no  |   1101 | 33.0 |     39.0 | 48.0 |  15.0 |
| self-employed | yes |    142 | 30.2 |     35.0 | 44.8 |  14.5 |
| services      | no  |   3050 | 32.0 |     37.0 | 46.0 |  14.0 |
| services      | yes |    298 | 30.0 |     36.0 | 43.0 |  13.0 |
| student       | no  |    539 | 24.0 |     26.0 | 30.0 |   6.0 |
| student       | yes |    219 | 22.0 |     25.0 | 28.0 |   6.0 |
| technician    | no  |   5405 | 32.0 |     38.0 | 45.0 |  13.0 |
| technician    | yes |    663 | 32.0 |     37.0 | 46.0 |  14.0 |
| unemployed    | no  |    866 | 33.0 |     40.0 | 48.0 |  15.0 |
| unemployed    | yes |    158 | 31.0 |     39.0 | 48.0 |  17.0 |
| unknown       | no  |    206 | 42.0 |     47.0 | 55.0 |  13.0 |
| unknown       | yes |     28 | 36.2 |     47.5 | 54.0 |  17.8 |

**Figure 11 - conclusion.** *Why this pair:* `age` and `job` are strongly linked (students and retirees are at the age extremes), and both have high-rate groups in 3B.
- **Location:** median age ranges from 26 (students, `no`) to 59 (retired, `no`).
- **Within groups:** for most jobs, the `yes` and `no` boxes are within a few years of each other. The clear exception is **retired** clients: those who subscribed are older (median 67 vs 59) and more spread out (IQR 14 vs 7).
- **Overall:** the age effect is non-linear. Clients aged 60+ subscribe at 33.7% (n = 1,447) and those under 25 at 25.6% (n = 657), against 9.8% for ages 30-59. The overall medians by class are almost equal (38 vs 39), which is why a linear age effect would be weak. Part 2 should consider age bins or non-linear models.

![Figure 12. Balance by education, split by subscription outcome (training set)](part1_figures/fig12_box_balance_education_y.png)

| education   | y   |   rows |    Q1 |   median |     Q3 |    IQR |
|:------------|:----|-------:|------:|---------:|-------:|-------:|
| primary     | no  |   5054 |  56.0 |    390.5 | 1334.2 | 1278.2 |
| primary     | yes |    477 | 217.0 |    796.0 | 2580.0 | 2363.0 |
| secondary   | no  |  16580 |  46.0 |    373.0 | 1177.0 | 1131.0 |
| secondary   | yes |   1981 | 173.0 |    651.0 | 1757.0 | 1584.0 |
| tertiary    | no  |   9017 |  89.0 |    538.0 | 1666.0 | 1577.0 |
| tertiary    | yes |   1577 | 242.0 |    882.0 | 2559.0 | 2317.0 |
| unknown     | no  |   1286 |  79.0 |    534.5 | 1499.8 | 1420.8 |
| unknown     | yes |    196 | 390.8 |   1007.0 | 2667.8 | 2277.0 |

**Figure 12 - conclusion.** *Why this pair:* balance is a likely proxy for the ability to make a deposit, and education is a socioeconomic grouping with an ordinal rate gradient in 3B. The figure checks whether balance still matters within an education level.
- **Location:** within every education level, subscribers have a higher median balance:
  - primary: 796 vs 390.5
  - secondary: 651 vs 373
  - tertiary: 882 vs 538
- **Spread:** subscribers also have a larger IQR (for example tertiary 2,317 vs 1,577).
- The balance effect is therefore not explained by education alone. Long tails on both sides (a symlog axis is needed) confirm that `balance` needs a robust transform.

![Figure 13. Campaign contacts by contact type, split by subscription outcome (training set)](part1_figures/fig13_box_campaign_contact_y.png)

| contact   | y   |   rows |   Q1 |   median |   Q3 |   IQR |
|:----------|:----|-------:|-----:|---------:|-----:|------:|
| cellular  | no  |  19984 |  1.0 |      2.0 |  3.0 |   2.0 |
| cellular  | yes |   3481 |  1.0 |      2.0 |  2.0 |   1.0 |
| telephone | no  |   1994 |  1.0 |      2.0 |  4.0 |   3.0 |
| telephone | yes |    323 |  1.0 |      2.0 |  3.0 |   2.0 |
| unknown   | no  |   9959 |  1.0 |      2.0 |  3.0 |   2.0 |
| unknown   | yes |    427 |  1.0 |      2.0 |  3.0 |   2.0 |

**Figure 13 - conclusion.** *Why this pair:* `campaign` measures call effort, and `contact` is the channel used. This checks whether extra calls relate to subscription in the same way for each channel, and whether the `unknown` (early-period) group behaves differently.
- **Location:** medians are equal (2) in every box, so location barely differs.
- **Spread:** the difference is in the upper tail. For `cellular` and `telephone`, subscribers have a narrower IQR (cellular 1 vs 2; telephone 2 vs 3), while for `unknown` they are equal (2 vs 2). The extreme counts belong mostly to non-subscribers.
- **Overall:** the rate is 14.6% after a single contact and 5.9% after more than 5 contacts (n = 3,483). Repeated calls have diminishing returns. The signal is in the tail, so a capped or log-transformed `campaign` is preferable to the raw count.

---

## Handoff for Part 2 (preprocessing, stages 4-5)

**Split - use exactly this.**
- Function: `train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`, with `X = df[numerical + categorical]` in the column order of `part1_outputs/feature_lists.json`, and `y = (df["y"] == "yes").astype(int)`.
- Input: `data/raw/bank-full.csv` (SHA-256 `d1513ec63b385506f7cfce9f2c5caa9fe99e7ba4e8c3fa264b3aaf0f849ed32d`), read with `pd.read_csv(path, sep=";")`.
- Result: train 36,168 rows (4,231 positive), test 9,043 rows (1,058 positive).
- Row membership: `part1_outputs/train_test_split_indices.csv`. Specification: `part1_outputs/train_test_split.json`.

**Retained features (`duration` excluded).**
- Numerical (6): `age`, `balance`, `day`, `campaign`, `pdays`, `previous`
- Categorical (9): `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
- Target: `y` (`yes` = 1, `no` = 0). The excluded column is `duration` (leakage).

**Findings that justify preprocessing choices** (fit everything on the training set only, inside a pipeline):
1. **No NaNs.** `"unknown"` is a legitimate category in `job`, `education`, `contact` and `poutcome`. Do not impute it, and keep it in one-hot encoding.
2. **`pdays = -1`** is a sentinel identical to `previous = 0` (and to `poutcome = unknown`, except for 5 rows). Create a "previously contacted" flag, and neutralise or mask `pdays` for never-contacted clients rather than scaling `-1` as a number.
3. **Skew and outliers:** `balance` (skew 8.34, 10.42% IQR outliers, negative values present, so use Yeo-Johnson or a signed log rather than a plain log), `campaign` (skew 4.94) and `previous` (skew 45.03). No rows were removed. Use robust scaling and/or transforms or caps fitted on train.
4. **Cardinality is low** (max 12, 44 levels in total), so one-hot encoding is fine. Rare levels: `job`: unknown; `month`: dec. Use `handle_unknown="ignore"`.
5. **`default`** is near-constant (1.80% `yes`). Keep it or test dropping it.
6. **Imbalance** 7.55 : 1. Keep stratification in any CV, and use imbalance-aware metrics and/or class weights.
7. **Time drift:** the file is chronological, and the rate goes from 3.0% to 47.1% across deciles. `month` and `contact = unknown` partly encode the period. Never use the row index as a feature, and state the optimism of a random split as a limitation.
8. **Duplicates:** 16 feature-level duplicates after dropping `duration`, with no conflicting labels. They were kept.

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
| 1 | `fig01_hist_age_balance_day.png` | Distributions of age, balance and day (training set) |
| 2 | `fig02_counts_campaign_pdays_previous.png` | Count-type features: campaign, pdays and previous (training set) |
| 3 | `fig03_boxplots_numerical.png` | Boxplots and 1.5 x IQR outliers of retained numerical features (training set) |
| 4 | `fig04_categorical_frequencies.png` | Level frequencies of the retained categorical features (training set) |
| 5 | `fig05_cardinality_unknown.png` | Cardinality and share of "unknown" per categorical feature (training set) |
| 6 | `fig06_corr_spearman_pearson.png` | Correlation among retained numerical features (training set) |
| 7 | `fig07_pdays_vs_previous.png` | pdays versus previous once the -1 sentinel is removed (training set) |
| 8 | `fig08_rate_poutcome_contact_housing_loan.png` | Subscription rate by poutcome, contact, housing and loan, with 95% Wilson CIs (training set) |
| 9 | `fig09_month_volume_rate.png` | Contact volume and subscription rate by month (training set) |
| 10 | `fig10_rate_job_education.png` | Subscription rate by job and education, with 95% Wilson CIs (training set) |
| 11 | `fig11_box_age_job_y.png` | Age by job, split by subscription outcome (training set) |
| 12 | `fig12_box_balance_education_y.png` | Balance by education, split by subscription outcome (training set) |
| 13 | `fig13_box_campaign_contact_y.png` | Campaign contacts by contact type, split by subscription outcome (training set) |

**Reproduce:** `python scripts/part1_download_data.py && python scripts/part1_eda.py` (Python 3.9, pandas, numpy, matplotlib, seaborn, scikit-learn, tabulate). No model is trained in Part 1.
