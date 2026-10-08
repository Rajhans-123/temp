# Anime Recommendations - Data Warehousing & Mining

An end-to-end data warehousing and mining project over the Kaggle dataset
**[Anime Recommendations Database](https://www.kaggle.com/datasets/CooperUnion/anime-recommendations-database)**
(CooperUnion, scraped from MyAnimeList): 12,294 anime, 73,515 users and
7,813,737 interactions.

The repository contains a reproducible mining pipeline, a deployable Streamlit
UI that walks through every stage, and a trained prediction/recommendation
system. Every table the pipeline produces is a CSV.

---

## 1. Quick start

```bash
python -m pip install -r requirements.txt

python build_dataset.py      # ~13 s   clean, engineer, aggregate, split  -> data/processed/*.csv
python train.py              # ~5 min  all six mining stages            -> reports/*.csv, models/*.joblib
python smoke_test.py         # ~2 min  headless check that the UI runs

streamlit run app.py         # then open http://localhost:8501

python report/make_report.py # rebuild the 7-page assignment report       -> report/dwm_report.html
```

`report/make_report.py` regenerates the written report from the CSV reports, so the
document cannot drift from the measured results. `report/dwm_report.html` prints to
exactly seven A4 pages (`Ctrl+P`, margins: default); `report/dwm_report.pdf` is a
pre-rendered copy. Screenshot slots are marked dashed frames - paste the captured
UI view over each one.

`train.py` also accepts stage numbers, and re-running one stage keeps the
metrics of the others in the summary:

```bash
python train.py 3 4          # only classification and regression
```

| stage | what it does | key artefact |
| --- | --- | --- |
| 1 preprocessing | profile, clean, engineer, discretise, split | `data/processed/*.csv`, `reports/preprocessing_report.csv` |
| 2 association | Apriori + FP-Growth frequent itemsets, rule metrics | `reports/association_rules_*.csv` |
| 3 classification | J48 (own C4.5), Naive Bayes, 5-fold model comparison | `reports/j48_rules.csv`, `reports/classification_comparison.csv` |
| 4 regression | six regressors for the continuous MAL score | `reports/regression_comparison.csv` |
| 5 clustering | K-Means + Ward hierarchical on two datasets | `reports/cluster_*_profile.csv` |
| 6 prediction | blended score regressor + hybrid recommender | `models/*.joblib` |

---

## 2. Data

### Source

The raw files live in `data/raw/`:

| file | rows | columns |
| --- | --- | --- |
| `anime.csv` | 12,294 | `anime_id, name, genre, type, episodes, rating, members` |
| `ratings.csv` | 7,813,737 | `user_id, anime_id, rating` |

`ratings.csv` was taken from the public GitHub mirror of the Kaggle dataset
(`vineethvs23/Anime-Recommendation`) because no Kaggle API credentials were
available in this environment; it is the same data, and
`data/raw/anime.csv` is byte-identical in schema. To use Kaggle directly:

```bash
# Kaggle CLI: writes anime.csv and rating.csv into data/raw/
kaggle datasets download -d cooperunion/anime-recommendations-database -p data/raw --unzip
# the project expects the ratings table as ratings.csv
Rename-Item data\raw\rating.csv ratings.csv
```

### Data quality (measured, see `reports/preprocessing_report.csv`)

* `rating = -1` means *watched but not scored*. It is **excluded**, not treated
  as a rating: 1,476,496 such rows, leaving 6,337,241 scored interactions and
  69,600 users with a usable rating history.
* `episodes` is a string column containing `"Unknown"`; it is coerced to numeric
  and imputed with the **media-type median**.
* Missing `genre`/`type`/`rating` are imputed (genre/type to `"Unknown"`, rating
  to the type-wise mean).
* Duplicate `anime_id` values are dropped.
* `members` and `episodes` are heavily right-skewed, so `log1p` transforms are
  added; outliers are **flagged** (`is_outlier`) by the 1.5 x IQR fence rather
  than deleted, and titles below the member threshold are excluded from the
  modelling frames.

### Derived tables (`data/processed/`)

| file | rows | purpose |
| --- | --- | --- |
| `anime_clean.csv` | 12,294 | cleaned + feature-engineered master table |
| `anime_rating_stats.csv` | ~10,600 | per-anime mean/std/max/min of user ratings |
| `user_features.csv` | 69,600 | per-user behaviour profile - **clustering dataset B** |
| `baskets_genre.csv` | 12,294 | genre basket (transaction = anime, item = genre) |
| `baskets_user.csv` | 6,000 users | long-format basket of liked titles - association + recommender |
| `classification_split.csv` / `classification_test.csv` | 9,513 / 2,379 | stratified 80/20, target `rating_tier` |
| `regression_split.csv` / `regression_test.csv` | 9,513 / 2,379 | 80/20, target `rating` (continuous) |

Two tables feed clustering, as required: the **anime** table (content features)
and the **user** table (behaviour features), both standardised with z-scores.

---

## 3. The mining stages

### 3.1 Association rule mining

Two rule bases, both mined with **Apriori and FP-Growth** and compared on
runtime (`reports/apriori_vs_fpgrowth.csv`):

* **Genre basket** - 18 rules at support >= 0.04, the strongest being
  `Mecha => Action` (support 0.047, confidence 0.618, lift 2.67) and
  `Fantasy => Adventure` (support 0.077, confidence 0.412, lift 2.16).
* **User basket** - 33 rules over the 1,500 most popular titles each user rated
  >= 8; these are the "watched this, then that" rules the recommender uses, and
  the strongest are sequel pairs such as
  `Neon Genesis Evangelion => The End of Evangelion` (lift 7.96) and
  `Dragon Ball => Dragon Ball Z` (lift 7.04).

Rules carry support, confidence, lift, conviction, leverage, Zhang's metric and
Kulczynski's number, and the UI exposes support/confidence/lift thresholds so
the ruleset can be re-cut interactively.

### 3.2 Classification - J48 and Naive Bayes

`core/decision_tree.py` is a **from-scratch C4.5** (no library tree learner):
numeric pre-discretisation into 10 quantile intervals, **gain ratio** splitting,
confidence-factor bottom-up **pruning**, and IF/THEN rule export.

The target is `rating` binned into four classes at the 30/30/20/20 quantiles
(`Low`, `Medium`, `High`, `Top`). Held-out results (2,379 test rows) from
`reports/classification_comparison.csv`:

| model | CV accuracy | test accuracy | macro F1 | ROC-AUC (OvR) |
| --- | --- | --- | --- | --- |
| Random Forest | 0.5812 | 0.5864 | 0.5649 | 0.8287 |
| Decision Tree (sklearn, entropy) | 0.5572 | 0.5628 | 0.5505 | 0.8034 |
| J48 / C4.5 (ours) | 0.5435 | 0.5595 | 0.5291 | - |
| Naive Bayes (Gaussian) | 0.4710 | 0.4758 | 0.4396 | 0.7403 |
| Majority baseline | 0.2999 | 0.2997 | 0.1153 | 0.5000 |

Every row above is fed the *same* ordinal encoding, fitted on the training frame and
reused for the test frame (`_as_int_frame(Xtr, mapping)` -> `_as_int_frame(Xte, mapping)`).
Encoding the two frames independently - the earlier bug - let a category carry
different integer codes in train and test and cost Gaussian NB 3 points of accuracy,
the sklearn tree 2 and the forest 2.6.

J48 yields a 5-level tree with 27 rules; the gain-ratio ranking is led by
`is_shoujo`, `members_band`, `is_shounen`, `members` and `episodes_band`, which
is why the J48 predictions are close to but slightly below the forest - the
quantile discretization of the numeric attributes throws away some resolution
that the forest recovers on the raw values.

Stage 3b additionally fits a **Categorical NB** on the one-hot encoded
categorical attributes (it has no place in the table above because it uses a
different feature representation): test accuracy 0.5443, macro F1 0.5112 - the
strongest of the two Naive Bayes variants. Its standalone Gaussian NB run, which
scales the numeric attributes instead of ordinal-encoding them, scores
0.4758 / 0.4396, matching the row in the table.

### 3.3 Regression - continuous estimation

Predicting the numeric MAL score from the same content features, with mean /
linear / ridge / tree / random forest / gradient boosting, evaluated by MAE,
RMSE, R2 and MAPE plus a learning curve and an error breakdown by media type:

| model | CV RMSE | MAE | RMSE | R2 | MAPE % |
| --- | --- | --- | --- | --- | --- |
| Gradient Boosting | 0.6393 | 0.4833 | 0.6545 | 0.5956 | 8.164 |
| Random Forest | 0.6487 | 0.4937 | 0.6690 | 0.5774 | 8.309 |
| Decision Tree | 0.6831 | 0.5168 | 0.6971 | 0.5412 | 8.730 |
| Linear Regression | 0.7016 | 0.5508 | 0.7354 | 0.4894 | 9.405 |
| Ridge (alpha=1) | 0.7016 | 0.5508 | 0.7354 | 0.4894 | 9.405 |
| Mean baseline | 0.9897 | 0.7973 | 1.0292 | -0.0001 | 13.655 |

The blend of the two tree ensembles in stage 6 improves on both base learners
(held-out RMSE 0.6531, R2 0.5973, weight 0.42 RF + 0.58 GB).

### 3.4 Clustering - two datasets, two algorithms

`k` is chosen by the **elbow (max distance to the chord)** rule and cross-checked
against the silhouette maximum; Calinski-Harabasz and Davies-Bouldin are reported
alongside.

| dataset | rows | features | k | K-Means silhouette | Ward silhouette |
| --- | --- | --- | --- | --- | --- |
| A - Anime (content) | 12,294 | 6 | 4 | 0.3408 | 0.3097 |
| B - Users (behaviour) | 69,600 | 8 | 5 | 0.2764 | 0.2200 |

Hierarchical clustering uses Ward linkage via `scipy.cluster.hierarchy.linkage`
on a fixed 1,200-row subsample, because the exact algorithm needs an n x n
distance matrix (38 GB for dataset B). The UI renders the dendrogram and cuts it
at k, so every k from 2 to 8 can be read off one run. Clusters are named from the
**rank** of their centroids (e.g. *blockbuster / acclaimed / long-form*,
*casual rater / harsh critic / erratic*), which keeps the labels correct
whatever the column distributions, and a **cluster x class cube** provides the
OLAP-style roll-up over the discretised dimension.

### 3.5 Prediction system

* **Score predictor** - Random Forest (200 trees, depth 12) + Gradient Boosting
  (400 stages). The blend weight is fitted on a validation split carved out of
  the training data (never on the test set) and both learners are then refitted
  on the full training split. The UI reports a 95% interval of
  `score +/- 1.96 x held-out RMSE`. An `engagement_index` feature was removed
  because it leaked the target. Held-out RMSE 0.6531 / R2 0.5973.
* **Hybrid recommender** - 6,941 candidate titles, 47,153 users, 5 taste
  segments. The score blends the viewer's segment genre affinity, the segment's
  media-type preference, the community prior, how far a title is above the
  average for its own type, a mild popularity preference, and lift from the
  association rules that fire on what the user already watched. It can be driven
  by an existing user, by genre preferences, or by a watch list.

---

## 4. The Streamlit app

`app.py` has one tab per stage, and every number on screen is computed live from
`core/` (results are cached, so switching tabs does not re-mine):

1. **Overview** - headline KPIs, raw data preview, artefact map, and the results
   of the last pipeline run.
2. **Preprocessing** - missing values, IQR outliers, derived features,
   normalisation, discretisation, and the resulting splits.
3. **Association Rules** - interactive support/confidence/lift thresholds, a
   support-vs-confidence scatter sized by lift, and the Apriori/FP-Growth
   benchmark.
4. **Classification** - J48 with sliders for depth, leaf size, pruning confidence
   and rule confidence; the rendered tree, the IF/THEN rules, gain-ratio ranking,
   confusion matrices, Naive Bayes densities and the head-to-head comparison.
5. **Regression** - model table, actual-vs-predicted, residuals, learning curve,
   error by media type, importance, and the regressed tree.
6. **Clustering** - dataset switch, k diagnostics, K-Means and Ward results, a
   dendrogram, and the cluster x class cube.
7. **Prediction System** - the score form, its interval and feature importance,
   plus the three recommender modes and a retrain button.

---

## 5. Deployment

The app is a plain Streamlit app with no database, so it deploys to Streamlit
Community Cloud as-is. Before deploying:

1. `python -m pip install -r requirements.txt` and `python train.py` so that
   `models/` and `reports/` exist.
2. Keep the repository under ~100 MB: `data/raw/ratings.csv` (1 GB) is excluded
   by `.gitignore` and is **not** read by the app - only `data/processed/`,
   `reports/` and `models/` are. A bundle of about 50 MB is expected
   (21 MB predictor, 7 MB recommender, 18 MB processed CSV).
3. Deploy with `app.py` as the entry point, `.streamlit/config.toml` as
   configuration, and `requirements.txt` as the dependency file.

Local run without the raw files present is supported: the app falls back to the
processed tables.

The app's **8 - Live Feed** tab is the UI demo for this feed: sync status KPIs
(last sync, snapshot count, last weekly run), the per-title score/member trend
from `data/live/score_history.csv`, and a single-title fetch-and-sync demo that
diffs the live Tenrai row and writes `anime.csv` plus the score history.

---

## 6. Layout

```
app.py                 Streamlit UI (7 tabs)
build_dataset.py       stage 1 driver
train.py               stages 2-6 driver (+ smoke test of the predictor)
smoke_test.py          headless AppTest check of the UI
core/
  config.py            paths, thresholds, tiers, band edges
  preprocessing.py     cleaning, features, aggregation, baskets, splits
  association.py       Apriori / FP-Growth / rule metrics / benchmark
  decision_tree.py     from-scratch C4.5 (gain ratio + confidence pruning)
  classification.py    J48, Naive Bayes, model comparison, gain-ratio ranking
  regression.py        six regressors, diagnostics, importance
  clustering.py        k selection, K-Means, Ward, profiles, cubes
  predictor.py         score regressor + hybrid recommender
  reporting.py         long-format CSV summaries
report/
  make_report.py       builds the 7-page A4 report from the CSV reports
  diagrams.py          SVG flowcharts used by the report
  dwm_report.html      the report (prints to 7 A4 pages)
  dwm_report.pdf       pre-rendered copy
data/raw/              source CSV
data/processed/        generated datasets
reports/               generated result tables (all CSV)
models/                trained joblib artefacts
```

## 7. Reproducibility notes
  seed, and the raw ratings file is streamed in 500k-row chunks so peak memory
  stays low.
* Every reported accuracy is on data the model has not seen. Where a metric is
  in-sample it is labelled as such in the UI.
* The sklearn learners in the stage-3 comparison share one ordinal encoding fitted
  on the training frame; encoding train and test separately is a real bug that
  silently costs 2-3 accuracy points, so it is covered by the assertion-style
  single-mapping call rather than by a comment.
* `smoke_test.py` executes the entire UI headlessly and drives the interactive
  paths, so a broken widget fails before deployment rather than in the browser.

## 8. Live feed (Tenrai) - optional, additive

`sync_jikan.py` refreshes `data/raw/anime.csv` from the free Tenrai v1 API
(Jikan-compatible, no key; `anime_id` IS the MyAnimeList id, so rows join 1:1).
Tenrai is used because the public Jikan API was discontinued on 2026-10-01;
it serves the same response shape, so the client and cache work unchanged.
The batch pipeline is untouched - run sync first, then the usual build + train:

```bash
python sync_jikan.py --self-test     # offline check, no network
python sync_jikan.py --dry-run --top 5
python sync_jikan.py --top 200       # refresh 200 most-membered titles
python sync_jikan.py --stale 30      # anything not refreshed in 30 days
python build_dataset.py && python train.py
```

Rules that protect the DWM logic: local `name` is never overwritten (it is
the join key for baskets and rules); only sane API values overwrite
(`episodes > 0`, `0 < rating <= 10`, known `type`, non-empty `genre`);
unknown ids are skipped unless `--allow-new`; every run writes
`data/live/score_history.csv` (append-only snapshots for retraining) and
`data/live/last_sync.csv`, keeps one `.bak` of anime.csv, and respects the
3 req/s API limit with retries + a JSON cache under `data/live/cache/`.

### Weekly retrain

`retrain_weekly.py` chains the three steps (sync -> rebuild -> retrain) and
appends one row per run to `reports/retrain_log.csv`:

```bash
python retrain_weekly.py                    # weekly default: titles stale 7d, stages 2-6
python retrain_weekly.py --top 200          # refresh 200 most-membered instead
python retrain_weekly.py --stages 6         # only the predictor stage
python retrain_weekly.py --skip-sync        # retrain on current CSVs, no network
python retrain_weekly.py --dry-run          # select ids only, no I/O
python retrain_weekly.py --force            # retrain even if 0 titles changed
```

Quiet weeks cost nothing: if fewer than `--min-changed` titles changed, the
rebuild is skipped (override with `--force`). Without the 1 GB `ratings.csv`
(e.g. CI) only anime-derived tables are rebuilt and each train stage runs in
isolation, so one stage failing never stops the rest.

Schedule it:

* **GitHub Actions** - `.github/workflows/weekly-retrain.yml` runs every
  Monday 03:00 UTC (plus manual dispatch with the same knobs) and commits
  back `reports/retrain_log.csv`, `reports/mining_summary.csv` and the
  `data/live/` state.
* **Windows** - `weekly_task.ps1` registers the same job in Task Scheduler:

```powershell
powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action install
powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action run  # run once now
```
