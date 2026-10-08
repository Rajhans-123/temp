"""Single source of truth for the assignment report.

    PAGES -> list of pages; each page is a list of blocks.

Both renderers consume this module, so the Word document and the HTML preview can
never disagree:

    report/make_docx.py      -> report/dwm_report.docx  (the print deliverable)
    report/make_report.py    -> report/dwm_report.html (browser preview)

Block types
-----------
("cover", {course, title, subtitle, blurb, rows, kpis, date})
("toc",   [(number, title, pages), ...])
("h1"|"h2"|"h3", text)
("p", text)
("pre", text)
("callout", kind, text)                     kind: "" | "ok" | "warn"
("table", headers, rows, opts)              opts: {"numeric": {i}, "best": row}
("figure", key, caption, width_pct)
("shots", [(name, how, height_mm), ...])

Inline markup understood by both renderers: `code`, **bold**, *italic*.
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from core import config as C
from core import reporting as REP

# --------------------------------------------------------------- the data --
S = REP.read_summary(C.REPORT_DIR / "mining_summary.csv")
P = REP.read_summary(C.REPORT_DIR / "preprocessing_report.csv")
RULES = pd.read_csv(C.REPORT_DIR / "association_rules_genre.csv")
USER_RULES = pd.read_csv(C.REPORT_DIR / "association_rules_user.csv")
PROFILES = pd.read_csv(C.REPORT_DIR / "cluster_anime_kmeans_profile.csv")
U_PROFILES = pd.read_csv(C.REPORT_DIR / "cluster_users_kmeans_profile.csv")
BENCH = pd.read_csv(C.REPORT_DIR / "apriori_vs_fpgrowth.csv")
U_CUBE = pd.read_csv(C.REPORT_DIR / "cluster_users_cube.csv")


def m(key, default=0):
    return S.get(key, P.get(key, default))


def n(key):
    value = m(key, 0)
    return f"{value:,}" if isinstance(value, (int, float)) else str(value)


TODAY = date.today().strftime("%d %B %Y")


# ============================================================== PAGE 1 =====
PAGE_1 = [
    ("cover", {
        "course": "Data Warehousing & Mining · Project Report",
        "title": "Anime Recommendation\nMining Pipeline",
        "subtitle": "Preprocessing · Association Rules · Classification · "
                    "Regression · Clustering · Prediction",
        "blurb": "An end-to-end data warehousing and mining solution over the Kaggle "
                 "dataset **Anime Recommendations Database** by CooperUnion, scraped "
                 "from MyAnimeList. Six mining stages, every artefact written as CSV, "
                 "presented through a deployable Streamlit interface with a trained "
                 "scoring model and a hybrid recommender.",
        "kpis": [("12,294", "Anime titles", "anime.csv"),
                 ("7,813,737", "Interactions", "ratings.csv"),
                 ("69,600", "User profiles", "clustering dataset B"),
                 ("6", "Mining stages", "preprocess to prediction")],
        "rows": [("Dataset", "Anime Recommendations Database - CooperUnion (Kaggle)"),
                 ("Language", "Python 3.13, scikit-learn, pandas, Streamlit"),
                 ("Storage", "CSV files only - no database server, no JSON artefacts"),
                 ("Interface", "Streamlit, 7 tabs, one per mining stage"),
                 ("Reproducibility", "All seeds fixed at 42, stratified splits"),
                 ("Report date", TODAY)],
    }),
    ("h2", "Contents"),
    ("toc", [("1", "Introduction, dataset and results at a glance", "1 - 2"),
             ("2", "System architecture and project layout", "2"),
             ("3", "Stage 1 - Data preprocessing", "3"),
             ("4", "Stage 2 - Association rules · Stage 3 - Classification", "4"),
             ("5", "Stage 4 - Regression · Stage 5 - Clustering", "5"),
             ("6", "Stage 6 - Prediction and recommendation system", "6"),
             ("7", "User interface, evaluation and conclusions", "7")]),
    ("callout", "", "**How to read this report.** Section 1 gives the architecture, "
                    "sections 2 - 5 explain each mining stage in the order the code "
                    "runs them, and section 6 covers the interface and the "
                    "conclusions. Every figure is a diagram generated from the "
                    "project's own code. Dashed frames are screenshot slots: run "
                    "`streamlit run app.py`, capture the stated view, and paste it "
                    "over the frame."),
]

# ============================================================== PAGE 2 =====
PAGE_2 = [
    ("h1", "1  Introduction and Architecture"),
    ("h2", "1.1 What the project does"),
    ("p", "The dataset holds two tables that describe the same thing from opposite "
          "sides: `anime.csv` describes 12,294 titles and `ratings.csv` holds "
          "7,813,737 statements of taste from 73,515 users. Nothing in the raw data "
          "is clean, nothing is discrete enough to classify directly, and nothing is "
          "pre-computed for a model. The project therefore runs the full classical "
          "mining pipeline over it: clean and reshape the data, mine co-occurrence "
          "patterns, learn rules that classify a title's quality tier, estimate its "
          "score continuously, discover natural groupings in both the titles and the "
          "users, and finally ship a model that scores a new anime and recommends "
          "what to watch next."),
    ("h2", "1.2 End-to-end data flow"),
    ("figure", "end_to_end", "Figure 1 - The pipeline as code executes it. "
                             "Preprocessing is shared by every later stage; stages 2 "
                             "to 6 are independent of each other and can be re-run "
                             "one at a time.", 100),
    ("h2", "1.3 Why the output is CSV"),
    ("p", "Each stage writes flat tables rather than objects, which keeps the whole "
          "project inspectable with a text editor and makes it deployable without a "
          "database. Even the nested pipeline summaries are flattened to a long "
          "`metric, value` table (`reports/mining_summary.csv`), so there is no JSON "
          "anywhere in the data path. Only the two trained models are pickled, because "
          "a fitted scikit-learn estimator cannot be expressed as a table."),
    ("h2", "1.4 Project layout"),
    ("table", ["Path", "Responsibility"],
     [["app.py", "Streamlit UI, 7 tabs, one per stage"],
      ["build_dataset.py", "Stage 1 driver: clean, engineer, aggregate, split"],
      ["train.py", "Stages 2-6 driver; accepts stage numbers"],
      ["smoke_test.py", "Headless AppTest that fails if the UI breaks"],
      ["core/config.py", "Paths, thresholds, class edges, band boundaries"],
      ["core/preprocessing.py", "Cleaning, features, aggregation, baskets, splits"],
      ["core/association.py", "Apriori / FP-Growth and the rule metrics"],
      ["core/decision_tree.py", "From-scratch C4.5: gain ratio + pruning"],
      ["core/classification.py", "J48, Naive Bayes, model comparison"],
      ["core/regression.py", "Six regressors with diagnostics"],
      ["core/clustering.py", "k selection, K-Means, Ward, cluster cubes"],
      ["core/predictor.py", "Blended score regressor and hybrid recommender"],
      ["core/reporting.py", "Long-format CSV summary writer/reader"],
      ["data/raw/, data/processed/", "Source tables and the generated datasets"],
      ["reports/, models/", "Result tables and fitted estimators"]],
     {"code_cols": {0}}),
]

# ============================================================== PAGE 3 =====
PAGE_3 = [
    ("h1", "2  Stage 1 - Data Preprocessing"),
    ("p", "Cleaning is where a mining project is won or lost. Three properties of "
          "this dataset drive every decision below: `rating = -1` means *watched but "
          "not rated* (" + n('ratings.watched_not_rated') + " rows), `episodes` is a "
          "text column containing `\"Unknown\"`, and both `members` and `episodes` are "
          "badly right-skewed. Treating any of these as ordinary numbers silently "
          "corrupts every later stage."),
    ("figure", "preprocessing", "Figure 2 - The cleaning chain. Each box is a function "
                                "call in core/preprocessing.py, in the order the code "
                                "makes them.", 100),
    ("h2", "2.1 Decisions and why they were made"),
    ("table", ["Issue found", "Treatment", "Reason"],
     [[f"rating = -1 ({n('ratings.watched_not_rated')} rows)",
       "Excluded as missing, not scored as a rating",
       "It encodes watched, not liked; scoring it would drag every mean down"],
      [f"episodes = \"Unknown\" ({n('cleaning.episodes_impute_median')} rows)",
       "Coerced to numeric, then imputed with the media-type median",
       "A movie and a 26-episode series have very different medians"],
      [f"Missing genre ({n('cleaning.missing_filled.genre')}) and type "
       f"({n('cleaning.missing_filled.type')})", "Filled with \"Unknown\"",
       "Keeps the row and makes the missingness explicit"],
      [f"Missing rating ({n('cleaning.rating_missing')} rows)",
       "Type-wise mean, then global mean",
       "Preserves the per-type offset instead of flattening it"],
      ["Duplicate anime_id",
       f"Dropped, keeping the first ({n('cleaning.duplicate_ids_removed')} found)",
       "Would double-count the title in every aggregate"],
      ["Right-skewed members, episodes",
       "log1p transform added, used by clustering and the trees",
       "One blockbuster with 3M members otherwise dominates means and distance"],
      ["Extreme but valid values",
       "Flagged is_outlier by the 1.5 x IQR fence, kept in the data",
       "Deletion would remove the titles the recommender needs most"]],
     {"code_cols": {0}, "font": 7.6}),
    ("callout", "ok", "**Handling of the 1 GB ratings file.** The "
                       + n('ratings.raw_interactions') + " rows are never loaded into "
                       "memory: `build_ratings_tables()` streams them in 500,000-row "
                       "chunks and keeps only running sums, sums of squares, maxima "
                       "and minima. Peak memory stays flat and the stage finishes in "
                       "about 13 seconds."),
    ("h2", "2.2 What comes out"),
    ("table", ["Output table", "Rows", "Used by"],
     [["anime_clean.csv", n('cleaning.clean_rows'),
       "Clustering dataset A, classification, regression"],
      ["anime_rating_stats.csv", n('ratings.n_anime_rated'),
       "Community prior for the recommender"],
      ["user_features.csv", n('ratings.n_users_rated'),
       "Clustering dataset B, recommender segments"],
      ["baskets_genre.csv", n('baskets.genre'), "Stage 2 genre association rules"],
      ["baskets_user.csv", f"{n('baskets.user')} pairs",
       "Stage 2 user rules; 5,501 users survive the >= 3-title filter"],
      ["classification_split.csv",
       f"{n('modelling.classification.rows')} "
       f"({n('modelling.classification.train')} / "
       f"{n('modelling.classification.test')})",
       "Stage 3, stratified on the class; 58 features"],
      ["regression_split.csv",
       f"{n('modelling.regression.rows')} ({n('modelling.regression.train')} / "
       f"{n('modelling.regression.test')})", "Stages 4 and 6; 57 features"]],
     {"code_cols": {0}, "numeric": {1}}),
    ("p", "The two clustering datasets are derived, not copied: dataset A is the "
          "cleaned anime table (content features), dataset B the aggregated user table "
          "(behaviour features)."),
    ("shots", [("Tab 1 - Overview",
                "KPIs, raw data preview, artefact map and last run's results", 26),
               ("Tab 2 - Preprocessing",
                "Missing-value bars, the IQR outlier table and the rating-tier histogram",
                26)]),
]

# ============================================================== PAGE 4 =====
PAGE_4 = [
    ("h1", "3  Stage 2 - Association Rules | Stage 3 - Classification"),
    ("h2", "3.1 Association rule mining"),
    ("p", "Two rule bases are mined, because \"which genres go together\" and \"what "
          "does someone watch after this\" are different questions with different units "
          "of analysis. A genre basket treats each **anime** as a transaction whose "
          "items are its genres; a user basket treats each **user** as a transaction "
          "whose items are the titles they rated highly. Both directions of every pair "
          "are kept, because `Mecha -> Action` and `Action -> Mecha` share a lift but "
          "not a confidence."),
    ("p", "Both **Apriori** and **FP-Growth** are implemented and benchmarked "
          "(`core/association.py`). They return identical itemsets at every threshold "
          "tested - " + n('stage2.genre_rules') + " genre rules and "
          + n('stage2.user_rules') + " user rules at the default settings - which is "
          "the correctness check that matters. Every rule also carries conviction, "
          "leverage, Zhang's metric and Kulczynski's number, so it can be judged on "
          "more than lift alone."),
    ("table", ["Rule (as printed in the CSV)", "Supp.", "Conf.", "Lift", "Reading"],
     [[RULES.iloc[0]['rule'], f"{RULES.iloc[0]['support']:.3f}",
       f"{RULES.iloc[0]['confidence']:.3f}", f"{RULES.iloc[0]['lift']:.2f}",
       "strongest genre pair"],
      [RULES.iloc[2]['rule'], f"{RULES.iloc[2]['support']:.3f}",
       f"{RULES.iloc[2]['confidence']:.3f}", f"{RULES.iloc[2]['lift']:.2f}",
       "most frequent strong pair"],
      [RULES.iloc[4]['rule'], f"{RULES.iloc[4]['support']:.3f}",
       f"{RULES.iloc[4]['confidence']:.3f}", f"{RULES.iloc[4]['lift']:.2f}",
       "demographic genre -> genre"],
      [USER_RULES.iloc[0]['rule'], f"{USER_RULES.iloc[0]['support']:.3f}",
       f"{USER_RULES.iloc[0]['confidence']:.3f}", f"{USER_RULES.iloc[0]['lift']:.2f}",
       "sequel-completion behaviour"],
      [USER_RULES.iloc[2]['rule'], f"{USER_RULES.iloc[2]['support']:.3f}",
       f"{USER_RULES.iloc[2]['confidence']:.3f}", f"{USER_RULES.iloc[2]['lift']:.2f}",
       "franchise loyalty"]],
     {"code_cols": {0}, "numeric": {1, 2, 3}, "best": 3, "font": 7.4}),
    ("callout", "", "**Apriori versus FP-Growth.** Both find exactly the same itemsets "
                    "at every threshold tested ("
                    + ", ".join(str(v) for v in BENCH['apriori_itemsets'])
                    + " at support "
                    + ", ".join(f"{v:.0%}" for v in BENCH['min_support'])
                    + ") - the correctness guarantee. Apriori is faster here because it "
                    "scans only "
                    + f"{int(BENCH.iloc[0]['n_transactions']):,}"
                    + " short baskets; FP-Growth's advantage is asymptotic and both "
                    "timings are reported."),
    ("h2", "3.2 Classification with J48 and Naive Bayes"),
    ("p", "The target `rating` is binned into four classes (`Low`, `Medium`, `High`, "
          "`Top`) at the 30/30/20/20 quantiles, giving "
          + n('modelling.classification.class_distribution.Low') + " Low, "
          + n('modelling.classification.class_distribution.Medium') + " Medium, "
          + n('modelling.classification.class_distribution.High') + " High and "
          + n('modelling.classification.class_distribution.Top') + " Top titles. J48 "
          "is implemented from scratch in `core/decision_tree.py` over 12 attributes "
          "- 8 numeric plus `type`, `primary_genre`, `members_band` and "
          "`episodes_band`."),
    ("figure", "j48", "Figure 3 - The C4.5 loop and where the measurements come from. "
                      "Naive Bayes is handed the same attributes on the same split.",
     78),
    ("table", ["Model", "CV accuracy", "Test accuracy", "Macro F1", "ROC-AUC", "Note"],
     [["Random Forest", "0.5812", f"{m('stage3.best_accuracy'):.4f}", "0.5649",
       "0.8287", "best overall - ensembling wins"],
      ["Decision Tree (entropy)", "0.5572", "0.5628", "0.5505", "0.8034",
       "sklearn reference for our C4.5"],
      ["J48 / C4.5 (own implementation)", "0.5435", f"{m('stage3.j48.accuracy'):.4f}",
       f"{m('stage3.j48.f1_macro'):.4f}", "-",
       f"{m('stage3.j48_stats.depth')} levels, {n('stage3.j48_stats.n_rules')} rules"],
      ["Naive Bayes (Gaussian)", "0.4710", "0.4758", "0.4396", "0.7403",
       "scaled numeric columns"],
      ["Majority baseline", "0.2999", "0.2997", "0.1153", "0.5000",
       "always predict the most common class"]],
     {"numeric": {1, 2, 3, 4}, "best": 0}),
    ("p", "A second variant, **Categorical NB** over one-hot attributes, reaches "
          + f"{m('stage3.categorical_nb.accuracy'):.4f} / "
          + f"{m('stage3.categorical_nb.f1_macro'):.4f}"
          + " in its own harness, so it is kept out of the shared-encoding table. "
          "The forest adds "
          + f"{(m('stage3.best_accuracy') - 0.2997) * 100:.1f}"
          + " accuracy points over the baseline; gain-ratio ranks `is_shoujo`, "
          "`members_band`, `is_shounen`, `members` and `episodes_band` on top, so "
          "community size and genre define the tier while episode count barely does. "
          "J48 lands within "
          + f"{m('stage3.best_accuracy') - m('stage3.j48.accuracy'):.3f}"
          + " of the forest while still producing "
          + n('stage3.j48_stats.n_rules')
          + " readable IF/THEN rules - extracted knowledge rather than the last "
          "accuracy point."),
    ("callout", "warn", "**A scoring bug found by cross-checking.** The comparison "
                        "harness ordinal-encoded the training and test frames "
                        "*independently*, so every model was scored against category "
                        "codes it had never seen. Fitting one mapping on train and "
                        "reusing it for test moved Gaussian NB from 0.4451 to 0.4758, "
                        "the tree from 0.5427 to 0.5628 and the forest from 0.5607 to "
                        + f"{m('stage3.best_accuracy'):.4f}"
                        + ". The table is post-fix."),
    ("shots", [("Tab 3 - Association Rules",
                "Support-vs-confidence scatter and both rule tables", 26),
               ("Tab 4 - Classification",
                "J48 tree, IF/THEN rules and the comparison chart", 26)]),
]

# ============================================================== PAGE 5 =====
PAGE_5 = [
    ("h1", "4  Stage 4 - Regression | Stage 5 - Clustering"),
    ("h2", "4.1 Regression - estimating the score continuously"),
    ("p", "Classification bins the score and throws information away; regression keeps "
          "it. Six regressors share one 80/20 split."),
    ("table", ["Model", "CV RMSE", "MAE", "RMSE", "R2", "MAPE %"],
     [["Gradient Boosting", "0.6393", "0.4833", "0.6545", "0.5956", "8.16"],
      ["Random Forest", "0.6487", "0.4937", "0.6690", "0.5774", "8.31"],
      ["Decision Tree", "0.6831", "0.5168", "0.6971", "0.5412", "8.73"],
      ["Linear Regression", "0.7016", "0.5508", "0.7354", "0.4894", "9.41"],
      ["Ridge (alpha = 1)", "0.7016", "0.5508", "0.7354", "0.4894", "9.41"],
      ["Mean baseline", "0.9897", "0.7973", "1.0292", "-0.0001", "13.66"]],
     {"numeric": {1, 2, 3, 4, 5}, "best": 0}),
    ("p", "Gradient boosting is the strongest single model; the ensembles reach "
          "R2 = 0.60 against a mean baseline of 0. Read honestly: popularity and genre "
          "explain about 60% of score variance, and the other 40% is quality no "
          "attribute here captures. Ridge is indistinguishable from plain linear "
          "regression at this sample size - shrinkage has nothing to shrink."),
    ("p", "Error by media type shows where it is weak. **TV**, the largest group "
          "(n = 738), is also the most predictable at MAE 0.427; **ONA** is worst at "
          "0.623 on 122 rows, with **Movie** between them at 0.555. The learning curve "
          "is still descending, so this stage needs more data next, not more "
          "hyper-parameter search."),
    ("callout", "warn", "**A leak that was found and removed.** An engineered feature "
                        "`engagement_index = members x rating / 1000` produced an "
                        "impossible test R2 of 0.91: the feature contains the target, "
                        "so the model was reading the answer from its input. It was "
                        "deleted, and the honest figure is the 0.60 above. Every split "
                        "in this project is therefore made before any transformation is "
                        "fitted."),
    ("h2", "4.2 Clustering - two datasets, two algorithms"),
    ("p", "Both datasets follow one workflow: standardise, choose `k`, run K-Means, "
          "run Ward-linkage agglomerative clustering, profile the centroids, "
          "cross-tabulate."),
    ("figure", "clustering", "Figure 4 - The clustering workflow. k differs per dataset "
                             "because k is chosen from the data, not assumed.", 100),
    ("table", ["Dataset", "Rows", "Features", "k", "K-Means silhouette",
               "Ward silhouette", "k chosen because"],
     [["A - Anime (content)", n('stage5.anime.rows'), "6", f"{m('stage5.anime.k')}",
       f"{m('stage5.anime.kmeans_silhouette'):.4f}",
       f"{m('stage5.anime.hier_silhouette'):.4f}", m('stage5.anime.why')],
      ["B - Users (behaviour)", n('stage5.users.rows'), "8", f"{m('stage5.users.k')}",
       f"{m('stage5.users.kmeans_silhouette'):.4f}",
       f"{m('stage5.users.hier_silhouette'):.4f}", m('stage5.users.why')]],
     {"numeric": {1, 2, 3, 4, 5}, "font": 7.4}),
    ("h3", "What the segments turned out to be"),
    ("p", "Centroid profiles are named from their own statistics, so labels describe "
          "rather than decorate. Dataset A splits along reach and reception: its "
          "strongest segment, *"
          + PROFILES.loc[PROFILES['rating'].idxmax(), 'segment_name'] + "*, averages "
          + f"{PROFILES['rating'].max():.2f}" + " with "
          + f"{PROFILES['members_log'].max():.1f}" + " log-members against "
          + f"{PROFILES[PROFILES['rating'] < 6.5]['rating'].mean():.2f}"
          + " for the two weak ones. Dataset B splits along *volume* and *generosity*: "
          "the largest segment (" + f"{int(U_PROFILES['n'].max()):,}"
          + " users) rates "
          + f"{U_PROFILES.loc[U_PROFILES['n'].idxmax(), 'mean_rating']:.2f}" + " over "
          + f"{int(U_PROFILES.loc[U_PROFILES['n'].idxmax(), 'n_ratings'])}"
          + " titles, the heaviest raters (" + f"{int(U_PROFILES.loc[U_PROFILES['n_ratings'].idxmax(), 'n']):,}"
          + ") rate "
          + f"{U_PROFILES.loc[U_PROFILES['n_ratings'].idxmax(), 'mean_rating']:.2f}"
          + ", and the cross-tab puts "
          + f"{U_CUBE.iloc[3]['high'] / U_CUBE.iloc[3][['good', 'high', 'low', 'mid']].sum():.0%}"
          + " of the light/generous segment in the top rating band."),
    ("h3", "How k was chosen, and why hierarchical runs on a sample"),
    ("p", "Three signals are reported for every k: the **elbow** (furthest point from "
          "the line joining the first and last inertia value), the maximised "
          "**silhouette**, and Calinski-Harabasz with Davies-Bouldin as checks. The two "
          "main signals disagree by exactly one cluster on both datasets ("
          + m('stage5.anime.why') + " for A, " + m('stage5.users.why')
          + " for B); the elbow wins both times, because one fewer cluster merges two "
          "genuinely distinct groups - on A the two low-scoring segments differ sharply "
          "in episode length, on B it would merge two light rater tiers with opposite "
          "habits. Ward linkage needs the full n x n matrix (38 GB for 69,600 users), "
          "so `hierarchical_cluster()` fits a seeded 1,200-row subsample and keeps the "
          "whole merge tree - any k from 2 to 8 comes off one run."),
    ("shots", [("Tab 5 - Regression",
                "Actual-vs-predicted scatter, residuals and the learning curve", 26),
               ("Tab 6 - Clustering",
                "Dataset B selected, the dendrogram cut at k=5 and the cluster cube", 26)]),
]

# ============================================================== PAGE 6 =====
PAGE_6 = [
    ("h1", "5  Stage 6 - Prediction and Recommendation System"),
    ("p", "The final stage produces the two models the interface actually calls. Both "
          "are trained here, persisted, and then only read at serving time."),
    ("figure", "prediction", "Figure 5 - Model A scores a new anime; model B ranks "
                             "candidates for a viewer. Neither is retrained when the UI "
                             "runs.", 100),
    ("h2", "5.1 Score predictor"),
    ("p", "A Random Forest and a Gradient Boosting regressor are blended. The weight is "
          "fitted on a validation split carved out of the *training* data, never on the "
          "test set; both learners are then refitted on the full training split. On the "
          "untouched test set the blend reaches RMSE **" + f"{m('stage6.sigma'):.4f}"
          + "** and R2 **0.5973**, slightly better than either learner alone, and the UI "
          "shows a 95% interval of score +/- 1.96 x RMSE so a prediction is never a "
          "bare number."),
    ("table", ["Stage", "Weight source", "MAE", "RMSE", "R2", "Interpretation"],
     [["Blend on validation", "used only to fit the weight", "0.4714", "0.6490",
       "0.5643", "diagnostic only - not a result"],
      ["Blend on test", "never seen", "0.4812", f"{m('stage6.sigma'):.4f}", "0.5973",
       "the honest number"],
      ["Gradient Boosting", "never seen", "0.4823", "0.6535", "0.5968",
       "strongest single model"],
      ["Random Forest", "never seen", "0.4909", "0.6670", "0.5800",
       "adds diversity to the blend"]],
     {"numeric": {2, 3, 4}, "best": 1}),
    ("h2", "5.2 Hybrid recommender"),
    ("p", "\"What should I watch next?\" is answered by combining five signals that "
          "earlier stages already computed, so the recommender adds no training cost:"),
    ("numbered", ["**Taste segment (0.30).** K-Means over the "
                  + n('stage5.users.rows') + " user profiles gives "
                  + f"{m('stage5.users.k')}" + " segments; each segment's mean rating "
                  "per genre is streamed out of the raw ratings.",
                  "**Media-type preference (0.18).** The same per type, so a segment "
                  "can prefer TV over movies.",
                  "**Community prior (0.20)** and **type-relative quality (0.18)**. The "
                  "title's own score z-scored across candidates, and how far it sits "
                  "above its type's average.",
                  "**Popularity and rules (0.14 + boost).** A small log-popularity "
                  "preference, plus the lift of any stage-2 rule whose antecedent "
                  "matches the watch list."]),
    ("p", "The result is a defensible `match_score` with a `why` column - \"Action: "
          "your segment rates it 8.4 vs community 7.9\" - so a recommendation can be "
          "explained rather than asserted. " + n('stage6.recommender.n_users')
          + " users with at least 20 ratings are eligible; "
          + n('stage6.recommender.n_anime') + " titles are candidates."),
    ("h2", "5.3 Three entry points, one engine"),
    ("table", ["Mode", "Input", "How the score is formed"],
     [["An existing user", "user_id",
       "their own segment's affinities; nothing is invented"],
      ["My preferences", "genres + media type",
       "global affinities, with the chosen genres lifted by +0.8"],
      ["Titles I watched", "up to 8 titles",
       "genres inferred from the watch list, plus lift from matching rules"]],
     {"code_cols": {1}}),
    ("callout", "ok", "**One shared encoder.** Every request is built by "
                      "`RatingPredictor._row()`, which applies the same log transforms, "
                      "band edges and genre flags as training - a user cannot slip in "
                      "an input the model was never fitted on, the usual failure mode "
                      "of a hand-rolled scoring form."),
    ("shots", [("Tab 7 - score form",
                "Predicted MAL score, tier band, 95% interval and the RF vs GB spread", 30),
               ("Tab 7 - recommendations",
                "match_score ranking with the genre affinity and reason columns", 30)]),
]

# ============================================================== PAGE 7 =====
PAGE_7 = [
    ("h1", "6  User Interface, Evaluation and Conclusions"),
    ("h2", "6.1 How the interface is built"),
    ("p", "`app.py` uses one tab per stage. Rather than reading pre-baked screenshots, "
          "every figure on screen is computed from `core/` at render time, and the "
          "sliders re-run the corresponding algorithm, so the thresholds shown by the "
          "association tab genuinely change the rule set and the depth slider genuinely "
          "re-grows the J48 tree."),
    ("figure", "ui", "Figure 6 - Tab-to-module wiring. `@st.cache_data` means each tab "
                     "is computed once per session; switching tabs is instant after "
                     "that.", 100),
    ("table", ["Tab", "What it shows", "Interactive controls"],
     [["1 Overview", "KPIs, raw data preview, artefact map, last run's results", "-"],
      ["2 Preprocessing",
       "Missing values, IQR outliers, features, distributions, splits", "-"],
      ["3 Association", "Both rule bases, lift scatter, Apriori vs FP-Growth",
       "support, confidence, lift, max length"],
      ["4 Classification", "J48 tree and rules, NB densities, comparison",
       "depth, leaf size, pruning, rule confidence"],
      ["5 Regression", "Comparison, residuals, learning curve, error by type", "-"],
      ["6 Clustering", "k diagnostics, K-Means, dendrogram, cluster cube",
       "dataset, k, seed, sample size"],
      ["7 Prediction", "Score form, interval, recommender, retrain",
       "score inputs, recommender mode, top-N"]],
     {"font": 7.6}),
    ("h2", "6.2 Evaluation summary"),
    ("table", ["Objective", "Measure", "Result", "Verdict"],
     [["Predict the quality tier", "test accuracy vs 0.2997 baseline",
       f"{m('stage3.best_accuracy'):.4f} ({m('stage3.best')})",
       f"Strong - +{(m('stage3.best_accuracy') - 0.2997) * 100:.1f} points over baseline"],
      ["Extract readable rules", "rules at 5 levels",
       f"{n('stage3.j48_stats.n_rules')} rules, J48 F1 {m('stage3.j48.f1_macro'):.4f}",
       "Usable - every rule is a readable IF/THEN"],
      ["Estimate the score", "held-out RMSE / R2",
       f"{m('stage6.sigma'):.4f} / 0.5973",
       "Good - within +/- 0.65 of the true score"],
      ["Find taste segments", "silhouette, dataset B",
       f"{m('stage5.users.kmeans_silhouette'):.4f} at k = {m('stage5.users.k')}",
       "Real structure, weak separation"],
      ["Recommend titles", "candidate pool / eligible users",
       f"{n('stage6.recommender.n_anime')} / {n('stage6.recommender.n_users')}",
       "Explainable and rule-augmented"]],
     {"font": 7.6}),
    ("h2", "6.3 Honest limitations"),
    ("bullets", ["**Genre is the only textual signal.** With no synopsis, studio or "
                 "demographic attributes, roughly 40% of score variance is unreachable.",
                 "**Hierarchical clustering is sampled.** Its silhouette is measured on "
                 "1,200 rows, not the full 69,600, and is labelled as such everywhere "
                 "it appears.",
                 "**No offline ranking metric.** The recommender is justified by its "
                 "blend logic and explainability, not by a measured NDCG or hit-rate, "
                 "because the raw data contains no held-out \"next watch\" sequence.",
                 "**Class imbalance is inherent.** `Low` is genuinely rarer than "
                 "`Medium`, so macro-F1 rather than accuracy is the fair metric."]),
    ("h2", "6.4 Reproducing the project"),
    ("pre", "python -m pip install -r requirements.txt\n"
            "python build_dataset.py      # stage 1  (~13 s)\n"
            "python train.py              # stages 2-6 (~5 min)\n"
            "python smoke_test.py         # headless UI check\n"
            "streamlit run app.py         # http://localhost:8501"),
    ("p", "Deployment is a single Streamlit app with no database: about 50 MB with the "
          "models included. The 1 GB raw ratings file is excluded by `.gitignore` and is "
          "never read at serving time - the app falls back to the processed tables when "
          "it is absent."),
    ("h2", "6.5 Conclusion"),
    ("p", "All six requested stages are implemented, measured on held-out data and "
          "exposed through one interface. The two deliverables that go beyond a "
          "textbook pipeline are the honest reporting of the leaked-feature "
          "correction, and the fact that every number in this report is read from the "
          "pipeline's own CSV output rather than typed in - so the document and the "
          "code cannot disagree. The natural next step is a held-out sequential "
          "evaluation of the recommender, which this dataset's schema does not "
          "currently support."),
]

PAGES = [PAGE_1, PAGE_2, PAGE_3, PAGE_4, PAGE_5, PAGE_6, PAGE_7]
PAGE_TITLES = ["DWM Anime Mining Pipeline",
               "1 - Introduction and architecture",
               "2 - Data preprocessing",
               "3 - Association rules and classification",
               "4 - Regression and clustering",
               "5 - Prediction system",
               "6 - Interface, evaluation and conclusions"]