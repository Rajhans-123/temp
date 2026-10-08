"""Streamlit front-end for the Data Warehousing & Mining project.

    streamlit run app.py

Everything the app shows is computed by the modules in `core/`; heavy results are
cached so switching tabs does not re-run the miners.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from core import association as A
from core import classification as CL
from core import clustering as CLC
from core import config as C
from core import live_feed as LF
from core import predictor as PRED
from core import preprocessing as P
from core import regression as R
from core import reporting as REP

warnings.filterwarnings("ignore")
C.ensure_dirs()

st.set_page_config(
    page_title="Anime DWM Lab - Data Warehousing & Mining",
    page_icon="[ DWM ]",
    layout="wide",
    initial_sidebar_state="expanded",
)
CSS = """
<style>
  .block-container { padding-top: 2.2rem; }
  .kpi { border-radius: 12px; padding: 14px 18px; background: #f6f7fb;
         border: 1px solid #e3e6ef; text-align: left; }
  .kpi .label { font-size: .78rem; letter-spacing: .04em; text-transform: uppercase;
                color: #6b7280; }
  .kpi .value { font-size: 1.6rem; font-weight: 700; color: #111827; }
  .kpi .sub   { font-size: .78rem; color: #6b7280; }
  pre.tree { background:#0f172a; color:#e2e8f0; padding:14px; border-radius:10px;
             font-size:.76rem; max-height:460px; overflow:auto; line-height:1.45 }
  .stTabs [data-baseweb="tab"] { font-weight:600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ============================================================ helpers =======
def kpi(label: str, value, sub: str = "") -> None:
    st.markdown(
        f'<div class="kpi"><div class="label">{label}</div>'
        f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def live_feed_status() -> dict:
    """Summarise the optional Tenrai live feed (missing files = never synced)."""
    info: dict = {"synced_titles": 0, "last_sync": None, "snapshots": 0}
    try:
        if C.LIVE_STATE.exists():
            st_ = pd.read_csv(C.LIVE_STATE)
            info["synced_titles"] = len(st_)
            info["last_sync"] = str(st_["fetched_at"].max())
        if C.LIVE_HISTORY.exists():
            info["snapshots"] = sum(1 for _ in open(C.LIVE_HISTORY, encoding="utf-8")) - 1
    except Exception:
        pass
    return info


class _Cols(list):
    """`st.columns` result that also works with `with ... as (a, b, c):`."""

    def __enter__(self):
        return tuple(self)

    def __exit__(self, *exc):
        return False


def cols(n):
    return _Cols(st.columns(n))


def read_report(name: str) -> pd.DataFrame:
    p = C.REPORT_DIR / name
    return pd.read_csv(p) if p.exists() else pd.DataFrame()


def table(df: pd.DataFrame, height: int = 320, **kw):
    if df is None or df.empty:
        st.info("no rows for the current settings")
        return
    # format only the numeric columns - a blanket precision breaks on string columns
    fmt = {c: "{:,.4f}" for c in df.columns
           if pd.api.types.is_numeric_dtype(df[c])
           and not pd.api.types.is_bool_dtype(df[c])}
    st.dataframe(df.style.format(fmt, na_rep="-"), height=height,
                 width="stretch", **kw)


def download(df: pd.DataFrame, name: str) -> None:
    st.download_button("download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv",
                       key=f"dl_{name}_{abs(hash(name)) % 9999}")


# ------------------------------------------------------------ data cache ---
@st.cache_data(show_spinner="loading the cleaned dataset ...")
def load_anime() -> pd.DataFrame:
    return pd.read_csv(C.CLEAN_ANIME)


@st.cache_data(show_spinner="loading user profiles ...")
def load_users() -> pd.DataFrame:
    return pd.read_csv(C.USER_FEATURES)


@st.cache_data(show_spinner="loading the preprocessing report ...")
def load_prep_report() -> dict:
    return REP.read_summary(C.REPORT_DIR / "preprocessing_report.csv")


@st.cache_data(show_spinner="loading the mining summary ...")
def load_mining_summary() -> dict:
    return REP.read_summary(C.REPORT_DIR / "mining_summary.csv")


@st.cache_data(show_spinner="mining association rules ...")
def get_genre_rules(ms: float, mc: float, ml: float) -> pd.DataFrame:
    return A.mine_genre_rules(min_support=ms, min_confidence=mc, min_lift=ml)


@st.cache_data(show_spinner="mining user basket rules ...")
def get_user_rules(ms: float, mc: float, ml: float) -> pd.DataFrame:
    return A.mine_user_rules(min_support=ms, min_confidence=mc, min_lift=ml)


@st.cache_data(show_spinner="building transaction baskets ...")
def get_baskets() -> pd.DataFrame:
    return P.load_baskets_user(min_items=3)


@st.cache_data(show_spinner="training J48 (C4.5) ...")
def get_j48(depth: int, leaf: int, conf: float):
    Xtr, ytr, Xte = CL.load_classification_data()
    yte = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))["target"]
    from core.decision_tree import C45Tree
    t = C45Tree(max_depth=depth, min_samples_leaf=leaf, confidence=conf).fit(Xtr, ytr)
    rules = t.to_rules(min_conf=0.5)
    from sklearn.metrics import accuracy_score, f1_score
    p = t.predict(Xte)
    return {
        "model": t, "rules": rules, "text": t.text_tree(), "stats": t.stats(),
        "accuracy": round(float(accuracy_score(yte, p)), 4),
        "f1": round(float(f1_score(yte, p, average="macro", zero_division=0)), 4),
        "cm": pd.crosstab(pd.Series(yte.to_numpy(), name="actual"),
                          pd.Series(p, name="predicted"))
                   .reindex(columns=C.TIER_ORDER, fill_value=0)
                   .reindex(index=C.TIER_ORDER, fill_value=0),
        "n_test": len(yte),
    }


@st.cache_data(show_spinner="training Naive Bayes ...")
def get_nb():
    Xtr, ytr, Xte = CL.load_classification_data()
    yte = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))["target"]
    r = CL.run_naive_bayes(Xtr, ytr, Xte, yte)
    r["cm_gnb"] = pd.crosstab(
        pd.Series(yte.to_numpy(), name="actual"),
        pd.Series(r["predictions"]["gaussian_nb"], name="predicted")
    ).reindex(columns=C.TIER_ORDER, fill_value=0).reindex(index=C.TIER_ORDER, fill_value=0)
    r["cm_dnb"] = pd.crosstab(
        pd.Series(yte.to_numpy(), name="actual"),
        pd.Series(r["predictions"]["categorical_nb"], name="predicted")
    ).reindex(columns=C.TIER_ORDER, fill_value=0).reindex(index=C.TIER_ORDER, fill_value=0)
    return r


@st.cache_data(show_spinner="comparing classifiers (5-fold CV) ...")
def get_compare():
    # Cloud-safe: prefer the precomputed table from `python train.py`.
    # Live 5-fold CV (RF + J48 + NB) needs >500 MB and OOMs the 1 GB Cloud
    # container, so it only runs on explicit request via `live=True`.
    pre = C.REPORT_DIR / "classification_comparison.csv"
    pred_path = C.REPORT_DIR / "classification_predictions.csv"
    if pre.exists():
        table = pd.read_csv(pre)
        preds = pd.read_csv(pred_path) if pred_path.exists() else pd.DataFrame()
        best = table.sort_values("accuracy", ascending=False).iloc[0]["model"] \
            if not table.empty else "-"
        return {"table": table, "predictions": preds, "best": best,
                "precomputed": True}
    Xtr, ytr, Xte = CL.load_classification_data()
    yte = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))["target"]
    r = CL.compare_models(Xtr, ytr, Xte, yte)
    r["precomputed"] = False
    return r


@st.cache_data(show_spinner="loading regression results ...")
def get_regression():
    # Cloud-safe: serve precomputed CSVs. Training 6 regressors x 5-fold CV +
    # learning curve live costs ~800 MB and reliably exceeds Cloud limits.
    comp = C.REPORT_DIR / "regression_comparison.csv"
    if comp.exists():
        try:
            table = pd.read_csv(comp)
            scatter = pd.read_csv(C.REPORT_DIR / "regression_actual_vs_predicted.csv")
            # cap scatter to 2000 points so Plotly JSON stays small
            if len(scatter) > 2000:
                scatter = scatter.sample(min(2000, len(scatter)), random_state=42)
            residuals = pd.read_csv(C.REPORT_DIR / "regression_residuals.csv")
            if len(residuals) > 2000:
                residuals = residuals.sample(min(2000, len(residuals)), random_state=42)
            by_type = pd.read_csv(C.REPORT_DIR / "regression_by_type.csv")
            lc = pd.read_csv(C.REPORT_DIR / "regression_learning_curve.csv")
            imp = pd.read_csv(C.REPORT_DIR / "regression_importance.csv")
            tree_txt = ""
            tp = C.REPORT_DIR / "sklearn_tree.txt"
            if tp.exists():
                tree_txt = tp.read_text(encoding="utf-8")[:3500]
            return {"table": table, "scatter": scatter, "residuals": residuals,
                    "by_type": by_type, "learning_curve": lc,
                    "importance": imp, "tree_text": tree_txt,
                    "precomputed": True}
        except Exception:
            pass
    Xtr, ytr, Xte, yte = R.load_regression_data()
    r = R.run_regression(Xtr, ytr, Xte, yte)
    r["importance"] = R.feature_importance(Xtr, ytr)
    r["precomputed"] = False
    return r


@st.cache_data(show_spinner="scanning k (elbow + silhouette) ...")
def get_kdiag(key: str):
    df = CLC.load_dataset(key)
    Z, _, _ = CLC.scaled_matrix(df, key)
    return df, CLC.k_selection(Z)


@st.cache_data(show_spinner="running K-Means ...")
def get_kmeans(key: str, k: int, seed: int):
    df = CLC.load_dataset(key)
    r = CLC.kmeans_cluster(df, key, k, seed=seed)
    r["named"] = CLC.name_clusters(r["profile"], key)
    r["cube"] = CLC.cluster_vs_class(df, key, r["labels"])
    return r


@st.cache_data(show_spinner="running hierarchical clustering ...")
def get_hier(key: str, k: int, sample: int):
    df = CLC.load_dataset(key)
    r = CLC.hierarchical_cluster(df, key, k, sample=sample)
    r["named"] = CLC.name_clusters(r["profile"], key)
    return r


@st.cache_resource(show_spinner="loading the trained predictor ...")
def get_predictor():
    return PRED.RatingPredictor()


@st.cache_resource(show_spinner="loading the recommender ...")
def get_recommender():
    return PRED.Recommender()


# ============================================================= sidebar ======
with st.sidebar:
    st.markdown("### DWM Lab")
    st.markdown("*Anime Recommendations Database*")
    st.caption("Kaggle \u2013 CooperUnion (MyAnimeList).")
    st.caption("12,294\u00a0titles / 73,515\u00a0users / 7,813,737\u00a0ratings.")
    _live = live_feed_status()
    if _live["synced_titles"]:
        st.caption(f"Live feed: Tenrai API (MyAnimeList) - "
                   f"{_live['synced_titles']:,} titles refreshed "
                   f"(last sync {_live['last_sync']}).")
    else:
        st.caption("Live feed: Tenrai API (MyAnimeList) \u2013 fetching live. "
                   "Try the single-title demo in 8 - Live Feed; bulk refresh with "
                   "`python sync_jikan.py --top 200`.")
    st.divider()
    st.markdown("**Pipeline**")
    st.markdown(
        "1. Data preprocessing\n"
        "2. Association rule mining\n"
        "3. Classification - J48 & Naive Bayes\n"
        "4. Regression - continuous estimation\n"
        "5. Clustering - K-Means & hierarchical\n"
        "6. Prediction system"
    )
    st.divider()
    if not C.MODEL_PATH.exists() or not C.RECOMMENDER_PATH.exists():
        st.error("models not trained yet - run `python train.py`")
    else:
        st.success("models ready")
    st.caption("All intermediate data is stored as CSV in data/ and reports/.")

TABS = ["1 - Overview", "2 - Preprocessing", "3 - Association Rules",
        "4 - Classification", "5 - Regression", "6 - Clustering",
        "7 - Prediction System", "8 - Live Feed"]
# NOTE: sidebar navigation (not st.tabs) is intentional for Streamlit Cloud.
# st.tabs executes *every* tab's block on each rerun, so opening the app would
# simultaneously train classifiers + regressors + K-Means on 69k users and blow
# the 1 GB Cloud memory limit. A radio renders only the selected page.
page = st.sidebar.radio("Go to", TABS, index=0)


# ========================================================== 1 OVERVIEW ======
if page == TABS[0]:
    st.title("Data Warehousing & Mining on the Anime Recommendations Database")
    st.markdown(
        "A single end-to-end mining pipeline over the MyAnimeList data scraped by "
        "CooperUnion and published on Kaggle. Every stage below is reproducible "
        "from `python build_dataset.py` followed by `python train.py`."
    )
    anime = load_anime()
    users = load_users()
    prep = load_prep_report()
    mine = load_mining_summary()

    with cols(6):
        kpi("Anime titles", f"{len(anime):,}", "anime.csv")
        kpi("Ratings", f"{REP.get(prep, 'ratings.scored_interactions', 0):,}", "scored interactions")
        kpi("Users profiled", f"{len(users):,}", "ratings.csv aggregated")
        kpi("Association rules", f"{len(read_report('association_rules_genre.csv')) + len(read_report('association_rules_user.csv')):,}", "genre + user baskets")
        kpi("Classification", f"{float(REP.get(mine, 'stage3.best_accuracy', 0)):.1%}", "best test accuracy")
        kpi("Regression R2", f"{float(REP.get(mine, 'stage4.metrics[0].R2', 0)):.3f}", "best model")

    st.subheader("What the source data looks like")
    live = live_feed_status()
    if live["synced_titles"]:
        st.success(
            f"Live feed active - **Tenrai API** (live MyAnimeList data): "
            f"{live['synced_titles']:,} titles refreshed, "
            f"last sync `{live['last_sync']}`, "
            f"{live['snapshots']:,} score snapshots stored. "
            f"Refresh via `python sync_jikan.py --stale 30`, then re-run "
            f"`build_dataset.py` + `train.py`."
        )
    else:
        st.info(
            "Static base: Kaggle - CooperUnion (MyAnimeList scrape). "
            "No bulk refresh yet - run `python sync_jikan.py --top 200` to "
            "pull live scores/members from the free **Tenrai API** "
            "(api.tenrai.org, MyAnimeList data, no key needed), or try a "
            "single-title live fetch on the **8 - Live Feed** page."
        )
    a, b = cols(2)
    with a:
        src = C.RAW_ANIME if C.RAW_ANIME.exists() else C.CLEAN_ANIME
        st.markdown("**`anime.csv`** - 12,294 rows x 7 columns")
        st.dataframe(pd.read_csv(src, nrows=8), width="stretch", height=290)
        st.caption(f"read from `{src.parent.name}/{src.name}`")
    with b:
        st.markdown("**`ratings.csv`** - 7,813,737 rows x 3 columns")
        if C.RAW_RATINGS.exists():
            st.dataframe(pd.read_csv(C.RAW_RATINGS, nrows=8), width="stretch", height=290)
            st.caption(f"read from `{C.RAW_RATINGS.parent.name}/{C.RAW_RATINGS.name}`")
        else:
            st.dataframe(read_report("dataset_summary.csv"), width="stretch", height=290)
        st.caption("`rating = -1` means the user watched the title but did not "
                   "score it - treated as a missing value, not a rating. The 1 GB "
                   "raw file is not needed once the pipeline has run.")

    st.subheader("Where each mining stage lands")
    rows = []
    for stage, artefact, what in [
        ("Preprocessing", "data/processed/*.csv",
         "cleaned master, per-anime and per-user aggregates, 80/20 splits"),
        ("Association rules", "reports/association_rules_*.csv",
         "genre co-occurrence and watched-then-watched rules with lift"),
        ("Classification", "reports/j48_rules.csv, classification_comparison.csv",
         "J48 rule set + Naive Bayes + 5-fold model comparison"),
        ("Regression", "reports/regression_comparison.csv",
         "MAE / RMSE / R2 / MAPE for six regressors"),
        ("Clustering", "reports/cluster_*_profile.csv",
         "K-Means and Ward on the anime table and the user table"),
        ("Prediction", "models/*.joblib",
         "blended score regressor + hybrid recommender"),
    ]:
        rows.append({"stage": stage, "csv artefact": artefact, "contents": what})
    table(pd.DataFrame(rows), height=280)

    st.subheader("Headline results from the last pipeline run")
    mine = load_mining_summary()
    if mine:
        with cols(5):
            kpi("J48 accuracy", f"{float(REP.get(mine, 'stage3.j48.accuracy', 0)):.1%}",
                f"macro F1 {float(REP.get(mine, 'stage3.j48.f1_macro', 0)):.1%}")
            kpi("Best regressor", str(REP.get(mine, "stage4.best", "-")),
                f"RMSE {float(REP.get(mine, 'stage4.metrics[0].RMSE', 0)):.4f}")
            kpi("Anime clusters", str(REP.get(mine, "stage5.anime.k", "-")),
                f"k-means sil {float(REP.get(mine, 'stage5.anime.kmeans_silhouette', 0)):.3f}")
            kpi("User segments", str(REP.get(mine, "stage5.users.k", "-")),
                f"k-means sil {float(REP.get(mine, 'stage5.users.kmeans_silhouette', 0)):.3f}")
            kpi("Predictor", str(REP.get(mine, "stage6.best", "-")),
                f"held-out RMSE {float(REP.get(mine, 'stage6.sigma', 0)):.4f}")
        st.caption(f"read from `reports/mining_summary.csv` "
                   f"({len(mine):,} metrics). Re-run `python train.py` to refresh it.")


# ====================================================== 2 PREPROCESSING ====
if page == TABS[1]:
    st.title("Data Preprocessing")
    st.caption("Profiling, cleaning, type casting, outlier detection, "
               "normalisation, discretisation and train/test splitting.")

    rep = load_prep_report()
    anime = load_anime()
    raw_rows = int(REP.get(rep, "cleaning.raw_rows", len(anime)))
    filled = sum(int(REP.get(rep, f"cleaning.missing_filled.{k}", 0))
                 for k in ("genre", "type", "rating"))
    with cols(5):
        kpi("Rows in", f"{raw_rows:,}", "raw anime.csv")
        kpi("Rows out", f"{int(REP.get(rep, 'cleaning.out_rows', len(anime))):,}", "after cleaning")
        kpi("Missing filled", f"{filled:,}", "genre / type / rating")
        kpi("Episodes reparsed", f"{int(REP.get(rep, 'cleaning.episodes_unparsed', 0)):,}", "'Unknown' -> median by type")
        kpi("Duplicates removed", f"{int(REP.get(rep, 'cleaning.duplicate_ids_removed', 0)):,}", "on anime_id")

    st.subheader("1 - Raw profiling")
    if rep:
        miss = read_report("preprocessing_missing_values.csv")
        a, b = cols([1, 1.4])
        with a:
            table(miss, height=250)
        with b:
            fig = px.bar(miss, x="column", y="missing_pct", color="missing_pct",
                         color_continuous_scale="OrRd", title="Missing values per column (%)")
            fig.update_layout(showlegend=False, height=280, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")

    st.subheader("2 - Outlier detection (IQR fence rule)")
    out = P.detect_outliers(anime)
    a, b = cols([1.3, 1])
    with a:
        table(out, height=290)
    with b:
        fig = px.bar(out, x="attribute", y="pct_outliers", text="pct_outliers",
                     title="Share of rows beyond 1.5 x IQR",
                     color="skewness", color_continuous_scale="Viridis")
        fig.update_layout(height=290, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    st.caption("`members`, `episodes` and `genre_count` are strongly right-skewed, "
               "which is why a log transform is applied before scaling and why the "
               "outliers are kept but flagged in `is_outlier` rather than deleted.")

    st.subheader("3 - Feature engineering")
    a, b = cols(2)
    with a:
        st.markdown("**Derived attributes**")
        table(pd.DataFrame([
            {"feature": c, "dtype": str(anime[c].dtype),
             "example": str(anime[c].iloc[0])[:40]}
            for c in ["members_log", "episodes_log", "members_per_episode",
                      "genre_count", "primary_genre", "is_multi_genre",
                      "is_shounen", "is_shoujo", "is_seinen", "has_isekai",
                      "members_band", "episodes_band", "rating_tier"]
            if c in anime.columns]), height=430)
    with b:
        st.markdown("**Normalisation (min-max and z-score) + discretisation**")
        table(anime[[c for c in ["members", "members_log", "members_minmax",
                                 "members_zscore", "members_band", "rating",
                                 "rating_tier"] if c in anime.columns]].head(10),
              height=430)

    st.subheader("4 - Distribution of the discretised target")
    a, b = cols(2)
    with a:
        if "rating_tier" in anime.columns:
            cnt = anime["rating_tier"].value_counts().reindex(C.TIER_ORDER)
            fig = px.bar(x=C.TIER_ORDER, y=cnt.values, text=cnt.values,
                         labels={"x": "rating_tier class", "y": "anime"},
                         color=C.TIER_ORDER, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(showlegend=False, height=300, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
            st.caption("Class edges are the 30/30/20/20 quantiles of `rating`, so no "
                       "class is too small to learn.")
    with b:
        fig = px.histogram(anime, x="members", nbins=60, log_x=True,
                           title="Members (log scale) - the heavy right tail")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    st.subheader("5 - Modelling splits produced")
    if rep:
        cls_rows, reg_rows = "classification", "regression"
        table(pd.DataFrame([
            {"dataset": "classification_split.csv",
             "rows": int(REP.get(rep, f"modelling.{cls_rows}.rows", 0)),
             "features": int(REP.get(rep, f"modelling.{cls_rows}.features", 0)),
             "train": int(REP.get(rep, f"modelling.{cls_rows}.train", 0)),
             "test": int(REP.get(rep, f"modelling.{cls_rows}.test", 0)),
             "target": "rating_tier (4 classes)"},
            {"dataset": "regression_split.csv",
             "rows": int(REP.get(rep, f"modelling.{reg_rows}.rows", 0)),
             "features": int(REP.get(rep, f"modelling.{reg_rows}.features", 0)),
             "train": int(REP.get(rep, f"modelling.{reg_rows}.train", 0)),
             "test": int(REP.get(rep, f"modelling.{reg_rows}.test", 0)),
             "target": "rating (continuous 0-10)"},
        ]), height=200)


# ================================================== 3 ASSOCIATION RULES =====
if page == TABS[2]:
    st.title("Association Rule Mining")
    st.caption("Apriori and FP-Growth frequent itemset mining, then rule "
               "generation with support / confidence / lift and the extra metrics "
               "conviction, leverage, Zhang's metric and Kulczynski.")

    st.subheader("Rule base A - genre basket (transaction = anime, item = genre)")
    c1, c2, c3, c4 = cols(4)
    ms = c1.slider("min support", 0.01, 0.30, 0.04, 0.01,
                   help="share of anime that contain the whole itemset")
    mc = c2.slider("min confidence", 0.10, 0.90, 0.20, 0.05)
    ml = c3.slider("min lift", 1.0, 4.0, 1.2, 0.1, help=">1 means the rule is not independent")
    maxlen = c4.select_slider("max items per rule", options=[2, 3, 4], value=3)
    if c4.button("re-mine", key="remine_genre"):
        get_genre_rules.clear()

    gr = get_genre_rules(ms, mc, ml)
    gr = gr[gr["n_items"] <= maxlen] if not gr.empty else gr
    a, b = cols([1.6, 1])
    with a:
        st.markdown(f"**{len(gr)} genre rules**")
        table(gr, height=300)
        if not gr.empty:
            download(gr, "association_rules_genre.csv")
    with b:
        if not gr.empty:
            fig = px.scatter(gr, x="support", y="confidence", size="lift",
                             color="lift", hover_name="rule",
                             color_continuous_scale="Plasma",
                             title="Support vs confidence (bubble = lift)",
                             log_x=True)
            fig.add_hline(y=mc, line_dash="dash", line_color="red")
            fig.update_layout(height=330, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
            st.caption("Rules above the red line clear the confidence threshold; "
                       "bubble size is how much stronger than chance the rule is.")

    st.subheader("Rule base B - user basket (transaction = user, item = top-12 anime rated >= 8)")
    HAS_USER_BASKETS = C.BASKET_USER.exists()
    if not HAS_USER_BASKETS:
        st.warning(
            "`data/processed/baskets_user.csv` is missing - it is gitignored and "
            "needs the 1 GB `ratings.csv` to rebuild (`python build_dataset.py`). "
            "Showing the precomputed rule set instead; sliders and re-mining are "
            "disabled until the file exists."
        )
        ur = read_report("association_rules_user.csv")
        st.markdown(f"**{len(ur)} anime-to-anime rules (precomputed)** - these drive the "
                    "recommender in the prediction tab")
        table(ur, height=300)
        if not ur.empty:
            download(ur, "association_rules_user.csv")
    else:
        c1, c2, c3, c4 = cols(4)
        ums = c1.slider("min support", 0.01, 0.20, 0.03, 0.01, key="ums")
        umc = c2.slider("min confidence", 0.10, 0.90, 0.25, 0.05, key="umc")
        uml = c3.slider("min lift", 1.0, 8.0, 1.5, 0.5, key="uml")
        if c4.button("re-mine", key="remine_user"):
            get_user_rules.clear()
        ur = get_user_rules(ums, umc, uml)
        a, b = cols([1.6, 1])
        with a:
            st.markdown(f"**{len(ur)} anime-to-anime rules** - these drive the "
                        "recommender in the prediction tab")
            table(ur, height=300)
            if not ur.empty:
                download(ur, "association_rules_user.csv")
        with b:
            baskets = get_baskets()
            st.markdown("**Basket statistics**")
            kpi("users (transactions)", f"{len(baskets):,}")
            kpi("avg items / basket", f"{baskets['n'].mean():.1f}",
                f"max {baskets['n'].max()}")
            st.markdown("")
            fig = px.histogram(baskets, x="n", nbins=20, title="Items per basket")
            fig.update_layout(height=250, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")

    st.subheader("Apriori vs FP-Growth")
    st.caption("Both algorithms produce the identical itemset count; the "
               "difference is the number of database passes they need.")
    if not HAS_USER_BASKETS:
        st.warning("Live timing needs `baskets_user.csv` - showing the precomputed "
                   "benchmark instead.")
        bench = read_report("apriori_vs_fpgrowth.csv")
        table(bench, height=230)
    else:
        bench = A.apriori_vs_fpgrowth()
        a, b = cols([1.2, 1])
        with a:
            table(bench, height=230)
        with b:
            fig = px.bar(bench, x="min_support", y=["apriori_sec", "fpgrowth_sec"],
                         barmode="group", labels={"value": "seconds", "variable": "algorithm"},
                         title="Runtime by min_support (same itemsets)")
            fig.update_layout(height=280, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
    st.caption("At this basket size Apriori wins on wall clock; FP-Growth's "
               "advantage appears once the itemset count explodes, because it never "
               "generates a candidate set it has not already confirmed in the tree.")


# ===================================================== 4 CLASSIFICATION =====
if page == TABS[3]:
    st.title("Classification Rule Mining")
    st.caption("Predicting the discretised rating class (`rating_tier`) of an "
               "anime from its content attributes, with J48 and Naive Bayes.")

    Xtr, ytr, Xte = CL.load_classification_data()
    yte = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))["target"]
    a, b, c = cols(3)
    with a:
        st.markdown("**training set**")
        kpi("rows", f"{len(Xtr):,}", f"{Xtr.shape[1]} features")
        st.dataframe(pd.DataFrame({"class": ytr.value_counts().reindex(C.TIER_ORDER)}),
                     width="stretch", height=150)
    with b:
        st.markdown("**test set**")
        kpi("rows", f"{len(yte):,}", "stratified 80/20 split")
        st.dataframe(pd.DataFrame({"class": yte.value_counts().reindex(C.TIER_ORDER)}),
                     width="stretch", height=150)
    with c:
        st.markdown("**majority-class baseline**")
        maj = ytr.value_counts().max() / len(ytr)
        kpi("accuracy", f"{maj:.1%}", "any model must beat this")
        st.caption("Attributes are ordinal-encoded for the scikit-learn learners; "
                   "J48 consumes the raw frame and encodes it itself.")

    st.divider()
    st.subheader("A - J48 / C4.5 decision tree")
    st.markdown("C4.5 pre-discretises each continuous attribute into 10 quantile "
                "intervals, splits on **gain ratio** = information gain / split "
                "information, grows the tree, then prunes it bottom-up with the "
                "confidence factor before exporting the IF/THEN rules.")
    c1, c2, c3, c4 = cols(4)
    depth = c1.slider("max depth", 3, 12, 5)
    leaf = c2.slider("min samples / leaf", 2, 60, 10)
    conf = c3.slider("confidence factor", 0.0, 0.5, 0.25, 0.05,
                     help="higher prunes more aggressively")
    minconf = c4.slider("min rule confidence", 0.3, 0.95, 0.5, 0.05)

    j = get_j48(depth, leaf, conf)
    rules = j["rules"]
    rules = (rules[rules["confidence"] >= minconf] if not rules.empty else rules)
    a, b, c, d = cols(4)
    kpi("test accuracy", f"{j['accuracy']:.1%}", f"vs {maj:.1%} baseline")
    kpi("macro F1", f"{j['f1']:.1%}")
    kpi("tree", f"{j['stats']['depth']} levels", f"{j['stats']['internal_nodes']} splits")
    kpi("rules", f"{j['stats']['n_rules']}", f"{len(rules)} above {minconf:.0%} confidence")

    st.markdown("**Pruned tree as IF/THEN rules**")
    table(rules, height=280)
    if not rules.empty:
        download(rules, "j48_rules.csv")
    st.markdown("**Rendered tree**")
    st.markdown(f'<pre class="tree">{j["text"]}</pre>', unsafe_allow_html=True)

    st.markdown("**Gain ratio ranking** - which attribute the tree trusts most")
    imp = CL.feature_importance_j48(Xtr, ytr)
    a, b = cols([1, 1.2])
    with a:
        fig = px.bar(imp, x="mean_gain_ratio", y="attribute", orientation="h",
                     color="cumulative", color_continuous_scale="Blues",
                     title="Cumulative gain ratio")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.imshow(j["cm"], text_auto=True, color_continuous_scale="Blues",
                        labels=dict(x="predicted", y="actual", color="anime"),
                        title="Confusion matrix")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    st.divider()
    st.subheader("B - Naive Bayes")
    nb = get_nb()
    a, b, c, d = cols(4)
    kpi("Gaussian NB accuracy", f"{nb['gaussian_nb']['accuracy']:.1%}")
    kpi("Gaussian NB macro F1", f"{nb['gaussian_nb']['f1_macro']:.1%}")
    kpi("Categorical NB accuracy", f"{nb['categorical_nb']['accuracy']:.1%}")
    kpi("Categorical NB macro F1", f"{nb['categorical_nb']['f1_macro']:.1%}")

    st.markdown("**Class-conditional score distributions** - this is exactly what "
                "Gaussian NB multiplies together")
    # a log view of the popularity attribute: the continuous feature the
    # Gaussian NB class-conditional densities are actually estimated on
    pop_col = next((c for c in ("members_log", "members") if c in Xtr.columns),
                   Xtr.select_dtypes(include=[np.number]).columns[0])
    vals = np.log1p(Xtr[pop_col].astype(float)) if pop_col == "members" \
        else Xtr[pop_col].astype(float)
    mu, sd = float(vals.mean()), float(vals.std()) or 1.0
    z = (vals - mu) / sd
    yb = nb["label_encoder"].transform(ytr)
    fig = go.Figure()
    for i, cls in enumerate(nb["label_encoder"].classes_):
        sub = z.to_numpy()[yb == i]
        fig.add_trace(go.Histogram(
            x=sub, name=f"{cls} (mu={sub.mean():.2f}, sigma={sub.std():.2f})",
            opacity=0.55, nbinsx=45))
    label = f"log1p({pop_col})" if pop_col == "members" else pop_col
    fig.update_layout(barmode="overlay", xaxis_title=f"{label} (standardised)",
                      yaxis_title="anime", height=340, margin=dict(t=40, l=0, r=0))
    st.plotly_chart(fig, width="stretch")
    st.caption("Each trace is one class's estimated normal density argument. "
               "Gaussian NB assumes these are normal and independent given the "
               "class, then predicts argmax of P(class) x P(attribute | class).")

    a, b = cols(2)
    with a:
        fig = px.imshow(nb["cm_gnb"], text_auto=True, color_continuous_scale="Purples",
                        labels=dict(x="predicted", y="actual", color="anime"),
                        title="Gaussian NB confusion matrix")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.imshow(nb["cm_dnb"], text_auto=True, color_continuous_scale="Oranges",
                        labels=dict(x="predicted", y="actual", color="anime"),
                        title="Categorical NB confusion matrix")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Per-class precision / recall / F1**")
    rep = pd.DataFrame(nb["gaussian_nb"]["report"]).T.round(3)
    rep = rep.loc[[c for c in C.TIER_ORDER if c in rep.index]]
    table(rep.reset_index().rename(columns={"index": "class"}), height=230)

    st.divider()
    st.subheader("C - Head-to-head comparison (5-fold stratified CV)")
    cmp_ = get_compare()
    a, b = cols([1.15, 1])
    with a:
        table(cmp_["table"], height=290)
        download(cmp_["table"], "classification_comparison.csv")
    with b:
        fig = px.bar(cmp_["table"], x="model", y=["accuracy", "f1_macro"],
                     barmode="group", labels={"value": "score", "variable": "metric"},
                     title="Test accuracy and macro F1", text_auto=".3f")
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    st.markdown("**Where each model puts its predictions right**")
    fig = px.bar(cmp_["predictions"].melt(id_vars="actual", var_name="model",
                                           value_name="predicted"),
                 x="predicted", color="model", facet_col="model",
                 title="Predicted class distribution vs the actual distribution")
    fig.update_layout(height=280, margin=dict(t=40, l=0, r=0))
    st.plotly_chart(fig, width="stretch")


# ========================================================= 5 REGRESSION ====
if page == TABS[4]:
    st.title("Regression / Continuous Estimation")
    st.caption("Predicting the MyAnimeList community score (a continuous 0-10 "
               "value) from content attributes - the model behind the prediction widget.")
    r = get_regression()
    a, b, c, d = cols(4)
    best = r["table"].iloc[0]
    kpi("best model", str(best["model"]))
    kpi("RMSE", f"{best['RMSE']:.4f}", f"CV {best['cv_RMSE']:.4f}")
    kpi("R2", f"{best['R2']:.4f}")
    kpi("MAE", f"{best['MAE']:.4f}", f"MAPE {best['MAPE_%']:.2f}%")

    a, b = cols([1, 1.3])
    with a:
        st.markdown("**Model comparison**")
        table(r["table"], height=250)
        download(r["table"], "regression_comparison.csv")
    with b:
        fig = px.scatter(r["scatter"], x="actual", y="predicted",
                         color="residual" if "residual" in r["scatter"] else "actual",
                         color_continuous_scale="RdBu_r", opacity=0.6,
                         title="Actual vs predicted (best model)",
                         labels={"actual": "actual rating", "predicted": "predicted rating"})
        fig.add_shape(type="line", x0=1.5, y0=1.5, x1=10, y1=10,
                      line=dict(color="black", dash="dash"))
        fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    a, b, c = cols(3)
    with a:
        fig = px.histogram(r["residuals"], x="residual", nbins=50,
                           title="Residual distribution (0 = perfect)")
        fig.add_vline(x=0, line_color="black")
        fig.update_layout(height=260, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    with b:
        fig = px.box(r["residuals"], x="abs_residual", title="Absolute error spread")
        fig.update_layout(height=260, margin=dict(t=40, l=0, r=0), showlegend=False)
        st.plotly_chart(fig, width="stretch")
    with c:
        fig = px.line(r["learning_curve"], x="train_size", y=["train_RMSE", "test_RMSE"],
                      markers=True, title="Learning curve (gradient boosting)")
        fig.update_layout(height=260, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Error broken down by media type**")
    a, b = cols([1.1, 1])
    with a:
        table(r["by_type"], height=250)
    with b:
        fig = px.bar(r["by_type"], x="type", y="MAE", color="mean_actual",
                     title="MAE by media type", text_auto=".3f",
                     labels={"mean_actual": "mean actual rating"})
        fig.update_layout(height=280, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Feature importance (Random Forest)**")
    a, b = cols([1, 1.4])
    with a:
        fig = px.bar(r["importance"], x="importance", y="attribute", orientation="h",
                     color="cumulative", color_continuous_scale="Greens",
                     title="Cumulative importance")
        fig.update_layout(height=340, margin=dict(t=40, l=0, r=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Regressed tree (readable approximation)**")
        st.markdown(f'<pre class="tree">{r["tree_text"][:3500]}</pre>',
                    unsafe_allow_html=True)


# ========================================================= 6 CLUSTERING ====
if page == TABS[5]:
    st.title("Clustering - K-Means and Hierarchical, on Two Datasets")
    st.caption("Dataset A clusters the 12,294 anime by content. Dataset B clusters "
               "the 69,600 user profiles by rating behaviour. Both are clustered with "
               "K-Means and with Ward-linkage agglomerative clustering.")

    ds = st.radio("Dataset", list(CLC.DATASETS), horizontal=True)
    spec = CLC.DATASETS[ds]
    df, diag = get_kdiag(ds)
    auto_k, why = CLC.best_k(diag)

    a, b = cols([1, 1.4])
    with a:
        st.markdown(f"**{spec['source']}**")
        kpi("rows", f"{len(df):,}", f"{len(spec['features'])} clustered features")
        st.markdown("**Clustered attributes**")
        st.markdown(", ".join(f"`{f}`" for f in spec["features"]))
    with b:
        st.markdown("**Choosing k - elbow method + silhouette**")
        fig = px.line(diag, x="k", y=["inertia", "silhouette"], markers=True,
                      labels={"value": "value", "variable": "metric"},
                      title=f"k selection  ({why})")
        fig.update_layout(height=260, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    table(diag, height=230)

    c1, c2, c3 = cols([1, 1, 1.3])
    k = c1.slider("k (number of clusters)", 2, 8, int(auto_k))
    seed = c2.slider("random seed", 0, 100, C.RANDOM_STATE)
    sample = c3.slider("hierarchical sample size", 300, 2000, min(CLC.HIER_SAMPLE, 2000), 100,
                       help="Ward linkage is O(n^2) in memory, so it is fitted on a "
                            "random subsample (capped at 2000 for Cloud)")
    st.caption(f"`distance_to_chord` marks the elbow: the point furthest from the "
               f"straight line joining the first and last inertia value. "
               f"Selected k = **{auto_k}** ({why}).")

    km = get_kmeans(ds, k, seed)
    hc = get_hier(ds, k, sample)

    a, b, c, d = cols(4)
    kpi("K-Means silhouette", f"{km['silhouette']:.4f}")
    kpi("K-Means inertia", f"{km['inertia']:,.0f}")
    kpi("Hierarchical silhouette", f"{hc['silhouette']:.4f}", f"ward, n={hc['n_sample']:,}")
    kpi("largest cluster", f"{max(km['size'].values()):,}",
        f"of {len(df):,} rows")

    st.markdown("**A - K-Means**")
    a, b = cols([1.25, 1])
    with a:
        st.markdown("Cluster profile (centroid means, profiled back to a name)")
        table(km["named"], height=260)
    with b:
        name_of = dict(zip(km["named"]["cluster"], km["named"]["segment_name"]))
        _pidx = km.get("pca_idx", None)
        _labs = km["labels"][_pidx] if _pidx is not None else km["labels"]
        px_df = pd.DataFrame({
            "PC1": km["pca"][:, 0], "PC2": km["pca"][:, 1],
            "cluster": [f"C{int(c)} - {name_of.get(int(c), '')}" for c in _labs],
        })
        fig = px.scatter(px_df, x="PC1", y="PC2", color="cluster", opacity=0.5,
                         title=f"K-Means on 2 principal components (k={k}, "
                               f"{len(px_df):,} pts sampled)",
                         labels={"PC1": "PC1", "PC2": "PC2"})
        fig.update_layout(height=360, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")

    a, b = cols(2)
    with a:
        st.markdown("**Cluster sizes**")
        sizes = pd.DataFrame({"cluster": [f"C{c}" for c in km["size"]],
                              "rows": list(km["size"].values())})
        fig = px.bar(sizes, x="cluster", y="rows", text="rows", color="cluster",
                     title="K-Means cluster sizes")
        fig.update_layout(height=250, showlegend=False, margin=dict(t=40, l=0, r=0))
        st.plotly_chart(fig, width="stretch")
    with b:
        st.markdown("**Standardised cluster centroids** (the K-Means means in Z-space)")
        table(km["centers_raw"], height=250)

    st.divider()
    st.markdown("**B - Hierarchical (agglomerative, Ward linkage)**")
    from scipy.cluster.hierarchy import dendrogram
    st.caption("A dendrogram needs a linkage matrix, which the sample makes "
               "affordable. Cutting the tree at k clusters recovers the partitions below.")
    a, b = cols([1.15, 1])
    with a:
        Zl = hc["Z_linkage"]
        dn = dendrogram(Zl, no_labels=True, color_threshold=None,
                        above_threshold_color="#94a3b8")
        leaves = dn["ivl"]
        leaves = [int(x) for x in leaves]
        labels = hc["sample_labels"]
        fig = go.Figure()
        palette = px.colors.qualitative.Set2
        for c in range(k):
            members = [l for l in leaves if labels[l] == c]
            if members:
                fig.add_trace(go.Bar(x=members, y=[c] * len(members), name=f"cluster {c}",
                                     marker_color=palette[c % len(palette)]))
        fig.update_layout(barmode="stack", height=340, margin=dict(t=40, l=0, r=0),
                          xaxis_title="sample index (ordered by the dendrogram)",
                          yaxis_title="cluster", legend_title_text="")
        st.plotly_chart(fig, width="stretch")
        st.caption("The bar chart is the dendrogram read left to right: bars are "
                   "the leaves in merge order, stacked by the cluster they end up in. "
                   "A long run of one colour is a tight, well-separated cluster.")
    with b:
        st.markdown("**Hierarchical cluster profile**")
        table(hc["named"], height=250)
        st.markdown("**Merge distances of the last 12 joins**")
        tail = pd.DataFrame({
            "step": np.arange(len(Zl) - 12, len(Zl)),
            "distance": np.round(Zl[-12:, 2], 3),
            "child_left": Zl[-12:, 0].astype(int),
            "child_right": Zl[-12:, 1].astype(int),
        })
        table(tail, height=200)

    st.divider()
    st.markdown("**C - Cluster x class cube** (the OLAP roll-up over the discretised dimension)")
    if not km["cube"].empty:
        a, b = cols([1, 1.3])
        with a:
            table(km["cube"], height=250)
        with b:
            melted = km["cube"].melt(id_vars="cluster", var_name="band", value_name="rows")
            fig = px.bar(melted, x="band", y="rows", color="cluster", barmode="stack",
                         title="Cluster composition across the discretised dimension")
            fig.update_layout(height=300, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
    else:
        st.info("no discretised dimension available for this dataset")

    st.markdown("**D - Side-by-side verdict**")
    a, b = cols(2)
    verdict = pd.DataFrame([
        {"algorithm": "K-Means", "silhouette": round(km["silhouette"], 4),
         "rows clustered": len(df), "cost": "O(n x k x d) per iteration, linear memory",
         "shape": "flat, k partitions"},
        {"algorithm": "Hierarchical (Ward)", "silhouette": round(hc["silhouette"], 4),
         "rows clustered": hc["n_sample"], "cost": "O(n^2) memory, O(n^2 log n) time",
         "shape": "dendrogram, any k by cutting"},
    ])
    with a:
        table(verdict, height=200)
    with b:
        st.caption("K-Means scores marginally better here, which is the usual "
                   "outcome when the clusters really are convex and roughly equal "
                   "in density. Hierarchical clustering earns its place because it "
                   "returns the whole dendrogram - the k=2..8 partitions can all be "
                   "read off a single run, and unlike K-Means it is deterministic.")


# =================================================== 7 PREDICTION SYSTEM ====
if page == TABS[6]:
    st.title("Prediction System")
    st.caption("A trained model scores an anime you describe, and a hybrid "
               "recommender suggests what to watch next.")

    st.subheader("A - Anime score predictor")
    st.caption("The blend weight was fitted on a held-out validation split; both "
               "base learners were then refitted on the whole training set. The "
               "interval below is score +/- 1.96 x the held-out RMSE.")

    b = get_predictor()
    a, c = cols([1.2, 1])
    with a:
        with st.form("pred_form"):
            name = st.text_input("Title (informational only)", "Untitled anime")
            anime_type = st.selectbox("Media type", C.ANIME_TYPES, index=0)
            episodes = st.number_input("Episodes", min_value=1, max_value=5000,
                                       value=26, step=1,
                                       help="1 for a movie")
            members = st.number_input("Community members", min_value=0, max_value=5_000_000,
                                      value=250_000, step=10_000)
            genres = st.multiselect("Genres", C.ASSOC_GENRES, default=["Action", "Fantasy"])
            submitted = st.form_submit_button("Predict score", type="primary")
        c2, c3, c4 = cols(3)
        kpi("RMSE (held out)", f"{b.sigma:.3f}", f"R2 {b.bundle['r2_full']:.3f} in-sample")
        kpi("blend", f"{b.blend[0]:.2f} RF + {b.blend[1]:.2f} GB")
        kpi("training rows", f"{b.bundle['n_train']:,}")
    with c:
        st.markdown("**Model reference metrics**")
        m = pd.DataFrame(b.bundle["metrics"])
        m = m[["split", "model", "MAE", "RMSE", "R2", "MAPE_%"]]
        table(m, height=230)
        st.markdown("**Feature importance**")
        imp = pd.DataFrame({
            "attribute": b.feature_cols,
            "importance": b.models["Random Forest"].feature_importances_.round(4)})
        imp = imp.sort_values("importance", ascending=False)
        fig = px.bar(imp.head(8), x="importance", y="attribute", orientation="h",
                     title="What drives the predicted score")
        fig.update_layout(height=250, margin=dict(t=40, l=0, r=0),
                          yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, width="stretch")

    if submitted:
        out = b.predict(name=name, anime_type=anime_type, episodes=episodes,
                        members=members, genres=genres)
        a, b2, c2 = cols(3)
        kpi("predicted MAL score", f"{out['score']:.2f} / 10", f"class: {out['tier']}")
        kpi("95% interval", f"[{out['lower']:.2f}, {out['upper']:.2f}]",
            f"+/- 1.96 x {out['sigma']:.3f}")
        kpi("model spread", f"{out['random_forest']:.2f} / {out['gradient_boosting']:.2f}",
            "RF vs GB")
        fig = go.Figure(go.Scatter(
            x=[out["lower"], out["score"], out["upper"]], y=["score"] * 3,
            mode="markers", marker=dict(size=[6, 22, 6], color=["#94a3b8", "#4f46e5", "#94a3b8"]),
            error_x=dict(type="data", symmetric=False,
                         array=[out["upper"] - out["score"], 0],
                         arrayminus=[out["score"] - out["lower"], 0]),
            name="prediction"))
        for lo, hi, label in [(v[0], v[1], k) for k, v in C.RATING_TIERS.items()]:
            fig.add_vrect(x0=lo, x1=min(hi, 10), fillcolor="#e0e7ff", opacity=0.25,
                          line_width=0, annotation_text=label)
        fig.update_layout(height=250, xaxis_range=[3, 10], showlegend=False,
                          margin=dict(t=40, l=0, r=0),
                          xaxis_title="predicted rating", yaxis_visible=False)
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Input -> type={anime_type}, episodes={episodes}, "
                   f"members={members:,}, genres={', '.join(genres) or 'none'}")

    st.divider()
    st.subheader("B - Hybrid recommender")
    st.caption("Scoring blends the viewer's taste segment, that segment's "
               "per-genre affinity, the community prior, how good a title is for "
               "its own media type, a mild popularity preference, and the lift of "
               "any association rule that fires on what was already watched.")
    rec = get_recommender()

    mode = st.radio("Recommend by", ["an existing user", "my genre preferences",
                                     "titles I already watched"], horizontal=True)
    topn = st.slider("number of recommendations", 5, 30, 10)
    result = None
    if mode == "an existing user":
        vu = rec.valid_users()
        lookup = dict(zip(vu["user_id"], vu["label"]))
        pick = st.selectbox("Pick a user", vu["user_id"].tolist(),
                            format_func=lambda u: lookup.get(u, str(u)))
        prof = rec.profile_of(pick)
        seg = rec.cluster_of(pick)
        a, b2, c2, d = cols(4)
        kpi("user", str(pick), f"segment {seg}")
        kpi("anime rated", f"{int(prof.get('n_ratings', 0)):,}")
        kpi("mean rating given", f"{prof.get('mean_rating', 0):.2f}",
            f"sigma {prof.get('std_rating', 0):.2f}")
        kpi("like ratio", f"{prof.get('like_ratio', 0):.1%}",
            f"rated >= 8: {int(prof.get('n_liked', 0)):,}")
        try:
            result = rec.recommend_by_user(pick, topn)
        except KeyError as e:
            st.warning(str(e))
        a, b2 = cols([1, 1.3])
        with a:
            st.markdown(f"**What segment {seg} rates highly (vs the whole population)**")
            table(rec.segment_profile(seg), height=300)
        with b2:
            if result is not None and not result.empty:
                st.markdown("**Recommendations**")
                table(result, height=300)
                download(result, "recommendations_user.csv")
    elif mode == "my genre preferences":
        g = st.multiselect("Genres I like", C.ASSOC_GENRES,
                           default=["Action", "Adventure"])
        t = st.selectbox("Media type", ["Any"] + C.ANIME_TYPES)
        if g:
            result = rec.recommend_by_preferences(g, t, topn)
        else:
            st.info("pick at least one genre")
    else:
        titles = (rec.anime.sort_values("members", ascending=False)
                  .head(1500)["name"].drop_duplicates().tolist())
        titles = sorted(titles)
        w = st.multiselect("Titles already watched (1,500 most popular)",
                           titles, max_selections=8)
        if w:
            result = rec.recommend_by_watched(w, topn)
            st.caption("Items reached by an association rule are boosted; the "
                       "rules themselves are in the Association Rules tab.")
        else:
            st.info("pick at least one title")

    if result is not None and not result.empty:
        st.markdown("**Recommendation table**")
        table(result, height=340)
        download(result, "recommendations.csv")
        a, b2 = cols(2)
        with a:
            fig = px.bar(result, x="name", y="community_score", color="primary_genre",
                         title="Community score of each recommendation")
            fig.update_layout(height=320, margin=dict(t=40, l=0, r=0))
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, width="stretch")
        with b2:
            fig = px.bar(result, x="name", y="genre_affinity", color="match_score",
                         color_continuous_scale="Plasma",
                         title="Segment affinity vs overall match score")
            fig.add_hline(y=7.0, line_dash="dash", line_color="grey")
            fig.update_layout(height=320, margin=dict(t=40, l=0, r=0))
            fig.update_xaxes(tickangle=-45)
            st.plotly_chart(fig, width="stretch")
        if "genre_affinity" in result.columns:
            st.caption("`genre_affinity` is the mean rating this viewer's segment "
                       "gives to the item's primary genre; the dashed line at 7.0 is "
                       "the population average. `match_score` is the weighted blend "
                       "described above.")

    st.divider()
    st.subheader("C - Retrain the prediction models")
    st.caption("Useful for demonstrating that the models are genuinely fitted here, "
               "not downloaded.")
    if st.button("Retrain score predictor + recommender"):
        progress = st.progress(0, text="training the score regressor ...")
        with st.spinner("training ..."):
            res = PRED.train_rating_predictor()
            progress.progress(0.6, text="fitting user segments and affinity ...")
            PRED.train_recommender()
            progress.progress(1.0, text="done")
        get_predictor.clear()
        get_recommender.clear()
        st.success(f"done - best model {res['best']}, RMSE {res['sigma']:.4f}")
        st.dataframe(res["metrics"], width="stretch")
        st.rerun()


# ======================================================== 8 LIVE FEED =======
if page == TABS[7]:
    st.title("Live Feed - Tenrai API (MyAnimeList)")
    st.caption("The batch pipeline trains on `data/raw/anime.csv`. The live feed "
               "refreshes that file from MyAnimeList through the free Tenrai API "
               "and the weekly retrain rebuilds + retrains on the fresher input. "
               "Fetching a row is read-only - the Sync button under the diff is "
               "what writes `anime.csv` and appends to the score history.")

    try:
        _names = (pd.read_csv(C.RAW_ANIME, usecols=["anime_id", "name"])
                  .set_index("anime_id")["name"].to_dict())
    except Exception:
        _names = {}

    def _label(mid: int) -> str:
        return f"{mid} - {_names.get(int(mid), 'unknown title')}"

    live = live_feed_status()
    log = read_report("retrain_log.csv")
    last = log.iloc[-1].to_dict() if not log.empty else {}
    with cols(3):
        kpi("titles refreshed", f"{live['synced_titles']:,}",
            f"last sync {live['last_sync'] or 'never'}")
        kpi("score snapshots", f"{live['snapshots']:,}",
            "data/live/score_history.csv")
        if last:
            kpi("last weekly run", str(last.get("run_at", ""))[:10],
                f"stages {last.get('stages_ok', '-')} in {last.get('duration_s', '?')}s")
        else:
            kpi("weekly retrain", "not run yet", "python retrain_weekly.py")

    st.subheader("A - Score history (what the retrain learns from)")
    if C.LIVE_HISTORY.exists():
        hist = pd.read_csv(C.LIVE_HISTORY, parse_dates=["fetched_at"])
        ids = sorted(hist["anime_id"].unique().tolist())
        pick = st.selectbox("Title", ids, format_func=_label)
        sub = hist[hist["anime_id"] == pick].sort_values("fetched_at")
        a, b = cols(2)
        with a:
            fig = px.line(sub, x="fetched_at", y="rating", markers=True,
                          title="Community score over syncs")
            fig.update_layout(height=280, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
        with b:
            fig = px.line(sub, x="fetched_at", y="members", markers=True,
                          title="Community members over syncs")
            fig.update_layout(height=280, margin=dict(t=40, l=0, r=0))
            st.plotly_chart(fig, width="stretch")
        table(sub.sort_values("fetched_at", ascending=False), height=250)
        download(sub, "score_history_title.csv")
    else:
        st.info("No snapshots yet - run `python sync_jikan.py --top 200` to pull "
                "live scores/members, then revisit this tab for the trend.")

    st.subheader("B - Live row demo (fetch one title from Tenrai)")
    st.caption("Fetches `GET /anime/{id}`, maps it onto the 7-column anime.csv "
               "schema, and diffs it against your local row. Repeat fetches are "
               "served from the JSON cache under `data/live/cache/` (gitignored).")
    c1, c2 = cols([1, 2])
    mid = c1.number_input("MyAnimeList id", min_value=1, value=1, step=1,
                          help="anime_id IS the MyAnimeList id, so rows join 1:1")
    nocache = c2.checkbox("ignore JSON cache, refetch from Tenrai",
                          help="bypasses data/live/cache/ for this fetch")
    _cached = (C.LIVE_CACHE_DIR / f"{int(mid)}.json").exists()
    if nocache:
        st.caption("Cache bypassed - this click makes a live network call "
                   "(~1-3 s normally, up to ~25 s if the API is slow). "
                   "Untick for the instant cached read.")
    elif _cached:
        st.caption("Cached locally - this fetch is instant.")
    else:
        st.caption("Not cached yet - first fetch makes one live network call "
                   "(~1-3 s), then it is cached.")
    # interactive client: short timeout + single retry so a slow API
    # fails fast with a warning instead of hanging the spinner.
    _client = LF.JikanClient(timeout=12, max_retries=1)
    if st.button("Fetch live row", type="primary"):
        with st.spinner(f"fetching anime {int(mid)} from api.tenrai.org ..."):
            try:
                payload = _client.fetch(int(mid), use_cache=not nocache)
                norm = LF.normalize(payload)
            except Exception as e:  # noqa: BLE001 - show, don't crash the page
                st.warning(f"API request failed ({type(e).__name__}: {e}). "
                           "Check your connection and retry - nothing was written.")
                payload, norm = None, None
        if norm is not None:
            # stash in session state so the diff + sync buttons below survive
            # reruns (a clicked button resets to False on the next run).
            st.session_state["live_demo"] = {"mid": int(mid), "payload": payload,
                                             "norm": norm}
    demo = st.session_state.get("live_demo")
    if demo is not None:
        _dmid, payload, norm = demo["mid"], demo["payload"], demo["norm"]
        if _dmid != int(mid):
            st.caption(f"showing last fetched id {_dmid} - change the id above "
                       "and fetch again to refresh.")
        local = pd.read_csv(C.RAW_ANIME)
        hit = local[local["anime_id"] == _dmid]

        def _show(v) -> str:
            # plain strings only: mixed numpy scalars in an object
            # column break Arrow serialization in st.dataframe.
            if v is None or (not isinstance(v, str) and pd.isna(v)):
                return "-"
            if isinstance(v, np.generic):
                v = v.item()
            return str(v)

        rows = []
        for f in ("genre", "type", "episodes", "rating", "members"):
            lv = hit.iloc[0][f] if not hit.empty else "-"
            rv = norm["row"][f] if norm["row"][f] is not None else "-"
            rows.append({"field": f, "local anime.csv": _show(lv),
                         "live Tenrai": _show(rv),
                         "would update": "yes" if (not hit.empty and str(lv) != str(rv)
                                                   and LF._valid(f, norm["row"][f])) else "-"})
        st.markdown(f"**{_label(_dmid)}**")
        table(pd.DataFrame(rows), height=230)
        a, b2 = cols(2)
        kpi("scored_by (voters)", f"{norm['extras'].get('scored_by') or '-'}",
            "history-only field, never in anime.csv")
        kpi("favorites", f"{norm['extras'].get('favorites') or '-'}",
            "history-only field, never in anime.csv")
        _msg = st.session_state.pop("live_sync_msg", None)
        if _msg:
            st.success(_msg)
        if st.button("Sync this title (update anime.csv + score history)",
                     key="sync_live_title"):
            with st.spinner(f"syncing anime {_dmid} ..."):
                _name = (payload.get("title_english") or payload.get("title")
                         or f"MAL {_dmid}")
                rep = LF.sync([_dmid], client=_client, allow_new=True,
                              name_overrides={_dmid: str(_name)})
            if rep["changed"]:
                st.session_state["live_sync_msg"] = (
                    f"synced - {', '.join(sorted(rep['fields']))} updated, "
                    f"snapshot appended to data/live/score_history.csv. "
                    f"Re-run `python build_dataset.py && python train.py` "
                    f"to retrain on it.")
            else:
                st.session_state["live_sync_msg"] = (
                    "already up to date - snapshot appended, nothing changed.")
            st.rerun()
        st.caption("Syncing appends a snapshot row and refreshes the trend in "
                   "section A above. Retrain after with "
                   "`python build_dataset.py && python train.py` - or "
                   "`python retrain_weekly.py --skip-sync` for all stages.")
        with st.expander("raw API payload (trimmed)"):
                trim = {k: payload.get(k) for k in
                        ("mal_id", "title", "title_english", "type", "episodes",
                         "status", "score", "scored_by", "members", "favorites",
                         "year", "season")}
                trim["genres"] = [(g or {}).get("name")
                                  for g in (payload.get("genres") or [])]
                st.json(trim)
