"""Run every DWM stage and persist the result tables as CSV.

    python train.py            # all stages
    python train.py 3 4        # only association rule mining + classification

Stages
    1 preprocessing   (delegates to build_dataset.py when artefacts are missing)
    2 association     genre rules + user basket rules + Apriori vs FP-Growth
    3 classification  J48 rules, Naive Bayes, model comparison
    4 regression      continuous estimation of the MAL score
    5 clustering      K-Means + hierarchical on both datasets
    6 prediction      rating predictor + hybrid recommender
"""
from __future__ import annotations

import sys
import time
import warnings

import pandas as pd

from core import association as A
from core import classification as CL
from core import clustering as CLC
from core import config as C
from core import predictor as PRED
from core import regression as R
from core import reporting as REP

warnings.filterwarnings("ignore")


def hr(title: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def ensure_dataset() -> None:
    if not C.CLS_SPLIT.exists() or not C.USER_FEATURES.exists():
        hr("STAGE 1 - DATA PREPROCESSING (rebuilding)")
        import build_dataset
        build_dataset.main()


# ------------------------------------------------------------- 2. ARM ------
def stage_association() -> dict:
    hr("STAGE 2 - ASSOCIATION RULE MINING")
    t = time.time()
    genre = A.mine_genre_rules()
    print(f"genre rules        : {len(genre):>5}  ({time.time() - t:.1f}s)")
    if len(genre):
        print(genre.head(5)[["rule", "support", "confidence", "lift"]]
              .to_string(index=False))
    t = time.time()
    user = A.mine_user_rules()
    print(f"\nuser basket rules  : {len(user):>5}  ({time.time() - t:.1f}s)")
    if len(user):
        print(user.head(5)[["rule", "support", "confidence", "lift"]]
              .to_string(index=False))
    bench = A.apriori_vs_fpgrowth()
    bench.to_csv(C.REPORT_DIR / "apriori_vs_fpgrowth.csv", index=False)
    print("\nApriori vs FP-Growth:")
    print(bench.to_string(index=False))
    return {"genre_rules": len(genre), "user_rules": len(user),
            "best_genre_lift": float(genre["lift"].max()) if len(genre) else None,
            "best_user_lift": float(user["lift"].max()) if len(user) else None}


# ------------------------------------------------- 3. CLASSIFICATION -------
def stage_classification() -> dict:
    hr("STAGE 3a - CLASSIFICATION WITH J48 (C4.5)")
    Xtr, ytr, Xte = CL.load_classification_data()
    yte = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))["target"]
    t = time.time()
    j = CL.run_j48(Xtr, ytr, Xte, yte)
    print(f"J48   accuracy {j['j48']['accuracy']:.4f}   f1 {j['j48']['f1_macro']:.4f}"
          f"   tree {j['j48_stats']['depth']} levels, {j['j48_stats']['n_rules']} rules"
          f"   ({time.time() - t:.1f}s)")
    print(f"sklearn entropy tree accuracy {j['sklearn_entropy_tree']['accuracy']:.4f}")
    if len(j["rules"]):
        print("\ntop J48 rules:")
        print(j["rules"].head(5)[["conditions", "conclusion", "confidence"]]
              .to_string(index=False))
    (C.REPORT_DIR / "j48_tree.txt").write_text(
        j["j48_tree_text"], encoding="utf-8")
    (C.REPORT_DIR / "sklearn_tree.txt").write_text(
        j["sklearn_tree_text"], encoding="utf-8")
    pd.DataFrame(j["j48"]["confusion_matrix"],
                 index=[f"actual {c}" for c in j["j48"]["classes"]],
                 columns=[f"pred {c}" for c in j["j48"]["classes"]]).to_csv(
        C.REPORT_DIR / "j48_confusion_matrix.csv")

    hr("STAGE 3b - CLASSIFICATION WITH NAIVE BAYES")
    nb = CL.run_naive_bayes(Xtr, ytr, Xte, yte)
    print(f"Gaussian NB    accuracy {nb['gaussian_nb']['accuracy']:.4f}"
          f"   f1 {nb['gaussian_nb']['f1_macro']:.4f}")
    print(f"Categorical NB accuracy {nb['categorical_nb']['accuracy']:.4f}"
          f"   f1 {nb['categorical_nb']['f1_macro']:.4f}")
    nb["predictions"].to_csv(C.REPORT_DIR / "naive_bayes_predictions.csv", index=False)

    hr("STAGE 3c - MODEL COMPARISON (5-fold CV)")
    cmp_ = CL.compare_models(Xtr, ytr, Xte, yte)
    print(cmp_["table"].to_string(index=False))
    print(f"\nbest by test accuracy: {cmp_['best']}")
    cmp_["table"].to_csv(C.REPORT_DIR / "classification_comparison.csv", index=False)
    cmp_["predictions"].to_csv(C.REPORT_DIR / "classification_predictions.csv", index=False)
    imp = CL.feature_importance_j48(Xtr, ytr)
    imp.to_csv(C.REPORT_DIR / "j48_gain_ratio_ranking.csv", index=False)
    print("\nJ48 gain-ratio ranking (information gain ratios summed per attribute):")
    print(imp.head(8).to_string(index=False))
    return {"j48": j["j48"], "gaussian_nb": nb["gaussian_nb"],
            "categorical_nb": nb["categorical_nb"],
            "comparison": cmp_["table"].to_dict("records"),
            "j48_stats": j["j48_stats"], "best": cmp_["best"],
            "best_accuracy": round(float(cmp_["table"]["accuracy"].max()), 4)}


# ----------------------------------------------------- 4. REGRESSION -------
def stage_regression() -> dict:
    hr("STAGE 4 - REGRESSION / CONTINUOUS ESTIMATION")
    Xtr, ytr, Xte, yte = R.load_regression_data()
    t = time.time()
    r = R.run_regression(Xtr, ytr, Xte, yte)
    print(r["table"].to_string(index=False))
    print(f"\nbest: {r['best_model']}   ({time.time() - t:.1f}s)")
    r["table"].to_csv(C.REPORT_DIR / "regression_comparison.csv", index=False)
    r["scatter"].to_csv(C.REPORT_DIR / "regression_actual_vs_predicted.csv", index=False)
    r["residuals"].to_csv(C.REPORT_DIR / "regression_residuals.csv", index=False)
    r["by_type"].to_csv(C.REPORT_DIR / "regression_by_type.csv", index=False)
    r["learning_curve"].to_csv(C.REPORT_DIR / "regression_learning_curve.csv", index=False)
    imp = R.feature_importance(Xtr, ytr)
    imp.to_csv(C.REPORT_DIR / "regression_importance.csv", index=False)
    print("\nerror by media type:")
    print(r["by_type"].to_string(index=False))
    return {"best": r["best_model"],
            "metrics": r["table"].to_dict("records"),
            "importance": imp.head(10).to_dict("records")}


# ------------------------------------------------------ 5. CLUSTERING ------
def stage_clustering() -> dict:
    hr("STAGE 5 - CLUSTERING: K-MEANS AND HIERARCHICAL, TWO DATASETS")
    out = {}
    for key in CLC.DATASETS:
        spec = CLC.DATASETS[key]
        df = CLC.load_dataset(key)
        Z, cols, _ = CLC.scaled_matrix(df, key)
        diag = CLC.k_selection(Z)
        k, why = CLC.best_k(diag)
        print(f"\n--- {key}   {spec['source']}   rows={len(df):,}")
        print(diag[["k", "inertia", "silhouette", "calinski_harabasz",
                    "davies_bouldin"]].to_string(index=False))
        print(f"selected k = {k}   ({why})")

        km = CLC.kmeans_cluster(df, key, k)
        hc = CLC.hierarchical_cluster(df, key, k)
        km_prof = CLC.name_clusters(km["profile"], key)
        hc_prof = CLC.name_clusters(hc["profile"], key)
        cube = CLC.cluster_vs_class(df, key, km["labels"])
        print(f"\nK-Means     silhouette {km['silhouette']:.4f}  "
              f"inertia {km['inertia']:.0f}  sizes {km['size']}")
        print(km_prof[["cluster", "n", "segment_name"]].to_string(index=False))
        print(f"\nHierarchical (ward, n={hc['n_sample']:,} sample)  "
              f"silhouette {hc['silhouette']:.4f}")
        print(hc_prof[["cluster", "n", "segment_name"]].to_string(index=False))
        if not cube.empty:
            print("\ncluster x class cube (OLAP roll-up):")
            print(cube.to_string(index=False))

        tag = "anime" if key.startswith("A") else "users"
        diag.to_csv(C.REPORT_DIR / f"cluster_{tag}_k_selection.csv", index=False)
        km_prof.to_csv(C.REPORT_DIR / f"cluster_{tag}_kmeans_profile.csv", index=False)
        hc_prof.to_csv(C.REPORT_DIR / f"cluster_{tag}_hierarchical_profile.csv", index=False)
        km["centers_raw"].to_csv(C.REPORT_DIR / f"cluster_{tag}_kmeans_centroids.csv")
        cube.to_csv(C.REPORT_DIR / f"cluster_{tag}_cube.csv", index=False)
        out[tag] = {"k": k, "why": why, "rows": int(len(df)),
                    "kmeans_silhouette": round(km["silhouette"], 4),
                    "hier_silhouette": round(hc["silhouette"], 4),
                    "sizes": {str(a): int(b) for a, b in km["size"].items()}}
    return out


# ------------------------------------------------------ 6. PREDICTION ------
def stage_prediction() -> dict:
    hr("STAGE 6 - PREDICTION SYSTEM (model training + inference)")
    t = time.time()
    r = PRED.train_rating_predictor()
    print(f"rating predictor  : {r['best']}  blend={r['blend']}  "
          f"RMSE={r['sigma']}  R2(full)={r['r2_full']}  ({time.time() - t:.1f}s)")
    print(r["metrics"].to_string(index=False))
    print("\nfeature importance:")
    print(r["importance"].head(8).to_string(index=False))
    t = time.time()
    rec = PRED.train_recommender()
    print(f"\nrecommender       : {rec['n_anime']:,} items, {rec['n_users']:,} users, "
          f"{rec['k']} segments, {rec['rules']} rules  ({time.time() - t:.1f}s)")

    p = PRED.RatingPredictor()
    demo = p.predict(name="Demo", anime_type="TV", episodes=26, members=250_000,
                     genres=["Action", "Adventure", "Fantasy", "Shounen"])
    print(f"\nsmoke test -> predicted MAL score {demo['score']} "
          f"({demo['tier']}), 95% interval [{demo['lower']}, {demo['upper']}]")
    return {"predictor": r["metrics"].to_dict("records"), "best": r["best"],
            "blend": r["blend"], "sigma": r["sigma"], "r2_full": r["r2_full"],
            "recommender": {k: v for k, v in rec.items() if k != "affinity"}}


STAGES = {2: stage_association, 3: stage_classification, 4: stage_regression,
          5: stage_clustering, 6: stage_prediction}


def main() -> None:
    C.ensure_dirs()
    ensure_dataset()
    wanted = [int(a) for a in sys.argv[1:]] or sorted(STAGES)
    # keep the metrics of stages that are not re-run, so a partial run
    # (e.g. `python train.py 6`) does not wipe the rest of the summary
    path = C.REPORT_DIR / "mining_summary.csv"
    summary = REP.read_summary(path) if path.exists() else {}
    for n in wanted:
        if n not in STAGES:
            print(f"unknown stage {n}; choose from {sorted(STAGES)}")
            continue
        res = STAGES[n]()
        for k, v in res.items():
            summary[f"stage{n}.{k}"] = v
    REP.write_summary(path, summary)
    hr("ALL REQUESTED STAGES COMPLETE - tables written to reports/")


if __name__ == "__main__":
    main()
