"""Run the full DWM pipeline once: clean -> features -> baskets -> splits.

    python build_dataset.py
"""
from __future__ import annotations

import time

from core import config as C
from core import preprocessing as P
from core import reporting as REP


def main() -> None:
    C.ensure_dirs()
    t0 = time.time()
    report: dict = {}

    print("[1/6] profiling & cleaning anime.csv ...")
    anime, profile = P.clean_anime()
    report["cleaning"] = profile

    print("[2/6] outlier detection (IQR fences) ...")
    out = P.detect_outliers(anime)
    out.to_csv(C.REPORT_DIR / "preprocessing_outliers.csv", index=False)
    report["outliers"] = out.to_dict("records")
    anime = P.add_outlier_flags(anime)
    anime.to_csv(C.CLEAN_ANIME, index=False)

    print("[3/6] aggregating 7.8M ratings (per-anime + per-user) ...")
    astats, ufeat, meta = P.build_ratings_tables(anime)
    report["ratings"] = meta

    print("[4/6] building transaction baskets ...")
    gb = P.build_genre_baskets(anime)
    ub = P.build_user_baskets()
    report["baskets"] = {"genre": int(len(gb)), "user": int(len(ub))}

    print("[5/6] scaling + discretisation + train/test splits ...")
    report["modelling"] = P.build_modelling_frames(anime)

    print("[6/6] summary tables ...")
    import pandas as pd

    summary = pd.DataFrame([
        {"table": "anime_clean", "rows": len(anime),
         "description": "cleaned + feature-engineered anime master"},
        {"table": "anime_rating_stats", "rows": len(astats),
         "description": "per-anime aggregated user ratings"},
        {"table": "user_features", "rows": len(ufeat),
         "description": "per-user behavioural profile (clustering dataset B)"},
    ])
    summary.to_csv(C.REPORT_DIR / "dataset_summary.csv", index=False)

    # raw missing-value profile as its own table, then the flattened summary
    miss = pd.DataFrame(
        [{"column": k, "missing": v,
          "missing_pct": round(100 * v / profile["raw_rows"], 3)}
         for k, v in profile["raw_missing"].items()])
    miss.to_csv(C.REPORT_DIR / "preprocessing_missing_values.csv", index=False)

    REP.write_summary(C.REPORT_DIR / "preprocessing_report.csv", report)

    print("\n" + "=" * 62)
    print(f"  anime  rows : {profile['out_rows']:,}")
    print(f"  users  rows : {len(ufeat):,}")
    print(f"  ratings     : {meta['scored_interactions']:,} scored / "
          f"{meta['watched_not_rated']:,} unrated")
    print(f"  class split : {report['modelling']['classification']['train']} train / "
          f"{report['modelling']['classification']['test']} test")
    print(f"  reg   split : {report['modelling']['regression']['train']} train / "
          f"{report['modelling']['regression']['test']} test")
    print(f"  done in {time.time() - t0:.1f}s")
    print("=" * 62)


if __name__ == "__main__":
    main()
