"""Weekly retrain driver: Jikan live sync -> rebuild -> retrain -> log.

Chains the existing pieces without duplicating their logic:

    Jikan API --sync_jikan--> data/raw/anime.csv
              --build_dataset--> data/processed/*.csv
              --train.py -------> reports/*.csv, models/*.joblib

Designed for a weekly cron (see .github/workflows/weekly-retrain.yml) but
equally runnable by hand::

    python retrain_weekly.py                    # weekly default (stale 7d)
    python retrain_weekly.py --top 200          # refresh 200 most-membered
    python retrain_weekly.py --stages 6         # only the predictor stage
    python retrain_weekly.py --skip-sync        # retrain on current CSVs
    python retrain_weekly.py --dry-run          # select ids only, no I/O
    python retrain_weekly.py --force             # retrain even if 0 changed

Robustness rules (CI has no 1 GB ratings.csv and no baskets_user.csv):

* if ``data/raw/ratings.csv`` is missing, only anime-derived tables are
  rebuilt (clean + genre baskets + modelling splits); user aggregates and
  baskets are preserved from the previous run;
* each train stage runs in isolation - one stage failing (e.g. user-basket
  rules without baskets_user.csv, recommender affinity without ratings.csv)
  is recorded in the log and the remaining stages still run;
* if Jikan reports fewer than ``--min-changed`` updated titles the expensive
  rebuild is skipped (override with ``--force``) so quiet weeks cost nothing.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time

import pandas as pd

from core import config as C

RETRAIN_LOG = C.REPORT_DIR / "retrain_log.csv"
LOG_COLUMNS = ["run_at", "mode", "selected", "fetched", "changed", "failed",
               "skipped", "stages", "stages_ok", "duration_s",
               "predictor_best", "predictor_rmse", "recommender_users",
               "git_sha"]


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def append_log(row: dict) -> None:
    C.ensure_dirs()
    df = pd.DataFrame([{c: row.get(c, "") for c in LOG_COLUMNS}])
    hdr = not RETRAIN_LOG.exists()
    df.to_csv(RETRAIN_LOG, mode="a", header=hdr, index=False)


def rebuild_anime_derived() -> dict:
    """Rebuild everything derivable from anime.csv alone (no ratings.csv).

    Returns a small summary dict. User aggregates (user_features.csv,
    anime_rating_stats.csv, baskets_user.csv) are left untouched.
    """
    from core import preprocessing as P

    anime, profile = P.clean_anime()
    out = P.detect_outliers(anime)
    out.to_csv(C.REPORT_DIR / "preprocessing_outliers.csv", index=False)
    anime = P.add_outlier_flags(anime)
    anime.to_csv(C.CLEAN_ANIME, index=False)
    gb = P.build_genre_baskets(anime)
    modelling = P.build_modelling_frames(anime)
    return {"anime_rows": int(len(anime)), "genre_baskets": int(len(gb)),
            "modelling": modelling}


def run_stages(wanted: list[int]) -> dict:
    """Run train stages one by one, isolating failures. Returns per-stage info."""
    import train as T

    C.ensure_dirs()
    from core import reporting as REP
    path = C.REPORT_DIR / "mining_summary.csv"
    summary = REP.read_summary(path) if path.exists() else {}
    info: dict = {}
    for n in wanted:
        if n not in T.STAGES:
            info[str(n)] = "unknown-stage"
            continue
        t0 = time.time()
        try:
            res = T.STAGES[n]()
            for k, v in res.items():
                summary[f"stage{n}.{k}"] = v
            info[str(n)] = f"ok ({time.time() - t0:.0f}s)"
        except Exception as e:  # noqa: BLE001 - weekly job must continue
            info[str(n)] = f"FAILED: {type(e).__name__}: {e}"[:200]
    REP.write_summary(path, summary)
    return info


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Weekly Jikan retrain.")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--ids", nargs="+", help="MAL ids, e.g. --ids 1 5 20")
    src.add_argument("--top", type=int, help="top N titles by members")
    src.add_argument("--stale", type=int, default=7, metavar="DAYS",
                     help="ids not synced in the last DAYS (default: 7)")
    src.add_argument("--all", action="store_true", help="whole catalog")
    ap.add_argument("--limit", type=int, default=500,
                    help="cap number of ids fetched (default: 500)")
    ap.add_argument("--stages", nargs="+", type=int, default=[2, 3, 4, 5, 6],
                    help="train stages to run (default: 2 3 4 5 6)")
    ap.add_argument("--min-changed", type=int, default=1,
                    help="skip retrain if fewer titles changed (default: 1)")
    ap.add_argument("--force", action="store_true",
                    help="retrain even if nothing changed")
    ap.add_argument("--skip-sync", action="store_true",
                    help="skip Jikan sync, retrain on current CSVs")
    ap.add_argument("--dry-run", action="store_true",
                    help="select ids only; no network calls, no writes")
    ap.add_argument("--allow-new", action="store_true",
                    help="append Jikan titles missing from anime.csv")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached Jikan JSON, refetch everything")
    args = ap.parse_args(argv)

    t0 = time.time()
    import sync_jikan as SJ
    from core import live_feed as LF

    # ---------------------------------------------------------- 1. sync ----
    sync_rep: dict = {"fetched": 0, "changed": 0, "failed": {}}
    selected: list[int] = []
    if args.skip_sync:
        mode = "retrain-only (no sync)"
        print("skip-sync: using current data/raw/anime.csv as-is.")
    else:
        # reuse sync_jikan's selection so flags behave identically
        # (default weekly path: anything not refreshed in --stale days)
        ns = argparse.Namespace(ids=args.ids, top=args.top, stale=args.stale,
                                all=args.all, limit=args.limit)
        selected = SJ.select_ids(ns)
        print(f"selected {len(selected)} ids"
              + (f" (first 10: {selected[:10]})" if selected else ""))
        if args.dry_run or not selected:
            print("dry run: no network calls, no writes.")
            return 0
        client = LF.JikanClient()
        sync_rep = LF.sync(selected, client=client, allow_new=args.allow_new,
                           use_cache=not args.no_cache)
        print(f"fetched={sync_rep['fetched']} changed={sync_rep['changed']} "
              f"calls={sync_rep.get('calls_made')} "
              f"cache_hits={sync_rep.get('cache_hits')}")
        if sync_rep.get("fields"):
            print("fields updated:", sync_rep["fields"])
        if sync_rep.get("failed"):
            print(f"failed ({len(sync_rep['failed'])}), first few:")
            for k, v in list(sync_rep["failed"].items())[:10]:
                print(f"  {k}: {v}")
        mode = (f"sync(stale={args.stale}d,limit={args.limit})"
                if not (args.ids or args.top or args.all)
                else f"sync(ids={len(selected)})")

    changed = int(sync_rep.get("changed", 0)) if not args.skip_sync else 10**9
    if not args.skip_sync and not args.force and changed < args.min_changed:
        dur = round(time.time() - t0, 1)
        print(f"only {changed} title(s) changed (< min {args.min_changed}) - "
              "skipping rebuild/retrain. Use --force to override.")
        append_log({"run_at": LF.utcnow_iso(), "mode": mode,
                    "selected": len(selected), "fetched": sync_rep.get("fetched", 0),
                    "changed": changed, "failed": len(sync_rep.get("failed", {})),
                    "skipped": True, "stages": "", "stages_ok": "",
                    "duration_s": dur, "git_sha": git_sha()})
        return 0

    # -------------------------------------------------------- 2. rebuild ---
    print("\n== rebuild ==")
    if C.RAW_RATINGS.exists():
        import build_dataset
        build_dataset.main()
        rebuilt = "full (ratings.csv present)"
    else:
        info = rebuild_anime_derived()
        print(f"ratings.csv missing - anime-only rebuild: {info['anime_rows']:,} "
              f"anime rows, {info['genre_baskets']:,} genre baskets.")
        rebuilt = "anime-only (no ratings.csv)"
    print(f"rebuild done: {rebuilt}")

    # -------------------------------------------------------- 3. retrain ---
    print(f"\n== retrain stages {args.stages} ==")
    stage_info = run_stages([int(s) for s in args.stages])
    for k, v in stage_info.items():
        print(f"  stage {k}: {v}")
    ok = [k for k, v in stage_info.items() if v.startswith("ok")]

    # ------------------------------------------------------------ 4. log ---
    from core import reporting as REP
    summary = REP.read_summary(C.REPORT_DIR / "mining_summary.csv")
    dur = round(time.time() - t0, 1)
    append_log({
        "run_at": LF.utcnow_iso(), "mode": f"{mode} | rebuild:{rebuilt}",
        "selected": len(selected), "fetched": sync_rep.get("fetched", 0),
        "changed": changed if not args.skip_sync else "",
        "failed": len(sync_rep.get("failed", {})),
        "skipped": False, "stages": " ".join(str(s) for s in args.stages),
        "stages_ok": " ".join(ok), "duration_s": dur,
        "predictor_best": summary.get("stage6.best", ""),
        "predictor_rmse": summary.get("stage6.sigma", ""),
        "recommender_users": (summary.get("stage6.recommender.n_users", "")
                              or summary.get("stage6.recommender['n_users']", "")),
        "git_sha": git_sha()})
    print(f"\nweekly retrain done in {dur:.0f}s - "
          f"{len(ok)}/{len(stage_info)} stages ok. Log: reports/retrain_log.csv")
    failed = [k for k in stage_info if not stage_info[k].startswith("ok")]
    return 0 if not failed else 2


if __name__ == "__main__":
    sys.exit(main())
