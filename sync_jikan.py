"""Sync anime.csv with the free live Jikan (MyAnimeList) API.

The batch DWM pipeline is untouched: this script only refreshes
``data/raw/anime.csv`` (same 7 columns) plus an append-only score history.
Run it, then ``build_dataset.py`` and ``train.py`` exactly as before::

    python sync_jikan.py --ids 1 5 20        # specific MAL ids
    python sync_jikan.py --top 200           # top 200 titles by members
    python sync_jikan.py --stale 30          # not refreshed in 30 days
    python sync_jikan.py --all --limit 500   # whole catalog, capped
    python sync_jikan.py --dry-run --top 5   # select only, no network/write
    python sync_jikan.py --self-test         # offline fixture check, no network
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from core import config as C
from core import live_feed as LF


def select_ids(args) -> list[int]:
    anime = pd.read_csv(C.RAW_ANIME, dtype={"anime_id": "int64"})
    if args.ids:
        return [int(x) for x in args.ids]
    if args.top:
        ids = (anime.sort_values("members", ascending=False)
               .head(args.top)["anime_id"].astype(int).tolist())
        return ids[:args.limit] if args.limit else ids
    if args.stale is not None:
        synced: set[int] = set()
        if C.LIVE_STATE.exists():
            try:
                st = pd.read_csv(C.LIVE_STATE, dtype={"anime_id": "int64"},
                                 parse_dates=["fetched_at"])
                times = pd.to_datetime(st["fetched_at"], utc=True)
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=args.stale)
                synced = set(st.loc[times >= cutoff, "anime_id"])
            except (ValueError, TypeError):
                synced = set()  # unreadable state: refresh everything
        ids = [i for i in anime["anime_id"].astype(int) if i not in synced]
    else:  # --all
        ids = anime["anime_id"].astype(int).tolist()
    if args.limit:
        ids = ids[:args.limit]
    return ids


def self_test() -> int:
    """Offline end-to-end check of normalize -> upsert -> history (no network)."""
    import tempfile
    fix = json.loads((C.LIVE_FIXTURES / "jikan_anime_1.json").read_text(
        encoding="utf-8"))["data"]
    norm = LF.normalize(fix)
    assert norm["row"]["anime_id"] == 1
    assert norm["row"]["episodes"] == 26
    assert norm["row"]["rating"] == 8.75
    assert "Action" in norm["row"]["genre"] and "Seinen" in norm["row"]["genre"], \
        norm["row"]["genre"]
    assert norm["row"]["type"] == "TV"
    assert norm["extras"]["scored_by"] == 600123

    with tempfile.TemporaryDirectory() as td:
        csv = Path(td) / "anime.csv"
        pd.DataFrame([{"anime_id": 1, "name": "Cowboy Bebop", "genre": "Unknown",
                       "type": "TV", "episodes": "Unknown", "rating": 7.0,
                       "members": 100}]).to_csv(csv, index=False)
        changed, changes = LF.upsert(pd.read_csv(csv), norm)
        assert changed and set(changes) >= {"genre", "episodes", "rating",
                                            "members"}, changes
        # invalid Jikan values must never clobber local data
        bad = {"row": {"anime_id": 1, "name": "X", "genre": "", "type": "XX",
                       "episodes": None, "rating": None, "members": -5},
               "extras": {}}
        df = pd.read_csv(csv)
        # NOTE: upsert works in place on the passed frame; reload fixture state
        df.loc[0, ["genre", "episodes", "rating", "members"]] = [
            "Action", 26, 8.75, 1723456]
        changed2, _ = LF.upsert(df, bad)
        assert not changed2, "invalid values must be ignored"
        assert df.loc[0, "name"] == "Cowboy Bebop", "name must never change"
        # unknown id is skipped unless allow_new with a name
        new_norm = {"row": {"anime_id": 999999, "name": None, "genre": "Action",
                            "type": "TV", "episodes": 12, "rating": 8.0,
                            "members": 5000}, "extras": {}}
        c3, _ = LF.upsert(df, new_norm)
        assert not c3
        c4, ch4 = LF.upsert(df, new_norm, allow_new=True,
                            default_name="Brand New Anime")
        assert c4 and ch4 == {"added": 999999}
    print("self-test OK: normalize, guarded upsert, allow_new all behave")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Refresh anime.csv from Jikan.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--ids", nargs="+", help="MAL ids, e.g. --ids 1 5 20")
    src.add_argument("--top", type=int, help="top N titles by members")
    src.add_argument("--stale", type=int, metavar="DAYS",
                     help="ids not synced in the last DAYS (or never)")
    src.add_argument("--all", action="store_true", help="whole catalog")
    src.add_argument("--self-test", action="store_true",
                     help="offline fixture check, no network")
    ap.add_argument("--limit", type=int, help="cap number of ids fetched")
    ap.add_argument("--dry-run", action="store_true",
                    help="select ids only; no network calls, no writes")
    ap.add_argument("--allow-new", action="store_true",
                    help="append Jikan titles missing from anime.csv")
    ap.add_argument("--no-cache", action="store_true",
                    help="ignore cached JSON, refetch everything selected")
    args = ap.parse_args(argv)

    if args.self_test:
        return self_test()
    ids = select_ids(args)
    print(f"selected {len(ids)} ids"
          + (f" (showing first 10: {ids[:10]})" if ids else ""))
    if args.dry_run or not ids:
        print("dry run: no network calls, no writes.")
        return 0
    client = LF.JikanClient()
    rep = LF.sync(ids, client=client, allow_new=args.allow_new,
                use_cache=not args.no_cache)
    print(f"fetched={rep['fetched']} changed={rep['changed']} "
          f"calls={rep['calls_made']} cache_hits={rep['cache_hits']}")
    if rep["fields"]:
        print("fields updated:", rep["fields"])
    if rep["failed"]:
        print(f"failed ({len(rep['failed'])}), first few:")
        for k, v in list(rep["failed"].items())[:10]:
            print(f"  {k}: {v}")
    print("next: python build_dataset.py  &&  python train.py")
    return 0 if not rep["failed"] else 2


if __name__ == "__main__":
    sys.exit(main())
