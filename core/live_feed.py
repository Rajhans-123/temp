"""Live anime (MyAnimeList) feed: refresh anime.csv without touching DWM logic.

The batch pipeline (`build_dataset.py` -> `train.py`) only ever reads
``data/raw/anime.csv``. This module fetches the same 7-column schema from the
free Tenrai v1 API (Jikan-v4-compatible, no key; ``anime_id`` IS the MAL id,
so rows join 1:1), upserts changed fields, and appends an append-only score
history that later becomes the retraining signal. Cleaning, imputation, splits
and all six mining stages are unchanged - they simply see fresher input.
"""
from __future__ import annotations

import datetime as _dt
import json
import time
import urllib.error
import urllib.request

import pandas as pd

from . import config as C

ANIME_COLUMNS = ["anime_id", "name", "genre", "type", "episodes", "rating", "members"]
HISTORY_COLUMNS = ["fetched_at", "anime_id", "rating", "members",
                   "scored_by", "favorites", "episodes"]


def utcnow_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ------------------------------------------------------------------ client ---
class JikanClient:
    """Minimal stdlib client: polite rate limit, retries, on-disk JSON cache.

    (Name is historical: the backend is now Tenrai, the Jikan-v4
    continuation, which serves the same response shape.)"""

    def __init__(self, min_interval: float = C.JIKAN_MIN_INTERVAL,
                 timeout: int = C.JIKAN_TIMEOUT,
                 max_retries: int = C.JIKAN_MAX_RETRIES,
                 cache_dir=C.LIVE_CACHE_DIR) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self.max_retries = max_retries
        self.cache_dir = cache_dir
        self._last_call = 0.0
        self.calls_made = 0
        self.cache_hits = 0

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.time()

    def fetch(self, mal_id: int, use_cache: bool = True) -> dict:
        """Return the raw ``data`` object of GET /anime/{id}."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / f"{int(mal_id)}.json"
        if use_cache and cached.exists():
            try:
                payload = json.loads(cached.read_text(encoding="utf-8"))
                self.cache_hits += 1
                return payload["data"]
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError):
                cached.unlink(missing_ok=True)  # corrupt cache: fall to refetch
        url = f"{C.ANIME_API_BASE_URL}/anime/{int(mal_id)}"
        last_err: Exception | None = None
        for attempt in range(self.max_retries + 1):
            self._throttle()
            req = urllib.request.Request(
                url, headers={"User-Agent": "animeRecommendation-DWM/1.0"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                last_err = e
                if e.code == 404:
                    raise KeyError(f"API has no anime {mal_id}") from e
                if e.code in (429,) or 500 <= e.code < 600:
                    time.sleep(2.0 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError) as e:
                last_err = e
                time.sleep(2.0 * (attempt + 1))
                continue
            self.calls_made += 1
            cached.write_text(json.dumps({"data": payload["data"]},
                                         ensure_ascii=False), encoding="utf-8")
            return payload["data"]
        raise RuntimeError(f"API fetch failed for {mal_id}: {last_err}")


# -------------------------------------------------------------- normalize ---
def _names(items) -> list[str]:
    out: list[str] = []
    for g in items or []:
        name = (g.get("name") or "").strip()
        if name and name not in out:
            out.append(name)
    return out


def normalize(payload: dict) -> dict:
    """Map a Tenrai (Jikan-v4) ``/anime`` object onto the anime.csv 7-column schema.

    Returns the schema row plus an ``_extras`` dict (score voters, favorites)
    that is stored in the history table, never in anime.csv.
    """
    genres = (_names(payload.get("genres")) + _names(payload.get("themes"))
              + _names(payload.get("demographics")))
    episodes = payload.get("episodes")
    episodes = int(episodes) if isinstance(episodes, int) and episodes > 0 else None
    score = payload.get("score")
    score = float(score) if isinstance(score, (int, float)) and score > 0 else None
    members = payload.get("members")
    members = int(members) if isinstance(members, int) and members > 0 else None
    atype = (payload.get("type") or "").strip()
    row = {
        "anime_id": int(payload["mal_id"]),
        "name": None,  # never overwritten; filled from the local row on upsert
        "genre": ", ".join(genres) if genres else None,
        "type": atype if atype in C.ANIME_TYPES else None,
        "episodes": episodes,
        "rating": round(score, 2) if score is not None else None,
        "members": members,
    }
    extras = {
        "scored_by": payload.get("scored_by"),
        "favorites": payload.get("favorites"),
    }
    return {"row": row, "extras": extras}


# ----------------------------------------------------------------- upsert ---
def _valid(col: str, new) -> bool:
    """Write-boundary guard: only sane values may overwrite local data."""
    if new is None:
        return False
    if col == "genre":
        return isinstance(new, str) and bool(new.strip())
    if col == "type":
        return new in C.ANIME_TYPES
    if col in ("episodes", "members"):
        try:
            iv = int(new)
        except (TypeError, ValueError):
            return False
        return iv > 0 if col == "episodes" else iv >= 0
    if col == "rating":
        try:
            rv = float(new)
        except (TypeError, ValueError):
            return False
        return 0.0 < rv <= 10.0
    return False


# ----------------------------------------------------------------- upsert ---
def upsert(anime: pd.DataFrame, norm: dict, allow_new: bool = False,
           default_name: str | None = None) -> tuple[bool, dict]:
    """Apply one normalized row. Only valid API values overwrite local data.

    The local ``name`` is never touched (it is the join key for baskets and
    rules). Unknown ids are appended only with ``allow_new`` (needs a name).
    Returns (changed, {field: (old, new)}).
    """
    row = norm["row"]
    changes: dict = {}
    # pandas 3 infers arrow-string dtypes; assigning an int (episodes) or
    # float (rating/members) into a str column then raises TypeError, so
    # relax the mutable columns to object dtype before upserting.
    for col in ("genre", "type", "episodes", "rating", "members"):
        if col in anime.columns:
            try:
                if pd.api.types.is_string_dtype(anime[col].dtype):
                    anime[col] = anime[col].astype(object)
            except TypeError:
                pass
    hit = anime.index[anime["anime_id"] == row["anime_id"]]
    if len(hit) == 0:
        if not allow_new or not default_name:
            return False, {"skipped": "unknown anime_id"}
        new = {c: row[c] for c in ANIME_COLUMNS}
        new["name"] = default_name
        anime.loc[len(anime)] = [new[c] for c in ANIME_COLUMNS]
        return True, {"added": row["anime_id"]}
    i = int(hit[0])
    for col in ("genre", "type", "episodes", "rating", "members"):
        new = row[col]
        if not _valid(col, new):
            continue
        old = anime.at[i, col]
        try:
            same = float(old) == float(new)
        except (TypeError, ValueError):
            same = str(old) == str(new)
        if not same:
            changes[col] = (old, new)
            anime.at[i, col] = new
    return bool(changes), changes


# ------------------------------------------------------------------ sync ----
def sync(mal_ids: list[int], client: JikanClient | None = None,
         allow_new: bool = False, anime_path=C.RAW_ANIME,
         name_overrides: dict[int, str] | None = None,
         use_cache: bool = True) -> dict:
    """Fetch ids, upsert anime.csv (with .bak), append history + state.

    ``name_overrides`` maps anime_id -> name and is only used together with
    ``allow_new`` (offline/self-test path where no local row exists).
    """
    client = client or JikanClient()
    C.LIVE_DIR.mkdir(parents=True, exist_ok=True)
    anime = pd.read_csv(anime_path, dtype={"anime_id": "int64"})
    fetched_at = utcnow_iso()
    hist_rows: list[dict] = []
    state: dict[int, str] = {}
    report = {"fetched": 0, "changed": 0, "failed": {}, "fields": {}}
    for mid in mal_ids:
        try:
            norm = normalize(client.fetch(int(mid), use_cache=use_cache))
        except Exception as e:  # noqa: BLE001 - per-id failure must not stop sync
            report["failed"][int(mid)] = f"{type(e).__name__}: {e}"[:160]
            continue
        report["fetched"] += 1
        mid_key = norm["row"]["anime_id"]
        changed, changes = upsert(
            anime, norm, allow_new=allow_new,
            default_name=(name_overrides or {}).get(mid_key))
        if changed:
            report["changed"] += 1
            for f in changes:
                if f not in ("skipped", "added"):
                    report["fields"][f] = report["fields"].get(f, 0) + 1
        hist_rows.append({"fetched_at": fetched_at, "anime_id": mid_key,
                          "rating": norm["row"]["rating"],
                          "members": norm["row"]["members"],
                          "scored_by": norm["extras"].get("scored_by"),
                          "favorites": norm["extras"].get("favorites"),
                          "episodes": norm["row"]["episodes"]})
        state[mid_key] = fetched_at
    if report["changed"]:
        bak = anime_path.with_suffix(".csv.bak")
        anime_path.replace(bak)  # previous version kept aside, one level
        tmp = anime_path.with_suffix(".csv.tmp")
        anime.to_csv(tmp, index=False)
        tmp.replace(anime_path)
    if hist_rows:
        hist = pd.DataFrame(hist_rows, columns=HISTORY_COLUMNS)
        hdr = not C.LIVE_HISTORY.exists()
        hist.to_csv(C.LIVE_HISTORY, mode="a", header=hdr, index=False)
    if state:
        prev = {}
        if C.LIVE_STATE.exists():
            old = pd.read_csv(C.LIVE_STATE, dtype={"anime_id": "int64"})
            prev = dict(zip(old["anime_id"], old["fetched_at"]))
        prev.update(state)
        pd.DataFrame([{"anime_id": k, "fetched_at": v} for k, v in
                      sorted(prev.items())]).to_csv(C.LIVE_STATE, index=False)
    report["calls_made"] = client.calls_made
    report["cache_hits"] = client.cache_hits
    return report
