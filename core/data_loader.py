"""CSV loading helpers. All I/O is CSV, results are cached on disk."""
from __future__ import annotations

import pandas as pd

from . import config as C


def _cache_read(path, **kwargs) -> pd.DataFrame:
    """Read a CSV once and memoise it by (path, mtime) for the session."""
    key = (str(path), path.stat().st_mtime_ns)
    return _READ_CACHE.get(key) if _READ_CACHE.get(key) is not None else _store(
        key, pd.read_csv(path, **kwargs)
    )


_READ_CACHE: dict = {}


def _store(key, df):
    _READ_CACHE.clear()          # keep memory bounded, one frame at a time
    _READ_CACHE[key] = df
    return df


def load_raw_anime() -> pd.DataFrame:
    """anime.csv - 12,294 rows: anime_id, name, genre, type, episodes, rating, members."""
    return _cache_read(C.RAW_ANIME, dtype={"anime_id": "int64", "members": "int64"})


def load_raw_ratings() -> pd.DataFrame:
    """ratings.csv - user_id, anime_id, rating (-1 = watched but not scored)."""
    return _cache_read(C.RAW_RATINGS, dtype={"user_id": "int64", "anime_id": "int64"})


def load_processed(name: str) -> pd.DataFrame:
    path = C.PROC_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"{path.name} not built yet - run `python build_dataset.py` first."
        )
    return _cache_read(path)
