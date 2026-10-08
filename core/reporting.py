"""Long-format CSV reporting helpers.

Every artefact this project writes is a CSV, including the pipeline summaries.
A nested summary dictionary is flattened to `metric,value` rows, which keeps the
CSV self-describing and trivially readable back into a dict.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _walk(prefix: str, obj, out: list) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            _walk(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _walk(f"{prefix}[{i}]", v, out)
    elif isinstance(obj, (bool, np.bool_)):
        out.append((prefix, str(bool(obj))))
    elif isinstance(obj, (int, np.integer)):
        out.append((prefix, int(obj)))
    elif isinstance(obj, (float, np.floating)):
        out.append((prefix, round(float(obj), 6)))
    elif obj is None:
        out.append((prefix, ""))
    elif isinstance(obj, str):
        out.append((prefix, obj))
    else:
        out.append((prefix, str(obj)))


def flatten(payload: dict) -> pd.DataFrame:
    rows: list = []
    _walk("", payload, rows)
    return pd.DataFrame(rows, columns=["metric", "value"])


def write_summary(path: Path, payload: dict) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = flatten(payload)
    df.to_csv(path, index=False, encoding="utf-8")
    return df


def _coerce(v: str):
    """Turn a CSV cell back into int / float / bool where it obviously is one."""
    s = str(v).strip()
    if s in ("", "-", "nan", "NaN", "None", "null"):
        return ""
    if s in ("True", "False"):
        return s == "True"
    try:
        return int(s) if s.lstrip("+-").isdigit() else float(s)
    except ValueError:
        return s


def read_summary(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    df = pd.read_csv(path, keep_default_na=False, dtype=str)
    return {m: _coerce(v) for m, v in zip(df["metric"], df["value"])}


def get(flat: dict, metric: str, default=None):
    """Fetch a single metric from a flattened summary, with a safe default."""
    v = flat.get(metric, default)
    return default if v == "" else v
