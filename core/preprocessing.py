"""Data Warehousing & Mining - Stage 1: Preprocessing.

Raw anime.csv / ratings.csv are cleaned, validated, feature-engineered,
outlier-flagged, normalised, discretised and split into modelling frames.
Every artefact is written back out as CSV.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C

REPORT: dict[str, object] = {}


# =============================================================== ANIME ======
def clean_anime() -> tuple[pd.DataFrame, dict]:
    """Stage 1-3: profiling, cleaning, type casting for anime.csv."""
    C.ensure_dirs()
    raw = pd.read_csv(C.RAW_ANIME)
    profile = {
        "raw_rows": int(len(raw)),
        "raw_cols": int(raw.shape[1]),
        "raw_missing": raw.isna().sum().to_dict(),
        "raw_dupes": int(raw.duplicated().sum()),
        "episodes_dtype": str(raw["episodes"].dtype),
    }

    df = raw.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # ---- duplicate handling ------------------------------------------------
    dup_ids = int(df["anime_id"].duplicated().sum())
    df = df.drop_duplicates(subset="anime_id", keep="first")
    profile["duplicate_ids_removed"] = dup_ids
    profile["clean_rows"] = int(len(df))

    # ---- missing values ----------------------------------------------------
    for col in ("name", "genre", "type"):
        df[col] = df[col].fillna("Unknown").astype("string").str.strip()
    profile["missing_filled"] = {
        "genre": int((raw["genre"].isna()).sum()),
        "type": int((raw["type"].isna()).sum()),
        "rating": int((raw["rating"].isna()).sum()),
    }

    # ---- episodes is a string column containing 'Unknown' ------------------
    ep = pd.to_numeric(df["episodes"], errors="coerce")
    profile["episodes_unparsed"] = int(ep.isna().sum())
    df["episodes"] = ep
    # impute with the median of each media type (Movie/ONA/Special/Music = 1 ep)
    med_by_type = df.groupby("type")["episodes"].transform("median")
    df["episodes"] = df["episodes"].fillna(med_by_type).fillna(df["episodes"].median())
    profile["episodes_impute_median"] = int(profile["episodes_unparsed"])

    # ---- rating ------------------------------------------------------------
    profile["rating_missing"] = int(df["rating"].isna().sum())
    # impute with the type-wise mean, then the global mean
    df["rating"] = df["rating"].fillna(df.groupby("type")["rating"].transform("mean"))
    df["rating"] = df["rating"].fillna(df["rating"].mean())
    df["rating"] = df["rating"].round(2)

    # ---- categorical hygiene ----------------------------------------------
    df["type"] = df["type"].where(df["type"].isin(C.ANIME_TYPES), "Unknown")

    # ---- genre engineering -------------------------------------------------
    df["genre_list"] = df["genre"].str.split(",").apply(
        lambda xs: [x.strip() for x in xs if x and x.strip() not in ("", "Unknown")]
    )
    df["genre_count"] = df["genre_list"].apply(len)
    df["primary_genre"] = df["genre_list"].apply(
        lambda xs: xs[0] if xs else "Unknown"
    )
    df["is_multi_genre"] = (df["genre_count"] > 1).astype(int)
    df["is_shounen"] = df["genre"].str.contains("Shounen", na=False).astype(int)
    df["is_seinen"] = df["genre"].str.contains("Seinen", na=False).astype(int)
    df["is_shoujo"] = df["genre"].str.contains("Shoujo", na=False).astype(int)
    df["has_isekai"] = df["genre"].str.contains("Isekai", na=False).astype(int)

    # genre binary flags used by the association-rule baskets
    for g in C.ASSOC_GENRES:
        flag = "g_" + g.replace("-", "").replace(" ", "_")
        df[flag] = df["genre"].str.contains(g, na=False, regex=False).astype(int)

    # ---- derived numeric attributes ---------------------------------------
    df["members_log"] = np.log1p(df["members"].clip(lower=0))
    df["episodes_log"] = np.log1p(df["episodes"])
    df["members_per_episode"] = df["members"] / df["episodes"].clip(lower=1)
    df["engagement_index"] = df["members"] * df["rating"] / 1000.0

    df = df.drop(columns=["genre_list"])
    df.to_csv(C.CLEAN_ANIME, index=False)
    profile["out_rows"] = int(len(df))
    return df, profile


# ============================================================== OUTLIERS ====
def detect_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """IQR fence rule on the skewed numeric attributes."""
    rows = []
    for col in ("members", "rating", "episodes", "genre_count"):
        s = df[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (s < lo) | (s > hi)
        rows.append({
            "attribute": col,
            "q1": round(float(q1), 3),
            "q3": round(float(q3), 3),
            "iqr": round(float(iqr), 3),
            "lower_fence": round(float(lo), 3),
            "upper_fence": round(float(hi), 3),
            "n_outliers": int(mask.sum()),
            "pct_outliers": round(float(mask.mean() * 100), 2),
            "skewness": round(float(s.skew()), 3),
        })
    return pd.DataFrame(rows)


def add_outlier_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    flags = []
    for col in ("members", "rating", "episodes"):
        s = out[col].astype(float)
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        flags.append(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).astype(int))
    out["is_outlier"] = np.maximum.reduce(flags)
    out.loc[out["members"] < C.MIN_MEMBERS, "is_outlier"] = 1
    return out


# ======================================================== NORMALISATION =====
def add_scaled(df: pd.DataFrame) -> pd.DataFrame:
    """min-max and z-score versions of the continuous attributes."""
    out = df.copy()
    for col in ("members_log", "episodes_log", "rating", "genre_count"):
        s = out[col].astype(float)
        lo, hi = s.min(), s.max()
        out[f"{col}_minmax"] = ((s - lo) / (hi - lo)).round(4) if hi > lo else 0.0
        sd = s.std(ddof=0)
        out[f"{col}_zscore"] = ((s - s.mean()) / sd).round(4) if sd > 0 else 0.0
    return out


# ======================================================== DISCRETISATION =====
def add_class_label(df: pd.DataFrame) -> pd.DataFrame:
    """Equal-width binning of the target `rating` into an ordinal class."""
    out = df.copy()
    edges = [v[0] for v in C.RATING_TIERS.values()] + [C.RATING_TIERS["Top"][1]]
    out["rating_tier"] = pd.cut(
        out["rating"], bins=edges, labels=C.TIER_ORDER, right=False
    ).astype("string")
    out["members_band"] = pd.cut(
        out["members"],
        bins=[-1, 100, 1_000, 10_000, 100_000, 10**9],
        labels=["tiny", "small", "medium", "large", "huge"],
    ).astype("string")
    out["episodes_band"] = pd.cut(
        out["episodes"],
        bins=[-1, 1, 12, 26, 52, 10**6],
        labels=["single", "short", "mid", "long", "series"],
    ).astype("string")
    return out


# ============================================================== RATINGS =====
def build_ratings_tables(anime: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stream the 7.8M-row ratings file and aggregate it into two CSV tables.

    Returns (per-anime rating statistics, per-user behaviour profile).
    """
    C.ensure_dirs()
    usecols = ["user_id", "anime_id", "rating"]
    scored = 0
    unrated = 0
    per_anime: dict[int, list] = {}
    per_user: dict[int, dict] = {}

    for chunk in pd.read_csv(C.RAW_RATINGS, usecols=usecols, chunksize=500_000):
        # rating == -1 means "watched but not scored" -> missing, not a value
        unrated += int((chunk["rating"] == -1).sum())
        scored_chunk = chunk[chunk["rating"] > 0]
        scored += len(scored_chunk)
        for a, g in scored_chunk.groupby("anime_id", sort=False):
            d = per_anime.setdefault(a, {"s": 0.0, "s2": 0.0, "n": 0, "mx": 0, "mn": 10})
            v = g["rating"].to_numpy(dtype=float)
            d["s"] += v.sum()
            d["s2"] += (v * v).sum()
            d["n"] += v.size
            d["mx"] = max(d["mx"], float(v.max()))
            d["mn"] = min(d["mn"], float(v.min()))
        for u, g in scored_chunk.groupby("user_id", sort=False):
            d = per_user.setdefault(
                u, {"s": 0.0, "s2": 0.0, "n": 0, "mx": 0.0, "mn": 10.0, "liked": 0, "disliked": 0}
            )
            v = g["rating"].to_numpy(dtype=float)
            d["s"] += v.sum()
            d["s2"] += (v * v).sum()
            d["n"] += v.size
            d["mx"] = max(d["mx"], float(v.max()))
            d["mn"] = min(d["mn"], float(v.min()))
            d["liked"] += int((v >= C.LIKED_THRESHOLD).sum())
            d["disliked"] += int((v <= 4).sum())

    meta = {"raw_interactions": int(scored + unrated), "scored_interactions": int(scored),
            "watched_not_rated": int(unrated), "n_anime_rated": len(per_anime),
            "n_users_rated": len(per_user)}

    # ---- per-anime statistics ---------------------------------------------
    arows = []
    for a, d in per_anime.items():
        n = d["n"]
        mean = d["s"] / n
        var = max((d["s2"] - d["s"] * d["s"] / n) / (n - 1), 0.0) if n > 1 else 0.0
        arows.append({
            "anime_id": a, "n_ratings": n, "mean_user_rating": round(mean, 4),
            "std_user_rating": round(float(np.sqrt(var)), 4),
            "max_user_rating": d["mx"], "min_user_rating": d["mn"],
        })
    astats = pd.DataFrame(arows).sort_values("anime_id").reset_index(drop=True)
    astats.to_csv(C.ANIME_STATS, index=False)

    # ---- per-user behaviour profile (Dataset B for clustering) --------------
    urows = []
    for u, d in per_user.items():
        n = d["n"]
        mean = d["s"] / n
        var = max((d["s2"] - d["s"] * d["s"] / n) / (n - 1), 0.0) if n > 1 else 0.0
        urows.append({
            "user_id": u, "n_ratings": n, "mean_rating": round(mean, 4),
            "std_rating": round(float(np.sqrt(var)), 4),
            "max_rating": d["mx"], "min_rating": d["mn"],
            "range_rating": round(d["mx"] - d["mn"], 4),
            "n_liked": d["liked"], "n_disliked": d["disliked"],
            "like_ratio": round(d["liked"] / n, 4),
            "critic_score": round(mean + np.sqrt(var), 4),   # mean + 1 sigma
        })
    ufeat = pd.DataFrame(urows).sort_values("user_id").reset_index(drop=True)
    ufeat.to_csv(C.USER_FEATURES, index=False)
    return astats, ufeat, meta


# ============================================================== BASKETS =====
def build_genre_baskets(anime: pd.DataFrame) -> pd.DataFrame:
    """Transactions = anime, Items = genres. One row per anime."""
    genre_cols = [c for c in anime.columns if c.startswith("g_")]
    b = anime[["anime_id"] + genre_cols].copy()
    b.to_csv(C.BASKET_GENRE, index=False)
    return b


def build_user_baskets(max_users: int = 6000) -> pd.DataFrame:
    """Transactions = users, Items = the top-N anime each user rated >= 8.

    Written out in long format (user_id, item, rank, rating) - one row per
    user-anime pair - because a 6,000 x 6,000 one-hot matrix does not belong in
    a CSV. `mine_user_rules` regroups it into transactions.
    """
    C.ensure_dirs()
    parts = []
    for chunk in pd.read_csv(C.RAW_RATINGS, usecols=["user_id", "anime_id", "rating"],
                             chunksize=500_000):
        c = chunk[chunk["rating"] >= C.LIKED_THRESHOLD]
        if len(c):
            parts.append(c)
    if not parts:
        raise RuntimeError("no scored ratings found")
    top = pd.concat(parts, ignore_index=True)

    # only "recommendable" titles (enough community to be a real suggestion)
    pop = (pd.read_csv(C.CLEAN_ANIME, usecols=["anime_id", "name", "members"])
           .query("members >= @C.LIKED_MIN_MEMBERS"))
    pop_ids = set(pop["anime_id"].tolist())
    names = pop.set_index("anime_id")["name"].to_dict()
    top = top[top["anime_id"].isin(pop_ids)]
    top["item"] = top["anime_id"].map(lambda a: names[int(a)])

    top = (top.sort_values(["user_id", "rating", "anime_id"],
                           ascending=[True, False, True])
              .groupby("user_id", sort=True)
              .head(C.BASKET_SIZE)
              .reset_index(drop=True))
    top["rank"] = top.groupby("user_id").cumcount() + 1

    # a random but reproducible sample of engaged users
    rng = np.random.RandomState(C.RANDOM_STATE)
    users = top["user_id"].unique()
    if len(users) > max_users:
        keep = set(rng.choice(users, size=max_users, replace=False).tolist())
        top = top[top["user_id"].isin(keep)]
    top = top[["user_id", "rank", "item", "rating", "anime_id"]].sort_values(
        ["user_id", "rank"]).reset_index(drop=True)
    top.to_csv(C.BASKET_USER, index=False)
    return top


def load_baskets_user(min_items: int = 3) -> pd.DataFrame:
    """Regroup the long-format user basket into a transaction list."""
    b = pd.read_csv(C.BASKET_USER)
    tx = (b.groupby("user_id")["item"]
            .apply(lambda s: sorted(set(s)))
            .reset_index())
    tx["n"] = tx["item"].apply(len)
    return tx[tx["n"] >= min_items].reset_index(drop=True)


# ============================================================== OUTPUTS =====
def build_modelling_frames(anime: pd.DataFrame) -> dict:
    """Assemble the classification + regression datasets and export as CSV."""
    from sklearn.model_selection import train_test_split

    C.ensure_dirs()
    df = add_class_label(add_scaled(add_outlier_flags(anime)))

    cls = df[(df["members"] >= C.MIN_MEMBERS) & df["rating_tier"].notna()].copy()
    # keep classes that are actually learnable
    vc = cls["rating_tier"].value_counts()
    keep = vc[vc >= 25].index
    cls = cls[cls["rating_tier"].isin(keep)].reset_index(drop=True)
    X_cls = cls.drop(columns=["genre", "name"])
    y_cls = cls["rating_tier"]
    Xc_tr, Xc_te, yc_tr, yc_te = train_test_split(
        X_cls, y_cls, test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE, stratify=y_cls
    )
    cls_tr = Xc_tr.copy(); cls_tr["target"] = yc_tr.to_numpy()
    cls_te = Xc_te.copy(); cls_te["target"] = yc_te.to_numpy()
    cls_tr.to_csv(C.CLS_SPLIT, index=False)
    cls_te.to_csv(C.CLS_SPLIT.with_name("classification_test.csv"), index=False)

    reg = df[df["members"] >= C.MIN_MEMBERS].copy()
    reg = reg.drop(columns=["genre", "name"])
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        reg.drop(columns=["rating"]), reg["rating"],
        test_size=C.TEST_SIZE, random_state=C.RANDOM_STATE
    )
    rtr = Xr_tr.copy(); rtr["target"] = yr_tr.to_numpy()
    rte = Xr_te.copy(); rte["target"] = yr_te.to_numpy()
    rtr.to_csv(C.REG_SPLIT, index=False)
    rte.to_csv(C.REG_SPLIT.with_name("regression_test.csv"), index=False)

    return {
        "classification": {
            "rows": int(len(cls)), "features": int(X_cls.shape[1]),
            "class_distribution": vc.reindex(C.TIER_ORDER).fillna(0).astype(int).to_dict(),
            "train": int(len(cls_tr)), "test": int(len(cls_te)),
        },
        "regression": {
            "rows": int(len(reg)), "features": int(reg.shape[1] - 1),
            "target": "rating (0-10 MAL score)", "train": int(len(rtr)), "test": int(len(rte)),
        },
    }
