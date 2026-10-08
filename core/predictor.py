"""Prediction system.

1. `train_rating_predictor` - regresses the MAL score of an anime from its
   content attributes (Random Forest + Gradient Boosting, blended). This is the
   model behind the prediction widget in the UI.
2. `train_recommender`      - a hybrid recommender combining user-behaviour
   clusters, per-segment genre affinity, the stage-2 association rules and the
   community prior. Everything is written to models/*.joblib for the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from . import config as C
from . import regression as R
from . import clustering as CL

GENRE_FLAGS = {
    "Action": "g_Action", "Adventure": "g_Adventure", "Comedy": "g_Comedy",
    "Drama": "g_Drama", "Fantasy": "g_Fantasy", "Horror": "g_Horror",
    "Kids": "g_Kids", "Music": "g_Music", "Mystery": "g_Mystery",
    "Romance": "g_Romance", "Sci-Fi": "g_SciFi", "Slice of Life": "g_Slice_of_Life",
    "Sports": "g_Sports", "Supernatural": "g_Supernatural",
    "Thriller": "g_Thriller", "Shounen": "g_Shounen", "Shoujo": "g_Shoujo",
    "Seinen": "g_Seinen", "Mecha": "g_Mecha", "Historical": "g_Historical",
    "Magic": "g_Magic", "Ecchi": "g_Ecchi", "Hentai": "g_Hentai",
    "School": "g_School", "Psychological": "g_Psychological",
    "Super Power": "g_Super_Power", "Military": "g_Military",
    "Parody": "g_Parody", "Space": "g_Space", "Demons": "g_Demons",
    "Martial Arts": "g_Martial_Arts", "Samurai": "g_Samurai",
    "Gag Humor": "g_Gag_Humor",
}
TYPE_MEDIAN_EPISODES = {"TV": 26, "OVA": 4, "Movie": 1, "Special": 1,
                        "ONA": 13, "Music": 1, "Unknown": 12}


# ================================================== 1. RATING PREDICTOR ====
def train_rating_predictor(force: bool = False) -> dict:
    """Train the model behind the prediction widget.

    Two base learners (Random Forest + Gradient Boosting) are blended. The blend
    weight is fitted on a held-out validation split - never on the test set -
    and both learners are then refitted on the whole training split.
    """
    C.ensure_dirs()
    df = pd.read_csv(C.CLEAN_ANIME)
    df = df[df["members"] >= C.MIN_MEMBERS].copy()
    feat_cols = [c for c in R.features(df) if c in df.columns]
    X, y = df[feat_cols], df["rating"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=C.TEST_SIZE,
                                          random_state=C.RANDOM_STATE)
    Xfit, Xval, yfit, yval = train_test_split(Xtr, ytr, test_size=0.25,
                                              random_state=C.RANDOM_STATE)
    Xfit_e, (Xval_e, Xte_e, X_e) = R._encode(Xfit, [Xval, Xte, X])
    yfit, yval = yfit.to_numpy(float), yval.to_numpy(float)
    yte, y_e = yte.to_numpy(float), y.to_numpy(float)

    models = {
        # depth/leaf caps keep the persisted forest deployable (~10 MB instead of
        # ~85 MB) at a negligible cost in accuracy
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=4, n_jobs=-1,
            random_state=C.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.9,
            random_state=C.RANDOM_STATE),
    }
    # ---- 1. fit on the fit-split, learn the blend weight on validation -----
    p_val = []
    for m in models.values():
        m.fit(Xfit_e, yfit)
        p_val.append(m.predict(Xval_e))
    w = _best_blend_weight(np.column_stack(p_val), yval)
    blend_val = w[0] * p_val[0] + w[1] * p_val[1]

    # ---- 2. refit on the full training split -------------------------------
    Xtr_e = R._encode(Xtr, [])[0]
    fitted, rows, p_te = {}, [], []
    for name, m in models.items():
        m.fit(Xtr_e, ytr)
        p = m.predict(Xte_e)
        fitted[name] = m
        p_te.append(p)
    blend_te = w[0] * p_te[0] + w[1] * p_te[1]
    rows.append({"split": "test", "model": f"BLEND {w[0]:.2f}*RF + {w[1]:.2f}*GB",
                 **R._metrics(yte, blend_te)})
    for name, p in zip(models, p_te):
        rows.append({"split": "test", "model": name, **R._metrics(yte, p)})
    rows.append({"split": "validation (weight fitting only)",
                 "model": f"BLEND {w[0]:.2f}*RF + {w[1]:.2f}*GB",
                 **R._metrics(yval, blend_val)})

    # held-out residual spread -> the prediction interval shown in the UI
    sigma = float(np.sqrt(np.mean((yte - blend_te) ** 2)))
    full_pred = w[0] * fitted["Random Forest"].predict(X_e) + \
        w[1] * fitted["Gradient Boosting"].predict(X_e)
    r2_full = float(r2_score(y_e, full_pred))

    table = pd.DataFrame(rows)
    # "best" is decided on the test split only; the validation row is a diagnostic
    best = table[table["split"] == "test"].sort_values("RMSE").iloc[0]
    table = table.sort_values(["split", "RMSE"]).reset_index(drop=True)
    import joblib
    joblib.dump({
        "models": fitted, "blend": w, "feature_cols": feat_cols,
        "encoders": _encoder_map(Xtr), "sigma": sigma, "r2_full": r2_full,
        "n_train": int(len(Xtr)), "metrics": table.to_dict("records"),
        "actual": np.round(yte, 3), "predicted": np.round(blend_te, 3),
        "residuals": np.round(yte - blend_te, 3),
        "actual_all": np.round(y_e, 3), "predicted_all": np.round(full_pred, 3),
    }, C.MODEL_PATH)

    imp = pd.DataFrame({"attribute": feat_cols,
                        "importance": np.round(
                            fitted["Random Forest"].feature_importances_, 4)})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["cumulative"] = imp["importance"].cumsum().round(4)
    imp.to_csv(C.REPORT_DIR / "predictor_importance.csv", index=False)
    table.to_csv(C.REPORT_DIR / "predictor_metrics.csv", index=False)
    return {"metrics": table, "best": best["model"],
            "blend": (round(float(w[0]), 3), round(float(w[1]), 3)),
            "sigma": round(sigma, 4), "r2_full": round(r2_full, 4),
            "n_train": int(len(Xtr)), "importance": imp}


def _best_blend_weight(P: np.ndarray, y: np.ndarray, step: float = 0.02) -> np.ndarray:
    """Grid-search w so that w*P0 + (1-w)*P1 minimises SSE (weights sum to 1)."""
    best_w, best_sse = 1.0, np.inf
    for w in np.arange(0.0, 1.0 + step / 2, step):
        sse = float(np.sum((y - (w * P[:, 0] + (1 - w) * P[:, 1])) ** 2))
        if sse < best_sse:
            best_w, best_sse = float(w), sse
    return np.array([best_w, 1.0 - best_w])


def _encoder_map(X: pd.DataFrame) -> dict:
    enc = {}
    for c in R.CATEGORICAL:
        if c in X.columns:
            _, uniq = pd.factorize(X[c].astype(str), sort=True)
            enc[c] = {v: i for i, v in enumerate(uniq)}
    return enc


class RatingPredictor:
    """Loads the trained regressor and scores unseen anime."""

    def __init__(self, path=C.MODEL_PATH):
        import joblib
        if not path.exists():
            raise FileNotFoundError(
                "rating_predictor.joblib missing - run `python train.py` first")
        self.bundle = joblib.load(path)
        self.models = self.bundle["models"]
        self.blend = self.bundle.get("blend", np.array([0.5, 0.5]))
        self.feature_cols = self.bundle["feature_cols"]
        self.encoders = self.bundle["encoders"]
        self.sigma = float(self.bundle["sigma"])
        self._defaults = _load_defaults()

    # ---------------------------------------------------------------- build --
    def _row(self, *, name, anime_type, episodes, members, genres,
             extra: dict | None = None) -> pd.DataFrame:
        d = self._defaults.copy()
        d["name"] = name
        d["type"] = anime_type if anime_type in C.ANIME_TYPES else "Unknown"
        d["episodes"] = (float(episodes) if episodes and episodes > 0
                         else TYPE_MEDIAN_EPISODES.get(d["type"], 12))
        d["episodes_log"] = float(np.log1p(max(d["episodes"], 0)))
        d["members"] = (float(members) if members and members > 0
                        else self._defaults["members"])
        d["members_log"] = float(np.log1p(d["members"]))
        d["members_per_episode"] = d["members"] / max(d["episodes"], 1)

        gset = {g.strip() for g in (genres or []) if g and g.strip() not in ("", "Unknown")}
        d["genre"] = ", ".join(sorted(gset)) if gset else "Unknown"
        d["genre_count"] = max(len(gset), 1)
        d["primary_genre"] = sorted(gset)[0] if gset else "Unknown"
        d["is_multi_genre"] = int(d["genre_count"] > 1)
        for flag, genre in (("is_shounen", "Shounen"), ("is_seinen", "Seinen"),
                            ("is_shoujo", "Shoujo"), ("has_isekai", "Isekai")):
            d[flag] = int(genre in gset)
        for f in GENRE_FLAGS.values():
            if f in d:
                d[f] = 0
        for g in gset:
            if g in GENRE_FLAGS:
                d[GENRE_FLAGS[g]] = 1
        d["members_band"] = str(pd.cut(
            [d["members"]], [-1, 100, 1e3, 1e4, 1e5, 1e9],
            labels=["tiny", "small", "medium", "large", "huge"])[0])
        d["episodes_band"] = str(pd.cut(
            [d["episodes"]], [-1, 1, 12, 26, 52, 1e6],
            labels=["single", "short", "mid", "long", "series"])[0])
        if extra:
            d.update(extra)
        row = pd.DataFrame([d])
        for c, m in self.encoders.items():
            if c in row.columns:
                row[c] = row[c].astype(str).map(m).fillna(len(m)).astype(int)
        return row.reindex(columns=self.feature_cols)

    # ----------------------------------------------------------------- API ---
    def predict(self, *, name="New anime", anime_type="TV", episodes=None,
                members=None, genres=None, extra=None) -> dict:
        X = self._row(name=name, anime_type=anime_type, episodes=episodes,
                      members=members, genres=genres, extra=extra)
        rf = float(self.models["Random Forest"].predict(X)[0])
        gb = float(self.models["Gradient Boosting"].predict(X)[0])
        w = self.blend
        score = float(np.clip(w[0] * rf + w[1] * gb, 1.0, 10.0))
        tier = next((t for t, (a, b) in C.RATING_TIERS.items() if a <= score < b),
                    C.TIER_ORDER[-1])
        return {
            "score": round(score, 3),
            "lower": round(float(np.clip(score - 1.96 * self.sigma, 1.0, 10.0)), 3),
            "upper": round(float(np.clip(score + 1.96 * self.sigma, 1.0, 10.0)), 3),
            "tier": tier,
            "random_forest": round(rf, 3),
            "gradient_boosting": round(gb, 3),
            "blend": (round(float(w[0]), 3), round(float(w[1]), 3)),
            "sigma": round(self.sigma, 3),
        }


def _load_defaults() -> dict:
    df = pd.read_csv(C.CLEAN_ANIME, nrows=3000)
    d = df.iloc[0].to_dict()
    d["members"] = float(df["members"].median())
    d["episodes"] = float(df["episodes"].median())
    for f in GENRE_FLAGS.values():
        if f in df.columns:
            d[f] = 0
    return d


# ===================================================== 2. RECOMMENDER =====
def train_recommender(user_clusters: int = 5) -> dict:
    """Fit the hybrid recommender and persist everything the UI needs.

    Ingredients
      1. K-Means over the user-behaviour profile (dataset B of the clustering
         stage) -> every user is assigned to a taste segment.
      2. Per-segment genre and media-type affinity streamed out of the 7.8M raw
         ratings: "how does segment 3 rate Action anime?"
      3. The stage-2 association rules, boosting items that follow whatever the
         user has already watched.
      4. The community prior of each candidate item.
    """
    C.ensure_dirs()
    import joblib

    anime = pd.read_csv(C.CLEAN_ANIME)
    anime = anime[anime["members"] >= C.LIKED_MIN_MEMBERS].reset_index(drop=True)
    users = pd.read_csv(C.USER_FEATURES)
    users = users[users["n_ratings"] >= 20].reset_index(drop=True)

    Z, cols, _ = CL.scaled_matrix(users, "B - Users (behaviour)")
    km = KMeans(n_clusters=user_clusters, n_init=10,
                random_state=C.RANDOM_STATE).fit(Z)
    users["cluster"] = km.labels_
    user_cluster = {int(u): int(c) for u, c in zip(users["user_id"], users["cluster"])}

    aff, type_aff, n_ratings = _cluster_affinity(users[["user_id", "cluster"]], anime)
    try:
        rules = pd.read_csv(C.RULES_USER)
        rules = rules[rules["lift"] >= 1.2].head(5000)
    except FileNotFoundError:
        rules = pd.DataFrame(columns=["antecedents", "consequents", "lift", "confidence"])

    joblib.dump({
        "anime": anime[["anime_id", "name", "type", "episodes", "rating", "members",
                        "primary_genre", "genre"]].to_dict("records"),
        "user_cluster": user_cluster,
        "user_profiles": users.to_dict("records"),
        "user_feature_cols": cols,
        "affinity": aff, "type_affinity": type_aff, "n_ratings": n_ratings,
        "global_affinity": _global_affinity(aff),
        "rules": rules,
        "item_stats": _item_stats(anime),
    }, C.RECOMMENDER_PATH)
    return {"n_anime": len(anime), "n_users": len(users), "k": user_clusters,
            "genres": len(aff), "rules": int(len(rules)), "affinity": aff}


def _cluster_affinity(user_clusters: pd.DataFrame,
                      anime: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Mean rating per (cluster, genre) and per (cluster, type) from raw ratings."""
    cmap = user_clusters.set_index("user_id")["cluster"].to_dict()
    gmap = anime.set_index("anime_id")["primary_genre"].to_dict()
    tmap = anime.set_index("anime_id")["type"].to_dict()
    g_acc: dict[tuple[int, str], list] = {}
    t_acc: dict[tuple[int, str], list] = {}
    n_acc: dict[str, int] = {}

    for chunk in pd.read_csv(C.RAW_RATINGS, usecols=["user_id", "anime_id", "rating"],
                             chunksize=500_000):
        c = chunk[chunk["rating"] > 0]
        c = c[c["user_id"].isin(cmap)]
        if c.empty:
            continue
        c = c.assign(cluster=c["user_id"].map(cmap),
                     genre=c["anime_id"].map(gmap),
                     atype=c["anime_id"].map(tmap)).dropna(subset=["genre", "atype"])
        for cl, g in c.groupby("cluster", sort=False):
            n_acc[str(int(cl))] = n_acc.get(str(int(cl)), 0) + len(g)
            for gg, sub in g.groupby("genre", sort=False):
                g_acc.setdefault((int(cl), gg), []).append(float(sub["rating"].mean()))
            for tt, sub in g.groupby("atype", sort=False):
                t_acc.setdefault((int(cl), tt), []).append(float(sub["rating"].mean()))
    aff = {f"{k[0]}|{k[1]}": round(float(np.mean(v)), 3) for k, v in g_acc.items()}
    type_aff = {f"{k[0]}|{k[1]}": round(float(np.mean(v)), 3) for k, v in t_acc.items()}
    return aff, type_aff, n_acc


def _global_affinity(aff: dict) -> dict:
    """Genre -> mean rating across all segments (fallback for unseen segments)."""
    g: dict[str, list] = {}
    for k, v in aff.items():
        g.setdefault(k.split("|", 1)[1], []).append(v)
    return {k: round(float(np.mean(v)), 3) for k, v in g.items()}


def _item_stats(anime: pd.DataFrame) -> dict:
    return {
        "type_mean_rating": anime.groupby("type")["rating"].mean().round(3).to_dict(),
        "genre_mean_rating": anime.groupby("primary_genre")["rating"].mean().round(3).to_dict(),
        "median_members": float(anime["members"].median()),
        "median_episodes": float(anime["episodes"].median()),
    }


_GENRE_INDEX = None


def _genres_of(watched: set[str]) -> set[str]:
    """Genres covered by a set of already-watched anime titles."""
    global _GENRE_INDEX
    if _GENRE_INDEX is None:
        a = pd.read_csv(C.CLEAN_ANIME, usecols=["name", "genre"])
        _GENRE_INDEX = {
            r["name"]: {x.strip() for x in str(r["genre"]).split(",")
                        if x.strip() and x.strip() != "Unknown"}
            for _, r in a.iterrows()
        }
    out: set[str] = set()
    for w in watched:
        out |= _GENRE_INDEX.get(w, set())
    return out


class Recommender:
    """Hybrid recommender: user segment x genre affinity x rules x prior."""

    def __init__(self, path=C.RECOMMENDER_PATH):
        import joblib
        if not path.exists():
            raise FileNotFoundError(
                "recommender.joblib missing - run `python train.py` first")
        self.b = joblib.load(path)
        self.anime = pd.DataFrame(self.b["anime"])
        self.user_cluster = {int(k): int(v) for k, v in self.b["user_cluster"].items()}
        self.user_profiles = pd.DataFrame(self.b["user_profiles"])
        self.affinity = self.b["affinity"]
        self.type_affinity = self.b.get("type_affinity", {})
        self.global_affinity = self.b.get("global_affinity", {})
        self.rules = self.b["rules"]
        self.stats = self.b["item_stats"]

    # ------------------------------------------------------------ internals --
    def _affinity_for(self, cluster: int) -> dict[str, float]:
        pref = f"{cluster}|"
        aff = {k.split("|", 1)[1]: v for k, v in self.affinity.items() if k.startswith(pref)}
        return {g: aff.get(g, self.global_affinity.get(g, 7.0))
                for g in self.global_affinity}

    def _type_affinity_for(self, cluster: int) -> dict[str, float]:
        pref = f"{cluster}|"
        return {k.split("|", 1)[1]: v for k, v in self.type_affinity.items()
                if k.startswith(pref)}

    def _rule_boost(self, watched: set[str]) -> dict[str, float]:
        """Lift contributed by the association rules that fire for what was watched."""
        if self.rules.empty or not watched:
            return {}
        mask = self.rules["antecedents"].astype(str).apply(
            lambda a: any(w in a for w in watched))
        out: dict[str, float] = {}
        for _, r in self.rules[mask].iterrows():
            for item in str(r["consequents"]).split(", "):
                out[item] = out.get(item, 0.0) + float(r["lift"])
        return out

    def _score(self, d: pd.DataFrame, affinity: dict[str, float],
               type_aff: dict[str, float], boost: dict[str, float],
               cluster: int | None, exclude: set[str]) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame(
                columns=["anime_id", "name", "type", "primary_genre", "episodes",
                         "rating", "members", "genre_affinity", "community_score",
                         "match_score", "reasons"])
        tmean = self.stats["type_mean_rating"]
        prior = d["rating"].to_numpy(dtype=float)
        pop = (np.log1p(d["members"].to_numpy(dtype=float))
               - np.log1p(self.stats["median_members"])) / 5.0
        g_aff = np.array([affinity.get(g, 7.0) - 7.0 for g in d["primary_genre"]])
        t_aff = np.array([type_aff.get(t, 7.0) - 7.0 for t in d["type"]])
        prior_z = (prior - prior.mean()) / max(prior.std(), 1e-9)
        type_dev = np.array([prior[i] - tmean.get(t, prior[i])
                             for i, t in enumerate(d["type"])])

        score = (0.30 * g_aff        # taste match of the user's segment
                 + 0.18 * t_aff      # media-type preference of the segment
                 + 0.20 * prior_z    # community quality prior
                 + 0.18 * type_dev   # better than usual for its own media type
                 + 0.14 * np.clip(pop, -1, 1.5))   # mild popularity preference
        if boost:
            b = np.array([boost.get(nm, 0.0) for nm in d["name"]])
            score = score + 0.05 * np.clip(b - 1.2, 0, 2.0)

        out = d[["anime_id", "name", "type", "primary_genre", "episodes",
                 "rating", "members"]].copy()
        out.insert(0, "rank", np.arange(1, len(out) + 1))
        if cluster is not None:
            out.insert(1, "user_segment", cluster)
        out["genre_affinity"] = np.round(7.0 + g_aff, 3)
        out["community_score"] = np.round(prior, 3)
        out["match_score"] = np.round(score, 4)
        if exclude:
            out = out[~out["name"].isin(exclude)]
        out = out.sort_values("match_score", ascending=False).reset_index(drop=True)
        out["rank"] = np.arange(1, len(out) + 1)
        out["why"] = [
            f"{g}: your segment rates it {a:.2f} vs community {r:.2f}"
            for g, a, r in zip(out["primary_genre"], out["genre_affinity"], prior)
        ]
        return out

    # ----------------------------------------------------------------- API ---
    def valid_users(self, n: int = 300) -> pd.DataFrame:
        """A spread of users for the UI dropdown, with profile and segment."""
        p = self.user_profiles.copy()
        p = p.sort_values("n_ratings", ascending=False).head(n)
        p["label"] = (p["user_id"].astype(int).astype(str)
                      + "  -  " + p["n_ratings"].astype(int).astype(str) + " anime rated"
                      + "  -  mean " + p["mean_rating"].round(2).astype(str)
                      + "  -  segment " + p["cluster"].astype(int).astype(str))
        return p

    def segment_profile(self, cluster: int) -> pd.DataFrame:
        """Genres that define one segment, for display in the UI."""
        pref = f"{cluster}|"
        rows = [{"genre": k.split("|", 1)[1], "segment_mean_rating": v}
                for k, v in self.affinity.items() if k.startswith(pref)]
        d = pd.DataFrame(rows)
        if d.empty:
            return d
        g = self.global_affinity
        d["overall_mean_rating"] = d["genre"].map(g)
        d["delta"] = (d["segment_mean_rating"] - d["overall_mean_rating"]).round(3)
        return d.sort_values("delta", ascending=False).reset_index(drop=True)

    def cluster_of(self, user_id: int) -> int | None:
        return self.user_cluster.get(int(user_id))

    def profile_of(self, user_id: int) -> dict:
        row = self.user_profiles[self.user_profiles["user_id"] == int(user_id)]
        return row.iloc[0].to_dict() if len(row) else {}

    def recommend_by_user(self, user_id: int, n: int = 10) -> pd.DataFrame:
        c = self.cluster_of(user_id)
        if c is None:
            raise KeyError(
                f"user {int(user_id)} has fewer than 20 ratings, so no segment was "
                f"learned for them - pick a more active user")
        return self._score(self.anime, self._affinity_for(c),
                           self._type_affinity_for(c), {}, c, set()).head(n)

    def recommend_by_preferences(self, genres: list[str], anime_type: str = "Any",
                                n: int = 10) -> pd.DataFrame:
        gset = {g.strip() for g in genres if g and g.strip()}
        d = self.anime
        if gset:
            d = d[d["genre"].apply(
                lambda s: bool(gset & {x.strip() for x in str(s).split(",")}))]
        if anime_type != "Any":
            d = d[d["type"] == anime_type]
        if d.empty:
            d = (self.anime[self.anime["primary_genre"].isin(gset)] if gset
                 else self.anime)
        if d.empty:
            d = self.anime
        # the visitor's own picks act as a pseudo-segment
        aff = dict(self.global_affinity)
        for g in gset:
            aff[g] = min(aff.get(g, 7.0) + 0.8, 10.0)
        return self._score(d, aff, {}, {}, None, set()).head(n)

    def recommend_by_watched(self, watched: list[str], n: int = 10) -> pd.DataFrame:
        watched = {w.strip() for w in watched if w and w.strip()}
        d = self.anime[~self.anime["name"].isin(watched)] if watched else self.anime
        if d.empty:
            d = self.anime
        boost = self._rule_boost(watched)
        aff = dict(self.global_affinity)
        for g in _genres_of(watched):
            aff[g] = min(aff.get(g, 7.0) + 0.7, 10.0)
        return self._score(d, aff, {}, boost, None, watched).head(n)
