"""Data Warehousing & Mining - Stage 5: Unsupervised clustering on TWO datasets.

Dataset A  - CONTENT     : one row per anime  (12k rows, 6 engineered features)
Dataset B  - BEHAVIOUR   : one row per user   (73k rows, 8 behavioural features)

Both are clustered with K-Means and Agglomerative (Ward) hierarchical clustering,
k is chosen from the elbow method + silhouette score, and the clusters are then
profiled back onto human-readable labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from . import config as C

DATASETS = {
    "A - Anime (content)": {
        "file": C.CLEAN_ANIME,
        "source": "anime.csv - 12,294 titles",
        "features": ["members_log", "episodes_log", "rating", "genre_count",
                     "is_shounen", "is_multi_genre"],
        "labels": ["members_log", "episodes_log", "rating", "genre_count",
                   "is_shounen", "is_multi_genre"],
        "id": "anime_id", "name": "name",
    },
    "B - Users (behaviour)": {
        "file": C.USER_FEATURES,
        "source": "ratings.csv aggregated per user - 73,515 users",
        "features": ["n_ratings", "mean_rating", "std_rating", "like_ratio",
                     "range_rating", "max_rating", "n_liked", "critic_score"],
        "labels": ["n_ratings", "mean_rating", "std_rating", "like_ratio",
                   "range_rating", "max_rating", "n_liked", "critic_score"],
        "id": "user_id", "name": None,
    },
}

K_RANGE = range(2, 9)
HIER_SAMPLE = 1200          # dendrogram cost is O(n^2) -> subsample for Ward
SIL_SAMPLE = 5000           # silhouette is O(n^2) in memory -> never score more
PCA_PLOT_SAMPLE = 3000      # max points forwarded for the 2-D scatter


def _sampled_silhouette(Z: np.ndarray, labels: np.ndarray,
                        max_n: int = SIL_SAMPLE) -> float:
    """Silhouette on a fixed random subsample.

    Full silhouette needs a pairwise distance matrix (n^2 floats: ~19 GB for
    the 69.6k user rows), which is what kills 1 GB Streamlit Cloud containers.
    A 5k sample gives the same value to ~0.01 at <10 MB.
    """
    Z = np.asarray(Z)
    labels = np.asarray(labels)
    n = len(Z)
    if n > max_n:
        rs = np.random.RandomState(C.RANDOM_STATE)
        idx = rs.choice(n, size=max_n, replace=False)
        Z, labels = Z[idx], labels[idx]
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(Z, labels))


def load_dataset(key: str, min_rows: int = 1) -> pd.DataFrame:
    spec = DATASETS[key]
    df = pd.read_csv(spec["file"])
    for f in spec["features"]:
        df[f] = pd.to_numeric(df[f], errors="coerce")
    df = df.dropna(subset=spec["features"])
    if spec["id"] and spec["id"] in df.columns:
        df = df[df[spec["id"]].notna()]
    if len(df) < min_rows:
        raise ValueError(f"{key}: only {len(df)} usable rows")
    return df.reset_index(drop=True)


def scaled_matrix(df: pd.DataFrame, key: str) -> tuple[np.ndarray, list[str], np.ndarray]:
    spec = DATASETS[key]
    X = df[spec["features"]].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    Z = StandardScaler().fit_transform(X)
    return Z, spec["features"], X


# ------------------------------------------------------------- diagnostics --
def k_selection(Z: np.ndarray, k_range=K_RANGE) -> pd.DataFrame:
    """Elbow + silhouette: pick k at the elbow, or the silhouette maximum."""
    rows = []
    inertia, sil, ch, db = [], [], [], []
    ss = np.random.RandomState(C.RANDOM_STATE)
    # diagnostics are scored on a fixed random subsample (silhouette is O(n^2))
    Zs = Z[ss.permutation(len(Z))[:2000]]
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=C.RANDOM_STATE).fit(Zs)
        lab = km.labels_
        inertia.append(km.inertia_)
        sil.append(silhouette_score(Zs, lab) if len(set(lab)) > 1 else -1)
        ch.append(calinski_harabasz_score(Zs, lab))
        db.append(davies_bouldin_score(Zs, lab))
        rows.append({"k": k, "inertia": round(float(km.inertia_), 1),
                     "elbow_knee": None})
    df = pd.DataFrame({"k": list(k_range), "inertia": [round(i, 1) for i in inertia],
                       "silhouette": np.round(sil, 4),
                       "calinski_harabasz": np.round(ch, 1),
                       "davies_bouldin": np.round(db, 4)})
    # knee = max distance to the chord between the first and last inertia point
    x = df["k"].to_numpy(dtype=float)
    y = df["inertia"].to_numpy(dtype=float)
    x1, y1, x2, y2 = x[0], y[0], x[-1], y[-1]
    d = np.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1)
    df["distance_to_chord"] = np.round(d, 1)
    df["elbow_knee"] = np.where(d == d.max(), df["k"], None)
    return df


def best_k(diag: pd.DataFrame, prefer: int | None = None) -> tuple[int, str]:
    knee = diag.loc[diag["elbow_knee"].notna(), "k"]
    sil = int(diag.loc[diag["silhouette"].idxmax(), "k"])
    elbow = int(knee.iloc[0]) if len(knee) else sil
    if prefer:
        return int(prefer), "user-selected"
    return elbow, f"elbow(knee)={elbow}, silhouette-max={sil}"


# --------------------------------------------------------------- K-Means ---
def kmeans_cluster(df: pd.DataFrame, key: str, k: int, seed: int = C.RANDOM_STATE) -> dict:
    Z, cols, Xraw = scaled_matrix(df, key)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(Z)
    labels = km.labels_
    sil = _sampled_silhouette(Z, km.labels_)
    prof = profile(df, key, labels, Z, cols)
    # PCA is fitted on a subsample for speed, then applied to a capped sample
    # for plotting; returning 69k scatter points would OOM the browser anyway.
    rs = np.random.RandomState(seed)
    fit_idx = rs.choice(len(Z), size=min(5000, len(Z)), replace=False)
    pca_model = PCA(n_components=2, random_state=seed).fit(Z[fit_idx])
    plot_idx = rs.choice(len(Z), size=min(PCA_PLOT_SAMPLE, len(Z)), replace=False)
    plot_idx = np.sort(plot_idx)
    return {
        "labels": labels, "Z": Z, "pca": pca_model.transform(Z[plot_idx]),
        "pca_idx": plot_idx, "features": cols,
        "inertia": float(km.inertia_), "silhouette": float(sil),
        "centers_raw": pd.DataFrame(km.cluster_centers_, columns=cols).round(3),
        "profile": prof, "k": k, "n": len(df),
        "size": pd.Series(labels).value_counts().sort_index().to_dict(),
    }


# ---------------------------------------------------------- Hierarchical ---
def hierarchical_cluster(df: pd.DataFrame, key: str, k: int,
                          sample: int = HIER_SAMPLE,
                          seed: int = C.RANDOM_STATE) -> dict:
    """Ward-linkage agglomerative clustering on a representative subsample.

    Agglomerative clustering needs an n x n distance matrix, which is 38 GB for
    the 69.6k user rows, so it is fitted on a fixed random subsample and the
    result is reported for that sample.

    scipy's `linkage` is used rather than sklearn's AgglomerativeClustering
    because it also returns the full merge tree, which is what makes the
    dendrogram in the UI (and a k for any k) available.
    """
    from scipy.cluster.hierarchy import fcluster, linkage

    Z, cols, _ = scaled_matrix(df, key)
    rs = np.random.RandomState(seed)
    n = len(Z)
    idx = np.sort(rs.choice(n, size=min(sample, n), replace=False))
    Zs = Z[idx]
    link = linkage(Zs, method="ward", metric="euclidean", optimal_ordering=False)
    labels = fcluster(link, t=k, criterion="maxclust") - 1     # back to 0..k-1
    return {
        "sample_labels": labels, "sample_idx": idx,
        "sample_df": df.iloc[idx].reset_index(drop=True),
        "silhouette": float(_sampled_silhouette(Zs, labels)), "k": k,
        "n_sample": len(idx), "n": n, "linkage": "ward",
        "Z_linkage": link, "linkage_matrix": link, "features": cols,
        "profile": profile(df.iloc[idx].reset_index(drop=True), key, labels, Zs, cols),
    }


def _linkage_matrix(Z: np.ndarray) -> np.ndarray:
    """Condensed distance matrix - kept for the O(n^2) cost comparison."""
    from scipy.spatial.distance import pdist
    return pdist(Z, metric="euclidean")


# -------------------------------------------------------------- profiles ---
def profile(df: pd.DataFrame, key: str, labels: np.ndarray, Z: np.ndarray,
            cols: list[str], show_cols: int = 6) -> pd.DataFrame:
    spec = DATASETS[key]
    d = df.copy()
    d["cluster"] = labels
    rows = []
    for c in sorted(np.unique(labels)):
        sub = d[d["cluster"] == c]
        row = {"cluster": int(c), "n": int(len(sub)),
               "share_%": round(100 * len(sub) / len(d), 2)}
        for col in cols[:show_cols]:
            row[col] = round(float(sub[col].mean()), 3)
        if spec["name"]:
            top = sub[spec["name"]].astype(str).value_counts().head(3)
            row["top_" + spec["name"]] = " | ".join(f"{k} ({v})" for k, v in top.items())
        if "type" in sub.columns:
            row["top_type"] = " | ".join(sub["type"].value_counts().head(2).index)
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------- cluster naming ----
# Segment labels are derived from the *rank* of each cluster's centroid on the
# key features, so the names stay correct whatever the column distributions are.
_RANK_TIERS = {
    "members_log": ["niche", "mid-reach", "blockbuster"],
    "rating": ["poorly rated", "solid", "acclaimed"],
    "episodes_log": ["short-form", "mid-length", "long-form"],
    "n_ratings": ["light rater", "casual rater", "heavy rater"],
    "mean_rating": ["harsh critic", "balanced", "generous"],
    "std_rating": ["consistent", "opinionated", "erratic"],
    "like_ratio": ["selective", "moderate", "enthusiastic"],
}


def name_clusters(profile_df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Turn numeric cluster centroids into distinct human-readable labels."""
    out = profile_df.copy().sort_values("cluster").reset_index(drop=True)
    n = len(out)
    ranked: dict[str, list[int]] = {}
    for col, tiers in _RANK_TIERS.items():
        if col not in out.columns or n == 0:
            continue
        # rank clusters by centroid value, then bucket into thirds (or halves)
        order = out[col].rank(method="first", ascending=True)
        bucket = np.floor((order - 1) / max(n / len(tiers), 1)).astype(int)
        labels = [tiers[min(b, len(tiers) - 1)] for b in bucket]
        ranked[col] = labels

    feats = ["members_log", "rating", "episodes_log"] if key.startswith("A") else \
            ["n_ratings", "mean_rating", "std_rating"]
    feats = [f for f in feats if f in ranked]
    out["segment_name"] = [
        " / ".join(dict.fromkeys(ranked[f][i] for f in feats)) for i in range(n)
    ]
    # guarantee uniqueness (two clusters can tie on every feature)
    seen: dict[str, int] = {}
    names = []
    for s in out["segment_name"]:
        seen[s] = seen.get(s, 0) + 1
        names.append(s if seen[s] == 1 else f"{s} (variant {seen[s]})")
    out["segment_name"] = names
    return out


# ---------------------------------------------------------- association ----
def cluster_vs_class(df: pd.DataFrame, key: str, labels: np.ndarray) -> pd.DataFrame:
    """Cross-tabulate the clusters against the discretised target - a DWM cube."""
    d = df.copy()
    d["cluster"] = labels
    if "members_band" in d.columns:
        band = d["members_band"].astype(str)
    elif "members" in d.columns:
        band = pd.cut(d["members"], [-1, 100, 1e3, 1e4, 1e5, 1e9],
                      labels=["tiny", "small", "medium", "large", "huge"]).astype(str)
    else:
        band = None
    if band is not None:
        order = ["tiny", "small", "medium", "large", "huge"]
        band = pd.Categorical(band, categories=[c for c in order if c in set(band)])
        return pd.crosstab(d["cluster"], band).reset_index()
    if "mean_rating" in d.columns:
        b = pd.cut(d["mean_rating"], [0, 6, 7, 8, 11],
                   labels=["low", "mid", "good", "high"]).astype(str)
        return pd.crosstab(d["cluster"], b).reset_index()
    return pd.DataFrame()
