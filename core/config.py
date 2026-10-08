"""Central configuration: paths, constants and shared vocabularies."""
from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------- paths ----
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
FIG_DIR = ROOT / "reports" / "figures"

RAW_ANIME = RAW_DIR / "anime.csv"
RAW_RATINGS = RAW_DIR / "ratings.csv"

CLEAN_ANIME = PROC_DIR / "anime_clean.csv"
ANIME_FEATURES = PROC_DIR / "anime_features.csv"
ANIME_LABELED = PROC_DIR / "anime_labeled.csv"
USER_FEATURES = PROC_DIR / "user_features.csv"
BASKET_GENRE = PROC_DIR / "baskets_genre.csv"
BASKET_USER = PROC_DIR / "baskets_user.csv"
RULES_GENRE = REPORT_DIR / "association_rules_genre.csv"
RULES_USER = REPORT_DIR / "association_rules_user.csv"
CLS_SPLIT = PROC_DIR / "classification_split.csv"
REG_SPLIT = PROC_DIR / "regression_split.csv"
ANIME_STATS = PROC_DIR / "anime_rating_stats.csv"

# ---------------------------------------------------------- model files ----
MODEL_PATH = MODEL_DIR / "rating_predictor.joblib"
RECOMMENDER_PATH = MODEL_DIR / "recommender.joblib"
CLUSTER_MODEL_PATH = MODEL_DIR / "cluster_models.joblib"
PREPROC_META = MODEL_DIR / "preprocess_meta.joblib"

# ------------------------------------------------------------- constants ---
# Ordinal class for the rating target. The edges are the ~30/30/20/20
# quantiles of `rating`, so the four classes are learnable instead of leaving
# a 93-row tail class that no classifier can score.
RATING_TIERS = {
    "Low": (0.0, 6.1),
    "Medium": (6.1, 6.8),
    "High": (6.8, 7.3),
    "Top": (7.3, 11.0),
}
TIER_ORDER = ["Low", "Medium", "High", "Top"]

PRIMARY_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror",
    "Mecha", "Music", "Mystery", "Romance", "Sci-Fi", "Slice of Life",
    "Sports", "Supernatural", "Thriller",
]

# wider item set used only for the genre association baskets (43 genres exist)
ASSOC_GENRES = [
    "Action", "Adventure", "Comedy", "Drama", "Fantasy", "Horror", "Kids",
    "School", "Slice of Life", "Hentai", "Supernatural", "Mecha", "Music",
    "Historical", "Magic", "Ecchi", "Shoujo", "Seinen", "Shounen", "Sports",
    "Mystery", "Super Power", "Military", "Parody", "Space", "Demons",
    "Martial Arts", "Psychological", "Samurai", "Gag Humor",
]

ANIME_TYPES = ["TV", "OVA", "Movie", "Special", "ONA", "Music", "Unknown"]

RANDOM_STATE = 42
TEST_SIZE = 0.2
LIKED_THRESHOLD = 8          # a user "liked" an anime when rating >= 8
BASKET_SIZE = 12             # top-N liked anime kept per user basket
MIN_MEMBERS = 50             # drop near-empty anime from modelling
LIKED_MIN_MEMBERS = 1000     # 10% lift target must be a real recommendation


# ------------------------------------------------------- live Jikan feed ----
# Additive live-feed settings. Nothing in the batch pipeline reads these;
# `sync_jikan.py` refreshes data/raw/anime.csv, after which build_dataset.py
# and train.py run exactly as before.
LIVE_DIR = ROOT / "data" / "live"
LIVE_CACHE_DIR = LIVE_DIR / "cache"          # raw Jikan JSON per anime_id
LIVE_HISTORY = LIVE_DIR / "score_history.csv"  # append-only score snapshots
LIVE_STATE = LIVE_DIR / "last_sync.csv"        # anime_id -> last fetched_at
LIVE_FIXTURES = LIVE_DIR / "fixtures"

JIKAN_BASE_URL = "https://api.jikan.moe/v4"
JIKAN_MIN_INTERVAL = 0.5   # seconds between calls (limit is 3 req/s)
JIKAN_TIMEOUT = 20         # seconds per request
JIKAN_MAX_RETRIES = 3      # retries on HTTP 429 / 5xx with backoff


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROC_DIR, MODEL_DIR, REPORT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
