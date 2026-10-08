"""Data Warehousing & Mining - Stage 4: Regression / continuous estimation.

Target: the MyAnimeList community score of an anime (continuous, 0-10) predicted
from its content attributes (media type, episode count, genre make-up, reach).
This is the model that also backs the prediction widget in the UI.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor, export_text

from . import config as C

NUMERIC = ["episodes", "episodes_log", "members", "members_log", "genre_count",
           "is_multi_genre", "is_shounen", "is_shoujo", "is_seinen", "has_isekai",
           "members_per_episode"]
CATEGORICAL = ["type", "primary_genre", "members_band", "episodes_band"]


def features(df: pd.DataFrame) -> list[str]:
    return ([c for c in NUMERIC if c in df.columns] +
            [c for c in CATEGORICAL if c in df.columns])


def load_regression_data():
    tr = pd.read_csv(C.REG_SPLIT)
    te = pd.read_csv(C.REG_SPLIT.with_name("regression_test.csv"))
    cols = features(tr)
    return tr[cols], tr["target"], te[cols], te["target"]


def _encode(Xtr: pd.DataFrame, Xs: list[pd.DataFrame]):
    enc = {}
    o = Xtr.copy()
    for c in o.columns:
        if o[c].dtype == object or str(o[c].dtype) in ("string", "str"):
            _, uniq = pd.factorize(o[c].astype(str), sort=True)
            enc[c] = {v: i for i, v in enumerate(uniq)}
            o[c] = o[c].astype(str).map(enc[c]).astype(int)
    outs = [o]
    for X in Xs:
        t = X.copy()
        for c, m in enc.items():
            t[c] = t[c].astype(str).map(m).fillna(-1).astype(int)
        outs.append(t)
    return outs[0], outs[1:]


def _metrics(y, p) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y, p)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 4),
        "R2": round(float(r2_score(y, p)), 4),
        "MAPE_%": round(float(np.mean(np.abs((y - p) / np.maximum(y, 1e-9))) * 100), 3),
    }


def run_regression(Xtr, ytr, Xte, yte, n_splits: int = 5) -> dict:
    Xtr_e, (Xte_e,) = _encode(Xtr, [Xte])
    models = {
        "Mean baseline": DummyRegressor(),
        "Linear Regression": LinearRegression(),
        "Ridge (alpha=1)": Ridge(alpha=1.0, random_state=C.RANDOM_STATE),
        "Decision Tree": DecisionTreeRegressor(max_depth=8, min_samples_leaf=10,
                                                random_state=C.RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, max_depth=14, min_samples_leaf=3, n_jobs=-1,
            random_state=C.RANDOM_STATE),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=4,
            random_state=C.RANDOM_STATE),
    }
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    rows, fitted, preds = [], {}, {}
    for name, m in models.items():
        cv_rmse = -cross_val_score(clone(m), Xtr_e, ytr, cv=cv,
                                   scoring="neg_root_mean_squared_error", n_jobs=-1).mean()
        m.fit(Xtr_e, ytr)
        p = m.predict(Xte_e)
        fitted[name] = m
        preds[name] = p
        rows.append({"model": name, "cv_RMSE": round(float(cv_rmse), 4), **_metrics(yte, p)})

    table = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = table.iloc[0]["model"]
    p = np.asarray(preds[best], dtype=float)
    y = np.asarray(yte, dtype=float)

    # per media-type error breakdown for the best model
    by_type = (pd.DataFrame({"type": Xte["type"].to_numpy(), "y": y, "p": p})
               .assign(abs_err=lambda d: (d.y - d.p).abs())
               .groupby("type")[["y", "p", "abs_err"]]
               .agg(n=("abs_err", "size"), mean_actual=("y", "mean"),
                    mean_pred=("p", "mean"), MAE=("abs_err", "mean"))
               .round(3).reset_index())

    # learning curve for the boosted model
    lc = []
    gb = models["Gradient Boosting"]
    for size in range(1000, len(Xtr_e) + 1, max(1000, len(Xtr_e) // 10)):
        s = clone_fit(gb, Xtr_e.iloc[:size], ytr.iloc[:size])
        lc.append({"train_size": size, **{
            "train_RMSE": _metrics(ytr.iloc[:size], s.predict(Xtr_e.iloc[:size]))["RMSE"],
            "test_RMSE": _metrics(yte, s.predict(Xte_e))["RMSE"]}})
    return {
        "table": table, "best_model": best,
        "scatter": pd.DataFrame({"actual": y, "predicted": np.round(p, 3)}),
        "residuals": pd.DataFrame({
            "actual": y, "predicted": np.round(p, 3),
            "residual": np.round(y - p, 3),
            "abs_residual": np.round(np.abs(y - p), 3)}),
        "by_type": by_type,
        "learning_curve": pd.DataFrame(lc),
        "fitted": fitted, "predictions": pd.DataFrame(preds).assign(actual=y),
        "tree_text": export_text(fitted["Decision Tree"],
                                 feature_names=list(Xtr.columns), max_depth=6),
    }


def clone_fit(model, X, y):
    from sklearn.base import clone
    m = clone(model)
    m.fit(X, y)
    return m


def feature_importance(X, y) -> pd.DataFrame:
    X_e, _ = _encode(X, [])
    rf = RandomForestRegressor(n_estimators=250, max_depth=14, n_jobs=-1,
                               random_state=C.RANDOM_STATE).fit(X_e, y)
    imp = pd.DataFrame({
        "attribute": X.columns,
        "importance": np.round(rf.feature_importances_, 4),
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    imp["rank"] = np.arange(1, len(imp) + 1)
    imp["cumulative"] = imp["importance"].cumsum().round(4)
    return imp
