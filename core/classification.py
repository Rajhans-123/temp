"""Data Warehousing & Mining - Stage 3: Classification (J48 and Naive Bayes)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text

from . import config as C
from .decision_tree import C45Tree

# Every column the learners are allowed to see. Only genuinely distinct
# attributes are listed: `members_log` and `members_per_episode` are monotone
# transforms of `members`, and letting a tree split on all three just burns
# depth on redundant, unreadable rules.
NUMERIC_FEATURES = [
    "episodes", "members", "genre_count", "is_multi_genre",
    "is_shounen", "is_shoujo", "is_seinen", "has_isekai",
]
CATEGORICAL_FEATURES = ["type", "primary_genre", "members_band", "episodes_band"]


def feature_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cols += [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return cols


def load_classification_data() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    tr = pd.read_csv(C.CLS_SPLIT)
    te = pd.read_csv(C.CLS_SPLIT.with_name("classification_test.csv"))
    cols = feature_columns(tr)
    return tr[cols], tr["target"], te[cols]


def _encode(X_fit: pd.DataFrame, Xs: list[pd.DataFrame]) -> tuple[pd.DataFrame, list]:
    """Ordinal-encode categoricals for the sklearn models (trees accept ints)."""
    enc = {}
    out_fit = X_fit.copy()
    for c in X_fit.columns:
        if out_fit[c].dtype == object or str(out_fit[c].dtype) in ("string", "str"):
            codes, uniques = pd.factorize(out_fit[c].astype(str), sort=True)
            enc[c] = {v: i for i, v in enumerate(uniques)}
            out_fit[c] = codes
    outs = [out_fit]
    for X in Xs:
        o = X.copy()
        for c, m in enc.items():
            o[c] = o[c].astype(str).map(m).fillna(-1).astype(int)
        outs.append(o)
    return outs[0], outs[1:]


# =========================================================== J48 / C4.5 =====
def run_j48(Xtr, ytr, Xte, yte, max_depth: int = 7, min_leaf: int = 10,
            confidence: float = 0.25) -> dict:
    """Train J48 (own C4.5) and sklearn's entropy tree, compare, return rules."""
    j48 = C45Tree(max_depth=max_depth, min_samples_leaf=min_leaf,
                  confidence=confidence).fit(Xtr, ytr)
    pred_j48 = j48.predict(Xte)

    Xtr_e, (Xte_e,) = _encode(Xtr, [Xte])
    sk = DecisionTreeClassifier(criterion="entropy", max_depth=max_depth,
                                min_samples_leaf=min_leaf, random_state=C.RANDOM_STATE)
    sk.fit(Xtr_e, ytr)
    pred_sk = sk.predict(Xte_e)
    sk_text = export_text(sk, feature_names=list(Xtr.columns), max_depth=max_depth)

    rules = j48.to_rules(min_conf=0.5)
    res = {
        "j48": _score(yte, pred_j48, j48.classes_),
        "sklearn_entropy_tree": _score(yte, pred_sk, sorted(ytr.unique())),
        "j48_tree_text": j48.text_tree(),
        "sklearn_tree_text": sk_text,
        "j48_stats": j48.stats(),
        "rules": rules,
        "j48_model": j48,
    }
    if not rules.empty:
        rules.to_csv(C.REPORT_DIR / "j48_rules.csv", index=False)
    return res


# ========================================================== NAIVE BAYES =====
def run_naive_bayes(Xtr, ytr, Xte, yte) -> dict:
    """Gaussian NB on the standardised numeric attributes + discrete NB on the
    one-hot encoded categoricals. Both are reported, as in the textbook."""
    le = LabelEncoder().fit(ytr)
    # split the *raw* frame into numeric / categorical before any encoding
    num = Xtr.select_dtypes(include=[np.number]).columns.tolist()
    cat = [c for c in Xtr.columns if c not in num]
    Xtr_e, (Xte_e,) = _encode(Xtr, [Xte])

    gn_pipe = Pipeline([
        ("prep", ColumnTransformer([("num", StandardScaler(), num)],
                                   remainder="passthrough")),
        ("gnb", GaussianNB()),
    ])
    gn_pipe.fit(Xtr_e, le.transform(ytr))
    pred_gnb = le.inverse_transform(gn_pipe.predict(Xte_e))

    # categorical part -> discrete Naive Bayes over one-hot encoded attributes
    from sklearn.naive_bayes import CategoricalNB
    dummies = pd.get_dummies(
        pd.concat([Xtr[cat], Xte[cat]], ignore_index=True).astype("category"),
        dtype=np.int8,
    )
    n_tr = len(Xtr)
    dnb = CategoricalNB(alpha=1.0).fit(dummies.iloc[:n_tr], le.transform(ytr))
    pred_dnb = le.inverse_transform(dnb.predict(dummies.iloc[n_tr:]))

    return {
        "gaussian_nb": _score(yte, pred_gnb, sorted(ytr.unique())),
        "categorical_nb": _score(yte, pred_dnb, sorted(ytr.unique())),
        "model": gn_pipe, "label_encoder": le,
        "x_columns": list(Xtr_e.columns), "numeric": num, "categorical": cat,
        "predictions": pd.DataFrame({
            "actual": np.asarray(yte).astype(str),
            "gaussian_nb": pred_gnb,
            "categorical_nb": pred_dnb,
        }),
    }


# ============================================================== REPORTING ===
def _score(y_true, y_pred, classes) -> dict:
    y_true = np.asarray(y_true).astype(str)
    y_pred = np.asarray(y_pred).astype(str)
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=list(classes)),
        "classes": list(classes),
        "report": classification_report(y_true, y_pred, labels=list(classes),
                                         zero_division=0, output_dict=True),
    }


def _as_int_frame(X: pd.DataFrame, mapping: dict | None = None) -> pd.DataFrame:
    """Ordinal-encode categoricals to ints for the sklearn learners.

    `mapping` is fitted on the training frame and applied unchanged to the test
    frame. Fitting the two frames separately would give a category different
    codes in train and test, so a model would be scored against labels it never
    saw. Unknown test categories become -1. If `mapping` is None a new one is
    fitted (only safe for a frame that is never split).
    """
    X = X.copy()
    mapping = {} if mapping is None else mapping
    for c in X.columns:
        if X[c].dtype == object or str(X[c].dtype) in ("string", "str"):
            if c not in mapping:
                mapping[c] = {v: i for i, v in enumerate(sorted(X[c].astype(str).unique()))}
            X[c] = X[c].astype(str).map(mapping[c]).fillna(-1).astype(int)
    return X


def compare_models(Xtr, ytr, Xte, yte, n_splits: int = 5) -> dict:
    """Full model comparison: majority baseline, J48, sklearn tree, NB, RF + 5-fold CV.

    J48 consumes the raw (string) frame; every sklearn learner consumes an
    ordinal-encoded copy of the same attributes, so the comparison is fair.
    """
    le = LabelEncoder().fit(ytr)
    ytr_i, yte_i = le.transform(ytr), le.transform(yte)
    # one shared encoding: fitted on train, applied unchanged to test
    _map: dict = {}
    Xtr_i, Xte_i = _as_int_frame(Xtr, _map), _as_int_frame(Xte, _map)
    num_i = [c for c in Xtr_i.columns if pd.api.types.is_numeric_dtype(Xtr_i[c])]

    models = {
        "Majority baseline": (DummyClassifier(strategy="most_frequent"), "sk"),
        "J48 (C4.5, ours)": (C45Tree(max_depth=7, min_samples_leaf=10), "raw"),
        "Decision Tree (entropy)": (DecisionTreeClassifier(
            criterion="entropy", max_depth=7, min_samples_leaf=10,
            random_state=C.RANDOM_STATE), "sk"),
        "Naive Bayes (Gaussian)": (Pipeline([
            ("prep", ColumnTransformer([("num", StandardScaler(), num_i)],
                                       remainder="passthrough")),
            ("gnb", GaussianNB())]), "sk"),
        "Random Forest": (RandomForestClassifier(
            n_estimators=120, max_depth=12, n_jobs=-1,
            random_state=C.RANDOM_STATE), "sk"),
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=C.RANDOM_STATE)
    rows, preds = [], {}
    for name, (m, kind) in models.items():
        if kind == "raw":
            acc_cv = cross_val_score(clone(m), Xtr, ytr, cv=cv, scoring="accuracy").mean()
            mc = clone(m).fit(Xtr, ytr)
            p = np.asarray(mc.predict(Xte), dtype=str)
            auc = None
        else:
            acc_cv = cross_val_score(clone(m), Xtr_i, ytr_i, cv=cv,
                                     scoring="accuracy", n_jobs=-1).mean()
            mc = clone(m).fit(Xtr_i, ytr_i)
            p = le.inverse_transform(mc.predict(Xte_i))
            try:
                proba = mc.predict_proba(Xte_i)
                auc = round(float(roc_auc_score(yte_i, proba, multi_class="ovr",
                                                average="macro")), 4)
            except Exception:
                auc = None
        sc = _score(yte, p, le.classes_)
        rows.append({"model": name, "cv_accuracy": round(float(acc_cv), 4), **sc,
                     "roc_auc_ovr": auc})
        preds[name] = p

    return {
        "table": pd.DataFrame(rows)[["model", "cv_accuracy", "accuracy", "precision_macro",
                                    "recall_macro", "f1_macro", "roc_auc_ovr"]],
        "predictions": pd.DataFrame(preds).assign(actual=np.asarray(yte).astype(str)),
        "best": max(rows, key=lambda r: r["accuracy"])["model"],
        "label_encoder": le,
    }


def feature_importance_j48(X, y, max_depth: int = 6) -> pd.DataFrame:
    """J48-style gain-ratio ranking aggregated over the tree's splits."""
    t = C45Tree(max_depth=max_depth, min_samples_leaf=20).fit(X, y)
    acc: dict[str, float] = {}
    stack = [t.root]
    while stack:
        n = stack.pop()
        if n.is_leaf:
            continue
        acc[n.feature] = acc.get(n.feature, 0.0) + n.gain_ratio
        stack.extend(n.children.values())
    df = pd.DataFrame(sorted(acc.items(), key=lambda kv: -kv[1]),
                      columns=["attribute", "mean_gain_ratio"])
    df["rank"] = np.arange(1, len(df) + 1)
    df["mean_gain_ratio"] = df["mean_gain_ratio"].round(4)
    df["cumulative"] = (df["mean_gain_ratio"] / df["mean_gain_ratio"].sum()).round(4).cumsum()
    return df
