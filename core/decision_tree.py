"""J48 / C4.5 decision tree implemented from scratch.

`sklearn.tree.DecisionTreeClassifier` optimises information gain (ID3-style);
J48 = C4.5 optimises *gain ratio*, which penalises attributes with many
distinct values and therefore avoids over-fitting on high-cardinality columns.

Implementing it explicitly is the point of the exercise, so this module is a
faithful C4.5:

  * continuous attributes are **pre-discretised once** into `MAX_BINS` quantile
    intervals (C4.5's FindThreshold + discretization step), so a node splits on
    the *interval* of an attribute, not on a fresh threshold recomputed inside
    that node. This is what makes the extracted rules readable: a path can only
    ever walk down the interval chain of one attribute.
  * the split criterion is InformationGainRatio = Gain(a) / SplitInfo(a).
  * the tree is grown to the maximum depth allowed and only **then** pruned
    bottom-up with the confidence factor, because a locally useless split can
    still contain a deeper split that pays off.
  * the pruned tree is exported as IF/THEN classification rules.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

MAX_BINS = 10


# ------------------------------------------------------------------ utils --
def _entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _split_info(weights: np.ndarray) -> float:
    """Entropy of the branch distribution - the denominator of the gain ratio."""
    w = weights[weights > 0]
    if w.size == 0:
        return 0.0
    frac = w / w.sum()
    return float(-(frac * np.log2(frac)).sum())


def _leaf(node: Node) -> Node:
    """Collapse an internal node into a pure leaf, keeping its label."""
    return Node(n_samples=node.n_samples, n_correct=node.n_correct,
                majority_class=node.majority_class, value=node.value)


@dataclass
class Node:
    is_leaf: bool = True
    feature: str | None = None
    bin_index: int | None = None       # for discrete attributes
    categories: tuple | None = None    # None => the split is on a bin interval
    gain_ratio: float = 0.0
    gain: float = 0.0
    value: np.ndarray = field(default_factory=lambda: np.zeros(1))
    majority_class: int = 0
    n_samples: int = 0
    n_correct: int = 0
    children: dict | None = None


class C45Tree(BaseEstimator, ClassifierMixin):
    """C4.5 / J48 classifier with confidence-factor pruning.

    Subclasses scikit-learn's estimator API so it can be cloned, cross-validated
    and dropped into a Pipeline like any other learner.
    """

    def __init__(self, min_samples_split: int = 20, min_samples_leaf: int = 10,
                 max_depth: int = 8, confidence: float = 0.25):
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.confidence = confidence
        self.root: Node | None = None
        self.classes_: list = []
        self.classes_to_idx: dict = {}
        self.n_classes_ = 0
        self.numeric_features_: list = []
        self.categorical_features_: list = []
        self.bin_edges: dict = {}          # feature -> (B+1,) interval boundaries
        self.raw_levels: dict = {}         # feature -> its observed values, in order
        self.cat_labels: dict = {}         # categorical feature -> level labels
        self._encoding: dict = {}
        self._X_fit = None
        self._raw_fit = None
        self._y_fit = None

    # --------------------------------------------------------- discretising --
    def _fit_bins(self, X: pd.DataFrame) -> None:
        self.bin_edges, self.raw_levels = {}, {}
        for col in X.columns:
            if col in self.categorical_features_:
                continue
            x = X[col].to_numpy(dtype=float)
            uniq = np.unique(x)
            self.raw_levels[col] = uniq
            if uniq.size <= MAX_BINS:
                # already discrete: one interval per observed value
                edges = np.concatenate([[uniq[0] - 0.5] if uniq.size else [0.0],
                                        (uniq[:-1] + uniq[1:]) / 2 if uniq.size > 1 else [],
                                        [uniq[-1] + 0.5] if uniq.size else []])
                edges = np.unique(edges) if uniq.size else np.array([0.0, 1.0])
            else:
                qs = np.quantile(x, np.linspace(0, 1, MAX_BINS + 1))
                edges = np.unique(np.round(qs, 6))
                if edges.size < 2:
                    edges = np.array([x.min() - 0.5, x.max() + 0.5])
            self.bin_edges[col] = edges

    def _bin_of(self, col: str, values: np.ndarray) -> np.ndarray:
        """Map raw values onto interval indices 0..B-1 (clipped into range)."""
        edges = self.bin_edges[col]
        idx = np.searchsorted(edges, values, side="right") - 1
        return np.clip(idx, 0, edges.size - 2).astype(int)

    def _prepare(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Ordinal-encode categoricals, then discretise every numeric attribute."""
        X = X.copy()
        if fit:
            self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_features_ = [c for c in X.columns
                                          if c not in self.numeric_features_]
            self.cat_labels = {
                c: sorted(X[c].astype(str).unique()) for c in self.categorical_features_
            }
            self._encoding = {
                c: {v: i for i, v in enumerate(self.cat_labels[c])}
                for c in self.categorical_features_
            }
        for c in self.categorical_features_:
            X[c] = X[c].astype(str).map(self._encoding[c]).fillna(0).astype(int)
        if fit:
            self._fit_bins(X)
        for c in self.numeric_features_:
            X[c] = self._bin_of(c, X[c].to_numpy(dtype=float))
        return X

    # ---------------------------------------------------------------- build --
    @staticmethod
    def _gain_ratio(y: np.ndarray, x: np.ndarray, n_classes: int) -> tuple[float, float]:
        """(gain ratio, information gain) of splitting y on the discrete x."""
        counts = np.bincount(y, minlength=n_classes).astype(float)
        h = _entropy(counts)
        if h == 0.0:
            return 0.0, 0.0
        total = y.size
        groups = np.bincount(x, minlength=int(x.max()) + 1).astype(float)
        info = _split_info(groups)
        if info <= 0:
            return 0.0, 0.0
        rem = 0.0
        for b in np.flatnonzero(groups):
            m = x == b
            c = np.bincount(y[m], minlength=n_classes).astype(float)
            rem += (groups[b] / total) * _entropy(c)
        gain = h - rem
        return (float(gain / info) if gain > 0 else 0.0), float(gain)

    def _build(self, X: pd.DataFrame, y: np.ndarray, depth: int) -> Node:
        counts = np.bincount(y, minlength=self.n_classes_).astype(float)
        maj = int(np.argmax(counts))
        node = Node(n_samples=int(y.size), n_correct=int(counts[maj]),
                    majority_class=maj, value=counts)
        if depth >= self.max_depth or y.size < self.min_samples_split:
            return node
        if counts[maj] == y.size:                      # pure node
            return node

        best = None
        for col in X.columns:
            x = X[col].to_numpy(dtype=int)
            if np.unique(x).size <= 1 or np.unique(x).size >= y.size:
                continue
            gr, ig = self._gain_ratio(y, x, self.n_classes_)
            if gr <= 0:
                continue
            if best is None or gr > best[0]:
                best = (gr, ig, col, tuple(int(v) for v in np.unique(x)))

        if best is None:
            return node
        gr, ig, col, bins = best
        node.feature, node.gain_ratio, node.gain = col, gr, ig
        node.categories = None if col in self.numeric_features_ else bins
        node.bin_index = None if node.categories is None else bins[0]
        x = X[col].to_numpy(dtype=int)
        groups = {b: x == b for b in bins}

        node.children, too_small = {}, False
        for b, m in groups.items():
            node.children[b] = self._build(X[m].reset_index(drop=True), y[m], depth + 1)
            if m.sum() < self.min_samples_leaf:
                too_small = True
        if len(node.children) < 2 or too_small:
            return _leaf(node)
        # No pruning here: C4.5 grows the full tree first and prunes bottom-up.
        node.is_leaf = False
        return node

    # ------------------------------------------------------------- pruning ---
    def prune(self, node: Node | None = None) -> Node:
        """Subtree pruning with the confidence factor, applied bottom-up."""
        node = self.root if node is None else node
        if node.is_leaf:
            return node
        for child in node.children.values():
            self.prune(child)
        correct = sum(c.n_correct for c in node.children.values())
        if self._pessimistic_error(correct, node.n_samples) >= \
                self._pessimistic_error(node.n_correct, node.n_samples):
            return _leaf(node)
        return node

    def _pessimistic_error(self, correct: int, n: int) -> float:
        """J48's confidence factor: correct/n pulled down by one standard error."""
        if n == 0:
            return 0.0
        p = correct / n
        se = np.sqrt(max(p * (1 - p), 1e-9) / n)
        return p - self.confidence * se

    # ----------------------------------------------------------------- API ---
    def fit(self, X: pd.DataFrame, y) -> "C45Tree":
        self._raw_fit = X
        X = self._prepare(X, fit=True)
        y = pd.Series(np.asarray(y)).astype(str)
        self.classes_ = sorted(y.unique().tolist())
        self.classes_to_idx = {c: i for i, c in enumerate(self.classes_)}
        yi = y.map(self.classes_to_idx).to_numpy(dtype=int)
        self.n_classes_ = len(self.classes_)
        self._X_fit, self._y_fit = X, y.to_numpy()
        self.root = self._build(X, yi, 0)
        if self.confidence > 0:
            self.prune()
        return self

    def _walk(self, X: pd.DataFrame) -> np.ndarray:
        D = self._prepare(X)
        out = np.empty(len(D), dtype=object)
        for i in range(len(D)):
            node, row = self.root, D.iloc[i]
            while not node.is_leaf:
                node = node.children.get(int(row[node.feature]), node.children[
                    min(node.children)])
            out[i] = node.majority_class
        return out

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "root")
        return np.array([self.classes_[i] for i in self._walk(X)], dtype=object).astype(str)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        check_is_fitted(self, "root")
        D = self._prepare(X)
        P = np.zeros((len(D), self.n_classes_))
        for i in range(len(D)):
            node, row = self.root, D.iloc[i]
            while not node.is_leaf:
                node = node.children.get(int(row[node.feature]), node.children[
                    min(node.children)])
            tot = node.value.sum()
            P[i] = node.value / tot if tot else np.ones(self.n_classes_) / self.n_classes_
        return P

    # ------------------------------------------------------------ reporting --
    def stats(self) -> dict:
        leaves = internals = depth = 0
        stack = [(self.root, 1)]
        while stack:
            node, d = stack.pop()
            depth = max(depth, d)
            if node.is_leaf:
                leaves += 1
            else:
                internals += 1
                stack.extend((c, d + 1) for c in node.children.values())
        return {"leaves": leaves, "internal_nodes": internals, "depth": depth,
                "n_rules": self.count_paths(), "n_bins": MAX_BINS,
                "n_samples": self.root.n_samples,
                "train_accuracy": self.train_accuracy()}

    def train_accuracy(self) -> float:
        return float((self.predict(self._raw_fit) == self._y_fit).mean()) \
            if getattr(self, "_raw_fit", None) is not None else float("nan")

    def count_paths(self) -> int:
        return _count_paths(self.root)

    def text_tree(self, max_rules: int = 500) -> str:
        return render_text(self.root, self.classes_, self, max_rules)

    def to_rules(self, min_conf: float = 0.5) -> pd.DataFrame:
        return extract_rules(self.root, self.classes_, self, min_conf=min_conf)


def _count_paths(node: Node) -> int:
    if node.is_leaf:
        return 1
    return sum(_count_paths(c) for c in node.children.values())


# ------------------------------------------------------------- rendering ---
def _condition(feature: str, key, node: Node, tree: "C45Tree") -> str:
    """Human-readable form of one branch condition."""
    k = int(key)
    if feature in tree.cat_labels:
        labels = tree.cat_labels[feature]
        return f"{feature} = {labels[k] if k < len(labels) else 'other'}"
    levels = tree.raw_levels.get(feature)
    edges = tree.bin_edges.get(feature)
    if levels is not None and len(levels) <= 2:
        # a 0/1 flag reads far better as an equality than as an interval
        return f"{feature} = {_num(levels[k] if k < len(levels) else levels[-1])}"
    if edges is not None:
        lo = edges[k] if k < len(edges) else edges[-1]
        hi = edges[k + 1] if k + 1 < len(edges) else edges[-1]
        return f"{feature} in [{_num(lo)}, {_num(hi)})"
    return f"{feature} = level {k}"


def _num(v) -> str:
    if isinstance(v, (int, np.integer)):
        return f"{int(v)}"
    v = float(v)
    return f"{int(round(v))}" if abs(v - round(v)) < 1e-9 else f"{v:.3g}"


def render_text(node: Node, classes: list, tree: 'C45Tree',
                max_rules: int = 500) -> str:
    lines: list[str] = []
    counter = [0]

    def walk(n: Node, cond: str):
        if counter[0] >= max_rules:
            return
        p = n.n_correct / n.n_samples if n.n_samples else 0.0
        if n.is_leaf:
            counter[0] += 1
            lines.append(
                f"RULE {counter[0]:>3}: IF {cond or 'TRUE'} "
                f"THEN rating_tier = {classes[n.majority_class]}  "
                f"[support={n.n_samples}, confidence={p:.3f}]"
            )
            return
        for key, child in n.children.items():
            edge = _condition(n.feature, key, n, tree)
            walk(child, f"{cond} AND {edge}" if cond else edge)

    walk(node, "")
    return "\n".join(lines)


def extract_rules(node: Node, classes: list, tree: "C45Tree",
                  min_conf: float = 0.5) -> pd.DataFrame:
    """Export the pruned tree as IF/THEN classification rules."""
    total = node.n_samples or 1
    rows: list[dict] = []
    stack = [(node, [])]
    while stack:
        n, cond = stack.pop()
        conf = n.n_correct / n.n_samples if n.n_samples else 0.0
        if n.is_leaf:
            if conf >= min_conf and cond:
                rows.append({
                    "conditions": " AND ".join(cond),
                    "antecedent": cond[0],
                    "n_conditions": len(cond),
                    "conclusion": f"rating_tier = {classes[n.majority_class]}",
                    "confidence": round(conf, 4),
                    "support_abs": int(n.n_samples),
                    "support_pct": round(100 * n.n_samples / total, 2),
                    "error_count": int(n.n_samples - n.n_correct),
                })
            continue
        for key, child in n.children.items():
            stack.append((child, cond + [_condition(n.feature, key, n, tree)]))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = (df.sort_values(["confidence", "support_abs"], ascending=False)
                .drop_duplicates(subset=["conditions", "conclusion"])
                .reset_index(drop=True))
        df.insert(0, "rule_id", np.arange(1, len(df) + 1))
    return df
