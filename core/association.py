"""Data Warehousing & Mining - Stage 2: Association Rule Mining.

Two rule bases are mined with Apriori + FP-Growth and the classic
support / confidence / lift / conviction metrics:

  A. Genre basket   - transaction = anime, item = genre
                     -> interpretable "co-occurring genre" rules
  B. User basket    - transaction = user, item = the top rated anime
                     -> "watched A -> also liked B" recommendation rules
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules, fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from . import config as C
from . import preprocessing as P


def mine_genre_rules(min_support: float = 0.04, min_confidence: float = 0.20,
                     min_lift: float = 1.2, max_len: int = 3) -> pd.DataFrame:
    """Dataset A: which genres are watched together."""
    C.ensure_dirs()
    baskets = pd.read_csv(C.BASKET_GENRE)
    item_cols = [c for c in baskets.columns if c.startswith("g_")]
    baskets = baskets[item_cols].astype(bool)          # boolean indicator matrix
    itemsets = fpgrowth(baskets, min_support=min_support, use_colnames=True)
    rules = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    if rules.empty:
        return rules
    rules = rules[rules["lift"] >= min_lift]
    rules = _decorate(rules, max_len)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules.to_csv(C.RULES_GENRE, index=False)
    return rules

def mine_user_rules(min_support: float = 0.03, min_confidence: float = 0.25,
                    min_lift: float = 1.5, max_len: int = 2) -> pd.DataFrame:
    """Dataset B: anime -> anime recommendation rules from user baskets."""
    C.ensure_dirs()
    tx = P.load_baskets_user(min_items=3)
    if len(tx) < 100:
        return pd.DataFrame()
    oh_df = _onehot(tx["item"].tolist())       # one-hot matrix, in memory only
    itemsets = fpgrowth(oh_df, min_support=min_support, use_colnames=True)
    rules = association_rules(
        itemsets, metric="confidence", min_threshold=min_confidence
    )
    if rules.empty:
        return rules
    rules = rules[rules["lift"] >= min_lift]
    rules = _decorate(rules, max_len)
    rules = rules.sort_values(["lift", "confidence"], ascending=False).reset_index(drop=True)
    rules.to_csv(C.RULES_USER, index=False)
    return rules


def _onehot(transactions: list[list[str]]) -> pd.DataFrame:
    """list-of-item-lists -> boolean indicator DataFrame (in memory only)."""
    te = TransactionEncoder()
    arr = te.fit_transform(transactions)
    return pd.DataFrame(arr, columns=te.columns_).astype(bool)


def _pretty(sets) -> str:
    """`{g_Action, g_Comedy}` -> `Action, Comedy`"""
    items = list(sets)
    if len(items) == 1 and isinstance(items[0], str):
        items = items[0].split(", ")
    return ", ".join(sorted(str(i).replace("g_", "").replace("_", " ") for i in items))


def _decorate(rules: pd.DataFrame, max_len: int) -> pd.DataFrame:
    r = rules.rename(columns={"antecedent support": "antecedent_support",
                              "consequent support": "consequent_support"}).copy()
    for c in ("conviction", "leverage", "zhangs_metric", "jaccard", "certainty",
              "kulczynski", "representativity"):
        if c not in r.columns:
            r[c] = np.nan
    r["antecedents"] = r["antecedents"].apply(_pretty)
    r["consequents"] = r["consequents"].apply(_pretty)
    r["rule"] = r["antecedents"] + "  =>  " + r["consequents"]
    r["n_items"] = r["antecedents"].str.count(",") + r["consequents"].str.count(",") + 2
    r = r[r["n_items"] <= max_len]
    # conviction < 1 => the rule is still wrong more often than not; leverage <= 0
    # => the rule adds nothing over the independence assumption
    for c in ("support", "confidence", "lift", "antecedent_support",
              "consequent_support", "conviction", "leverage", "zhangs_metric",
              "jaccard", "certainty", "kulczynski"):
        r[c] = r[c].astype(float).round(5)
    cols = ["rule", "antecedents", "consequents", "n_items", "support", "confidence",
            "lift", "antecedent_support", "consequent_support", "conviction",
            "leverage", "zhangs_metric", "certainty", "kulczynski", "jaccard"]
    return r[cols]


def apriori_vs_fpgrowth(min_support_list=(0.02, 0.03, 0.05, 0.08)) -> pd.DataFrame:
    """Timing comparison of the two frequent-itemset algorithms (Apriori vs FP-Growth)."""
    import time

    rows = []
    tx = P.load_baskets_user(min_items=3)
    tx_df = _onehot(tx["item"].tolist())
    for ms in min_support_list:
        if ms * len(tx_df) < 3:
            continue
        t = time.perf_counter()
        try:
            a = apriori(tx_df, min_support=ms, use_colnames=True)
            ta = time.perf_counter() - t
        except Exception as exc:
            a, ta = pd.DataFrame(), float("nan")
            print("apriori failed:", exc)
        t = time.perf_counter()
        f = fpgrowth(tx_df, min_support=ms, use_colnames=True)
        tf = time.perf_counter() - t
        rows.append({
            "min_support": ms, "n_transactions": len(tx_df),
            "apriori_itemsets": len(a), "fpgrowth_itemsets": len(f),
            "apriori_sec": round(ta, 4), "fpgrowth_sec": round(tf, 4),
            "speedup": round(ta / tf, 2) if tf else np.nan,
        })
    return pd.DataFrame(rows)
