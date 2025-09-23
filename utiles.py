"""
Utility functions for supervised customer segmentation and evaluation.

These helpers are extracted from the notebook so they can be reused across
experiments. They cover data preparation, decision-tree based clustering,
confidence intervals, profiling, statistical tests, and readable tree rules.

Dependencies:
- numpy, pandas
- scikit-learn (DecisionTreeClassifier, OrdinalEncoder, metrics)
- scipy (stats)
- statsmodels (proportions_ztest) for pairwise z-tests
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

import math
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

from scipy import stats
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder

try:
    # Optional but recommended for two-proportion z-tests
    from statsmodels.stats.proportion import proportions_ztest
except Exception:  # pragma: no cover - fallback if statsmodels missing
    proportions_ztest = None  # type: ignore


# -----------------------------
# Data preparation
# -----------------------------
def prepare_features(
    df: pd.DataFrame,
    categorical_cols: Sequence[str],
    numeric_cols: Sequence[str],
    target_col: str = "target",
    encoder: Optional[OrdinalEncoder] = None,
    handle_unknown: str = "use_encoded_value",
    unknown_value: int = -1,
) -> Tuple[np.ndarray, np.ndarray, List[str], OrdinalEncoder]:
    """Prepare model matrix X, target y, and feature names with ordinal encoding.

    Returns (X, y, feature_names, fitted_encoder).
    """
    if encoder is None:
        encoder = OrdinalEncoder(
            handle_unknown=handle_unknown, unknown_value=unknown_value
        )
        X_cat = encoder.fit_transform(df[categorical_cols])
    else:
        X_cat = encoder.transform(df[categorical_cols])

    X_num = df[numeric_cols].to_numpy()
    X = np.hstack([X_num, X_cat])
    y = df[target_col].values
    feature_names = list(numeric_cols) + list(categorical_cols)
    return X, y, feature_names, encoder


# -----------------------------
# Modeling
# -----------------------------
def fit_tree_for_clusters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    *,
    max_leaf_nodes: int = 8,
    min_samples_leaf: int = 500,
    max_depth: Optional[int] = None,
    criterion: str = "gini",
    random_state: int = 42,
    class_weight: Optional[Dict[int, float]] = None,
) -> Tuple[DecisionTreeClassifier, Optional[float]]:
    """Fit a DecisionTreeClassifier intended for supervised clustering.

    Returns (fitted_tree, auc_on_val_or_None).
    """
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        criterion=criterion,
        random_state=random_state,
        class_weight=class_weight,
    )
    tree.fit(X_train, y_train)
    auc = None
    if X_val is not None and y_val is not None:
        val_preds = tree.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, val_preds)
    return tree, auc


def evaluate_leaf_bad_rates(
    tree: DecisionTreeClassifier,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    X_all: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, np.ndarray, Dict[int, int]]:
    """Evaluate leaf-level bad rates on an evaluation set and map leaves to clusters.

    - Creates a leaf stats DataFrame (sorted by bad_rate ascending) with Wilson CIs.
    - Assigns rank-based cluster labels 1..N (low to high bad rate).
    - Returns (leaf_df, clusters_all, mapping) where clusters_all is computed on X_all
      if provided, else on X_eval.
    """
    leaf_eval = tree.apply(X_eval)
    stats_list: List[Dict[str, Any]] = []
    for leaf_id in np.unique(leaf_eval):
        mask = leaf_eval == leaf_id
        y_leaf = y_eval[mask]
        size = y_leaf.size
        bad_rate = y_leaf.mean() if size > 0 else np.nan
        ci_low, ci_high = proportion_ci_wilson(y_leaf.sum(), size)
        stats_list.append(
            dict(
                leaf_id=int(leaf_id),
                size=int(size),
                pct=float(size / len(y_eval) if len(y_eval) else np.nan),
                bad_rate=float(bad_rate),
                ci_low=float(ci_low),
                ci_high=float(ci_high),
            )
        )
    leaf_df = pd.DataFrame(stats_list).sort_values("bad_rate").reset_index(drop=True)
    leaf_df["cluster"] = np.arange(1, len(leaf_df) + 1)
    mapping: Dict[int, int] = dict(zip(leaf_df.leaf_id, leaf_df.cluster))

    X_assign = X_all if X_all is not None else X_eval
    all_leaves = tree.apply(X_assign)
    clusters_all = np.vectorize(mapping.get)(all_leaves)
    return leaf_df, clusters_all, mapping


# -----------------------------
# Statistics
# -----------------------------
def proportion_ci_wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Returns (lower, upper). If n == 0, returns (nan, nan).
    """
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    adj = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2))
    low = (centre - adj) / denom
    high = (centre + adj) / denom
    return low, high


def chi_square_test_cluster_target(
    df: pd.DataFrame, cluster_col: str = "cluster", target_col: str = "target"
) -> Tuple[float, float, int, pd.DataFrame]:
    """Global chi-square test of independence between cluster and target.

    Returns (chi2, p_value, dof, expected_counts_df).
    """
    tab = pd.crosstab(df[cluster_col], df[target_col])
    chi2, p_chi, dof, expected = stats.chi2_contingency(tab)
    expected_df = pd.DataFrame(expected, index=tab.index, columns=tab.columns)
    return float(chi2), float(p_chi), int(dof), expected_df


def _benjamini_hochberg_adjust(p_values: Sequence[float]) -> List[float]:
    """Benjamini-Hochberg FDR adjustment. Returns adjusted p-values.

    Monotonicity is enforced with a reverse cumulative minimum.
    """
    m = len(p_values)
    order = np.argsort(p_values)
    p_sorted = np.array(p_values)[order]
    adj = np.minimum(1.0, p_sorted * m / (np.arange(1, m + 1)))
    # enforce monotonicity from largest rank to smallest
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    # back to original order
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def pairwise_bad_rate_ztests(
    df: pd.DataFrame,
    cluster_col: str = "cluster",
    target_col: str = "target",
    adjust: str = "bh",
) -> pd.DataFrame:
    """Compute all pairwise two-proportion z-tests of bad rates across clusters.

    Returns a DataFrame with columns: cluster_a, cluster_b, p_raw, p_adj (if adjusted), significant_0_05.
    Requires statsmodels for exact matching of notebook's behaviour; otherwise falls back
    to a manual normal-approximation implementation.
    """
    agg = df.groupby(cluster_col)[target_col].agg(["sum", "count"]).rename(
        columns={"sum": "bad", "count": "n"}
    )
    clusters = agg.index.tolist()

    rows: List[Dict[str, Any]] = []
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            a, b = clusters[i], clusters[j]
            x = np.array([agg.loc[a, "bad"], agg.loc[b, "bad"]], dtype=float)
            n = np.array([agg.loc[a, "n"], agg.loc[b, "n"]], dtype=float)

            if proportions_ztest is not None:
                stat, pval = proportions_ztest(count=x, nobs=n)
            else:
                # Manual two-proportion z-test (pooled)
                p1, p2 = x[0] / n[0], x[1] / n[1]
                p_pool = (x[0] + x[1]) / (n[0] + n[1])
                se = math.sqrt(p_pool * (1 - p_pool) * (1 / n[0] + 1 / n[1]))
                stat = (p1 - p2) / se if se > 0 else 0.0
                # two-sided p-value
                pval = 2 * (1 - stats.norm.cdf(abs(stat)))

            rows.append({
                "cluster_a": a,
                "cluster_b": b,
                "p_raw": float(pval),
            })

    out = pd.DataFrame(rows).sort_values("p_raw").reset_index(drop=True)
    if adjust and not out.empty:
        if adjust.lower() in {"bh", "fdr_bh", "benjamini-hochberg"}:
            out["p_adj"] = _benjamini_hochberg_adjust(out["p_raw"].tolist())
        else:
            raise ValueError(f"Unsupported adjust method: {adjust}")
        out["significant_0_05"] = out["p_adj"] < 0.05
    return out


# -----------------------------
# Profiling
# -----------------------------
def compute_cluster_profiles(
    df: pd.DataFrame,
    cluster_col: str,
    target_col: str,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
) -> pd.DataFrame:
    """Compute per-cluster profiles: size, % portfolio, bad rate, Wilson CI, lift, numeric stats, and dominant categories."""
    overall_bad_rate = df[target_col].mean() if len(df) else float("nan")
    profiles: List[Dict[str, Any]] = []
    for c, grp in df.groupby(cluster_col):
        size = len(grp)
        bad = int(grp[target_col].sum())
        bad_rate = grp[target_col].mean() if size > 0 else float("nan")
        ci_low, ci_high = proportion_ci_wilson(bad, size)
        profile: Dict[str, Any] = dict(
            cluster=c,
            size=int(size),
            pct_portfolio=float(size / len(df) if len(df) else float("nan")),
            bad_rate=float(bad_rate),
            ci_low=float(ci_low),
            ci_high=float(ci_high),
            lift_vs_overall=float(bad_rate / overall_bad_rate) if overall_bad_rate and not np.isnan(bad_rate) else float("nan"),
        )
        for col in numeric_cols:
            profile[f"{col}_mean"] = float(grp[col].mean()) if size else float("nan")
            profile[f"{col}_median"] = float(grp[col].median()) if size else float("nan")
        for col in categorical_cols:
            mode_val = grp[col].mode(dropna=True)
            if not mode_val.empty:
                mval = mode_val.iloc[0]
                profile[f"{col}_mode"] = mval
                profile[f"{col}_mode_pct"] = float((grp[col] == mval).mean())
            else:
                profile[f"{col}_mode"] = None
                profile[f"{col}_mode_pct"] = float("nan")
        profiles.append(profile)
    return pd.DataFrame(profiles).sort_values("cluster").reset_index(drop=True)


# -----------------------------
# Readable rules
# -----------------------------
def build_readable_rules(
    tree: DecisionTreeClassifier,
    feature_names: Sequence[str],
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    ordinal_encoder: OrdinalEncoder,
) -> str:
    """Generate a human-readable representation of tree splits, translating
    categorical thresholds back to category sets.
    """
    tree_ = tree.tree_
    lines: List[str] = []
    n_num = len(numeric_cols)

    def cat_split_description(feature_index: int, threshold: float) -> Tuple[List[Any], List[Any]]:
        # index inside categorical encoder arrays
        cat_idx = feature_index - n_num
        categories: List[Any] = list(ordinal_encoder.categories_[cat_idx])
        # Thresholds are at k + 0.5 for integer codes; left side codes <= floor(threshold)
        k = int(math.floor(threshold))
        left_codes = list(range(k + 1))
        left_cats = [categories[c] for c in left_codes if 0 <= c < len(categories)]
        right_cats = [c for c in categories if c not in left_cats]
        return left_cats, right_cats

    def recurse(node: int, depth: int) -> None:
        indent = "  " * depth
        feature_index = tree_.feature[node]
        if feature_index != -2:  # internal node
            fname = feature_names[feature_index]
            threshold = tree_.threshold[node]
            if fname in categorical_cols:
                left_cats, right_cats = cat_split_description(feature_index, threshold)
                lines.append(f"{indent}IF {fname} IN {left_cats}")
                recurse(tree_.children_left[node], depth + 1)
                lines.append(f"{indent}ELSE  # {fname} IN {right_cats}")
                recurse(tree_.children_right[node], depth + 1)
            else:
                lines.append(f"{indent}IF {fname} <= {threshold:.4g}")
                recurse(tree_.children_left[node], depth + 1)
                lines.append(f"{indent}ELSE  # {fname} > {threshold:.4g}")
                recurse(tree_.children_right[node], depth + 1)
        else:
            # Leaf
            value = tree_.value[node][0]
            good, bad = value[0], value[1]
            total = good + bad
            bad_rate = bad / total if total else 0
            lines.append(
                f"{indent}LEAF: samples={int(total)} bad_rate={bad_rate:.2%} bad={int(bad)} good={int(good)}"
            )

    recurse(0, 0)
    return "\n".join(lines)


__all__ = [
    "prepare_features",
    "fit_tree_for_clusters",
    "evaluate_leaf_bad_rates",
    "proportion_ci_wilson",
    "chi_square_test_cluster_target",
    "pairwise_bad_rate_ztests",
    "compute_cluster_profiles",
    "build_readable_rules",
    # inference helpers
    "load_bands_artifacts",
    "assign_risk_band_to_customer",
]


# -----------------------------
# Inference helpers (LogReg bands)
# -----------------------------
def load_bands_artifacts(artifacts_dir: "str | Path") -> Tuple[Any, OrdinalEncoder, np.ndarray]:
    """Load Logistic Regression model, OrdinalEncoder, and band edges.

    Expects the following files inside artifacts_dir:
    - model_logreg.pkl
    - encoder_ordinal.pkl
    - band_edges.txt

    Returns (model, encoder, edges).
    """
    p = Path(artifacts_dir)
    model_path = p / "model_logreg.pkl"
    enc_path = p / "encoder_ordinal.pkl"
    edges_path = p / "band_edges.txt"

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")
    if not enc_path.exists():
        raise FileNotFoundError(f"Missing encoder file: {enc_path}")
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing band edges file: {edges_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(enc_path, "rb") as f:
        encoder: OrdinalEncoder = pickle.load(f)

    edges = np.loadtxt(edges_path, dtype=float)
    edges = np.atleast_1d(edges)
    return model, encoder, edges


def assign_risk_band_to_customer(
    customer: Dict[str, Any] | pd.Series | pd.DataFrame,
    numeric_cols: Sequence[str],
    categorical_cols: Sequence[str],
    *,
    artifacts_dir: Optional["str | Path"] = None,
    model: Optional[Any] = None,
    encoder: Optional[OrdinalEncoder] = None,
    edges: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Assign a new customer to a risk band using saved artifacts.

    Provide either `artifacts_dir` (to load model/encoder/edges) or pass
    `model`, `encoder`, and `edges` directly.

    Inputs:
    - customer: dict/Series/DataFrame with at least the feature columns
      in numeric_cols + categorical_cols. If DataFrame is provided, the first row is used.
    - numeric_cols: order of numeric columns used in training
    - categorical_cols: order of categorical columns used in training

    Returns a dict with keys: 'pd_score' (float), 'risk_band' (int, 1..K), 'edges' (np.ndarray).
    """
    if artifacts_dir is not None:
        model, encoder, edges = load_bands_artifacts(artifacts_dir)
    if model is None or encoder is None or edges is None:
        raise ValueError("Must provide artifacts_dir or model+encoder+edges.")

    # Normalize customer input to a dict of values
    if isinstance(customer, pd.DataFrame):
        if customer.empty:
            raise ValueError("Customer DataFrame is empty.")
        row = customer.iloc[0]
        cust = row.to_dict()
    elif isinstance(customer, pd.Series):
        cust = customer.to_dict()
    elif isinstance(customer, dict):
        cust = customer
    else:
        raise TypeError("customer must be dict, Series, or DataFrame")

    # Validate required fields
    required = list(numeric_cols) + list(categorical_cols)
    missing = [c for c in required if c not in cust]
    if missing:
        raise KeyError(f"Missing required feature(s) in customer: {missing}")

    # Build single-row feature matrix: [numeric ... , encoded categorical ...]
    X_num = np.array([[float(cust[col]) for col in numeric_cols]], dtype=float)
    # Encoder expects a DataFrame with categorical columns
    X_cat = encoder.transform(pd.DataFrame([{col: cust[col] for col in categorical_cols}]))
    X = np.hstack([X_num, X_cat])

    # Predict PD score
    if hasattr(model, "predict_proba"):
        pd_score = float(model.predict_proba(X)[:, 1][0])
    else:
        # Fallback: decision_function -> map via sigmoid
        if hasattr(model, "decision_function"):
            z = float(model.decision_function(X)[0])
            pd_score = 1.0 / (1.0 + math.exp(-z))
        else:
            raise AttributeError("Model does not support predict_proba or decision_function.")

    # Assign to band using [left, right) bins; impute to nearest if outside
    K = int(len(edges) - 1)
    if K <= 0:
        band_code = 0
    else:
        if pd_score < edges[0]:
            band_code = 0
        elif pd_score >= edges[-1]:
            band_code = K - 1
        else:
            band_code = int(np.searchsorted(edges, pd_score, side="right") - 1)
            band_code = min(max(band_code, 0), K - 1)

    result = {
        "pd_score": pd_score,
        "risk_band": int(band_code + 1),  # 1..K
        "edges": edges,
    }
    return result
