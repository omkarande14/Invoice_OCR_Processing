# app/rrsi/stats.py
"""
Statistical Utilities for RRSI Evaluation & Selection Gate.
1. Wilson Score Intervals for single proportions (never Wald).
2. Paired Cluster Bootstrap over invoice template groups (10,000 resamples).
3. McNemar Test for paired binary field outcomes.
"""

import math
from typing import Dict, Any, List, Tuple, Union, Optional
import numpy as np
from scipy import stats


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """
    Computes the Wilson score interval for a binomial proportion.
    Never uses Wald (p +/- z*sqrt(p(1-p)/n)), which breaks down near 0/1 and on small sample sizes.

    Returns:
        (ci_lower, ci_upper) as float in [0.0, 1.0].
    """
    if total <= 0:
        return 0.0, 0.0

    p = successes / total
    # Two-sided normal quantile
    alpha = 1.0 - confidence
    z = stats.norm.ppf(1.0 - alpha / 2.0)

    denominator = 1.0 + (z**2) / total
    center_adjusted = (p + (z**2) / (2.0 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p * (1.0 - p) / total) + (z**2) / (4.0 * (total**2)))

    if successes == 0:
        ci_lower = 0.0
    else:
        ci_lower = max(0.0, float(center_adjusted - margin))

    if successes == total:
        ci_upper = 1.0
    else:
        ci_upper = min(1.0, float(center_adjusted + margin))

    return float(ci_lower), float(ci_upper)


def paired_cluster_bootstrap(
    base_per_doc: Union[Dict[str, float], List[float]],
    cand_per_doc: Union[Dict[str, float], List[float]],
    groups: Union[Dict[str, str], List[str]],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs a paired cluster bootstrap across document template groups.
    Resamples clusters (template groups) with replacement to account for intra-template correlation.

    Args:
        base_per_doc: Mapping or list of doc_id -> metric score for baseline.
        cand_per_doc: Mapping or list of doc_id -> metric score for candidate.
        groups: Mapping or list of doc_id -> template group id.
        n_resamples: Number of bootstrap iterations (default 10000).
        confidence: Confidence level (default 0.95).
        seed: Random seed for reproducibility.

    Returns:
        dict with:
            mean_delta: Mean candidate - baseline delta.
            ci_lower: Lower bound of bootstrap percentile CI.
            ci_upper: Upper bound of bootstrap percentile CI.
            std_err: Standard error of the bootstrap deltas.
            p_value: Empirical two-sided p-value.
            n_clusters: Number of distinct template groups.
            n_docs: Number of documents evaluated.
    """
    rng = np.random.default_rng(seed)

    # Standardize to list of tuples: (doc_id, base_score, cand_score, group_id)
    if isinstance(base_per_doc, dict) and isinstance(cand_per_doc, dict):
        common_ids = [d for d in base_per_doc if d in cand_per_doc]
        if not common_ids:
            return {
                "mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
                "std_err": 0.0, "p_value": 1.0, "n_clusters": 0, "n_docs": 0
            }
        group_mapping = groups if isinstance(groups, dict) else {d: str(i) for i, d in enumerate(common_ids)}
        doc_records = [
            (d, float(base_per_doc[d]), float(cand_per_doc[d]), str(group_mapping.get(d, "default_grp")))
            for d in common_ids
        ]
    else:
        base_list = list(base_per_doc)
        cand_list = list(cand_per_doc)
        n = min(len(base_list), len(cand_list))
        grp_list = list(groups) if groups is not None else [f"grp_{i}" for i in range(n)]
        doc_records = [
            (str(i), float(base_list[i]), float(cand_list[i]), str(grp_list[i]))
            for i in range(n)
        ]

    n_docs = len(doc_records)
    if n_docs == 0:
        return {
            "mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0,
            "std_err": 0.0, "p_value": 1.0, "n_clusters": 0, "n_docs": 0
        }

    # Group documents by cluster
    clusters: Dict[str, List[Tuple[float, float]]] = {}
    for _, b_score, c_score, grp in doc_records:
        clusters.setdefault(grp, []).append((b_score, c_score))

    cluster_keys = list(clusters.keys())
    n_clusters = len(cluster_keys)

    # Calculate actual observed mean delta
    all_b = np.array([r[1] for r in doc_records])
    all_c = np.array([r[2] for r in doc_records])
    mean_delta = float(np.mean(all_c) - np.mean(all_b))

    # Fast path: if candidate and baseline are strictly identical across all documents
    if np.allclose(all_b, all_c):
        return {
            "mean_delta": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "std_err": 0.0,
            "p_value": 1.0,
            "n_clusters": n_clusters,
            "n_docs": n_docs
        }

    # Convert clusters into arrays of base & cand sums and sizes
    cluster_b_sums = np.array([sum(x[0] for x in clusters[k]) for k in cluster_keys])
    cluster_c_sums = np.array([sum(x[1] for x in clusters[k]) for k in cluster_keys])
    cluster_sizes = np.array([len(clusters[k]) for k in cluster_keys])

    # Sample cluster indices with replacement (n_resamples x n_clusters)
    sampled_indices = rng.integers(0, n_clusters, size=(n_resamples, n_clusters))

    sample_b_totals = np.take(cluster_b_sums, sampled_indices).sum(axis=1)
    sample_c_totals = np.take(cluster_c_sums, sampled_indices).sum(axis=1)
    sample_doc_counts = np.take(cluster_sizes, sampled_indices).sum(axis=1)

    # Avoid division by zero
    valid_mask = sample_doc_counts > 0
    bootstrap_deltas = np.zeros(n_resamples, dtype=float)
    bootstrap_deltas[valid_mask] = (
        (sample_c_totals[valid_mask] - sample_b_totals[valid_mask]) / sample_doc_counts[valid_mask]
    )

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(bootstrap_deltas, lower_pct))
    ci_upper = float(np.percentile(bootstrap_deltas, upper_pct))
    std_err = float(np.std(bootstrap_deltas))

    # Empirical two-sided p-value
    p_leq_0 = np.mean(bootstrap_deltas <= 0.0)
    p_geq_0 = np.mean(bootstrap_deltas >= 0.0)
    p_value = float(min(1.0, 2.0 * min(p_leq_0, p_geq_0)))

    return {
        "mean_delta": mean_delta,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "p_value": p_value,
        "n_clusters": n_clusters,
        "n_docs": n_docs
    }


def mcnemar_test(
    base_outcomes: List[bool],
    cand_outcomes: List[bool]
) -> Dict[str, Any]:
    """
    Computes McNemar's test for paired nominal / binary outcomes (e.g. per-field exact match).
    Uses continuity-corrected chi-square and exact binomial test.

    Args:
        base_outcomes: List of bool correctness flags for baseline.
        cand_outcomes: List of bool correctness flags for candidate.

    Returns:
        dict with:
            b: Baseline correct, candidate incorrect (discordant).
            c: Baseline incorrect, candidate correct (discordant).
            n_discordant: b + c.
            statistic: Continuity-corrected chi-square statistic.
            p_value: Two-sided p-value from exact binomial test.
    """
    n = min(len(base_outcomes), len(cand_outcomes))
    b = 0  # base=1, cand=0
    c = 0  # base=0, cand=1
    a = 0  # base=1, cand=1
    d = 0  # base=0, cand=0

    for i in range(n):
        b_val = bool(base_outcomes[i])
        c_val = bool(cand_outcomes[i])
        if b_val and not c_val:
            b += 1
        elif not b_val and c_val:
            c += 1
        elif b_val and c_val:
            a += 1
        else:
            d += 1

    n_discordant = b + c
    if n_discordant == 0:
        return {
            "a": a, "b": b, "c": c, "d": d,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0
        }

    # Continuity-corrected chi-square: (|b - c| - 1)^2 / (b + c)
    stat = ((abs(b - c) - 1.0) ** 2) / float(n_discordant) if abs(b - c) >= 1 else 0.0

    # Exact two-sided binomial test on discordant pairs under H0: p = 0.5
    binom_res = stats.binomtest(k=b, n=n_discordant, p=0.5, alternative="two-sided")
    p_val = float(binom_res.pvalue)

    return {
        "a": a, "b": b, "c": c, "d": d,
        "n_discordant": n_discordant,
        "statistic": float(stat),
        "p_value": p_val
    }


def cluster_bootstrap_mean(
    scores_per_cluster: Union[Dict[str, float], List[float]],
    n_resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Performs cluster bootstrap for an aggregate metric grouped by cluster (e.g. invoice_id).
    Resamples clusters with replacement to properly account for correlated within-invoice field outcomes.

    Returns:
        dict with:
            mean: Observed sample mean
            ci_lower: Lower bound of bootstrap percentile CI
            ci_upper: Upper bound of bootstrap percentile CI
            std_err: Standard error of bootstrap means
            n_clusters: Number of clusters (invoices)
    """
    rng = np.random.default_rng(seed)
    if isinstance(scores_per_cluster, dict):
        vals = np.array(list(scores_per_cluster.values()), dtype=float)
    else:
        vals = np.array(scores_per_cluster, dtype=float)

    n = len(vals)
    if n == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std_err": 0.0, "n_clusters": 0}
    if n == 1:
        return {"mean": float(vals[0]), "ci_lower": float(vals[0]), "ci_upper": float(vals[0]), "std_err": 0.0, "n_clusters": 1}

    mean_val = float(np.mean(vals))
    sample_indices = rng.integers(0, n, size=(n_resamples, n))
    resampled_means = np.take(vals, sample_indices).mean(axis=1)

    alpha = 1.0 - confidence
    lower_pct = 100.0 * (alpha / 2.0)
    upper_pct = 100.0 * (1.0 - alpha / 2.0)

    ci_lower = float(np.percentile(resampled_means, lower_pct))
    ci_upper = float(np.percentile(resampled_means, upper_pct))
    std_err = float(np.std(resampled_means))

    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_err": std_err,
        "n_clusters": n
    }

