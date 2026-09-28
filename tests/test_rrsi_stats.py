# tests/test_rrsi_stats.py
import pytest
from app.rrsi.stats import (
    wilson_interval,
    paired_cluster_bootstrap,
    mcnemar_test
)

def test_wilson_interval_known_values():
    # Known test case: 10 successes out of 20 trials at 95% confidence
    lower, upper = wilson_interval(successes=10, total=20, confidence=0.95)
    assert pytest.approx(lower, abs=1e-3) == 0.2993
    assert pytest.approx(upper, abs=1e-3) == 0.7007
    assert 0.0 <= lower <= upper <= 1.0

def test_wilson_interval_small_n_is_wide():
    # N=3 sample size (e.g. 2 successes out of 3) must give a very wide interval
    lower, upper = wilson_interval(successes=2, total=3, confidence=0.95)
    width = upper - lower
    # At N=3, the 95% Wilson interval width is > 0.60
    assert width > 0.60
    assert lower > 0.15
    assert upper < 0.99

def test_wilson_interval_edge_cases():
    # 0 successes out of 10: Wald gives [0, 0] with margin 0, but Wilson properly gives upper bound > 0
    lower, upper = wilson_interval(successes=0, total=10, confidence=0.95)
    assert lower == 0.0
    assert upper > 0.20
    assert upper <= 0.35

    # 10 successes out of 10
    lower, upper = wilson_interval(successes=10, total=10, confidence=0.95)
    assert upper == 1.0
    assert lower >= 0.65

    # Empty total
    lower, upper = wilson_interval(successes=0, total=0)
    assert lower == 0.0
    assert upper == 0.0

def test_paired_cluster_bootstrap_identical_runs():
    # Identical runs must yield exactly delta=0 and CI [0, 0]
    base_scores = {"doc1": 0.8, "doc2": 0.9, "doc3": 0.7, "doc4": 0.85}
    cand_scores = {"doc1": 0.8, "doc2": 0.9, "doc3": 0.7, "doc4": 0.85}
    groups = {"doc1": "grp_a", "doc2": "grp_a", "doc3": "grp_b", "doc4": "grp_b"}

    res = paired_cluster_bootstrap(base_scores, cand_scores, groups, n_resamples=1000, seed=42)
    assert pytest.approx(res["mean_delta"], abs=1e-6) == 0.0
    assert pytest.approx(res["ci_lower"], abs=1e-6) == 0.0
    assert pytest.approx(res["ci_upper"], abs=1e-6) == 0.0
    assert pytest.approx(res["std_err"], abs=1e-6) == 0.0

def test_paired_cluster_bootstrap_positive_delta():
    base_scores = {"doc1": 0.6, "doc2": 0.7, "doc3": 0.5, "doc4": 0.6}
    cand_scores = {"doc1": 0.8, "doc2": 0.9, "doc3": 0.7, "doc4": 0.8}
    groups = {"doc1": "grp_a", "doc2": "grp_a", "doc3": "grp_b", "doc4": "grp_b"}

    res = paired_cluster_bootstrap(base_scores, cand_scores, groups, n_resamples=1000, seed=42)
    assert pytest.approx(res["mean_delta"], abs=1e-3) == 0.2
    assert res["ci_lower"] > 0.0
    assert res["ci_upper"] > res["ci_lower"]

def test_mcnemar_test_known_values():
    # Contingency table:
    # base=1, cand=1: 50
    # base=1, cand=0: 5 (b)
    # base=0, cand=1: 20 (c)
    # base=0, cand=0: 25
    base = [True] * 50 + [True] * 5 + [False] * 20 + [False] * 25
    cand = [True] * 50 + [False] * 5 + [True] * 20 + [False] * 25

    res = mcnemar_test(base, cand)
    assert res["b"] == 5
    assert res["c"] == 20
    assert res["n_discordant"] == 25
    # Chi-square with continuity correction: (|5 - 20| - 1)^2 / (5 + 20) = 14^2 / 25 = 196 / 25 = 7.84
    assert pytest.approx(res["statistic"], abs=1e-2) == 7.84
    assert res["p_value"] < 0.01

def test_mcnemar_test_zero_discordant():
    base = [True, False, True]
    cand = [True, False, True]
    res = mcnemar_test(base, cand)
    assert res["b"] == 0
    assert res["c"] == 0
    assert res["p_value"] == 1.0
