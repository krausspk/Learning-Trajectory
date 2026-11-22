import itertools

import numpy as np
import pytest
from scipy.stats import t

from learning_trajectory.statistics import (
    Z95, correlation_summary, group_summary, jackknife, mean_t_interval,
)


def pairs(n):
    return np.array(list(itertools.combinations(range(n), 2)))


def test_pair_average_jackknife_recovers_independent_mean_standard_error():
    x = np.array([1., 4., 2., 7., 9., 3.])
    p = pairs(len(x))
    v = x[p].mean(axis=1)
    result = jackknife(p, v, np.zeros(len(x)))
    assert result["estimate"] == pytest.approx(x.mean())
    assert result["se"] == pytest.approx(x.std(ddof=1) / np.sqrt(len(x)))
    assert result["ci_low"] == pytest.approx(x.mean() - Z95 * x.std(ddof=1) / np.sqrt(len(x)))
    assert result["n_units"] == 6
    assert result["n_pairs"] == 15
    for i, replicate in enumerate(result["replicates"]):
        assert replicate == pytest.approx(np.delete(x, i).mean())


def test_dataset_stratified_jackknife_matches_manual_eq28():
    groups = np.repeat(["A", "B", "C"], 4)
    p = pairs(len(groups))
    rng = np.random.default_rng(4)
    v = rng.uniform(size=len(p)) + (groups[p[:, 0]] != groups[p[:, 1]])
    result = group_summary(p, v, groups)
    def contrast(mask):
        inside = groups[p[mask, 0]] == groups[p[mask, 1]]
        return v[mask][~inside].mean() - v[mask][inside].mean()
    replicates = [contrast(np.all(p != r, axis=1)) for r in range(len(groups))]
    variance = 0
    for g in np.unique(groups):
        r = np.asarray(replicates)[groups == g]
        variance += (len(r) - 1) / len(r) * np.sum((r - r.mean()) ** 2)
    assert result["contrast"]["estimate"] == pytest.approx(contrast(np.ones(len(p), dtype=bool)))
    assert result["contrast"]["se"] == pytest.approx(np.sqrt(variance))
    np.testing.assert_allclose(result["contrast"]["replicates"], replicates)
    assert result["between"]["estimate"] - result["within"]["estimate"] == pytest.approx(result["contrast"]["estimate"])


def test_paired_seed_blocks_remove_both_task_networks_together():
    groups = np.tile(["identity", "parity"], 5)
    blocks = np.repeat(np.arange(401, 406), 2)
    p = pairs(len(groups))
    v = np.arange(len(p), dtype=float) / 10
    observed_subsets = []
    def statistic(values, pair_rows):
        observed_subsets.append(pair_rows.copy())
        return values.mean()
    result = jackknife(p, v, groups, statistic, blocks=blocks)
    assert result["n_units"] == 5
    for b, remaining in zip(np.unique(blocks), observed_subsets[1:]):
        assert not np.any(blocks[remaining] == b)
    r = np.asarray(result["replicates"])
    expected = np.sqrt((5 - 1) / 5 * np.sum((r - r.mean()) ** 2))
    assert result["se"] == pytest.approx(expected)
    assert group_summary(p, v, groups, blocks)["contrast"]["method"] == "delete_one_block"


def test_vector_valued_statistics_and_correlation_subsets():
    p = pairs(12)
    groups = np.repeat([0, 1, 2], 4)
    rng = np.random.default_rng(6)
    x = rng.normal(size=len(p))
    y = .7 * x + rng.normal(size=len(p))
    vectors = jackknife(p, np.column_stack((x, y)), groups)
    np.testing.assert_allclose(vectors["estimate"], [x.mean(), y.mean()])
    for method in ("pearson", "spearman"):
        for subset in ("all", "within", "between"):
            result = correlation_summary(p, x, y, groups, method, subset)
            assert np.isfinite(result["estimate"])
            assert len(result["replicates"]) == 12
            assert result["se"] >= 0


def test_student_interval_uses_runs_and_t_quantile():
    x = np.array([1., 2., 4., 3., 5.])
    result = mean_t_interval(x)
    se = x.std(ddof=1) / np.sqrt(5)
    assert result["se"] == pytest.approx(se)
    assert result["ci_high"] == pytest.approx(x.mean() + t.ppf(.975, 4) * se)


def test_invalid_resampling_design_rejected():
    with pytest.raises(ValueError, match="stratum"):
        jackknife(pairs(3), np.ones(3), [0, 1, 1])
    with pytest.raises(ValueError, match="Self-pairs"):
        jackknife(np.array([[0, 0], [0, 1]]), np.ones(2), [0, 0])
