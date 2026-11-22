"""Run-level inference for dependent network-pair measurements.

No function treats the pair count as the number of independent replicates.
The primary estimator stays the full-sample estimate; the jackknife supplies
variance and normal 95% confidence intervals, exactly as in paper §2.16.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr, t

Z95 = 1.95996


def _plain(value):
    value = np.asarray(value)
    return float(value) if value.ndim == 0 else value.tolist()


def jackknife(pairs, values, groups, statistic=None, blocks=None):
    """Delete complete runs (stratified) or entire paired seed blocks.

    Parameters
    ----------
    pairs : int array, shape (n_pairs, 2)
        Zero-based run indices for each unordered pair.
    values : array, shape (n_pairs, ...)
        Scalar or multiple pair measurements.
    groups : sequence, length n_runs
        Dataset/task label of each run. Used as strata unless blocks is given.
    statistic : callable(values_subset, pairs_subset), optional
        Returns scalar/vector estimates; defaults to the mean along axis 0.
    blocks : sequence, length n_runs, optional
        Seed-block membership. All runs in the deleted block are removed.
        One ordinary delete-one-block variance is then used, without strata.
    """
    pairs, values, groups = np.asarray(pairs, dtype=int), np.asarray(values, dtype=float), np.asarray(groups)
    if pairs.ndim != 2 or pairs.shape[1] != 2 or len(pairs) != len(values):
        raise ValueError("pairs must be n x 2 and correspond one-to-one with values")
    if len(pairs) == 0 or np.min(pairs) < 0 or np.max(pairs) >= len(groups):
        raise ValueError("Pair indices must reference groups and cannot be empty")
    if np.any(pairs[:, 0] == pairs[:, 1]):
        raise ValueError("Self-pairs are not independent training-run comparisons")
    if statistic is None:
        statistic = lambda v, p: np.mean(v, axis=0)
    estimate = np.asarray(statistic(values, pairs), dtype=float)
    if blocks is None:
        units = np.arange(len(groups))
        membership = units
        strata = groups
    else:
        blocks = np.asarray(blocks)
        if blocks.shape != groups.shape:
            raise ValueError("blocks must specify one label per run")
        units, membership = np.unique(blocks, return_inverse=True)
        strata = np.zeros(len(units), dtype=int)
    replicates, deleted = [], []
    for i, unit in enumerate(units):
        keep = np.all(membership[pairs] != i, axis=1)
        if not keep.any():
            raise ValueError("Deleting a replication unit leaves no network pairs")
        replicates.append(np.asarray(statistic(values[keep], pairs[keep]), dtype=float))
        deleted.append(unit.item() if hasattr(unit, "item") else unit)
    replicates = np.asarray(replicates)
    variance = np.zeros_like(estimate)
    for group in np.unique(strata):
        r = replicates[strata == group]
        n = len(r)
        if n < 2:
            raise ValueError("Each independent jackknife stratum needs at least two units")
        variance += (n - 1) / n * np.sum((r - r.mean(axis=0)) ** 2, axis=0)
    se = np.sqrt(variance)
    return {"estimate": _plain(estimate), "se": _plain(se),
            "ci_low": _plain(estimate - Z95 * se), "ci_high": _plain(estimate + Z95 * se),
            "replicates": replicates.tolist(), "deleted": deleted, "strata": strata.tolist(),
            "n_units": len(units), "n_pairs": len(pairs),
            "method": "delete_one_block" if blocks is not None else "stratified_delete_one_run"}


def group_summary(pairs, values, groups, blocks=None):
    """Within, between, and between-minus-within with run/block-level CIs."""
    groups = np.asarray(groups)
    def estimator(v, p):
        within = groups[p[:, 0]] == groups[p[:, 1]]
        if not within.any() or within.all():
            raise ValueError("Both within- and between-group pairs are required")
        inside, outside = np.mean(v[within], axis=0), np.mean(v[~within], axis=0)
        return np.stack((inside, outside, outside - inside))
    combined = jackknife(pairs, values, groups, estimator, blocks)
    results = {}
    for index, name in enumerate(("within", "between", "contrast")):
        result = {k: v for k, v in combined.items() if k not in ("estimate", "se", "ci_low", "ci_high", "replicates")}
        for key in ("estimate", "se", "ci_low", "ci_high"):
            result[key] = _plain(np.asarray(combined[key])[index])
        result["replicates"] = np.asarray(combined["replicates"])[:, index].tolist()
        results[name] = result
    return results


def correlation_summary(pairs, x, y, groups, method="pearson", subset="all", blocks=None):
    """Pairwise correlation with uncertainty from deleting complete runs.

    Subsets are selected inside the statistic; all replication units are kept
    in the resampling design, even when only within/between pairs are scored.
    """
    groups = np.asarray(groups)
    if method not in ("pearson", "spearman") or subset not in ("all", "within", "between"):
        raise ValueError("Unknown correlation method or subset")
    values = np.column_stack((x, y))
    correlation = pearsonr if method == "pearson" else spearmanr
    def estimator(v, p):
        keep = np.ones(len(v), dtype=bool)
        if subset != "all":
            inside = groups[p[:, 0]] == groups[p[:, 1]]
            keep = inside if subset == "within" else ~inside
        if keep.sum() < 3:
            return float("nan")
        a, b = v[keep, 0], v[keep, 1]
        if np.ptp(a) == 0 or np.ptp(b) == 0:
            return float("nan")
        return float(correlation(a, b).statistic)
    return jackknife(pairs, values, groups, estimator, blocks)


def mean_t_interval(values, confidence=0.95):
    """Mean and Student-t interval over independent runs (early displacement)."""
    values = np.asarray(values, dtype=float)
    if values.ndim == 0 or len(values) < 2:
        raise ValueError("At least two independent observations required")
    mean = values.mean(axis=0)
    se = values.std(axis=0, ddof=1) / np.sqrt(len(values))
    margin = t.ppf((1 + confidence) / 2, len(values) - 1) * se
    return {"estimate": _plain(mean), "se": _plain(se), "ci_low": _plain(mean - margin),
            "ci_high": _plain(mean + margin), "n_units": len(values)}
