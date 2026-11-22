"""Independent checks of the scientific metric and region definitions."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


_spec = importlib.util.spec_from_file_location("perturbation_metrics", Path(__file__).with_name("metrics.py"))
metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(metrics)


def network(seed=11, width=4):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(9, width)), rng.normal(size=(width, 3))


def transformed(a, seed=5):
    rng = np.random.default_rng(seed)
    p = rng.permutation(a[0].shape[1])
    scale = np.exp(rng.uniform(-3, 3, len(p)))
    return a[0][:, p] * scale, a[1][p, :] / scale[:, None]


def test_permutation_and_positive_scaling_are_zero_distance_and_preserve_function():
    a = network()
    b = transformed(a)
    x = np.random.default_rng(61).normal(size=(13, 9))
    np.testing.assert_allclose(np.maximum(x @ a[0], 0) @ a[1],
                               np.maximum(x @ b[0], 0) @ b[1], atol=1e-12)
    result = metrics.paired_metrics(a, b)
    for key in ("distance", "pif_distance", "alpha_distance"):
        assert result[key] < 1e-12
    assert result["alpha"] == 0


def test_distance_matches_exhaustive_canonical_frobenius_optimum():
    a, b = network(21), network(22)
    ca, cb = metrics.canonical_pair(*a), metrics.canonical_pair(*b)
    candidates = []
    for perm in itertools.permutations(range(4)):
        p = list(perm)
        candidates.append(np.sqrt(np.square(ca[0] - cb[0][:, p]).sum()
                                  + np.square(ca[1] - cb[1][p]).sum()))
    assert metrics.distance(a, b) == pytest.approx(min(candidates), abs=1e-12)
    assert metrics.distance(a, b) == pytest.approx(metrics.distance(b, a), abs=1e-12)
    assert metrics.distance(transformed(a, 8), transformed(b, 9)) == pytest.approx(min(candidates), abs=1e-12)


def test_analysis_does_not_mutate_or_share_input_storage():
    a, b = network(31), network(32)
    originals = tuple(w.copy() for w in (*a, *b))
    canonical = metrics.canonical_pair(*a)
    metrics.paired_metrics(a, b)
    for current, before in zip((*a, *b), originals):
        np.testing.assert_array_equal(current, before)
    assert all(not np.shares_memory(c, original) for c in canonical for original in a)


def test_reference_thresholds_exclude_self_and_use_networks_as_units():
    # One hidden unit, balanced equal matrices: analytic distance = 2*|a-b|.
    refs = [(np.full((2, 1), value), np.full((1, 2), value)) for value in (1., 2., 4., 8.)]
    accuracy = [.91, .93, .95, .97]
    loss = [.4, .3, .2, .1]
    calibration = metrics.threshold_calibration(refs, accuracy, loss)
    expected_pairwise = 2 * np.abs(np.array([1., 2., 4., 8.])[:, None] - np.array([1., 2., 4., 8.]))
    np.testing.assert_allclose(calibration["pairwise_distances"], expected_pairwise)
    expected_medians = [6., 4., 6., 12.]
    np.testing.assert_allclose([r["loo_median_distance"] for r in calibration["per_reference"]], expected_medians)
    assert calibration["distance_q95"] == pytest.approx(11.1)
    assert calibration["distance_q90"] == pytest.approx(10.2)
    assert calibration["accuracy_q05"] == pytest.approx(.913)
    assert calibration["loss_q95"] == pytest.approx(.385)


def test_region_requires_functional_and_structural_criteria_and_rejects_nan():
    refs = [(np.full((2, 1), value), np.full((1, 2), value)) for value in (1., 2., 4., 8.)]
    thresholds = metrics.threshold_calibration(refs, [.91, .93, .95, .97], [.4, .3, .2, .1])
    nearby = (np.full((2, 1), 3.), np.full((1, 2), 3.))
    distant = (np.full((2, 1), 100.), np.full((1, 2), 100.))
    good = metrics.region_membership(nearby, refs, .95, .2, thresholds)
    assert good["member"] and good["functional"] and good["structural"]
    assert not metrics.region_membership(nearby, refs, .90, .2, thresholds)["member"]
    assert not metrics.region_membership(nearby, refs, .95, .5, thresholds)["member"]
    far = metrics.region_membership(distant, refs, .95, .2, thresholds)
    assert far["functional"] and not far["structural"] and not far["member"]
    bad = (np.full((2, 1), np.nan), np.ones((1, 2)))
    assert not metrics.region_membership(bad, refs, .95, .2, thresholds)["member"]
    assert not metrics.region_membership(nearby, refs, np.nan, .2, thresholds)["member"]
    zero = (np.zeros((2, 1)), np.ones((1, 2)))
    assert not metrics.region_membership(zero, refs, .95, .2, thresholds)["member"]


def test_invalid_reference_calibration_and_candidate_shape_raise():
    refs = [network(seed) for seed in (1, 2, 3)]
    with pytest.raises(ValueError, match="at least three"):
        metrics.threshold_calibration(refs[:2], [.9, .9], [.1, .1])
    with pytest.raises(ValueError, match="finite"):
        metrics.threshold_calibration(refs, [.9, np.nan, .9], [.1, .1, .1])
    with pytest.raises(ValueError, match="one accuracy"):
        metrics.threshold_calibration(refs, [.9], [.1, .1, .1])
    thresholds = metrics.threshold_calibration(refs, [.9] * 3, [.1] * 3)
    with pytest.raises(ValueError, match="cohort size"):
        metrics.region_membership(network(4), refs[:2], .9, .1, thresholds)
    with pytest.raises(ValueError, match="identical architectures"):
        metrics.region_membership(network(4, width=3), refs, .9, .1, thresholds)
