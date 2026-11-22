"""Independent checks of final-only transformations and grouped inference."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

SOURCE = Path(__file__).resolve().parents[1] / 'code' / 'analyze.py'
spec = importlib.util.spec_from_file_location('trajectory_analysis_independent_test', SOURCE)
a = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = a
spec.loader.exec_module(a)


def trajectories(seed=81, steps=4, inputs=7, hidden=5, outputs=3):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(steps, inputs, hidden)), rng.normal(size=(steps, hidden, outputs))


def test_grouped_split_is_complete_disjoint_and_keeps_both_arms():
    seeds = np.repeat(np.arange(30), 2)
    split = a.grouped_split(seeds)
    assert [len(split[k]) for k in ('train', 'validation', 'test')] == [36, 12, 12]
    assert_array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(len(seeds)))
    for indices in split.values():
        for seed in np.unique(seeds[indices]):
            assert set(np.flatnonzero(seeds == seed)).issubset(set(indices))
    repeat = a.grouped_split(seeds)
    for key in split:
        assert_array_equal(split[key], repeat[key])


def test_grouped_stratification_preserves_dataset_without_crossing_seeds():
    seeds = np.repeat(np.arange(30), 2)
    strata = np.repeat(np.arange(3), 20)
    split = a.grouped_split(seeds, strata=strata)
    for key, expected_per_stratum in [('train', 12), ('validation', 4), ('test', 4)]:
        assert_array_equal(np.bincount(strata[split[key]], minlength=3), [expected_per_stratum] * 3)
    assert len(set(split['train']) & set(split['test'])) == 0


def test_alignment_final_features_ignore_all_earlier_weights():
    w0, w1 = trajectories()
    r0, r1 = trajectories(seed=96)
    reference = a.canonicalize(r0[-1], r1[-1])
    c0, c1, permutation = a.canonical_trajectory(w0, w1, reference)
    altered0, altered1 = w0.copy(), w1.copy()
    rng = np.random.default_rng(72)
    altered0[:-1] = rng.normal(size=altered0[:-1].shape) * 100
    altered1[:-1] = rng.normal(size=altered1[:-1].shape) * 100
    d0, d1, second = a.canonical_trajectory(altered0, altered1, reference)
    assert_array_equal(permutation, second)
    assert_array_equal(c0[-1], d0[-1])
    assert_array_equal(c1[-1], d1[-1])
    assert not np.allclose(c0[0], d0[0])


def test_alignment_freezes_final_permutation_for_every_checkpoint():
    w0, w1 = trajectories()
    r0, r1 = trajectories(seed=96)
    reference = a.canonicalize(r0[-1], r1[-1])
    c0, c1, permutation = a.canonical_trajectory(w0, w1, reference)
    for t in range(len(w0)):
        gauge0, gauge1 = a.canonicalize(w0[t], w1[t])
        assert_allclose(c0[t], gauge0[:, permutation], rtol=2e-7)
        assert_allclose(c1[t], gauge1[permutation], rtol=2e-7)


def test_alignment_invariant_to_hidden_unit_reindexing_and_positive_gauge():
    w0, w1 = trajectories()
    r0, r1 = trajectories(seed=96)
    reference = a.canonicalize(r0[-1], r1[-1])
    c0, c1, _ = a.canonical_trajectory(w0, w1, reference)
    permutation = np.array([2, 4, 0, 3, 1])
    scales = np.array([.2, .7, 2, 4, 9])
    altered0 = (w0 * scales[None, None, :])[:, :, permutation]
    altered1 = (w1 / scales[None, :, None])[:, permutation, :]
    d0, d1, _ = a.canonical_trajectory(altered0, altered1, reference)
    assert_allclose(c0, d0, rtol=2e-7, atol=2e-7)
    assert_allclose(c1, d1, rtol=2e-7, atol=2e-7)


def test_regression_metrics_match_manual_sums_and_displacement_errors():
    truth = np.array([[1., 3.], [3., 5.], [5., 9.]])
    pred = np.array([[0., 4.], [2., 5.], [7., 8.]])
    error = truth - pred
    result = a.regression_metrics(truth, pred)
    assert result['mse'] == pytest.approx(np.sum(error ** 2) / truth.size)
    assert result['mae'] == pytest.approx(np.sum(np.abs(error)) / truth.size)
    assert result['r2'] == pytest.approx(1 - np.sum(error ** 2) / np.sum((truth - truth.mean(0)) ** 2))
    final = np.array([[10., 2.], [0., 3.], [9., 4.]])
    shifted = a.regression_metrics(truth - final, pred - final)
    assert shifted['mse'] == pytest.approx(result['mse'])
    assert shifted['mae'] == pytest.approx(result['mae'])


def test_bootstrap_draws_whole_seed_blocks_and_keeps_duplicate_multiplicity():
    seeds = np.repeat(np.arange(12), 2)
    draws = a.bootstrap_indices(seeds, n=41, seed=792)
    assert draws.shape == (41, len(seeds))
    for draw in draws:
        counts = np.bincount(draw, minlength=len(seeds))
        assert_array_equal(counts[::2], counts[1::2])
    assert any(len(np.unique(draw)) < len(seeds) for draw in draws)
    assert_array_equal(draws, a.bootstrap_indices(seeds, n=41, seed=792))
    with pytest.raises(ValueError, match='equal-sized'):
        a.bootstrap_indices([0, 0, 1])


def test_history_null_swaps_entire_vectors_inside_pairs_only():
    rng = np.random.default_rng(71)
    base = rng.integers(0, 2, size=(12, 4))
    labels = np.stack((base, 1 - base), axis=1).reshape(-1, 4)
    seeds = np.repeat(np.arange(12), 2)
    swapped = a.paired_label_swap(labels, seeds, np.random.default_rng(99))
    changed = 0
    for i in range(0, len(labels), 2):
        assert np.array_equal(swapped[i:i+2], labels[i:i+2]) or np.array_equal(swapped[i:i+2], labels[i:i+2][::-1])
        assert_array_equal(swapped[i] + swapped[i + 1], np.ones(4, int))
        changed += int(not np.array_equal(swapped[i], labels[i]))
    assert 0 < changed < 12
    assert_array_equal(labels[::2], base)


def test_holm_matches_hand_calculation_and_preserves_original_order():
    pvalues = [0.04, 0.001, 0.02, 0.9]
    assert_allclose(a.holm(pvalues), [0.08, 0.004, 0.06, 0.9])


def test_linear_probe_generalizes_known_linear_map_and_exceeds_baselines():
    rng = np.random.default_rng(413)
    x = rng.normal(size=(120, 7))
    y = x @ rng.normal(size=(7, 3)) + np.array([4., -5., 12.])
    split = a.grouped_split(np.repeat(np.arange(60), 2))
    pred, alpha = a.LinearProbe(x, split).fit_select(y)
    test = split['test']
    assert a.regression_metrics(y[test], pred['test'])['r2'] > 0.999
    train_mean = np.broadcast_to(y[split['train']].mean(0), y[test].shape)
    assert a.regression_metrics(y[test], pred['test'])['mse'] < a.regression_metrics(y[test], train_mean)['mse'] / 1000
    assert alpha in a.ALPHAS


def test_linear_probe_training_normalization_ignores_validation_and_test_features():
    rng = np.random.default_rng(498)
    x = rng.normal(size=(80, 8))
    split = a.grouped_split(np.repeat(np.arange(40), 2))
    altered = x.copy()
    altered[np.r_[split['validation'], split['test']]] *= 1e5
    p1, p2 = a.LinearProbe(x, split), a.LinearProbe(altered, split)
    assert_array_equal(p1.mean, p2.mean)
    assert p1.scale == p2.scale
    assert_allclose(p1.eigen, p2.eigen, atol=1e-10)
    assert_array_equal(p1.x[split['train']], p2.x[split['train']])


def test_linear_probe_selection_and_predictions_ignore_test_targets():
    rng = np.random.default_rng(219)
    x = rng.normal(size=(80, 8))
    y = x @ rng.normal(size=(8, 3))
    split = a.grouped_split(np.repeat(np.arange(40), 2))
    altered = y.copy()
    altered[split['test']] += 1e8
    probe = a.LinearProbe(x, split)
    first, alpha1 = probe.fit_select(y)
    second, alpha2 = probe.fit_select(altered)
    assert alpha1 == alpha2
    for key in split:
        assert_array_equal(first[key], second[key])


def test_pca_fitted_to_training_slice_is_unchanged_by_holdout_targets():
    rng = np.random.default_rng(116)
    x = rng.normal(size=(60, 5, 14)).astype(np.float32)
    split = a.grouped_split(np.repeat(np.arange(30), 2))
    train = split['train']
    first = a.randomized_pca(x[train].reshape(-1, 14), components=8)
    altered = x.copy()
    altered[split['test'], :-1] = rng.normal(size=altered[split['test'], :-1].shape) * 1e5
    second = a.randomized_pca(altered[train].reshape(-1, 14), components=8)
    for actual, expected in zip(second, first):
        assert_array_equal(actual, expected)
    assert_allclose(first[1].T @ first[1], np.eye(8), atol=1e-6)


def test_event_auc_and_accuracy_match_known_rank_example():
    y = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.4, 0.35, 0.8])
    assert a.auc_score(y, score) == pytest.approx(.75)
    metrics = a.event_metrics(y, score)
    assert metrics['accuracy'] == pytest.approx(.75)
    assert metrics['auc'] == pytest.approx(.75)
    assert a.auc_score(y, np.ones(4)) == .5


def test_batched_history_null_matches_serial_retraining_and_selection():
    rng = np.random.default_rng(613)
    seeds = np.repeat(np.arange(40), 2)
    split = a.grouped_split(seeds)
    base = rng.integers(0, 2, size=(40, 4))
    labels = np.stack((base, 1-base), axis=1).reshape(-1, 4)
    x = rng.normal(size=(80, 12)) + labels @ rng.normal(size=(4, 12)) * .3
    probe = a.LinearProbe(x, split)
    permutations = 9
    shuffled = np.stack([a.paired_label_swap(labels, seeds, rng) for _ in range(permutations)], axis=1)
    batched, batched_alpha = probe.fit_select(shuffled.reshape(len(labels), -1), classification=True)
    batched_scores = batched['test'].reshape(len(split['test']), permutations, labels.shape[1])
    batched_null = np.mean((batched_scores >= .5) == shuffled[split['test']], axis=0)
    for p in range(permutations):
        serial, serial_alpha = probe.fit_select(shuffled[:, p], classification=True)
        assert_array_equal(batched_alpha.reshape(permutations, -1)[p], serial_alpha)
        assert_allclose(batched_scores[:, p], serial['test'], atol=1e-10, rtol=1e-10)
        serial_null = np.mean((serial['test'] >= .5) == shuffled[split['test'], p], axis=0)
        assert_array_equal(batched_null[p], serial_null)


def test_mean_early_minus_final_baseline_has_absolute_mean_prediction_error():
    rng = np.random.default_rng(983)
    early = rng.normal(size=(80, 8))
    final = rng.normal(size=(80, 8)) * 100
    split = a.grouped_split(np.repeat(np.arange(40), 2))
    mean_early = early[split['train']].mean(0)
    test = split['test']
    absolute_baseline = np.broadcast_to(mean_early, early[test].shape)
    displacement_target = early[test] - final[test]
    displacement_baseline = mean_early - final[test]
    absolute = a.regression_metrics(early[test], absolute_baseline)
    displacement = a.regression_metrics(displacement_target, displacement_baseline)
    assert displacement['mse'] == pytest.approx(absolute['mse'], abs=1e-12)
    assert displacement['mae'] == pytest.approx(absolute['mae'], abs=1e-12)
    # R2 differs because the known final position dominates displacement variance.
    assert displacement['r2'] > .99
    assert absolute['r2'] < .2
