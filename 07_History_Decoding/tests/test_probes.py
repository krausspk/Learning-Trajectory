"""Tests for inference correctness and explicit leakage boundaries."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from threadpoolctl import threadpool_limits

MODULE = Path(__file__).resolve().parents[1] / "code" / "probes.py"
spec = importlib.util.spec_from_file_location("history_probes", MODULE)
probes = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = probes
spec.loader.exec_module(probes)


def test_gram_map_preserves_linear_logistic_predictions():
    rng = np.random.RandomState(101)
    x = rng.normal(size=(40, 70)) / np.sqrt(70)
    y = np.tile([0, 1], 20)
    train = np.arange(28)
    with threadpool_limits(limits=1):
        mapped, diagnostic = probes.train_gram_map(x, train)
        theta1, _ = probes.fit_logistic(x[train], y[train], .01)
        theta2, _ = probes.fit_logistic(mapped[train], y[train], .01)
    first = x @ theta1[:-1] + theta1[-1]
    second = mapped @ theta2[:-1] + theta2[-1]
    np.testing.assert_allclose(first, second, atol=2e-4)
    assert diagnostic["gram_reconstruction_max_error"] < 1e-10


def test_preprocessing_never_uses_validation_or_test_rows():
    rng = np.random.RandomState(102)
    x = rng.normal(size=(24, 15))
    d = rng.normal(size=(24, 4))
    train = np.arange(14)
    x_changed = x.copy()
    x_changed[14:] += 1000
    scaled, metadata = probes.center_scale_block(x, train)
    altered, altered_meta = probes.center_scale_block(x_changed, train)
    assert metadata == altered_meta
    np.testing.assert_array_equal(scaled[train], altered[train])
    residual, residual_meta = probes.residualize(scaled, d, train)
    altered_residual, altered_residual_meta = probes.residualize(altered, d, train)
    assert residual_meta == altered_residual_meta
    np.testing.assert_array_equal(residual[train], altered_residual[train])


def test_reference_and_aligned_layer_scales_ignore_held_out_weights(tmp_path):
    """Changing all validation/test weights cannot change a fitted transform."""
    rng = np.random.RandomState(104)
    arrays = {"W0": rng.normal(size=(6, 3, 784, 64)),
              "W1": rng.normal(size=(6, 3, 64, 10))}
    altered = {name: value.copy() for name, value in arrays.items()}
    # These are independent replacement networks, not a symmetry that could
    # hide accidental inclusion of held-out samples in reference selection.
    for name in altered:
        altered[name][2:] = rng.normal(loc=4., scale=3., size=altered[name][2:].shape)
    train = np.array([0, 1])
    folders = [tmp_path / "original", tmp_path / "altered"]
    for folder in folders:
        (folder / "analysis").mkdir(parents=True)
    with threadpool_limits(limits=1):
        aligned, metadata = probes.aligned_features(arrays, train, "reset", folders[0])
        changed, changed_metadata = probes.aligned_features(altered, train, "reset", folders[1])
    assert metadata["reference_row_index"] in train
    assert metadata["reference_row_index"] == changed_metadata["reference_row_index"]
    assert metadata["reference_sha256"] == changed_metadata["reference_sha256"]
    for h in range(3):
        for layer in range(2):
            np.testing.assert_array_equal(aligned[h][layer][train], changed[h][layer][train])
            fitted, scale = probes.center_scale_block(aligned[h][layer], train)
            fitted_changed, scale_changed = probes.center_scale_block(changed[h][layer], train)
            assert scale == scale_changed
            np.testing.assert_array_equal(fitted[train], fitted_changed[train])


def test_binary_metrics_and_bootstrap_keep_pairs():
    y = np.tile([0, 1], 12)
    scores = np.tile([-2., 2.], 12)
    result = probes.binary_metrics(y, scores)
    assert result["balanced_accuracy"] == result["auc"] == 1
    ci = probes.paired_bootstrap(y, scores, n_bootstrap=100)
    assert all(value == 1 for value in ci.values())
    swapped = probes.swapped_labels(12, np.random.RandomState(55)).reshape(-1, 2)
    np.testing.assert_array_equal(swapped.sum(axis=1), np.ones(12))


def test_permutation_pipeline_refits_and_reselects():
    rng = np.random.RandomState(103)
    y = np.tile([0, 1], 30)
    x = np.column_stack((y * 2 - 1, rng.normal(scale=.05, size=(60, 3))))
    indices = (np.arange(36), np.arange(36, 48), np.arange(48, 60))
    with threadpool_limits(limits=1):
        fitted = probes.select_probe(x, y, *indices, lambdas=(.01, .1))
        p, null = probes.permutation_test(x, fitted["test"]["balanced_accuracy"], indices, 30, 19, 42, lambdas=(.01, .1))
    assert fitted["test"]["balanced_accuracy"] == 1
    assert 0.05 <= p <= .15
    assert np.mean(null) < .65


def test_holm_adjust_is_monotonic_and_includes_negative_control():
    rows = [{"primary": True, "negative_control": False, "permutation_p": .001},
            {"primary": False, "negative_control": False, "permutation_p": .01},
            {"primary": False, "negative_control": False, "permutation_p": .03},
            {"primary": False, "negative_control": True, "permutation_p": .2}]
    probes.holm_adjust(rows)
    assert rows[0]["secondary_holm_p"] == ""
    assert rows[1]["secondary_holm_p"] == .03
    assert rows[2]["secondary_holm_p"] == .06
    assert rows[3]["secondary_holm_p"] == .2


def test_checkpoint_wait_completes_or_times_out_without_reading_archives(tmp_path, monkeypatch):
    folder = tmp_path / "checkpoints" / "reset"
    folder.mkdir(parents=True)
    (folder / "seed_1.npz").write_bytes(b"test-only sentinel, deliberately not a valid archive")
    result = probes.wait_for_checkpoints(tmp_path, "reset", [1], timeout=1)
    assert result["expected_files"] == 1
    clock = [0.]
    monkeypatch.setattr(probes.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(probes.time, "sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))
    with pytest.raises(TimeoutError, match="1 checkpoint files missing"):
        probes.wait_for_checkpoints(tmp_path, "reset", [1, 2], timeout=1)
    assert clock[0] == 1.
