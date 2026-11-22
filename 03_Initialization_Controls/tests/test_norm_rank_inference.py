import importlib.util
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

SPEC = importlib.util.spec_from_file_location(
    "norm_rank_wrapper", Path(__file__).resolve().parents[1] / "code/norm_rank_inference.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)


def norm_distance(query, candidate):
    n = len(query)
    return np.abs(rankdata(query)[:, None] - rankdata(candidate)[None, :]) / (n - 1)


def manual_identity_statistics(matrix):
    n = len(matrix)
    ranks = np.empty_like(matrix)
    credit = []
    for i in range(n):
        for j in range(n):
            ranks[i, j] = np.count_nonzero(matrix[i] < matrix[i, j]) + (np.count_nonzero(matrix[i] == matrix[i, j]) + 1) / 2
        credit.append(1 / np.count_nonzero(matrix[i] == matrix[i].min()) if matrix[i, i] == matrix[i].min() else 0.)
    mean_rank = ranks.diagonal().mean()
    top1 = np.mean(credit)
    mrr = (1 / ranks.diagonal()).mean()
    chance = (1 / ranks).mean()
    return np.array([mean_rank, (n - mean_rank) / (n - 1), top1,
                     (top1 - 1 / n) / (1 - 1 / n), mrr, (mrr - chance) / (1 - chance)])


def test_recalibrated_jackknife_matches_independent_norm_reranking(tmp_path):
    module = wrapper.load_frozen_inference()
    rng = np.random.RandomState(883)
    n = 6
    norms = rng.normal(size=(2, n))
    full = np.stack([norm_distance(row, norms[0]) for row in norms])[None]
    deleted = np.empty((1, 2, n, n - 1, n - 1))
    for t in range(2):
        for i in range(n):
            keep = np.arange(n) != i
            deleted[0, t, i] = norm_distance(norms[t, keep], norms[0, keep])
    path = tmp_path / "fixture.npz"
    np.savez(path, distances=full, jackknife_distances=deleted,
             metric_keys=["norm_rank_difference|total|none|raw"], checkpoints=[0, 20],
             run_ids=np.arange(n), seeds=np.arange(n))
    naive = module.jackknife_retrieval(full[0, 1])[0]
    original = module.jackknife_retrieval
    with wrapper.RecalibratedJackknife(module) as adapter:
        module._load_input(path)
        for t in range(2):
            actual, chances = module.jackknife_retrieval(full[0, t])
            expected = np.stack([manual_identity_statistics(deleted[0, t, i]) for i in range(n)])
            np.testing.assert_allclose(actual, expected, atol=1e-14)
        assert not np.allclose(naive, actual)
        adapter.finish()
    assert module.jackknife_retrieval is original
    assert adapter.completed[0]["n_jackknife_calls"] == 2


def test_duplicate_full_matrices_use_ordered_distinct_deletions_and_restore_hooks(tmp_path):
    module = wrapper.load_frozen_inference()
    n = 5
    matrix = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]).astype(float)
    full = np.tile(matrix, (2, 2, 1, 1))
    small = np.abs(np.arange(n - 1)[:, None] - np.arange(n - 1)[None, :]).astype(float)
    jack = np.tile(small, (2, 2, n, 1, 1))
    jack[1] = jack[1, :, :, :, ::-1]
    path = tmp_path / "fixture.npz"
    np.savez(path, distances=full, jackknife_distances=jack,
             metric_keys=["norm_rank_difference|W0|none|raw", "norm_rank_difference|W1|none|raw"],
             checkpoints=[0, 20], run_ids=np.arange(n), seeds=np.arange(n))
    loader = module._load_input
    original = module.jackknife_retrieval
    with wrapper.RecalibratedJackknife(module) as adapter:
        module._load_input(path)
        results = [module.jackknife_retrieval(full[k, t])[0] for k in range(2) for t in range(2)]
        assert not np.allclose(results[0], results[2])
        adapter.finish()
    assert module._load_input is loader and module.jackknife_retrieval is original
    with np.testing.assert_raises(RuntimeError):
        with wrapper.RecalibratedJackknife(module) as adapter:
            module._load_input(path)
            module.jackknife_retrieval(full[0, 0] + .1)
    assert module._load_input is loader and module.jackknife_retrieval is original
