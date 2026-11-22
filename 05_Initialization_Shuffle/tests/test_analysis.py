"""Small independent references for crossed-factor inference, without training."""
import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

SPEC = importlib.util.spec_from_file_location("crossed_analysis", Path(__file__).resolve().parents[1] / "code/analysis.py")
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


def make_tensor(n_init=4, n_shuffle=3, seed=15):
    rng = np.random.RandomState(seed)
    initial = rng.normal(size=(n_init, 5))
    shuffle = rng.normal(size=(n_shuffle, 5))
    features = initial[:, None] + .7 * shuffle[None] + .2 * rng.normal(size=(n_init, n_shuffle, 5))
    flat = features.reshape(-1, 5)
    return np.linalg.norm(flat[:, None] - flat[None, :], axis=-1).reshape(n_init, n_shuffle, n_init, n_shuffle)


def direct_block_statistics(tensor, factor, permutations=None):
    ni, ns = tensor.shape[:2]
    groups, identities = (ns, ni) if factor == "initialization" else (ni, ns)
    permutations = np.broadcast_to(np.arange(identities), (groups, identities)) if permutations is None else permutations
    correct_ranks, top1, rr, all_rr = [], [], [], []
    for source in range(groups):
        for target in range(groups):
            if source == target:
                continue
            for physical in range(identities):
                label = int(np.where(permutations[source] == physical)[0][0])
                candidate = int(permutations[target, label])
                costs = tensor[physical, source, :, target] if factor == "initialization" else tensor[source, physical, target, :]
                less = np.count_nonzero(costs < costs[candidate])
                equal = np.count_nonzero(costs == costs[candidate])
                rank = less + (equal + 1) / 2
                correct_ranks.append(rank)
                top1.append(1 / equal if less == 0 else 0.)
                rr.append(1 / rank)
                all_rr.extend(1 / rankdata(costs, method="average"))
    meanrank, first, mrr = np.mean(correct_ranks), np.mean(top1), np.mean(rr)
    chance = np.mean(all_rr)
    return np.array([meanrank, (identities - meanrank) / (identities - 1), first,
                     (first - 1 / identities) / (1 - 1 / identities), mrr,
                     (mrr - chance) / (1 - chance)])


def test_block_null_inverse_composition_and_exact_small_enumeration():
    tensor = make_tensor(n_init=3, n_shuffle=2)
    permutations = np.stack([np.stack([np.arange(3), p]) for p in itertools.permutations(range(3))])
    actual = analysis.factor_result(tensor, "initialization", permutations)
    reference = np.stack([direct_block_statistics(tensor, "initialization", p) for p in permutations])
    np.testing.assert_allclose(actual["null"], reference, atol=1e-14)
    np.testing.assert_allclose(actual["null"].mean(axis=0)[[0, 1, 2]], [2., .5, 1 / 3], atol=1e-14)
    # Reordering common labels has no effect on the matched physical pairs.
    relabeled = permutations[:, :, [2, 0, 1]]
    np.testing.assert_allclose(analysis.factor_result(tensor, "initialization", relabeled)["null"], reference)
    maps = np.array([[[2, 0, 1], [1, 2, 0]], [[1, 0, 2], [0, 2, 1]]])
    physical = analysis.physical_candidate_mapping(maps, 0, 1)
    expected = np.array([[maps[b, 1, np.where(maps[b, 0] == i)[0][0]] for i in range(3)] for b in range(2)])
    np.testing.assert_array_equal(physical, expected)
    assert not np.array_equal(physical, maps[:, 1])


def test_initialization_and_shuffle_factor_directions_match_direct_reference():
    tensor = make_tensor()
    for factor, blocks, identities in (("initialization", 3, 4), ("shuffle", 4, 3)):
        maps = analysis.factor_label_permutations(blocks, identities, 400, n_permutations=13)
        np.testing.assert_array_equal(maps[:, 0], np.tile(np.arange(identities), (13, 1)))
        result = analysis.factor_result(tensor, factor, maps)
        np.testing.assert_allclose(result["observed"], direct_block_statistics(tensor, factor), atol=1e-14)
        np.testing.assert_allclose(result["null"], [direct_block_statistics(tensor, factor, p) for p in maps], atol=1e-14)
        np.testing.assert_allclose(result["per_identity"].mean(0), result["observed"])


def test_delete_whole_factor_levels_and_candidate_count_adjustment():
    tensor = make_tensor(seed=199)
    for factor in ("initialization", "shuffle"):
        raw, contrast, chances = analysis.factor_jackknife(tensor, factor)
        n = tensor.shape[0 if factor == "initialization" else 1]
        reference = []
        for deleted in range(n):
            axes = (0, 2) if factor == "initialization" else (1, 3)
            small = np.delete(np.delete(tensor, deleted, axis=axes[0]), deleted, axis=axes[1])
            reference.append(direct_block_statistics(small, factor))
        np.testing.assert_allclose(raw, reference, atol=1e-14)
        full = analysis.factor_result(tensor, factor)
        adjusted = analysis.BASE.harmonize_replicates(raw, n, full["mrr_chance"])
        np.testing.assert_allclose(adjusted[:, 0], n - (n - 1) * raw[:, 1])
        np.testing.assert_allclose(analysis.BASE.jackknife_se(adjusted[:, 0]),
                                   (n - 1) * analysis.BASE.jackknife_se(raw[:, 1]))
        assert contrast.shape == chances.shape == (n,)


def test_trajectory_uses_one_candidate_identity_mapping_across_all_shuffles():
    rng = np.random.RandomState(881)
    distances = rng.randint(0, 12, size=(4, 3, 4)).astype(float)
    permutations = np.array(list(itertools.permutations(range(4))))
    result = analysis.trajectory_result(distances, permutations)
    null_rank = []
    for mapping in permutations:
        selected = []
        for shuffle in range(3):
            for initial in range(4):
                selected.append(rankdata(distances[initial, shuffle], method="average")[mapping[initial]])
        null_rank.append(np.mean(selected))
    np.testing.assert_allclose(result["null"][:, 0], null_rank)
    np.testing.assert_allclose(np.mean(null_rank), 2.5)
    raw, _ = analysis.trajectory_jackknife(distances)
    for deleted in range(4):
        sub = np.delete(np.delete(distances, deleted, axis=0), deleted, axis=2)
        expected = np.mean([rankdata(sub[i, s])[i] for i in range(3) for s in range(3)])
        np.testing.assert_allclose(raw[deleted, 0], expected)


def test_distance_counts_and_factor_contrasts():
    tensor = make_tensor(n_init=10, n_shuffle=5)
    groups = analysis.distance_groups(tensor)
    assert [groups[key]["n_pairs"] for key in groups] == [100, 225, 900]
    both = groups["different_initialization_different_shuffle"]["mean"]
    np.testing.assert_allclose(analysis.factor_result(tensor, "initialization")["contrast"],
                               both - groups["same_initialization_different_shuffle"]["mean"])
    np.testing.assert_allclose(analysis.factor_result(tensor, "shuffle")["contrast"],
                               both - groups["different_initialization_same_shuffle"]["mean"])


def test_two_way_ss_and_constant_table():
    values = 10 + np.arange(3)[:, None] + np.array([0, 2])[None] + np.array([[-1, 1], [0, 0], [1, -1]])
    result, _, _ = analysis.performance_decomposition(values)
    np.testing.assert_allclose([result[k] for k in ("initialization_ss", "shuffle_ss", "interaction_residual_ss", "total_ss")], [4, 6, 4, 14])
    constant, _, _ = analysis.performance_decomposition(np.full((10, 5), .1))
    assert constant["total_ss"] == constant["initialization_ss"] == constant["shuffle_ss"] == constant["interaction_residual_ss"] == 0
    assert np.isnan(constant["initialization_fraction"])


def test_incomplete_training_manifest_blocks_analysis(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/manifest.json").write_text(json.dumps({"complete": False, "expected_runs": 150, "runs": []}))
    with pytest.raises(ValueError, match="150-run"):
        analysis.load_complete_manifest(tmp_path)


def test_three_stratum_artifacts_and_aggregate_variance(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "N_PERMUTATIONS", 30)
    (tmp_path / "data").mkdir()
    all_records = []
    for index, dataset in enumerate(analysis.DATASETS):
        rng = np.random.RandomState(1300 + index)
        tensor = make_tensor(10, 5, 500 + index)
        trajectory = rng.uniform(.1, 1., size=(21, 10, 5, 10))
        for i in range(10):
            trajectory[0, i, :, i] = 0.
        data = {"endpoint_distances": tensor, "endpoint_cka_distances": tensor / tensor.max(),
                "initial_distances": trajectory, "accuracy": rng.uniform(size=(21, 10, 5)),
                "loss": rng.uniform(size=(21, 10, 5)), "init_ids": np.arange(1, 11), "shuffle_ids": np.arange(1, 6)}
        rows, records = analysis.analyze_dataset(tmp_path, dataset, data, index)
        assert len(rows["initialization_retrieval"]) == 2 and len(rows["shuffle_retrieval"]) == 2
        assert sum(r["inferential_role"] == "primary" for r in rows["initialization_retrieval"]) == 1
        assert len(rows["initial_trajectory_retrieval"]) == 21
        all_records.extend(records)
        with np.load(tmp_path / "data" / f"{dataset}_inference.npz") as stored:
            assert stored["endpoint_null"].shape == (2, 2, 30, 6)
            assert stored["initialization_jackknife_raw"].shape == (2, 10, 6)
            assert stored["shuffle_jackknife_raw"].shape == (2, 5, 6)
            assert np.isnan(stored["trajectory_null"][0]).all()
    analysis.aggregate_results(tmp_path, all_records)
    aggregates = pd.read_csv(tmp_path / "aggregate.csv")
    assert len(aggregates) == 25
    for row in aggregates.itertuples():
        matches = [r for r in all_records if (r["analysis"], r["key"], r["checkpoint"]) == (row.analysis, row.metric_key, row.checkpoint)]
        expected = np.mean([r["result"]["observed"][0] for r in matches])
        variance = 0.
        for record in matches:
            reps = record["adjusted_jack"][:, 0] / 3
            n = len(reps)
            variance += (n - 1) / n * np.sum((reps - reps.mean()) ** 2)
        np.testing.assert_allclose(row.mean_rank, expected, atol=1e-13)
        np.testing.assert_allclose(row.mean_rank_jackknife_se, np.sqrt(variance), atol=1e-13)
