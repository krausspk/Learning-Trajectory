"""Independent small-enumeration references for network-identity retrieval."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SPEC = importlib.util.spec_from_file_location("retrieval_inference", Path(__file__).with_name("inference.py"))
inference = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inference)


def manual_statistics(distances, mapping):
    n = len(distances)
    selected, first_credit, recip = [], [], []
    all_recip = []
    for i, candidate in enumerate(mapping):
        row = distances[i]
        less = np.count_nonzero(row < row[candidate])
        equal = np.count_nonzero(row == row[candidate])
        correct_rank = less + (equal + 1) / 2
        selected.append(correct_rank)
        first_credit.append(1 / equal if less == 0 else 0.)
        recip.append(1 / correct_rank)
        all_recip.extend(1 / rankdata(row, method="average"))
    mean_rank, top1, mrr = np.mean(selected), np.mean(first_credit), np.mean(recip)
    chance = np.mean(all_recip)
    return np.array([mean_rank, (n - mean_rank) / (n - 1), top1,
                     (top1 - 1 / n) / (1 - 1 / n), mrr, (mrr - chance) / (1 - chance)])


def test_global_bijections_exact_ties_and_null_expectations():
    distances = np.array([[0., 0., 2., 3.], [2., 1., 1., 2.],
                          [3., 2., 0., 1.], [2., 2., 2., 2.]])
    mappings = np.array(list(itertools.permutations(range(4))))
    ranks, credits = inference.rank_candidates(distances)
    actual = inference.mapping_statistics(ranks, credits, mappings)
    reference = np.stack([manual_statistics(distances, mapping) for mapping in mappings])
    np.testing.assert_allclose(actual, reference, atol=1e-15)
    np.testing.assert_allclose(actual.mean(axis=0)[[0, 1, 2, 3, 5]], [2.5, .5, .25, 0., 0.], atol=1e-15)
    assert actual[0, 2] == (1 / 2 + 1 / 2 + 1 + 1 / 4) / 4
    assert not np.isclose((1 / ranks).mean(), np.sum(1 / np.arange(1, 5)) / 4)
    with np.testing.assert_raises(ValueError):
        inference.mapping_statistics(ranks, credits, [[0, 0, 1, 2]])


def test_paired_network_delete_reranking_and_chance_adjusted_ci():
    rng = np.random.RandomState(303)
    distances = rng.randint(0, 8, size=(6, 6)).astype(float)
    actual, chances = inference.jackknife_retrieval(distances)
    reference = []
    for delete in range(6):
        small = np.delete(np.delete(distances, delete, axis=0), delete, axis=1)
        reference.append(manual_statistics(small, np.arange(5)))
    np.testing.assert_allclose(actual, reference, atol=1e-15)
    ranks, credits = inference.rank_candidates(distances)
    full_chance = (1 / ranks).mean()
    adjusted = inference.harmonize_replicates(actual, 6, full_chance)
    np.testing.assert_allclose(adjusted[:, 0], 6 - 5 * actual[:, 1])
    np.testing.assert_allclose(adjusted[:, 2], 1 / 6 + 5 / 6 * actual[:, 3])
    np.testing.assert_allclose(adjusted[:, 4], full_chance + (1 - full_chance) * actual[:, 5])
    expected_se = np.sqrt(5 / 6 * np.sum((adjusted - adjusted.mean(axis=0)) ** 2, axis=0))
    np.testing.assert_allclose(inference.jackknife_se(adjusted), expected_se)
    np.testing.assert_allclose(expected_se[0], 5 * expected_se[1])
    assert chances.shape == (6,)


def test_add_one_pvalues_and_holm():
    assert inference.permutation_pvalue(1., np.array([2., 3., 4.]), "less") == .25
    assert inference.permutation_pvalue(1., np.array([2., 3., 4.]), "greater") == 1.
    assert inference.permutation_pvalue(1., np.array([1., 2., 3.]), "less") == .5
    np.testing.assert_allclose(inference.holm_adjust([.01, .04, .03]), [.03, .06, .06])


def test_all_ties_receive_fractional_credit_and_zero_evidence():
    distances = np.ones((5, 5))
    ranks, credits = inference.rank_candidates(distances)
    observed = inference.mapping_statistics(ranks, credits, np.arange(5))[0]
    np.testing.assert_allclose(observed, [3., .5, .2, 0., 1 / 3, 0.], atol=1e-15)
    raw, _ = inference.jackknife_retrieval(distances)
    adjusted = inference.harmonize_replicates(raw, 5, 1 / 3)
    np.testing.assert_allclose(adjusted, np.tile(observed, (5, 1)), atol=1e-15)
    np.testing.assert_allclose(inference.jackknife_se(adjusted), 0., atol=1e-15)


def test_generic_variants_controls_and_stratified_aggregate(tmp_path, monkeypatch):
    monkeypatch.setattr(inference, "N_PERMUTATIONS", 40)
    (tmp_path / "data").mkdir()
    keys = [inference.PRIMARY_KEY, "frobenius|total|none|raw"]
    checkpoints = np.array([0, 1, 20])
    for index, dataset in enumerate(inference.DATASETS):
        rng = np.random.RandomState(500 + index)
        distances = rng.uniform(size=(2, 3, 5, 5))
        distances[:, 0, np.arange(5), np.arange(5)] = 0.
        np.savez(tmp_path / "data" / f"{dataset}_retrieval.npz", distances=distances,
                 metric_keys=keys, checkpoints=checkpoints, run_ids=np.arange(5), seeds=np.arange(5) + index * 10)
    diagnostic = inference.run_inference(tmp_path, base_seed=9400, controls=True)
    frame = pd.read_csv(tmp_path / "retrieval_by_dataset.csv")
    assert len(frame) == 18 and diagnostic["n_variants"] == 2
    assert frame.p_holm_less.isna().all()
    assert set(frame.loc[frame.checkpoint > 0, "inferential_role"]) == {"exploratory"}
    assert set(frame.loc[frame.metric_key == keys[0], "protocol_role"]) == {"reference"}
    assert frame.loc[frame.checkpoint == 0, "p_perm_less"].isna().all()
    aggregate = pd.read_csv(tmp_path / "aggregate.csv")
    results = [np.load(tmp_path / "data" / f"{dataset}_inference.npz") for dataset in inference.DATASETS]
    combined = np.load(tmp_path / "data" / "aggregate_inference.npz")
    np.testing.assert_allclose(combined["observed_statistics"], np.mean([x["observed_statistics"] for x in results], axis=0))
    np.testing.assert_allclose(combined["null_statistics"], np.mean([x["null_statistics"] for x in results], axis=0), atol=3e-7)
    for k, key in enumerate(keys):
        for t, checkpoint in enumerate(checkpoints):
            variance = np.zeros(len(inference.STATISTIC_KEYS))
            for result in results:
                reps = result["jackknife_full_n_scale"][k, t] / 3
                variance += 4 / 5 * np.sum((reps - reps.mean(axis=0)) ** 2, axis=0)
            row = aggregate[(aggregate.metric_key == key) & (aggregate.checkpoint == checkpoint)].iloc[0]
            for j, statistic in enumerate(inference.STATISTIC_KEYS):
                np.testing.assert_allclose(row[statistic + "_jackknife_se"], np.sqrt(variance[j]), atol=1e-14)
    # Reuse the same prepared data for the prespecified primary family.
    inference.run_inference(tmp_path, base_seed=9400)
    frame = pd.read_csv(tmp_path / "retrieval_by_dataset.csv")
    assert (frame.inferential_role == "primary").sum() == 3
    assert frame.p_holm_less.notna().sum() == 3
    assert frame.p_holm_less.dropna().between(0, 1).all()
