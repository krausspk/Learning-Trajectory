"""Network-identity inference for cross-time initialization retrieval.

Entry point: run_inference(out: Path, base_seed=2026111700, primary_key=PRIMARY_KEY).
Input C[k,t,i,j] is the distance from current query i to initial candidate j.
Every permutation is one bijection of the candidate identities, shared by all
queries. Rows/columns are deleted together for the network-level jackknife.
Calculations use float64; stored Monte Carlo null arrays use float32.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE_SEED = 2026111700
N_PERMUTATIONS = 10000
DATASETS = ("mnist", "fashion_mnist", "kmnist")
PRIMARY_KEY = "normalized_frobenius|total|optimal|canonical"
Z95 = 1.95996
STATISTIC_KEYS = ("mean_rank", "rank_score", "top1", "top1_score", "mrr", "mrr_score")


def rank_candidates(distances):
    """Average exact ties; fractional credit for every tied nearest candidate.

    Ties are exact float64 equality of the saved distances, without rounding or
    a data-dependent tolerance. This preserves the supplied distance ordering.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[0] != distances.shape[1]:
        raise ValueError("Expected one square query-by-candidate matrix")
    if distances.shape[0] < 3 or not np.isfinite(distances).all():
        raise ValueError("At least three runs and finite distances are required")
    ranks = rankdata(distances, method="average", axis=1)
    nearest = distances == distances.min(axis=1, keepdims=True)
    top1_credit = nearest / nearest.sum(axis=1, keepdims=True)
    return ranks, top1_credit


def mapping_statistics(ranks, top1_credit, mappings):
    """Statistics for bijections mapping each query to its claimed true parent.

    rank_score=(n-mean_rank)/(n-1): chance=.5, perfect=1, worst=0.
    top1_score=(top1-1/n)/(1-1/n): chance=0, perfect=1.
    MRR uses reciprocal average ranks. Its exact random-label expectation is the
    average of reciprocal ranks over all cells, also in the presence of ties.
    mrr_score=(mrr-expectation)/(1-expectation): chance=0, perfect=1.
    """
    ranks = np.asarray(ranks, dtype=np.float64)
    top1_credit = np.asarray(top1_credit, dtype=np.float64)
    mappings = np.atleast_2d(np.asarray(mappings, dtype=np.int64))
    n = ranks.shape[0]
    if ranks.shape != (n, n) or top1_credit.shape != (n, n):
        raise ValueError("Ranks and credits must be equally sized square matrices")
    if mappings.shape[1] != n or not np.all(np.sort(mappings, axis=1) == np.arange(n)):
        raise ValueError("Each mapping must be one complete candidate-identity permutation")
    selected_ranks = ranks[np.arange(n), mappings]
    mean_rank = selected_ranks.mean(axis=1)
    top1 = top1_credit[np.arange(n), mappings].mean(axis=1)
    mrr = (1 / selected_ranks).mean(axis=1)
    mrr_chance = float((1 / ranks).mean())
    return np.column_stack((mean_rank, (n - mean_rank) / (n - 1),
                            top1, (top1 - 1 / n) / (1 - 1 / n),
                            mrr, (mrr - mrr_chance) / (1 - mrr_chance)))


def jackknife_retrieval(distances):
    """Rerank after jointly removing one query and its initial candidate."""
    distances = np.asarray(distances, dtype=np.float64)
    n = len(distances)
    replicates, mrr_chances = [], []
    for deleted in range(n):
        keep = np.arange(n) != deleted
        ranks, credits = rank_candidates(distances[np.ix_(keep, keep)])
        replicates.append(mapping_statistics(ranks, credits, np.arange(n - 1))[0])
        mrr_chances.append(float((1 / ranks).mean()))
    return np.asarray(replicates), np.asarray(mrr_chances)


def jackknife_se(replicates):
    replicates = np.asarray(replicates, dtype=np.float64)
    n = len(replicates)
    return np.sqrt((n - 1) / n * np.sum((replicates - replicates.mean(axis=0)) ** 2, axis=0))


def harmonize_replicates(replicates, full_n, full_mrr_chance):
    """Express deletion statistics on the full sample's chance/ceiling scale.

    Raw deletion ranks have n-1 candidates. Transform normalized scores back
    using full n before calculating raw-statistic standard errors and CIs.
    """
    values = np.asarray(replicates, dtype=np.float64).copy()
    values[..., 0] = full_n - (full_n - 1) * values[..., 1]
    values[..., 2] = 1 / full_n + (1 - 1 / full_n) * values[..., 3]
    values[..., 4] = full_mrr_chance + (1 - full_mrr_chance) * values[..., 5]
    return values


def permutation_pvalue(observed, null, alternative="greater"):
    """Add-one Monte Carlo probability, including numerical equality."""
    null = np.asarray(null, dtype=np.float64)
    if not np.isfinite(observed) or not np.isfinite(null).all():
        return np.nan
    if alternative == "greater":
        count = np.count_nonzero(null >= observed - 1e-12)
    elif alternative == "less":
        count = np.count_nonzero(null <= observed + 1e-12)
    else:
        raise ValueError("Expected greater or less alternative")
    return float((1 + count) / (len(null) + 1))


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(
        (len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


def _null_diagnostics(values, expected):
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    mcse = sd / np.sqrt(len(values))
    z = (mean - expected) / mcse if mcse > 0 else np.nan
    return {"null_mean": mean, "null_sd": sd, "null_expected": expected,
            "null_mean_mcse": mcse, "null_mean_z": z}


def _metric_fields(key):
    metric, layer, alignment, gauge = key.split("|")
    return {"metric": metric, "layer": layer, "alignment": alignment, "gauge": gauge}


def _summary(observed, adjusted_replicates, null, n, mrr_chance, se=None):
    """All raw CIs are unbounded normal CIs; plotting may clip their display."""
    observed = np.asarray(observed)
    if se is None:
        se = jackknife_se(adjusted_replicates)
    row = {"n_runs": n, "n_queries": n, "n_candidates": n,
           "n_permutations": 0 if null is None else len(null),
           "chance_mean_rank": (n + 1) / 2, "chance_rank_score": .5,
           "chance_top1": 1 / n, "chance_mrr": mrr_chance}
    expected = ((n + 1) / 2, .5, 1 / n, 0., mrr_chance, 0.)
    for index, name in enumerate(STATISTIC_KEYS):
        row[name] = float(observed[index])
        row[name + "_jackknife_se"] = float(se[index])
        row[name + "_ci_low"] = float(observed[index] - Z95 * se[index])
        row[name + "_ci_high"] = float(observed[index] + Z95 * se[index])
        row[name + "_jackknife_min"] = float(adjusted_replicates[:, index].min())
        row[name + "_jackknife_max"] = float(adjusted_replicates[:, index].max())
        alternative = "less" if name == "mean_rank" else "greater"
        row[name + "_p_perm_" + alternative] = (np.nan if null is None else
            permutation_pvalue(observed[index], null[:, index], alternative))
        if null is not None:
            row.update({name + "_" + field: value for field, value in
                        _null_diagnostics(null[:, index], expected[index]).items()})
    row.update({"p_perm_less": row["mean_rank_p_perm_less"],
                "p_perm_greater": row["rank_score_p_perm_greater"],
                "jackknife_se": row["mean_rank_jackknife_se"],
                "ci_low": row["mean_rank_ci_low"], "ci_high": row["mean_rank_ci_high"]})
    return row


def _write_csv(path, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.15g")
    temporary.replace(path)


def _write_npz(path, **arrays):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def _load_input(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    distances = np.asarray(data["distances"], dtype=np.float64)
    keys = data["metric_keys"].astype(str).tolist()
    checkpoints = data["checkpoints"].astype(int)
    run_ids = data["run_ids"].astype(str)
    n = len(run_ids)
    if distances.shape != (len(keys), len(checkpoints), n, n):
        raise ValueError(f"Incompatible distance dimensions in {path}")
    if n < 4 or not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError(f"Require finite nonnegative distances and at least four networks: {path}")
    if len(set(keys)) != len(keys) or len(set(run_ids)) != n:
        raise ValueError(f"Duplicate metric or run identifiers in {path}")
    if len(set(checkpoints)) != len(checkpoints) or 0 not in checkpoints or 20 not in checkpoints:
        raise ValueError(f"Expected unique checkpoints containing E0 and E20: {path}")
    for axis in ("query_run_ids", "candidate_run_ids"):
        if axis in data and not np.array_equal(data[axis].astype(str), run_ids):
            raise ValueError(f"Query/candidate run identity ordering differs in {path}")
    return data, distances, keys, checkpoints, run_ids


def run_inference(out: Path, base_seed=BASE_SEED, primary_key=PRIMARY_KEY, controls=False):
    """Read all three data/*_retrieval.npz files and write CSV/NPZ statistics.

    Set controls=True when reusing this analysis for purely exploratory controls;
    the given primary_key is then only a reference protocol, with no Holm family.
    Set primary_key=None for exploratory variants without a reference protocol.
    No metric, checkpoint, seed, or test direction is selected by data.
    """
    out = Path(out)
    reference_key = primary_key if controls else None
    if controls:
        primary_key = None
    rows, query_rows, jackknife_rows, dataset_results = [], [], [], []
    expected_keys = expected_checkpoints = expected_n = None
    for dataset_index, dataset in enumerate(DATASETS):
        data, distances, keys, checkpoints, run_ids = _load_input(out / "data" / f"{dataset}_retrieval.npz")
        k_count, t_count, n, _ = distances.shape
        if expected_keys is None:
            expected_keys, expected_checkpoints, expected_n = keys, checkpoints, n
        if keys != expected_keys or not np.array_equal(checkpoints, expected_checkpoints) or n != expected_n:
            raise ValueError("All dataset strata must have the same ordered variants, checkpoints, and n")
        if primary_key is not None and primary_key not in keys:
            raise ValueError(f"Prespecified primary key is absent: {primary_key}")
        permutation_seed = int(base_seed) + dataset_index + 1
        rng = np.random.RandomState(permutation_seed)
        mappings = np.stack([rng.permutation(n) for _ in range(N_PERMUTATIONS)]).astype(np.int16)
        observed = np.empty((k_count, t_count, len(STATISTIC_KEYS)))
        null = np.full((k_count, t_count, N_PERMUTATIONS, len(STATISTIC_KEYS)), np.nan)
        raw_jack = np.empty((k_count, t_count, n, len(STATISTIC_KEYS)))
        adjusted_jack = np.empty_like(raw_jack)
        ranks_all = np.empty_like(distances)
        credits_all = np.empty_like(distances)
        mrr_chances = np.empty((k_count, t_count))
        jack_mrr_chances = np.empty((k_count, t_count, n))
        for k, key in enumerate(keys):
            for t, checkpoint in enumerate(checkpoints):
                protocol_role = "reference" if key == reference_key else ("primary_protocol" if key == primary_key else "control")
                fields = {"dataset": dataset, "checkpoint": int(checkpoint), "metric_key": key,
                          "protocol_role": protocol_role, **_metric_fields(key)}
                ranks, credits = rank_candidates(distances[k, t])
                ranks_all[k, t], credits_all[k, t] = ranks, credits
                mrr_chance = float((1 / ranks).mean())
                mrr_chances[k, t] = mrr_chance
                observed[k, t] = mapping_statistics(ranks, credits, np.arange(n))[0]
                raw_jack[k, t], jack_mrr_chances[k, t] = jackknife_retrieval(distances[k, t])
                adjusted_jack[k, t] = harmonize_replicates(raw_jack[k, t], n, mrr_chance)
                current_null = None
                if checkpoint != 0:
                    current_null = mapping_statistics(ranks, credits, mappings)
                    null[k, t] = current_null
                role = "sanity" if checkpoint == 0 else ("primary" if checkpoint == 20 and key == primary_key else "exploratory")
                rows.append({**fields, **_summary(observed[k, t], adjusted_jack[k, t], current_null, n, mrr_chance),
                             "inferential_role": role, "permutation_seed": permutation_seed})
                for i, run_id in enumerate(run_ids):
                    query_rows.append({**fields, "query_index": i, "query_run_id": run_id,
                                       "correct_candidate_index": i, "correct_candidate_run_id": run_id,
                                       "correct_distance": float(distances[k, t, i, i]),
                                       "correct_rank": float(ranks[i, i]),
                                       "top1_credit": float(credits[i, i]), "reciprocal_rank": float(1 / ranks[i, i]),
                                       "n_minimum_ties": int(np.count_nonzero(credits[i])),
                                       "n_candidates": n})
                    jackknife_rows.append({**fields, "deleted_index": i, "deleted_run": run_id,
                                           "n_candidates_deleted_sample": n - 1,
                                           "mrr_chance_deleted_sample": jack_mrr_chances[k, t, i],
                                           **dict(zip(STATISTIC_KEYS, raw_jack[k, t, i])),
                                           **{name + "_full_n_scale": value for name, value in
                                              zip(STATISTIC_KEYS, adjusted_jack[k, t, i])}})
        _write_npz(out / "data" / f"{dataset}_inference.npz",
                   observed_statistics=observed, null_statistics=null.astype(np.float32),
                   statistic_keys=np.asarray(STATISTIC_KEYS), jackknife_statistics=raw_jack,
                   jackknife_full_n_scale=adjusted_jack, mrr_chance=mrr_chances,
                   jackknife_mrr_chance=jack_mrr_chances,
                   label_permutations=mappings, observed_identity=np.arange(n, dtype=np.int16),
                   candidate_ranks=ranks_all, top1_credits=credits_all,
                   metric_keys=np.asarray(keys), checkpoints=checkpoints, run_ids=run_ids,
                   seeds=data["seeds"], permutation_seed=permutation_seed)
        dataset_results.append({"dataset": dataset, "observed": observed, "null": null,
                                "jack": adjusted_jack, "mrr_chances": mrr_chances, "run_ids": run_ids})
    frame = pd.DataFrame(rows)
    frame["p_holm_less"] = np.nan
    frame["p_holm_greater"] = np.nan
    primary_mask = frame["inferential_role"] == "primary"
    if primary_key is not None:
        if primary_mask.sum() != len(DATASETS):
            raise ValueError("Expected exactly three prespecified primary endpoint tests")
        adjusted = holm_adjust(frame.loc[primary_mask, "p_perm_less"].to_numpy())
        frame.loc[primary_mask, "p_holm_less"] = adjusted
        frame.loc[primary_mask, "p_holm_greater"] = adjusted
    _write_csv(out / "retrieval_by_dataset.csv", frame)
    _write_csv(out / "per_query_ranks.csv", query_rows)
    _write_csv(out / "retrieval_jackknife.csv", jackknife_rows)
    aggregate_rows, aggregate_jack_rows = [], []
    aggregate_observed = np.mean([r["observed"] for r in dataset_results], axis=0)
    aggregate_null = np.mean([r["null"] for r in dataset_results], axis=0)
    aggregate_jack = np.empty((len(expected_keys), len(expected_checkpoints), 3 * expected_n, len(STATISTIC_KEYS)))
    for k, key in enumerate(expected_keys):
        for t, checkpoint in enumerate(expected_checkpoints):
            protocol_role = "reference" if key == reference_key else ("primary_protocol" if key == primary_key else "control")
            fields = {"checkpoint": int(checkpoint), "metric_key": key,
                      "protocol_role": protocol_role, **_metric_fields(key)}
            variance = np.zeros(len(STATISTIC_KEYS))
            for stratum, result in enumerate(dataset_results):
                replicas = (aggregate_observed[k, t] +
                            (result["jack"][k, t] - result["observed"][k, t]) / len(DATASETS))
                aggregate_jack[k, t, stratum * expected_n:(stratum + 1) * expected_n] = replicas
                variance += jackknife_se(replicas) ** 2
                for i, run_id in enumerate(result["run_ids"]):
                    aggregate_jack_rows.append({**fields, "dataset_stratum": result["dataset"],
                                               "deleted_index": i, "deleted_run": run_id,
                                               **dict(zip(STATISTIC_KEYS, replicas[i]))})
            mrr_chance = float(np.mean([r["mrr_chances"][k, t] for r in dataset_results]))
            current_null = None if checkpoint == 0 else aggregate_null[k, t]
            role = "sanity" if checkpoint == 0 else ("secondary_summary" if checkpoint == 20 and key == primary_key else "exploratory")
            aggregate_rows.append({**fields, **_summary(aggregate_observed[k, t], aggregate_jack[k, t],
                                                       current_null, expected_n, mrr_chance, np.sqrt(variance)),
                                   "n_datasets": len(DATASETS), "n_runs_total": len(DATASETS) * expected_n,
                                   "inferential_role": role,
                                   "aggregation": "equal mean of dataset statistics; independent identity permutations per stratum"})
    _write_csv(out / "aggregate.csv", aggregate_rows)
    _write_csv(out / "aggregate_jackknife.csv", aggregate_jack_rows)
    _write_npz(out / "data" / "aggregate_inference.npz",
               observed_statistics=aggregate_observed, null_statistics=aggregate_null.astype(np.float32),
               jackknife_statistics=aggregate_jack, statistic_keys=np.asarray(STATISTIC_KEYS),
               metric_keys=np.asarray(expected_keys), checkpoints=expected_checkpoints,
               datasets=np.asarray(DATASETS), permutation_seeds=np.arange(base_seed + 1, base_seed + 4))
    z_values = frame.loc[frame.checkpoint != 0, "rank_score_null_mean_z"].dropna().to_numpy()
    initial = frame.loc[frame.checkpoint == 0]
    diagnostics = {
        "base_seed": int(base_seed), "permutation_seeds": {name: int(base_seed + i + 1) for i, name in enumerate(DATASETS)},
        "n_permutations": N_PERMUTATIONS, "minimum_monte_carlo_p": 1 / (N_PERMUTATIONS + 1),
        "rng": "numpy.random.RandomState; independent dataset streams; iid uniform bijections with replacement",
        "n_variants": len(expected_keys), "n_checkpoints": len(expected_checkpoints), "n_runs_per_dataset": expected_n,
        "primary_key": primary_key, "reference_key": reference_key,
        "controls": bool(controls), "primary_checkpoint": 20,
        "primary_direction": "smaller mean correct-parent rank; equivalently greater rank_score",
        "holm_family": "three dataset E20 tests for prespecified primary key, absent for purely exploratory analyses",
        "calculation_dtype": "float64", "stored_null_dtype": "float32",
        "mapping_convention": "mapping[i] identifies the candidate treated as the parent of query i; one bijection for all rows",
        "tie_definition": "exact saved-distance equality, average ranks, fractional 1/number_of_minima top1 credit",
        "rank_score_definition": "(n-mean_rank)/(n-1); chance .5, perfect 1, worst 0",
        "top1_score_definition": "(top1-1/n)/(1-1/n); chance 0, perfect 1",
        "mrr_definition": "mean reciprocal average rank; chance is mean reciprocal rank across all matrix cells",
        "mrr_score_definition": "(mrr-mrr_chance)/(1-mrr_chance); chance 0, perfect 1",
        "jackknife": "delete paired query row and candidate column, rerank, normalize for changed candidate count; raw CIs back-transformed to full-n scale",
        "aggregate_jackknife": "sum of three within-stratum (n-1)/n centered sum-of-squares; equal dataset weights",
        "ci": "observed estimate plus/minus 1.95996 times jackknife SE; unbounded, uncorrected normal intervals",
        "e0_inference": "sanity only: no p-values and null arrays NaN; distances/diagonal never overridden",
        "e0_all_correct_rank_one": bool(np.allclose(initial.mean_rank, 1, atol=1e-12, rtol=0)),
        "e0_all_top1_one": bool(np.allclose(initial.top1, 1, atol=1e-12, rtol=0)),
        "null_centering": {"rank_score_expected": .5, "max_abs_mean_z": float(np.max(np.abs(z_values))) if len(z_values) else None,
                            "n_outside_6_mcse": int(np.count_nonzero(np.abs(z_values) > 6)),
                            "interpretation": "Monte Carlo diagnostic, not an assertion or a seed-selection rule"},
        "primary_endpoints": json.loads(frame.loc[primary_mask].to_json(orient="records")),
        "files": ["retrieval_by_dataset.csv", "per_query_ranks.csv", "retrieval_jackknife.csv",
                  "aggregate.csv", "aggregate_jackknife.csv", "data/{dataset}_inference.npz", "data/aggregate_inference.npz"],
    }
    (out / "inference_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--exploratory", action="store_true", help="No confirmatory primary key or Holm family")
    args = parser.parse_args()
    run_inference(args.output, base_seed=args.base_seed, controls=args.exploratory)
