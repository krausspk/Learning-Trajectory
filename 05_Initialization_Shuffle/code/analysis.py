"""Prespecified blocked analysis of the crossed initialization x shuffle design.

run_analysis(out: Path) requires the complete validated 150-run manifest. It
never trains models or changes earlier experiments. The primary endpoint is
cross-shuffle E20 initialization retrieval in canonical unit weight space.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "05_Initialization_Shuffle"
DATASETS = ("mnist", "fashion_mnist", "kmnist")
BASE_SEED = 2026112100
N_PERMUTATIONS = 10000
WEIGHT_KEY = "normalized_frobenius|total|optimal|canonical"
CKA_KEY = "one_minus_cka|hidden|none|raw"
METRIC_KEYS = (WEIGHT_KEY, CKA_KEY)
STATISTIC_KEYS = ("mean_rank", "rank_score", "top1", "top1_score", "mrr", "mrr_score")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, Path(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BASE = _module(ROOT / "02_Initialization_Retrieval/code/inference.py", "_crossed_frozen_retrieval_inference")


def metric_fields(key):
    metric, layer, alignment, gauge = key.split("|")
    return {"metric_key": key, "metric": metric, "layer": layer, "alignment": alignment, "gauge": gauge}


def save_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def rank_bank(matrices):
    """Rank candidates independently for every query and comparison matrix."""
    matrices = np.asarray(matrices, dtype=np.float64)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2] or matrices.shape[1] < 2:
        raise ValueError("Expected [comparison, query, candidate] square matrices with n>=2")
    if not np.isfinite(matrices).all():
        raise ValueError("All comparison distances must be finite")
    ranks = rankdata(matrices, axis=-1, method="average")
    nearest = matrices == matrices.min(axis=-1, keepdims=True)
    credit = nearest / nearest.sum(axis=-1, keepdims=True)
    return ranks, credit


def _statistics(mean_rank, top1, mrr, n, mrr_chance):
    return np.stack((mean_rank, (n - mean_rank) / (n - 1), top1,
                     (top1 - 1 / n) / (1 - 1 / n), mrr,
                     (mrr - mrr_chance) / (1 - mrr_chance)), axis=-1)


def evaluate_bank(ranks, credit, rows, columns):
    """Average a full set of bijective row/column mappings, never individual edges.

    rows/columns are [draw, comparison, common_identity]. Each mapping must
    cover all physical queries/candidates once within a comparison.
    """
    ranks, credit = np.asarray(ranks), np.asarray(credit)
    rows, columns = np.asarray(rows, dtype=int), np.asarray(columns, dtype=int)
    p, n, _ = ranks.shape
    if rows.shape != columns.shape or rows.ndim != 3 or rows.shape[1:] != (p, n):
        raise ValueError("Mappings must have shape [draw, comparison, identity]")
    if not np.all(np.sort(rows, axis=-1) == np.arange(n)) or not np.all(np.sort(columns, axis=-1) == np.arange(n)):
        raise ValueError("Each row and column map must be a bijection")
    selected = ranks[np.arange(p)[None, :, None], rows, columns]
    selected_credit = credit[np.arange(p)[None, :, None], rows, columns]
    return _statistics(selected.mean(axis=(1, 2)), selected_credit.mean(axis=(1, 2)),
                       (1 / selected).mean(axis=(1, 2)), n, float((1 / ranks).mean()))


def observed_bank(ranks, credit):
    p, n, _ = ranks.shape
    diagonal = np.diagonal(ranks, axis1=-2, axis2=-1)
    first = np.diagonal(credit, axis1=-2, axis2=-1)
    chance = float((1 / ranks).mean())
    per_identity = _statistics(diagonal.mean(axis=0), first.mean(axis=0),
                               (1 / diagonal).mean(axis=0), n, chance)
    return per_identity.mean(axis=0), per_identity, chance


def factor_matrices(tensor, factor):
    """[init,shuffle,init,shuffle] -> ordered cross-block retrieval matrices."""
    tensor = np.asarray(tensor, dtype=np.float64)
    i, s, ii, ss = tensor.shape
    if i != ii or s != ss:
        raise ValueError("Expected a complete crossed distance tensor")
    if factor == "initialization":
        block = tensor.transpose(1, 3, 0, 2)  # shuffle, shuffle, init, init
    elif factor == "shuffle":
        block = tensor.transpose(0, 2, 1, 3)  # init, init, shuffle, shuffle
    else:
        raise ValueError("Factor must be initialization or shuffle")
    pairs = np.asarray([(a, b) for a in range(len(block)) for b in range(len(block)) if a != b], dtype=int)
    if not len(pairs):
        raise ValueError("At least two blocks are required")
    return block[pairs[:, 0], pairs[:, 1]], pairs


def factor_label_permutations(n_blocks, n_identities, seed, n_permutations=N_PERMUTATIONS):
    """p[draw,block,label]=physical identity; block zero fixes the common gauge."""
    rng = np.random.RandomState(seed)
    maps = np.broadcast_to(np.arange(n_identities), (n_permutations, n_blocks, n_identities)).copy()
    for b in range(n_permutations):
        for block in range(1, n_blocks):
            maps[b, block] = rng.permutation(n_identities)
    return maps.astype(np.int16)


def physical_candidate_mapping(permutations, source, target):
    """For physical query i, target candidate is p[target, inverse(p[source])[i]]."""
    p = np.asarray(permutations)
    return np.take_along_axis(p[:, target], np.argsort(p[:, source], axis=1), axis=1)


def factor_result(tensor, factor, permutations=None):
    matrices, pairs = factor_matrices(tensor, factor)
    ranks, credit = rank_bank(matrices)
    observed, per_identity, chance = observed_bank(ranks, credit)
    p, n, _ = matrices.shape
    same = float(np.diagonal(matrices, axis1=-2, axis2=-1).mean())
    different = float((matrices.sum() - same * p * n) / (p * n * (n - 1)))
    result = {"observed": observed, "per_identity": per_identity, "mrr_chance": chance,
              "ranks": ranks, "credits": credit, "block_pairs": pairs,
              "same_factor_mean": same, "different_factor_mean": different,
              "contrast": different - same}
    if permutations is not None:
        rows, cols = permutations[:, pairs[:, 0]], permutations[:, pairs[:, 1]]
        result["null"] = evaluate_bank(ranks, credit, rows, cols)
        selected = matrices[np.arange(p)[None, :, None], rows, cols].mean(axis=(1, 2))
        nonselected = (matrices.sum() - selected * p * n) / (p * n * (n - 1))
        result["contrast_null"] = nonselected - selected
    return result


def subset_tensor(tensor, initial_indices=None, shuffle_indices=None):
    i, s = tensor.shape[:2]
    initial_indices = np.arange(i) if initial_indices is None else np.asarray(initial_indices)
    shuffle_indices = np.arange(s) if shuffle_indices is None else np.asarray(shuffle_indices)
    return tensor[np.ix_(initial_indices, shuffle_indices, initial_indices, shuffle_indices)]


def factor_jackknife(tensor, factor):
    i, s = tensor.shape[:2]
    n = i if factor == "initialization" else s
    replicas, contrasts, chances = [], [], []
    for deleted in range(n):
        keep = np.delete(np.arange(n), deleted)
        smaller = subset_tensor(tensor, initial_indices=keep) if factor == "initialization" else subset_tensor(tensor, shuffle_indices=keep)
        result = factor_result(smaller, factor)
        replicas.append(result["observed"])
        contrasts.append(result["contrast"])
        chances.append(result["mrr_chance"])
    return np.asarray(replicas), np.asarray(contrasts), np.asarray(chances)


def trajectory_result(distances, candidate_permutations=None):
    """One checkpoint [initialization,shuffle,candidate_initialization]."""
    matrices = np.asarray(distances).transpose(1, 0, 2)
    ranks, credit = rank_bank(matrices)
    observed, per_identity, chance = observed_bank(ranks, credit)
    result = {"observed": observed, "per_identity": per_identity, "mrr_chance": chance,
              "ranks": ranks, "credits": credit}
    if candidate_permutations is not None:
        b, n = candidate_permutations.shape
        p = len(matrices)
        rows = np.broadcast_to(np.arange(n), (b, p, n))
        cols = np.broadcast_to(candidate_permutations[:, None], (b, p, n))
        result["null"] = evaluate_bank(ranks, credit, rows, cols)
    return result


def trajectory_jackknife(distances):
    n, s, _ = distances.shape
    replicas, chances = [], []
    for deleted in range(n):
        keep = np.delete(np.arange(n), deleted)
        result = trajectory_result(distances[np.ix_(keep, np.arange(s), keep)])
        replicas.append(result["observed"])
        chances.append(result["mrr_chance"])
    return np.asarray(replicas), np.asarray(chances)


def performance_decomposition(values):
    """Descriptive two-way SS; one observation per cell cannot isolate residual."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Expected a complete finite initialization x shuffle table")
    i, s = values.shape
    grand = float(values.mean())
    initial_means, shuffle_means = values.mean(axis=1), values.mean(axis=0)
    if np.ptp(values) == 0:
        grand = float(values.flat[0])
        initial_means = np.full(i, grand)
        shuffle_means = np.full(s, grand)
    residual = values - initial_means[:, None] - shuffle_means[None, :] + grand
    total = float(np.sum((values - grand) ** 2))
    init_ss = float(s * np.sum((initial_means - grand) ** 2))
    shuffle_ss = float(i * np.sum((shuffle_means - grand) ** 2))
    residual_ss = float(np.sum(residual ** 2))
    return {"grand_mean": grand, "total_ss": total, "initialization_ss": init_ss,
            "shuffle_ss": shuffle_ss, "interaction_residual_ss": residual_ss,
            "initialization_fraction": init_ss / total if total > 0 else np.nan,
            "shuffle_fraction": shuffle_ss / total if total > 0 else np.nan,
            "interaction_residual_fraction": residual_ss / total if total > 0 else np.nan,
            "n_initializations": i, "n_shuffle_levels": s,
            "initialization_df": i - 1, "shuffle_df": s - 1,
            "interaction_residual_df": (i - 1) * (s - 1),
            "inferential_role": "descriptive_no_anova_p_one_observation_per_cell"}, initial_means, shuffle_means


def distance_groups(tensor):
    i, s = tensor.shape[:2]
    records = {"same_initialization_different_shuffle": [], "different_initialization_same_shuffle": [],
               "different_initialization_different_shuffle": []}
    for a, b in itertools.combinations(range(i * s), 2):
        ia, sa = divmod(a, s)
        ib, sb = divmod(b, s)
        name = ("same_initialization_different_shuffle" if ia == ib else
                "different_initialization_same_shuffle" if sa == sb else "different_initialization_different_shuffle")
        records[name].append(float(tensor[ia, sa, ib, sb]))
    return {key: {"n_pairs": len(value), "mean": float(np.mean(value)), "std_descriptive": float(np.std(value, ddof=1))}
            for key, value in records.items()}


def load_complete_manifest(out):
    """Fail closed until every cell of the 3x10x5 crossed design is complete."""
    path = Path(out) / "data/manifest.json"
    manifest = json.loads(path.read_text())
    records = manifest.get("runs", [])
    if not manifest.get("complete") or manifest.get("expected_runs") != 150 or len(records) != 150:
        raise ValueError("Analysis requires a complete validated 150-run manifest; training is still incomplete")
    expected = {(d, i, s) for d in DATASETS for i in range(1, 11) for s in range(1, 6)}
    actual = [(r["dataset"], r["init_id"], r["shuffle_id"]) for r in records]
    if len(set(actual)) != 150 or set(actual) != expected:
        raise ValueError("Manifest does not cover the complete initialization x shuffle grid exactly once")
    schedules = {r["shuffle_id"]: r for r in manifest["schedules"]}
    source_initials = manifest["sources"]["initializations"]
    for record in records:
        if record["checkpoints"] != 21 or sha256(record["path"]) != record["sha256"]:
            raise ValueError("A trained checkpoint archive is incomplete or differs from its validated hash")
        source = source_initials[record["original_run_id"]]
        if (record["dataset"], record["init_id"], record["init_seed"]) != (source["dataset"], source["init_id"], source["init_seed"]):
            raise ValueError("Training record initial identity differs from the original initialization manifest")
        schedule = schedules[record["shuffle_id"]]
        if record["shuffle_seed"] != schedule["shuffle_seed"] or record["epoch_order_sha256"] != schedule["epoch_order_sha256"]:
            raise ValueError("Training record does not use the prescribed shared shuffle schedule")
        for key in ("E0_W0_sha256", "E0_W1_sha256"):
            if record[key] != source[key]:
                raise ValueError("Training record initial weights differ from the original source")
    for source in source_initials.values():
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError("Original initialization source archive has changed")
    return manifest, sorted(records, key=lambda r: (DATASETS.index(r["dataset"]), r["init_id"], r["shuffle_id"]))


def _analysis_fingerprint(out, manifest):
    sources = {
        "analysis": {"path": str(Path(__file__).resolve()), "sha256": sha256(__file__)},
        "design": {"path": str(Path(out) / "analysis_design.json"), "sha256": sha256(Path(out) / "analysis_design.json")},
        "manifest": {"path": str(Path(out) / "data/manifest.json"), "sha256": sha256(Path(out) / "data/manifest.json")},
        "weight_distance": {"path": str(ROOT / "02_Initialization_Retrieval/code/distances.py"), "sha256": sha256(ROOT / "02_Initialization_Retrieval/code/distances.py")},
        "cka_distance": {"path": str(ROOT / "04_Representation_Retrieval/code/representations.py"), "sha256": sha256(ROOT / "04_Representation_Retrieval/code/representations.py")},
        "inference_helpers": {"path": str(ROOT / "02_Initialization_Retrieval/code/inference.py"), "sha256": sha256(ROOT / "02_Initialization_Retrieval/code/inference.py")},
        "probe": {"path": str(ROOT / "outputs/probes/main.npz"), "sha256": sha256(ROOT / "outputs/probes/main.npz")},
    }
    fingerprint = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    return fingerprint, sources


def build_dataset_distances(out, dataset, records, manifest, fingerprint, sources):
    path = Path(out) / "data" / f"{dataset}_distances.npz"
    if path.exists():
        with np.load(path, allow_pickle=False) as cached:
            if str(cached["analysis_fingerprint"]) == fingerprint:
                return {key: cached[key] for key in cached.files}
    weights = _module(ROOT / "02_Initialization_Retrieval/code/distances.py", "_crossed_weight_distance")
    representations = _module(ROOT / "04_Representation_Retrieval/code/representations.py", "_crossed_representation_distance")
    chosen = [r for r in records if r["dataset"] == dataset]
    runs = {}
    accuracies, losses = np.empty((21, 10, 5)), np.empty((21, 10, 5))
    run_ids = np.empty((10, 5), dtype="U32")
    for record in chosen:
        i, s = record["init_id"] - 1, record["shuffle_id"] - 1
        with np.load(record["path"], allow_pickle=False) as archive:
            w0, w1 = archive["W0"], archive["W1"]
            if w0.shape != (21, 784, 64) or w1.shape != (21, 64, 10) or not np.array_equal(archive["checkpoints"], np.arange(21)):
                raise ValueError("Unexpected crossed-training checkpoint dimensions")
            if not np.isfinite(w0).all() or not np.isfinite(w1).all() or np.any(archive["bias0"] != 0) or np.any(archive["bias1"] != 0):
                raise ValueError("Nonfinite weights or nonzero biases in crossed training")
            metadata = json.loads(str(archive["metadata"]))
            for field in ("dataset", "run_id", "init_id", "shuffle_id", "init_seed", "shuffle_seed", "original_run_id"):
                if metadata[field] != record[field]:
                    raise ValueError(f"Archive identity metadata differs from manifest: {field}")
            accuracies[:, i, s], losses[:, i, s] = archive["accuracy"], archive["loss"]
        runs[i, s] = (w0, w1)
        run_ids[i, s] = record["run_id"]
        source = manifest["sources"]["initializations"][record["original_run_id"]]
        with np.load(source["path"], allow_pickle=False) as original:
            if not np.array_equal(w0[0], original["W0"][0]) or not np.array_equal(w1[0], original["W1"][0]):
                raise ValueError("Crossed run does not reuse the exact original initial state")
    if not np.isfinite(accuracies).all() or not np.isfinite(losses).all():
        raise ValueError("Performance arrays must be complete and finite")
    initial = [weights.canonical_unit_network(runs[i, 0][0][0], runs[i, 0][1][0])[0] for i in range(10)]
    initial_distances = np.empty((21, 10, 5, 10))
    initial_permutations = np.empty((21, 10, 5, 10, 64), dtype=np.int16)
    with threadpool_limits(limits=1):
        for checkpoint in range(21):
            current = [weights.canonical_unit_network(runs[i, s][0][checkpoint], runs[i, s][1][checkpoint])[0]
                       for i in range(10) for s in range(5)]
            values, mappings = weights.cross_distances(current, initial, already_normalized=True)
            initial_distances[checkpoint] = values.reshape(10, 5, 10)
            initial_permutations[checkpoint] = mappings.reshape(10, 5, 10, 64)
            print(f"Crossed distances: {dataset}, E{checkpoint}, 50 x 10 initial candidates", flush=True)
        pairs = np.asarray(list(itertools.combinations(range(50), 2)), dtype=np.int16)
        endpoint = np.empty((50, 50))
        permutations = np.empty((len(pairs), 64), dtype=np.int16)
        for index in range(50):
            endpoint[index, index], _ = weights.match_unit_networks(current[index], current[index])
        for index, (a, b) in enumerate(pairs):
            value, mapping = weights.match_unit_networks(current[a], current[b])
            endpoint[a, b] = endpoint[b, a] = value
            permutations[index] = mapping
        with np.load(ROOT / "outputs/probes/main.npz", allow_pickle=False) as archive:
            probe = archive["X"].astype(np.float64)
        if probe.shape != (3000, 784):
            raise ValueError("Unexpected fixed representation probe dimensions")
        centered, normalizers = [], []
        for i in range(10):
            for s in range(5):
                hidden = np.maximum(probe @ runs[i, s][0][20].astype(np.float64), 0)
                features, normalizer, _ = representations.centered_features(hidden)
                centered.append(features)
                normalizers.append(normalizer)
        cka, raw_cka = representations.cka_cross(centered, centered, normalizers, normalizers)
    if not np.isfinite(endpoint).all() or not np.isfinite(initial_distances).all() or not np.isfinite(cka).all():
        raise AssertionError("Invalid crossed-design distances")
    e0_error = max(float(initial_distances[0, i, s, i]) for i in range(10) for s in range(5))
    if e0_error > 1e-12 or np.max(np.abs(np.diag(endpoint))) > 1e-12 or np.max(np.abs(np.diag(cka))) > 1e-10:
        raise AssertionError("Crossed-design self-distance sanity failed")
    if np.max(np.abs(cka - cka.T)) > 1e-10:
        raise AssertionError("Crossed endpoint CKA is not symmetric")
    data = {"endpoint_distances": endpoint.reshape(10, 5, 10, 5),
            "endpoint_cka_distances": cka.reshape(10, 5, 10, 5), "endpoint_cka_unclipped": raw_cka.reshape(10, 5, 10, 5),
            "initial_distances": initial_distances, "permutations_initial": initial_permutations,
            "permutations_endpoint": permutations, "endpoint_pairs": pairs,
            "init_ids": np.arange(1, 11), "shuffle_ids": np.arange(1, 6), "run_ids": run_ids,
            "shuffle_seeds": np.asarray([11001, 12001, 13001, 14001, 15001]),
            "checkpoints": np.arange(21), "metric_keys": np.asarray(METRIC_KEYS),
            "accuracy": accuracies, "loss": losses, "probe_sha256": np.asarray(sources["probe"]["sha256"]),
            "analysis_fingerprint": np.asarray(fingerprint), "e0_max_self_distance": np.asarray(e0_error)}
    BASE._write_npz(path, **data)
    return data


def _summary(result, raw_jack, n, role, null=None):
    adjusted = BASE.harmonize_replicates(raw_jack, n, result["mrr_chance"])
    summary = BASE._summary(result["observed"], adjusted, null, n, result["mrr_chance"])
    summary["inferential_role"] = role
    return summary, adjusted


def _append_identity_rows(rows, fields, values, identity_factor):
    for index, statistics in enumerate(values):
        rows.append({**fields, "identity_factor": identity_factor, "identity_id": index + 1,
                     **dict(zip(STATISTIC_KEYS, statistics))})


def _append_jack_rows(rows, fields, raw, adjusted, identity_factor, contrasts=None):
    for index in range(len(raw)):
        row = {**fields, "deleted_factor": identity_factor, "deleted_id": index + 1,
               **dict(zip(STATISTIC_KEYS, raw[index])),
               **{name + "_full_n_scale": value for name, value in zip(STATISTIC_KEYS, adjusted[index])}}
        if contrasts is not None:
            row["distance_contrast"] = contrasts[index]
        rows.append(row)


def analyze_dataset(out, dataset, data, dataset_index):
    """Blocked inference from an already computed crossed distance archive."""
    initialization_maps = factor_label_permutations(5, 10, BASE_SEED + dataset_index + 1, N_PERMUTATIONS)
    shuffle_maps = factor_label_permutations(10, 5, BASE_SEED + dataset_index + 101, N_PERMUTATIONS)
    rng = np.random.RandomState(BASE_SEED + dataset_index + 201)
    trajectory_maps = np.stack([rng.permutation(10) for _ in range(N_PERMUTATIONS)]).astype(np.int16)
    result_rows = {key: [] for key in ("initialization_retrieval", "shuffle_retrieval", "initial_trajectory_retrieval",
                                     "distance_contrasts", "distance_groups", "per_identity_scores", "jackknife",
                                     "leave_factor_level_out", "performance_decomposition", "performance_factor_means")}
    records = []
    endpoint_observed = np.empty((2, 2, 6))
    endpoint_null = np.empty((2, 2, N_PERMUTATIONS, 6))
    endpoint_contrast = np.empty((2, 2))
    endpoint_contrast_null = np.empty((2, 2, N_PERMUTATIONS))
    init_jack = np.empty((2, 10, 6)); shuffle_jack = np.empty((2, 5, 6))
    init_jack_raw = np.empty_like(init_jack); shuffle_jack_raw = np.empty_like(shuffle_jack)
    init_contrast_jack = np.empty((2, 10)); shuffle_contrast_jack = np.empty((2, 5))
    rank_arrays, credit_arrays = {}, {}
    endpoint_tensors = (data["endpoint_distances"], data["endpoint_cka_distances"])
    for k, (key, tensor) in enumerate(zip(METRIC_KEYS, endpoint_tensors)):
        for group, values in distance_groups(tensor).items():
            result_rows["distance_groups"].append({"dataset": dataset, "checkpoint": 20, **metric_fields(key),
                                                   "group": group, **values, "inferential_role": "descriptive"})
        for f, factor in enumerate(("initialization", "shuffle")):
            maps = initialization_maps if factor == "initialization" else shuffle_maps
            n = 10 if factor == "initialization" else 5
            result = factor_result(tensor, factor, maps)
            raw, contrast_jack, chance_jack = factor_jackknife(tensor, factor)
            analysis = factor + "_retrieval"
            fields = {"dataset": dataset, "checkpoint": 20, "analysis": analysis, **metric_fields(key)}
            role = "primary" if factor == "initialization" and key == WEIGHT_KEY else "exploratory"
            row, adjusted = _summary(result, raw, n, role, result["null"])
            row.update({**fields, "n_initializations": 10, "n_shuffle_levels": 5,
                        "n_ordered_block_comparisons": len(result["block_pairs"]),
                        "n_query_comparisons": len(result["block_pairs"]) * n,
                        "independent_identity_blocks": n,
                        "ci_conditioning": "five selected shuffle levels" if factor == "initialization" else "ten selected initializations"})
            result_rows[analysis].append(row)
            _append_identity_rows(result_rows["per_identity_scores"], fields, result["per_identity"], factor)
            _append_jack_rows(result_rows["jackknife"], fields, raw, adjusted, factor, contrast_jack)
            endpoint_observed[k, f], endpoint_null[k, f] = result["observed"], result["null"]
            endpoint_contrast[k, f], endpoint_contrast_null[k, f] = result["contrast"], result["contrast_null"]
            if factor == "initialization":
                init_jack[k], init_jack_raw[k], init_contrast_jack[k] = adjusted, raw, contrast_jack
            else:
                shuffle_jack[k], shuffle_jack_raw[k], shuffle_contrast_jack[k] = adjusted, raw, contrast_jack
            rank_arrays[f"{factor}_ranks_{k}"] = result["ranks"]
            credit_arrays[f"{factor}_top1_credits_{k}"] = result["credits"]
            contrast_se = float(BASE.jackknife_se(contrast_jack))
            result_rows["distance_contrasts"].append({**fields, "factor": factor,
                "same_factor_mean": result["same_factor_mean"], "different_factor_mean": result["different_factor_mean"],
                "contrast": result["contrast"], "jackknife_se": contrast_se,
                "ci_low": result["contrast"] - BASE.Z95 * contrast_se, "ci_high": result["contrast"] + BASE.Z95 * contrast_se,
                "p_perm_greater": BASE.permutation_pvalue(result["contrast"], result["contrast_null"], "greater"),
                "n_permutations": N_PERMUTATIONS, "inferential_role": "exploratory",
                "definition": "different-initialization/different-shuffle minus same-initialization/different-shuffle" if factor == "initialization" else
                              "different-initialization/different-shuffle minus different-initialization/same-shuffle"})
            records.append({"dataset": dataset, "analysis": analysis, "checkpoint": 20, "key": key,
                            "result": result, "adjusted_jack": adjusted, "n": n,
                            "contrast_jack": contrast_jack, "summary": row})
            # The opposite factor is omitted as a separate robustness estimate;
            # these overlapping estimates are not independent jackknife samples.
            other_n = 5 if factor == "initialization" else 10
            for deleted in range(other_n):
                keep = np.delete(np.arange(other_n), deleted)
                smaller = subset_tensor(tensor, shuffle_indices=keep) if factor == "initialization" else subset_tensor(tensor, initial_indices=keep)
                robust = factor_result(smaller, factor)
                result_rows["leave_factor_level_out"].append({**fields,
                    "omitted_factor": "shuffle" if factor == "initialization" else "initialization",
                    "omitted_level": deleted + 1, **dict(zip(STATISTIC_KEYS, robust["observed"])),
                    "distance_contrast": robust["contrast"], "inferential_role": "descriptive_sensitivity_no_independent_replicates"})
    trajectory_observed = np.empty((21, 6)); trajectory_null = np.full((21, N_PERMUTATIONS, 6), np.nan)
    trajectory_raw_jack = np.empty((21, 10, 6)); trajectory_adjusted_jack = np.empty_like(trajectory_raw_jack)
    trajectory_ranks = np.empty((21, 5, 10, 10)); trajectory_credits = np.empty_like(trajectory_ranks)
    for t in range(21):
        result = trajectory_result(data["initial_distances"][t], None if t == 0 else trajectory_maps)
        raw, chance_jack = trajectory_jackknife(data["initial_distances"][t])
        fields = {"dataset": dataset, "checkpoint": t, "analysis": "initial_trajectory_retrieval", **metric_fields(WEIGHT_KEY)}
        row, adjusted = _summary(result, raw, 10, "sanity" if t == 0 else "exploratory", result.get("null"))
        result_rows["initial_trajectory_retrieval"].append({**fields, **row,
            "n_initializations": 10, "n_shuffle_levels": 5, "n_queries_total": 50,
            "independent_identity_blocks": 10, "ci_conditioning": "five selected shuffle levels"})
        _append_identity_rows(result_rows["per_identity_scores"], fields, result["per_identity"], "initialization")
        _append_jack_rows(result_rows["jackknife"], fields, raw, adjusted, "initialization")
        trajectory_observed[t], trajectory_raw_jack[t], trajectory_adjusted_jack[t] = result["observed"], raw, adjusted
        trajectory_ranks[t], trajectory_credits[t] = result["ranks"], result["credits"]
        if t != 0:
            trajectory_null[t] = result["null"]
        records.append({"dataset": dataset, "analysis": fields["analysis"], "checkpoint": t, "key": WEIGHT_KEY,
                        "result": result, "adjusted_jack": adjusted, "n": 10, "summary": row})
    for metric in ("accuracy", "loss"):
        for t in range(21):
            decomposed, init_means, shuffle_means = performance_decomposition(data[metric][t])
            result_rows["performance_decomposition"].append({"dataset": dataset, "checkpoint": t, "metric": metric, **decomposed})
            for factor, means in (("initialization", init_means), ("shuffle", shuffle_means)):
                for identity, value in enumerate(means):
                    result_rows["performance_factor_means"].append({"dataset": dataset, "checkpoint": t, "metric": metric,
                        "factor": factor, "identity_id": identity + 1, "mean": value,
                        "effect_from_grand_mean": value - decomposed["grand_mean"], "inferential_role": "descriptive"})
    BASE._write_npz(Path(out) / "data" / f"{dataset}_inference.npz",
        metric_keys=np.asarray(METRIC_KEYS), statistic_keys=np.asarray(STATISTIC_KEYS),
        factor_keys=np.asarray(["initialization", "shuffle"]),
        initialization_label_permutations=initialization_maps, shuffle_label_permutations=shuffle_maps,
        trajectory_candidate_permutations=trajectory_maps,
        observed_initialization_identity=np.broadcast_to(np.arange(10), (5, 10)),
        observed_shuffle_identity=np.broadcast_to(np.arange(5), (10, 5)),
        endpoint_observed=endpoint_observed, endpoint_null=endpoint_null.astype(np.float32),
        initialization_jackknife_raw=init_jack_raw, initialization_jackknife_full_scale=init_jack,
        shuffle_jackknife_raw=shuffle_jack_raw, shuffle_jackknife_full_scale=shuffle_jack,
        endpoint_contrast=endpoint_contrast, endpoint_contrast_null=endpoint_contrast_null.astype(np.float32),
        initialization_contrast_jackknife=init_contrast_jack, shuffle_contrast_jackknife=shuffle_contrast_jack,
        trajectory_observed=trajectory_observed, trajectory_null=trajectory_null.astype(np.float32),
        trajectory_jackknife_raw=trajectory_raw_jack, trajectory_jackknife_full_scale=trajectory_adjusted_jack,
        trajectory_candidate_ranks=trajectory_ranks, trajectory_top1_credits=trajectory_credits,
        checkpoints=np.arange(21), init_ids=data["init_ids"], shuffle_ids=data["shuffle_ids"],
        seeds=np.asarray([BASE_SEED + dataset_index + offset for offset in (1, 101, 201)]),
        **rank_arrays, **credit_arrays)
    return result_rows, records


def aggregate_results(out, records):
    rows, contrast_rows, jack_rows = [], [], []
    groups = {}
    for record in records:
        groups.setdefault((record["analysis"], record["key"], record["checkpoint"]), []).append(record)
    null_arrays, observed_arrays, labels = [], [], []
    for (analysis, key, checkpoint), strata in groups.items():
        if len(strata) != 3:
            raise ValueError("Aggregate must contain exactly three dataset strata")
        observed = np.mean([s["result"]["observed"] for s in strata], axis=0)
        chance = float(np.mean([s["result"]["mrr_chance"] for s in strata]))
        null = None if checkpoint == 0 else np.mean([s["result"]["null"] for s in strata], axis=0)
        variance = np.zeros(6); replicas = []
        for stratum in strata:
            reps = observed + (stratum["adjusted_jack"] - stratum["result"]["observed"]) / 3
            variance += BASE.jackknife_se(reps) ** 2
            replicas.extend(reps)
            for deleted, values in enumerate(reps):
                jack_rows.append({"analysis": analysis, "checkpoint": checkpoint, **metric_fields(key),
                                  "dataset_stratum": stratum["dataset"], "deleted_id": deleted + 1,
                                  **dict(zip(STATISTIC_KEYS, values))})
        row = BASE._summary(observed, np.asarray(replicas), null, strata[0]["n"], chance, np.sqrt(variance))
        rows.append({"analysis": analysis, "checkpoint": checkpoint, **metric_fields(key), **row,
                     "n_datasets": 3, "n_identity_blocks_total": 3 * strata[0]["n"],
                     "inferential_role": "sanity" if checkpoint == 0 else "secondary_summary"})
        labels.append(f"{analysis}|{key}|{checkpoint}")
        observed_arrays.append(observed)
        null_arrays.append(np.full((N_PERMUTATIONS, 6), np.nan) if null is None else null)
        if "contrast_jack" in strata[0]:
            estimate = float(np.mean([s["result"]["contrast"] for s in strata]))
            contrast_null = np.mean([s["result"]["contrast_null"] for s in strata], axis=0)
            se = float(np.sqrt(sum(float(BASE.jackknife_se(s["contrast_jack"] / 3)) ** 2 for s in strata)))
            contrast_rows.append({"analysis": analysis, "checkpoint": checkpoint, **metric_fields(key),
                "contrast": estimate, "jackknife_se": se, "ci_low": estimate - BASE.Z95 * se,
                "ci_high": estimate + BASE.Z95 * se,
                "p_perm_greater": BASE.permutation_pvalue(estimate, contrast_null),
                "n_datasets": 3, "inferential_role": "exploratory"})
    BASE._write_csv(Path(out) / "aggregate.csv", rows)
    BASE._write_csv(Path(out) / "aggregate_distance_contrasts.csv", contrast_rows)
    BASE._write_csv(Path(out) / "aggregate_jackknife.csv", jack_rows)
    BASE._write_npz(Path(out) / "data/aggregate_inference.npz", analysis_keys=np.asarray(labels),
                    observed_statistics=np.asarray(observed_arrays), null_statistics=np.asarray(null_arrays, dtype=np.float32),
                    statistic_keys=np.asarray(STATISTIC_KEYS), datasets=np.asarray(DATASETS))


def run_analysis(out: Path = DEFAULT_OUT):
    out = Path(out).resolve()
    if out != DEFAULT_OUT:
        raise ValueError("Real crossed-design analyses must remain isolated in 05_Initialization_Shuffle")
    manifest, records = load_complete_manifest(out)
    fingerprint, sources = _analysis_fingerprint(out, manifest)
    all_rows, all_records, outputs = {}, [], []
    for index, dataset in enumerate(DATASETS):
        data = build_dataset_distances(out, dataset, records, manifest, fingerprint, sources)
        rows, summaries = analyze_dataset(out, dataset, data, index)
        for name, values in rows.items():
            all_rows.setdefault(name, []).extend(values)
        all_records.extend(summaries)
        path = out / "data" / f"{dataset}_distances.npz"
        outputs.append({"dataset": dataset, "path": str(path), "sha256": sha256(path),
                        "weight_endpoint_shape": list(data["endpoint_distances"].shape),
                        "cka_endpoint_shape": list(data["endpoint_cka_distances"].shape),
                        "initial_shape": list(data["initial_distances"].shape),
                        "e0_max_self_distance": float(data["e0_max_self_distance"])})
    primary = pd.DataFrame(all_rows["initialization_retrieval"])
    primary["p_holm_less"] = np.nan
    primary["p_holm_greater"] = np.nan
    mask = primary.inferential_role == "primary"
    if mask.sum() != 3:
        raise ValueError("Expected exactly three primary initialization-retrieval endpoints")
    adjusted = BASE.holm_adjust(primary.loc[mask, "p_perm_less"].to_numpy())
    primary.loc[mask, "p_holm_less"] = adjusted
    primary.loc[mask, "p_holm_greater"] = adjusted
    all_rows["initialization_retrieval"] = primary
    for name, rows in all_rows.items():
        BASE._write_csv(out / f"{name}.csv", rows)
    aggregate_results(out, all_records)
    z_scores = [r["summary"].get("rank_score_null_mean_z", np.nan) for r in all_records]
    finite_z = np.asarray([z for z in z_scores if np.isfinite(z)])
    diagnostics = {"passed": True, "analysis_fingerprint": fingerprint, "sources": sources, "outputs": outputs,
        "n_complete_training_runs": len(records), "n_datasets": 3, "n_initializations_per_dataset": 10, "n_shuffle_levels": 5,
        "metric_keys": list(METRIC_KEYS), "primary_key": WEIGHT_KEY, "primary_checkpoint": 20,
        "n_permutations": N_PERMUTATIONS, "minimum_monte_carlo_p": 1 / (N_PERMUTATIONS + 1),
        "permutation_seeds": {dataset: {"initialization": BASE_SEED + i + 1, "shuffle": BASE_SEED + i + 101,
                                        "trajectory": BASE_SEED + i + 201} for i, dataset in enumerate(DATASETS)},
        "null_rank_score_expected": .5,
        "null_mean_max_abs_mcse_z": float(np.max(np.abs(finite_z))) if len(finite_z) else None,
        "null_mean_n_outside_6mcse": int(np.count_nonzero(np.abs(finite_z) > 6)),
        "null_diagnostic_policy": "diagnostic only; no assertion or seed selection",
        "calculation_dtype": "float64", "stored_null_dtype": "float32",
        "primary_endpoints": json.loads(primary.loc[mask].to_json(orient="records")),
        "ci_conditioning": "initialization JK conditions on five observed shuffle levels; shuffle JK conditions on ten observed initializations",
        "performance_decomposition": "one observation per cell: interaction and residual are not separable, no ANOVA p-values",
        "old_experiments_modified": False}
    save_json(out / "analysis_validation.json", diagnostics)
    # The source inventory is checked again after computations; no stale or
    # changed old implementations may be silently attributed to the analysis.
    for entry in sources.values():
        if sha256(entry["path"]) != entry["sha256"]:
            raise AssertionError("Analysis input/source changed while the analysis was running")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    run_analysis(args.output)
