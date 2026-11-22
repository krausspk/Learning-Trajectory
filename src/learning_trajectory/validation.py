"""ReLU gauge, permutation stability, integrity and synthetic checks (S6/S8/S11)."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from threadpoolctl import threadpool_limits

from . import metrics as m
from .analysis import file_hash, load_runs, write_csv, write_json
from .data import _atomic_npz
from .statistics import group_summary

GAUGE_ENVELOPES = (1.0, 1.25, 1.5, 2.0, 4.0, 8.0)
GAUGE_SEED = 20261116
STABILITY_SEED = 2026
SYNTHETIC_SEED = 2026


def rescale_network(W0, W1, envelope, seed):
    """Deterministic neuronwise scales with exactly the specified exponent range."""
    if envelope < 1:
        raise ValueError("A rescaling envelope must be at least one")
    rng = np.random.RandomState(seed)
    exponent = rng.permutation(np.linspace(-1.0, 1.0, W0.shape[1]))
    scales = float(envelope) ** exponent
    return np.asarray(W0, dtype=np.float64) * scales, np.asarray(W1, dtype=np.float64) / scales[:, None], scales


def _logits(X, weights):
    result = X
    for i, weight in enumerate(weights):
        result = result @ weight
        if i < len(weights) - 1:
            result = np.maximum(result, 0)
    return result


def _gauge_job(args):
    output, envelope = args
    with threadpool_limits(limits=1):
        runs = load_runs(output, "main")
        probes = np.load(Path(output) / "probes" / "main.npz")["X"][:256].astype(np.float64)
        transformed, checks, pair_rows = [], [], []
        for run in runs:
            original = tuple(run[key][-1].astype(np.float64) for key in ("W0", "W1"))
            seed = GAUGE_SEED + int(run["meta"]["seed"])
            b0, b1, scales = rescale_network(*original, envelope, seed)
            transformed.append((b0, b1))
            canonical_a, canonical_b = m.canonicalize(*original), m.canonicalize(b0, b1)
            original_logits, transformed_logits = _logits(probes, original), _logits(probes, (b0, b1))
            checks.append({
                "C": envelope, "run": run["meta"]["run_id"], "seed": seed,
                "logit_error": float(np.max(np.abs(original_logits - transformed_logits))),
                "canonical_weight_error": max(float(np.max(np.abs(a-b))) for a, b in zip(canonical_a, canonical_b)),
                "predictions_equal": bool(np.array_equal(original_logits.argmax(1), transformed_logits.argmax(1))),
                "scale_geometric_mean": float(np.exp(np.log(scales).mean())),
            })
        for ia, ib in itertools.combinations(range(len(runs)), 2):
            a0, a1 = transformed[ia]
            b0, b1 = transformed[ib]
            base = {"C": envelope, "index_a": ia, "index_b": ib,
                    "run_a": runs[ia]["meta"]["run_id"], "run_b": runs[ib]["meta"]["run_id"],
                    "within": runs[ia]["meta"]["dataset"] == runs[ib]["meta"]["dataset"]}
            for method in ("raw_alpha", "canonical_alpha"):
                permutation = m.align(a0, a1, b0, b1, method=method)
                values = m.layers(a0, a1, b0, b1, permutation, metric="alpha", canonical=method == "canonical_alpha")
                for layer, value in values.items():
                    pair_rows.append({**base, "metric": method, "layer": layer, "value": value})
        return pair_rows, checks


def gauge_validation(output="outputs", workers=4):
    output = Path(output)
    jobs = [(str(output), envelope) for envelope in GAUGE_ENVELOPES]
    if workers == 1:
        results = list(map(_gauge_job, jobs))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(_gauge_job, jobs))
    pairs = pd.DataFrame([row for pair_rows, _ in results for row in pair_rows])
    checks = pd.DataFrame([row for _, checks in results for row in checks])
    runs = load_runs(output, "main")
    groups = np.array([r["meta"]["dataset"] for r in runs])
    estimates, replicates = [], []
    for (envelope, metric, layer), block in pairs.groupby(["C", "metric", "layer"], sort=False):
        index_pairs = block[["index_a", "index_b"]].to_numpy()
        summary = group_summary(index_pairs, block.value.to_numpy(), groups)
        for group, result in summary.items():
            base = {"C": envelope, "metric": metric, "layer": layer, "group": group}
            estimates.append({**base, **{key: result[key] for key in ("estimate", "se", "ci_low", "ci_high")}})
            for deleted, stratum, value in zip(result["deleted"], result["strata"], result["replicates"]):
                replicates.append({**base, "deleted_index": deleted, "deleted_run": runs[int(deleted)]["meta"]["run_id"], "stratum": stratum, "estimate": value})
    estimates = pd.DataFrame(estimates)
    sensitivity = []
    for layer in ("W0", "W1", "balanced", "size_weighted"):
        baseline = pairs[(pairs.C == 1) & (pairs.metric == "raw_alpha") & (pairs.layer == layer)].sort_values(["index_a", "index_b"]).value.to_numpy()
        baseline_contrast = float(estimates[(estimates.C == 1) & (estimates.metric == "raw_alpha") & (estimates.layer == layer) & (estimates.group == "contrast")].estimate.iloc[0])
        canonical_baseline = pairs[(pairs.C == 1) & (pairs.metric == "canonical_alpha") & (pairs.layer == layer)].sort_values(["index_a", "index_b"]).value.to_numpy()
        for envelope in GAUGE_ENVELOPES:
            current = pairs[(pairs.C == envelope) & (pairs.metric == "raw_alpha") & (pairs.layer == layer)].sort_values(["index_a", "index_b"]).value.to_numpy()
            canonical = pairs[(pairs.C == envelope) & (pairs.metric == "canonical_alpha") & (pairs.layer == layer)].sort_values(["index_a", "index_b"]).value.to_numpy()
            contrast = float(estimates[(estimates.C == envelope) & (estimates.metric == "raw_alpha") & (estimates.layer == layer) & (estimates.group == "contrast")].estimate.iloc[0])
            sensitivity.append({"C": envelope, "layer": layer, "mean_absolute_change": float(np.mean(np.abs(current-baseline))),
                                "correlation_with_original": float(pearsonr(current, baseline).statistic),
                                "dataset_contrast": contrast, "fraction_retained": contrast / baseline_contrast,
                                "canonical_max_absolute_change": float(np.max(np.abs(canonical-canonical_baseline)))})
    write_csv(output / "summary" / "gauge_pairs.csv", pairs)
    write_csv(output / "summary" / "gauge_estimates.csv", estimates)
    write_csv(output / "summary" / "gauge_jackknife_replicates.csv", pd.DataFrame(replicates))
    write_csv(output / "summary" / "gauge_sensitivity.csv", pd.DataFrame(sensitivity))
    write_csv(output / "validation" / "gauge_checks.csv", checks)
    return {"pairs_per_envelope": len(runs) * (len(runs)-1) // 2, "envelopes": list(GAUGE_ENVELOPES),
            "max_logit_error": float(checks.logit_error.max()), "all_predictions_equal": bool(checks.predictions_equal.all()),
            "max_canonical_weight_error": float(checks.canonical_weight_error.max()),
            "max_canonical_alpha_change": max(r["canonical_max_absolute_change"] for r in sensitivity),
            "scaling_seed_convention": f"RandomState({GAUGE_SEED} + training_seed) permutes linspace(-1,1,64); same exponent order at every envelope",
            "probe_convention": "first 256 rows of the saved fixed 3000-image probe; choice not fully specified in paper"}


def _objective(weights_a, weights_b, permutation, method):
    if method == "git_rebasin":
        b0, b1 = m.apply_permutation(*weights_b, permutation)
        return float(np.sum(weights_a[0] * b0) + np.sum(weights_a[1] * b1))
    return m.layers(*weights_a, *weights_b, permutation, metric="alpha", canonical=True)["size_weighted"]


def stability_validation(output="outputs", pair_count=30, repeats=10):
    output = Path(output)
    runs = load_runs(output, "main")
    available = list(itertools.combinations(range(len(runs)), 2))
    rng = np.random.RandomState(STABILITY_SEED)
    selected = [available[i] for i in sorted(rng.choice(len(available), pair_count, replace=False))]
    probes = np.load(output / "probes" / "main.npz")["X"][:256].astype(np.float64)
    rows = []
    with threadpool_limits(limits=1):
        for ia, ib in selected:
            a = tuple(runs[ia][key][-1].astype(np.float64) for key in ("W0", "W1"))
            b = tuple(runs[ib][key][-1].astype(np.float64) for key in ("W0", "W1"))
            references = {method: m.align(*a, *b, method=method) for method in ("benchmark", "git_rebasin")}
            objectives = {method: _objective(a, b, p, method) for method, p in references.items()}
            logits = _logits(probes, b)
            for repeat in range(repeats):
                renumber = rng.permutation(b[0].shape[1])
                permuted = m.apply_permutation(*b, renumber)
                for method in references:
                    local = m.align(*a, *permuted, method=method)
                    composed = renumber[local]
                    aligned = m.apply_permutation(*permuted, local)
                    rows.append({"method": method, "pair": f"{runs[ia]['meta']['run_id']}:{runs[ib]['meta']['run_id']}",
                                 "index_a": ia, "index_b": ib, "repeat": repeat,
                                 "assignment_equal": bool(np.array_equal(composed, references[method])),
                                 "objective_error": abs(_objective(a, b, composed, method)-objectives[method]),
                                 "logit_error": float(np.max(np.abs(logits-_logits(probes, aligned))))})
    frame = pd.DataFrame(rows)
    write_csv(output / "validation" / "alignment_stability.csv", frame)
    return {"pairs": pair_count, "repeats_per_pair": repeats, "methods": ["benchmark", "git_rebasin"],
            "all_assignments_equal": bool(frame.assignment_equal.all()), "max_objective_error": float(frame.objective_error.max()),
            "max_logit_error": float(frame.logit_error.max()),
            "selection_convention": f"RandomState({STABILITY_SEED}) chooses 30 pair indices without replacement from lexicographically sorted run pairs, then draws renumberings sequentially",
            "selected_pairs": [[runs[a]["meta"]["run_id"], runs[b]["meta"]["run_id"]] for a, b in selected]}


def apply_multilayer_permutations(weights, permutations):
    """Apply one shared column/row permutation at every hidden-layer boundary."""
    if len(permutations) != len(weights)-1:
        raise ValueError("Provide one permutation for every hidden layer")
    result = [np.asarray(w, dtype=np.float64).copy() for w in weights]
    for layer, p in enumerate(permutations):
        p = np.asarray(p, dtype=np.int64)
        if not np.array_equal(np.sort(p), np.arange(result[layer].shape[1])):
            raise ValueError("Invalid hidden-layer permutation")
        result[layer] = result[layer][:, p]
        result[layer+1] = result[layer+1][p, :]
    return result


def full_dissimilarity(a, b, metric):
    return m.dissimilarity(np.concatenate([w.ravel() for w in a]), np.concatenate([w.ravel() for w in b]), metric)


def iterative_match(reference, target, metric="alpha", max_passes=30, tolerance=1e-12):
    """PIF presorting followed by alternating backward/forward profile assignments.

    PIF components are invariant to neighboring neuron orders. Every refinement
    is an exact Hungarian assignment for the current two-neighbor profiles.
    The best complete-network value is retained since Alpha's local normalization
    does not guarantee monotonic improvement of its matrix-level objective.
    """
    if len(reference) != len(target) or any(a.shape != b.shape for a, b in zip(reference, target)):
        raise ValueError("Reference and target architectures must match")
    permutations = [m.pif_assignment(reference[i], reference[i+1], target[i], target[i+1]) for i in range(len(reference)-1)]
    current = apply_multilayer_permutations(target, permutations)
    best_weights, best_permutations = [w.copy() for w in current], [p.copy() for p in permutations]
    best = full_dissimilarity(reference, current, metric)
    trace = [{"pass": 0, "dissimilarity": best, "best_dissimilarity": best, "stage": "pif"}]
    stop_reason = "max_passes"
    for iteration in range(1, max_passes+1):
        previous_permutations = [p.copy() for p in permutations]
        previous_best = best
        for direction in (range(len(permutations)-1, -1, -1), range(len(permutations))):
            for layer in direction:
                cost = m.profile_cost(reference[layer], reference[layer+1], current[layer], current[layer+1], metric)
                p = m._assignment(cost)
                current[layer] = current[layer][:, p]
                current[layer+1] = current[layer+1][p, :]
                permutations[layer] = permutations[layer][p]
        value = full_dissimilarity(reference, current, metric)
        if value < best:
            best = value
            best_weights, best_permutations = [w.copy() for w in current], [p.copy() for p in permutations]
        trace.append({"pass": iteration, "dissimilarity": value, "best_dissimilarity": best, "stage": "backward_forward"})
        if all(np.array_equal(a,b) for a,b in zip(previous_permutations, permutations)):
            stop_reason = "orders_fixed"
            break
        if previous_best-best < tolerance:
            stop_reason = "improvement_below_tolerance"
            break
    return best_weights, best_permutations, {"trace": trace, "stop_reason": stop_reason, "passes": iteration}


def synthetic_validation(output="outputs"):
    output = Path(output)
    rng = np.random.RandomState(SYNTHETIC_SEED)
    architecture = [20, 12, 10, 8, 6]
    reference = [rng.normal(0, 1/np.sqrt(fan_in), (fan_in, fan_out)) for fan_in, fan_out in zip(architecture[:-1], architecture[1:])]
    imposed = [rng.permutation(width) for width in architecture[1:-1]]
    permuted = apply_multilayer_permutations(reference, imposed)
    noise = [rng.normal(0, 0.3*np.std(weight), weight.shape) for weight in permuted]
    target = [weight+perturbation for weight, perturbation in zip(permuted, noise)]
    probe = rng.normal(size=(256, architecture[0]))
    arrays = {}
    for index, (a,b,n) in enumerate(zip(reference,target,noise)):
        arrays[f"reference_W{index}"] = a
        arrays[f"target_W{index}"] = b
        arrays[f"noise_W{index}"] = n
    for index, p in enumerate(imposed): arrays[f"imposed_permutation_{index}"] = p
    results = []
    with threadpool_limits(limits=1):
        for metric in ("euclidean", "snd", "alpha"):
            aligned, inferred, details = iterative_match(reference, target, metric)
            for index, weights in enumerate(aligned): arrays[f"{metric}_aligned_W{index}"] = weights
            for index, p in enumerate(inferred): arrays[f"{metric}_inferred_permutation_{index}"] = p
            results.append({"metric": metric, "before": full_dissimilarity(reference,target,metric),
                            "after": full_dissimilarity(reference,aligned,metric),
                            "permutation_recovery_by_layer": [float(np.mean(p == np.argsort(q))) for p,q in zip(inferred,imposed)],
                            "logit_error": float(np.max(np.abs(_logits(probe,target)-_logits(probe,aligned)))), **details})
    result = {"architecture": architecture, "seed": SYNTHETIC_SEED, "noise_sd_fraction": 0.3,
              "initialization": "RandomState(2026), independent zero-mean Gaussian weights with standard deviation 1/sqrt(fan_in)",
              "unspecified_original_choices": "The supplement specifies architecture and relative noise, but not the random seed, initial distribution/scale, or exact full-network aggregation; these are explicit reconstruction choices.",
              "aggregation": "Concatenate all weight entries, then apply the selected dissimilarity once",
              "noise_sd_convention": "0.3 * population standard deviation (ddof=0) of each pre-noise matrix",
              "max_passes": 30, "tolerance": 1e-12, "results": results}
    _atomic_npz(output / "validation" / "synthetic_weights.npz", **arrays)
    write_json(output / "validation" / "synthetic.json", result)
    return result


def _direct_alpha(a, b):
    terms = [(float(x)*float(y))**2 for x,y in zip(np.ravel(a),np.ravel(b))]
    denominator = sum(terms)
    if denominator <= 1e-12: return 0.0
    return sum(term for term,x,y in zip(terms,np.ravel(a),np.ravel(b)) if np.sign(x) != np.sign(y)) / denominator


def _direct_cka(a, b):
    # Independent sample-Gram implementation, rather than feature covariance.
    a, b = a-a.mean(axis=0), b-b.mean(axis=0)
    ka, kb = a @ a.T, b @ b.T
    return float(np.sum(ka*kb) / np.sqrt(np.sum(ka*ka)*np.sum(kb*kb)))


def integrity_and_direct_checks(output="outputs"):
    output = Path(output)
    archive_checks, paired_checks, direct_checks = [], [], []
    for cohort in ("main", "task", "early"):
        runs = load_runs(output, cohort)
        manifest = json.loads((output / "manifests" / f"{cohort}.json").read_text())
        records = {r["run_id"]: r for r in manifest["runs"]}
        for run in runs:
            record = records[run["meta"]["run_id"]]
            archive_hash = file_hash(run["path"])
            state_path = run["path"].with_suffix(".pt")
            state_hash = file_hash(state_path)
            archive_checks.append({"cohort": cohort, "run": run["meta"]["run_id"],
                                   "npz_hash_equal": archive_hash == record["sha256"],
                                   "pt_hash_equal": state_hash == record["full_state_sha256"] == run["meta"]["full_state_sha256"],
                                   "all_finite": all(np.isfinite(run[key]).all() for key in ("W0","W1","accuracy","loss")),
                                   "zero_biases": not (np.any(run["bias0"]) or np.any(run["bias1"])),
                                   "checkpoints": len(run["checkpoints"]),
                                   "schedule_matches_all_runs": bool(np.array_equal(run["checkpoints"], runs[0]["checkpoints"]) and np.array_equal(run["updates"],runs[0]["updates"]))})
        if cohort == "task":
            for seed in sorted({r["meta"]["seed"] for r in runs}):
                pair = [r for r in runs if r["meta"]["seed"] == seed]
                paired_checks.append({"seed": seed, "initial_W0_equal": bool(np.array_equal(pair[0]["W0"][0], pair[1]["W0"][0])),
                                      "same_epoch_shuffle_seeds": pair[0]["meta"]["epoch_shuffle_seeds"] == pair[1]["meta"]["epoch_shuffle_seeds"],
                                      "same_shuffle_convention": pair[0]["meta"]["conventions"]["shuffle"] == pair[1]["meta"]["conventions"]["shuffle"]})
    runs = load_runs(output,"main")
    probe = np.load(output / "probes" / "main.npz")["X"].astype(np.float64)
    # Predetermined representative pairs cover within and between datasets.
    choices = [(0,1), (0,10), (10,20)]
    with threadpool_limits(limits=1):
        for ia,ib in choices:
            a = tuple(runs[ia][key][-1].astype(np.float64) for key in ("W0","W1"))
            b = tuple(runs[ib][key][-1].astype(np.float64) for key in ("W0","W1"))
            p = m.align(*a,*b,method="canonical_alpha")
            ca = m.canonicalize(*a)
            cb = m.apply_permutation(*m.canonicalize(*b),p)
            alpha_error = max(abs(m.dissimilarity(x,y,"alpha")-_direct_alpha(x,y)) for x,y in zip(ca,cb))
            ha,hb = np.maximum(probe @ a[0],0), np.maximum(probe @ b[0],0)
            cka_error = abs(m.linear_cka(ha,hb)-_direct_cka(ha,hb))
            bp = m.apply_permutation(*b,p)
            direct_checks.append({"pair": f"{runs[ia]['meta']['run_id']}:{runs[ib]['meta']['run_id']}",
                                  "alpha_recompute_error": alpha_error, "cka_recompute_error": cka_error,
                                  "permutation_logit_error": float(np.max(np.abs(_logits(probe[:256],b)-_logits(probe[:256],bp)))),
                                  "self_alpha": m.dissimilarity(ca[0],ca[0],"alpha"),
                                  "symmetry_alpha_error": abs(m.dissimilarity(ca[0],cb[0],"alpha")-m.dissimilarity(cb[0],ca[0],"alpha"))})
    passed = all(all(r[key] for key in ("npz_hash_equal","pt_hash_equal","all_finite","zero_biases","schedule_matches_all_runs")) for r in archive_checks)
    passed &= all(all(r[key] for key in ("initial_W0_equal","same_epoch_shuffle_seeds","same_shuffle_convention")) for r in paired_checks)
    passed &= all(max(r[key] for key in ("alpha_recompute_error","cka_recompute_error","permutation_logit_error","self_alpha","symmetry_alpha_error")) < 1e-10 for r in direct_checks)
    return {"passed": bool(passed), "archive_checks": archive_checks, "paired_initialization_checks": paired_checks, "independent_direct_checks": direct_checks}


def run_validation(output="outputs", workers=4):
    output = Path(output)
    print("Validation: checking archive integrity and independent formulas", flush=True)
    integrity = integrity_and_direct_checks(output)
    print("Validation: six ReLU rescaling envelopes, 435 pairs each", flush=True)
    gauge = gauge_validation(output, workers)
    print("Validation: 30 pair renumberings x 10 repeats x 2 methods", flush=True)
    stability = stability_validation(output)
    print("Validation: synthetic multilayer implementation check", flush=True)
    synthetic = synthetic_validation(output)
    passed = (integrity["passed"] and gauge["all_predictions_equal"] and gauge["max_logit_error"] < 1e-10
              and gauge["max_canonical_weight_error"] < 1e-10 and gauge["max_canonical_alpha_change"] < 1e-10
              and stability["all_assignments_equal"] and stability["max_objective_error"] < 1e-10 and stability["max_logit_error"] < 1e-10
              and all(r["after"] <= r["before"] and r["logit_error"] < 1e-10 for r in synthetic["results"]))
    result = {"passed": bool(passed), "integrity": integrity, "gauge": gauge, "alignment_stability": stability,
              "synthetic": synthetic, "validation_source_sha256": file_hash(__file__), "metrics_source_sha256": file_hash(m.__file__)}
    write_json(output / "validation" / "checks.json", result)
    print(f"Validation complete: passed={passed}", flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = run_validation(args.output,args.workers)
    if not result["passed"]: raise SystemExit(1)


if __name__ == "__main__":
    main()
