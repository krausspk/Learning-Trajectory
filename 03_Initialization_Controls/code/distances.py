"""Label-blind initialization-retrieval controls, isolated from experiments 01/02.

All cost functions receive weights and query-only masks, never run identities.
Candidate-pair residual quantiles are metric sensitivities, not interventions on
true-parent changes. Historical raw changes are written separately without tests.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
LAYERS = ("W0", "W1", "total")
PRIMARY_KEY = "normalized_frobenius|total|optimal|canonical"


def retrieval_source(root=ROOT):
    path = Path(root) / "02_Initialization_Retrieval/code/distances.py"
    spec = importlib.util.spec_from_file_location("_controls_retrieval_source", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


RETRIEVAL = retrieval_source()
SOURCE = RETRIEVAL.SOURCE
sha256 = RETRIEVAL.sha256


def network_norms(network):
    a, b = network
    return np.asarray([np.linalg.norm(a), np.linalg.norm(b),
                       np.sqrt(np.sum(a*a) + np.sum(b*b))], dtype=np.float64)


def unit_network(network):
    norm = network_norms(network)[2]
    if not np.isfinite(norm) or norm <= 0:
        raise ValueError("Cannot unit-normalize a nonfinite or zero full network")
    return tuple(np.asarray(w, dtype=np.float64) / norm for w in network)


def _aligned(candidate, permutation):
    return candidate[0][:, permutation], candidate[1][permutation, :]


def layer_distances(query, candidate, permutation, mask=None):
    """Frobenius contributions under one shared joint-layer neuron assignment."""
    b0, b1 = _aligned(candidate, permutation)
    e0, e1 = (query[0] - b0)**2, (query[1] - b1)**2
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != (query[0].shape[1],) or not mask.any():
            raise ValueError("A query mask must select at least one hidden neuron")
        e0, e1 = e0[:, mask], e1[mask, :]
    s0, s1 = float(e0.sum()), float(e1.sum())
    return np.sqrt([s0, s1, s0+s1])


def activity_masks(activations):
    """Query-only gauge-invariant activity quartiles, retaining all boundary ties."""
    h = np.asarray(activations)
    if h.ndim != 2 or not np.isfinite(h).all() or np.any(h < 0):
        raise ValueError("Expected finite nonnegative ReLU activations [probe, hidden]")
    rate = np.mean(h > 0, axis=0)
    low_threshold, high_threshold = np.quantile(rate, [.25, .75])
    masks = {"low": rate <= low_threshold, "high": rate >= high_threshold,
             "inactive": rate == 0}
    return masks, rate, np.asarray([low_threshold, high_threshold])


def masked_permutation(query, candidate, query_mask):
    """Optimize masked squared distance, including candidate norm terms.

    A dot-product-only objective would be incorrect: selecting only query rows
    makes the sum of selected candidate-neuron norms assignment-dependent.
    Unselected query rows have zero cost and complete the bijection neutrally.
    """
    a0, a1 = query
    b0, b1 = candidate
    mask = np.asarray(query_mask, dtype=bool)
    if mask.shape != (a0.shape[1],) or not mask.any():
        raise ValueError("Masked assignment requires a nonempty query-only mask")
    qa = np.sum(a0*a0, axis=0) + np.sum(a1*a1, axis=1)
    cb = np.sum(b0*b0, axis=0) + np.sum(b1*b1, axis=1)
    cost = (qa[:, None] + cb[None, :] - 2*(a0.T @ b0 + a1 @ b1.T)) * mask[:, None]
    rows, columns = linear_sum_assignment(cost)
    permutation = np.empty(len(mask), dtype=np.int64)
    permutation[rows] = columns
    return permutation


def residual_quantile_distances(query, candidate, permutation):
    """Pair-dependent residual RMS, selected separately within each weight layer.

    No parent identity enters selection. For each candidate pair, small/large
    residual subsets are newly selected after the same joint optimal matching.
    Inclusive thresholds avoid neuron-index-dependent tie breaking. RMS accounts
    for varying subset cardinalities; total pools selected entries from W0/W1.
    """
    aligned = _aligned(candidate, permutation)
    thresholds = np.empty((2, 2))
    counts = np.empty((2, 2), dtype=np.int64)
    sums = np.empty((2, 2))
    for layer, (a, b) in enumerate(zip(query, aligned)):
        residual = np.abs(a-b).ravel()
        low, high = np.quantile(residual, [.25, .75])
        thresholds[layer] = [low, high]
        for subset, mask in enumerate((residual <= low, residual >= high)):
            counts[layer, subset] = int(mask.sum())
            sums[layer, subset] = float(np.sum(residual[mask]**2))
    values = np.empty((2, 3))
    for subset in range(2):
        values[subset] = np.sqrt([sums[0, subset]/counts[0, subset],
                                 sums[1, subset]/counts[1, subset],
                                 sums[:, subset].sum()/counts[:, subset].sum()])
    return values, thresholds, counts


def definitions():
    result = {}
    common = {"orientation": "current query i at E=t versus candidate j at E=0; all 100 entries use identical rules",
              "run_identity_input": False, "interpretive_role": "exploratory_control",
              "layer_order": list(LAYERS), "shared_matching": "W0 and W1 always share the same joint-network permutation; no layer-specific rematching"}
    for gauge in ("raw", "canonical"):
        for alignment in ("identity", "optimal"):
            for metric in ("frobenius", "normalized_frobenius"):
                for layer in LAYERS:
                    key = "|".join((metric, layer, alignment, gauge))
                    result[key] = {**common, "metric": metric, "layer": layer, "alignment": alignment, "gauge": gauge,
                                   "formula": "Frobenius norm of query minus aligned candidate, restricted to the named layer (total concatenates W0/W1)",
                                   "normalization": "one full-vector norm divides BOTH W0 and W1; layer distances are contributions to that globally normalized vector" if metric.startswith("normalized") else "no global normalization",
                                   "matching": "native saved neuron coordinates" if alignment=="identity" else "global Hungarian maximizes joint W0/W1 inner product in the stated gauge; global normalization leaves the optimizer unchanged",
                                   "coordinate_caution": "Identity is a neuron-coordinate memory diagnostic, not permutation-invariant" if alignment=="identity" else "Permutation-adjusted comparison"}
        for layer in LAYERS:
            key = f"norm_only|{layer}|none|{gauge}"
            result[key] = {**common, "metric": "norm_only", "layer": layer, "alignment": "none", "gauge": gauge,
                           "formula": "abs(norm(query layer at E=t) - norm(candidate layer at E=0))",
                           "normalization": "pre-global-normalization norm; total=sqrt(norm(W0)^2+norm(W1)^2)"}
    for subset in ("small", "large"):
        for layer in LAYERS:
            metric = f"{subset}_residual_rms"
            key = f"{metric}|{layer}|optimal|canonical"
            result[key] = {**common, "metric": metric, "layer": layer, "alignment": "optimal", "gauge": "canonical",
                           "normalization": "canonical neuron gauge followed by one global unit norm",
                           "selection": f"own candidate-pair absolute residuals after joint matching, {'at or below the 25th' if subset=='small' else 'at or above the 75th'} percentile separately in W0 and W1, all boundary ties included",
                           "formula": "sqrt(mean(selected squared residuals)); total pools selected entries from both layers",
                           "caution": "Candidate-pair-dependent metric sensitivity, NOT true historical change, causal attribution, an ablation, or independent evidence"}
    for subset in ("low", "high"):
        for layer in LAYERS:
            metric = f"activity_{subset}_normalized_frobenius"
            key = f"{metric}|{layer}|masked_optimal|canonical"
            result[key] = {**common, "metric": metric, "layer": layer, "alignment": "masked_optimal", "gauge": "canonical",
                           "normalization": "canonical neuron gauge followed by one global unit norm",
                           "selection": f"query-only activation occupancy Pr(H>0) on the fixed original probe, {'<=25th' if subset=='low' else '>=75th'} query-neuron percentile, all boundary ties included; recomputed per query/checkpoint",
                           "matching": "Hungarian on full squared joint-neuron distances, weighted by the query mask; includes candidate norm terms; unselected rows have zero cost",
                           "formula": "Frobenius norm on selected query neurons after the mask-specific shared assignment",
                           "caution": "Query-activity sensitivity, not a functional fingerprint: no query/candidate activation comparison; inactive means zero on this probe only"}
    assert len(result) == 42 and PRIMARY_KEY in result
    return result


def pair_controls(query, candidate, query_masks):
    """All 42 candidate costs, with no IDs, diagonal logic, or parent-based masks."""
    values, assignments = {}, {}
    for gauge in ("raw", "canonical"):
        q = query if gauge=="raw" else SOURCE.canonical_gauge(*query)[:2]
        c = candidate if gauge=="raw" else SOURCE.canonical_gauge(*candidate)[:2]
        qu, cu = unit_network(q), unit_network(c)
        _, permutation = RETRIEVAL.match_unit_networks(qu, cu)
        assignments[gauge] = permutation
        for alignment, p in (("identity", np.arange(q[0].shape[1])), ("optimal", permutation)):
            for metric, a, b in (("frobenius", q, c), ("normalized_frobenius", qu, cu)):
                for layer, value in zip(LAYERS, layer_distances(a, b, p)):
                    values[f"{metric}|{layer}|{alignment}|{gauge}"] = float(value)
        for layer, value in zip(LAYERS, np.abs(network_norms(q)-network_norms(c))):
            values[f"norm_only|{layer}|none|{gauge}"] = float(value)
        if gauge=="canonical":
            residual_values, thresholds, counts = residual_quantile_distances(qu, cu, permutation)
            for subset, distances in zip(("small", "large"), residual_values):
                for layer, value in zip(LAYERS, distances):
                    values[f"{subset}_residual_rms|{layer}|optimal|canonical"] = float(value)
            for subset in ("low", "high"):
                p = masked_permutation(qu, cu, query_masks[subset])
                assignments[f"activity_{subset}"] = p
                for layer, value in zip(LAYERS, layer_distances(qu, cu, p, query_masks[subset])):
                    values[f"activity_{subset}_normalized_frobenius|{layer}|masked_optimal|canonical"] = float(value)
    return values, assignments, thresholds, counts


def _load_query_masks(root, groups):
    """Verify original activation archives before deriving query-only masks."""
    all_ids = {run["run_id"] for group in groups.values() for run in group}
    output, receipts = {}, []
    for dataset, runs in groups.items():
        output[dataset] = {name: np.empty((21, len(runs), 64), dtype=bool) for name in ("low", "high", "inactive")}
        output[dataset]["activity_rate"] = np.empty((21, len(runs), 64))
        output[dataset]["thresholds"] = np.empty((21, len(runs), 2))
    for cp in range(21):
        receipt_path = root / f"outputs/pairs/main/checkpoint_{cp:04d}.json"
        receipt = json.loads(receipt_path.read_text())
        path = Path(receipt["activation_path"])
        if not path.is_absolute(): path = root/path
        digest = sha256(path)
        if digest != receipt["activation_sha256"]:
            raise ValueError(f"Hash mismatch for activation archive {path}")
        with np.load(path, allow_pickle=False) as archive:
            ids = archive["run_ids"].astype(str).tolist()
            if len(ids)!=30 or len(set(ids))!=30 or set(ids)!=all_ids:
                raise ValueError(f"Activation run identities do not match main manifest: {path}")
            h = archive["H"]
            if h.shape != (30, 3000, 64):
                raise ValueError(f"Unexpected activation dimensions: {path}, {h.shape}")
            for dataset, runs in groups.items():
                for i, run in enumerate(runs):
                    masks, rate, thresholds = activity_masks(h[ids.index(run["run_id"])])
                    for name in ("low", "high", "inactive"):
                        output[dataset][name][cp, i] = masks[name]
                    output[dataset]["activity_rate"][cp, i] = rate
                    output[dataset]["thresholds"][cp, i] = thresholds
        receipts.append({"checkpoint": cp, "path": str(path), "sha256": digest,
                         "original_receipt_path": str(receipt_path), "original_receipt_sha256": sha256(receipt_path)})
    return output, receipts


def _historical_change_rows(dataset, runs):
    """Descriptive same-run change summaries; never used to construct costs."""
    rows = []
    percentiles = [0, 25, 50, 75, 90, 95, 99, 100]
    for run in runs:
        for cp in range(21):
            for layer in ("W0", "W1"):
                initial, final = run[layer][0], run[layer][cp]
                delta = np.abs(final-initial).ravel()
                rows.append({"dataset": dataset, "run_id": run["run_id"], "checkpoint": cp, "layer": layer,
                             "gauge": "raw", "alignment": "identity", "n_parameters": len(delta),
                             **{f"abs_change_p{q}": float(v) for q,v in zip(percentiles, np.percentile(delta,percentiles))},
                             "fraction_exactly_unchanged": float(np.mean(delta==0)),
                             "fraction_abs_change_le_1e_minus6": float(np.mean(delta<=1e-6)),
                             "relative_frobenius_change": float(np.linalg.norm(delta)/np.linalg.norm(initial)),
                             "inferential_role": "descriptive_true_parent_history_only", "used_for_retrieval": False})
    return rows


def build_distances(root: Path, out: Path):
    root, out = Path(root).resolve(), Path(out).resolve()
    if out != root/"03_Initialization_Controls":
        raise ValueError("Control outputs must be isolated in 03_Initialization_Controls")
    point1 = root/"02_Initialization_Retrieval"
    if not (point1/"artifact_manifest.json").is_file():
        raise ValueError("Complete and finalize Point 1 before running Point 2 controls")
    validation1 = json.loads((point1/"data/validation.json").read_text())
    if not validation1.get("passed") or validation1.get("code_sha256") != sha256(point1/"code/distances.py"):
        raise ValueError("Point 1 distances must be validated against their current source")
    source = RETRIEVAL.source_module(root)
    groups, manifest, run_checks = source.load_manifest_runs(root)
    _, pipeline_checks = source.load_pipeline_artifacts(root, manifest)
    masks, activation_receipts = _load_query_masks(root, groups)
    probe_path = root/"outputs/probes/main.npz"
    probe_hash = sha256(probe_path)
    data_dir = out/"data"
    data_dir.mkdir(parents=True, exist_ok=True)
    specification = definitions()
    keys = list(specification)
    key_index = {key:i for i,key in enumerate(keys)}
    source.save_json(data_dir/"distance_definitions.json", specification)
    source.save_json(data_dir/"source_manifest.json", manifest)
    source.save_json(data_dir/"activation_sources.json", {"probe_path":str(probe_path),"probe_sha256":probe_hash,"archives":activation_receipts,
                     "mask_definition":"Query-only fraction of original probe activations >0; <=25th / >=75th query-neuron quantile including all boundary ties; inactive iff occupancy=0"})
    records, mask_rows, change_rows = [], [], []
    with threadpool_limits(limits=1):
        for dataset, runs in groups.items():
            values = np.empty((len(keys), 21, 10, 10), dtype=np.float64)
            permutations = {name:np.empty((21,10,10,64),dtype=np.int16) for name in ("raw","canonical","activity_low","activity_high")}
            residual_thresholds = np.empty((21,10,10,2,2))
            residual_counts = np.empty((21,10,10,2,2),dtype=np.int32)
            norms = {gauge:np.empty((21,10,3)) for gauge in ("raw","canonical")}
            candidates = [(run["W0"][0],run["W1"][0]) for run in runs]
            for cp in range(21):
                for i, run in enumerate(runs):
                    query = run["W0"][cp], run["W1"][cp]
                    norms["raw"][cp,i] = network_norms(query)
                    norms["canonical"][cp,i] = network_norms(source.canonical_gauge(*query)[:2])
                    query_masks = {name:masks[dataset][name][cp,i] for name in ("low","high")}
                    for name in ("low","high","inactive"):
                        mask_rows.append({"dataset":dataset,"run_id":run["run_id"],"checkpoint":cp,"subset":name,
                                          "selected_neurons":int(masks[dataset][name][cp,i].sum()),"total_neurons":64,
                                          "activation_sha256":activation_receipts[cp]["sha256"],"probe_sha256":probe_hash})
                    for j, candidate in enumerate(candidates):
                        costs, assignments, thresholds, counts = pair_controls(query,candidate,query_masks)
                        for key, cost in costs.items(): values[key_index[key],cp,i,j] = cost
                        for name,p in assignments.items(): permutations[name][cp,i,j] = p
                        residual_thresholds[cp,i,j], residual_counts[cp,i,j] = thresholds,counts
                print(f"Initialization controls: {dataset} E={cp}, 100 label-blind candidates x {len(keys)} protocols",flush=True)
            if not np.isfinite(values).all() or np.any(values<0):
                raise AssertionError(f"Nonfinite or negative control distances: {dataset}")
            reference_path = point1/f"data/{dataset}_retrieval.npz"
            expected_record = next(row for row in validation1["outputs"] if row["dataset"]==dataset)
            if sha256(reference_path) != expected_record["sha256"]:
                raise ValueError(f"Point 1 reference hash mismatch: {reference_path}")
            run_ids = np.asarray([run["run_id"] for run in runs])
            with np.load(reference_path,allow_pickle=False) as reference:
                reference_index = reference["metric_keys"].astype(str).tolist().index(PRIMARY_KEY)
                if not np.array_equal(run_ids,reference["run_ids"]): raise ValueError("Reference run ordering mismatch")
                error = float(np.max(np.abs(values[key_index[PRIMARY_KEY]]-reference["distances"][reference_index])))
            if error>1e-12: raise AssertionError(f"Point 1 reference mismatch: {dataset}, maxabs={error}")
            path = data_dir/f"{dataset}_retrieval.npz"
            np.savez_compressed(path,distances=values,metric_keys=np.asarray(keys),checkpoints=np.arange(21),
                                run_ids=run_ids,query_run_ids=run_ids,candidate_run_ids=run_ids,seeds=np.asarray([r["seed"] for r in runs]),
                                permutations=permutations["canonical"],raw_permutations=permutations["raw"],
                                activity_low_permutations=permutations["activity_low"],activity_high_permutations=permutations["activity_high"],
                                norms_raw=norms["raw"],norms_canonical=norms["canonical"],norm_layers=np.asarray(LAYERS),
                                residual_thresholds=residual_thresholds,residual_counts=residual_counts,
                                residual_subset_names=np.asarray(["small","large"]),residual_layer_names=np.asarray(["W0","W1"]),
                                candidate_checkpoint=np.asarray(0),source_manifest_sha256=np.asarray(sha256(root/"outputs/manifests/main.json")))
            mask_path = data_dir/f"{dataset}_query_masks.npz"
            np.savez_compressed(mask_path,**masks[dataset],run_ids=run_ids,checkpoints=np.arange(21),
                                **{f"{name}_counts":masks[dataset][name].sum(axis=-1) for name in ("low","high","inactive")},
                                activation_sha256=np.asarray([r["sha256"] for r in activation_receipts]),probe_sha256=np.asarray(probe_hash))
            records.append({"dataset":dataset,"path":str(path),"sha256":sha256(path),"shape":list(values.shape),
                            "query_run_ids":run_ids.tolist(),"candidate_run_ids":run_ids.tolist(),
                            "point1_reference":str(reference_path),"point1_reference_sha256":sha256(reference_path),
                            "max_primary_difference_from_point1":error,"query_masks_path":str(mask_path),"query_masks_sha256":sha256(mask_path)})
            change_rows.extend(_historical_change_rows(dataset,runs))
    pd.DataFrame(mask_rows).to_csv(data_dir/"query_mask_counts.csv",index=False)
    pd.DataFrame(change_rows).to_csv(data_dir/"historical_weight_changes_descriptive.csv",index=False)
    validation = {"passed":True,"primary_key":PRIMARY_KEY,"code_sha256":sha256(__file__),"outputs":records,
                  "n_metric_keys":len(keys),"n_distances":len(keys)*3*21*100,"all_entries_computed_identically":True,
                  "diagonal_overridden":False,"candidate_checkpoint":0,"runs":run_checks,"original_pipeline_sources":pipeline_checks,
                  "point1_validation_sha256":sha256(point1/"data/validation.json"),
                  "source_code_sha256":{str(root/"01_RSA/code/distances.py"):sha256(root/"01_RSA/code/distances.py"),str(point1/"code/distances.py"):sha256(point1/"code/distances.py")},
                  "activation_sources":"activation_sources.json","masks_use_candidate_or_parent_identity":False,
                  "residual_control_interpretation":"candidate-pair-dependent residual sensitivity; not true training change, causal information attribution or independent evidence",
                  "historical_change_table_used_for_retrieval":False,"inactive_definition":"zero activation on the fixed probe, not globally dead",
                  "functional_fingerprints_computed":False,"training_performed":False,"previous_experiments_modified":False}
    source.save_json(data_dir/"validation.json",validation)
    return validation


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=ROOT/"03_Initialization_Controls")
    args=parser.parse_args()
    result=build_distances(args.root,args.out)
    print(json.dumps({"passed":result["passed"],"n_distances":result["n_distances"]}))
