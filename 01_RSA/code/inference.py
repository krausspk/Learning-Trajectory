"""Run-level RSA inference for Experiment 01, isolated from the paper pipeline.

Entry point: run_inference(out: pathlib.Path), where out is the 01_RSA root.
All calculations use float64; Monte-Carlo null arrays are stored as float32.
Network identity permutations simultaneously permute RDM rows and columns.
The 45 edges are never treated as 45 independent biological/training units.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

BASE_SEED = 2026111600
N_PERMUTATIONS = 10000
DATASETS = ("mnist", "fashion_mnist", "kmnist")
PRIMARY_KEY = "frobenius|total|optimal|canonical"
Z95 = 1.95996
FISHER_EPS = 1e-12


def normalized_ranks(values):
    ranks = rankdata(np.asarray(values, dtype=np.float64), method="average", axis=-1)
    ranks -= ranks.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ranks, axis=-1, keepdims=True)
    return np.divide(ranks, norms, out=np.full_like(ranks, np.nan), where=norms > 0)


def spearman(x, y):
    """Spearman with average ranks; constant vectors return NaN."""
    result = np.sum(normalized_ranks(x) * normalized_ranks(y), axis=-1)
    return np.clip(result, -1.0, 1.0)


def identity_edge_indices(permutations, pairs, n_runs):
    """Map upper-triangle edges after simultaneous row/column permutations."""
    permutations = np.asarray(permutations, dtype=np.int64)
    pairs = np.asarray(pairs, dtype=np.int64)
    if permutations.ndim != 2 or permutations.shape[1] != n_runs:
        raise ValueError("Expected a batch of full run-identity permutations")
    if not np.all(np.sort(permutations, axis=1) == np.arange(n_runs)):
        raise ValueError("Each identity mapping must be a permutation")
    lookup = np.full((n_runs, n_runs), -1, dtype=np.int64)
    lookup[pairs[:, 0], pairs[:, 1]] = np.arange(len(pairs))
    lookup[pairs[:, 1], pairs[:, 0]] = np.arange(len(pairs))
    mapped = lookup[permutations[:, pairs[:, 0]], permutations[:, pairs[:, 1]]]
    if np.any(mapped < 0):
        raise ValueError("Pairs do not span all unordered off-diagonal edges")
    return mapped


def permutation_pvalues(observed, null):
    """Add-one Monte-Carlo p-values; tiny tolerance absorbs floating roundoff."""
    null = np.asarray(null, dtype=np.float64)
    if not np.isfinite(observed) or not np.isfinite(null).all():
        return np.nan, np.nan
    b = len(null)
    greater = (1 + np.count_nonzero(null >= observed - 1e-12)) / (b + 1)
    two_sided = (1 + np.count_nonzero(np.abs(null) >= abs(observed) - 1e-12)) / (b + 1)
    return float(greater), float(two_sided)


def jackknife_interval(observed, replicates):
    """Delete-one-network variance; raw normal CI is deliberately not clipped."""
    replicates = np.asarray(replicates, dtype=np.float64)
    n = replicates.shape[-1]
    se = np.sqrt((n - 1) / n * np.sum((replicates - replicates.mean(axis=-1, keepdims=True)) ** 2, axis=-1))
    return se, observed - Z95 * se, observed + Z95 * se


def jackknife_rsa(initial, final, pairs, n_runs):
    """Delete all edges touching one network, rerank, and correlate."""
    return np.asarray([spearman(np.asarray(initial)[np.all(pairs != run, axis=1)],
                                np.asarray(final)[np.all(pairs != run, axis=1)]) for run in range(n_runs)])


def null_diagnostics(null):
    null = np.asarray(null, dtype=np.float64)
    mean = float(null.mean())
    sd = float(null.std(ddof=1))
    mcse = sd / np.sqrt(len(null))
    return {"null_mean": mean, "null_sd": sd, "null_mean_mcse": mcse,
            "null_mean_z": mean / mcse if mcse > 0 else np.nan,
            "null_mean_within_6mcse": bool(abs(mean) <= 6 * mcse) if mcse > 0 else bool(mean == 0)}


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=np.float64)
    order = np.argsort(pvalues)
    adjusted = np.empty_like(pvalues)
    adjusted[order] = np.minimum(1, np.maximum.accumulate((len(pvalues) - np.arange(len(pvalues))) * pvalues[order]))
    return adjusted


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


def _write_json(path, obj):
    def convert(value):
        if isinstance(value, np.generic): return value.item()
        if isinstance(value, np.ndarray): return value.tolist()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, default=convert, allow_nan=False) + "\n")


def _metric_fields(key):
    metric, layer, alignment, gauge = key.split("|")
    return {"metric": metric, "layer": layer, "alignment": alignment, "gauge": gauge}


def _concentration(dataset, key, checkpoint, values):
    ordered = np.sort(np.asarray(values, dtype=np.float64))
    gaps = np.diff(ordered)
    positive = gaps[gaps > 0]
    mean, sd = float(ordered.mean()), float(ordered.std(ddof=1))
    unique, counts = np.unique(ordered, return_counts=True)
    tied_pairs = int(counts[counts > 1].sum())
    return {"dataset": dataset, "checkpoint": int(checkpoint), **_metric_fields(key),
            "mean": mean, "std": sd, "cv": sd / abs(mean) if mean != 0 else np.nan,
            "min": float(ordered[0]), "max": float(ordered[-1]),
            "rank_gap_min": float(gaps.min()), "rank_gap_min_positive": float(positive.min()) if len(positive) else np.nan,
            "rank_gap_median": float(np.median(gaps)), "rank_gap_median_positive": float(np.median(positive)) if len(positive) else np.nan,
            "n_unique": len(unique), "tied_pairs": tied_pairs, "tie_fraction": tied_pairs / len(ordered),
            "n_runs": 10, "n_pairs": len(ordered), "ties_definition": "exact floating-point equality; all observations in a tied group counted"}


def _partial_setup(x, covariates):
    """Rank each variable, then remove intercept and both ranked norm predictors."""
    x = rankdata(x, method="average")
    covariates = np.asarray(covariates, dtype=np.float64)
    ranked_cov = rankdata(covariates, method="average", axis=0)
    design = np.column_stack((np.ones(len(x)), ranked_cov))
    u, s, _ = np.linalg.svd(design, full_matrices=False)
    rank = np.count_nonzero(s > s[0] * max(design.shape) * np.finfo(float).eps)
    q = u[:, :rank]
    residual = x - q @ (q.T @ x)
    norm = np.linalg.norm(residual)
    return residual / norm if norm > 1e-12 else np.full_like(residual, np.nan), q, rank


def partial_spearman(x, y, covariates):
    x_unit, q, _ = _partial_setup(x, covariates)
    yr = rankdata(y, method="average")
    residual = yr - q @ (q.T @ yr)
    norm = np.linalg.norm(residual)
    return float(np.clip(np.dot(x_unit, residual) / norm, -1, 1)) if norm > 1e-12 else np.nan


def _partial_null(x, y, covariates, edge_indices):
    x_unit, q, rank = _partial_setup(x, covariates)
    yr = rankdata(y, method="average")
    permuted = yr[edge_indices]
    projection = permuted @ q
    residual_norm = np.sqrt(np.maximum(np.sum(permuted ** 2, axis=1) - np.sum(projection ** 2, axis=1), 0))
    numerator = permuted @ x_unit
    null = np.divide(numerator, residual_norm, out=np.full(len(permuted), np.nan), where=residual_norm > 1e-12)
    return np.clip(null, -1, 1), rank


def _control_row(dataset, gauge, analysis, checkpoint, predictor, rho, replicates, null, n_runs=10, layer="total"):
    se, lo, hi = jackknife_interval(rho, replicates)
    pg, pt = permutation_pvalues(rho, null)
    return {"dataset": dataset, "checkpoint": checkpoint, "metric": "frobenius", "layer": layer,
            "alignment": "optimal", "gauge": gauge, "analysis": analysis, "predictor": predictor,
            "rho": rho, "p_perm_greater": pg, "p_perm_two_sided": pt, "jackknife_se": float(se),
            "ci_low": float(lo), "ci_high": float(hi), "jackknife_min": float(np.min(replicates)),
            "jackknife_max": float(np.max(replicates)), "n_runs": n_runs, "n_pairs": n_runs * (n_runs - 1) // 2,
            "n_permutations": len(null), "interpretive_role": "conditional_descriptive",
            "exchangeability_caveat": "Covariates and initial distances are fixed while final RDM identities are permuted; no claim of an exact conditional test under arbitrary covariate dependence.",
            **null_diagnostics(null)}


def _norm_controls(dataset, archive, vectors, metric_keys, pairs, label_permutations, edge_indices):
    rows, nulls, replicates, control_keys, norm_run_rows = [], [], [], [], []
    final_index = list(archive["checkpoints"].astype(int)).index(20)
    n = len(archive["run_ids"])
    for gauge in ("raw", "canonical"):
        for layer in ("W0", "W1", "total"):
            layer_index = list(archive["norm_layers"].astype(str)).index(layer)
            norm = archive["norms_" + gauge][:, :, layer_index].astype(float)
            initial_norm = norm[0]
            predictors = np.column_stack((initial_norm[pairs[:, 0]] + initial_norm[pairs[:, 1]],
                                          np.abs(initial_norm[pairs[:, 0]] - initial_norm[pairs[:, 1]])))
            metric_index = metric_keys.index(f"frobenius|{layer}|optimal|{gauge}")
            x, final = vectors[metric_index, 0], vectors[metric_index, final_index]
            for cp, cp_index in ((0, 0), (20, final_index)):
                y = vectors[metric_index, cp_index]
                for column, predictor in enumerate(("norm_sum", "norm_abs_difference")):
                    cov = predictors[:, column]
                    rho = float(spearman(cov, y))
                    jack = jackknife_rsa(cov, y, pairs, n)
                    null = normalized_ranks(y)[edge_indices] @ normalized_ranks(cov)
                    rows.append(_control_row(dataset, gauge, "norm_predicts_distance", cp, predictor, rho, jack, null, layer=layer))
                    control_keys.append(f"{gauge}|{layer}|norm_predicts_distance|{cp}|{predictor}")
                    nulls.append(null.astype(np.float32)); replicates.append(jack)
            rho = partial_spearman(x, final, predictors)
            jack = []
            for run in range(n):
                keep = np.all(pairs != run, axis=1)
                jack.append(partial_spearman(x[keep], final[keep], predictors[keep]))
            jack = np.asarray(jack)
            null, design_rank = _partial_null(x, final, predictors, edge_indices)
            row = _control_row(dataset, gauge, "partial_rsa", 20, "norm_sum+norm_abs_difference", rho, jack, null, layer=layer)
            row["design_rank"] = int(design_rank)
            rows.append(row)
            control_keys.append(f"{gauge}|{layer}|partial_rsa|20|norm_sum+norm_abs_difference")
            nulls.append(null.astype(np.float32)); replicates.append(jack)
            # Independent run-level check: initial versus final norm of the same run.
            rho = float(spearman(initial_norm, norm[final_index]))
            jack = np.asarray([spearman(np.delete(initial_norm, i), np.delete(norm[final_index], i)) for i in range(n)])
            null = normalized_ranks(norm[final_index])[label_permutations] @ normalized_ranks(initial_norm)
            se, lo, hi = jackknife_interval(rho, jack)
            pg, pt = permutation_pvalues(rho, null)
            norm_run_rows.append({"dataset": dataset, "gauge": gauge, "layer": layer, "rho": rho,
                                  "p_perm_greater": pg, "p_perm_two_sided": pt, "jackknife_se": float(se), "ci_low": float(lo), "ci_high": float(hi),
                                  "n_runs": n, "n_permutations": len(null), "inferential_role": "exploratory", **null_diagnostics(null)})
            control_keys.append(f"{gauge}|{layer}|run_norm_correlation|20|initial_vs_final")
            nulls.append(null.astype(np.float32)); replicates.append(jack)
    return rows, norm_run_rows, {"control_keys": np.asarray(control_keys), "null_rho": np.asarray(nulls), "jackknife_rho": np.asarray(replicates)}


def _neighbors(distance, k):
    masked = np.array(distance, copy=True)
    np.fill_diagonal(masked, np.inf)
    indices = np.argsort(masked, axis=1, kind="stable")[:, :k]
    result = np.zeros(masked.shape, dtype=bool)
    result[np.arange(len(masked))[:, None], indices] = True
    return result


def _neighbor_controls(dataset, initial, final, permutations):
    rows, nulls, replicas = [], [], []
    n = len(initial)
    for k in (1, 3):
        a, b = _neighbors(initial, k), _neighbors(final, k)
        effect = float(np.sum(a & b) / (n * k))
        permuted = b[permutations[:, :, None], permutations[:, None, :]]
        null = np.sum(a[None, :, :] & permuted, axis=(1, 2)) / (n * k)
        jack = []
        for run in range(n):
            keep = np.arange(n) != run
            ai, bi = _neighbors(initial[np.ix_(keep, keep)], k), _neighbors(final[np.ix_(keep, keep)], k)
            jack.append(np.sum(ai & bi) / ((n - 1) * k))
        jack = np.asarray(jack)
        se, lo, hi = jackknife_interval(effect, jack)
        pg = permutation_pvalues(effect, null)[0]
        rows.append({"dataset": dataset, "k": k, "retention": effect, "expected_uniform_retention": k / (n - 1),
                     "p_perm_greater": pg, "jackknife_se": float(se), "ci_low": float(lo), "ci_high": float(hi),
                     "n_runs": n, "n_permutations": len(null), "null_mean": float(null.mean()),
                     "null_mean_mcse": float(null.std(ddof=1)/np.sqrt(len(null))),
                     "tie_breaking": "stable run-index order", "inferential_role": "exploratory"})
        nulls.append(null.astype(np.float32)); replicas.append(jack)
    return rows, np.asarray(nulls), np.asarray(replicas)


def _combined_inference(primary, checkpoints):
    observed = np.stack([x["observed"] for x in primary])  # dataset, checkpoint
    nulls = np.stack([x["null"] for x in primary])  # dataset, checkpoint, Monte-Carlo sample
    jack = np.stack([x["jackknife"] for x in primary])  # dataset, checkpoint, deleted run
    transform = lambda a: np.arctanh(np.clip(a, -1 + FISHER_EPS, 1 - FISHER_EPS))
    rows, replicate_rows = [], []
    combined_null = np.full((len(checkpoints), N_PERMUTATIONS), np.nan, dtype=np.float32)
    combined_jack = np.full((len(checkpoints), 30), np.nan)
    for cp_index, cp in enumerate(checkpoints):
        if cp == 0:
            rows.append({"checkpoint": 0, "rho": 1.0, "fisher_z": np.nan, "p_perm_greater": np.nan,
                         "p_perm_two_sided": np.nan, "jackknife_se": 0.0, "ci_low": 1.0, "ci_high": 1.0,
                         "fisher_z_se": np.nan, "z_ci_low": np.nan, "z_ci_high": np.nan,
                         "n_datasets": 3, "n_runs": 30, "n_pairs": 135, "n_permutations": 0, "inferential_role": "sanity"})
            continue
        z = transform(observed[:, cp_index])
        estimate_z = float(z.mean()); estimate = float(np.tanh(estimate_z))
        null_z = transform(nulls[:, cp_index]).mean(axis=0)
        null = np.tanh(null_z)
        combined_null[cp_index] = null.astype(np.float32)
        rep_z = np.concatenate([(z.sum() - z[d] + transform(jack[d, cp_index])) / 3 for d in range(3)])
        rep_rho = np.tanh(rep_z)
        combined_jack[cp_index] = rep_rho
        variance_z = sum(9/10 * np.sum((rep_z[d*10:(d+1)*10] - rep_z[d*10:(d+1)*10].mean())**2) for d in range(3))
        variance = sum(9/10 * np.sum((rep_rho[d*10:(d+1)*10] - rep_rho[d*10:(d+1)*10].mean())**2) for d in range(3))
        se, se_z = float(np.sqrt(variance)), float(np.sqrt(variance_z))
        pg, pt = permutation_pvalues(estimate, null)
        rows.append({"checkpoint": int(cp), "rho": estimate, "fisher_z": estimate_z,
                     "p_perm_greater": pg, "p_perm_two_sided": pt, "jackknife_se": se,
                     "ci_low": estimate - Z95*se, "ci_high": estimate + Z95*se,
                     "fisher_z_se": se_z, "z_ci_low": estimate_z - Z95*se_z, "z_ci_high": estimate_z + Z95*se_z,
                     "n_datasets": 3, "n_runs": 30, "n_pairs": 135, "n_permutations": N_PERMUTATIONS,
                     "inferential_role": "secondary_summary" if cp==20 else "exploratory", **null_diagnostics(null)})
        for d in range(3):
            for deleted in range(10):
                index = d*10+deleted
                replicate_rows.append({"checkpoint": int(cp), "dataset_stratum": DATASETS[d],
                                       "deleted_run": str(primary[d]["run_ids"][deleted]), "rho": float(rep_rho[index]), "fisher_z": float(rep_z[index])})
    return rows, replicate_rows, combined_null, combined_jack


def run_inference(out: Path):
    """Compute all RSA, run-level inference, concentration and norm controls."""
    out = Path(out)
    (out / "data").mkdir(parents=True, exist_ok=True)
    rows, concentration, norm_rows, norm_run_rows, neighbor_rows, all_replicates, primary = [], [], [], [], [], [], []
    diagnostics = {"seed_by_dataset": {}, "n_permutations": N_PERMUTATIONS, "p_resolution": 1/(N_PERMUTATIONS+1),
                   "null_storage_dtype": "float32", "calculation_dtype": "float64", "e0_inference": "none; sanity only",
                   "ci_rule": "raw correlation scale normal delete-one-network jackknife; no clipping in saved data",
                   "fisher_boundary_clip": FISHER_EPS, "null_centering_rule": "Report mean and mean/MCSE; six-MCSE descriptive check across all null distributions",
                   "conditional_norm_test": "Descriptive, with exchangeability caveat; ranks residualized against intercept plus ranked pair norm sum and ranked absolute difference"}
    with threadpool_limits(limits=1):
        for dataset_index, dataset in enumerate(DATASETS):
            with np.load(out / "data" / f"{dataset}_distances.npz", allow_pickle=False) as a:
                archive = {k:a[k] for k in a.files}
            distances = archive["distances"].astype(np.float64)
            keys = archive["metric_keys"].astype(str).tolist()
            checkpoints = archive["checkpoints"].astype(int)
            pairs = archive["pairs"].astype(int)
            n = len(archive["run_ids"])
            if n != 10 or pairs.shape != (45,2) or distances.shape != (len(keys),21,10,10) or not np.array_equal(checkpoints,np.arange(21)):
                raise ValueError(f"Unexpected RDM dimensions or checkpoint schedule for {dataset}")
            if not np.isfinite(distances).all() or not np.allclose(distances,distances.swapaxes(-1,-2),rtol=0,atol=1e-12) or np.any(np.diagonal(distances,axis1=-1,axis2=-2)!=0):
                raise ValueError(f"Invalid RDM symmetry/diagonal/finiteness for {dataset}")
            vectors = distances[:,:,pairs[:,0],pairs[:,1]]
            unit = normalized_ranks(vectors)
            observed = np.clip(np.einsum("kp,ktp->kt",unit[:,0],unit),-1,1)
            if not np.isfinite(observed).all() or not np.allclose(observed[:,0],1,rtol=0,atol=1e-12):
                raise ValueError(f"Degenerate RSA or failing E0 identity sanity for {dataset}")
            observed[:,0] = 1.0
            jack = np.empty((len(keys),len(checkpoints),n))
            for run in range(n):
                keep = np.all(pairs != run,axis=1)
                partial_unit = normalized_ranks(vectors[:,:,keep])
                jack[:,:,run] = np.clip(np.einsum("kp,ktp->kt",partial_unit[:,0],partial_unit),-1,1)
            jack[:,0,:] = 1.0
            se, low, high = jackknife_interval(observed,jack)
            seed = BASE_SEED + dataset_index + 1
            diagnostics["seed_by_dataset"][dataset] = seed
            rng = np.random.RandomState(seed)
            label_permutations = np.asarray([rng.permutation(n) for _ in range(N_PERMUTATIONS)], dtype=np.int16)
            edge_indices = identity_edge_indices(label_permutations,pairs,n)
            null_store = np.full((len(keys),len(checkpoints),N_PERMUTATIONS),np.nan,dtype=np.float32)
            primary_null = np.full((len(checkpoints),N_PERMUTATIONS),np.nan,dtype=np.float64)
            primary_index = keys.index(PRIMARY_KEY)
            for ki,key in enumerate(keys):
                fields = _metric_fields(key)
                for ti,cp in enumerate(checkpoints):
                    pg,pt = np.nan,np.nan
                    nd = {"null_mean":np.nan,"null_sd":np.nan,"null_mean_mcse":np.nan,"null_mean_z":np.nan,"null_mean_within_6mcse":None}
                    if cp != 0:
                        null = unit[ki,ti][edge_indices] @ unit[ki,0]
                        null = np.clip(null,-1,1)
                        pg,pt = permutation_pvalues(observed[ki,ti],null)
                        null_store[ki,ti] = null.astype(np.float32)
                        nd = null_diagnostics(null)
                        if ki == primary_index: primary_null[ti] = null
                    rows.append({"dataset":dataset,"checkpoint":int(cp),**fields,"rho":observed[ki,ti],
                                 "p_perm_greater":pg,"p_perm_two_sided":pt,"jackknife_se":se[ki,ti],
                                 "ci_low":low[ki,ti],"ci_high":high[ki,ti],"jackknife_min":float(jack[ki,ti].min()),"jackknife_max":float(jack[ki,ti].max()),
                                 "n_runs":n,"n_pairs":len(pairs),"n_permutations":N_PERMUTATIONS if cp!=0 else 0,
                                 "inferential_role":"sanity" if cp==0 else ("primary" if cp==20 and key==PRIMARY_KEY else "exploratory"),**nd})
                    for run in range(n):
                        all_replicates.append({"dataset":dataset,"checkpoint":int(cp),**fields,
                                               "deleted_index":run,"deleted_run":str(archive["run_ids"][run]),"rho":float(jack[ki,ti,run])})
                    if cp in (0,20): concentration.append(_concentration(dataset,key,cp,vectors[ki,ti]))
            _write_npz(out/"data"/f"{dataset}_inference.npz",null_rho=null_store,jackknife_rho=jack,
                       label_permutations=label_permutations,observed_identity=np.arange(n),edge_indices=edge_indices.astype(np.int16),
                       metric_keys=np.asarray(keys),checkpoints=checkpoints,run_ids=archive["run_ids"],seeds=archive["seeds"],pairs=pairs,
                       rho=observed,permutation_seed=np.asarray(seed))
            nr,nrr,narrays = _norm_controls(dataset,archive,vectors,keys,pairs,label_permutations,edge_indices)
            norm_rows.extend(nr);norm_run_rows.extend(nrr)
            neighbors,nnull,njack = _neighbor_controls(dataset,distances[primary_index,0],distances[primary_index,-1],label_permutations)
            neighbor_rows.extend(neighbors)
            _write_npz(out/"data"/f"{dataset}_controls.npz",**narrays,neighbor_k=np.asarray([1,3]),neighbor_null_retention=nnull,
                       neighbor_jackknife_retention=njack,label_permutations=label_permutations,run_ids=archive["run_ids"],permutation_seed=np.asarray(seed))
            primary.append({"observed":observed[primary_index],"jackknife":jack[primary_index],"null":primary_null,"run_ids":archive["run_ids"]})
            print(f"RSA inference complete: {dataset}, {len(keys)} metric keys x 21 checkpoints, {N_PERMUTATIONS} identity permutations",flush=True)
        aggregate,aggregate_jack,aggregate_null,aggregate_replicates = _combined_inference(primary,checkpoints)
    frame = pd.DataFrame(rows)
    frame["p_holm_greater"] = np.nan;frame["p_holm_two_sided"] = np.nan
    mask = frame.inferential_role.eq("primary")
    if mask.sum()!=3:raise AssertionError("Expected exactly three dataset-specific primary endpoints")
    frame.loc[mask,"p_holm_greater"] = holm_adjust(frame.loc[mask,"p_perm_greater"].to_numpy())
    frame.loc[mask,"p_holm_two_sided"] = holm_adjust(frame.loc[mask,"p_perm_two_sided"].to_numpy())
    for name,data in [("rsa_by_dataset.csv",frame),("rsa_jackknife.csv",all_replicates),("aggregate.csv",aggregate),
                      ("aggregate_jackknife.csv",aggregate_jack),("norm_controls.csv",norm_rows),("norm_run_correlations.csv",norm_run_rows),
                      ("distance_concentration.csv",concentration),("neighbor_retention.csv",neighbor_rows)]:_write_csv(out/name,data)
    _write_npz(out/"data"/"aggregate_inference.npz",null_rho=aggregate_null,jackknife_rho=aggregate_replicates,
               checkpoints=checkpoints,datasets=np.asarray(DATASETS),permutation_seeds=np.arange(BASE_SEED+1,BASE_SEED+4))
    inferred = frame[frame.checkpoint>0]
    diagnostics["null_centering"] = {"distributions":len(inferred),"max_abs_z":float(inferred.null_mean_z.abs().max()),
                                    "outside_6mcse":int((~inferred.null_mean_within_6mcse.astype(bool)).sum())}
    diagnostics["primary_endpoints"] = frame[mask].replace({np.nan:None}).to_dict(orient="records")
    diagnostics["all_e0_rho_equal_one"] = bool((frame.loc[frame.checkpoint==0,"rho"]==1).all())
    diagnostics["csv_files"] = [p.name for p in out.glob("*.csv")]
    _write_json(out/"inference_diagnostics.json",diagnostics)
    return diagnostics


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    run_inference(args.output)
