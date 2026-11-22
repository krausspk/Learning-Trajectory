"""Independent recomputation of endpoints and numeric rank sensitivity."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr


def weights(archive, checkpoint, dtype=np.float64):
    a = archive["W0"][checkpoint].astype(dtype)
    b = archive["W1"][checkpoint].astype(dtype)
    ni, no = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=1)
    assert np.all(ni > 0) and np.all(no > 0)
    scale = np.sqrt(no / ni)
    return a * scale, b / scale[:, None]


def distance(a, b):
    inner = a[0].T @ b[0] + a[1] @ b[1].T
    rows, cols = linear_sum_assignment(-inner)
    assert np.array_equal(rows, np.arange(64))
    squared = np.sum((a[0] - b[0][:, cols])**2) + np.sum((a[1] - b[1][cols])**2)
    # Independent expression, useful for catching orientation mistakes.
    trace_squared = sum(np.sum(w*w) for net in (a,b) for w in net) - 2 * np.sum(inner[rows,cols])
    return float(np.sqrt(squared)), cols, float(abs(squared - trace_squared))


def audit(out: Path):
    root = out.parent
    manifest = json.loads((root / "outputs/manifests/main.json").read_text())
    paths = {r["run_id"]: Path(r["path"]) for r in manifest["runs"]}
    summaries, all_errors = [], []
    estimates = pd.read_csv(out / "rsa_by_dataset.csv")
    sanity = {"all_finite": True, "all_symmetric": True, "all_zero_diagonal": True,
              "all_e0_rho_one": True, "all_endpoint_correlations_match": True,
              "all_primary_distances_match": True, "all_reverse_distances_match": True}
    for dataset in ("mnist", "fashion_mnist", "kmnist"):
        with np.load(out / "data" / f"{dataset}_distances.npz") as stored:
            rdms = stored["distances"]
            keys = stored["metric_keys"].astype(str).tolist()
            idx = keys.index("frobenius|total|optimal|canonical")
            ids = stored["run_ids"].astype(str).tolist()
            pairs = stored["pairs"]
            expected_permutations = stored["permutations"]
        sanity["all_finite"] &= bool(np.isfinite(rdms).all())
        sanity["all_symmetric"] &= bool(np.array_equal(rdms, rdms.swapaxes(-1,-2)))
        sanity["all_zero_diagonal"] &= bool(np.all(np.diagonal(rdms,axis1=-2,axis2=-1)==0))
        assert pairs.shape == (45,2) and len(ids) == 10
        archives = []
        for run_id in ids:
            with np.load(paths[run_id]) as archive:
                archives.append({k: archive[k] for k in ("W0", "W1")})
        for checkpoint in (0,20):
            nets64 = [weights(a, checkpoint) for a in archives]
            nets32 = [weights(a, checkpoint, np.float32) for a in archives]
            values64, values32, reverse_errors, algebra_errors, changed = [], [], [], [], []
            for pair_index, (i,j) in enumerate(pairs):
                d64, p64, algebra_error = distance(nets64[i], nets64[j])
                d32, p32, _ = distance(nets32[i], nets32[j])
                reverse, _, _ = distance(nets64[j], nets64[i])
                values64.append(d64); values32.append(d32)
                reverse_errors.append(abs(d64-reverse)); algebra_errors.append(algebra_error)
                changed.append(np.mean(p64 != p32))
                assert np.array_equal(p64, expected_permutations[checkpoint,pair_index])
            exact = rdms[idx,checkpoint,pairs[:,0],pairs[:,1]]
            error = float(np.max(np.abs(exact-values64)))
            all_errors.append(error)
            sanity["all_primary_distances_match"] &= error < 1e-10
            sanity["all_reverse_distances_match"] &= max(reverse_errors) < 1e-10
            summaries.append({"dataset": dataset, "checkpoint": checkpoint, "n_pairs": 45,
                              "max_abs_independent_distance_error": error,
                              "max_abs_reverse_distance_error": max(reverse_errors),
                              "max_abs_squared_distance_algebra_error": max(algebra_errors),
                              "float32_vs_float64_rank_rho": float(spearmanr(values64,values32).statistic),
                              "float32_vs_float64_max_abs_distance_error": float(np.max(np.abs(np.array(values64)-values32))),
                              "float32_vs_float64_assignment_disagreement": float(np.mean(changed))})
        for k,key in enumerate(keys):
            metric,layer,alignment,gauge = key.split("|")
            rows = estimates[(estimates.dataset==dataset)&(estimates.metric==metric)&(estimates.layer==layer)&(estimates.alignment==alignment)&(estimates.gauge==gauge)]
            sanity["all_e0_rho_one"] &= bool(np.allclose(rows[rows.checkpoint==0].rho,1))
            x,y = rdms[k,0,pairs[:,0],pairs[:,1]], rdms[k,20,pairs[:,0],pairs[:,1]]
            calculated = float(spearmanr(x,y).statistic)
            reported = float(rows[rows.checkpoint==20].rho.iloc[0])
            sanity["all_endpoint_correlations_match"] &= abs(calculated-reported) < 1e-12
    pd.DataFrame(summaries).to_csv(out / "numerical_stability.csv",index=False)
    result = {"checks": sanity, "all_passed": all(sanity.values()),
              "independently_recomputed_pairs": 270,
              "max_primary_distance_error": max(all_errors),
              "note": "Independent float64 canonicalization/Hungarian/endpoint correlation; float32 control tests arithmetic sensitivity only, not stochastic retraining stability."}
    (out / "provenance/independent_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    assert result["all_passed"], result
    return result


if __name__ == "__main__":
    print(json.dumps(audit(Path(__file__).resolve().parents[1]),indent=2))
