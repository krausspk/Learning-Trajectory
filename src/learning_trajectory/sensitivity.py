"""Post-hoc early-training alignment sensitivity, separate from primary results.

The publication names canonical Alpha in the early experiment but does not
explicitly distinguish the direct-profile versus candidate-selected assignment
there. This diagnostic keeps every trained weight fixed and changes only the
assignment rule. It must not replace the prespecified primary calculation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from . import metrics as m
from .analysis import file_hash, load_runs, write_csv, write_json
from .data import _atomic_npz
from .statistics import group_summary, jackknife

METHODS = ("canonical_alpha", "benchmark", "git_rebasin")
DEFINITIONS = {
    "canonical_alpha": "Direct Hungarian assignment of canonical incoming-plus-outgoing Alpha profiles; unchanged primary definition.",
    "benchmark": "Choose coarse canonical-PIF or direct canonical-Alpha profile assignment by minimum size-weighted canonical Alpha; evaluate balanced canonical Alpha.",
    "git_rebasin": "Maximize raw whole-network Frobenius inner product by exact Hungarian assignment; evaluate balanced canonical Alpha.",
}


def _groups_and_pairs(runs):
    groups = np.asarray([r["meta"]["dataset"] for r in runs])
    pairs = np.asarray(list(itertools.combinations(range(len(runs)), 2)), dtype=np.int64)
    return groups, pairs


def _summarize_checkpoint(pairs, values, groups, checkpoint, method, run_ids):
    summaries = group_summary(pairs, values, groups)
    summaries["all"] = jackknife(pairs, values, groups)
    estimates, replicates = [], []
    for group, result in summaries.items():
        base = {"cohort": "early", "checkpoint": int(checkpoint), "metric": "canonical_alpha",
                "layer": "balanced", "alignment": method, "group": group, "post_hoc": True}
        estimates.append({**base, **{k: result[k] for k in ("estimate", "se", "ci_low", "ci_high")}})
        for deleted, stratum, value in zip(result["deleted"], result["strata"], result["replicates"]):
            replicates.append({**base, "deleted_index": int(deleted), "deleted_run": run_ids[int(deleted)],
                               "stratum": str(stratum), "value": float(value)})
    return estimates, replicates


def run_sensitivity(output="outputs"):
    output = Path(output)
    destination = output / "sensitivity"
    runs = load_runs(output, "early")
    run_ids = [r["meta"]["run_id"] for r in runs]
    groups, pairs = _groups_and_pairs(runs)
    checkpoints = runs[0]["checkpoints"].astype(int)
    if len(runs) != 15 or len(pairs) != 105 or len(checkpoints) != 16:
        raise ValueError("Sensitivity requires the complete published early cohort")
    if any(not np.array_equal(checkpoints, r["checkpoints"]) for r in runs):
        raise ValueError("Inconsistent early checkpoint schedules")
    pair_rows, estimates, replicates, checks = [], [], [], []
    permutations = np.empty((len(checkpoints), len(pairs), len(METHODS), 64), dtype=np.int64)
    inputs = [r["path"] for r in runs]
    source_files = [Path(__file__), Path(m.__file__), Path(__file__).with_name("statistics.py")]
    with threadpool_limits(limits=1):
        for cp_index, cp in enumerate(checkpoints):
            source_pairs = output / "pairs" / "early" / f"checkpoint_{cp:04d}.csv"
            inputs.append(source_pairs)
            existing = pd.read_csv(source_pairs)
            existing = existing[(existing.metric == "canonical_alpha") & (existing.layer == "balanced") & (existing.alignment == "profile")]
            old = existing.set_index(["index_a", "index_b"]).value
            if len(existing) != len(pairs) or not old.index.is_unique:
                raise ValueError(f"Incomplete or duplicated primary pair values at update {cp}")
            weights = [tuple(r[key][cp_index].astype(np.float64) for key in ("W0", "W1")) for r in runs]
            values_by_method = {method: [] for method in METHODS}
            for pair_index, (ia, ib) in enumerate(pairs):
                a, b = weights[ia], weights[ib]
                for method_index, method in enumerate(METHODS):
                    permutation = m.align(*a, *b, method=method)
                    value = m.layers(*a, *b, permutation, metric="alpha", canonical=True)["balanced"]
                    permutations[cp_index, pair_index, method_index] = permutation
                    values_by_method[method].append(value)
                    pair_rows.append({"cohort": "early", "checkpoint": int(cp), "index_a": int(ia), "index_b": int(ib),
                                      "run_a": run_ids[ia], "run_b": run_ids[ib], "group_a": str(groups[ia]), "group_b": str(groups[ib]),
                                      "within": bool(groups[ia] == groups[ib]), "metric": "canonical_alpha", "layer": "balanced",
                                      "alignment": method, "value": value, "post_hoc": True})
                    if method == "canonical_alpha":
                        error = abs(value - float(old.loc[(ia, ib)]))
                        if error > 1e-12:
                            raise AssertionError(f"Primary direct result mismatch at update {cp}, pair {ia}:{ib}: {error}")
                        checks.append({"checkpoint": int(cp), "index_a": int(ia), "index_b": int(ib), "absolute_error": error})
            for method in METHODS:
                mean_rows, replicate_rows = _summarize_checkpoint(pairs, np.asarray(values_by_method[method]), groups, cp, method, run_ids)
                estimates.extend(mean_rows)
                replicates.extend(replicate_rows)
            print(f"Post-hoc alignment sensitivity: update {cp}, {len(pairs)} pairs x {len(METHODS)} methods", flush=True)
    estimates = pd.DataFrame(estimates)
    pair_rows = pd.DataFrame(pair_rows)
    write_csv(destination / "early_alignment_pairs.csv", pair_rows)
    write_csv(destination / "early_alignment_estimates.csv", estimates)
    write_csv(destination / "early_alignment_jackknife.csv", pd.DataFrame(replicates))
    write_csv(destination / "direct_recomputation_checks.csv", pd.DataFrame(checks))
    _atomic_npz(destination / "early_alignment_permutations.npz", permutations=permutations,
                checkpoints=checkpoints, pairs=pairs, methods=np.asarray(METHODS), run_ids=np.asarray(run_ids), groups=groups)
    contrasts = estimates[estimates.group == "contrast"]
    onset, update4 = {}, {}
    for method in METHODS:
        selected = contrasts[contrasts.alignment == method]
        qualifying = selected[(selected.checkpoint > 0) & (selected.ci_low > 0)]
        onset[method] = int(qualifying.checkpoint.min()) if len(qualifying) else None
        row = selected[selected.checkpoint == 4].iloc[0]
        update4[method] = {k: float(row[k]) for k in ("estimate", "se", "ci_low", "ci_high")}
    fingerprints = {str(path): file_hash(path) for path in inputs + source_files}
    diagnostic = {
        "post_hoc": True, "primary_results_modified": False,
        "purpose": "Diagnose how alignment conventions change early canonical Alpha while holding all newly trained weights fixed.",
        "trigger": "The primary direct-profile update-4 contrast differed materially from the reported early-training value; this check was designed after seeing that discrepancy.",
        "inference_limit": "Sensitivity to assignment is not evidence of an error in the publication or proof of which unpublished implementation was used. No parameters or seeds were selected to fit reported values.",
        "cohort": "early", "independent_runs": len(runs), "runs_per_dataset": 5,
        "checkpoints": checkpoints.tolist(), "unordered_pairs_per_checkpoint": len(pairs),
        "methods": list(METHODS), "method_definitions": DEFINITIONS,
        "evaluation": "Canonically gauged layer-balanced Alpha (exponent=2) for every method, at every checkpoint.",
        "uncertainty": "Dataset-stratified delete-one-complete-run jackknife, normal pointwise 95% intervals; never treat pairs as independent replicates.",
        "onset_rule": "First saved post-initialization checkpoint whose complete 95% contrast interval is above zero.",
        "onset_by_method": onset, "update4_contrasts": update4,
        "published_update4_reference": {"estimate": 0.1322, "ci_low": 0.1072, "ci_high": 0.1571,
                                        "source": "Main paper section 3.9 / Figure 6; Supplement S7, page13",
                                        "role": "Comparison only; not an optimization or acceptance target"},
        "direct_recomputation": {"pairs_checked": len(checks), "max_absolute_error": max(x["absolute_error"] for x in checks), "tolerance": 1e-12, "passed": True},
        "permutation_array_axes": ["checkpoint", "unordered_pair", "method", "hidden_neuron"],
        "files": {"pairs": "early_alignment_pairs.csv", "estimates": "early_alignment_estimates.csv",
                  "jackknife": "early_alignment_jackknife.csv", "permutations": "early_alignment_permutations.npz",
                  "direct_checks": "direct_recomputation_checks.csv"},
        "input_and_source_sha256": fingerprints,
        "fingerprint": hashlib.sha256(json.dumps(fingerprints, sort_keys=True).encode()).hexdigest(),
    }
    write_json(destination / "diagnostic.json", diagnostic)
    return diagnostic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs")
    args = parser.parse_args()
    result = run_sensitivity(args.output)
    print(json.dumps({"onset_by_method": result["onset_by_method"], "update4_contrasts": result["update4_contrasts"],
                      "direct_recomputation": result["direct_recomputation"]}, indent=2))


if __name__ == "__main__":
    main()
