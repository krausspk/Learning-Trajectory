#!/usr/bin/env python3
"""Independently audit the completed MNIST pilot without rerunning analysis.

Canonical balancing and the Hungarian/Frobenius calculation are implemented
directly here. This module deliberately does not import experiment metrics,
analysis, or reporting code. The audit reads existing artifacts and writes one
validation receipt; it never changes checkpoints or source results.
"""
from __future__ import annotations

import os
for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(w0, w1):
    w0 = np.asarray(w0, dtype=np.float64)
    w1 = np.asarray(w1, dtype=np.float64)
    incoming = np.sqrt(np.sum(w0 * w0, axis=0))
    outgoing = np.sqrt(np.sum(w1 * w1, axis=1))
    assert np.all(incoming > 0) and np.all(outgoing > 0)
    scale = np.sqrt(outgoing / incoming)
    return w0 * scale[None, :], w1 / scale[:, None]


def direct_distance(a, b):
    a0, a1 = canonical(*a)
    b0, b1 = canonical(*b)
    rows, permutation = linear_sum_assignment(-(a0.T @ b0 + a1 @ b1.T))
    assert np.array_equal(rows, np.arange(len(rows)))
    squared = np.square(a0 - b0[:, permutation]).sum() + np.square(a1 - b1[permutation, :]).sum()
    return float(np.sqrt(squared))


def close(actual, expected, label, tolerance=1e-11):
    assert np.isfinite(actual) and np.isfinite(expected), label
    error = abs(float(actual) - float(expected))
    assert error <= tolerance, f"{label}: {actual} != {expected}; error={error}"
    return error


def run(root):
    root = Path(root).resolve()
    endpoints = pd.read_csv(root / "endpoint_metrics.csv")
    curves = pd.read_csv(root / "recovery_metrics.csv")
    summary = pd.read_csv(root / "endpoint_summary.csv")
    jobs = json.loads((root / "pilot_jobs.json").read_text())
    references = json.loads((root / "reference_jobs.json").read_text())
    thresholds = json.loads((root / "region_thresholds.json").read_text())["thresholds"]
    primary = endpoints[(endpoints.phase == "pilot") & (endpoints.epsilon > 0)].copy()
    assert len(primary) == 300 and primary.seed.nunique() == 10
    assert not primary.failed.any()
    assert primary.groupby(["seed", "epsilon", "momentum"]).size().eq(5).all()

    selected_id = "M1_E20_preserve_both_e0.01_d1_same"
    job = next(j for j in jobs if j["job_id"] == selected_id)
    endpoint = endpoints.set_index("job_id").loc[selected_id]
    curve = curves[curves.job_id == selected_id].sort_values("update")
    selected_path = root / "checkpoints" / f"{selected_id}.npz"
    baseline_path = root / "checkpoints" / f"{job['baseline_id']}.npz"
    with np.load(selected_path, allow_pickle=False) as saved:
        perturbation = {k: saved[k].copy() for k in ("W0", "W1", "recovery_updates", "accuracy", "loss")}
    with np.load(baseline_path, allow_pickle=False) as saved:
        baseline = {k: saved[k].copy() for k in ("W0", "W1", "recovery_updates")}
    np.testing.assert_array_equal(perturbation["recovery_updates"], baseline["recovery_updates"])
    np.testing.assert_array_equal(perturbation["recovery_updates"], curve["update"].to_numpy())
    raw_distances = [direct_distance((perturbation["W0"][k], perturbation["W1"][k]),
                                     (baseline["W0"][k], baseline["W1"][k]))
                     for k in range(len(curve))]
    initial_distance = raw_distances[0]
    errors = [close(value, row.distance, f"distance update {row.update}")
              for value, row in zip(raw_distances, curve.itertuples())]
    ratio_errors = [close(value / initial_distance, row.ratio, f"ratio update {row.update}")
                    for value, row in zip(raw_distances, curve.itertuples())]
    endpoint_ratio = raw_distances[-1] / initial_distance
    close(endpoint_ratio, endpoint.ratio, "endpoint ratio")

    ref_paths = [root / "checkpoints" / f"{j['job_id']}.npz" for j in references]
    ref_weights, ref_accuracy, ref_loss = [], [], []
    for path in ref_paths:
        with np.load(path, allow_pickle=False) as saved:
            ref_weights.append((saved["W0"][-1].copy(), saved["W1"][-1].copy()))
            ref_accuracy.append(float(saved["accuracy"][-1]))
            ref_loss.append(float(saved["loss"][-1]))
    matrix = np.zeros((len(ref_weights), len(ref_weights)))
    for i in range(len(ref_weights)):
        for j in range(i + 1, len(ref_weights)):
            matrix[i, j] = matrix[j, i] = direct_distance(ref_weights[i], ref_weights[j])
    stored_matrix = np.asarray(thresholds[-1]["pairwise_distances"])
    matrix_error = float(np.max(np.abs(matrix - stored_matrix)))
    assert matrix_error < 1e-11
    medians = np.array([np.median(np.delete(matrix[i], i)) for i in range(len(matrix))])
    q95 = float(np.quantile(medians, .95, method="linear"))
    close(q95, thresholds[-1]["distance_q95"], "independent structural threshold")
    close(float(np.quantile(ref_accuracy, .05, method="linear")), thresholds[-1]["accuracy_q05"], "accuracy threshold")
    close(float(np.quantile(ref_loss, .95, method="linear")), thresholds[-1]["loss_q95"], "loss threshold")
    chosen_weights = (perturbation["W0"][-1], perturbation["W1"][-1])
    median_distance = float(np.median([direct_distance(chosen_weights, ref) for ref in ref_weights]))
    close(median_distance, endpoint.region_distance, "selected reference median distance")
    accuracy, loss = float(perturbation["accuracy"][-1]), float(perturbation["loss"][-1])
    structural_member = median_distance <= q95
    functional_member = accuracy >= thresholds[-1]["accuracy_q05"] and loss <= thresholds[-1]["loss_q95"]
    assert structural_member == bool(endpoint.structural_member)
    assert functional_member == bool(endpoint.functional_member)
    assert (structural_member and functional_member) == bool(endpoint.region_member)

    means = []
    for (epsilon, momentum), frame in primary.groupby(["epsilon", "momentum"]):
        per_seed = frame.groupby("seed").ratio.agg(lambda values: values.mean(skipna=False))
        mean = float(per_seed.mean(skipna=False))
        row = summary[(summary.phase == "pilot") & (summary.epsilon == epsilon) & (summary.momentum == momentum)].iloc[0]
        error = close(mean, row.ratio_mean, f"seed mean {epsilon} {momentum}")
        assert row.ratio_low > 1
        means.append(dict(epsilon=float(epsilon), momentum=momentum, n_seeds=len(per_seed),
                          recomputed_mean=mean, csv_mean=float(row.ratio_mean), error=error,
                          reported_ci_low=float(row.ratio_low), reported_ci_high=float(row.ratio_high)))

    primary_curves = curves[curves.job_id.isin(primary.job_id)].copy()
    initial = primary_curves[primary_curves["update"] == 0].set_index("job_id")
    final = primary.set_index("job_id")
    initial = initial.loc[final.index]
    assert len(initial) == len(final)
    # Independently reconstruct logical events from initial and final records.
    logical_checks = {
        "fixed_return": ~initial.fixed_region_member & final.fixed_region_member & ~final.failed,
        "fixed_structural_return": ~initial.fixed_structural_member & final.fixed_structural_member & ~final.failed,
        "initially_outside_return": ~initial.region_member & final.region_member & ~final.failed,
        "structural_return": ~initial.structural_member & final.structural_member & ~final.failed,
    }
    logical_checks["perturbation_induced_return"] = final.initial_baseline_fixed_member & logical_checks["fixed_return"]
    for name, expected in logical_checks.items():
        np.testing.assert_array_equal(expected.to_numpy(), final[name].to_numpy(), err_msg=name)
    moving_ever_out = primary_curves.groupby("job_id").structural_member.apply(lambda values: bool((~values).any()))
    fixed_ever_out = primary_curves.groupby("job_id").fixed_structural_member.apply(lambda values: bool((~values).any()))
    fixed_reentry = fixed_ever_out.reindex(final.index) & final.fixed_structural_member
    counts = dict(
        total_perturbations=len(primary),
        initially_outside_moving_structure=int((~initial.structural_member).sum()),
        initially_outside_fixed_structure=int((~initial.fixed_structural_member).sum()),
        initially_outside_fixed_combined=int((~initial.fixed_region_member).sum()),
        initially_below_fixed_accuracy=int((initial.accuracy < thresholds[-1]["accuracy_q05"]).sum()),
        initially_above_fixed_loss=int((initial.loss > thresholds[-1]["loss_q95"]).sum()),
        moving_structural_outside_to_inside=int(logical_checks["structural_return"].sum()),
        moving_combined_outside_to_inside=int(logical_checks["initially_outside_return"].sum()),
        fixed_structural_outside_to_inside=int(logical_checks["fixed_structural_return"].sum()),
        fixed_combined_outside_to_inside=int(logical_checks["fixed_return"].sum()),
        perturbation_induced_return=int(logical_checks["perturbation_induced_return"].sum()),
        fixed_structural_ever_outside=int(fixed_ever_out.sum()),
        moving_structural_ever_outside=int(moving_ever_out.sum()),
        fixed_structural_ever_outside_and_final_inside=int(fixed_reentry.sum()),
    )
    assert counts["initially_outside_fixed_structure"] == 0
    assert counts["initially_below_fixed_accuracy"] == 300
    assert counts["fixed_structural_ever_outside_and_final_inside"] == 0
    baseline_initial = curves[(curves.phase == "pilot") & (curves.epsilon == 0) & (curves["update"] == 0)]
    assert baseline_initial.seed.nunique() == 10
    assert baseline_initial.fixed_structural_member.all()
    assert not baseline_initial.fixed_functional_member.any()
    assert (baseline_initial.accuracy < thresholds[-1]["accuracy_q05"]).all()
    assert not primary.initial_baseline_fixed_member.any()

    positive_updates = primary_curves[primary_curves["update"] > 0]
    first_update = primary_curves[primary_curves["update"] == 1]
    assert len(first_update) == 300 and (first_update.ratio < 1).all()
    assert (primary.ratio > 1).all()
    early_final = dict(first_update_contractions=int((first_update.ratio < 1).sum()),
                       minimum_ratio_after_injection=float(positive_updates.ratio.min()),
                       final_contractions=int((primary.ratio < 1).sum()),
                       minimum_final_ratio=float(primary.ratio.min()), maximum_final_ratio=float(primary.ratio.max()))

    barriers = json.loads((root / "loss_interpolations.json").read_text())
    saved_barrier = next(item for item in barriers if item["job_id"] == selected_id)
    values = np.asarray(saved_barrier["losses"], dtype=float)
    assert len(values) == 11 and np.isfinite(values).all()
    reconstructed_barrier = max(0., float(np.max(values - np.linspace(values[0], values[-1], len(values)))))
    close(reconstructed_barrier, endpoint.barrier, "barrier reconstructed from saved losses")

    inputs = [root / name for name in ("endpoint_metrics.csv", "recovery_metrics.csv", "endpoint_summary.csv",
              "pilot_jobs.json", "reference_jobs.json", "region_thresholds.json", "loss_interpolations.json")]
    inputs.extend([selected_path, baseline_path, *ref_paths])
    receipt = dict(
        passed=True, audited_at=datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
        scope="Independent numerical and logical audit of completed MNIST pilot; no training or full analysis rerun",
        implementation="Direct float64 canonical balancing, scipy Hungarian assignment, pooled Frobenius norm; no project metric imports",
        script_sha256=sha256(__file__),
        input_sha256={str(path.relative_to(root)): sha256(path) for path in inputs},
        selected_fork=dict(job_id=selected_id, initial_distance=initial_distance, final_distance=raw_distances[-1],
                           final_ratio=endpoint_ratio, checkpoints_verified=len(raw_distances),
                           maximum_distance_error=max(errors), maximum_ratio_error=max(ratio_errors)),
        references=dict(first_pair_distance=float(matrix[0, 1]), maximum_pairwise_error=matrix_error,
                        recomputed_q95=q95, selected_median_distance=median_distance,
                        selected_structural_member=bool(structural_member), selected_functional_member=bool(functional_member),
                        selected_combined_member=bool(structural_member and functional_member)),
        seed_level_means=means, region_events=counts, early_and_final_contraction=early_final,
        selected_barrier=dict(recomputed=reconstructed_barrier, csv=float(endpoint.barrier), interpolation_points=len(values)),
        inference_limits=[
            "All initial perturbations remain inside the fixed structural region: immediate structural departure-and-return was not tested at these amplitudes.",
            "All initial candidates and source baselines fail fixed endpoint accuracy; combined outside-to-inside events do not establish perturbation-induced structural return.",
            "Moving-reference-envelope entries are distinct from fixed-region return; small early contraction does not persist to the final horizon.",
            "This audit verifies reported point estimates and event logic; it does not add independent experimental seeds or validate population coverage of bootstrap intervals.",
        ],
    )
    output = root / "validation" / "independent_numeric_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(output)
    print(json.dumps({"passed": True, "output": str(output), "selected_ratio": endpoint_ratio,
                      "seed_means_verified": len(means), "region_events": counts}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    run(parser.parse_args().root)
