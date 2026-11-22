#!/usr/bin/env python3
"""Verify independent pilot/expanded repeats bitwise after MNIST E20 completes.

The stages differ in their job phase and output location. Their 20 baselines
and 60 common perturbations have identical training inputs and therefore must
produce identical arrays at every saved recovery update, irrespective of worker
concurrency. This runner never launches training or changes checkpoints.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import json
from pathlib import Path
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from learning_trajectory.data import _atomic_json, sha256_file


ARRAY_KEYS = ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss",
              "bias0", "bias1", "recovery_updates")
CELL = ROOT / "expanded/mnist_E20"


def selected_pilot_job(job):
    if job["phase"] != "pilot" or job["dataset"] != "mnist" or job["checkpoint"] != 20:
        return False
    if job["order"] != "same" or job["layer"] != "both":
        return False
    if job["epsilon"] == 0:
        return job["direction"] == 0
    return job["direction"] == 1 and job["epsilon"] in (.01, .05, .20)


def require_finished_cell():
    """An experimental failure is finished; missing receipts mean still running."""
    required = []
    for filename in ("pilot_jobs.json", "reference_jobs.json"):
        path = CELL / filename
        if not path.exists():
            raise RuntimeError("MNIST E20 expansion is not prepared; overlap validation cannot run yet")
        required.extend(json.loads(path.read_text()))
    if len(required) != 270:
        raise AssertionError("The registered MNIST E20 cell must contain 260 forks and 10 references")
    missing = [j["job_id"] for j in required if not (CELL / "receipts" / f"{j['job_id']}.json").exists()]
    if missing:
        raise RuntimeError(f"MNIST E20 expansion is not finished: {len(missing)} of 270 receipts missing; run this script after completion")
    for job in required:
        receipt = json.loads((CELL / "receipts" / f"{job['job_id']}.json").read_text())
        if receipt.get("status") not in ("complete", "failed"):
            raise RuntimeError(f"MNIST E20 receipt has no terminal status: {job['job_id']}")


def compare_pair(pilot_job, expanded_job):
    job_id = pilot_job["job_id"]
    pilot_path = ROOT / "checkpoints" / f"{job_id}.npz"
    expanded_path = CELL / "checkpoints" / f"{job_id}.npz"
    checks = {}
    left = {k: v for k, v in pilot_job.items() if k != "phase"}
    right = {k: v for k, v in expanded_job.items() if k != "phase"}
    checks["same_job_except_phase"] = left == right
    checks["expected_stages"] = pilot_job["phase"] == "pilot" and expanded_job["phase"] == "expanded"
    for label, folder, path in (("pilot", ROOT, pilot_path), ("expanded", CELL, expanded_path)):
        receipt = json.loads((folder / "receipts" / f"{job_id}.json").read_text())
        checks[f"{label}_complete"] = receipt["status"] == "complete"
        checks[f"{label}_15_snapshots"] = receipt["checkpoints"] == 15
        checks[f"{label}_2350_updates"] = receipt["final_recovery_update"] == 2350
        checks[f"{label}_archive_hash_valid"] = sha256_file(path) == receipt["npz_sha256"]
    observed_norms = {}
    with np.load(pilot_path, allow_pickle=False) as a, np.load(expanded_path, allow_pickle=False) as b:
        for key in ARRAY_KEYS:
            # Equal numerical values alone would not catch accidental dtype
            # changes. Require shape, dtype and raw array bytes to match.
            aa, bb = a[key], b[key]
            checks[f"bitwise_{key}"] = (aa.shape == bb.shape and aa.dtype == bb.dtype
                                          and aa.tobytes(order="C") == bb.tobytes(order="C"))
        am, bm = json.loads(str(a["metadata"])), json.loads(str(b["metadata"]))
        checks["same_epoch_batch_order_hashes"] = am["epoch_order_hashes"] == bm["epoch_order_hashes"]
        checks["ten_epoch_orders"] = len(am["epoch_order_hashes"]) == len(bm["epoch_order_hashes"]) == 10
        checks["same_source_archive"] = am["source_sha256"] == bm["source_sha256"]
        checks["same_training_implementation"] = am["source_code_sha256"] == bm["source_code_sha256"]
        checks["same_dataset_manifest"] = am["dataset_manifest_sha256"] == bm["dataset_manifest_sha256"]
        checks["same_perturbation_norm_metadata"] = am["perturbation"] == bm["perturbation"]
        checks["same_planned_recovery_updates"] = am["planned_recovery_updates"] == bm["planned_recovery_updates"]
        checks["all_15_updates_present"] = (a["recovery_updates"].tolist() == am["planned_recovery_updates"]
                                             and b["recovery_updates"].tolist() == bm["planned_recovery_updates"])
        for layer in ("W0", "W1"):
            target = float(pilot_job["epsilon"])
            checks[f"{layer}_correct_requested_norm"] = (
                am["perturbation"][layer]["requested_relative_norm"] == target
                and bm["perturbation"][layer]["requested_relative_norm"] == target)
            checks[f"{layer}_correct_actual_norm"] = (
                abs(am["perturbation"][layer]["actual_relative_norm"] - target) <= 1e-7
                and abs(bm["perturbation"][layer]["actual_relative_norm"] - target) <= 1e-7)
        # Recompute perturbation norms directly from saved initial weights and
        # their corresponding unperturbed baseline, not only metadata fields.
        for stage, folder, archive in (("pilot", ROOT, a), ("expanded", CELL, b)):
            baseline_path = folder / "checkpoints" / f"{pilot_job['baseline_id']}.npz"
            with np.load(baseline_path, allow_pickle=False) as baseline:
                for layer in ("W0", "W1"):
                    start = baseline[layer][0].astype(np.float64)
                    delta = archive[layer][0].astype(np.float64) - start
                    actual = float(np.linalg.norm(delta) / np.linalg.norm(start))
                    observed_norms[f"{stage}_{layer}"] = actual
                    checks[f"{stage}_{layer}_recomputed_norm"] = abs(actual - float(pilot_job["epsilon"])) <= 1e-7
    return {"job_id": job_id, "epsilon": pilot_job["epsilon"], "momentum": pilot_job["momentum"],
            "seed": pilot_job["seed"], "observed_relative_norms": observed_norms,
            "checks": checks, "passed": all(checks.values())}


def main():
    require_finished_cell()
    pilot_jobs = [j for j in json.loads((ROOT / "pilot_jobs.json").read_text()) if selected_pilot_job(j)]
    expanded_jobs = {j["job_id"]: j for j in json.loads((CELL / "pilot_jobs.json").read_text())}
    if len(pilot_jobs) != 80 or len({j["job_id"] for j in pilot_jobs}) != 80:
        raise AssertionError("Expected exactly 80 distinct overlap pairs")
    if sum(j["epsilon"] == 0 for j in pilot_jobs) != 20:
        raise AssertionError("Expected 20 common baselines")
    missing = [j["job_id"] for j in pilot_jobs if j["job_id"] not in expanded_jobs]
    if missing:
        raise AssertionError(f"Registered expanded jobs lack pilot duplicates: {missing}")
    records = [compare_pair(j, expanded_jobs[j["job_id"]]) for j in sorted(pilot_jobs, key=lambda j: j["job_id"])]
    result = {"checked_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
              "purpose": "Independent stage/worker-concurrency deterministic repeat consistency",
              "pilot_root": str(ROOT), "expanded_cell": str(CELL),
              "expected_pairs": 80, "checked_pairs": len(records), "baseline_pairs": 20,
              "perturbation_pairs": 60, "snapshots_per_pair": 15, "snapshots_compared": 80 * 15,
              "array_keys": list(ARRAY_KEYS),
              "excluded": "Different-batch-order controls, directions2..5, additional epsilon0.10 and single-layer interventions",
              "passed": len(records) == 80 and all(r["passed"] for r in records), "pairs": records}
    destination = ROOT / "validation/pilot_expanded_overlap.json"
    _atomic_json(destination, result)
    print(json.dumps({"passed": result["passed"], "checked_pairs": len(records),
                      "snapshots_compared": result["snapshots_compared"], "output": str(destination)}), flush=True)
    if not result["passed"]:
        errors = {r["job_id"]: [key for key, passed in r["checks"].items() if not passed]
                  for r in records if not r["passed"]}
        raise AssertionError(f"Pilot/expanded overlap differs: {errors}")


if __name__ == "__main__":
    main()
