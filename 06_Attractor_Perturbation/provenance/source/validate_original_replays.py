#!/usr/bin/env python3
"""Compare 60 expanded baseline trajectories with original archived E20 states.

No training is performed. For all 30 main source models, an unperturbed,
momentum-preserving E2 fork after 2,115 recovery updates and an E10 fork after
1,175 updates must reproduce the original ten-epoch endpoint bitwise. Run after
expanded training has completed; this verifies continuation across all tasks.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

from datetime import datetime
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from learning_trajectory.data import _atomic_json, sha256_file

DATASETS = ("mnist", "fashion_mnist", "kmnist")
RECOVERY_TARGETS = {2: 2115, 10: 1175}
ARRAY_KEYS = ("W0", "W1", "momentum_W0", "momentum_W1", "bias0", "bias1", "accuracy", "loss")


def selected_baseline(job):
    return (job["phase"] == "expanded" and job["momentum"] == "preserve"
            and job["epsilon"] == 0 and job["order"] == "same"
            and job["layer"] == "both" and job["direction"] == 0)


def registered_pairs():
    pairs = []
    for dataset in DATASETS:
        for checkpoint, recovery_update in RECOVERY_TARGETS.items():
            cell = ROOT / "expanded" / f"{dataset}_E{checkpoint}"
            path = cell / "pilot_jobs.json"
            if not path.exists():
                raise RuntimeError(f"Expanded cell has not been prepared: {cell.name}")
            jobs = [job for job in json.loads(path.read_text()) if selected_baseline(job)]
            if len(jobs) != 10 or len({job["seed"] for job in jobs}) != 10:
                raise AssertionError(f"Expected ten independent preserved baselines in {cell.name}")
            if any(job["dataset"] != dataset or job["checkpoint"] != checkpoint for job in jobs):
                raise AssertionError(f"Job/cell configuration mismatch in {cell.name}")
            pairs.extend((cell, job, recovery_update) for job in jobs)
    if len(pairs) != 60:
        raise AssertionError("Expected exactly 60 original-trajectory replay comparisons")
    return pairs


def require_completed_expansion():
    path = ROOT / "expanded_training_status.json"
    if not path.exists():
        raise RuntimeError("Finish expanded training before running original-trajectory replay validation")
    status = json.loads(path.read_text())
    if not status.get("complete") or status.get("completed_runs") != 2340:
        raise RuntimeError("All 2,340 expanded forks must have terminal results before replay validation")


def main():
    require_completed_expansion()
    pairs = registered_pairs()
    frozen = json.loads((ROOT / "provenance/expanded_source_checksums_before.json").read_text())
    source_hashes = {}
    fork_hashes = {}
    records = []
    for cell, job, target_update in pairs:
        source_pt = Path(job["source"])
        if not source_pt.is_absolute():
            source_pt = PROJECT / source_pt
        original_path = source_pt.with_suffix(".npz")
        fork_path = cell / "checkpoints" / f"{job['job_id']}.npz"
        receipt_path = cell / "receipts" / f"{job['job_id']}.json"
        if not receipt_path.exists():
            raise RuntimeError(f"Required baseline has not finished: {job['job_id']}")
        receipt = json.loads(receipt_path.read_text())
        checks = {"complete_baseline": receipt["status"] == "complete"}
        for source in (source_pt, original_path):
            key = str(source.relative_to(PROJECT))
            if key not in source_hashes:
                source_hashes[key] = sha256_file(source)
            checks["unchanged_source_" + source.suffix.lstrip(".")] = source_hashes[key] == frozen[key]
        fork_key = str(fork_path.relative_to(PROJECT))
        fork_hashes[fork_key] = sha256_file(fork_path)
        checks["valid_fork_archive_hash"] = fork_hashes[fork_key] == receipt["npz_sha256"]
        with np.load(original_path, allow_pickle=False) as original, np.load(fork_path, allow_pickle=False) as fork:
            metadata = json.loads(str(fork["metadata"]))
            checks["source_pt_matches_metadata"] = metadata["source_sha256"] == source_hashes[str(source_pt.relative_to(PROJECT))]
            checks["metadata_job_matches_registration"] = metadata["job"] == job
            source_indices = np.flatnonzero(original["checkpoints"] == 20)
            start_indices = np.flatnonzero(original["checkpoints"] == job["checkpoint"])
            recovery_indices = np.flatnonzero(fork["recovery_updates"] == target_update)
            checks["unique_original_E20"] = len(source_indices) == 1
            checks["unique_original_start"] = len(start_indices) == 1
            checks["required_recovery_snapshot_present"] = len(recovery_indices) == 1
            if len(source_indices) == len(start_indices) == len(recovery_indices) == 1:
                si, start, fi = int(source_indices[0]), int(start_indices[0]), int(recovery_indices[0])
                checks["matching_absolute_update"] = int(original["updates"][start]) + target_update == int(original["updates"][si]) == 2350
                for key in ARRAY_KEYS:
                    a, b = np.asarray(original[key][si]), np.asarray(fork[key][fi])
                    checks["bitwise_" + key] = (a.shape == b.shape and a.dtype == b.dtype
                                                 and a.tobytes(order="C") == b.tobytes(order="C"))
            else:
                checks["matching_absolute_update"] = False
                checks.update({"bitwise_" + key: False for key in ARRAY_KEYS})
        records.append({"job_id": job["job_id"], "dataset": job["dataset"], "seed": job["seed"],
                        "source_checkpoint": job["checkpoint"], "recovery_update": target_update,
                        "original_endpoint_checkpoint": 20, "original_absolute_update": 2350,
                        "checks": checks, "passed": all(checks.values())})
    result = {"checked_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
              "purpose": "Bitwise continuation of original archived trajectories across all three datasets",
              "expected_pairs": 60, "checked_pairs": len(records), "source_networks": 30,
              "datasets": list(DATASETS), "recovery_targets": RECOVERY_TARGETS,
              "array_keys": list(ARRAY_KEYS), "script_sha256": sha256_file(__file__),
              "source_hashes": source_hashes, "fork_npz_hashes": fork_hashes,
              "passed": len(records) == 60 and all(r["passed"] for r in records), "pairs": records}
    path = ROOT / "validation/original_trajectory_replays.json"
    _atomic_json(path, result)
    print(json.dumps({"checked_pairs": len(records), "source_networks": 30,
                      "passed": result["passed"], "output": str(path)}), flush=True)
    if not result["passed"]:
        failures = {r["job_id"]: [name for name, passed in r["checks"].items() if not passed]
                    for r in records if not r["passed"]}
        raise AssertionError(f"Original trajectory replays differed: {failures}")


if __name__ == "__main__":
    main()
