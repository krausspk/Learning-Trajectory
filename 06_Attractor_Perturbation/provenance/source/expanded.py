#!/usr/bin/env python3
"""Pilot-gated, prospectively frozen 3×3 factorial screening extension.

This is a separate screening stage: ten source models per task and one shared
Gaussian direction per source/checkpoint, paired across amplitudes, layers and
momentum protocols. The five-direction MNIST pilot remains independently saved.
"""
from __future__ import annotations

import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from learning_trajectory.data import _atomic_json, sha256_file
from run import freeze, job_for

DATASETS = (("mnist", "M", 100), ("fashion_mnist", "F", 200), ("kmnist", "K", 300))
CHECKPOINTS = (2, 10, 20)
EPSILONS = (.01, .05, .10, .20)
LAYERS = ("both", "W0", "W1")
MOMENTUM = ("preserve", "reset")


def stamp():
    return datetime.now(ZoneInfo("Europe/Berlin")).isoformat()


def cells():
    return [(dataset, prefix, seedbase, checkpoint, ROOT / "expanded" / f"{dataset}_E{checkpoint}")
            for dataset, prefix, seedbase in DATASETS for checkpoint in CHECKPOINTS]


def require_completed_pilot():
    status_path = ROOT / "analysis_status.json"
    validation_path = ROOT / "validation/checks.json"
    if not status_path.exists() or not validation_path.exists():
        raise RuntimeError("Finish pilot analysis and technical validation before preparing expansion")
    status = json.loads(status_path.read_text())
    validation = json.loads(validation_path.read_text())
    expected = len(json.loads((ROOT / "pilot_jobs.json").read_text()))
    if not status.get("complete") or status.get("analyzed_runs") != expected:
        raise RuntimeError("The full, nonselective pilot must be analyzed before expansion")
    if not validation.get("passed"):
        raise RuntimeError("Pilot technical validation failed; do not expand")
    return {"analysis_status_sha256": sha256_file(status_path),
            "validation_sha256": sha256_file(validation_path),
            "config_sha256": sha256_file(ROOT / "config.json"),
            "endpoint_summary_sha256": sha256_file(ROOT / "endpoint_summary.csv"),
            "pilot_analyzed_runs": expected,
            "pilot_numerical_failure_count": status.get("n_final_failures"),
            "scientific_effects_are_not_gate_criteria": True}


def new_reference_specs():
    return [{"run_id": f"R{prefix}{i}", "cohort": "heldout", "dataset": dataset,
             "task": "identity", "seed": seedbase + 1000 + i}
            for dataset, prefix, seedbase in DATASETS if dataset != "mnist" for i in range(1, 11)]


def reference_job(dataset, prefix, seedbase, checkpoint, index):
    if dataset == "mnist":
        return job_for(f"I{index}", dataset, 400 + index, checkpoint=checkpoint,
                       phase="reference", cohort="task")
    job = job_for(f"R{prefix}{index}", dataset, seedbase + 1000 + index,
                  checkpoint=checkpoint, phase="reference", cohort="heldout")
    job["source"] = str(ROOT / "reference_sources/checkpoints/heldout" / f"R{prefix}{index}.pt")
    return job


def cell_jobs(dataset, prefix, seedbase, checkpoint):
    jobs = []
    for index in range(1, 11):
        for protocol in MOMENTUM:
            jobs.append(job_for(f"{prefix}{index}", dataset, seedbase + index,
                                checkpoint=checkpoint, momentum=protocol, phase="expanded"))
            for layer in LAYERS:
                for epsilon in EPSILONS:
                    jobs.append(job_for(f"{prefix}{index}", dataset, seedbase + index,
                        checkpoint=checkpoint, momentum=protocol, epsilon=epsilon,
                        layer=layer, direction=1, phase="expanded"))
    if len(jobs) != 260 or len({j["job_id"] for j in jobs}) != len(jobs):
        raise AssertionError("Incorrect expanded job matrix")
    return jobs


def original_checksums():
    sources = [PROJECT / "outputs/checkpoints/main" / f"{prefix}{i}.{suffix}"
               for _, prefix, _ in DATASETS for i in range(1, 11) for suffix in ("pt", "npz")]
    sources += [PROJECT / "outputs/checkpoints/task" / f"I{i}.{suffix}"
                for i in range(1, 11) for suffix in ("pt", "npz")]
    return {str(p.relative_to(PROJECT)): sha256_file(p) for p in sorted(sources)}


def prepare():
    gate = require_completed_pilot()
    prereg_path = ROOT / "expanded_preregistration.json"
    existing = json.loads(prereg_path.read_text()) if prereg_path.exists() else None
    created = existing["created_at"] if existing else stamp()
    # A later validation rerun changes its timestamp/hash; the preregistration
    # retains the original successful gate receipt rather than rewriting it.
    gate_record = existing["pilot_gate"] if existing else gate
    prereg = {
        "created_at": created, "stage": "expanded_screening", "pilot_informed": True,
        "selection": "Full Cartesian matrix fixed after completed pilot; no inclusion/exclusion depends on observed effect or return",
        "pilot_gate": gate_record,
        "datasets": [d for d, _, _ in DATASETS], "checkpoints": list(CHECKPOINTS),
        "layers": list(LAYERS), "epsilon": [0., *EPSILONS], "momentum": list(MOMENTUM),
        "source_seeds": {d: list(range(s + 1, s + 11)) for d, _, s in DATASETS},
        "directions_per_nonzero_cell": 1,
        "direction_limitation": "One direction per source/checkpoint estimates seed-averaged screening behavior; it does not characterize within-network directional uncertainty or establish a multidirectional basin boundary. Pilot retains five directions.",
        "direction_pairing": "Same normalized Gaussian W0/W1 directions paired across epsilon, layer and momentum; perturb_seed=61000000+seed*10000+checkpoint*100+direction",
        "independent_reference_seeds": {"mnist": list(range(401, 411)),
                                        "fashion_mnist": list(range(1201, 1211)),
                                        "kmnist": list(range(1301, 1311))},
        "reference_source_runs": new_reference_specs(),
        "reference_continuations": 90, "reference_models_per_task": 10,
        "baseline_runs_per_dataset_checkpoint": 20, "perturbed_runs_per_dataset_checkpoint": 240,
        "total_baselines": 180, "total_perturbed": 2160, "total_forks": 2340,
        "recovery_epochs": 10, "training_samples": 60000, "test_samples": 10000,
        "original_training_rule": "784-64-10 ReLU; fixed zero biases; SGD lr0.1 momentum0.9 weight_decay0; batch256",
        "threshold_freezing": "Ten independent same-task references at every matched recovery update; all nine threshold files frozen before any expanded perturbation training",
        "control_c": "Different-future-batch-order control remains in the five-direction MNIST pilot; extension uses identical paired future batches",
        "inference": "Pointwise seed-level bootstrap intervals, ten seeds; full factorial screening is exploratory and intervals are not familywise multiplicity-adjusted",
        "local_family": "Prespecified operational local-distance radius from independent references; neither exact solution identity nor strict mathematical attractor",
        "planned_outputs": [str(folder.relative_to(ROOT)) for *_, folder in cells()],
    }
    freeze(prereg_path, prereg)
    freeze(ROOT / "provenance/expanded_source_checksums_before.json", original_checksums())
    pilot_config = json.loads((ROOT / "config.json").read_text())
    all_jobs = []
    for dataset, prefix, seedbase, checkpoint, folder in cells():
        for name in ("checkpoints", "receipts", "validation", "provenance"):
            (folder / name).mkdir(parents=True, exist_ok=True)
        config = {**pilot_config, "created_at": created, "stage": "expanded_screening",
                  "datasets": [dataset], "source_runs": [f"{prefix}{i}" for i in range(1, 11)],
                  "checkpoint": checkpoint, "epsilon": [0., *EPSILONS], "layers": list(LAYERS),
                  "directions": 1, "momentum": list(MOMENTUM),
                  "control_c": "Conducted separately in MNIST pilot; all expanded arms use paired identical future batches",
                  "scope": "Pilot-informed prospectively frozen full factorial screening, ten independent source seeds and one paired perturbation direction per source/checkpoint",
                  "direction_limitation": prereg["direction_limitation"],
                  "expanded_preregistration_sha256": sha256_file(prereg_path),
                  "data_dir": str(PROJECT / "data"),
                  "source_root": str(PROJECT),
                  "inference": prereg["inference"]}
        references = [reference_job(dataset, prefix, seedbase, checkpoint, i) for i in range(1, 11)]
        jobs = cell_jobs(dataset, prefix, seedbase, checkpoint)
        freeze(folder / "config.json", config)
        freeze(folder / "reference_jobs.json", references)
        freeze(folder / "pilot_jobs.json", jobs)
        pd.DataFrame(jobs).to_csv(folder / "perturbation_manifest.csv", index=False)
        all_jobs.extend({"output_root": str(folder), **j} for j in jobs)
    pd.DataFrame(all_jobs).to_csv(ROOT / "expanded_perturbation_manifest.csv", index=False)
    print(json.dumps({"prepared_cells": 9, "forks": len(all_jobs), "reference_continuations": 90,
                      "preregistration": str(prereg_path)}), flush=True)
    return prereg


def _source_worker(spec):
    from learning_trajectory.training import RunSpec, TrainingConfig, train_run
    with threadpool_limits(limits=1):
        return train_run(RunSpec(**spec), PROJECT / "data", ROOT / "reference_sources", TrainingConfig(epochs=10, threads=1))


def _fork_worker(argument):
    from training import train_job
    job, folder = argument
    return train_job(job, folder)


def parallel_map(worker, arguments, workers, phase):
    if workers < 1:
        raise ValueError("workers must be positive")
    records = []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(worker, argument) for argument in arguments]
        for count, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records.append(record)
            print(json.dumps({"phase": phase, "completed": count, "total": len(arguments),
                              "elapsed_seconds": round(time.monotonic() - started, 1), "result": record}), flush=True)
    return records


def references(workers):
    source_records = parallel_map(_source_worker, new_reference_specs(), workers, "reference_sources")
    _atomic_json(ROOT / "reference_sources/manifests/heldout.json",
                 {"complete": len(source_records) == 20, "expected_runs": 20,
                  "runs": sorted(source_records, key=lambda r: r["run_id"])})
    arguments = [(job, str(folder)) for *_, folder in cells()
                 for job in json.loads((folder / "reference_jobs.json").read_text())]
    records = parallel_map(_fork_worker, arguments, workers, "reference_continuations")
    if any(r["status"] != "complete" for r in records):
        raise RuntimeError("Independent reference cohort contains numerical failures; cannot calibrate")


def calibrate():
    from metrics import threshold_calibration
    with threadpool_limits(limits=1):
        for *_, folder in cells():
            jobs = json.loads((folder / "reference_jobs.json").read_text())
            threshold_path = folder / "region_thresholds.json"
            perturb_ids = {j["job_id"] for j in json.loads((folder / "pilot_jobs.json").read_text())}
            if not threshold_path.exists() and any((folder / "receipts" / f"{key}.json").exists() for key in perturb_ids):
                raise RuntimeError(f"Cannot freeze thresholds retrospectively after expanded forks: {folder}")
            arrays = []
            hashes = {}
            for job in jobs:
                receipt = json.loads((folder / "receipts" / f"{job['job_id']}.json").read_text())
                path = folder / "checkpoints" / f"{job['job_id']}.npz"
                digest = sha256_file(path)
                if receipt["status"] != "complete" or digest != receipt["npz_sha256"]:
                    raise RuntimeError(f"Incomplete or corrupt reference: {path}")
                hashes[job["job_id"]] = digest
                with np.load(path, allow_pickle=False) as z:
                    arrays.append({k: z[k].copy() for k in ("W0", "W1", "accuracy", "loss", "recovery_updates")})
            updates = arrays[0]["recovery_updates"]
            if not all(np.array_equal(a["recovery_updates"], updates) for a in arrays):
                raise AssertionError("Reference snapshot schedules differ")
            thresholds = []
            for k, update in enumerate(updates):
                threshold = threshold_calibration([(a["W0"][k], a["W1"][k]) for a in arrays],
                                                   [float(a["accuracy"][k]) for a in arrays],
                                                   [float(a["loss"][k]) for a in arrays])
                threshold.update(recovery_update=int(update),
                    local_radius=.1 * float(np.median([r["loo_median_distance"] for r in threshold["per_reference"]])))
                thresholds.append(threshold)
            if threshold_path.exists():
                previous = json.loads(threshold_path.read_text())
                if previous["reference_checksums"] != hashes or previous["thresholds"] != thresholds:
                    raise ValueError(f"Frozen reference calibration changed: {folder}")
            else:
                freeze(threshold_path, {"created_at": stamp(), "frozen_before_perturbation_training": True,
                    "reference_ids": [j["job_id"] for j in jobs], "reference_checksums": hashes,
                    "thresholds": thresholds})
            print(json.dumps({"thresholds_frozen": str(threshold_path), "checkpoints": len(thresholds),
                              "final_distance_q95": thresholds[-1]["distance_q95"]}), flush=True)
    freeze(ROOT / "expanded_threshold_checksums.json",
           {str((folder / "region_thresholds.json").relative_to(ROOT)): sha256_file(folder / "region_thresholds.json")
            for *_, folder in cells()})


def train(workers, selected_cell=None, max_new_jobs=None):
    expected = json.loads((ROOT / "expanded_threshold_checksums.json").read_text())
    for path, digest in expected.items():
        if sha256_file(ROOT / path) != digest:
            raise RuntimeError(f"Frozen expanded threshold changed: {path}")
    if len(expected) != 9:
        raise RuntimeError("Freeze all nine reference criteria before expanded perturbations")
    selected=[folder for *_,folder in cells() if selected_cell is None or folder.name==selected_cell]
    if not selected:raise ValueError(f"Unknown registered cell: {selected_cell}")
    arguments = [(job, str(folder)) for folder in selected
                 for job in json.loads((folder / "pilot_jobs.json").read_text())]
    if max_new_jobs is not None:
        if max_new_jobs<1:raise ValueError("max_new_jobs must be positive")
        arguments=[(job,folder) for job,folder in arguments if not (Path(folder)/"receipts"/f"{job['job_id']}.json").exists()][:max_new_jobs]
    parallel_map(_fork_worker, arguments, workers, "expanded_forks")
    records=[]
    for *_,folder in cells():
        for job in json.loads((folder / "pilot_jobs.json").read_text()):
            receipt=folder/"receipts"/f"{job['job_id']}.json"
            if receipt.exists():records.append(json.loads(receipt.read_text()))
    _atomic_json(ROOT / "expanded_training_status.json",
                 {"completed_at": stamp(), "complete": len(records) == 2340,
                  "expected_runs": 2340, "completed_runs": len(records),
                  "numerical_failures": sum(r["status"] != "complete" for r in records)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "references", "calibrate", "train"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cell", choices=[folder.name for *_,folder in cells()])
    parser.add_argument("--max-new-jobs",type=int,help="Execute a bounded batch of unfinished jobs; preserves every registered condition")
    args = parser.parse_args()
    prepare()
    if args.phase == "references":
        references(args.workers)
    elif args.phase == "calibrate":
        calibrate()
    elif args.phase == "train":
        train(args.workers,args.cell,args.max_new_jobs)


if __name__ == "__main__":
    main()
