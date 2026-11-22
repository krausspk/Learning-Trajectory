"""Independent checks of fork execution, symmetry, and original preservation."""
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
import torch
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / "src"))

from learning_trajectory.checkpoints import restore_checkpoint
from learning_trajectory.data import _atomic_json, sha256_file
from metrics import distance
from training import _arrays, train_job


def benchmark_job(job_id, checkpoint=20, epochs=10):
    job = {"job_id": job_id, "run_id": "M1", "source": "outputs/checkpoints/main/M1.pt",
           "dataset": "mnist", "seed": 101, "checkpoint": checkpoint, "momentum": "preserve",
           "epsilon": 0.0, "layer": "both", "direction": 0, "perturb_seed": 200001, "order_seed": 101}
    if epochs != 10:
        job["recovery_epochs"] = epochs
    return job


def check_real_replay():
    receipt = train_job(benchmark_job("replay_M1_E2_to_E4", 2, 1), ROOT / "validation/benchmark")
    with np.load(receipt["npz_path"]) as new, np.load(PROJECT / "outputs/checkpoints/main/M1.npz") as original:
        checks = {key: bool(np.array_equal(new[key][-1], original[key][4]))
                  for key in ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss", "bias0", "bias1")}
    return {"from_checkpoint": 2, "to_checkpoint": 4, "updates": 235,
            "bitwise_checks": checks, "passed": all(checks.values())}


def check_zero_intervention_repeat():
    first = train_job(benchmark_job("benchmark_M1_E20_preserve"), ROOT / "validation/benchmark")
    second = train_job(benchmark_job("benchmark_M1_E20_preserve_repeat"), ROOT / "validation/benchmark")
    with np.load(first["npz_path"]) as a, np.load(second["npz_path"]) as b:
        checks = {key: bool(np.array_equal(a[key], b[key])) for key in a.files if key != "metadata"}
        am, bm = json.loads(str(a["metadata"])), json.loads(str(b["metadata"]))
        checks["epoch_order_hashes"] = am["epoch_order_hashes"] == bm["epoch_order_hashes"]
    return {"updates": 2350, "checkpoints": first["checkpoints"],
            "bitwise_checks": checks, "passed": all(checks.values())}


def check_independent_buffers():
    path = PROJECT / "outputs/checkpoints/main/M1.pt"
    a, oa, _ = restore_checkpoint(path, 20)
    b, ob, _ = restore_checkpoint(path, 20)
    checks = {}
    with torch.no_grad():
        for index in (0, 2):
            pa, pb = a[index].weight, b[index].weight
            ma = oa.state[pa]["momentum_buffer"]
            mb = ob.state[pb]["momentum_buffer"]
            saved_b, saved_m = pb.clone(), mb.clone()
            checks[f"layer{index}_initial_weights_equal"] = torch.equal(pa, pb)
            checks[f"layer{index}_initial_momentum_equal"] = torch.equal(ma, mb)
            checks[f"layer{index}_weight_storage_independent"] = pa.data_ptr() != pb.data_ptr()
            checks[f"layer{index}_momentum_storage_independent"] = ma.data_ptr() != mb.data_ptr()
            pa.add_(0.01)
            ma.zero_()
            checks[f"layer{index}_second_weights_unchanged"] = torch.equal(saved_b, pb)
            checks[f"layer{index}_second_momentum_unchanged"] = torch.equal(saved_m, mb)
    return {"checks": checks, "passed": all(checks.values())}


def check_symmetries():
    with np.load(PROJECT / "outputs/checkpoints/main/M1.npz") as archive:
        a0, a1 = archive["W0"][20].astype(float), archive["W1"][20].astype(float)
    rng = np.random.default_rng(8172026)
    permutation = rng.permutation(64)
    scales = np.exp(rng.uniform(-2, 2, 64))
    b0, b1 = (a0 * scales)[..., permutation], (a1 / scales[:, None])[permutation]
    before0, before1 = a0.copy(), a1.copy()
    quotient_error = distance((a0, a1), (b0, b1))
    x = _arrays("mnist", str(PROJECT / "data"))[2][:256].numpy().astype(float)
    logits_a = np.maximum(x @ a0, 0) @ a1
    logits_b = np.maximum(x @ b0, 0) @ b1
    logit_error = float(np.max(np.abs(logits_a - logits_b)))
    immutable = np.array_equal(before0, a0) and np.array_equal(before1, a1)
    return {"permutation_and_positive_scaling_distance": quotient_error,
            "maximum_logit_error": logit_error, "analysis_input_unchanged": immutable,
            "tolerance": 1e-10,
            "passed": quotient_error < 1e-10 and logit_error < 1e-10 and immutable}


def check_saved_jobs():
    records = []
    orders = {}
    errors = []
    failed_jobs = []
    same_order_comparisons = 0
    for path in sorted((ROOT / "receipts").glob("*.json")):
        receipt = json.loads(path.read_text())
        job = receipt["job"]
        if receipt["status"] == "failed":
            failed_jobs.append(job["job_id"])
        for suffix in ("npz", "pt"):
            artifact = Path(receipt[suffix + "_path"])
            if sha256_file(artifact) != receipt[suffix + "_sha256"]:
                errors.append(f"{job['job_id']}: corrupt {suffix} hash")
        with np.load(receipt["npz_path"]) as archive:
            metadata = json.loads(str(archive["metadata"]))
            for key in ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss"):
                if not np.isfinite(archive[key]).all():
                    errors.append(f"{job['job_id']}: nonfinite saved {key}")
            if np.any(archive["bias0"]) or np.any(archive["bias1"]):
                errors.append(f"{job['job_id']}: nonzero bias")
            if receipt["status"] == "complete" and not np.array_equal(archive["recovery_updates"], metadata["planned_recovery_updates"]):
                errors.append(f"{job['job_id']}: incomplete checkpoint schedule")
            if job["momentum"] == "reset" and (np.any(archive["momentum_W0"][0]) or np.any(archive["momentum_W1"][0])):
                errors.append(f"{job['job_id']}: initial reset momentum nonzero")
            for label, norm in metadata["perturbation"].items():
                target = float(job["epsilon"]) if job["layer"] in ("both", label) else 0.0
                if abs(norm["actual_relative_norm"] - target) > 1e-7:
                    errors.append(f"{job['job_id']}: incorrect {label} perturbation norm")
            order_key = (job["source"], job["checkpoint"], job["order_seed"], job.get("recovery_epochs", 10))
            if order_key in orders:
                same_order_comparisons += 1
                if metadata["epoch_order_hashes"] != orders[order_key]:
                    errors.append(f"{job['job_id']}: paired future batch-order mismatch")
            else:
                orders[order_key] = metadata["epoch_order_hashes"]
            if sha256_file(metadata["source"]) != metadata["source_sha256"]:
                errors.append(f"{job['job_id']}: source changed")
        records.append({"job_id": job["job_id"], "status": receipt["status"]})
    return {"checked_receipts": len(records), "same_order_comparisons": same_order_comparisons,
            "failed_jobs": failed_jobs, "numerical_failure_count": len(failed_jobs),
            "errors": errors, "passed": not errors}


def check_originals():
    original = json.loads((ROOT / "provenance/source_checksums_before.json").read_text())
    changed = [p for p, expected in original.items() if sha256_file(PROJECT / p) != expected]
    return {"checked_files": len(original), "changed": changed, "passed": not changed}


def main():
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        checks = {"real_continuation": check_real_replay(),
                  "epsilon_zero_deterministic_repeat": check_zero_intervention_repeat(),
                  "independent_optimizer_and_model_buffers": check_independent_buffers(),
                  "symmetry_sanity": check_symmetries(),
                  "saved_forks": check_saved_jobs(),
                  "originals_unchanged": check_originals()}
    result = {"checked_at": datetime.now(ZoneInfo("Europe/Berlin")).isoformat(),
              "passed": all(v["passed"] for v in checks.values()), "checks": checks}
    _atomic_json(ROOT / "validation/checks.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
