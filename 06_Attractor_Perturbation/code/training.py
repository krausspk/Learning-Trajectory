"""Deterministic, resumable raw-gauge perturbation forks.

Original checkpoint archives are read only. Canonicalization belongs exclusively
to the analysis module and is deliberately absent from this training code.
"""
from __future__ import annotations

import copy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
import time

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from learning_trajectory.checkpoints import restore_checkpoint
from learning_trajectory.data import _atomic_json, _atomic_npz, load_dataset, sha256_file
from learning_trajectory.training import _configure_threads, evaluate, image_order


PROJECT = Path(__file__).resolve().parents[2]
VERSION = "perturbation-fork-v1"


@lru_cache(maxsize=3)
def _arrays(dataset: str, data_dir: str):
    return load_dataset(dataset, data_dir)


def perturb_weights(model, epsilon: float, layer: str, perturb_seed: int) -> dict:
    """Inject independently normalized Gaussian matrices without changing RNG.

    Both matrices' directions are always drawn, including in single-layer arms,
    so matched direction seeds really do share their W0/W1 noise directions.
    """
    if epsilon < 0 or not np.isfinite(epsilon):
        raise ValueError("epsilon must be finite and nonnegative")
    if layer not in ("both", "W0", "W1"):
        raise ValueError(f"Unknown layer: {layer}")
    rng = np.random.Generator(np.random.PCG64(perturb_seed))
    records = {}
    with torch.no_grad():
        for label, module in (("W0", model[0]), ("W1", model[2])):
            original = module.weight.detach().clone()
            # Draw in archive (input, output) orientation, then transpose.
            noise = rng.standard_normal(tuple(reversed(original.shape)))
            norm = float(np.linalg.norm(original.numpy().astype(np.float64)))
            active = epsilon if layer in ("both", label) else 0.0
            delta = noise * (active * norm / np.linalg.norm(noise))
            module.weight.add_(torch.from_numpy(delta.T.copy()).to(original.dtype))
            actual = module.weight.detach().numpy().astype(np.float64) - original.numpy()
            records[label] = {
                "norm_before": norm,
                "requested_relative_norm": active,
                "actual_relative_norm": float(np.linalg.norm(actual) / norm),
            }
    return records


def reset_momentum(optimizer) -> None:
    for state in optimizer.state.values():
        if "momentum_buffer" in state:
            state["momentum_buffer"].zero_()


def recovery_schedule(n_train: int, batch_size: int, epochs: int = 10) -> list[int]:
    batches = math.ceil(n_train / batch_size)
    end = batches * epochs
    dense = (0, 1, 10, 50, math.ceil(batches / 2))
    return sorted({s for s in dense if s <= end} | {batches * e for e in range(1, epochs + 1)})


def continue_arrays(model, optimizer, state: dict, arrays: tuple, *, order_seed: int,
                    recovery_epochs: int = 10) -> tuple[dict, dict, dict | None]:
    """Continue a completed-epoch checkpoint; injectable arrays support tests.

    Returns trajectory, final full state, and a failure record or None. Only
    numerical divergence is converted into a recorded experimental failure;
    programming/configuration/IO errors propagate and cannot look like a run.
    """
    if recovery_epochs < 1:
        raise ValueError("recovery_epochs must be positive")
    x_train, y_train, x_test, y_test = arrays
    cfg = state["config"]
    batch = cfg["batch_size"]
    n_batches = math.ceil(len(x_train) / batch)
    if state["batch_position"] != n_batches:
        raise ValueError("This protocol starts only at completed-epoch checkpoints")
    if state["spec"]["task"] != "identity":
        raise ValueError("The perturbation protocol uses the identity classification task")
    first_epoch = int(state["epoch"]) + 1
    schedule = recovery_schedule(len(x_train), batch, recovery_epochs)
    targets = set(schedule)
    collected = {k: [] for k in ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss", "bias0", "bias1", "recovery_updates")}
    update, epoch, batch_position = 0, first_epoch - 1, n_batches
    order = state["epoch_order"].clone()
    order_hashes = []
    failure = None

    def snapshot():
        parameters = (model[0].weight, model[2].weight)
        if not all(bool(torch.isfinite(p).all()) for p in parameters):
            raise FloatingPointError("nonfinite_weights")
        accuracy, loss = evaluate(model, x_test, y_test, cfg["test_batch_size"])
        if not np.isfinite(loss):
            raise FloatingPointError("nonfinite_test_loss")
        for label, module in (("W0", model[0]), ("W1", model[2])):
            collected[label].append(module.weight.detach().numpy().T.copy())
            momentum = optimizer.state.get(module.weight, {}).get("momentum_buffer", torch.zeros_like(module.weight))
            collected["momentum_" + label].append(momentum.detach().numpy().T.copy())
        collected["bias0"].append(model[0].bias.detach().numpy().copy())
        collected["bias1"].append(model[2].bias.detach().numpy().copy())
        collected["accuracy"].append(accuracy)
        collected["loss"].append(loss)
        collected["recovery_updates"].append(update)

    try:
        snapshot()
        for epoch in range(first_epoch, first_epoch + recovery_epochs):
            order = image_order(order_seed, epoch, len(x_train))
            order_hashes.append({"epoch": epoch, "seed": order_seed + epoch,
                                 "sha256": hashlib.sha256(order.numpy().tobytes()).hexdigest()})
            batch_position = 0
            for start in range(0, len(x_train), batch):
                select = order[start:start + batch]
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(x_train[select]), y_train[select])
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite_training_loss")
                loss.backward()
                optimizer.step()
                update += 1
                batch_position += 1
                if update in targets:
                    snapshot()
    except FloatingPointError as error:
        failure = {"kind": str(error), "recovery_update": update, "epoch": epoch,
                   "batch_position": batch_position}
    if bool(torch.count_nonzero(model[0].bias)) or bool(torch.count_nonzero(model[2].bias)):
        raise AssertionError("A frozen bias changed")
    trajectory = {name: np.asarray(values) for name, values in collected.items()}
    final_generator = torch.Generator().manual_seed(order_seed + epoch)
    if order_hashes:
        torch.randperm(len(x_train), generator=final_generator)
    final = {
        "checkpoint": int(state["checkpoint"]) + 2 * recovery_epochs,
        "update": int(state["update"]) + update,
        "recovery_update": update,
        "model_state_dict": copy.deepcopy(model.state_dict()),
        "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
        "torch_rng_state": torch.get_rng_state().clone(),
        "numpy_rng_state": copy.deepcopy(np.random.get_state()),
        "python_rng_state": random.getstate(),
        "epoch": epoch, "batch_position": batch_position,
        "samples_consumed_this_epoch": min(batch_position * batch, len(x_train)),
        "epoch_order": order.clone(),
        "order_seed": order_seed + epoch,
        "order_generator_state": final_generator.get_state(),
        "config": {**cfg, "epochs": first_epoch + recovery_epochs},
        "spec": copy.deepcopy(state["spec"]),
        "continuation_order_seed_base": order_seed,
        "continuation_order_convention": "torch.randperm(N, generator=Generator().manual_seed(order_seed_base + zero_based_epoch)); next full epoch is epoch+1",
        "epoch_order_hashes": order_hashes,
        "failure": failure,
    }
    return trajectory, final, failure


def _atomic_torch(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".pt", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def train_job(job: dict, output_root: str | Path) -> dict:
    """Execute/reuse one full fork, returning its machine-readable receipt.

    Optional job keys: recovery_epochs (default10), data_dir, snapshot_source.
    All random choices and all completed snapshot states remain reproducible.
    """
    started = time.monotonic()
    root = Path(output_root).resolve()
    source = Path(job["source"])
    if not source.is_absolute():
        source = PROJECT / source
    source = source.resolve()
    if job["momentum"] not in ("preserve", "reset"):
        raise ValueError("momentum must be preserve or reset")
    job_id = job["job_id"]
    if Path(job_id).name != job_id or job_id in (".", ".."):
        raise ValueError("job_id must be a plain filename stem")
    npz_path = root / "checkpoints" / f"{job_id}.npz"
    pt_path = npz_path.with_suffix(".pt")
    receipt_path = root / "receipts" / f"{job_id}.json"
    data_dir = Path(job.get("data_dir", PROJECT / "data")).resolve()
    dependencies = (Path(__file__), PROJECT / "src/learning_trajectory/checkpoints.py",
                    PROJECT / "src/learning_trajectory/training.py", PROJECT / "src/learning_trajectory/data.py")
    provenance = {"version": VERSION, "job": job, "source": str(source),
                  "source_sha256": sha256_file(source),
                  "source_code_sha256": {str(p.relative_to(PROJECT)): sha256_file(p) for p in dependencies},
                  "dataset_manifest_sha256": sha256_file(data_dir / job["dataset"] / "manifest.json"),
                  "torch_version": torch.__version__, "numpy_version": np.__version__}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if receipt_path.exists():
        previous = json.loads(receipt_path.read_text())
        if previous["fingerprint"] != fingerprint:
            raise ValueError(f"Refusing incompatible cached job {job_id}")
        if sha256_file(npz_path) != previous["npz_sha256"] or sha256_file(pt_path) != previous["pt_sha256"]:
            raise ValueError(f"Corrupt cached fork {job_id}")
        return {**previous, "reused": True}
    if npz_path.exists() or pt_path.exists():
        raise ValueError(f"Incomplete unreceipted output for {job_id}; preserve it and choose a new job ID")
    _configure_threads(1)
    with threadpool_limits(limits=1):
        model, optimizer, state = restore_checkpoint(source, int(job["checkpoint"]))
        if state["spec"]["seed"] != job["seed"] or state["spec"]["dataset"] != job["dataset"]:
            raise ValueError("Job source seed/dataset mismatch")
        cfg = state["config"]
        if (cfg["hidden_width"], cfg["batch_size"], cfg["learning_rate"], cfg["momentum"]) != (64, 256, 0.1, 0.9):
            raise ValueError("Original architecture/SGD parameters do not match this experiment")
        if any(group.get("weight_decay", 0) != 0 for group in optimizer.param_groups):
            raise ValueError("This experiment assumes zero weight decay")
        if job["momentum"] == "reset":
            reset_momentum(optimizer)
        perturbation = perturb_weights(model, float(job["epsilon"]), job["layer"], int(job["perturb_seed"]))
        trajectory, final, failure = continue_arrays(model, optimizer, state,
            _arrays(job["dataset"], str(data_dir)), order_seed=int(job["order_seed"]),
            recovery_epochs=int(job.get("recovery_epochs", 10)))
    metadata = {**provenance, "fingerprint": fingerprint, "perturbation": perturbation,
                "epoch_order_hashes": final["epoch_order_hashes"],
                "start_epoch": state["epoch"] + 1,
                "planned_recovery_updates": recovery_schedule(60000, 256, int(job.get("recovery_epochs", 10))),
                "failure": failure, "elapsed_seconds": time.monotonic() - started}
    _atomic_torch(pt_path, {"implementation_version": VERSION, "checkpoints": [final], "metadata": metadata})
    metadata["final_state_sha256"] = sha256_file(pt_path)
    trajectory["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    _atomic_npz(npz_path, **trajectory)
    receipt = {"job_id": job_id, "job": job, "fingerprint": fingerprint,
               "status": "failed" if failure else "complete", "failure": failure,
               "npz_path": str(npz_path), "pt_path": str(pt_path),
               "npz_sha256": sha256_file(npz_path), "pt_sha256": sha256_file(pt_path),
               "checkpoints": len(trajectory["recovery_updates"]),
               "final_recovery_update": int(final["recovery_update"]),
               "final_accuracy": float(trajectory["accuracy"][-1]) if len(trajectory["accuracy"]) else None,
               "final_loss": float(trajectory["loss"][-1]) if len(trajectory["loss"]) else None,
               "elapsed_seconds": time.monotonic() - started, "reused": False}
    _atomic_json(receipt_path, receipt)
    return receipt
