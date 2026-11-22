"""Deterministic CPU training for all three experimental cohorts.

This is a reconstruction of the published methods, not the authors' unpublished
training code. The paper leaves the per-epoch seed derivation and the odd-length
half-epoch boundary unspecified; both conventions are explicit below and saved
inside every archive. All requested runs use the full official training sets.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import hashlib
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import tempfile
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .data import NORMALIZATION, _atomic_json, _atomic_npz, ensure_dataset, load_dataset, sha256_file


EARLY_UPDATES = (0, 1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 235, 352, 470)
IMPLEMENTATION_VERSION = "paper-reconstruction-1"
CONVENTIONS = {
    "initialization": "torch.manual_seed(seed); nn.Linear with bias=True; construct both layers; zero and freeze both biases",
    "shuffle": "torch.randperm(N, generator=torch.Generator().manual_seed(seed + zero_based_epoch))",
    "half_epoch": "ceil(number_of_minibatches / 2), i.e. update 118 of each 235-update epoch",
    "task_pairing": "same seed and first Linear construction give bitwise identical W0; image order depends only on seed and epoch",
    "evaluation": "unchanged test partition, sequential batches, sum cross entropy divided by test count",
    "numeric_precision": "CPU float32 training; stored weights float32; losses/accuracy float64 summaries",
    "unpublished_details": "Original loader seed formula, half-epoch boundary, and exact initialization code were not supplied in the paper; conventions above are explicit reconstruction choices.",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    cohort: str
    dataset: str
    task: str
    seed: int
    block: int | None = None


@dataclass(frozen=True)
class TrainingConfig:
    hidden_width: int = 64
    batch_size: int = 256
    learning_rate: float = 0.1
    momentum: float = 0.9
    epochs: int = 10
    test_batch_size: int = 2048
    threads: int = 1


def cohort_specs(cohort: str = "all") -> list[RunSpec]:
    """The exact 30 + 20 + 15 independent training runs specified in the paper."""
    if cohort not in ("all", "main", "task", "early"):
        raise ValueError(f"Unknown cohort {cohort!r}")
    specs = []
    if cohort in ("all", "main"):
        for dataset, prefix, seed_base in (("mnist", "M", 100), ("fashion_mnist", "F", 200), ("kmnist", "K", 300)):
            specs.extend(RunSpec(f"{prefix}{i}", "main", dataset, "identity", seed_base + i) for i in range(1, 11))
    if cohort in ("all", "task"):
        for i in range(1, 11):
            for task, prefix in (("identity", "I"), ("parity", "P")):
                specs.append(RunSpec(f"{prefix}{i}", "task", "mnist", task, 400 + i, 400 + i))
    if cohort in ("all", "early"):
        for dataset, prefix, seed_base in (("mnist", "M", 700), ("fashion_mnist", "F", 800), ("kmnist", "K", 900)):
            specs.extend(RunSpec(f"{prefix}{i}", "early", dataset, "identity", seed_base + i) for i in range(1, 6))
    return specs


def build_model(seed: int, output_size: int = 10, hidden_width: int = 64) -> nn.Sequential:
    """Preserve PyTorch default weight initialization, then freeze zero biases."""
    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(784, hidden_width), nn.ReLU(), nn.Linear(hidden_width, output_size))
    for layer in (model[0], model[2]):
        nn.init.zeros_(layer.bias)
        layer.bias.requires_grad_(False)
    return model


def image_order(seed: int, epoch: int, n_samples: int) -> torch.Tensor:
    return torch.randperm(n_samples, generator=torch.Generator().manual_seed(seed + epoch))


def checkpoint_schedule(cohort: str, config: TrainingConfig, n_train: int = 60000) -> tuple[np.ndarray, np.ndarray]:
    batches = math.ceil(n_train / config.batch_size)
    if cohort == "early":
        if batches * config.epochs < EARLY_UPDATES[-1]:
            raise ValueError("The early cohort requires at least 470 updates for its specified checkpoints")
        updates = np.asarray(EARLY_UPDATES, dtype=np.int64)
        return updates.copy(), updates
    updates = [0]
    for epoch in range(config.epochs):
        updates.extend((epoch * batches + math.ceil(batches / 2), (epoch + 1) * batches))
    if len(set(updates)) != len(updates):
        raise ValueError("At least two batches per epoch are needed for half-epoch checkpoints")
    return np.arange(len(updates), dtype=np.int64), np.asarray(updates, dtype=np.int64)


def _configure_threads(threads: int) -> None:
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # This setting can only be applied once per process.
    torch.use_deterministic_algorithms(True)


@torch.inference_mode()
def evaluate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, batch_size: int = 2048) -> tuple[float, float]:
    model.eval()
    total_loss, correct = 0.0, 0
    for start in range(0, len(images), batch_size):
        logits = model(images[start : start + batch_size])
        target = labels[start : start + batch_size]
        total_loss += float(F.cross_entropy(logits, target, reduction="sum"))
        correct += int((logits.argmax(1) == target).sum())
    model.train()
    return correct / len(images), total_loss / len(images)


def train_arrays(spec: RunSpec, arrays: tuple[torch.Tensor, ...], config: TrainingConfig, full_states: list[dict] | None = None) -> dict[str, np.ndarray]:
    """Train one complete trajectory; injection of arrays supports fast unit tests."""
    _configure_threads(config.threads)
    random.seed(spec.seed)
    np.random.seed(spec.seed)
    x_train, y_train, x_test, y_test = arrays
    if x_train.dtype != torch.float32 or x_test.dtype != torch.float32:
        raise ValueError("Training and test image tensors must be float32")
    if spec.task == "parity":
        y_train, y_test = y_train.remainder(2), y_test.remainder(2)
    output_size = 2 if spec.task == "parity" else 10
    model = build_model(spec.seed, output_size, config.hidden_width)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, momentum=config.momentum)
    checkpoints, requested_updates = checkpoint_schedule(spec.cohort, config, len(x_train))
    targets = set(requested_updates.tolist())
    collected = {key: [] for key in ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss", "bias0", "bias1")}
    current_epoch, current_batch_position, update = 0, 0, 0
    order_generator = torch.Generator().manual_seed(spec.seed)
    current_order = torch.randperm(len(x_train), generator=order_generator)

    def save_checkpoint() -> None:
        accuracy, loss = evaluate(model, x_test, y_test, config.test_batch_size)
        for label, layer in (("W0", model[0]), ("W1", model[2])):
            collected[label].append(layer.weight.detach().numpy().T.copy())
            buffer = optimizer.state.get(layer.weight, {}).get("momentum_buffer", torch.zeros_like(layer.weight))
            collected["momentum_" + label].append(buffer.detach().numpy().T.copy())
        collected["bias0"].append(model[0].bias.detach().numpy().copy())
        collected["bias1"].append(model[2].bias.detach().numpy().copy())
        collected["accuracy"].append(accuracy)
        collected["loss"].append(loss)
        if full_states is not None:
            full_states.append({
                "checkpoint": int(checkpoints[len(full_states)]),
                "update": update,
                "model_state_dict": copy.deepcopy(model.state_dict()),
                "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                "torch_rng_state": torch.get_rng_state().clone(),
                "numpy_rng_state": copy.deepcopy(np.random.get_state()),
                "python_rng_state": random.getstate(),
                "epoch": current_epoch,
                "batch_position": current_batch_position,
                "samples_consumed_this_epoch": min(current_batch_position * config.batch_size, len(x_train)),
                "epoch_order": current_order.clone(),
                "order_generator_state": order_generator.get_state().clone(),
                "order_seed": spec.seed + current_epoch,
                "config": asdict(config),
                "spec": asdict(spec),
            })

    save_checkpoint()
    for epoch in range(config.epochs):
        current_epoch = epoch
        current_batch_position = 0
        order_generator = torch.Generator().manual_seed(spec.seed + epoch)
        current_order = torch.randperm(len(x_train), generator=order_generator)
        for start in range(0, len(x_train), config.batch_size):
            selection = current_order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train[selection]), y_train[selection])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss in {spec.run_id} at update {update}")
            loss.backward()
            optimizer.step()
            update += 1
            current_batch_position += 1
            if torch.count_nonzero(model[0].bias) or torch.count_nonzero(model[2].bias):
                raise AssertionError("A fixed zero bias changed during training")
            if update in targets:
                save_checkpoint()
    if len(collected["W0"]) != len(checkpoints):
        raise AssertionError("Incomplete checkpoint schedule")
    result = {key: np.asarray(values) for key, values in collected.items()}
    result["checkpoints"] = checkpoints
    result["updates"] = requested_updates
    return result


def _config_hash(spec: RunSpec, config: TrainingConfig) -> str:
    value = {"spec": asdict(spec), "config": asdict(config), "conventions": CONVENTIONS, "version": IMPLEMENTATION_VERSION}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def validate_archive(path: str | Path, spec: RunSpec, config: TrainingConfig) -> dict:
    """Fully validate a finished archive before accepting it for resume/analysis."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        if metadata.get("config_hash") != _config_hash(spec, config):
            raise ValueError(f"Stored run configuration differs: {path}")
        checkpoints, updates = checkpoint_schedule(spec.cohort, config)
        for name, expected in (("checkpoints", checkpoints), ("updates", updates)):
            if not np.array_equal(archive[name], expected):
                raise ValueError(f"Wrong {name} in {path}")
        n = len(checkpoints)
        outputs = 2 if spec.task == "parity" else 10
        for name, shape in (("W0", (n, 784, config.hidden_width)), ("W1", (n, config.hidden_width, outputs)), ("accuracy", (n,)), ("loss", (n,))):
            if archive[name].shape != shape or not np.isfinite(archive[name]).all():
                raise ValueError(f"Invalid {name} in {path}")
        if np.any((archive["accuracy"] < 0) | (archive["accuracy"] > 1)) or np.any(archive["loss"] < 0):
            raise ValueError(f"Invalid performance values in {path}")
        if np.any(archive["bias0"]) or np.any(archive["bias1"]):
            raise ValueError(f"Nonzero frozen bias in {path}")
        accuracy = float(archive["accuracy"][-1])
        loss = float(archive["loss"][-1])
    state_path = path.with_suffix(".pt")
    if not state_path.exists():
        raise ValueError(f"Full optimizer/RNG checkpoint archive missing: {state_path}")
    state_hash = sha256_file(state_path)
    if state_hash != metadata.get("full_state_sha256"):
        raise ValueError(f"Full optimizer/RNG archive hash mismatch: {state_path}")
    return {**asdict(spec), "path": str(path.resolve()), "sha256": sha256_file(path), "full_state_path": str(state_path.resolve()), "full_state_sha256": state_hash, "checkpoints": n, "final_accuracy": accuracy, "final_loss": loss, "config_hash": metadata["config_hash"]}


@lru_cache(maxsize=3)
def _cached_dataset(name: str, data_dir: str) -> tuple[torch.Tensor, ...]:
    return load_dataset(name, data_dir)


def train_run(spec: RunSpec, data_dir: str | Path = "data", output_dir: str | Path = "outputs", config: TrainingConfig | None = None, force: bool = False) -> dict:
    config = config or TrainingConfig(epochs=2 if spec.cohort == "early" else 10)
    path = Path(output_dir) / "checkpoints" / spec.cohort / f"{spec.run_id}.npz"
    if path.exists() and not force:
        record = validate_archive(path, spec, config)
        record["resumed"] = True
        return record
    _configure_threads(config.threads)
    started = time.monotonic()
    arrays = _cached_dataset(spec.dataset, str(Path(data_dir).resolve()))
    full_states = []
    trajectory = train_arrays(spec, arrays, config, full_states=full_states)
    state_path = path.with_suffix(".pt")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=state_path.stem + ".", suffix=".pt", dir=state_path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save({"implementation_version": IMPLEMENTATION_VERSION, "conventions": CONVENTIONS, "checkpoints": full_states}, stream)
        os.replace(temporary, state_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    metadata = {
        **asdict(spec), "config": asdict(config), "config_hash": _config_hash(spec, config),
        "implementation_version": IMPLEMENTATION_VERSION, "conventions": CONVENTIONS,
        "normalization": NORMALIZATION[spec.dataset], "training_samples": len(arrays[0]), "test_samples": len(arrays[2]),
        "training_source_sha256": sha256_file(__file__), "data_source_sha256": sha256_file(Path(__file__).with_name("data.py")),
        "dataset_manifest_sha256": sha256_file(Path(data_dir) / spec.dataset / "manifest.json"),
        "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__,
        "elapsed_seconds": time.monotonic() - started,
        "optimizer_state": "momentum_W0 and momentum_W1 saved at every checkpoint; zero before first update",
        "epoch_shuffle_seeds": [spec.seed + epoch for epoch in range(config.epochs)],
        "full_state_path": str(state_path.resolve()), "full_state_sha256": sha256_file(state_path),
    }
    trajectory["metadata"] = np.asarray(json.dumps(metadata, sort_keys=True))
    _atomic_npz(path, **trajectory)
    record = validate_archive(path, spec, config)
    record["elapsed_seconds"] = metadata["elapsed_seconds"]
    record["resumed"] = False
    return record


def _worker(args: tuple) -> dict:
    return train_run(*args)


def train_cohort(cohort: str = "all", data_dir: str | Path = "data", output_dir: str | Path = "outputs", workers: int = 4, threads: int = 1, force: bool = False, run_ids: list[str] | None = None) -> dict:
    """Run or resume a cohort, writing manifests after every completed trajectory."""
    specs = cohort_specs(cohort)
    if run_ids:
        requested = set(run_ids)
        specs = [s for s in specs if s.run_id in requested or f"{s.cohort}/{s.run_id}" in requested]
        if not specs:
            raise ValueError("No requested run IDs matched the selected cohort")
    for dataset in sorted({spec.dataset for spec in specs}):
        ensure_dataset(dataset, data_dir)
    records = []
    manifests_dir = Path(output_dir) / "manifests"

    def write_manifests() -> None:
        for name in sorted({s.cohort for s in specs}):
            expected = [s for s in specs if s.cohort == name]
            actual = sorted([r for r in records if r["cohort"] == name], key=lambda r: r["run_id"])
            _atomic_json(manifests_dir / f"{name}.json", {"cohort": name, "expected_runs": len(cohort_specs(name)), "requested_runs": len(expected), "complete": len(actual) == len(cohort_specs(name)), "runs": actual, "conventions": CONVENTIONS})

    arguments = [(spec, str(data_dir), str(output_dir), TrainingConfig(epochs=2 if spec.cohort == "early" else 10, threads=threads), force) for spec in specs]
    if workers == 1:
        iterator = map(_worker, arguments)
        for record in iterator:
            records.append(record)
            write_manifests()
            print(json.dumps({"completed": len(records), "total": len(specs), **record}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = {executor.submit(_worker, arguments): arguments[0] for arguments in arguments}
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                write_manifests()
                print(json.dumps({"completed": len(records), "total": len(specs), **record}), flush=True)
    if cohort in ("all", "task") and not run_ids:
        for i in range(1, 11):
            folder = Path(output_dir) / "checkpoints" / "task"
            with np.load(folder / f"I{i}.npz") as identity, np.load(folder / f"P{i}.npz") as parity:
                if not np.array_equal(identity["W0"][0], parity["W0"][0]):
                    raise AssertionError(f"Paired initialization failed for seed block {400 + i}")
    return {"cohort": cohort, "completed": len(records), "runs": records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort", choices=("all", "main", "task", "early"), default="all")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output", default="outputs")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--run-ids", nargs="+", help="Optional selected runs; omitted for the complete reproduction")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    train_cohort(args.cohort, args.data_dir, args.output, args.workers, args.threads, args.force, args.run_ids)


if __name__ == "__main__":
    main()
