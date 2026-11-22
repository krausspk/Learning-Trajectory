"""Crossed initialization x independent example-order training experiment.

This isolated extension reads the frozen reconstruction; it never modifies it.
The 50 networks per dataset are crossed repetitions (10 initializations x five
shuffle schedules), NOT 50 independent initializations. Epochs are zero-based.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import platform
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from learning_trajectory import training as baseline
from learning_trajectory.data import IDX_FILES, NORMALIZATION, _atomic_json, _atomic_npz, read_idx, sha256_file

VERSION = "crossed-initialization-shuffle-v1"
SHUFFLE_SEEDS = (11001, 12001, 13001, 14001, 15001)
DATASETS = (("mnist", "M", 100), ("fashion_mnist", "F", 200), ("kmnist", "K", 300))
CONVENTIONS = {
    "design": "3 datasets x 10 stored initializations x 5 crossed shuffle schedules; 150 runs",
    "ordering": "dataset order mnist,fashion_mnist,kmnist; then numeric init_id; then numeric shuffle_id",
    "initialization": "construct original architecture with init_seed; assert bitwise equality to stored original E0; load that E0",
    "shuffle": "torch.randperm(N, generator=torch.Generator().manual_seed(shuffle_seed + zero_based_epoch))",
    "crossing": "the same five shuffle schedules apply to all initialization IDs and datasets; order generation never uses init_seed",
    "epoch_seed_ranges": "11001..11010,12001..12010,13001..13010,14001..14010,15001..15010 are disjoint",
    "half_epoch": "ceil(235/2)=118 updates; 21 checkpoints including E0 and E20",
    "precision": "CPU float32 weights/momentum/training, float64 test-summary arrays",
    "biases": "both biases are exactly zero, frozen and stored at every checkpoint",
    "evaluation": "official test split, sequential batches, mean cross entropy and classification accuracy",
    "resume": "atomic .progress.pt after each checkpoint; restore weights, optimizer, all RNGs, epoch order and position",
}


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    dataset: str
    init_id: int
    shuffle_id: int
    init_seed: int
    shuffle_seed: int
    original_run_id: str


TrainingConfig = baseline.TrainingConfig


def run_specs() -> list[RunSpec]:
    return [RunSpec(f"{prefix}_i{i:02d}_s{s:02d}", dataset, i, s, base + i, seed, f"{prefix}{i}")
            for dataset, prefix, base in DATASETS for i in range(1, 11)
            for s, seed in enumerate(SHUFFLE_SEEDS, 1)]


def digest_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def array_hash(value) -> str:
    a = np.ascontiguousarray(value)
    return hashlib.sha256(a.tobytes()).hexdigest()


def snapshots_hash(value) -> str:
    """Stable content hash independent of pickle storage IDs, including all RNGs."""
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            digest.update(b"torch")
            visit(x.detach().cpu().numpy())
        elif isinstance(x, np.ndarray):
            digest.update(b"ndarray")
            visit(str(x.dtype))
            visit(x.shape)
            digest.update(np.ascontiguousarray(x).tobytes())
        elif isinstance(x, dict):
            digest.update(b"dict")
            for key in sorted(x, key=lambda key: (type(key).__name__, str(key))):
                visit(key)
                visit(x[key])
        elif isinstance(x, (tuple, list)):
            digest.update(type(x).__name__.encode())
            visit(len(x))
            for item in x:
                visit(item)
        else:
            digest.update(json.dumps(x, sort_keys=True).encode() + b";")
    visit(value)
    return digest.hexdigest()


def state_payload(spec, context, states):
    return {"context_fingerprint": context["fingerprint"], "spec": asdict(spec),
            "checkpoints": states, "checkpoints_sha256": snapshots_hash(states)}


def validate_payload(payload, spec, context):
    if payload["context_fingerprint"] != context["fingerprint"] or payload["spec"] != asdict(spec):
        raise ValueError("Full-state/partial-cache provenance mismatch; use --force or a separate output directory")
    if payload.get("checkpoints_sha256") != snapshots_hash(payload["checkpoints"]):
        raise ValueError("Corrupted checkpoint content hash; use --force or a separate output directory")


def image_order(shuffle_seed: int, epoch: int, n: int):
    generator = torch.Generator().manual_seed(shuffle_seed + epoch)
    return torch.randperm(n, generator=generator), generator.get_state().clone()


def _atomic_pt(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            torch.save(value, stream)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model(spec, config, initial):
    model = baseline.build_model(spec.init_seed, 10, config.hidden_width)
    for name, layer in (("W0", model[0]), ("W1", model[2])):
        stored = np.asarray(initial[name], dtype=np.float32)
        if not np.array_equal(layer.weight.detach().numpy().T, stored):
            raise ValueError(f"Default initialization no longer matches original E0: {spec.run_id}/{name}")
        with torch.no_grad():
            layer.weight.copy_(torch.from_numpy(stored.T.copy()))
    return model


def train_arrays(spec: RunSpec, arrays, config: TrainingConfig, initial: dict,
                 full_states: list | None = None, on_checkpoint=None) -> dict:
    """Train or resume from a prefix of full snapshots; used also by toy tests."""
    baseline._configure_threads(config.threads)
    x_train, y_train, x_test, y_test = arrays
    if x_train.dtype != torch.float32 or x_test.dtype != torch.float32:
        raise ValueError("Only float32 training/test inputs are supported")
    random.seed(spec.init_seed)
    np.random.seed(spec.init_seed)
    model = _model(spec, config, initial)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad],
                                lr=config.learning_rate, momentum=config.momentum)
    labels, updates = baseline.checkpoint_schedule("main", config, len(x_train))
    targets = set(updates.tolist())
    states = full_states if full_states is not None else []
    batches = (len(x_train) + config.batch_size - 1) // config.batch_size
    current_epoch, batch_position, update = 0, 0, 0
    order, generator_state = image_order(spec.shuffle_seed, 0, len(x_train))
    if states:
        if len(states) > len(labels):
            raise ValueError("Resume archive contains too many checkpoints")
        for k, state in enumerate(states):
            if state["checkpoint"] != int(labels[k]) or state["update"] != int(updates[k]):
                raise ValueError("Resume checkpoints are not a complete schedule prefix")
            if state["spec"] != asdict(spec) or state["config"] != asdict(config):
                raise ValueError("Resume checkpoint factors/configuration changed")
        last = states[-1]
        model.load_state_dict(last["model_state_dict"])
        # load_state_dict can retain CPU momentum tensor storage: a deepcopy is
        # essential so continuation cannot mutate its historical checkpoint.
        optimizer.load_state_dict(copy.deepcopy(last["optimizer_state_dict"]))
        torch.set_rng_state(last["torch_rng_state"])
        np.random.set_state(last["numpy_rng_state"])
        random.setstate(last["python_rng_state"])
        current_epoch, batch_position, update = last["epoch"], last["batch_position"], last["update"]
        order, generator_state = last["epoch_order"].clone(), last["order_generator_state"].clone()
        expected_order, expected_generator = image_order(spec.shuffle_seed, current_epoch, len(x_train))
        if not torch.equal(order, expected_order) or not torch.equal(generator_state, expected_generator):
            raise ValueError("Stored resume order disagrees with the declared shuffle factor")
        if update != current_epoch * batches + batch_position:
            raise ValueError("Inconsistent update/epoch/batch resume position")

    def save():
        accuracy, loss = baseline.evaluate(model, x_test, y_test, config.test_batch_size)
        states.append({
            "checkpoint": int(labels[len(states)]), "update": update,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
            "torch_rng_state": torch.get_rng_state().clone(),
            "numpy_rng_state": copy.deepcopy(np.random.get_state()),
            "python_rng_state": random.getstate(), "epoch": current_epoch,
            "batch_position": batch_position,
            "samples_consumed_this_epoch": min(batch_position * config.batch_size, len(x_train)),
            "epoch_order": order.clone(), "order_generator_state": generator_state.clone(),
            "order_seed": spec.shuffle_seed + current_epoch,
            "epoch_order_sha256": array_hash(order.numpy()),
            "init_seed": spec.init_seed, "shuffle_seed": spec.shuffle_seed,
            "spec": asdict(spec), "config": asdict(config), "accuracy": accuracy, "loss": loss,
        })
        if on_checkpoint is not None:
            on_checkpoint(states)

    if not states:
        save()
    resume_epoch, resume_position = current_epoch, batch_position
    for epoch in range(resume_epoch, config.epochs):
        current_epoch = epoch
        if epoch == resume_epoch:
            batch_position = resume_position
        else:
            batch_position = 0
            order, generator_state = image_order(spec.shuffle_seed, epoch, len(x_train))
        for start in range(batch_position * config.batch_size, len(x_train), config.batch_size):
            selection = order[start:start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x_train[selection]), y_train[selection])
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite training loss: {spec.run_id}/{update}")
            loss.backward()
            optimizer.step()
            update += 1
            batch_position += 1
            if torch.count_nonzero(model[0].bias) or torch.count_nonzero(model[2].bias):
                raise AssertionError("Frozen zero bias changed")
            if update in targets:
                save()
    if len(states) != len(labels):
        raise AssertionError("Incomplete checkpoint trajectory")
    result = {"checkpoints": labels, "updates": updates}
    for name, state_key, shape in (("W0", "0.weight", (784, config.hidden_width)),
                                   ("W1", "2.weight", (config.hidden_width, 10))):
        parameter_index = 0 if name == "W0" else 1
        result[name] = np.stack([s["model_state_dict"][state_key].numpy().T for s in states])
        result["momentum_" + name] = np.stack([
            s["optimizer_state_dict"]["state"].get(parameter_index, {}).get(
                "momentum_buffer", torch.zeros(shape[::-1])).numpy().T for s in states])
    for name, key in (("bias0", "0.bias"), ("bias1", "2.bias")):
        result[name] = np.stack([s["model_state_dict"][key].numpy() for s in states])
    for key in ("accuracy", "loss"):
        result[key] = np.asarray([s[key] for s in states], dtype=np.float64)
    return result


def source_inventory(root: Path) -> dict:
    """Read-only, byte-level validation of all baseline initializations and IDX."""
    main_path = root / "outputs/manifests/main.json"
    manifest = json.loads(main_path.read_text())
    if manifest.get("complete") is not True or len(manifest["runs"]) != 30:
        raise ValueError("The frozen original main cohort must be complete (30 runs)")
    sources = {"main_manifest": {"path": str(main_path), "sha256": sha256_file(main_path)},
               "code": {}, "datasets": {}, "initializations": {}}
    for name, path in (("shuffle_training", Path(__file__)),
                       ("original_training", root / "src/learning_trajectory/training.py"),
                       ("original_data", root / "src/learning_trajectory/data.py")):
        sources["code"][name] = {"path": str(path), "sha256": sha256_file(path)}
    for dataset, _, _ in DATASETS:
        folder = root / "data" / dataset
        path = folder / "manifest.json"
        dm = json.loads(path.read_text())
        if dm["dataset"] != dataset or dm["normalization"] != list(NORMALIZATION[dataset]):
            raise ValueError(f"Dataset/normalization mismatch: {dataset}")
        entries = {entry["filename"]: entry for entry in dm["files"]}
        verified = []
        for filename, shape in IDX_FILES:
            entry = entries[filename]
            for suffix, key in (("", "sha256"), (".gz", "gzip_sha256")):
                fp = folder / (filename + suffix)
                if sha256_file(fp) != entry[key]:
                    raise ValueError(f"Changed dataset file: {fp}")
            read_idx(folder / filename, shape)
            verified.append(entry)
        sources["datasets"][dataset] = {"path": str(path), "sha256": sha256_file(path), "files": verified}
    records = {r["run_id"]: r for r in manifest["runs"]}
    for dataset, prefix, base in DATASETS:
        for i in range(1, 11):
            run_id = f"{prefix}{i}"
            record = records[run_id]
            path = Path(record["path"])
            if record["dataset"] != dataset or record["seed"] != base + i or record["task"] != "identity":
                raise ValueError(f"Wrong original factor mapping: {run_id}")
            if sha256_file(path) != record["sha256"] or sha256_file(record["full_state_path"]) != record["full_state_sha256"]:
                raise ValueError(f"Changed original archive: {run_id}")
            with np.load(path, allow_pickle=False) as a:
                meta = json.loads(str(a["metadata"]))
                for key, expected in (("run_id", run_id), ("dataset", dataset), ("seed", base + i),
                                      ("training_source_sha256", sources["code"]["original_training"]["sha256"]),
                                      ("data_source_sha256", sources["code"]["original_data"]["sha256"]),
                                      ("dataset_manifest_sha256", sources["datasets"][dataset]["sha256"])):
                    if meta.get(key) != expected:
                        raise ValueError(f"Original metadata mismatch: {run_id}/{key}")
                if not np.array_equal(a["checkpoints"], np.arange(21)) or a["W0"].shape != (21, 784, 64) or a["W1"].shape != (21, 64, 10):
                    raise ValueError(f"Invalid original trajectory: {run_id}")
                if np.any(a["bias0"]) or np.any(a["bias1"]) or not np.isfinite(a["W0"]).all() or not np.isfinite(a["W1"]).all():
                    raise ValueError(f"Invalid original weights/biases: {run_id}")
                initial = {key: a[key][0].copy() for key in ("W0", "W1")}
                _model(RunSpec("check", dataset, i, 1, base + i, SHUFFLE_SEEDS[0], run_id), TrainingConfig(), initial)
                sources["initializations"][run_id] = {
                    "path": str(path), "sha256": record["sha256"], "init_id": i, "init_seed": base + i,
                    "dataset": dataset, "full_state_path": record["full_state_path"],
                    "full_state_sha256": record["full_state_sha256"],
                    "E0_W0_sha256": array_hash(initial["W0"]), "E0_W1_sha256": array_hash(initial["W1"])}
    return sources


def prepare(root: Path, out: Path, config: TrainingConfig) -> dict:
    baseline._configure_threads(config.threads)
    if asdict(config) != asdict(TrainingConfig(threads=config.threads)):
        raise ValueError("Production experiment requires the fixed original architecture/hyperparameters")
    sources = source_inventory(root)
    schedules = []
    for shuffle_id, seed in enumerate(SHUFFLE_SEEDS, 1):
        orders = np.stack([image_order(seed, e, 60000)[0].numpy() for e in range(config.epochs)])
        path = out / "data/shuffle_schedules" / f"s{shuffle_id:02d}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as a:
                if not np.array_equal(a["orders"], orders) or int(a["shuffle_seed"]) != seed:
                    raise ValueError(f"Existing shuffle schedule changed: {path}; use a separate output directory")
        else:
            _atomic_npz(path, orders=orders, epoch_seeds=np.arange(seed, seed + config.epochs), shuffle_seed=np.asarray(seed))
        schedules.append({"shuffle_id": shuffle_id, "shuffle_seed": seed, "path": str(path),
                          "sha256": sha256_file(path), "epoch_seeds": list(range(seed, seed + config.epochs)),
                          "epoch_order_sha256": [array_hash(order) for order in orders]})
    context = {"version": VERSION, "config": asdict(config), "conventions": CONVENTIONS,
               "sources": sources, "schedules": schedules,
               "environment": {"python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__}}
    context["fingerprint"] = digest_json(context)
    _atomic_json(out / "provenance/source_inventory.json", sources)
    _atomic_json(out / "training_plan.json", {**context, "expected_runs": 150, "factors": [asdict(s) for s in run_specs()]})
    return context


@lru_cache(maxsize=1)
def dataset_arrays(root: str, name: str):
    """Read cached inputs without invoking the baseline downloader/manifest writer."""
    mean, std = NORMALIZATION[name]
    tensors = []
    for i, (filename, shape) in enumerate(IDX_FILES):
        a = read_idx(Path(root) / "data" / name / filename, shape)
        if i % 2 == 0:
            value = torch.from_numpy(a.copy()).reshape(len(a), 784).to(torch.float32)
            value.div_(255).sub_(mean).div_(std)
        else:
            value = torch.from_numpy(a.astype(np.int64))
        tensors.append(value)
    return tuple(tensors)


def expected_metadata(spec, context):
    return {**asdict(spec), "cohort": "initialization_shuffle", "task": "identity", "seed": spec.init_seed,
            "config": context["config"], "context_fingerprint": context["fingerprint"],
            "source_initialization": context["sources"]["initializations"][spec.original_run_id],
            "schedule": context["schedules"][spec.shuffle_id - 1], "version": VERSION}


def validate_archive(path: Path, spec: RunSpec, context: dict, deep: bool = True) -> dict:
    """Reject stale/corrupted caches, including full-state and provenance failures."""
    expected = expected_metadata(spec, context)
    config = TrainingConfig(**context["config"])
    checkpoints, updates = baseline.checkpoint_schedule("main", config)
    with np.load(path, allow_pickle=False) as a:
        meta = json.loads(str(a["metadata"]))
        for key, value in expected.items():
            if meta.get(key) != value:
                raise ValueError(f"Cache mismatch {spec.run_id}/{key}; use --force or a separate output directory")
        for key, values in (("checkpoints", checkpoints), ("updates", updates)):
            if not np.array_equal(a[key], values):
                raise ValueError(f"Wrong {key}: {path}")
        for key, shape in (("W0", (21, 784, 64)), ("W1", (21, 64, 10)),
                           ("momentum_W0", (21, 784, 64)), ("momentum_W1", (21, 64, 10)),
                           ("bias0", (21, 64)), ("bias1", (21, 10)), ("accuracy", (21,)), ("loss", (21,))):
            if a[key].shape != shape or not np.isfinite(a[key]).all():
                raise ValueError(f"Invalid {key}: {path}")
            if key not in ("accuracy", "loss") and a[key].dtype != np.float32:
                raise ValueError(f"Wrong stored precision: {key}")
        if np.any(a["bias0"]) or np.any(a["bias1"]) or np.any(a["momentum_W0"][0]) or np.any(a["momentum_W1"][0]):
            raise ValueError("Nonzero bias or initial momentum")
        if np.any((a["accuracy"] < 0) | (a["accuracy"] > 1)) or np.any(a["loss"] < 0):
            raise ValueError("Invalid performance")
        for key in ("W0", "W1"):
            if array_hash(a[key][0]) != expected["source_initialization"][f"E0_{key}_sha256"]:
                raise ValueError("E0 differs from original stored initialization")
        state_path = path.with_suffix(".pt")
        state_hash = sha256_file(state_path)
        if meta["full_state_sha256"] != state_hash:
            raise ValueError(f"Full state archive hash changed: {state_path}")
        if deep:
            payload = torch.load(state_path, weights_only=False, map_location="cpu")
            validate_payload(payload, spec, context)
            states = payload["checkpoints"]
            if len(states) != 21:
                raise ValueError("Full state checkpoint count changed")
            for k, s in enumerate(states):
                if s["checkpoint"] != k or s["update"] != int(updates[k]) or s["spec"] != asdict(spec) or s["config"] != context["config"]:
                    raise ValueError("Full state metadata mismatch")
                for key, state_key, pi in (("W0", "0.weight", 0), ("W1", "2.weight", 1)):
                    w = s["model_state_dict"][state_key]
                    momentum = s["optimizer_state_dict"]["state"].get(pi, {}).get("momentum_buffer", torch.zeros_like(w))
                    if not np.array_equal(w.numpy().T, a[key][k]) or not np.array_equal(momentum.numpy().T, a["momentum_" + key][k]):
                        raise ValueError("NPZ/full-state weight or momentum mismatch")
                for key in ("0.bias", "2.bias"):
                    if torch.count_nonzero(s["model_state_dict"][key]):
                        raise ValueError("Full state bias is nonzero")
                order_hash = array_hash(s["epoch_order"].numpy())
                if order_hash != expected["schedule"]["epoch_order_sha256"][s["epoch"]] or order_hash != s["epoch_order_sha256"]:
                    raise ValueError("Full state order violates the shared shuffle schedule")
                if s["order_seed"] != spec.shuffle_seed + s["epoch"] or s["init_seed"] != spec.init_seed or s["shuffle_seed"] != spec.shuffle_seed:
                    raise ValueError("Full state seed mismatch")
                _, generator_state = image_order(spec.shuffle_seed, s["epoch"], 60000)
                if not torch.equal(s["order_generator_state"], generator_state):
                    raise ValueError("Full state order-generator state mismatch")
                if s["update"] != s["epoch"] * 235 + s["batch_position"]:
                    raise ValueError("Full state position mismatch")
                if s["accuracy"] != float(a["accuracy"][k]) or s["loss"] != float(a["loss"][k]):
                    raise ValueError("Full state evaluation mismatch")
                for field in ("torch_rng_state", "numpy_rng_state", "python_rng_state", "order_generator_state"):
                    if field not in s:
                        raise ValueError(f"Missing continuation field: {field}")
        accuracy, loss = float(a["accuracy"][-1]), float(a["loss"][-1])
    return {**asdict(spec), "path": str(path.resolve()), "sha256": sha256_file(path),
            "full_state_path": str(state_path.resolve()), "full_state_sha256": state_hash,
            "checkpoints": 21, "final_accuracy": accuracy, "final_loss": loss,
            "context_fingerprint": context["fingerprint"], "E0_W0_sha256": expected["source_initialization"]["E0_W0_sha256"],
            "E0_W1_sha256": expected["source_initialization"]["E0_W1_sha256"],
            "epoch_order_sha256": expected["schedule"]["epoch_order_sha256"]}


def train_run(spec, root: Path, out: Path, context: dict, force=False):
    baseline._configure_threads(context["config"]["threads"])
    for item in context["sources"]["code"].values():
        if sha256_file(item["path"]) != item["sha256"]:
            raise ValueError("Training source changed after preparation; restart with current sources")
    path = out / "data/checkpoints" / spec.dataset / f"{spec.run_id}.npz"
    if path.exists() and not force:
        try:
            return {**validate_archive(path, spec, context), "resumed": True}
        except Exception as error:
            raise ValueError(f"Cannot reuse {path}: {error}; use --force or a separate output directory") from error
    progress = path.with_suffix(".progress.pt")
    states = []
    if progress.exists() and not force:
        payload = torch.load(progress, weights_only=False, map_location="cpu")
        validate_payload(payload, spec, context)
        states = payload["checkpoints"]
    elif path.with_suffix(".pt").exists() and not force:
        # Recover a completed PT if the process died before publishing the NPZ.
        payload = torch.load(path.with_suffix(".pt"), weights_only=False, map_location="cpu")
        validate_payload(payload, spec, context)
        states = payload["checkpoints"]
    original_path = context["sources"]["initializations"][spec.original_run_id]["path"]
    with np.load(original_path, allow_pickle=False) as a:
        initial = {key: a[key][0].copy() for key in ("W0", "W1")}
    def persist(current):
        _atomic_pt(progress, state_payload(spec, context, current))
    started = time.monotonic()
    resumed_checkpoints = len(states)
    result = train_arrays(spec, dataset_arrays(str(root), spec.dataset), TrainingConfig(**context["config"]), initial, states, persist)
    state_path = path.with_suffix(".pt")
    _atomic_pt(state_path, state_payload(spec, context, states))
    result["metadata"] = np.asarray(json.dumps({**expected_metadata(spec, context), "full_state_sha256": sha256_file(state_path),
                                               "elapsed_seconds": time.monotonic() - started, "resumed_checkpoints": resumed_checkpoints}, sort_keys=True))
    _atomic_npz(path, **result)
    record = validate_archive(path, spec, context)
    progress.unlink(missing_ok=True)
    return {**record, "resumed": bool(resumed_checkpoints), "elapsed_seconds": time.monotonic() - started}


def resource_check(root: Path, out: Path, workers: int, threads: int) -> dict:
    available = next(int(line.split()[1]) * 1024 for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
    free = shutil.disk_usage(out).free
    report = {"cpu_count": os.cpu_count(), "load_average": list(os.getloadavg()), "available_ram_bytes": available,
              "disk_free_bytes": free, "workers": workers, "torch_threads_per_worker": threads,
              "required_ram_bytes": workers * 2_400_000_000 + 2_000_000_000, "required_disk_bytes": 8_000_000_000,
              "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    report["passed"] = available >= report["required_ram_bytes"] and free >= report["required_disk_bytes"] and workers * threads <= os.cpu_count()
    _atomic_json(out / "provenance/prelaunch_resources.json", report)
    if not report["passed"]:
        raise RuntimeError(f"Insufficient launch resources: {report}")
    return report


def train_cohort(root: Path, out: Path, workers=4, threads=1, force=False, run_ids=None,
                 prepare_only=False, validate_only=False):
    root, out = root.resolve(), out.resolve()
    context = prepare(root, out, TrainingConfig(threads=threads))
    specs = run_specs()
    requested = specs if not run_ids else [s for s in specs if s.run_id in set(run_ids)]
    if run_ids and set(run_ids) != {s.run_id for s in requested}:
        raise ValueError("Unknown run ID in selection")
    records = {}
    def write_manifest():
        ordered = [records[s.run_id] for s in specs if s.run_id in records]
        manifest = {"version": VERSION, "expected_runs": 150, "complete": len(ordered) == 150,
                    "context_fingerprint": context["fingerprint"], "factors": [asdict(s) for s in specs],
                    "factor_order": CONVENTIONS["ordering"], "sources": context["sources"],
                    "schedules": context["schedules"], "config": context["config"], "conventions": CONVENTIONS,
                    "runs": ordered}
        _atomic_json(out / "data/manifest.json", manifest)
        return manifest
    # Validate already completed artifacts before accepting them into this manifest.
    # --force replaces only selected runs; other old caches still require validation.
    for spec in specs:
        path = out / "data/checkpoints" / spec.dataset / f"{spec.run_id}.npz"
        if path.exists() and not (force and spec in requested):
            records[spec.run_id] = validate_archive(path, spec, context)
    manifest = write_manifest()
    if prepare_only:
        print(json.dumps({"prepared": True, "expected_runs": 150, "existing_validated_runs": len(records), "fingerprint": context["fingerprint"]}), flush=True)
        return manifest
    if validate_only:
        _atomic_json(out / "provenance/training_validation.json", {
            "passed": manifest["complete"], "validated_runs": len(records), "expected_runs": 150,
            "E0_matches_original_and_all_shuffles": True, "same_order_for_equal_shuffle_seed": True,
            "context_fingerprint": context["fingerprint"], "manifest_sha256": sha256_file(out / "data/manifest.json")})
        if not manifest["complete"]:
            raise ValueError("Training experiment is incomplete")
        return manifest
    resource_check(root, out, workers, threads)
    pending = [s for s in requested if s.run_id not in records]
    if workers == 1:
        iterator = (train_run(s, root, out, context, force) for s in pending)
        for record in iterator:
            records[record["run_id"]] = record
            manifest = write_manifest()
            print(json.dumps({"completed": len(records), "expected_runs": 150, **record}), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as executor:
            futures = [executor.submit(train_run, s, root, out, context, force) for s in pending]
            for future in as_completed(futures):
                record = future.result()
                records[record["run_id"]] = record
                manifest = write_manifest()
                print(json.dumps({"completed": len(records), "expected_runs": 150, **record}), flush=True)
    _atomic_json(out / "provenance/training_validation.json", {
        "passed": manifest["complete"], "validated_runs": len(records), "expected_runs": 150,
        "E0_matches_original_and_all_shuffles": True, "same_order_for_equal_shuffle_seed": True,
        "context_fingerprint": context["fingerprint"], "manifest_sha256": sha256_file(out / "data/manifest.json")})
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "05_Initialization_Shuffle")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--run-ids", nargs="+")
    parser.add_argument("--force", action="store_true")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--prepare-only", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers/threads must be positive")
    train_cohort(args.root, args.output, args.workers, args.threads, args.force,
                 args.run_ids, args.prepare_only, args.validate_only)


if __name__ == "__main__":
    main()
