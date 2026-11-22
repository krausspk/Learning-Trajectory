"""Validated IDX data and fixed representation probes for the paper reproduction.

Only the official 60,000/10,000 splits are used. Downloads are written atomically;
the compressed and decoded bytes are fingerprinted in each dataset manifest.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import time
from urllib.request import Request, urlopen

import numpy as np
import torch


NORMALIZATION = {
    "mnist": (0.1307, 0.3081),
    "fashion_mnist": (0.2860, 0.3530),
    "kmnist": (0.1918, 0.3483),
}
BASE_URLS = {
    "mnist": ("https://ossci-datasets.s3.amazonaws.com/mnist/",),
    "fashion_mnist": (
        "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/",
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
    ),
    "kmnist": (
        "https://codh.rois.ac.jp/kmnist/dataset/kmnist/",
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/",
    ),
}
IDX_FILES = (
    ("train-images-idx3-ubyte", (60000, 28, 28)),
    ("train-labels-idx1-ubyte", (60000,)),
    ("t10k-images-idx3-ubyte", (10000, 28, 28)),
    ("t10k-labels-idx1-ubyte", (10000,)),
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_json(path: Path, value: object) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def read_idx(path: str | Path, expected_shape: tuple[int, ...] | None = None) -> np.ndarray:
    """Read an unsigned-byte IDX, rejecting wrong magic, dimensions and byte count."""
    path = Path(path)
    content = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    if len(content) < 4 or content[:3] != b"\x00\x00\x08":
        raise ValueError(f"Invalid unsigned-byte IDX magic in {path}")
    ndim = content[3]
    if ndim not in (1, 3) or len(content) < 4 + 4 * ndim:
        raise ValueError(f"Invalid IDX dimension count in {path}")
    shape = struct.unpack(">" + "I" * ndim, content[4 : 4 + 4 * ndim])
    if expected_shape is not None and shape != expected_shape:
        raise ValueError(f"IDX shape {shape} != expected {expected_shape} in {path}")
    offset = 4 + 4 * ndim
    if len(content) - offset != int(np.prod(shape)):
        raise ValueError(f"Truncated or overlong IDX in {path}")
    result = np.frombuffer(content, dtype=np.uint8, offset=offset).reshape(shape)
    if ndim == 1 and (result.size == 0 or int(result.max()) > 9):
        raise ValueError(f"Unexpected class label in {path}")
    return result


def ensure_dataset(name: str, data_dir: str | Path = "data") -> dict:
    """Download once, validate every source, and record hashes and source URLs."""
    if name not in NORMALIZATION:
        raise ValueError(f"Unknown dataset {name!r}; choose {tuple(NORMALIZATION)}")
    folder = Path(data_dir) / name
    folder.mkdir(parents=True, exist_ok=True)
    old = {}
    manifest_path = folder / "manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
    records = []
    for filename, shape in IDX_FILES:
        path = folder / filename
        compressed_path = folder / (filename + ".gz")
        old_entry = next((r for r in old.get("files", []) if r["filename"] == filename), {})
        source_url = old_entry.get("source_url")
        if not path.exists():
            if not compressed_path.exists():
                errors = []
                for base_url in BASE_URLS[name]:
                    url = base_url + filename + ".gz"
                    for attempt in range(3):
                        try:
                            with urlopen(Request(url, headers={"User-Agent": "Learning-Trajectory-reproduction/1.0"}), timeout=90) as response:
                                compressed = response.read()
                            decoded = gzip.decompress(compressed)
                            _atomic_bytes(compressed_path, compressed)
                            _atomic_bytes(path, decoded)
                            read_idx(path, shape)
                            source_url = url
                            break
                        except Exception as error:
                            errors.append(f"{url}: {type(error).__name__}: {error}")
                            if path.exists():
                                path.unlink()
                            if compressed_path.exists():
                                compressed_path.unlink()
                            if attempt < 2:
                                time.sleep(attempt + 1)
                    if path.exists():
                        break
                if not path.exists():
                    raise RuntimeError("Unable to obtain validated dataset: " + "\n".join(errors))
            else:
                _atomic_bytes(path, gzip.decompress(compressed_path.read_bytes()))
        array = read_idx(path, shape)
        decoded_hash = sha256_file(path)
        if old_entry.get("sha256") and old_entry["sha256"] != decoded_hash:
            raise ValueError(f"Cached dataset changed since manifest: {path}")
        records.append({
            "filename": filename,
            "shape": list(array.shape),
            "bytes": path.stat().st_size,
            "sha256": decoded_hash,
            "gzip_sha256": sha256_file(compressed_path) if compressed_path.exists() else None,
            "source_url": source_url or "preexisting local IDX; original URL not recorded",
        })
    result = {"dataset": name, "normalization": NORMALIZATION[name], "files": records}
    _atomic_json(manifest_path, result)
    return result


def load_dataset(name: str, data_dir: str | Path = "data") -> tuple[torch.Tensor, ...]:
    """Return normalized float32 image and int64 label tensors on CPU."""
    ensure_dataset(name, data_dir)
    folder = Path(data_dir) / name
    mean, std = NORMALIZATION[name]
    arrays = [read_idx(folder / file, shape) for file, shape in IDX_FILES]
    tensors = []
    for index, array in enumerate(arrays):
        if index % 2 == 0:
            values = torch.from_numpy(array.copy()).reshape(len(array), 784).to(torch.float32)
            values.div_(255.0).sub_(mean).div_(std)
        else:
            values = torch.from_numpy(array.astype(np.int64))
        tensors.append(values)
    return tuple(tensors)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.stem + ".", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build_probes(data_dir: str | Path = "data", output_dir: str | Path = "outputs") -> dict[str, str]:
    """Persist the shared 3,000-image and paired-task 2,000-image probe matrices.

    The paper specifies the seed but not the NumPy RNG API; this reconstruction
    explicitly uses RandomState(5052026), with datasets and classes in fixed order.
    """
    folder = Path(output_dir) / "probes"
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(5052026)
    images, labels, indices, datasets = [], [], [], []
    task = None
    for name in NORMALIZATION:
        ensure_dataset(name, data_dir)
        test_images = read_idx(Path(data_dir) / name / IDX_FILES[2][0], IDX_FILES[2][1])
        test_labels = read_idx(Path(data_dir) / name / IDX_FILES[3][0], IDX_FILES[3][1])
        selected = np.concatenate([rng.choice(np.flatnonzero(test_labels == label), 100, replace=False) for label in range(10)])
        mean, std = NORMALIZATION[name]
        x = test_images[selected].reshape(-1, 784).astype(np.float32)
        x /= np.float32(255)
        x -= np.float32(mean)
        x /= np.float32(std)
        images.append(x)
        labels.append(test_labels[selected])
        indices.append(selected)
        datasets.append(np.full(len(selected), name))
        if name == "mnist":
            task_indices = np.concatenate([np.flatnonzero(test_labels == label)[:200] for label in range(10)])
            task_x = test_images[task_indices].reshape(-1, 784).astype(np.float32)
            task_x /= np.float32(255)
            task_x -= np.float32(mean)
            task_x /= np.float32(std)
            task = {"X": task_x, "indices": task_indices, "labels": test_labels[task_indices]}
    combined = {"X": np.concatenate(images), "labels": np.concatenate(labels), "indices": np.concatenate(indices), "datasets": np.concatenate(datasets)}
    paths = {"main": folder / "main.npz", "task": folder / "task.npz"}
    _atomic_npz(paths["main"], **combined)
    _atomic_npz(paths["task"], **task)
    manifest = {
        "seed": 5052026,
        "rng": "numpy.random.RandomState (MT19937)",
        "dataset_order": list(NORMALIZATION),
        "main_selection": "100 without replacement per nominal class per dataset, concatenated dataset then class",
        "task_selection": "first 200 per digit in official MNIST test IDX order, concatenated by digit",
        "main": {"path": str(paths["main"]), "shape": list(combined["X"].shape), "sha256": sha256_file(paths["main"]), "matrix_sha256": hashlib.sha256(combined["X"].tobytes()).hexdigest()},
        "task": {"path": str(paths["task"]), "shape": list(task["X"].shape), "sha256": sha256_file(paths["task"]), "matrix_sha256": hashlib.sha256(task["X"].tobytes()).hexdigest()},
    }
    _atomic_json(folder / "manifest.json", manifest)
    return {key: str(value) for key, value in paths.items()}
