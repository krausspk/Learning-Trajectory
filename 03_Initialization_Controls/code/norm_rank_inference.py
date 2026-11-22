"""Isolated adapter for the post-hoc within-checkpoint norm-rank control.

The frozen Experiment 02 inference code is imported read-only. Its identity
permutation tests, summaries, and aggregation are reused unchanged. Only the
delete-one-network input matrices differ: norms must be ranked anew among the
nine retained networks before calculating the retrieval ranks. Cutting a row
and column out of the ten-network rank-distance matrix is not equivalent.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

BASE_SEED = 2026111900
FROZEN_SOURCE = Path(__file__).resolve().parents[2] / "02_Initialization_Retrieval/code/inference.py"


def load_frozen_inference():
    spec = importlib.util.spec_from_file_location("frozen_norm_rank_retrieval_inference", FROZEN_SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RecalibratedJackknife:
    """Supply saved re-ranked delete-one costs in verified dataset/K/T order.

    The input hook records the dataset and its ordered full matrices. Every
    jackknife call must equal the expected full matrix. This avoids ambiguous
    lookups when two variants have the same full matrix but distinct deletions.
    Shape of jackknife_distances: [K, T, deleted_network, n-1, n-1].
    Both hooks are restored when leaving the context, including on an error.
    """

    def __init__(self, module):
        self.module = module
        self.original_loader = module._load_input
        self.original_jackknife = module.jackknife_retrieval
        self.current = None
        self.completed = []

    def __enter__(self):
        self.module._load_input = self._load_input
        self.module.jackknife_retrieval = self._jackknife
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.module._load_input = self.original_loader
        self.module.jackknife_retrieval = self.original_jackknife

    def _finish_current(self):
        if self.current is not None:
            expected = self.current["distances"].shape[0] * self.current["distances"].shape[1]
            if self.current["cursor"] != expected:
                raise RuntimeError("Frozen inference did not consume exactly all ordered checkpoint matrices")
            self.completed.append({"input": str(self.current["path"]), "n_jackknife_calls": expected})
            self.current = None

    def _load_input(self, path):
        self._finish_current()
        result = self.original_loader(path)
        data, distances, keys, checkpoints, run_ids = result
        k, t, n, _ = distances.shape
        if "jackknife_distances" not in data:
            raise ValueError("Norm-rank controls require separately recalibrated jackknife_distances")
        jack = np.asarray(data["jackknife_distances"], dtype=np.float64)
        if jack.shape != (k, t, n, n - 1, n - 1):
            raise ValueError(f"Expected jackknife_distances[{k},{t},{n},{n-1},{n-1}], got {jack.shape}")
        if not np.isfinite(jack).all() or np.any(jack < 0):
            raise ValueError("Precomputed delete-one costs must be finite and nonnegative")
        if "jackknife_deleted_indices" in data and not np.array_equal(data["jackknife_deleted_indices"], np.arange(n)):
            raise ValueError("Precomputed deletion order differs from the network identity order")
        remaining = np.asarray([np.delete(np.arange(n), i) for i in range(n)])
        if "jackknife_remaining_indices" in data and not np.array_equal(data["jackknife_remaining_indices"], remaining):
            raise ValueError("Precomputed retained query/candidate identities are not in their original order")
        if "jackknife_axes" in data and list(data["jackknife_axes"].astype(str)) != [
                "metric", "checkpoint", "deleted_run", "remaining_query", "remaining_candidate"]:
            raise ValueError("Precomputed jackknife axis labels do not match the adapter convention")
        self.current = {"path": Path(path).resolve(), "distances": distances, "jack": jack, "cursor": 0}
        return result

    def _jackknife(self, distances):
        if self.current is None:
            raise RuntimeError("Jackknife called before loading a norm-rank dataset")
        all_distances = self.current["distances"]
        k, t = divmod(self.current["cursor"], all_distances.shape[1])
        if k >= all_distances.shape[0] or not np.array_equal(distances, all_distances[k, t]):
            raise RuntimeError("Frozen inference matrix order does not match the verified norm-rank adapter")
        n = len(distances)
        replicas, chances = [], []
        for deleted in range(n):
            ranks, credit = self.module.rank_candidates(self.current["jack"][k, t, deleted])
            replicas.append(self.module.mapping_statistics(ranks, credit, np.arange(n - 1))[0])
            chances.append(float((1 / ranks).mean()))
        self.current["cursor"] += 1
        return np.asarray(replicas), np.asarray(chances)

    def finish(self):
        self._finish_current()


def run_inference(out: Path, base_seed=BASE_SEED):
    """Run the six exploratory norm-rank variants without changing old files."""
    out = Path(out)
    source_hash = hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest()
    module = load_frozen_inference()
    with RecalibratedJackknife(module) as adapter:
        diagnostics = module.run_inference(out, base_seed=base_seed, controls=True, primary_key=None)
        adapter.finish()
    if hashlib.sha256(FROZEN_SOURCE.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError("Frozen Experiment 02 source changed during the analysis")
    provenance = {
        "frozen_source": str(FROZEN_SOURCE), "frozen_source_sha256": source_hash,
        "wrapper": str(Path(__file__).resolve()),
        "wrapper_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "base_seed": int(base_seed), "interpretive_role": "posthoc_exploratory_norm_rank_control",
        "reason": "Absolute norm-distance retrieval can have identical candidate ordering for every query after common norm growth; this control compares within-checkpoint norm ranks.",
        "jackknife": "For each deleted trajectory, rerank current and initial norms separately among the remaining nine networks before forming the 9x9 distance matrix; then rerank candidates and use the frozen chance-corrected jackknife summaries.",
        "null": "Full-sample label-blind norm-rank distances held fixed; unchanged global bijective candidate-identity permutations.",
        "adapter_order_check": "Dataset loader records full matrices; each K/T jackknife call must match its expected matrix exactly; all matrices must be consumed once.",
        "completed_datasets": adapter.completed,
    }
    diagnostics["norm_rank_control"] = provenance
    (out / "inference_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
    (out / "inference_wrapper_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return diagnostics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "norm_rank_controls")
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    args = parser.parse_args()
    run_inference(args.output, base_seed=args.base_seed)
