"""Reproduce experiment 01_RSA without writing to the original pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "01_RSA"
sys.path.insert(0, str(ROOT / "src"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_preservation(out=OUT):
    snapshot = json.loads((out / "provenance/source_inventory_before.json").read_text())
    checked = []
    for row in snapshot["files"]:
        path = ROOT / row["path"]
        checked.append({"path": row["path"], "unchanged": path.is_file() and sha(path) == row["sha256"]})
    result = {"all_unchanged": all(x["unchanged"] for x in checked), "checked_files": len(checked),
              "checked_utc": datetime.now(timezone.utc).isoformat(),
              "changed": [x["path"] for x in checked if not x["unchanged"]]}
    (out / "provenance/preservation_check.json").write_text(json.dumps(result, indent=2) + "\n")
    if not result["all_unchanged"]:
        raise RuntimeError(f"Original source files changed: {result['changed']}")
    return result


def write_config(out=OUT):
    import numpy, scipy, pandas, matplotlib
    from inference import BASE_SEED, N_PERMUTATIONS
    config = {
        "experiment": "01_RSA", "version": "1.0.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "specification": {"path": "01_RSA_Initialisierung_Endzustand.md", "sha256": sha(ROOT / "01_RSA_Initialisierung_Endzustand.md")},
        "source_manifest": {"path": "outputs/manifests/main.json", "sha256": sha(ROOT / "outputs/manifests/main.json")},
        "source_inventory": "provenance/source_inventory_before.json",
        "cohort": "main", "n_runs": 30, "runs_per_dataset": 10, "pairs_per_dataset": 45,
        "datasets": ["mnist", "fashion_mnist", "kmnist"], "checkpoints": list(range(21)),
        "checkpoint_units": "half epochs; E20 = 10 epochs", "arithmetic": "float64 from saved float32 weights",
        "primary_metric_key": "frobenius|total|optimal|canonical",
        "primary_distance": "sqrt(||canonical_A0-canonical_B0[:,p]||_F^2 + ||canonical_A1-canonical_B1[p,:]||_F^2)",
        "gauge": "s_h=sqrt(||out_h||_2/||in_h||_2); in*=s_h, out/=s_h; analysis-only",
        "alignment": "single-hidden-layer global Hungarian maximizes common W0/W1 inner product after canonicalization; recomputed per pair and checkpoint",
        "fixed_alignment_sensitivity": "For fixed_E0/fixed_E20, use that pair assignment at every checkpoint including the baseline RDM",
        "all_distance_definitions": "data/distance_definitions.json",
        "statistic": "Spearman rank correlation of within-dataset upper triangles, no diagonal",
        "replication_unit": "network identity; 10 independent training runs and 45 dependent pairs per dataset",
        "n_permutations": N_PERMUTATIONS, "base_seed": BASE_SEED,
        "dataset_permutation_seeds": {"mnist": 2026111601, "fashion_mnist": 2026111602, "kmnist": 2026111603},
        "null_test": "simultaneous row/column permutation of target RDM within dataset; independent permutations per stratum; plus-one Monte Carlo p",
        "primary_test": "one-sided positive association at E20; additionally two-sided p; Holm family correction over three dataset endpoints",
        "uncertainty": "delete-one-network jackknife; normal 95% intervals; raw bounds saved, clipped to [-1,1] only in plots",
        "aggregate": "equal-weight mean Fisher z, inverse tanh for presentation; independent-stratum permutation null and stratified run-level jackknife",
        "e0": "identity sanity check only, p omitted",
        "exploratory": "all other checkpoints, metric/layer/alignment/norm/neighbor controls; no family-wide error control claimed",
        "optional_early_replication": False, "optional_neighbor_retention": True,
        "new_training": False, "original_pipeline_written": False,
        "software": {"python": platform.python_version(), "numpy": numpy.__version__, "scipy": scipy.__version__,
                     "pandas": pandas.__version__, "matplotlib": matplotlib.__version__},
        "source_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in sorted((out / "code").glob("*.py"))},
        "reused_code_sha256": {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / "src/learning_trajectory/metrics.py", ROOT / "src/learning_trajectory/statistics.py"]},
        "reproduction_command": ".venv/bin/python 01_RSA/code/run.py --stage all",
    }
    (out / "rsa_config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n")


def artifact_manifest(out=OUT):
    excluded = {"artifact_manifest.json"}
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name not in excluded and "__pycache__" not in str(p) and ".pytest_cache" not in str(p) and "logs" not in p.relative_to(out).parts)
    (out / "artifact_manifest.json").write_text(json.dumps({"files": [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "distances", "inference", "plots", "report", "finalize"), default="all")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    stage = args.stage
    if stage in ("all", "distances"):
        from distances import build_distances
        build_distances(ROOT, OUT)
    if stage in ("all", "inference"):
        from inference import run_inference
        run_inference(OUT)
    if stage in ("all", "finalize"):
        from audit import audit
        audit(OUT)
    if stage in ("all", "plots"):
        from plots import make_plots
        make_plots(OUT)
    if stage in ("all", "report"):
        from reporting import make_reports
        make_reports(OUT)
    if stage in ("all", "finalize"):
        write_config()
        verify_preservation()
        artifact_manifest()
    print(f"Completed stage {stage}", flush=True)


if __name__ == "__main__":
    main()
