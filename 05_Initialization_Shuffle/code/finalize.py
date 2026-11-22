"""Read-only integrity checks and publication inventory for experiment 05.

Training archives and their manifest are immutable inputs to this finalizer.
Only the experiment's completion config, preservation receipt and artifact
inventory are written. Existing experiments are never changed.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import tempfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_Initialization_Shuffle"
DATASETS = ("mnist", "fashion_mnist", "kmnist")
SHUFFLE_SEEDS = (11001, 12001, 13001, 14001, 15001)
EXPECTED_PRESERVED_FILES = 703
CSV_OUTPUTS = ("initialization_retrieval.csv", "shuffle_retrieval.csv", "initial_trajectory_retrieval.csv",
               "aggregate.csv", "aggregate_distance_contrasts.csv", "aggregate_jackknife.csv",
               "distance_contrasts.csv", "distance_groups.csv", "jackknife.csv", "leave_factor_level_out.csv",
               "per_identity_scores.csv", "performance_decomposition.csv", "performance_factor_means.csv")


def sha(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_hash(path, expected, label="file"):
    path = Path(path)
    require(path.is_file(), f"Missing {label}: {path}")
    actual = sha(path)
    require(actual == expected, f"Changed {label}: {path}")
    return actual


def preservation_check(out: Path, expected_count=EXPECTED_PRESERVED_FILES):
    """Verify the pre-existing-file inventory; authorized additions are retained."""
    out = Path(out).resolve()
    inventory_path = out / "provenance/source_inventory_before.json"
    inventory = read_json(inventory_path)
    records = inventory["files"]
    require(len(records) == expected_count, f"Expected {expected_count} inventoried original files")
    require(len({r["path"] for r in records}) == expected_count, "Duplicate original inventory paths")
    changed = []
    for record in records:
        path = (out.parent / record["path"]).resolve()
        require(path.is_relative_to(out.parent), "Original inventory path escapes project")
        require(not path.is_relative_to(out), "Original inventory unexpectedly includes experiment 05")
        if not path.is_file() or path.stat().st_size != record["bytes"] or sha(path) != record["sha256"]:
            changed.append(record["path"])
    return {"all_unchanged": not changed, "files_checked": len(records), "changed": changed,
            "source_inventory_sha256": sha(inventory_path),
            "authorized_additions": inventory.get("authorized_additions"),
            "checked_utc": datetime.now(timezone.utc).isoformat()}


def validate_training(out: Path):
    """Bind the successful deep validation receipt to unchanged 150 NPZ/PT pairs.

    This does not rerun training, mutate its manifest, or deserialize PyTorch.
    Hashes bind the current bytes to the previously completed deep state checks.
    """
    out = Path(out).resolve()
    manifest_path = out / "data/manifest.json"
    manifest = read_json(manifest_path)
    receipt = read_json(out / "provenance/training_validation.json")
    plan = read_json(out / "training_plan.json")
    require(receipt.get("passed") is True and receipt.get("validated_runs") == 150 and receipt.get("expected_runs") == 150,
            "Training deep-validation receipt is missing, failed or incomplete")
    require_hash(manifest_path, receipt["manifest_sha256"], "validated training manifest")
    require(manifest.get("complete") is True and manifest.get("expected_runs") == 150, "Training manifest is incomplete")
    records = manifest["runs"]
    require(len(records) == 150 and sum(r["checkpoints"] for r in records) == 3150, "Expected 150 runs and 3150 checkpoints")
    expected = {(d, i, s) for d in DATASETS for i in range(1, 11) for s in range(1, 6)}
    actual = [(r["dataset"], r["init_id"], r["shuffle_id"]) for r in records]
    require(len(set(actual)) == 150 and set(actual) == expected, "Incomplete or duplicated crossed factor grid")
    context = {key: plan[key] for key in ("version", "config", "conventions", "sources", "schedules", "environment")}
    fingerprint = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    require(plan["fingerprint"] == fingerprint == receipt["context_fingerprint"] == manifest["context_fingerprint"],
            "Training context fingerprint mismatch")
    for key in ("config", "conventions", "sources", "schedules"):
        require(manifest[key] == plan[key], f"Training plan/manifest mismatch: {key}")
    require(receipt.get("E0_matches_original_and_all_shuffles") is True and receipt.get("same_order_for_equal_shuffle_seed") is True,
            "Deep validation did not confirm E0 and order invariants")
    sources = manifest["sources"]
    for entry in list(sources["code"].values()) + [sources["main_manifest"]]:
        require_hash(entry["path"], entry["sha256"], "training source")
    for entry in sources["datasets"].values():
        require_hash(entry["path"], entry["sha256"], "dataset manifest")
        for source in entry["files"]:
            for suffix, key in (("", "sha256"), (".gz", "gzip_sha256")):
                require_hash(Path(entry["path"]).parent / (source["filename"] + suffix), source[key], "dataset bytes")
    for entry in sources["initializations"].values():
        require_hash(entry["path"], entry["sha256"], "original initialization")
        require_hash(entry["full_state_path"], entry["full_state_sha256"], "original full-state archive")
    schedules = {s["shuffle_id"]: s for s in manifest["schedules"]}
    require(set(schedules) == set(range(1, 6)), "Expected exactly five shuffle schedules")
    for shuffle_id, seed in enumerate(SHUFFLE_SEEDS, 1):
        schedule = schedules[shuffle_id]
        require(schedule["shuffle_seed"] == seed and schedule["epoch_seeds"] == list(range(seed, seed + 10)), "Incorrect independent epoch-seed schedule")
        require_hash(schedule["path"], schedule["sha256"], "shared shuffle schedule")
    for record in records:
        initial = sources["initializations"][record["original_run_id"]]
        schedule = schedules[record["shuffle_id"]]
        require(record["checkpoints"] == 21 and record["context_fingerprint"] == fingerprint, "Invalid per-run checkpoint/provenance metadata")
        require((record["dataset"], record["init_id"], record["init_seed"]) ==
                (initial["dataset"], initial["init_id"], initial["init_seed"]), "Incorrect original initialization mapping")
        require(record["shuffle_seed"] == schedule["shuffle_seed"] and record["epoch_order_sha256"] == schedule["epoch_order_sha256"], "Shared shuffle order mismatch")
        require(all(record[key] == initial[key] for key in ("E0_W0_sha256", "E0_W1_sha256")), "Original E0 mismatch")
        for key, hash_key in (("path", "sha256"), ("full_state_path", "full_state_sha256")):
            require(Path(record[key]).resolve().is_relative_to(out / "data/checkpoints"), "Checkpoint path escapes experiment")
            require_hash(record[key], record[hash_key], "validated training archive")
    require(not list((out / "data/checkpoints").rglob("*.progress.pt")), "Partial training archives remain")
    return {"passed": True, "runs": 150, "checkpoints": 3150, "files_hashed": 300,
            "manifest_sha256": sha(manifest_path), "context_fingerprint": fingerprint,
            "deep_validation_receipt_sha256": sha(out / "provenance/training_validation.json"),
            "verification": "unchanged full NPZ/PT bytes bound to successful deep training validation; E0 and shared-order metadata rechecked"}


def validate_analysis(out: Path, training: dict):
    path = out / "analysis_validation.json"
    diagnostic = read_json(path)
    require(diagnostic.get("passed") is True, "Analysis validation failed")
    for key, count in (("n_complete_training_runs", 150), ("n_datasets", 3), ("n_initializations_per_dataset", 10), ("n_shuffle_levels", 5)):
        require(diagnostic[key] == count, f"Analysis design count mismatch: {key}")
    require(diagnostic["sources"]["manifest"]["sha256"] == training["manifest_sha256"], "Analysis used a different training manifest")
    for entry in diagnostic["sources"].values():
        require_hash(entry["path"], entry["sha256"], "analysis source")
    fingerprint = hashlib.sha256(json.dumps(diagnostic["sources"], sort_keys=True).encode()).hexdigest()
    require(diagnostic["analysis_fingerprint"] == fingerprint, "Analysis source fingerprint mismatch")
    require({entry["dataset"] for entry in diagnostic["outputs"]} == set(DATASETS) and len(diagnostic["outputs"]) == 3,
            "Expected exactly three analyzed datasets")
    for entry in diagnostic["outputs"]:
        require_hash(entry["path"], entry["sha256"], "analysis distance archive")
    required = [out / name for name in CSV_OUTPUTS] + [out / "data" / f"{dataset}_inference.npz" for dataset in DATASETS] + [out / "data/aggregate_inference.npz"]
    require(all(p.is_file() and p.stat().st_size > 0 for p in required), "Missing analysis tables or saved inferential distributions")
    return {"passed": True, "analysis_validation_sha256": sha(path), "analysis_fingerprint": fingerprint,
            "sources": diagnostic["sources"], "metric_keys": diagnostic["metric_keys"],
            "primary_key": diagnostic["primary_key"], "primary_checkpoint": diagnostic["primary_checkpoint"],
            "n_permutations": diagnostic["n_permutations"], "permutation_seeds": diagnostic["permutation_seeds"],
            "ci_conditioning": diagnostic["ci_conditioning"], "performance_decomposition": diagnostic["performance_decomposition"]}


def presentation_status(out: Path):
    """Validate available PDF/QA bindings without inventing a visual inspection."""
    status = {"results_pdf_present": (out / "results.pdf").is_file(), "qa_receipts": {}}
    if status["results_pdf_present"]:
        receipt = read_json(out / "results_manifest.json")
        require_hash(out / "results.pdf", receipt["report_sha256"], "results PDF")
        require_hash(out / "results_content.json", receipt["content_sha256"], "results PDF content")
        require_hash(receipt["generator"], receipt["generator_sha256"], "PDF generator")
        for entry in receipt["figure_sources"]:
            require_hash(out / entry["source"], entry["sha256"], "PDF source figure")
        for path, digest in receipt["data_sources"].items():
            require_hash(out / path, digest, "PDF source data")
        visual = receipt.get("visual_qa")
        if isinstance(visual, dict):
            require_hash(out / visual["receipt"], visual["receipt_sha256"], "linked PDF QA receipt")
            require(visual["report_sha256"] == receipt["report_sha256"], "Linked QA references a different results PDF")
        status.update({"results_pdf_sha256": receipt["report_sha256"], "pages": receipt["pages"],
                       "results_manifest_sha256": sha(out / "results_manifest.json"), "pdf_visual_qa": receipt.get("visual_qa", "pending")})
    plots = out / "plot_sources.json"
    if plots.exists():
        receipt = read_json(plots)
        require_hash(out / "code/plots.py", receipt["plotting_source_sha256"], "plotting code")
        for path, digest in receipt["sources"].items():
            require_hash(out / path, digest, "plot input")
        status["plot_sources_sha256"] = sha(plots)
    for relative in ("provenance/visual_qa/receipt.json", "results_qa/receipt.json"):
        path = out / relative
        if not path.exists():
            continue
        receipt = read_json(path)
        if "report_sha256" in receipt:
            require_hash(out / receipt.get("report", "results.pdf"), receipt["report_sha256"], "visually reviewed PDF")
        if "plotting_source_sha256" in receipt:
            require_hash(out / "code/plots.py", receipt["plotting_source_sha256"], "visually reviewed plotting code")
        for figure in receipt.get("figures", []):
            for key in ("png", "pdf", "pdf_preview"):
                if key in figure and key + "_sha256" in figure:
                    require_hash(out / figure[key], figure[key + "_sha256"], "visually reviewed figure")
        for page in receipt.get("pages", []):
            if "file" in page and "sha256" in page:
                require_hash(out / page["file"], page["sha256"], "reviewed PDF page rendering")
        status["qa_receipts"][relative] = {"sha256": sha(path), "status": receipt.get("status", receipt.get("passed", "recorded"))}
    return status


def artifact_manifest(out: Path):
    paths = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "artifact_manifest.json"
                   and not ({"__pycache__", ".pytest_cache", "logs"} & set(p.relative_to(out).parts))
                   and not p.name.endswith(".tmp"))
    records = [{"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": sha(p)} for p in paths]
    write_json(out / "artifact_manifest.json", {"files": records})
    return len(records)


def finalize(out: Path = OUT):
    out = Path(out).resolve()
    preserved = preservation_check(out)
    write_json(out / "provenance/preservation_check.json", preserved)
    require(preserved["all_unchanged"], f"Previous files changed: {preserved['changed']}")
    training = validate_training(out)
    analysis = validate_analysis(out, training)
    presentation = presentation_status(out)
    config = {
        "version": "1.0.0", "experiment": "05_Initialization_Shuffle", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_plan": read_json(out / "analysis_plan.json"), "analysis_plan_sha256": sha(out / "analysis_plan.json"),
        "analysis_design": read_json(out / "analysis_design.json"), "analysis_design_sha256": sha(out / "analysis_design.json"),
        "training": training, "analysis": analysis, "presentation": presentation,
        "code_sha256": {str(p.relative_to(out.parent)): sha(p) for p in sorted((out / "code").glob("*.py"))},
        "test_code_sha256": {str(p.relative_to(out.parent)): sha(p) for p in sorted((out / "tests").glob("*.py"))},
        "software": {"python": platform.python_version(), **{name: version(name) for name in ("numpy", "scipy", "pandas", "matplotlib", "torch")}},
        "independence": "The 50 networks per dataset form a 10-initialization x 5-shuffle crossed design, not 50 independent initializations.",
        "reproduction_command": ".venv/bin/python 05_Initialization_Shuffle/code/run.py --stage all",
        "read_only_validation_command": ".venv/bin/python 05_Initialization_Shuffle/code/run.py --stage validate",
    }
    write_json(out / "experiment_config.json", config)
    files = artifact_manifest(out)
    return {"passed": True, "preserved_files": preserved["files_checked"], "training_runs": 150,
            "checkpoints": 3150, "artifact_files": files, "artifact_manifest_sha256": sha(out / "artifact_manifest.json"),
            "results_pdf_present": presentation["results_pdf_present"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--validate-only", action="store_true", help="Read-only training checks; no files written")
    args = parser.parse_args()
    print(json.dumps(validate_training(args.output) if args.validate_only else finalize(args.output), indent=2))
