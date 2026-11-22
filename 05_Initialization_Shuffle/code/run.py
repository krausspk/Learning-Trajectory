"""Reproduce experiment 05 in explicit stages; validation never writes archives.

Default 'all' resumes/validates the full 150-run training before rebuilding the
analysis, figures and reports. --dry-run only prints the selected ordered stages.
PDF visual review remains a separate inspection recorded in the QA receipts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "05_Initialization_Shuffle"
PDF_PYTHON = Path(sys.executable)
STAGES = ("train", "validate", "analysis", "plots", "report", "pdf", "finalize")


def load_module(out, name):
    module_name = f"_experiment05_stage_{name}"
    spec = importlib.util.spec_from_file_location(module_name, Path(out) / "code" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def execution_plan(stage="all", workers=4, threads=1):
    if stage not in ("all", *STAGES):
        raise ValueError(f"Unknown stage: {stage}")
    if workers < 1 or threads < 1:
        raise ValueError("workers and threads must be positive")
    commands = {
        "train": [[str(ROOT / ".venv/bin/python"), str(OUT / "code/training.py"), "--root", str(ROOT), "--output", str(OUT),
                   "--workers", str(workers), "--threads", str(threads)]],
        "validate": ["finalize.validate_training(out) [read-only]"],
        "analysis": ["analysis.run_analysis(out)"], "plots": ["plots.make_plots(out)"],
        "report": ["reporting.make_report(out)"],
        "pdf": [[str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/prepare_results_content.py"), "05"],
                [str(PDF_PYTHON), str(ROOT / "scripts/build_results_pdf.py"), str(OUT)]],
        "finalize": ["finalize.finalize(out)"],
    }
    return [{"stage": name, "operations": commands[name]} for name in (STAGES if stage == "all" else (stage,))]


def run(stage="all", workers=4, threads=1, dry_run=False):
    plan = execution_plan(stage, workers, threads)
    if dry_run:
        print(json.dumps(plan, indent=2))
        return plan
    # These values are set before importing any scientific stage modules.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[name] = str(threads)
    for item in plan:
        name = item["stage"]
        print(f"Starting {name}", flush=True)
        if name in ("train", "pdf"):
            for command in item["operations"]:
                if not Path(command[0]).is_file():
                    raise FileNotFoundError(f"Required Python interpreter is missing: {command[0]}")
                subprocess.run(command, cwd=ROOT, check=True)
        else:
            module_name, function = {
                "validate": ("finalize", "validate_training"), "analysis": ("analysis", "run_analysis"),
                "plots": ("plots", "make_plots"), "report": ("reporting", "make_report"),
                "finalize": ("finalize", "finalize"),
            }[name]
            result = getattr(load_module(OUT, module_name), function)(OUT)
            if name in ("validate", "finalize"):
                print(json.dumps(result, indent=2), flush=True)
        print(f"Completed {name}", flush=True)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES), default="all")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print stage operations without running or modifying anything")
    args = parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error("workers and threads must be positive")
    run(args.stage, args.workers, args.threads, args.dry_run)


if __name__ == "__main__":
    main()
