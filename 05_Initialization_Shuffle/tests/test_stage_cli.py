"""Stage dispatch/integrity unit checks; no real training or analysis is run."""
from pathlib import Path
import importlib.util
import json
import sys

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"


def module(name):
    spec = importlib.util.spec_from_file_location(f"_stage_test_{name}", CODE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = result
    spec.loader.exec_module(result)
    return result


def test_all_orders_training_before_validation_and_analysis():
    run = module("run")
    plan = run.execution_plan("all")
    assert [p["stage"] for p in plan] == ["train", "validate", "analysis", "plots", "report", "pdf", "finalize"]
    assert "--force" not in plan[0]["operations"][0]
    assert "--workers" in plan[0]["operations"][0]


def test_pdf_commands_match_real_positional_clis():
    run = module("run")
    commands = run.execution_plan("pdf")[0]["operations"]
    assert commands[0] == [str(run.ROOT / ".venv/bin/python"), str(run.ROOT / "scripts/prepare_results_content.py"), "05"]
    assert commands[1] == [str(run.PDF_PYTHON), str(run.ROOT / "scripts/build_results_pdf.py"), str(run.OUT)]


def test_dry_run_does_not_import_or_execute_stages(monkeypatch, capsys):
    run = module("run")
    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run executed a stage")
    monkeypatch.setattr(run, "load_module", forbidden)
    monkeypatch.setattr(run.subprocess, "run", forbidden)
    run.run("all", dry_run=True)
    assert len(json.loads(capsys.readouterr().out)) == 7


def test_validate_dispatches_readonly_checker_not_training(monkeypatch):
    run = module("run")
    calls = []
    class Fake:
        def validate_training(self, out):
            calls.append(out)
            return {"passed": True}
    monkeypatch.setattr(run, "load_module", lambda out, name: Fake() if name == "finalize" else pytest.fail(name))
    monkeypatch.setattr(run.subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    run.run("validate")
    assert calls == [run.OUT]


def test_invalid_stage_or_parallelism_rejected():
    run = module("run")
    with pytest.raises(ValueError):
        run.execution_plan("unknown")
    with pytest.raises(ValueError):
        run.execution_plan("train", workers=0)


def test_preservation_hashes_bytes_and_retains_authorized_additions(tmp_path):
    final = module("finalize")
    old = tmp_path / "old/checkpoint.npz"
    old.parent.mkdir()
    old.write_bytes(b"unchanged checkpoint")
    out = tmp_path / "05"
    (out / "provenance").mkdir(parents=True)
    inventory = {"files": [{"path": "old/checkpoint.npz", "bytes": old.stat().st_size, "sha256": final.sha(old)}],
                 "authorized_additions": "new results PDF"}
    final.write_json(out / "provenance/source_inventory_before.json", inventory)
    (old.parent / "results.pdf").write_bytes(b"authorized addition")
    assert final.preservation_check(out, expected_count=1)["all_unchanged"]
    old.write_bytes(b"changed checkpoint!!")
    assert final.preservation_check(out, expected_count=1)["changed"] == ["old/checkpoint.npz"]


def test_artifact_manifest_excludes_logs_caches_and_itself(tmp_path):
    final = module("finalize")
    for name in ("data/run.pt", "code/run.py", "results.pdf", "results_qa/receipt.json", "logs/test.txt", "code/__pycache__/run.pyc", ".pytest_cache/info", "partial.tmp"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    final.write_json(tmp_path / "artifact_manifest.json", {"old": True})
    assert final.artifact_manifest(tmp_path) == 4
    files = json.loads((tmp_path / "artifact_manifest.json").read_text())["files"]
    assert {r["path"] for r in files} == {"data/run.pt", "code/run.py", "results.pdf", "results_qa/receipt.json"}
    assert all(final.sha(tmp_path / r["path"]) == r["sha256"] for r in files)


def test_stale_or_incomplete_training_receipt_rejected_before_archive_io(tmp_path):
    final = module("finalize")
    for name, value in (("data/manifest.json", {}), ("training_plan.json", {}),
                        ("provenance/training_validation.json", {"passed": True, "validated_runs": 149, "expected_runs": 150})):
        final.write_json(tmp_path / name, value)
    with pytest.raises(ValueError, match="incomplete"):
        final.validate_training(tmp_path)


def test_source_hash_verification_rejects_changed_content(tmp_path):
    final = module("finalize")
    p = tmp_path / "source"
    p.write_text("before")
    digest = final.sha(p)
    p.write_text("after")
    with pytest.raises(ValueError, match="Changed"):
        final.require_hash(p, digest, "analysis source")


def test_visual_receipt_rejects_stale_figure_bytes(tmp_path):
    final = module("finalize")
    figure = tmp_path / "figure.png"
    figure.write_bytes(b"visually reviewed version")
    receipt = {"status": "passed", "figures": [{"png": "figure.png", "png_sha256": final.sha(figure)}]}
    final.write_json(tmp_path / "provenance/visual_qa/receipt.json", receipt)
    assert final.presentation_status(tmp_path)["qa_receipts"]["provenance/visual_qa/receipt.json"]["status"] == "passed"
    figure.write_bytes(b"changed after visual review")
    with pytest.raises(ValueError, match="visually reviewed figure"):
        final.presentation_status(tmp_path)
