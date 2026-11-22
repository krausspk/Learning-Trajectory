import copy
import json
from pathlib import Path
import struct

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from learning_trajectory.data import read_idx
from learning_trajectory import training
from learning_trajectory.training import (
    EARLY_UPDATES, RunSpec, TrainingConfig, build_model, checkpoint_schedule,
    cohort_specs, image_order, train_arrays,
)


def test_all_paper_cohorts_and_seeds():
    assert len(cohort_specs()) == 65
    assert [len(cohort_specs(name)) for name in ("main", "task", "early")] == [30, 20, 15]
    main = cohort_specs("main")
    assert [r.seed for r in main[:10]] == list(range(101, 111))
    task = cohort_specs("task")
    for i in range(10):
        assert task[2 * i].seed == task[2 * i + 1].seed == 401 + i
        assert task[2 * i].block == task[2 * i + 1].block


def test_checkpoint_schedule_matches_paper():
    labels, updates = checkpoint_schedule("main", TrainingConfig())
    assert np.array_equal(labels, np.arange(21))
    assert updates[:5].tolist() == [0, 118, 235, 353, 470]
    assert updates[-1] == 2350
    labels, updates = checkpoint_schedule("early", TrainingConfig(epochs=2))
    assert tuple(updates) == EARLY_UPDATES
    assert np.array_equal(labels, updates)


def test_task_pairs_have_identical_hidden_initialization_and_shuffle():
    identity = build_model(401, 10)
    parity = build_model(401, 2)
    assert torch.equal(identity[0].weight, parity[0].weight)
    for model in (identity, parity):
        assert not model[0].bias.requires_grad
        assert not model[2].bias.requires_grad
        assert torch.count_nonzero(model[0].bias) == 0
        assert torch.count_nonzero(model[2].bias) == 0
    assert torch.equal(image_order(401, 0, 60000), image_order(401, 0, 60000))
    assert not torch.equal(image_order(401, 0, 60000), image_order(401, 1, 60000))


def _small_arrays():
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(64, 784, generator=generator)
    y = torch.arange(64).remainder(10)
    return x, y, x[:16].clone(), y[:16].clone()


def test_training_repeats_bitwise_and_preserves_biases():
    config = TrainingConfig(epochs=2, batch_size=16, test_batch_size=8)
    spec = RunSpec("T", "main", "mnist", "identity", 91)
    first = train_arrays(spec, _small_arrays(), config)
    second = train_arrays(spec, _small_arrays(), config)
    assert first["W0"].shape == (5, 784, 64)
    assert first["W1"].shape == (5, 64, 10)
    for key in first:
        assert np.array_equal(first[key], second[key]), key
        assert np.isfinite(first[key]).all()
    assert not first["bias0"].any()
    assert not first["bias1"].any()
    assert not np.array_equal(first["W0"][0], first["W0"][-1])


def test_checkpoint_state_restores_exact_optimizer_trajectory():
    config = TrainingConfig(epochs=2, batch_size=16, test_batch_size=8)
    spec = RunSpec("T", "main", "mnist", "identity", 91)
    arrays = _small_arrays()
    states = []
    complete = train_arrays(spec, arrays, config, full_states=states)
    assert len(states) == 5
    state = states[1]  # First half-epoch checkpoint, after two optimizer updates.
    assert state["update"] == 2
    assert state["batch_position"] == 2
    model = build_model(spec.seed)
    model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=config.learning_rate, momentum=config.momentum)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer_state_dict"]))
    torch.set_rng_state(state["torch_rng_state"])
    for epoch in range(state["epoch"], config.epochs):
        order = state["epoch_order"] if epoch == state["epoch"] else image_order(spec.seed, epoch, len(arrays[0]))
        start_at = state["samples_consumed_this_epoch"] if epoch == state["epoch"] else 0
        for start in range(start_at, len(order), config.batch_size):
            selected = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            F.cross_entropy(model(arrays[0][selected]), arrays[1][selected]).backward()
            optimizer.step()
    assert np.array_equal(model[0].weight.detach().numpy().T, complete["W0"][-1])
    assert np.array_equal(model[2].weight.detach().numpy().T, complete["W1"][-1])
    # The first stored model is a true snapshot, unaffected by later optimization.
    assert np.array_equal(states[0]["model_state_dict"]["0.weight"].numpy().T, complete["W0"][0])


def test_parity_training_outputs_two_logits_and_same_initial_w0():
    config = TrainingConfig(epochs=1, batch_size=16)
    identity = train_arrays(RunSpec("I", "task", "mnist", "identity", 401, 401), _small_arrays(), config)
    parity = train_arrays(RunSpec("P", "task", "mnist", "parity", 401, 401), _small_arrays(), config)
    assert parity["W1"].shape == (3, 64, 2)
    assert np.array_equal(identity["W0"][0], parity["W0"][0])


def test_idx_decoder_checks_shape_magic_and_size(tmp_path):
    path = tmp_path / "labels"
    content = struct.pack(">II", 2049, 3) + bytes([0, 4, 9])
    path.write_bytes(content)
    assert read_idx(path, (3,)).tolist() == [0, 4, 9]
    with pytest.raises(ValueError, match="shape"):
        read_idx(path, (4,))
    path.write_bytes(content[:-1])
    with pytest.raises(ValueError, match="Truncated"):
        read_idx(path)
    path.write_bytes(b"abcd")
    with pytest.raises(ValueError, match="magic"):
        read_idx(path)


def _finished_archive_fixture(tmp_path):
    spec = RunSpec("T", "main", "mnist", "identity", 91)
    config = TrainingConfig(epochs=1)
    data_dir = tmp_path / "data"
    manifest = data_dir / "mnist" / "manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"dataset":"mnist","files":[]}\n')
    output_dir = tmp_path / "outputs"
    path = output_dir / "checkpoints" / "main" / "T.npz"
    path.parent.mkdir(parents=True)
    torch.save({"test_fixture": True}, path.with_suffix(".pt"))
    checkpoints,updates = checkpoint_schedule("main",config)
    count = len(checkpoints)
    metadata = {
        "config_hash": training._config_hash(spec,config),
        "training_source_sha256": training.sha256_file(training.__file__),
        "data_source_sha256": training.sha256_file(Path(training.__file__).with_name("data.py")),
        "dataset_manifest_sha256": training.sha256_file(manifest),
        "full_state_sha256": training.sha256_file(path.with_suffix(".pt")),
    }
    arrays = {
        "W0": np.zeros((count,784,64),dtype=np.float32),
        "W1": np.zeros((count,64,10),dtype=np.float32),
        "accuracy": np.zeros(count),"loss": np.ones(count),
        "bias0": np.zeros((count,64)),"bias1": np.zeros((count,10)),
        "checkpoints": checkpoints,"updates": updates,
    }
    np.savez_compressed(path,metadata=np.asarray(json.dumps(metadata)),**arrays)
    return path,spec,config,data_dir,output_dir,metadata,arrays


@pytest.mark.parametrize("field",["training_source_sha256","data_source_sha256","dataset_manifest_sha256"])
def test_archive_rejects_stale_or_forged_provenance(tmp_path,field):
    path,spec,config,data_dir,_,metadata,arrays = _finished_archive_fixture(tmp_path)
    assert training.validate_archive(path,spec,config,data_dir)["checkpoints"] == 3
    metadata[field] = "0"*64
    np.savez_compressed(path,metadata=np.asarray(json.dumps(metadata)),**arrays)
    with pytest.raises(ValueError,match=field+".*--force"):
        training.validate_archive(path,spec,config,data_dir)


def test_archive_rejects_changed_dataset_manifest(tmp_path):
    path,spec,config,data_dir,_,_,_ = _finished_archive_fixture(tmp_path)
    (data_dir / "mnist" / "manifest.json").write_text('{"changed":true}\n')
    with pytest.raises(ValueError,match="dataset_manifest_sha256"):
        training.validate_archive(path,spec,config,data_dir)


def test_cache_reuse_validates_dataset_bytes_before_archive(tmp_path,monkeypatch):
    path,spec,config,data_dir,output_dir,_,_ = _finished_archive_fixture(tmp_path)
    calls = []
    original_validate = training.validate_archive
    monkeypatch.setattr(training,"ensure_dataset",lambda name,directory: calls.append("dataset"))
    def checked_validate(*args,**kwargs):
        assert calls == ["dataset"]
        calls.append("archive")
        return original_validate(*args,**kwargs)
    monkeypatch.setattr(training,"validate_archive",checked_validate)
    result = training.train_run(spec,data_dir,output_dir,config)
    assert result["resumed"] is True
    assert calls == ["dataset","archive"]


def test_corrupt_cached_archive_is_rejected_with_recovery_instructions(tmp_path,monkeypatch):
    path,spec,config,data_dir,output_dir,_,_ = _finished_archive_fixture(tmp_path)
    path.write_bytes(b"not an npz archive")
    monkeypatch.setattr(training,"ensure_dataset",lambda *_: None)
    with pytest.raises(ValueError,match="Cannot reuse cached run.*--force"):
        training.train_run(spec,data_dir,output_dir,config)
