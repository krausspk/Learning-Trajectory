"""Fast, synthetic-array checks; never launch the 150 production trajectories."""
import copy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

MODULE_PATH = Path(__file__).with_name("training.py")
spec = importlib.util.spec_from_file_location("shuffle_training_under_test", MODULE_PATH)
t = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = t
spec.loader.exec_module(t)


def fixture(seed=101, shuffle=11001):
    gen = torch.Generator().manual_seed(890)
    arrays = (torch.randn(19, 784, generator=gen), torch.randint(10, (19,), generator=gen),
              torch.randn(11, 784, generator=gen), torch.randint(10, (11,), generator=gen))
    config = t.TrainingConfig(hidden_width=5, batch_size=4, epochs=3, test_batch_size=7)
    rs = t.RunSpec("toy", "mnist", 1, 1, seed, shuffle, "M1")
    model = t.baseline.build_model(seed, hidden_width=5)
    initial = {"W0": model[0].weight.detach().numpy().T.copy(), "W1": model[2].weight.detach().numpy().T.copy()}
    return rs, arrays, config, initial


def test_design_is_full_crossed_and_epoch_seed_ranges_disjoint():
    specs = t.run_specs()
    assert len(specs) == len({s.run_id for s in specs}) == 150
    for dataset, _, base in t.DATASETS:
        selected = [s for s in specs if s.dataset == dataset]
        assert {(s.init_id, s.shuffle_id) for s in selected} == {(i, j) for i in range(1, 11) for j in range(1, 6)}
        assert all(s.init_seed == base + s.init_id for s in selected)
    epoch_seeds = [s + e for s in t.SHUFFLE_SEEDS for e in range(10)]
    assert len(set(epoch_seeds)) == 50


def test_initialization_and_shuffle_are_decoupled():
    rs, arrays, config, initial = fixture()
    states1, states2, states3 = [], [], []
    a = t.train_arrays(rs, arrays, config, initial, states1)
    b = t.train_arrays(replace(rs, shuffle_seed=12001), arrays, config, initial, states2)
    other, _, _, initial2 = fixture(seed=102)
    c = t.train_arrays(other, arrays, config, initial2, states3)
    for key in ("W0", "W1"):
        assert np.array_equal(a[key][0], b[key][0])
        assert not np.array_equal(a[key][-1], b[key][-1])
        assert not np.array_equal(a[key][0], c[key][0])
    assert all(torch.equal(x["epoch_order"], z["epoch_order"]) for x, z in zip(states1, states3))
    assert not torch.equal(states1[0]["epoch_order"], states2[0]["epoch_order"])


def test_identical_factors_produce_bitwise_identical_trajectory():
    rs, arrays, config, initial = fixture()
    a = t.train_arrays(rs, arrays, config, initial)
    b = t.train_arrays(rs, arrays, config, initial)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    assert not a["bias0"].any() and not a["bias1"].any()
    assert not a["momentum_W0"][0].any()


@pytest.mark.parametrize("resume_index", [0, 1, 2, 3, 6])
def test_every_checkpoint_type_continues_bitwise_including_momentum(resume_index, tmp_path):
    rs, arrays, config, initial = fixture()
    full = []
    a = t.train_arrays(rs, arrays, config, initial, full)
    path = tmp_path / "state.pt"
    t._atomic_pt(path, full[:resume_index + 1])
    prefix = torch.load(path, weights_only=False)
    old_momentum = copy.deepcopy(prefix[-1]["optimizer_state_dict"])
    b = t.train_arrays(rs, arrays, config, initial, prefix)
    for key in a:
        assert np.array_equal(a[key], b[key]), key
    for pi, state in old_momentum["state"].items():
        assert torch.equal(prefix[resume_index]["optimizer_state_dict"]["state"][pi]["momentum_buffer"], state["momentum_buffer"])
    assert torch.equal(prefix[-1]["torch_rng_state"], full[-1]["torch_rng_state"])
    assert prefix[-1]["python_rng_state"] == full[-1]["python_rng_state"]
    assert np.array_equal(prefix[-1]["numpy_rng_state"][1], full[-1]["numpy_rng_state"][1])


def test_resume_rejects_changed_factors_or_order():
    rs, arrays, config, initial = fixture()
    states = []
    t.train_arrays(rs, arrays, config, initial, states)
    with pytest.raises(ValueError, match="factors"):
        t.train_arrays(replace(rs, shuffle_seed=12001), arrays, config, initial, copy.deepcopy(states[:2]))
    corrupt = copy.deepcopy(states[:2])
    corrupt[-1]["epoch_order"] = corrupt[-1]["epoch_order"].roll(1)
    with pytest.raises(ValueError, match="order"):
        t.train_arrays(rs, arrays, config, initial, corrupt)


def test_expected_metadata_tracks_both_factors_and_source_fingerprints():
    rs, _, config, _ = fixture()
    context = {"config": t.asdict(config), "fingerprint": "abc", "sources": {"initializations": {"M1": {"sha256": "original"}}}, "schedules": [{"sha256": "schedule"}]}
    metadata = t.expected_metadata(rs, context)
    assert metadata["init_seed"] == 101 and metadata["shuffle_seed"] == 11001
    assert metadata["context_fingerprint"] == "abc"
    assert metadata["source_initialization"]["sha256"] == "original"
    assert metadata["schedule"]["sha256"] == "schedule"


def test_halfepoch_schedule_is_original_21_checkpoints():
    labels, updates = t.baseline.checkpoint_schedule("main", t.TrainingConfig())
    assert np.array_equal(labels, np.arange(21))
    assert list(updates[:5]) == [0, 118, 235, 353, 470]
    assert updates[-1] == 2350


def test_partial_cache_checks_actual_content_and_source_fingerprint(tmp_path):
    rs, arrays, config, initial = fixture()
    states = []
    t.train_arrays(rs, arrays, config, initial, states)
    context = {"fingerprint": "code-data-config-source-initialization-hash"}
    payload = t.state_payload(rs, context, states[:3])
    path = tmp_path / "checkpoint.pt"
    t._atomic_pt(path, payload)
    restored = torch.load(path, weights_only=False)
    t.validate_payload(restored, rs, context)
    with pytest.raises(ValueError, match="provenance"):
        t.validate_payload(restored, rs, {"fingerprint": "forged-source"})
    restored["checkpoints"][1]["model_state_dict"]["0.weight"][0, 0] += 0.1
    with pytest.raises(ValueError, match="content hash"):
        t.validate_payload(restored, rs, context)


def test_complete_archive_validator_rejects_forged_sources_and_momentum(tmp_path):
    """A synthetic 21-checkpoint archive exercises the production schema validator."""
    rs = t.run_specs()[0]
    config = t.TrainingConfig()
    model = t.baseline.build_model(rs.init_seed)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.1, momentum=.9)
    initial = {"W0": model[0].weight.detach().numpy().T.copy(), "W1": model[2].weight.detach().numpy().T.copy()}
    orders = [t.image_order(rs.shuffle_seed, e, 60000) for e in range(10)]
    schedule = {"epoch_order_sha256": [t.array_hash(o.numpy()) for o, _ in orders]}
    context = {"config": t.asdict(config), "fingerprint": "source-code-and-data-hash",
               "sources": {"initializations": {"M1": {f"E0_{k}_sha256": t.array_hash(v) for k, v in initial.items()}}},
               "schedules": [schedule]}
    labels, updates = t.baseline.checkpoint_schedule("main", config)
    states = []
    for k, update in enumerate(updates):
        epoch = 0 if k == 0 else (k - 1) // 2
        position = int(update) - epoch * 235
        states.append({"checkpoint": k, "update": int(update), "epoch": epoch, "batch_position": position,
                       "spec": t.asdict(rs), "config": t.asdict(config), "accuracy": .1, "loss": 1.,
                       "model_state_dict": copy.deepcopy(model.state_dict()),
                       "optimizer_state_dict": copy.deepcopy(optimizer.state_dict()),
                       "epoch_order": orders[epoch][0], "order_generator_state": orders[epoch][1],
                       "epoch_order_sha256": schedule["epoch_order_sha256"][epoch],
                       "order_seed": rs.shuffle_seed + epoch, "init_seed": rs.init_seed, "shuffle_seed": rs.shuffle_seed,
                       "torch_rng_state": torch.get_rng_state(), "numpy_rng_state": np.random.get_state(),
                       "python_rng_state": t.random.getstate()})
    arrays = {k: np.repeat(v[None], 21, axis=0) for k, v in initial.items()}
    arrays.update({"momentum_W0": np.zeros_like(arrays["W0"]), "momentum_W1": np.zeros_like(arrays["W1"]),
                   "bias0": np.zeros((21, 64), np.float32), "bias1": np.zeros((21, 10), np.float32),
                   "accuracy": np.full(21, .1), "loss": np.ones(21), "checkpoints": labels, "updates": updates})
    path = tmp_path / "run.npz"
    t._atomic_pt(path.with_suffix(".pt"), t.state_payload(rs, context, states))
    metadata = {**t.expected_metadata(rs, context), "full_state_sha256": t.sha256_file(path.with_suffix(".pt"))}
    arrays["metadata"] = np.asarray(json.dumps(metadata))
    t._atomic_npz(path, **arrays)
    assert t.validate_archive(path, rs, context)["checkpoints"] == 21
    forged_context = {**context, "fingerprint": "forged-source"}
    with pytest.raises(ValueError, match="context_fingerprint"):
        t.validate_archive(path, rs, forged_context)
    arrays["momentum_W0"][3, 0, 0] = 1
    t._atomic_npz(path, **arrays)
    with pytest.raises(ValueError, match="momentum mismatch"):
        t.validate_archive(path, rs, context)
