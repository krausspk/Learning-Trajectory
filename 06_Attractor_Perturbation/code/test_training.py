"""Meaningful intervention, momentum and deterministic continuation checks."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch

from learning_trajectory.training import build_model, RunSpec, TrainingConfig, train_arrays

spec = importlib.util.spec_from_file_location("perturb_training", Path(__file__).with_name("training.py"))
fork = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fork)


def arrays():
    generator = torch.Generator().manual_seed(71)
    x = torch.randn(64, 784, generator=generator)
    y = torch.arange(64).remainder(10)
    return x, y, x[:16].clone(), y[:16].clone()


def setup_checkpoint():
    cfg = TrainingConfig(epochs=2, batch_size=16, test_batch_size=8)
    run = RunSpec("T", "main", "mnist", "identity", 91)
    states = []
    complete = train_arrays(run, arrays(), cfg, full_states=states)
    state = states[2]
    model = build_model(91)
    model.load_state_dict(state["model_state_dict"])
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.1, momentum=0.9)
    optimizer.load_state_dict(copy.deepcopy(state["optimizer_state_dict"]))
    return model, optimizer, state, complete


def test_full_epoch_boundary_replays_next_epoch_bitwise():
    model, optimizer, state, complete = setup_checkpoint()
    trajectory, final, failure = fork.continue_arrays(model, optimizer, state, arrays(), order_seed=91, recovery_epochs=1)
    assert failure is None
    assert trajectory["recovery_updates"].tolist() == [0, 1, 2, 4]
    for name in ("W0", "W1", "momentum_W0", "momentum_W1", "accuracy", "loss"):
        assert np.array_equal(trajectory[name][-1], complete[name][-1]), name
    assert final["epoch"] == 1
    assert final["batch_position"] == 4
    assert final["update"] == 8
    generator = torch.Generator().manual_seed(92)
    expected_order = torch.randperm(64, generator=generator)
    assert torch.equal(expected_order, final["epoch_order"])
    assert torch.equal(generator.get_state(), final["order_generator_state"])


def test_gaussian_norms_layers_and_shared_directions():
    model = build_model(19)
    initial = copy.deepcopy(model)
    only0, only1 = copy.deepcopy(model), copy.deepcopy(model)
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state().clone()
    norms = fork.perturb_weights(model, 0.2, "both", 81)
    fork.perturb_weights(only0, 0.2, "W0", 81)
    fork.perturb_weights(only1, 0.2, "W1", 81)
    for label in ("W0", "W1"):
        assert norms[label]["actual_relative_norm"] == pytest.approx(0.2, rel=1e-6)
    assert torch.equal(model[0].weight, only0[0].weight)
    assert torch.equal(model[2].weight, only1[2].weight)
    assert torch.equal(initial[2].weight, only0[2].weight)
    assert torch.equal(initial[0].weight, only1[0].weight)
    assert torch.equal(torch_state, torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(numpy_state,np.random.get_state()))


@pytest.mark.parametrize("momentum", ["preserve", "reset"])
def test_zero_intervention_twins_recover_identically(momentum):
    model, optimizer, state, _ = setup_checkpoint()
    model2 = copy.deepcopy(model)
    optimizer2 = torch.optim.SGD([p for p in model2.parameters() if p.requires_grad], lr=0.1, momentum=0.9)
    optimizer2.load_state_dict(copy.deepcopy(optimizer.state_dict()))
    if momentum == "reset":
        fork.reset_momentum(optimizer)
        fork.reset_momentum(optimizer2)
    fork.perturb_weights(model2, 0.0, "both", 9087)
    a, fa, ea = fork.continue_arrays(model,optimizer,state,arrays(),order_seed=91,recovery_epochs=1)
    b, fb, eb = fork.continue_arrays(model2,optimizer2,state,arrays(),order_seed=91,recovery_epochs=1)
    assert ea is None and eb is None
    for key in a:
        assert np.array_equal(a[key],b[key]), key
    assert fa["epoch_order_hashes"] == fb["epoch_order_hashes"]


def test_momentum_reset_and_shuffle_noise_are_distinct_controls():
    results = []
    for reset, seed in ((False,91),(True,91),(False,8101)):
        model, optimizer, state, _ = setup_checkpoint()
        if reset:
            fork.reset_momentum(optimizer)
        result, _, failure = fork.continue_arrays(model,optimizer,state,arrays(),order_seed=seed,recovery_epochs=1)
        assert failure is None
        results.append(result)
    assert not np.array_equal(results[0]["W0"][-1], results[1]["W0"][-1])
    assert not np.array_equal(results[0]["W0"][-1], results[2]["W0"][-1])
    assert not np.any(results[1]["momentum_W0"][0])


def test_divergence_is_recorded_and_not_a_success():
    model, optimizer, state, _ = setup_checkpoint()
    with torch.no_grad():
        model[0].weight[0,0] = float("nan")
    result, final, failure = fork.continue_arrays(model,optimizer,state,arrays(),order_seed=91,recovery_epochs=1)
    assert failure["kind"] == "nonfinite_weights"
    assert failure["recovery_update"] == 0
    assert final["failure"] == failure
    assert len(result["recovery_updates"]) == 0


def test_rejects_partial_epoch_checkpoint():
    model, optimizer, state, _ = setup_checkpoint()
    state = copy.deepcopy(state)
    state["batch_position"] = 1
    with pytest.raises(ValueError,match="completed-epoch"):
        fork.continue_arrays(model,optimizer,state,arrays(),order_seed=91,recovery_epochs=1)
