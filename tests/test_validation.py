import json

import numpy as np
import pytest

from learning_trajectory import metrics as m
from learning_trajectory.validation import (
    _direct_alpha, _direct_cka, _logits, apply_multilayer_permutations,
    full_dissimilarity, iterative_match, rescale_network, synthetic_validation,
)


def test_neuronwise_rescaling_preserves_function_and_canonical_weights():
    rng = np.random.RandomState(23)
    a,b = rng.normal(size=(13,5)),rng.normal(size=(5,4))
    x = rng.normal(size=(19,13))
    ca = m.canonicalize(a,b)
    for envelope in (1,1.25,1.5,2,4,8):
        a2,b2,scales = rescale_network(a,b,envelope,42)
        assert np.exp(np.log(scales).mean()) == pytest.approx(1,abs=1e-15)
        assert np.min(scales) == pytest.approx(1/envelope)
        assert np.max(scales) == pytest.approx(envelope)
        np.testing.assert_allclose(_logits(x,[a,b]),_logits(x,[a2,b2]),atol=1e-13)
        for original,transformed in zip(ca,m.canonicalize(a2,b2)):
            np.testing.assert_allclose(original,transformed,atol=1e-14)
    with pytest.raises(ValueError): rescale_network(a,b,0.5,42)


def test_multilayer_permutations_preserve_logits_and_compose():
    rng = np.random.RandomState(9)
    widths = [20,12,10,8,6]
    weights = [rng.normal(size=(a,b)) for a,b in zip(widths[:-1],widths[1:])]
    permutations = [rng.permutation(n) for n in widths[1:-1]]
    transformed = apply_multilayer_permutations(weights,permutations)
    recovered = apply_multilayer_permutations(transformed,[np.argsort(p) for p in permutations])
    for a,b in zip(weights,recovered): assert np.array_equal(a,b)
    x = rng.normal(size=(20,20))
    np.testing.assert_allclose(_logits(x,weights),_logits(x,transformed),atol=2e-12)


@pytest.mark.parametrize("metric",["euclidean","snd","alpha"])
def test_iterative_matching_recovers_exact_multilayer_permutation(metric):
    rng = np.random.RandomState(51)
    widths = [20,12,10,8,6]
    weights = [rng.normal(size=(a,b)) for a,b in zip(widths[:-1],widths[1:])]
    permutations = [rng.permutation(n) for n in widths[1:-1]]
    target = apply_multilayer_permutations(weights,permutations)
    aligned,recovered,info = iterative_match(weights,target,metric)
    assert full_dissimilarity(weights,aligned,metric) == pytest.approx(0,abs=1e-15)
    for p,q in zip(recovered,permutations): assert np.array_equal(p,np.argsort(q))
    assert info["passes"] <= 30


def test_independent_scalar_alpha_and_sample_gram_cka_agree():
    rng = np.random.RandomState(55)
    a,b = rng.normal(size=(23,9)),rng.normal(size=(23,9))
    assert _direct_alpha(a,b) == pytest.approx(m.dissimilarity(a,b,"alpha"),abs=1e-14)
    assert _direct_cka(a,b) == pytest.approx(m.linear_cka(a,b),abs=1e-14)


def test_synthetic_noisy_case_is_auditable_and_improves_all_metrics(tmp_path):
    result = synthetic_validation(tmp_path)
    assert result["architecture"] == [20,12,10,8,6]
    assert result["seed"] == 2026
    assert result["noise_sd_fraction"] == 0.3
    assert len(result["results"]) == 3
    for row in result["results"]:
        assert row["after"] < row["before"]
        assert row["logit_error"] < 1e-10
    persisted = json.loads((tmp_path / "validation" / "synthetic.json").read_text())
    assert persisted == result
    with np.load(tmp_path / "validation" / "synthetic_weights.npz",allow_pickle=False) as data:
        for i in range(4):
            assert np.isfinite(data[f"reference_W{i}"]).all()
            assert np.isfinite(data[f"target_W{i}"]).all()
