import itertools

import numpy as np
import pytest

from learning_trajectory.metrics import (
    activation_correspondence, align, align_incoming, apply_permutation,
    canonicalize, dissimilarity, incoming_cost, layers, linear_cka,
    pif_assignment, profile_cost, weight_agreement,
)


def network(seed=17, width=7):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(23, width)), rng.normal(size=(width, 5))


def test_canonical_gauge_preserves_function_and_removes_rescaling():
    a0, a1 = network()
    rng = np.random.default_rng(19)
    x = rng.normal(size=(31, a0.shape[0]))
    scaling = np.exp(rng.uniform(-4, 4, a0.shape[1]))
    c0, c1 = canonicalize(a0, a1)
    d0, d1 = canonicalize(a0 * scaling, a1 / scaling[:, None])
    np.testing.assert_allclose(c0, d0, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(c1, d1, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(c0, axis=0), np.linalg.norm(c1, axis=1))
    np.testing.assert_allclose(np.maximum(x @ a0, 0) @ a1,
                               np.maximum(x @ c0, 0) @ c1, atol=1e-12)


def test_permutation_preserves_logits_and_alignment_recovers_known_neurons():
    a0, a1 = network()
    original_permutation = np.random.default_rng(2).permutation(a0.shape[1])
    b0, b1 = apply_permutation(a0, a1, original_permutation)
    x = np.random.default_rng(4).normal(size=(10, a0.shape[0]))
    np.testing.assert_allclose(np.maximum(x @ a0, 0) @ a1,
                               np.maximum(x @ b0, 0) @ b1, atol=1e-12)
    for method in ("canonical_alpha", "raw_alpha", "euclidean", "snd", "pif", "benchmark", "git_rebasin"):
        p = align(a0, a1, b0, b1, method)
        np.testing.assert_array_equal(original_permutation[p], np.arange(a0.shape[1]))
        result = layers(a0, a1, b0, b1, p, "euclidean")
        assert result["balanced"] == 0


@pytest.mark.parametrize("metric", ["alpha", "euclidean", "snd"])
def test_vectorized_profile_cost_equals_independent_pairwise_definition(metric):
    a0, a1 = network(18, 5)
    b0, b1 = network(19, 5)
    a0[0, 0] = b0[0, 1] = 0
    costs = profile_cost(a0, a1, b0, b1, metric)
    for i, j in itertools.product(range(5), repeat=2):
        a, b = np.r_[a0[:, i], a1[i]], np.r_[b0[:, j], b1[j]]
        assert costs[i, j] == pytest.approx(dissimilarity(a, b, metric), abs=1e-14)
    np.testing.assert_allclose(costs.T, profile_cost(b0, b1, a0, a1, metric), atol=1e-14)


def test_alpha_weighted_sign_disagreement_and_zero_conventions():
    a, b = np.array([1., 2., 0.]), np.array([-2., 3., 9.])
    assert dissimilarity(a, b) == pytest.approx(4 / 40)
    assert dissimilarity(a, 3 * a) == 0
    assert dissimilarity(a, b) == pytest.approx(dissimilarity(5 * a, 7 * b))
    assert dissimilarity(np.zeros(3), b) == 0
    assert dissimilarity(a * 1e-20, b) == 0
    assert dissimilarity(a, b) == dissimilarity(b, a)


def test_snd_absolute_fraction_sign_and_zero_rules():
    a, b = np.array([1., -2., 0., 0., 3., 1e-15]), np.array([3., -1., 0., 2., -4., 1e-15])
    assert dissimilarity(a, b, "snd") == pytest.approx((.5 + 1/3 + 0 + 1 + 1 + 1) / 6)
    assert dissimilarity(a, b, "snd") == dissimilarity(b, a, "snd")


def test_git_rebasin_is_exact_assignment_optimum():
    a0, a1 = network(12, 4)
    b0, b1 = network(13, 4)
    p = align(a0, a1, b0, b1, "git_rebasin")
    score = weight_agreement(a0, a1, b0, b1, p)["inner_product"]
    exhaustive = [weight_agreement(a0, a1, b0, b1, np.asarray(q))["inner_product"]
                  for q in itertools.permutations(range(4))]
    assert score == pytest.approx(max(exhaustive))


def test_canonical_alignment_and_benchmark_selection():
    a0, a1 = network(21)
    b0, b1 = network(22)
    scales_a, scales_b = np.exp(np.arange(7) / 3), np.exp(-np.arange(7) / 4)
    for method in ("canonical_alpha", "benchmark"):
        p = align(a0, a1, b0, b1, method)
        transformed = align(a0 * scales_a, a1 / scales_a[:, None],
                            b0 * scales_b, b1 / scales_b[:, None], method)
        np.testing.assert_array_equal(p, transformed)
    c0, c1 = canonicalize(a0, a1)
    d0, d1 = canonicalize(b0, b1)
    candidates = [pif_assignment(c0, c1, d0, d1), align(a0, a1, b0, b1)]
    minimum = min(layers(c0, c1, d0, d1, p)["size_weighted"] for p in candidates)
    chosen = align(a0, a1, b0, b1, "benchmark")
    assert layers(c0, c1, d0, d1, chosen)["size_weighted"] == minimum


def test_incoming_assignment_does_not_require_equal_output_dimensions():
    a0, _ = network()
    q = np.random.default_rng(4).permutation(a0.shape[1])
    for metric in ("alpha", "euclidean", "snd"):
        p = align_incoming(a0, a0[:, q], metric)
        np.testing.assert_array_equal(q[p], np.arange(len(p)))
        assert incoming_cost(a0, a0[:, q], metric).shape == (7, 7)


def test_cka_matches_centered_gram_definition_and_is_orthogonally_invariant():
    rng = np.random.default_rng(23)
    a, b = rng.normal(size=(30, 6)), rng.normal(size=(30, 8))
    ca, cb = a - a.mean(axis=0), b - b.mean(axis=0)
    ka, kb = ca @ ca.T, cb @ cb.T
    expected = np.sum(ka * kb) / (np.linalg.norm(ka) * np.linalg.norm(kb))
    assert linear_cka(a, b) == pytest.approx(expected)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert linear_cka(a, 3 * a @ q + np.arange(6)) == pytest.approx(1)
    assert linear_cka(a, b) == pytest.approx(linear_cka(b, a))
    with pytest.raises(ValueError, match="zero centered variance"):
        linear_cka(a, np.ones((30, 6)))


def test_matched_activation_validation_is_independent_of_weight_objectives():
    a = np.random.default_rng(15).normal(size=(90, 8))
    p = np.array([4, 7, 3, 2, 0, 1, 5, 6])
    b = a[:, p] * np.arange(1, 9) + 3
    assert activation_correspondence(a, b, np.argsort(p)) == pytest.approx(1)
    assert activation_correspondence(a, b) < .5
    a[:, 0] = 0
    assert activation_correspondence(a, a) == pytest.approx(7 / 8)


def test_layer_weighting_and_invalid_network_inputs():
    a0, a1 = network()
    b0, b1 = a0 + .1, a1 + .3
    result = layers(a0, a1, b0, b1, metric="euclidean")
    assert result["balanced"] == pytest.approx(.2)
    assert result["size_weighted"] == pytest.approx((23 * .1 + 5 * .3) / 28)
    with pytest.raises(ValueError):
        canonicalize(a0 * 0, a1)
    with pytest.raises(ValueError):
        apply_permutation(a0, a1, np.zeros(7, dtype=int))
