import importlib.util
import itertools
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("initialization_retrieval_distances_tests",Path(__file__).with_name("distances.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def networks(count=3,seed=19):
    rng = np.random.RandomState(seed)
    return [(rng.normal(size=(7,4)),rng.normal(size=(4,3))) for _ in range(count)]


def test_hungarian_retrieval_matches_exhaustive_assignment():
    query,candidate = networks(2)
    actual,p = d.retrieval_distance(query,candidate)
    (a0,a1),_ = d.canonical_unit_network(*query)
    (b0,b1),_ = d.canonical_unit_network(*candidate)
    brute = min(np.sqrt(np.sum((a0-b0[:,q])**2)+np.sum((a1-b1[list(q),:])**2)) for q in itertools.permutations(range(4)))
    assert actual == pytest.approx(brute,abs=1e-14)
    assert np.array_equal(np.sort(p),np.arange(4))


def test_neuron_reindexing_and_positive_relu_gauge_leave_distance_unchanged():
    query,candidate = networks(2)
    baseline,_ = d.retrieval_distance(query,candidate)
    p = np.array([2,0,3,1])
    scales = np.array([0.03,0.8,5.,30.])
    transformed = (candidate[0][:,p]*scales,candidate[1][p,:]/scales[:,None])
    value,_ = d.retrieval_distance(query,transformed)
    assert value == pytest.approx(baseline,abs=1e-14)
    self_value,_ = d.retrieval_distance(candidate,transformed)
    assert self_value < 1e-14


def test_unit_norm_and_cosine_equivalence_use_both_layers_together():
    query,candidate = networks(2)
    q,_ = d.canonical_unit_network(*query)
    c,_ = d.canonical_unit_network(*candidate)
    for a,b in (q,c):
        assert np.sum(a*a)+np.sum(b*b) == pytest.approx(1,abs=1e-14)
    value,p = d.match_unit_networks(q,c)
    cosine = np.sum(q[0]*c[0][:,p])+np.sum(q[1]*c[1][p,:])
    assert value**2 == pytest.approx(2*(1-cosine),abs=1e-14)


def test_cross_time_orientation_rows_are_queries_columns_candidates():
    initial = networks(3)
    current = networks(2,seed=91)
    values,permutations = d.cross_distances(current,initial)
    assert values.shape == (2,3)
    assert permutations.shape == (2,3,4)
    for i in range(2):
        for j in range(3):
            expected,p = d.retrieval_distance(current[i],initial[j])
            assert values[i,j] == pytest.approx(expected)
            assert np.array_equal(permutations[i,j],p)


def test_E0_identity_is_computed_normally_and_candidate_reordering_moves_matches():
    initial = networks(3)
    values,_ = d.cross_distances(initial,initial)
    np.testing.assert_array_equal(np.diag(values),np.zeros(3))
    np.testing.assert_allclose(values,values.T,atol=1e-14)
    order = [2,0,1]
    reordered,_ = d.cross_distances(initial,[initial[i] for i in order])
    np.testing.assert_allclose(reordered,values[:,order],atol=1e-14)
    assert reordered.argmin(axis=1).tolist() == [1,2,0]
    assert np.all(np.diag(reordered)>0)  # There is no diagonal preference.


def test_zero_neurons_are_supported_but_zero_complete_vector_is_undefined():
    a,b = networks(1)[0]
    a[:,0],b[0,:] = 0,0
    unit,counts = d.canonical_unit_network(a,b)
    assert counts["both_zero"]==1
    assert np.isfinite(unit[0]).all() and np.isfinite(unit[1]).all()
    with pytest.raises(ValueError,match="positive finite norm"):
        d.canonical_unit_network(np.zeros((7,4)),np.zeros((4,3)))
