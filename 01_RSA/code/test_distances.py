import importlib.util
import itertools
import json
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("rsa_distances",Path(__file__).with_name("distances.py"))
d = importlib.util.module_from_spec(spec)
spec.loader.exec_module(d)


def test_canonical_gauge_zero_cases_are_finite_and_function_preserving():
    a = np.array([[3.,2.,0.,0.],[4.,0.,0.,0.]])
    b = np.array([[6.,8.],[0.,0.],[0.,4.],[0.,0.]])
    ca,cb,counts = d.canonical_gauge(a,b)
    assert counts == {"both_positive":1,"incoming_only":1,"outgoing_only":1,"both_zero":1}
    assert np.linalg.norm(ca[:,0]) == pytest.approx(np.linalg.norm(cb[0]))
    assert np.linalg.norm(ca[:,1]) == pytest.approx(1)
    assert np.linalg.norm(cb[2]) == pytest.approx(1)
    x = np.array([[1.,2.],[-2.,1.],[0.,0.]])
    np.testing.assert_allclose(np.maximum(x@a,0)@b,np.maximum(x@ca,0)@cb,atol=1e-14)
    assert np.isfinite(ca).all() and np.isfinite(cb).all()


def test_gauge_matches_pipeline_and_removes_positive_scaling():
    rng = np.random.RandomState(12)
    a,b = rng.normal(size=(7,4)),rng.normal(size=(4,3))
    ca,cb,_ = d.canonical_gauge(a,b)
    pa,pb = d.pipeline_metrics.canonicalize(a,b)
    assert np.array_equal(ca,pa) and np.array_equal(cb,pb)
    scale = np.array([0.01,2.,5.,100.])
    sa,sb,_ = d.canonical_gauge(a*scale,b/scale[:,None])
    np.testing.assert_allclose(ca,sa,atol=1e-14)
    np.testing.assert_allclose(cb,sb,atol=1e-14)


def test_joint_git_assignment_is_global_minimum_by_exhaustive_check():
    rng = np.random.RandomState(3)
    a0,a1 = rng.normal(size=(5,4)),rng.normal(size=(4,3))
    b0,b1 = rng.normal(size=(5,4)),rng.normal(size=(4,3))
    p = d.git_permutation(a0,a1,b0,b1)
    actual = d.geometric_distances(a0,a1,b0,b1,p)[("frobenius","total")]
    brute = min(d.geometric_distances(a0,a1,b0,b1,np.array(q))[("frobenius","total")] for q in itertools.permutations(range(4)))
    assert actual == pytest.approx(brute,abs=1e-14)
    reverse = d.git_permutation(b0,b1,a0,a1)
    assert d.geometric_distances(b0,b1,a0,a1,reverse)[("frobenius","total")] == pytest.approx(actual,abs=1e-14)


def test_balanced_and_normalized_definitions_are_exact():
    rng = np.random.RandomState(14)
    a0,a1 = rng.normal(size=(5,4)),rng.normal(size=(4,3))
    b0,b1 = rng.normal(size=(5,4)),rng.normal(size=(4,3))
    p = np.array([1,3,0,2])
    values = d.geometric_distances(a0,a1,b0,b1,p)
    assert values[("frobenius","balanced")] == pytest.approx(np.sqrt(.5*(np.mean((a0-b0[:,p])**2)+np.mean((a1-b1[p])**2))))
    a = np.r_[a0.ravel(),a1.ravel()]
    b = np.r_[b0[:,p].ravel(),b1[p].ravel()]
    assert values[("normalized_frobenius","total")] == pytest.approx(np.linalg.norm(a/np.linalg.norm(a)-b/np.linalg.norm(b)))


def test_cycle_composition_and_assignment_change_conventions():
    rng = np.random.RandomState(1)
    neuron_orders = [rng.permutation(4) for _ in range(10)]
    pairs = np.array(list(itertools.combinations(range(10),2)))
    pair_permutations = np.array([np.argsort(neuron_orders[b])[neuron_orders[a]] for a,b in pairs])
    permutations = np.repeat(pair_permutations[None,:,:],21,axis=0)
    changes,cycles = d.assignment_diagnostics("test",np.array([str(i) for i in range(10)]),pairs,permutations)
    assert len(changes)==21*45 and len(cycles)==2*120
    assert all(row["change_fraction_vs_E0"]==row["change_fraction_vs_E20"]==0 for row in changes)
    assert all(row["exactly_consistent"] and row["fraction_consistent"]==1 for row in cycles)


def test_metric_schema_has_primary_and_52_unique_variants():
    definitions = d.definitions()
    assert len(definitions)==52
    assert d.PRIMARY_KEY in definitions
    for key,value in definitions.items():
        assert key.split("|")==[value["metric"],value["layer"],value["alignment"],value["gauge"]]
    assert "BOTH" in definitions["frobenius|total|fixed_E20|canonical"]["assignment_timing"]


def test_incomplete_manifest_does_not_substitute_other_experiment(tmp_path):
    folder = tmp_path/"outputs/manifests"
    folder.mkdir(parents=True)
    (folder/"main.json").write_text(json.dumps({"cohort":"main","complete":False,"expected_runs":30,"runs":[]}))
    with pytest.raises(ValueError,match="not complete"):
        d.load_manifest_runs(tmp_path)
