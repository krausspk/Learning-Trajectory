"""Small independent reference tests; no experimental controls are computed."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "code/distances.py"
SPEC = importlib.util.spec_from_file_location("controls_distances_under_test", PATH)
dist = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dist)


def network(seed, width=4):
    rng = np.random.RandomState(seed)
    return rng.normal(size=(7,width)), rng.normal(size=(width,3))


def masks(width=4):
    return {"low":np.arange(width)<2,"high":np.arange(width)>=2}


def test_protocols_and_joint_layer_decomposition():
    values, assignments, thresholds, counts = dist.pair_controls(network(1),network(2),masks())
    assert set(values)==set(dist.definitions()) and len(values)==42
    assert np.isfinite(list(values.values())).all()
    assert np.all(np.asarray(list(values.values()))>=0)
    for gauge in ("raw","canonical"):
        for alignment in ("identity","optimal"):
            for metric in ("frobenius","normalized_frobenius"):
                val=[values[f"{metric}|{layer}|{alignment}|{gauge}"] for layer in dist.LAYERS]
                assert np.isclose(val[2]**2,val[0]**2+val[1]**2)
    assert thresholds.shape==counts.shape==(2,2)
    assert all(np.array_equal(np.sort(p),np.arange(4)) for p in assignments.values())


def test_primary_matches_point1_and_canonical_gauge_invariance():
    a,b=network(3),network(4)
    values,*_=dist.pair_controls(a,b,masks())
    reference,_=dist.RETRIEVAL.retrieval_distance(a,b)
    assert np.isclose(values[dist.PRIMARY_KEY],reference,atol=1e-14)
    p=np.array([2,0,3,1]);scale=np.array([.2,4.,1.7,.6])
    transformed=(b[0][:,p]*scale,b[1][p,:]/scale[:,None])
    changed,*_=dist.pair_controls(a,transformed,masks())
    assert np.isclose(changed[dist.PRIMARY_KEY],reference,atol=1e-14)


def test_masked_assignment_is_global_squared_distance_minimum():
    a,b=dist.unit_network(network(11)),dist.unit_network(network(12))
    mask=np.array([True,False,True,False])
    p=dist.masked_permutation(a,b,mask)
    measured=dist.layer_distances(a,b,p,mask)[2]
    brute=min(dist.layer_distances(a,b,np.asarray(order),mask)[2]
              for order in itertools.permutations(range(4)))
    assert np.isclose(measured,brute,atol=1e-14)


def test_activity_selection_uses_only_query_and_preserves_all_ties():
    h=np.array([[0.,1.,0.,2.],[0.,0.,0.,3.],[0.,1.,0.,0.],[0.,0.,0.,4.]])
    chosen,rate,thresholds=dist.activity_masks(h)
    assert np.array_equal(rate,[0.,.5,0.,.75])
    assert np.array_equal(chosen["inactive"],[True,False,True,False])
    assert np.array_equal(chosen["low"],[True,False,True,False])
    p=np.array([3,1,0,2])
    transformed,_,_=dist.activity_masks(h[:,p]*np.array([.2,4.,2.,.1]))
    for name in chosen: assert np.array_equal(transformed[name],chosen[name][p])
    all_dead,_,_=dist.activity_masks(np.zeros((5,4)))
    assert all(mask.all() for mask in all_dead.values())


def test_candidate_reordering_does_not_change_pair_costs():
    query=network(5);candidates=[network(6),network(7),network(8)]
    baseline=[dist.pair_controls(query,c,masks())[0] for c in candidates]
    order=[2,0,1]
    reordered=[dist.pair_controls(query,candidates[j],masks())[0] for j in order]
    for item,j in zip(reordered,order):
        for key in baseline[j]: assert item[key]==baseline[j][key]


def test_query_neuron_permutation_equivariance_for_masked_controls():
    a,b=network(21),network(22);selection=masks()
    before=dist.pair_controls(a,b,selection)[0]
    p=np.array([2,3,1,0])
    transformed=(a[0][:,p],a[1][p,:])
    after=dist.pair_controls(transformed,b,{k:m[p] for k,m in selection.items()})[0]
    for key in before:
        if "masked_optimal" in key or "|optimal|" in key:
            assert np.isclose(before[key],after[key],atol=1e-13)


def test_residual_quantiles_are_candidate_specific_and_tie_safe():
    a=network(30)
    equal,thresholds,counts=dist.residual_quantile_distances(a,a,np.arange(4))
    assert np.array_equal(equal,np.zeros((2,3)))
    assert np.array_equal(thresholds,np.zeros((2,2)))
    assert np.array_equal(counts,[[28,28],[12,12]])
    b=network(31)
    values,thresholds,counts=dist.residual_quantile_distances(a,b,np.arange(4))
    for li,(x,y) in enumerate(zip(a,b)):
        absolute=np.abs(x-y).ravel()
        for si,mask in enumerate((absolute<=np.quantile(absolute,.25),absolute>=np.quantile(absolute,.75))):
            assert counts[li,si]==mask.sum()
            assert np.isclose(values[si,li],np.sqrt(np.mean(absolute[mask]**2)))
