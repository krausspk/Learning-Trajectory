import importlib.util
from pathlib import Path

import numpy as np
import pytest

SPEC=importlib.util.spec_from_file_location('representations_under_test',Path(__file__).resolve().parents[1]/'code/representations.py')
rep=importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rep)


def sample_gram_cka(x,y):
    n=len(x);center=np.eye(n)-np.ones((n,n))/n
    k=center@(x@x.T)@center;l=center@(y@y.T)@center
    return np.sum(k*l)/(np.linalg.norm(k,'fro')*np.linalg.norm(l,'fro'))


def test_feature_cka_matches_independent_sample_gram_reference():
    rng=np.random.RandomState(15);x=rng.normal(size=(17,5))+3;y=rng.normal(size=(17,4))-2
    xc,_,_=rep.centered_features(x);yc,_,_=rep.centered_features(y)
    distances,raw=rep.cka_cross([xc],[yc])
    assert np.isclose(raw[0,0],sample_gram_cka(x,y),atol=1e-13)
    assert np.isclose(distances[0,0],1-sample_gram_cka(x,y),atol=1e-13)


def test_batched_cross_pairs_and_candidate_reordering():
    rng=np.random.RandomState(16)
    queries=[rep.centered_features(rng.normal(size=(19,4)))[0] for _ in range(3)]
    candidates=[rep.centered_features(rng.normal(size=(19,4)))[0] for _ in range(3)]
    d,_=rep.cka_cross(queries,candidates)
    for i in range(3):
        for j in range(3):assert np.isclose(d[i,j],1-sample_gram_cka(queries[i],candidates[j]),atol=1e-13)
    changed,_=rep.cka_cross(queries,[candidates[i] for i in [2,0,1]])
    assert np.allclose(changed,d[:,[2,0,1]],atol=1e-13)


def test_standardized_hidden_is_positive_neuron_scale_invariant():
    rng=np.random.RandomState(17);x=np.maximum(rng.normal(size=(23,6)),0);x[:,2]=0
    scaled=x*np.array([.1,2,7,.5,30,4])
    xc,_,zeros=rep.centered_features(x,True);sc,_,zeros_scaled=rep.centered_features(scaled,True)
    assert zeros==zeros_scaled==1
    assert np.all(xc[:,2]==0)
    assert np.allclose(xc,sc,atol=1e-13)
    assert rep.cka_cross([xc],[sc])[0][0,0]<1e-13


def test_constant_columns_stay_zero_despite_mean_roundoff():
    x=np.column_stack((np.full(3000,.1),np.arange(3000)))
    xc,_,zeros=rep.centered_features(x,True)
    assert zeros==1 and np.all(xc[:,0]==0)
    assert np.isclose(xc[:,1].std(),1)


def test_zero_variance_representations_fail_without_substitute():
    for standard in (False,True):
        with pytest.raises(ValueError,match='zero variance'):
            rep.centered_features(np.full((17,10),.1),standard)
    with pytest.raises(ValueError,match='normalizers'):
        rep.cka_cross([np.zeros((17,10))],[np.ones((17,10))])


def test_no_neuron_correspondence_and_global_scale_invariance():
    rng=np.random.RandomState(18);x=rng.normal(size=(13,5));xc,_,_=rep.centered_features(x)
    changed=xc[:,[2,4,1,0,3]]*8
    d,_=rep.cka_cross([xc],[changed])
    assert d[0,0]<1e-13


def test_all_declared_endpoints_are_fixed():
    definitions=rep.definitions()
    assert tuple(definitions)==rep.METRIC_KEYS
    assert definitions[rep.PRIMARY_KEY]['inferential_role']=='prespecified_primary'
    assert all(d['alignment']=='none' for d in definitions.values())
    assert 'softmax' in definitions[rep.METRIC_KEYS[2]]['scaling_caution']
