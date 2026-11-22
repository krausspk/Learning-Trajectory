"""Failure inclusion and fixed-region regressions for the perturbation analysis."""
import importlib.util
import io
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('perturbation_analysis',HERE/'analysis.py')
analysis=importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_failed_direction_is_not_silently_removed_from_seed_mean():
    frame=pd.read_csv(io.StringIO('group,seed,ratio,member,failed\na,1,0.5,1,0\na,1,,0,1\na,2,0.7,1,0\na,2,0.9,1,0\n'))
    result=analysis.bootstrap_summary(frame,['ratio','member','failed'],['group']).iloc[0]
    assert result.n_seeds==2 and result.n_runs==4
    assert np.isnan(result.ratio_mean) and np.isnan(result.ratio_low)
    assert result.member_mean==pytest.approx(.75)
    assert result.failed_mean==pytest.approx(.25)


def test_seed_bootstrap_weights_seeds_equally_instead_of_directions():
    frame=pd.read_csv(io.StringIO('group,seed,value\na,1,1\na,1,1\na,1,1\na,2,0\n'))
    result=analysis.bootstrap_summary(frame,['value'],['group']).iloc[0]
    assert result.value_mean==pytest.approx(.5)


def setup_case(tmp_path,monkeypatch,values,baseline_values,status='complete',baseline_status='complete'):
    (tmp_path/'checkpoints').mkdir();(tmp_path/'receipts').mkdir()
    job=dict(job_id='perturbed',baseline_id='baseline',phase='pilot',seed=101,epsilon=0.,momentum='preserve',
             layer='both',order='same',direction=1,dataset='mnist',checkpoint=20)
    schedule=[0,2350]
    def save(name,weights,run_status):
        count=len(weights)
        trajectory={key:np.asarray([np.array([[w]],dtype=np.float32) for w in weights]) for key in ['W0','W1']}
        trajectory.update(recovery_updates=np.array(schedule[:count]),accuracy=np.full(count,.95),loss=np.full(count,.1),
                          bias0=np.zeros((count,1)),bias1=np.zeros((count,1)))
        orders=[{'epoch':10,'sha256':'same'}] if count else []
        if count==2:orders.append({'epoch':11,'sha256':'same-next'})
        trajectory['metadata']=np.asarray(json.dumps({'epoch_order_hashes':orders}))
        np.savez(tmp_path/'checkpoints'/f'{name}.npz',**trajectory)
        (tmp_path/'receipts'/f'{name}.json').write_text(json.dumps({'status':run_status}))
    save('perturbed',values,status);save('baseline',baseline_values,baseline_status)
    threshold=dict(recovery_update=0,distance_q95=.5,distance_q90=.4,accuracy_q05=.8,loss_q95=.3,local_radius=.1)
    monkeypatch.setattr(analysis,'_ROOT',tmp_path)
    monkeypatch.setattr(analysis,'_THRESHOLDS',[threshold,{**threshold,'recovery_update':2350}])
    # A fixed reference set at W=1, with mocked simple analytic distance.
    monkeypatch.setattr(analysis,'reference_state',lambda k:[(np.ones((1,1)),np.ones((1,1)))]*10)
    monkeypatch.setattr(analysis,'distance',lambda a,b:abs(float(a[0][0,0]-b[0][0,0])))
    monkeypatch.setattr(analysis,'paired_metrics',lambda a,b:{key:abs(float(a[0][0,0]-b[0][0,0])) for key in ['distance','pif_distance','alpha','alpha_distance']})
    monkeypatch.setattr(analysis,'subset_thresholds',lambda k,which:threshold)
    monkeypatch.setattr(analysis,'loss_barrier_and_cka',lambda a,b,dataset:(0.,1.,[.1]*11))
    return job


@pytest.mark.parametrize('values,baselines,status,bstatus',[
    ([],[1.,1.],'failed','complete'),
    ([1.],[1.,1.],'failed','complete'),
    ([1.,1.],[],'complete','failed'),
    ([1.,1.],[1.],'complete','failed'),
])
def test_empty_partial_and_failed_baseline_are_included(tmp_path,monkeypatch,values,baselines,status,bstatus):
    job=setup_case(tmp_path,monkeypatch,values,baselines,status,bstatus)
    curves,endpoint,validation,barrier=analysis.analyze_one(job)
    assert len(curves)==2 and endpoint['update']==2350
    assert endpoint['failed'] and not endpoint['fixed_region_member']
    assert not endpoint['fixed_return'] and not endpoint['local_return']
    assert np.isnan(endpoint['ratio']) and np.isnan(endpoint['accuracy'])
    assert validation['same_batch_order']
    assert barrier['losses']==[]


def test_fixed_return_requires_departure_and_source_membership_for_induced_return(tmp_path,monkeypatch):
    job=setup_case(tmp_path,monkeypatch,[10.,1.],[1.,1.])
    job['epsilon']=9.
    curves,endpoint,_,_=analysis.analyze_one(job)
    assert not curves[0]['fixed_region_member'] and curves[-1]['fixed_region_member']
    assert endpoint['fixed_return'] and endpoint['fixed_structural_return']
    assert endpoint['initial_baseline_fixed_member'] and endpoint['perturbation_induced_return']
    assert not endpoint['initial_local_member'] and endpoint['local_return']


def test_identical_baseline_has_undefined_ratio_and_retention_not_return(tmp_path,monkeypatch):
    job=setup_case(tmp_path,monkeypatch,[1.,1.],[1.,1.])
    _,endpoint,_,_=analysis.analyze_one(job)
    assert np.isnan(endpoint['ratio'])
    assert endpoint['fixed_region_member'] and endpoint['retained_region']
    assert not endpoint['fixed_return'] and not endpoint['perturbation_induced_return']


def test_moving_envelope_is_distinct_from_fixed_region_and_ordinary_training_return(tmp_path,monkeypatch):
    job=setup_case(tmp_path,monkeypatch,[10.,1.],[10.,1.])
    monkeypatch.setattr(analysis,'reference_state',lambda k:[(np.full((1,1),10. if k==0 else 1.),
                                                           np.full((1,1),10. if k==0 else 1.))]*10)
    curves,endpoint,_,_=analysis.analyze_one(job)
    assert all(row['region_member'] for row in curves)
    assert not curves[0]['fixed_region_member'] and curves[-1]['fixed_region_member']
    assert endpoint['fixed_return'] and not endpoint['initially_outside_return']
    assert not endpoint['initial_baseline_fixed_member']
    assert not endpoint['perturbation_induced_return']
