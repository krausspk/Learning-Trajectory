"""Prespecified paired/seed-level analysis of complete and failed forks."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
from learning_trajectory.data import load_dataset, _atomic_json, sha256_file
from learning_trajectory.metrics import align, apply_permutation, linear_cka
from metrics import paired_metrics, canonical_pair, distance, threshold_calibration

_ROOT=None
_THRESHOLDS=None
_REFS=None


def load_arrays(path):
    with np.load(path,allow_pickle=False) as z:
        return {k:z[k].copy() for k in z.files if k!='metadata'}


def initialize(root):
    global _ROOT,_THRESHOLDS,_REFS
    _ROOT=Path(root)
    torch.set_num_threads(1)
    threadpool_limits(1)
    _THRESHOLDS=json.loads((_ROOT/'region_thresholds.json').read_text())['thresholds']
    refs=json.loads((_ROOT/'reference_jobs.json').read_text())
    _REFS=[load_arrays(_ROOT/'checkpoints'/f"{j['job_id']}.npz") for j in refs]


@lru_cache(maxsize=3)
def test_arrays(dataset):
    _,_,x,y=load_dataset(dataset,Path(__file__).resolve().parents[2]/'data')
    return x,y


def reference_state(k):
    return [(a['W0'][k],a['W1'][k]) for a in _REFS]


@lru_cache(maxsize=8)
def subset_thresholds(k,which):
    ids=range(5) if which=='a' else range(5,10)
    return threshold_calibration([reference_state(k)[i] for i in ids],
          [_REFS[i]['accuracy'][k] for i in ids],[_REFS[i]['loss'][k] for i in ids])


def membership(distances,accuracy,loss,threshold):
    median=float(np.median(distances))
    structural=bool(np.isfinite(median) and median<=threshold['distance_q95'])
    functional=bool(np.isfinite(accuracy) and np.isfinite(loss) and accuracy>=threshold['accuracy_q05'] and loss<=threshold['loss_q95'])
    return dict(region_distance=median,structural_member=structural,functional_member=functional,region_member=structural and functional)


@torch.inference_mode()
def loss_barrier_and_cka(a,b,dataset):
    ca,cb=canonical_pair(*a),canonical_pair(*b)
    p=align(*ca,*cb,method='git_rebasin')
    cb=apply_permutation(*cb,p)
    at=[torch.from_numpy(w.astype(np.float32)) for w in ca]
    bt=[torch.from_numpy(w.astype(np.float32)) for w in cb]
    x,y=test_arrays(dataset)
    values=[]
    for alpha in np.linspace(0,1,11):
        w0=(1-alpha)*at[0]+alpha*bt[0]
        w1=(1-alpha)*at[1]+alpha*bt[1]
        logits=(x@w0).relu()@w1
        values.append(float(torch.nn.functional.cross_entropy(logits,y)))
    excess=np.asarray(values)-np.linspace(values[0],values[-1],11)
    # Raw hidden representations, fixed 1,024 original test images; not gauge-invariant CKA.
    ha=(x[:1024]@torch.from_numpy(a[0])).relu().numpy()
    hb=(x[:1024]@torch.from_numpy(b[0])).relu().numpy()
    cka=linear_cka(ha,hb)
    return max(0.,float(excess.max())),cka,values


def _empty_pair():
    return {name:np.nan for name in ['distance','pif_distance','alpha','alpha_distance']}


def _empty_membership():
    return dict(region_distance=np.nan,structural_member=False,functional_member=False,region_member=False)


def analyze_one(job):
    path=_ROOT/'checkpoints'/f"{job['job_id']}.npz"
    baseline_path=_ROOT/'checkpoints'/f"{job['baseline_id']}.npz"
    receipt=json.loads((_ROOT/'receipts'/f"{job['job_id']}.json").read_text())
    baseline_receipt=json.loads((_ROOT/'receipts'/f"{job['baseline_id']}.json").read_text())
    state,baseline=load_arrays(path),load_arrays(baseline_path)
    common={k:job[k] for k in ['job_id','phase','seed','epsilon','momentum','layer','order','direction','dataset','checkpoint']}
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata'].item()))
    with np.load(baseline_path,allow_pickle=False) as z:
        bm=json.loads(str(z['metadata'].item()))
    planned=[int(t['recovery_update']) for t in _THRESHOLDS]
    target=planned[-1]
    state_index={int(update):i for i,update in enumerate(state['recovery_updates'])}
    baseline_index={int(update):i for i,update in enumerate(baseline['recovery_updates'])}
    perturbation_failed=receipt['status']!='complete' or target not in state_index
    baseline_failed=baseline_receipt['status']!='complete' or target not in baseline_index
    failed=perturbation_failed or baseline_failed
    validation={'job_id':job['job_id'],'status':receipt['status'],'baseline_status':baseline_receipt['status']}
    sh,bh=metadata['epoch_order_hashes'],bm['epoch_order_hashes']
    shared=min(len(sh),len(bh))
    validation['same_batch_order']=(sh[:shared]==bh[:shared])
    validation['batch_order_epochs_compared']=shared
    validation['complete_batch_order_lists_equal']=sh==bh
    if job['order']=='same' and not validation['same_batch_order']:
        raise AssertionError('paired batch order prefixes differ')
    if job['order']=='same' and not failed and sh!=bh:
        raise AssertionError('complete paired batch order schedules differ')
    errors=[]
    if 0 in state_index and 0 in baseline_index:
        for layer in ['W0','W1']:
            actual=np.linalg.norm(state[layer][state_index[0]].astype(np.float64)-baseline[layer][baseline_index[0]].astype(np.float64))/np.linalg.norm(baseline[layer][baseline_index[0]].astype(np.float64))
            expected=job['epsilon'] if job['layer'] in ('both',layer) else 0.
            errors.append(abs(float(actual)-expected))
    validation['perturbation_norm_error']=max(errors) if errors else np.nan
    validation['perturbation_norm_checked']=bool(errors)
    if errors and max(errors)>1.e-6:raise AssertionError(validation)
    validation['zero_biases']=bool(not np.any(state['bias0']) and not np.any(state['bias1']))
    if not validation['zero_biases']:raise AssertionError(validation)
    final_refs=reference_state(len(_THRESHOLDS)-1)
    final_threshold=_THRESHOLDS[-1]
    baseline_initial_fixed=False
    if 0 in baseline_index:
        bi=baseline_index[0]
        a=(baseline['W0'][bi],baseline['W1'][bi])
        baseline_initial_fixed=membership([distance(a,r) for r in final_refs],float(baseline['accuracy'][bi]),float(baseline['loss'][bi]),final_threshold)['region_member']
    records=[]
    d0=np.nan
    for k,update in enumerate(planned):
        si,bi=state_index.get(update),baseline_index.get(update)
        available=si is not None
        paired_available=available and bi is not None
        pair=_empty_pair()
        member=_empty_membership()
        fixed=_empty_membership()
        accuracy=loss=baseline_accuracy=baseline_loss=np.nan
        distances=[]
        if bi is not None:
            baseline_accuracy=float(baseline['accuracy'][bi]);baseline_loss=float(baseline['loss'][bi])
        if available:
            b=(state['W0'][si],state['W1'][si])
            accuracy=float(state['accuracy'][si]);loss=float(state['loss'][si])
            distances=[distance(b,r) for r in reference_state(k)]
            member=membership(distances,accuracy,loss,_THRESHOLDS[k])
            fixed=membership([distance(b,r) for r in final_refs],accuracy,loss,final_threshold)
        if paired_available:
            a=(baseline['W0'][bi],baseline['W1'][bi])
            pair=paired_metrics(a,b)
        if update==0:d0=pair['distance']
        row={**common,'update':int(update),**pair,'ratio':pair['distance']/d0 if np.isfinite(d0) and d0>1e-12 else np.nan,
             'accuracy':accuracy,'loss':loss,'baseline_accuracy':baseline_accuracy,'baseline_loss':baseline_loss,
             **member,'local_member':bool(paired_available and pair['distance']<=_THRESHOLDS[k]['local_radius'] and member['functional_member']),
             'q90_member':bool(available and np.median(distances)<=_THRESHOLDS[k]['distance_q90'] and member['functional_member']),
             'fixed_region_distance':fixed['region_distance'],'fixed_region_member':fixed['region_member'],
             'fixed_structural_member':fixed['structural_member'],'fixed_functional_member':fixed['functional_member'],
             'state_available':available,'baseline_available':bi is not None}
        records.append(row)
    first=records[0]
    final=dict(records[-1])
    final.update(failed=failed,perturbation_failed=perturbation_failed,baseline_failed=baseline_failed,
                 last_observed_update=max(state_index,default=-1),initial_state_available=first['state_available'])
    final['initial_region_member']=first['region_member']
    final['initially_outside_return']=bool(first['state_available'] and not first['region_member'] and final['region_member'] and not failed)
    final['retained_region']=bool(all(r['region_member'] for r in records) and not failed)
    final['sampled_return']=bool(any(r['state_available'] and not r['region_member'] for r in records[:-1]) and final['region_member'] and not failed)
    final['final_contracts']=bool(np.isfinite(final['ratio']) and final['ratio']<1 and not failed)
    final['structural_return']=bool(first['state_available'] and not first['structural_member'] and final['structural_member'] and not failed)
    final['initial_structural_member']=first['structural_member']
    final['initial_functional_member']=first['functional_member']
    final['initial_local_member']=first['local_member']
    final['local_return']=bool(first['state_available'] and first['baseline_available'] and not first['local_member'] and final['local_member'] and not failed)
    final['initial_fixed_region_member']=first['fixed_region_member']
    final['initial_fixed_structural_member']=first['fixed_structural_member']
    final['initial_fixed_functional_member']=first['fixed_functional_member']
    final['fixed_return']=bool(first['state_available'] and not first['fixed_region_member'] and final['fixed_region_member'] and not failed)
    final['fixed_structural_return']=bool(first['state_available'] and not first['fixed_structural_member'] and final['fixed_structural_member'] and not failed)
    final['initial_baseline_fixed_member']=baseline_initial_fixed
    final['perturbation_induced_return']=bool(baseline_initial_fixed and final['fixed_return'])
    final['initial_distance']=d0
    final['accuracy_change']=final['accuracy']-first['accuracy']
    final['loss_change']=final['loss']-first['loss']
    final['accuracy_vs_baseline']=final['accuracy']-final['baseline_accuracy']
    final['loss_vs_baseline']=final['loss']-final['baseline_loss']
    if failed:
        for key in ['region_member','structural_member','functional_member','local_member','q90_member','subset_a_member','subset_b_member','fixed_region_member','fixed_structural_member','fixed_functional_member']:
            final[key]=False
        for key in ['ratio','distance','accuracy','loss','barrier','cka','pif_distance','alpha','alpha_distance','accuracy_change','loss_change','accuracy_vs_baseline','loss_vs_baseline']:
            final[key]=np.nan
        barrier_values=[]
    else:
        si,bi=state_index[target],baseline_index[target]
        b=(state['W0'][si],state['W1'][si])
        for which,indices in [('a',range(5)),('b',range(5,10))]:
            d=[distance(b,final_refs[i]) for i in indices]
            m=membership(d,final['accuracy'],final['loss'],subset_thresholds(len(_THRESHOLDS)-1,which))
            final['subset_'+which+'_member']=m['region_member']
        a=(baseline['W0'][bi],baseline['W1'][bi])
        final['barrier'],final['cka'],barrier_values=loss_barrier_and_cka(a,b,job['dataset'])
    return records,final,validation,dict(job_id=job['job_id'],losses=barrier_values)


def bootstrap_summary(df,columns,group_columns,seed=620260116):
    rng=np.random.default_rng(seed)
    output=[]
    for group,g in df.groupby(group_columns,dropna=False,sort=True):
        if not isinstance(group,tuple):group=(group,)
        row=dict(zip(group_columns,group))
        means=g.groupby('seed')[columns].agg(lambda values: values.mean(skipna=False))
        row.update(n_seeds=len(means),n_runs=len(g))
        ids=rng.integers(0,len(means),size=(10000,len(means)))
        for col in columns:
            vals=means[col].to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                # Undefined ratios for epsilon=0, or nonfinite failed endpoints are never silently imputed.
                row.update({col+'_mean':np.nan,col+'_low':np.nan,col+'_high':np.nan})
                continue
            bs=vals[ids].mean(axis=1)
            row.update({col+'_mean':float(vals.mean()),col+'_low':float(np.quantile(bs,.025)),col+'_high':float(np.quantile(bs,.975))})
        output.append(row)
    return pd.DataFrame(output)


def analyze(root,workers=8,report=True):
    root=Path(root)
    jobs=json.loads((root/'pilot_jobs.json').read_text())
    records=[];ends=[];valid=[];bars=[]
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn'),initializer=initialize,initargs=(str(root),)) as pool:
        futures=[pool.submit(analyze_one,j) for j in jobs]
        for n,future in enumerate(as_completed(futures),1):
            r,e,v,b=future.result();records.extend(r);ends.append(e);valid.append(v);bars.append(b)
            if n%20==0:print(json.dumps({'analyzed':n,'total':len(jobs),'seconds':round(time.monotonic()-started,1)}),flush=True)
    curves=pd.DataFrame(records).sort_values(['job_id','update'])
    endpoints=pd.DataFrame(ends).sort_values('job_id')
    curves.to_csv(root/'recovery_metrics.csv',index=False)
    endpoints.to_csv(root/'endpoint_metrics.csv',index=False)
    columns=['ratio','distance','accuracy','loss','region_member','structural_member','initially_outside_return','retained_region','sampled_return','functional_member','local_member','final_contracts','barrier','cka','q90_member','subset_a_member','subset_b_member','pif_distance','alpha','alpha_distance','accuracy_vs_baseline','loss_vs_baseline','failed','perturbation_failed','baseline_failed','fixed_region_member','fixed_structural_member','fixed_functional_member','fixed_return','fixed_structural_return','perturbation_induced_return','initial_baseline_fixed_member','initial_fixed_region_member','initial_fixed_structural_member','initial_fixed_functional_member','initial_local_member','local_return','region_distance','fixed_region_distance','accuracy_change','loss_change']
    summary=bootstrap_summary(endpoints,columns,['phase','dataset','checkpoint','epsilon','momentum','order','layer'])
    summary.to_csv(root/'endpoint_summary.csv',index=False)
    curve_summary=bootstrap_summary(curves,['distance','ratio','accuracy','loss','region_member','structural_member','functional_member','fixed_region_member','fixed_structural_member','fixed_functional_member','fixed_region_distance','region_distance','local_member'],['phase','dataset','checkpoint','epsilon','momentum','order','layer','update'])
    curve_summary.to_csv(root/'recovery_summary.csv',index=False)
    # Paired seed-block comparison of preserve minus reset; directions paired by seed/epsilon.
    primary=endpoints[(endpoints.phase.isin(['pilot','expanded']))&(endpoints.epsilon>0)]
    paired=[]
    for group,g in primary.groupby(['dataset','checkpoint','layer','epsilon']):
        dataset,checkpoint,layer,eps=group
        for metric in ['ratio','distance','accuracy','loss','region_member','local_member']:
            table=g.groupby(['seed','momentum'])[metric].agg(lambda values: values.mean(skipna=False)).unstack()
            diff=table['preserve']-table['reset']
            rng=np.random.default_rng(620260117)
            bs=diff.to_numpy()[rng.integers(0,len(diff),(10000,len(diff)))].mean(1)
            paired.append(dict(dataset=dataset,checkpoint=int(checkpoint),layer=layer,epsilon=eps,metric=metric,mean=float(diff.mean(skipna=False)),low=float(np.quantile(bs,.025)),high=float(np.quantile(bs,.975)),n_seeds=len(diff)))
    pd.DataFrame(paired).to_csv(root/'momentum_paired_comparison.csv',index=False)
    pd.DataFrame(valid).to_csv(root/'validation/fork_checks.csv',index=False)
    _atomic_json(root/'loss_interpolations.json',bars)
    status=dict(complete=len(ends)==len(jobs),expected_runs=len(jobs),analyzed_runs=len(ends),n_final_failures=int(endpoints.failed.sum()),
                primary_perturbations=int(((endpoints.phase.isin(['pilot','expanded']))&(endpoints.epsilon>0)).sum()),
                nan_endpoints=int(endpoints[['distance','accuracy','loss']].isna().any(axis=1).sum()),
                undefined_epsilon_zero_ratios=int(((endpoints.epsilon==0)&endpoints['ratio'].isna()).sum()),
                undefined_nonzero_ratios=int(((endpoints.epsilon>0)&endpoints['ratio'].isna()).sum()),
                analysis_source_sha256=sha256_file(__file__),
                metrics_source_sha256=sha256_file(Path(__file__).with_name('metrics.py')),
                max_perturbation_norm_error=max((v['perturbation_norm_error'] for v in valid if np.isfinite(v['perturbation_norm_error'])),default=None),
                seed_count=int(endpoints.seed.nunique()),elapsed_seconds=time.monotonic()-started,
                config_sha256=sha256_file(root/'config.json'),region_thresholds_sha256=sha256_file(root/'region_thresholds.json'))
    _atomic_json(root/'analysis_status.json',status)
    if report:
        from reporting import make_report
        make_report(root)
    print(json.dumps(status),flush=True)
