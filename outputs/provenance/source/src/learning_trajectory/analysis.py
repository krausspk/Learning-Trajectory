"""Complete checkpoint-wise weight and representation analysis.

Every CSV contains actual re-computed pair values. Checkpoint jobs are independent,
atomic, and reusable only when their input and analysis fingerprints still match.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t
from threadpoolctl import threadpool_limits

from . import metrics as m
from .statistics import group_summary, correlation_summary

CKA_CHECKPOINTS = [0, 1, 2, 5, 10, 20]


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        if isinstance(x, Path): return str(x)
        raise TypeError(type(x).__name__)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, default=convert, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    frame.to_csv(temporary, index=False, float_format='%.15g')
    temporary.replace(path)


def load_runs(output, cohort):
    runs = []
    for path in sorted((Path(output) / 'checkpoints' / cohort).glob('*.npz')):
        with np.load(path, allow_pickle=False) as z:
            run = {k: z[k] for k in z.files if k != 'metadata'}
            run['meta'] = json.loads(str(z['metadata'].item()))
        run['path'] = path
        for key in ['W0', 'W1', 'accuracy', 'loss']:
            if not np.isfinite(run[key]).all():
                raise ValueError(f'Nonfinite {key}: {path}')
        runs.append(run)
    expected = {'main': 30, 'task': 20, 'early': 15}[cohort]
    if len(runs) != expected:
        raise ValueError(f'{cohort}: expected {expected} complete runs; found {len(runs)}')
    return runs


def _group(run, cohort):
    return run['meta']['task' if cohort == 'task' else 'dataset']


def _checkpoint_job(args):
    output, cohort, checkpoint, fingerprint = args
    with threadpool_limits(limits=1):
        return _analyze_checkpoint(output, cohort, checkpoint, fingerprint)


def _analyze_checkpoint(output, cohort, checkpoint, fingerprint):
    output = Path(output)
    destination = output / 'pairs' / cohort / f'checkpoint_{checkpoint:04d}.csv'
    receipt = destination.with_suffix('.json')
    if receipt.exists() and destination.exists():
        old = json.loads(receipt.read_text())
        extras_valid=all(old.get(k+'_path') and Path(old[k+'_path']).exists() and file_hash(old[k+'_path'])==old.get(k+'_sha256') for k in ['alignment','activation'])
        if extras_valid and old.get('fingerprint') == fingerprint and old.get('sha256') == file_hash(destination):
            return str(destination), 'reused'
    runs = load_runs(output, cohort)
    probe = np.load(output / 'probes' / ('task.npz' if cohort == 'task' else 'main.npz'))['X'].astype(np.float64)
    weights, centered, norms, gram_norms, activations = [], [], [], [], []
    for r in runs:
        ix = int(np.flatnonzero(r['checkpoints'] == checkpoint)[0])
        a, b = r['W0'][ix].astype(np.float64), r['W1'][ix].astype(np.float64)
        weights.append((a, b))
        h = np.maximum(probe @ a, 0)
        activations.append(h)
        hc = h - h.mean(axis=0)
        centered.append(hc)
        norms.append(np.linalg.norm(hc, axis=0))
        gram_norms.append(np.linalg.norm(hc.T @ hc, 'fro'))
    rows, assignments = [], []
    for ia, ib in itertools.combinations(range(len(runs)), 2):
        ra, rb = runs[ia], runs[ib]
        a0, a1 = weights[ia]; b0, b1 = weights[ib]
        base = {'cohort': cohort, 'checkpoint': int(checkpoint), 'index_a': ia, 'index_b': ib,
                'run_a': ra['meta']['run_id'], 'run_b': rb['meta']['run_id'],
                'group_a': _group(ra, cohort), 'group_b': _group(rb, cohort),
                'within': _group(ra, cohort) == _group(rb, cohort)}
        def add(metric, layer, alignment, value):
            if not np.isfinite(value): raise ValueError((base, metric, value))
            rows.append(dict(base, metric=metric, layer=layer, alignment=alignment, value=float(value)))
        def remember(method, p):
            assignments.append((ia, ib, method, np.asarray(p)))
        if cohort == 'task':
            for metric in ['alpha', 'euclidean', 'snd']:
                p = m.align_incoming(a0, b0, metric=metric)
                remember('incoming_'+metric,p)
                add('raw_alpha' if metric == 'alpha' else metric, 'W0', 'incoming', m.dissimilarity(a0, b0[:, p], metric=metric))
        else:
            for method, metric, canonical in [('canonical_alpha','alpha',True), ('raw_alpha','alpha',False), ('euclidean','euclidean',False), ('snd','snd',False)]:
                p = m.align(a0, a1, b0, b1, method=method)
                remember(method,p)
                values = m.layers(a0, a1, b0, b1, p, metric=metric, canonical=canonical)
                for layer, value in values.items(): add(method, layer, 'profile', value)
        cross = centered[ia].T @ centered[ib]
        if cohort != 'main' or checkpoint in CKA_CHECKPOINTS:
            denominator = gram_norms[ia] * gram_norms[ib]
            if denominator <= 0: raise ValueError('Degenerate hidden representation')
            add('one_minus_cka', 'hidden', 'none', 1 - np.clip(np.sum(cross ** 2) / denominator, 0, 1))
        if cohort == 'main':
            denom = norms[ia][:,None] * norms[ib][None,:]
            correlation = np.divide(cross, denom, out=np.zeros_like(cross), where=denom>0)
            for method in ['identity', 'benchmark', 'git_rebasin']:
                p = m.align(a0, a1, b0, b1, method=method)
                remember(method,p)
                b0p, b1p = m.apply_permutation(b0, b1, p)
                dot = float(np.sum(a0*b0p) + np.sum(a1*b1p))
                norm = np.sqrt((np.sum(a0*a0)+np.sum(a1*a1)) * (np.sum(b0*b0)+np.sum(b1*b1)))
                rmse = np.sqrt((np.sum((a0-b0p)**2)+np.sum((a1-b1p)**2))/(a0.size+a1.size))
                add('weight_inner_product','total',method,dot)
                add('weight_cosine','total',method,dot/norm)
                add('weight_rmse','total',method,rmse)
                add('activation_correlation','hidden',method,np.mean(correlation[np.arange(len(p)),p]))
                for layer,value in m.layers(a0,a1,b0,b1,p,metric='alpha',canonical=True).items():
                    add('canonical_alpha',layer,method,value)
            p1 = m.align(a0,a1,b0,b1,method='benchmark')
            p2 = m.align(a0,a1,b0,b1,method='git_rebasin')
            add('assignment_agreement','hidden','benchmark_vs_git',np.mean(p1==p2))
    frame = pd.DataFrame(rows)
    from .data import _atomic_npz
    alignment_path=output/'alignments'/cohort/f'checkpoint_{checkpoint:04d}.npz'
    activation_path=output/'activations'/cohort/f'checkpoint_{checkpoint:04d}.npz'
    _atomic_npz(alignment_path, pairs=np.array([[a,b] for a,b,_,_ in assignments]),
                methods=np.array([method for _,_,method,_ in assignments]),
                permutations=np.stack([p for _,_,_,p in assignments]),
                run_ids=np.array([r['meta']['run_id'] for r in runs]))
    _atomic_npz(activation_path, H=np.stack(activations),run_ids=np.array([r['meta']['run_id'] for r in runs]))
    write_csv(destination, frame)
    write_json(receipt, {'fingerprint':fingerprint, 'sha256':file_hash(destination), 'rows':len(frame),
                         'unordered_pairs':len(runs)*(len(runs)-1)//2, 'cohort':cohort,'checkpoint':checkpoint,
                         'alignment_path':str(alignment_path),'alignment_sha256':file_hash(alignment_path),
                         'activation_path':str(activation_path),'activation_sha256':file_hash(activation_path)})
    return str(destination), 'computed'


def analyze(output='outputs', cohorts=('main','task','early'), workers=4):
    output = Path(output)
    jobs = []
    code = [Path(__file__), Path(m.__file__), Path(__file__).with_name('statistics.py')]
    for cohort in cohorts:
        runs = load_runs(output,cohort)
        checkpoints = CKA_CHECKPOINTS if cohort == 'task' else runs[0]['checkpoints'].tolist()
        files = code + [r['path'] for r in runs] + [output/'probes'/('task.npz' if cohort=='task' else 'main.npz')]
        fingerprint = hashlib.sha256(''.join(file_hash(p) for p in files).encode()).hexdigest()
        jobs += [(str(output),cohort,int(cp),fingerprint) for cp in checkpoints]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(_checkpoint_job, job) for job in jobs]
        for future in as_completed(pending):
            path,status = future.result()
            print(f'Analysis {status}: {path}',flush=True)
    summarize(output,cohorts)


def summarize(output='outputs', cohorts=('main','task','early')):
    output = Path(output)
    # A selected-cohort rerun must not erase summaries of already completed
    # cohorts. Rebuild every available complete cohort, including the requested
    # one, from the preserved pair tables.
    requested=set(cohorts)
    available=[]
    expected={'main':(30,list(range(21))), 'task':(20,CKA_CHECKPOINTS),
              'early':(15,[0,1,2,4,8,16,32,48,64,96,128,160,192,235,352,470])}
    for cohort,(count,checkpoints) in expected.items():
        complete=(len(list((output/'checkpoints'/cohort).glob('*.npz')))==count and
                  all((output/'pairs'/cohort/f'checkpoint_{cp:04d}.csv').exists() for cp in checkpoints))
        if cohort in requested or complete:available.append(cohort)
    cohorts=tuple(available)
    estimates, replicates, correlations, contributions = [], [], [], []
    for cohort in cohorts:
        runs = load_runs(output,cohort)
        groups = np.array([_group(r,cohort) for r in runs])
        blocks = np.array([r['meta']['seed'] for r in runs]) if cohort=='task' else None
        for path in sorted((output/'pairs'/cohort).glob('checkpoint_*.csv')):
            frame = pd.read_csv(path)
            checkpoint = int(frame.checkpoint.iloc[0])
            for (metric,layer,alignment), block in frame.groupby(['metric','layer','alignment'],sort=False):
                pair = block[['index_a','index_b']].to_numpy(dtype=int)
                values = block.value.to_numpy()
                summary = group_summary(pair,values,groups,blocks=blocks)
                base = dict(cohort=cohort,checkpoint=checkpoint,metric=metric,layer=layer,alignment=alignment)
                for group,stat in summary.items():
                    estimates.append(dict(base,group=group,**{key:float(stat[key]) for key in ['estimate','se','ci_low','ci_high']}))
                    for j, v in enumerate(stat['replicates']):
                        replicates.append(dict(base,group=group,replicate=j,deleted=stat['deleted'][j],stratum=stat['strata'][j],value=float(v)))
                if checkpoint==20 and cohort=='main' and alignment=='profile':
                    for i,r in enumerate(runs):
                        mask=(pair==i).any(axis=1)
                        within=groups[pair[:,0]]==groups[pair[:,1]]
                        value=values[mask & ~within].mean()-values[mask & within].mean()
                        contributions.append(dict(base,run=r['meta']['run_id'],dataset=_group(r,cohort),value=value))
            if cohort=='main' and checkpoint in CKA_CHECKPOINTS:
                cka=frame[frame.metric.eq('one_minus_cka')][['index_a','index_b','value']].rename(columns={'value':'cka'})
                for layer in ['W0','W1','balanced']:
                    alpha=frame[frame.metric.eq('raw_alpha') & frame.layer.eq(layer) & frame.alignment.eq('profile')]
                    merged=alpha.merge(cka,on=['index_a','index_b'])
                    pair=merged[['index_a','index_b']].to_numpy(int)
                    for method in ['pearson','spearman']:
                        for group in ['all','within','between']:
                            mask=np.ones(len(pair),bool) if group=='all' else ((groups[pair[:,0]]==groups[pair[:,1]]) == (group=='within'))
                            stat=correlation_summary(pair[mask],merged.value.to_numpy()[mask],merged.cka.to_numpy()[mask],groups,method=method)
                            correlations.append(dict(cohort=cohort,checkpoint=checkpoint,layer=layer,method=method,group=group,**{key:float(stat[key]) for key in ['estimate','se','ci_low','ci_high']}))
                            for j,v in enumerate(stat['replicates']):
                                replicates.append(dict(cohort=cohort,checkpoint=checkpoint,metric=f'{method}_alpha_cka',layer=layer,alignment='profile',group=group,replicate=j,deleted=stat['deleted'][j],stratum=stat['strata'][j],value=float(v)))
    write_csv(output/'summary'/'estimates.csv',pd.DataFrame(estimates))
    write_csv(output/'summary'/'jackknife_replicates.csv',pd.DataFrame(replicates))
    write_csv(output/'summary'/'correlations.csv',pd.DataFrame(correlations))
    write_csv(output/'summary'/'run_contributions.csv',pd.DataFrame(contributions))
    training_summaries(output,cohorts)


def training_summaries(output='outputs',cohorts=('main','task','early')):
    output=Path(output)
    training,sign,displacements=[],[],[]
    for cohort in cohorts:
        for r in load_runs(output,cohort):
            for j,cp in enumerate(r['checkpoints']):
                base=dict(cohort=cohort,run=r['meta']['run_id'],dataset=r['meta']['dataset'],task=r['meta']['task'],seed=r['meta']['seed'],checkpoint=int(cp),update=int(r['updates'][j]))
                training.append(dict(base,accuracy=float(r['accuracy'][j]),loss=float(r['loss'][j])))
                for layer in ['W0','W1']:
                    w0=r[layer][0].astype(float); w=r[layer][j].astype(float)
                    sign.append(dict(base,layer=layer,retention=float(np.mean(np.sign(w0)==np.sign(w)))))
                    displacements.append(dict(base,layer=layer,relative_frobenius=float(np.linalg.norm(w-w0)/np.linalg.norm(w0)),rms=float(np.sqrt(np.mean((w-w0)**2)))))
    write_csv(output/'summary'/'training.csv',pd.DataFrame(training))
    write_csv(output/'summary'/'sign_retention.csv',pd.DataFrame(sign))
    write_csv(output/'summary'/'displacements.csv',pd.DataFrame(displacements))
    if 'early' in cohorts:
        frame=pd.DataFrame(training); frame=frame[frame.cohort.eq('early')]
        mean=frame.groupby(['dataset','update']).accuracy.mean().unstack(0)
        progress=(mean-mean.iloc[0])/(mean.iloc[-1]-mean.iloc[0])
        progress['mean']=progress.mean(axis=1)
        write_csv(output/'summary'/'early_progress.csv',progress.reset_index())
        functional={col:int(progress.index[progress[col]>=.9][0]) for col in progress}
        est=pd.read_csv(output/'summary'/'estimates.csv')
        onset={}
        for metric,layer in [('canonical_alpha','balanced'),('raw_alpha','balanced'),('euclidean','balanced'),('snd','balanced'),('one_minus_cka','hidden')]:
            x=est[est.cohort.eq('early') & est.metric.eq(metric) & est.layer.eq(layer) & est.group.eq('contrast') & est.checkpoint.gt(0) & est.ci_low.gt(0)]
            onset[metric]=int(x.checkpoint.min()) if len(x) else None
        write_json(output/'summary'/'early_timing.json',{'functional_90_percent':functional,'structural_onset':onset,'pointwise_threshold':True})


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='outputs'); p.add_argument('--workers',type=int,default=4)
    p.add_argument('--cohort',choices=['all','main','task','early'],default='all')
    args=p.parse_args()
    analyze(args.output,('main','task','early') if args.cohort=='all' else (args.cohort,),args.workers)
