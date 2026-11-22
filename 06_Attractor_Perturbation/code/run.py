#!/usr/bin/env python3
"""Run the prospectively specified perturbation experiment in its own directory."""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'src'))
import numpy as np
import pandas as pd
from learning_trajectory.data import sha256_file, _atomic_json


def stamp():
    return datetime.now(ZoneInfo('Europe/Berlin')).isoformat()


def freeze(path, value):
    if path.exists():
        old = json.loads(path.read_text())
        if old != value:
            raise ValueError(f'Refusing to change a frozen specification: {path}')
    else:
        _atomic_json(path, value)


def job_for(run_id, dataset, seed, checkpoint=20, momentum='preserve', epsilon=0.,
            layer='both', direction=0, phase='pilot', order='same', cohort='main'):
    suffix = f'{run_id}_E{checkpoint}_{momentum}_{layer}_e{epsilon:.2f}_d{direction}_{order}'
    perturb_seed = 61000000 + seed*10000 + checkpoint*100 + direction
    # Same direction is used across epsilon/layer/momentum cells, enabling paired comparisons.
    order_seed = seed if order == 'same' else 71000000 + seed*100 + direction
    baseline = f'{run_id}_E{checkpoint}_{momentum}_both_e0.00_d0_same'
    return dict(job_id=suffix, run_id=run_id, source=str(PROJECT/'outputs/checkpoints'/cohort/f'{run_id}.pt'),
                dataset=dataset, seed=seed, checkpoint=checkpoint, momentum=momentum,
                epsilon=epsilon, layer=layer, direction=direction, perturb_seed=perturb_seed,
                order_seed=order_seed, order=order, phase=phase,
                baseline_id=baseline, recovery_epochs=10)


def prepare_pilot():
    path = ROOT/'config.json'
    if path.exists():
        return json.loads(path.read_text())
    config = dict(created_at=stamp(), stage='pilot', datasets=['mnist'], source_runs=[f'M{i}' for i in range(1,11)],
        checkpoint=20, epsilon=[0,.01,.05,.20], layers=['both'], directions=5,
        momentum=['preserve','reset'], recovery_epochs=10, batch_size=256, lr=.1, momentum_coefficient=.9,
        weight_decay=0, architecture=[784,64,10], frozen_zero_biases=True, train_samples=60000, test_samples=10000,
        primary_metric='Frobenius norm of concatenated canonical weights after exact one-hidden-layer Git Re-Basin assignment',
        region='median distance to 10 independent matched-horizon references <= q95 of leave-one-out reference medians; accuracy >= reference q05 and loss <= reference q95',
        return_definition='initially outside region, final inside; separately report membership, retention and sampled exit/re-entry',
        h1='final distance / immediate post-perturbation distance; epsilon=0 ratio undefined; seed-aggregate 95% bootstrap CI',
        h3='individual distance <= 0.1 * median leave-one-out independent reference distance, and functional criterion; operational family only',
        inference='10000 bootstrap resamples of 10 source seeds, averaging directions within seed first; paired protocol differences',
        bootstrap_seed=620260116,
        robustness=['canonical PIF/Hungarian Frobenius','canonical Alpha matching and dissimilarity','q90 structural threshold','two disjoint 5-reference subsets'],
        control_c='preserve momentum, alternate perturbation-arm batch order; epsilon 0 and .20, directions1..5 at .20',
        loss_barrier='11 equally spaced points, canonical and permutation-aligned endpoints, full test set; max loss minus linearly interpolated endpoint losses',
        extension_gate='technical validation and completed nonselective pilot; scientific null result does not count as failure',
        scope='Pilot first; prospective expanded matrix recorded after pilot; all full datasets, no shortened recovery')
    freeze(path, config)
    refs=[job_for(f'I{i}','mnist',400+i,phase='reference',cohort='task') for i in range(1,11)]
    jobs=[]
    for i in range(1,11):
        for protocol in config['momentum']:
            jobs.append(job_for(f'M{i}','mnist',100+i,momentum=protocol))
            for eps in config['epsilon'][1:]:
                for direction in range(1,6):
                    jobs.append(job_for(f'M{i}','mnist',100+i,momentum=protocol,epsilon=eps,direction=direction))
        jobs.append(job_for(f'M{i}','mnist',100+i,phase='noise_control',order='different'))
        for direction in range(1,6):
            jobs.append(job_for(f'M{i}','mnist',100+i,epsilon=.20,direction=direction,phase='noise_control',order='different'))
    freeze(ROOT/'reference_jobs.json', refs)
    freeze(ROOT/'pilot_jobs.json', jobs)
    allpaths = sorted({j['source'] for j in refs+jobs})
    originals = {str(Path(p).relative_to(PROJECT)):sha256_file(p) for p in allpaths}
    originals.update({str(Path(p).with_suffix('.npz').relative_to(PROJECT)):sha256_file(Path(p).with_suffix('.npz')) for p in allpaths})
    freeze(ROOT/'provenance/source_checksums_before.json', originals)
    pd.DataFrame(jobs).to_csv(ROOT/'perturbation_manifest.csv',index=False)
    return config


def worker(job):
    from training import train_job
    return train_job(job, ROOT)


def train_jobs(jobs, workers):
    started=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context('spawn')) as pool:
        futures={pool.submit(worker,j):j for j in jobs}
        for n,future in enumerate(as_completed(futures),1):
            result=future.result()
            print(json.dumps(dict(completed=n,total=len(jobs),elapsed_seconds=round(time.monotonic()-started,1),result=result)),flush=True)


def calibrate():
    from metrics import threshold_calibration
    refs=json.loads((ROOT/'reference_jobs.json').read_text())
    arrays=[]
    for job in refs:
        receipt=json.loads((ROOT/'receipts'/f"{job['job_id']}.json").read_text())
        assert receipt['status']=='complete' and receipt['final_recovery_update']==2350
        assert sha256_file(ROOT/'checkpoints'/f"{job['job_id']}.npz")==receipt['npz_sha256']
        with np.load(ROOT/'checkpoints'/f"{job['job_id']}.npz", allow_pickle=False) as a:
            arrays.append({k:a[k].copy() for k in a.files if k!='metadata'})
    updates=arrays[0]['recovery_updates']
    assert all(np.array_equal(a['recovery_updates'],updates) for a in arrays)
    thresholds=[]
    for k,update in enumerate(updates):
        networks=[(a['W0'][k],a['W1'][k]) for a in arrays]
        t=threshold_calibration(networks,[float(a['accuracy'][k]) for a in arrays],[float(a['loss'][k]) for a in arrays])
        t.update(recovery_update=int(update), local_radius=.1*float(np.median([r['loo_median_distance'] for r in t['per_reference']])))
        thresholds.append(t)
    value=dict(created_at=stamp(),frozen_before_perturbation_training=True,reference_ids=[r['job_id'] for r in refs],
               reference_checksums={r['job_id']:sha256_file(ROOT/'checkpoints'/f"{r['job_id']}.npz") for r in refs}, thresholds=thresholds)
    path=ROOT/'region_thresholds.json'
    if path.exists():
        old=json.loads(path.read_text())
        assert old['reference_checksums']==value['reference_checksums'] and old['thresholds']==thresholds
    else:
        freeze(path,value)
    print(json.dumps({'thresholds_frozen':str(path),'final':thresholds[-1]}),flush=True)


def check_originals():
    expected=json.loads((ROOT/'provenance/source_checksums_before.json').read_text())
    mismatches=[p for p,h in expected.items() if sha256_file(PROJECT/p)!=h]
    value=dict(checked_at=stamp(),checked_files=len(expected),mismatches=mismatches,passed=not mismatches)
    _atomic_json(ROOT/'validation/originals_unchanged.json',value)
    if mismatches:raise AssertionError(mismatches)
    print(json.dumps(value))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=['prepare','references','calibrate','pilot','analyze','verify'])
    parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    prepare_pilot()
    if args.phase=='references':train_jobs(json.loads((ROOT/'reference_jobs.json').read_text()),args.workers)
    elif args.phase=='calibrate':calibrate()
    elif args.phase=='pilot':
        assert (ROOT/'region_thresholds.json').exists(), 'Freeze independent criteria before perturbation training'
        train_jobs(json.loads((ROOT/'pilot_jobs.json').read_text()),args.workers)
    elif args.phase=='analyze':
        from analysis import analyze
        analyze(ROOT,workers=args.workers)
    elif args.phase=='verify':check_originals()

if __name__=='__main__':main()
