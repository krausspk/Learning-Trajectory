"""Post-hoc transductive norm-rank controls; prior 42 cost matrices stay intact."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("mnist", "fashion_mnist", "kmnist")
LAYERS = ("W0", "W1", "total")
GAUGES = ("raw", "canonical")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def norm_rank_costs(query_norms, candidate_norms):
    """Label-blind ranks calibrated separately within the two complete cohorts."""
    q, c = np.asarray(query_norms, dtype=np.float64), np.asarray(candidate_norms, dtype=np.float64)
    if q.ndim!=1 or c.ndim!=1 or len(q)!=len(c) or len(q)<2:
        raise ValueError("Equal one-dimensional query/candidate cohorts with n>=2 required")
    if not np.isfinite(q).all() or not np.isfinite(c).all() or np.any(q<0) or np.any(c<0):
        raise ValueError("Norms must be finite and nonnegative")
    qr = rankdata(q, method="average")
    cr = rankdata(c, method="average")
    return np.abs(qr[:,None]-cr[None,:])/(len(q)-1)


def calibrated_leave_one_costs(query_norms, candidate_norms):
    """Recalibrate ranks after deleting the same independent run in both cohorts."""
    n = len(query_norms)
    if n<3:
        raise ValueError("At least three original runs required for leave-one calibration")
    return np.asarray([norm_rank_costs(np.delete(query_norms,i),np.delete(candidate_norms,i)) for i in range(n)])


def _write_json(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def build_norm_rank_controls(root:Path=ROOT):
    root=Path(root).resolve()
    controls=root/'03_Initialization_Controls'
    out=controls/'norm_rank_controls'
    data=out/'data'
    validation_path=controls/'data/validation.json'
    validation=json.loads(validation_path.read_text())
    if not validation.get('passed') or validation['code_sha256']!=sha256(controls/'code/distances.py'):
        raise ValueError('Original control distance validation is missing or stale')
    keys=[f'norm_rank_difference|{layer}|none|{gauge}' for gauge in GAUGES for layer in LAYERS]
    note={
        'analysis':'post_hoc_norm_rank_control','created_utc':datetime.now(timezone.utc).isoformat(),
        'status':'additional exploratory artifact control, not part of the prespecified main/control family',
        'reason':'The uncalibrated absolute norm gap produced identical candidate orderings across final queries because training shifted query norms beyond the initial candidate range. Its mean correct-parent rank can then equal 5.5 algebraically for every bijection; that result does not establish absence of norm information.',
        'definition':'C[t,i,j]=abs(average_rank(norm_current[t,:])[i] - average_rank(norm_initial[0,:])[j])/(n-1)',
        'calibration':'Separate transductive rank calibration within all current queries and within all initial candidates, independently of their true pairing',
        'interpretation':'Tests relative norm ordering after allowing a separate monotone scale change between initial and trained cohorts; uses the full query ensemble and is not standalone single-network retrieval.',
        'ties':'Exact numeric ties receive average ranks; same rule for every query/candidate.',
        'jackknife':'Delete paired current-query and initial-candidate identities, then recompute both norm ranks and divide by n_remaining-1. Precomputed recalibrated leave-one matrices are saved; simply subsetting the full calibrated matrix would be insufficient.',
        'null':'Globally bijective candidate-identity permutations of the full label-blind cost matrix; calibration contains no parent labels.',
        'inference_base_seed':2026111900,'inference_role':'post_hoc_exploratory','confirmatory_primary_key':None,
        'selection_policy':'One fixed rank-based calibration for all six gauge/layer variants and all checkpoints, without selection for positive results.',
        'prespecified_42_matrices_changed':False,'new_training':False,
        'source_validation_path':str(validation_path),'source_validation_sha256':sha256(validation_path),
        'source_code_path':str(Path(__file__).resolve()),'source_code_sha256':sha256(__file__),
    }
    _write_json(out/'analysis_note.json',note)
    definitions={key:{'metric':'norm_rank_difference','layer':key.split('|')[1],'alignment':'none','gauge':key.split('|')[3],
                      'formula':note['definition'],'calibration':note['calibration'],'inferential_role':'post_hoc_exploratory',
                      'matching':'none; norm-only costs contain no weight or activation matching',
                      'units':'absolute within-cohort average-rank difference divided by n-1; dimensionless',
                      'candidate_checkpoint':0,'identity_input_to_cost_function':False} for key in keys}
    _write_json(data/'distance_definitions.json',definitions)
    records=[]
    for dataset in DATASETS:
        record=next(r for r in validation['outputs'] if r['dataset']==dataset)
        source=controls/f'data/{dataset}_retrieval.npz'
        source_hash=sha256(source)
        if source_hash!=record['sha256']:
            raise ValueError(f'Source control NPZ hash mismatch: {source}')
        with np.load(source,allow_pickle=False) as archive:
            cp=archive['checkpoints'].copy();ids=archive['run_ids'].copy();seeds=archive['seeds'].copy()
            qids=archive['query_run_ids'].copy();cids=archive['candidate_run_ids'].copy()
            layers=archive['norm_layers'].astype(str).tolist()
            norms={gauge:archive[f'norms_{gauge}'].astype(np.float64) for gauge in GAUGES}
        if not np.array_equal(cp,np.arange(21)) or len(ids)!=10 or not np.array_equal(ids,qids) or not np.array_equal(ids,cids):
            raise ValueError(f'Unexpected source run/checkpoint mapping: {source}')
        if any(a.shape!=(21,10,3) for a in norms.values()) or len(set(ids.tolist()))!=10:
            raise ValueError(f'Unexpected norm dimensions or duplicate identities: {source}')
        costs=np.empty((6,21,10,10))
        jack=np.empty((6,21,10,9,9))
        for k,key in enumerate(keys):
            _,layer,_,gauge=key.split('|');column=layers.index(layer)
            initial=norms[gauge][0,:,column]
            for t in range(21):
                query=norms[gauge][t,:,column]
                costs[k,t]=norm_rank_costs(query,initial)
                jack[k,t]=calibrated_leave_one_costs(query,initial)
        if not np.isfinite(costs).all() or costs.min()<0 or costs.max()>1 or not np.all(np.diagonal(costs[:,0],axis1=-2,axis2=-1)==0):
            raise AssertionError(f'Invalid norm-rank costs: {dataset}')
        path=data/f'{dataset}_retrieval.npz'
        np.savez_compressed(path,distances=costs,metric_keys=np.asarray(keys),checkpoints=cp,run_ids=ids,seeds=seeds,
                            query_run_ids=qids,candidate_run_ids=cids,candidate_checkpoint=np.asarray(0),
                            jackknife_distances=jack,jackknife_deleted_indices=np.arange(10),
                            jackknife_axes=np.asarray(['metric','checkpoint','deleted_run','remaining_query','remaining_candidate']),
                            jackknife_remaining_indices=np.asarray([np.delete(np.arange(10),i) for i in range(10)]),
                            norms_raw=norms['raw'],norms_canonical=norms['canonical'],norm_layers=np.asarray(layers),
                            source_npz_sha256=np.asarray(source_hash))
        records.append({'dataset':dataset,'path':str(path),'sha256':sha256(path),'shape':list(costs.shape),
                        'jackknife_shape':list(jack.shape),'source_path':str(source),'source_sha256':source_hash,
                        'query_run_ids':ids.tolist(),'candidate_run_ids':ids.tolist(),'E0_max_diagonal_distance':0.0})
    unchanged=all(sha256(Path(r['source_path']))==r['source_sha256'] for r in records)
    result={'passed':bool(unchanged),'code_sha256':sha256(__file__),'outputs':records,'n_metric_keys':6,'n_distances':6*3*21*100,
            'original_42_costs_unchanged':unchanged,'identity_labels_used_for_costs':False,'all_entries_computed_identically':True,
            'diagonal_overridden':False,'jackknife_recalibrates_cohorts':True,'analysis_note_sha256':sha256(out/'analysis_note.json'),
            'inferential_role':'post_hoc_exploratory','source_control_validation_sha256':sha256(validation_path)}
    _write_json(data/'validation.json',result)
    if not unchanged:raise AssertionError('Source controls unexpectedly changed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    result=build_norm_rank_controls(args.root)
    print(json.dumps({'passed':result['passed'],'n_distances':result['n_distances']}))
