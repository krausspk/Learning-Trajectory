#!/usr/bin/env python3
"""Independent consistency checks of serialized predictions, scores and controls."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def close(a,b):
    # Saved trajectory predictions are float32; CSV metrics use float64 predictions.
    if not np.isclose(a,b,rtol=2e-5,atol=2e-6):raise AssertionError((float(a),float(b)))

def main():
    reg=pd.read_csv(ROOT/'trajectory_regression_results.csv');events=pd.read_csv(ROOT/'event_decoding_results.csv')
    splits=json.loads((ROOT/'trajectory_probe_splits.json').read_text())
    freeze=json.loads((ROOT/'provenance/analysis_freeze.json').read_text())
    digest=hashlib.sha256((ROOT/'code/analyze.py').read_bytes()).hexdigest()
    assert digest==freeze['files']['code/analyze.py']
    assert reg.groupby('cohort').size().to_dict()=={'exploratory':702,'one_bit':492,'multi_bit':912}
    assert events.groupby('cohort').size().to_dict()=={'one_bit':8,'multi_bit':32}
    assert np.isfinite(reg[['mse','mae','r2','r2_ci_low','r2_ci_high']]).all().all()
    assert not reg.duplicated(['cohort','target','layer','pca_dim','model','time_index']).any()
    checked_reg=0;checked_events=0
    for cohort in ('exploratory','one_bit','multi_bit'):
        folder=ROOT/'analysis'/cohort
        summary=json.loads((folder/'summary.json').read_text());assert summary['source_sha256']==digest
        data=np.load(folder/'regression_predictions.npz')
        expected=splits[cohort]['splits']['test']['run_ids']
        assert list(data['run_ids'])==expected
        examples=np.load(folder/'heldout_examples.npz')
        assert set(examples['run_ids']).issubset(expected)
        assert np.array_equal(data['pred_z'][:,-1],data['true_z'][:,-1])
        rr=reg[reg.cohort==cohort]
        assert (rr.time_index<len(data['updates'])-1).all()
        assert np.all(rr['update']+rr.lag_updates==data['updates'][-1])
        for t in range(len(data['updates'])-1):
            row=rr[(rr.target=='absolute')&(rr.layer=='both')&(rr.pca_dim==8)&(rr.model=='ridge')&(rr.time_index==t)].iloc[0]
            y=data['true_z'][:,t].astype(float);p=data['pred_z'][:,t].astype(float)
            mse=((p-y)**2).mean();mae=np.abs(p-y).mean()
            r2=1-((p-y)**2).sum()/((y-y.mean(0))**2).sum()
            close(mse,row.mse);close(mae,row.mae);close(r2,row.r2);checked_reg+=1
        for dim in (8,16,32):
            mean=rr[(rr.target=='absolute')&(rr.layer=='both')&(rr.pca_dim==dim)&(rr.model=='train_mean')].sort_values('time_index')
            shifted=rr[(rr.target=='displacement')&(rr.layer=='both')&(rr.pca_dim==dim)&(rr.model=='mean_early_minus_final')].sort_values('time_index')
            # Float32 subtraction of large known final coordinates induces tiny rounding.
            assert np.allclose(mean.mse,shifted.mse,rtol=5e-4,atol=5e-5)
        if cohort=='exploratory':
            assert rr.p_value.isna().all();continue
        for feature in ('both','W0','W1','performance','norms','diagnostics'):
            saved=np.load(folder/f'event_{feature}.npz')
            labels=saved['labels'];scores=saved['scores'];null=saved['null_accuracy']
            assert null.shape==(999,labels.shape[1])
            assert (labels.sum(0)==40).all()
            for b in range(labels.shape[1]):
                row=events[(events.cohort==cohort)&(events.features==feature)&(events.model=='ridge_classifier')&(events.bit==b)].iloc[0]
                acc=np.mean((scores[:,b]>=.5)==labels[:,b])
                pos=scores[labels[:,b]==1,b];neg=scores[labels[:,b]==0,b]
                auc=np.mean((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg))
                p=(1+(null[:,b]>=acc).sum())/1000
                close(acc,row.accuracy);close(auc,row.auc);close(p,row.p_value);checked_events+=1
        er=events[(events.cohort==cohort)&(events.model=='ridge_classifier')].sort_values('p_value')
        running=0
        for rank,(_,row) in enumerate(er.iterrows()):
            running=max(running,min(1,row.p_value*(len(er)-rank)));close(running,row.p_holm)
    validation=json.loads((ROOT/'validation.json').read_text())
    assert validation['passed'] and validation['runs_audited']==800 and validation['original_inventory']['passed']
    result={'passed':True,'regression_rows':len(reg),'event_rows':len(events),'primary_timepoint_metrics_independently_recomputed':checked_reg,
            'linear_event_metrics_and_permutation_p_recomputed':checked_events,'test_only_examples':True,'endpoint_excluded':True,
            'baseline_translation_identity':True,'holm_recomputed':True,'analysis_freeze_hash_unchanged':True,
            'numeric_tolerance_note':'float32 saved trajectory predictions versus float64 CSV scoring: rtol2e-5/atol2e-6',
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'provenance/output_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
