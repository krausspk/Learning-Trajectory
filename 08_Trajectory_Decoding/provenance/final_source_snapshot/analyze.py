#!/usr/bin/env python3
"""Run-grouped, final-only trajectory and event probes; all fits use train only."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import warnings
import numpy as np
import pandas as pd
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT/'src'))
from learning_trajectory.metrics import canonicalize, align, apply_permutation
from learning_trajectory.data import sha256_file

ALPHAS = np.array([.0001,.001,.01,.1,1.,10.])
SPLIT_SEED=20261116
BOOTSTRAPS=1000
PERMUTATIONS=999

def write_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    temporary.replace(path)


def grouped_split(seeds, seed=SPLIT_SEED, strata=None):
    seeds=np.asarray(seeds)
    unique=np.unique(seeds)
    rng=np.random.default_rng(seed)
    output={k:[] for k in ['train','validation','test']}
    if strata is None:
        groups=[unique]
    else:
        strata=np.asarray(strata)
        groups=[np.unique(seeds[strata==s]) for s in np.unique(strata)]
    for group in groups:
        group=rng.permutation(group); n=len(group)
        ntrain=max(1,int(n*.6)); nval=max(1,int(n*.2))
        for key, part in zip(output,[group[:ntrain],group[ntrain:ntrain+nval],group[ntrain+nval:]]):
            output[key].extend(int(x) for x in part)
    return {key:np.flatnonzero(np.isin(seeds,part)) for key,part in output.items()}


def canonical_trajectory(w0,w1,reference):
    """Reference and permutation inspect FINAL weights only. Gauge is local."""
    f0,f1=canonicalize(w0[-1],w1[-1])
    permutation=align(*reference,f0,f1,method='git_rebasin')
    out0=np.empty_like(w0,dtype=np.float32); out1=np.empty_like(w1,dtype=np.float32)
    for t in range(len(w0)):
        a,b=canonicalize(w0[t],w1[t]); a,b=apply_permutation(a,b,permutation)
        out0[t],out1[t]=a,b
    return out0,out1,permutation


def regression_metrics(y,p):
    y=np.asarray(y,dtype=float); p=np.asarray(p,dtype=float)
    error=(y-p)**2
    denominator=np.sum((y-y.mean(axis=0))**2)
    return {'mse':float(error.mean()),'mae':float(np.abs(y-p).mean()),
            'r2':float(1-error.sum()/denominator) if denominator>1e-20 else 0.0}


def bootstrap_indices(seeds, n=BOOTSTRAPS, seed=5632):
    seeds=np.asarray(seeds); groups=[np.flatnonzero(seeds==s) for s in np.unique(seeds)]
    lengths={len(g) for g in groups}
    if len(lengths)!=1: raise ValueError('Bootstrap expects equal-sized complete seed groups')
    draws=np.random.default_rng(seed).integers(0,len(groups),size=(n,len(groups)))
    return np.asarray(groups)[draws].reshape(n,-1)


def regression_ci(y,p,draws):
    # Vectorized seed bootstrap, conditional on the trained probe.
    a=y[draws].astype(float); b=p[draws].astype(float)
    mse=np.mean((a-b)**2,axis=(1,2)); denom=np.sum((a-a.mean(axis=1,keepdims=True))**2,axis=(1,2))
    r2=1-np.sum((a-b)**2,axis=(1,2))/np.maximum(denom,1e-20)
    return {'mse_ci_low':float(np.quantile(mse,.025)), 'mse_ci_high':float(np.quantile(mse,.975)),
            'r2_ci_low':float(np.quantile(r2,.025)), 'r2_ci_high':float(np.quantile(r2,.975))}


def holm(pvalues):
    p=np.asarray(pvalues,float); order=np.argsort(p)
    vals=np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]); out=np.empty_like(p)
    out[order]=np.minimum(vals,1); return out


def paired_label_swap(labels,seeds,rng):
    labels=np.asarray(labels); out=labels.copy()
    for seed in np.unique(seeds):
        ix=np.flatnonzero(seeds==seed)
        if len(ix)!=2: raise ValueError('Expected two histories per independent seed')
        if rng.integers(2): out[ix]=labels[ix[::-1]]
    return out


def auc_score(y,s):
    y=np.asarray(y); s=np.asarray(s)
    n1=int(y.sum()); n0=len(y)-n1
    if n1==0 or n0==0: return .5
    ranks=stats.rankdata(s)
    return float((ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0))


def event_metrics(y,s,draws=None):
    result={'accuracy':float(np.mean((s>=.5)==y)), 'auc':auc_score(y,s)}
    if draws is not None:
        acc=np.mean((s[draws]>=.5)==y[draws],axis=1)
        auc=np.array([auc_score(y[d],s[d]) for d in draws])
        result.update(accuracy_ci_low=float(np.quantile(acc,.025)),accuracy_ci_high=float(np.quantile(acc,.975)),
                      auc_ci_low=float(np.quantile(auc,.025)),auc_ci_high=float(np.quantile(auc,.975)))
    return result


class LinearProbe:
    """Dual ridge with train-only centering/scaling; reusable linear operators."""
    def __init__(self,x,split):
        self.split=split
        x=np.asarray(x,dtype=np.float64)
        self.mean=x[split['train']].mean(axis=0)
        xc=x-self.mean
        # One global train RMS avoids individual near-zero weight amplification.
        self.scale=max(float(np.sqrt(np.mean(xc[split['train']]**2))),1e-12)
        self.x=xc/self.scale/np.sqrt(x.shape[1])
        train=self.x[split['train']]
        kernel=train@train.T
        eigen,u=linalg.eigh(kernel)
        eigen=np.maximum(eigen,0)
        self.eigen=eigen; self.u=u
        self.operators=[]
        for alpha in ALPHAS:
            inv=(u/(eigen+alpha))@u.T
            self.operators.append({key:(self.x[ix]@train.T)@inv for key,ix in split.items()})

    def candidates(self,y):
        y=np.asarray(y,float); yt=y[self.split['train']]
        mean=yt.mean(axis=0); centered=yt-mean
        return [{key: op@centered+mean for key,op in ops.items()} for ops in self.operators]

    def fit_select(self,y,classification=False):
        predictions=self.candidates(y)
        val=y[self.split['validation']]
        losses=[]
        for candidate in predictions:
            if classification:
                # Proper squared loss for the linear probability score, no test tuning.
                losses.append(np.mean((candidate['validation']-val)**2,axis=0))
            else:
                losses.append(np.mean((candidate['validation']-val)**2))
        losses=np.asarray(losses)
        if classification:
            best=np.argmin(losses,axis=0)
            out={key:np.column_stack([predictions[int(best[b])][key][:,b] for b in range(y.shape[1])]) for key in self.split}
            return out, ALPHAS[best]
        best=int(np.argmin(losses))
        return predictions[best],float(ALPHAS[best])


def randomized_pca(x,components=32,seed=232):
    """Fixed randomized SVD; x contains training trajectories only."""
    x=np.asarray(x,dtype=np.float32)
    mean=x.mean(axis=0,dtype=np.float64).astype(np.float32)
    a=x-mean
    k=min(components,min(a.shape)-1); sketch=min(k+8,min(a.shape))
    rng=np.random.default_rng(seed)
    omega=rng.standard_normal((a.shape[1],sketch),dtype=np.float32)
    q,_=linalg.qr(a@omega,mode='economic',check_finite=False)
    # One power iteration; deterministic and no test information.
    q,_=linalg.qr(a@(a.T@q),mode='economic',check_finite=False)
    _,singular,vt=linalg.svd(q.T@a,full_matrices=False,check_finite=False)
    basis=vt[:k].T.copy()
    explained=singular[:k]**2/np.sum(a*a,dtype=np.float64)
    return mean,basis,explained


def fit_mlp(x,y,split,classification=False):
    """Small secondary model; fixed architecture, validation selects stopping time."""
    import torch
    torch.set_num_threads(2); torch.manual_seed(825)
    x=np.asarray(x,np.float32); y=np.asarray(y,np.float32)
    mean=x[split['train']].mean(0); scale=x[split['train']].std(0); scale[scale<1e-6]=1
    xx=torch.from_numpy((x-mean)/scale)
    ym=np.zeros(y.shape[1],np.float32) if classification else y[split['train']].mean(0)
    ys=np.ones(y.shape[1],np.float32) if classification else y[split['train']].std(0)
    ys[ys<1e-6]=1; yy=torch.from_numpy((y-ym)/ys)
    model=torch.nn.Sequential(torch.nn.Linear(x.shape[1],32),torch.nn.ReLU(),torch.nn.Linear(32,y.shape[1]))
    optimizer=torch.optim.Adam(model.parameters(),lr=.003,weight_decay=.001)
    best=np.inf; saved=None; best_epoch=0
    loss_fn=torch.nn.functional.binary_cross_entropy_with_logits if classification else torch.nn.functional.mse_loss
    for epoch in range(1,301):
        optimizer.zero_grad(); loss=loss_fn(model(xx[split['train']]),yy[split['train']]); loss.backward(); optimizer.step()
        if epoch%5==0:
            with torch.no_grad(): val=float(loss_fn(model(xx[split['validation']]),yy[split['validation']]))
            if val<best:
                best=val; best_epoch=epoch; saved={k:v.detach().clone() for k,v in model.state_dict().items()}
    model.load_state_dict(saved)
    with torch.no_grad():
        pred=model(xx); pred=torch.sigmoid(pred).numpy() if classification else pred.numpy()*ys+ym
    return pred,best_epoch


def load_records(cohort):
    if cohort=='exploratory':
        records=[]
        for path in sorted((PROJECT/'outputs/checkpoints/main').glob('*.npz')):
            with np.load(path) as a: meta=json.loads(str(a['metadata']))
            records.append(dict(cohort=cohort,run_id=meta['run_id'],seed=meta['seed'],arm=0,bits='',path=str(path),dataset=meta['dataset']))
        return records
    frame=pd.read_csv(ROOT/'trajectory_run_manifest.csv',dtype={'bits':str})
    frame=frame[frame.cohort==cohort].sort_values(['seed','arm'])
    if len(frame)!=400: raise ValueError(f'Expected complete 400-run {cohort} cohort, found {len(frame)}')
    return frame.to_dict('records')


def prepare(cohort):
    records=load_records(cohort); seeds=np.array([r['seed'] for r in records])
    split=grouped_split(seeds,strata=[r['dataset'] for r in records] if cohort=='exploratory' else None)
    out=ROOT/'analysis'/cohort; out.mkdir(parents=True,exist_ok=True)
    refidx=int(split['train'][0]); reference_record=records[refidx]
    def path_for(r):
        p=Path(r['path']); return p if p.is_absolute() else ROOT/p
    with np.load(path_for(reference_record)) as a:
        reference=canonicalize(a['W0'][-1],a['W1'][-1]); updates=a['updates'].copy()
        nt=len(updates); width=a['W0'].shape[2]
    d0=784*width; d1=width*10
    aligned=np.lib.format.open_memmap(out/'aligned_trajectories.npy',mode='w+',dtype=np.float32,shape=(len(records),nt,d0+d1))
    diagnostics=[]; permutations=[]; final_feature_error=0.; permutation_error=0.; sources=[]
    for i,record in enumerate(records):
        path=path_for(record)
        with np.load(path) as a:
            if not np.array_equal(a['updates'],updates): raise ValueError('Checkpoint schedule mismatch')
            w0,w1=a['W0'],a['W1']; c0,c1,p=canonical_trajectory(w0,w1,reference)
            aligned[i,:,:d0]=c0.reshape(nt,-1); aligned[i,:,d0:]=c1.reshape(nt,-1)
            norms=[np.linalg.norm(w0[-1]),np.linalg.norm(w1[-1]),np.linalg.norm(c0[-1]),np.linalg.norm(c1[-1])]
            diagnostics.append([float(a['accuracy'][-1]),float(a['loss'][-1]),*norms])
            if i in (0,len(records)-1):
                rng=np.random.default_rng(12); q=rng.permutation(width)
                r0,r1,_=canonical_trajectory(w0[:,:,q],w1[:,q,:],reference)
                permutation_error=max(permutation_error,float(np.max(np.abs(c0-r0))),float(np.max(np.abs(c1-r1))))
                altered=w0.copy(); altered[:-1]*=1.8
                z0,z1,p2=canonical_trajectory(altered,w1,reference)
                final_feature_error=max(final_feature_error,float(np.max(np.abs(z0[-1]-c0[-1]))),float(np.max(np.abs(z1[-1]-c1[-1]))))
                assert np.array_equal(p,p2)
            permutations.append(p)
        sources.append({'run_id':record['run_id'],'path':str(path),'sha256':sha256_file(path)})
    aligned.flush()
    assert final_feature_error==0 and permutation_error<1e-5
    splitmeta={'seed':SPLIT_SEED,'reference_run_id':reference_record['run_id'],'reference_source_sha256':sources[refidx]['sha256'],
               'reference_selection':'first training row; independent of targets and performance',
               'splits':{key:{'seeds':sorted(set(int(seeds[i]) for i in ix)), 'run_ids':[records[i]['run_id'] for i in ix]} for key,ix in split.items()},
               'alignment':'canonical gauge per checkpoint, final-only Git Re-Basin permutation frozen over time',
               'target_pca':'fit all training trajectories only; fixed dimensions 8/16/32; endpoint excluded from scores',
               'input_features':'aligned canonical final weights only; layer RMS and centering train only'}
    splitpath=ROOT/'trajectory_probe_splits.json'
    allsplits=json.loads(splitpath.read_text()) if splitpath.exists() else {}
    allsplits[cohort]=splitmeta; write_json(splitpath,allsplits)
    write_json(out/'source_manifest.json',sources)
    np.savez_compressed(out/'alignment.npz',permutations=np.array(permutations),reference_W0=reference[0],reference_W1=reference[1])
    return records,seeds,split,aligned,np.asarray(diagnostics),updates,{'final_feature_early_perturbation_error':final_feature_error,'renumbering_error':permutation_error,'reference_run_id':reference_record['run_id']}


def regression_analysis(cohort,records,seeds,split,aligned,updates,summary):
    out=ROOT/'analysis'/cohort; rows=[]
    n,nt,d=aligned.shape; d0=d-640
    train=split['train']; test=split['test']; draws=bootstrap_indices(seeds[test]); rng=np.random.default_rng(820)
    # Layer scale is fitted exclusively to training trajectories, then frozen.
    scales=[]
    for sl in (slice(0,d0),slice(d0,d)):
        sample=np.asarray(aligned[train,:,sl]); scales.append(float(np.sqrt(np.mean(sample.astype(np.float64)**2))))
    target_train=np.array(aligned[train],copy=True).reshape(-1,d)
    target_train[:,:d0]/=scales[0]; target_train[:,d0:]/=scales[1]
    mean,basis,explained=randomized_pca(target_train,32); del target_train
    z=np.empty((n,nt,basis.shape[1]),np.float32)
    for i in range(n):
        a=np.array(aligned[i],copy=True); a[:,:d0]/=scales[0]; a[:,d0:]/=scales[1]
        z[i]=(a-mean)@basis
    np.savez_compressed(out/'target_pca.npz',mean=mean,basis=basis,layer_scales=scales,explained_variance_ratio=explained,train_indices=train)
    summary['target_pca_explained_variance']={str(dim):float(explained[:dim].sum()) for dim in (8,16,32)}
    primary_predictions=None
    for layer,sl in [('both',slice(None)),('W0',slice(0,d0)),('W1',slice(d0,d))]:
        x=np.array(aligned[:,-1,sl],copy=True)
        if layer=='both':
            x[:,:d0]/=scales[0]; x[:,d0:]/=scales[1]
        probe=LinearProbe(x,split)
        for dim in (8,16,32):
            if dim>z.shape[2]: continue
            for target in ('absolute','displacement'):
                y=z[:,:,:dim].copy()
                if target=='displacement': y-=z[:,-1:,:dim]
                for t in range(nt-1):
                    pred,alpha=probe.fit_select(y[:,t])
                    baseline=np.broadcast_to(y[train,t].mean(0),(len(test),dim))
                    persistence=z[test,-1,:dim] if target=='absolute' else np.zeros((len(test),dim))
                    candidates={'ridge':pred['test']}
                    if layer=='both':
                        candidates.update(train_mean=baseline,persistence=persistence)
                        if target=='displacement':
                            # Remove the trivial known -z(T) contribution: this is
                            # exactly the absolute-position train-mean baseline in MSE.
                            candidates['mean_early_minus_final']=z[train,t,:dim].mean(0)-z[test,-1,:dim]
                    for model,yp in candidates.items():
                        row=dict(cohort=cohort,target=target,layer=layer,pca_dim=dim,model=model,time_index=t,update=int(updates[t]),lag_updates=int(updates[-1]-updates[t]),alpha=alpha if model=='ridge' else np.nan,
                                 **regression_metrics(y[test,t],yp),**regression_ci(y[test,t],yp,draws))
                        rows.append(row)
                    if layer=='both' and dim==8 and target=='absolute':
                        if primary_predictions is None: primary_predictions=np.empty((len(test),nt,8),np.float32)
                        primary_predictions[:,t]=pred['test']
        if layer=='both':
            # Fixed final-weight embedding for the nonlinear sensitivity analysis.
            mx,bx,ex=randomized_pca(x[train],32,seed=928)
            embedding=(x-mx)@bx
            np.savez_compressed(out/'input_pca.npz',mean=mx,basis=bx,train_indices=train)
            for target in ('absolute','displacement'):
                y=z[:,:-1,:8].copy()
                if target=='displacement': y-=z[:,-1:,:8]
                flat=y.reshape(n,-1)
                pp,epoch=fit_mlp(embedding,flat,split)
                for t in range(nt-1):
                    yp=pp.reshape(n,nt-1,8)[test,t]
                    rows.append(dict(cohort=cohort,target=target,layer='both',pca_dim=8,model='mlp',time_index=t,update=int(updates[t]),lag_updates=int(updates[-1]-updates[t]),alpha=np.nan,epoch=epoch,
                                     **regression_metrics(y[test,t],yp),**regression_ci(y[test,t],yp,draws)))
            # Truly independent random targets, fitted and selected using train/val only.
            random_y=rng.normal(size=(n,8))*z[train,0,:8].std(0)+z[train,0,:8].mean(0)
            pp,alpha=probe.fit_select(random_y)
            for model,yp in [('random_target_ridge',pp['test']),('random_target_mean',np.broadcast_to(random_y[train].mean(0),(len(test),8)))]:
                rows.append(dict(cohort=cohort,target='random',layer='both',pca_dim=8,model=model,time_index=0,update=0,lag_updates=int(updates[-1]),alpha=alpha,
                                 **regression_metrics(random_y[test],yp),**regression_ci(random_y[test],yp,draws)))
    primary_predictions[:,-1]=z[test,-1,:8]
    # Conditional test-only block randomization; train/val never see these labels.
    primary_idx=[i for i,row in enumerate(rows) if row['layer']=='both' and row['pca_dim']==8 and row['target']=='absolute' and row['model']=='ridge']
    groups=np.array([np.flatnonzero(seeds[test]==s) for s in np.unique(seeds[test])])
    pvals=[]
    for row_idx in (primary_idx if cohort != 'exploratory' else []):
        t=rows[row_idx]['time_index']; truth=z[test,t,:8]; pred=primary_predictions[:,t]
        observed=regression_metrics(truth,pred)['mse']; null=[]
        for b in range(999):
            ix=groups[rng.permutation(len(groups))].ravel()
            null.append(np.mean((truth[ix]-pred)**2))
        p=(1+np.count_nonzero(np.asarray(null)<=observed))/(1+len(null))
        rows[row_idx]['p_value']=p; pvals.append(p)
    for i,p in zip(primary_idx,holm(pvals)): rows[i]['p_holm']=float(p)
    np.savez_compressed(out/'regression_predictions.npz',run_ids=np.array([records[i]['run_id'] for i in test]),seeds=seeds[test],updates=updates,true_z=z[test,:,:8],pred_z=primary_predictions,final_z=z[test,-1,:8])
    np.savez_compressed(out/'heldout_examples.npz',run_ids=np.array([records[i]['run_id'] for i in test[:6]]),updates=updates,true_z=z[test[:6],:,:8],pred_z=primary_predictions[:6],final_z=z[test[:6],-1,:8])
    return rows


def event_analysis(cohort,records,seeds,split,aligned,diagnostics,updates):
    if cohort=='exploratory': return []
    labels=np.array([[int(b) for b in str(r['bits'])] for r in records])
    test=split['test']; train=split['train']; draws=bootstrap_indices(seeds[test]); d=aligned.shape[2]; d0=d-640
    rows=[]; rng=np.random.default_rng(610)
    feature_sets={'both':np.array(aligned[:,-1]),'W0':np.array(aligned[:,-1,:d0]),'W1':np.array(aligned[:,-1,d0:]),
                  'performance':diagnostics[:,:2],'norms':diagnostics[:,2:],'diagnostics':diagnostics}
    np.savez_compressed(ROOT/'analysis'/cohort/'event_labels.npz',labels=labels,seeds=seeds,run_ids=np.array([r['run_id'] for r in records]))
    for feature,x in feature_sets.items():
        if feature=='both':
            for sl in (slice(0,d0),slice(d0,d)):
                scale=np.sqrt(np.mean(x[train,sl]**2)); x[:,sl]/=max(scale,1e-12)
        if feature in ('performance','norms','diagnostics'):
            scale=x[train].std(0); scale[scale<1e-12]=1; x=x/scale
        probe=LinearProbe(x,split)
        pred,alpha=probe.fit_select(labels,classification=True)
        # Batch independent permutations through the same linear operators. Each
        # permuted bit separately reselects alpha on its own validation labels.
        shuffled=np.stack([paired_label_swap(labels,seeds,rng) for _ in range(PERMUTATIONS)],axis=1)
        flat_shuffled=shuffled.reshape(len(labels),-1)
        pp,_=probe.fit_select(flat_shuffled,classification=True)
        null=np.mean((pp['test'].reshape(len(test),PERMUTATIONS,labels.shape[1])>=.5)==shuffled[test],axis=0)
        for b in range(labels.shape[1]):
            metrics=event_metrics(labels[test,b],pred['test'][:,b],draws)
            p=(1+np.count_nonzero(null[:,b]>=metrics['accuracy']))/(PERMUTATIONS+1)
            rows.append(dict(cohort=cohort,bit=b,age_updates=int(updates[-1]-(b+1)*2*235),features=feature,model='ridge_classifier',alpha=float(alpha[b]),p_value=float(p),**metrics))
        np.savez_compressed(ROOT/'analysis'/cohort/f'event_{feature}.npz',scores=pred['test'],labels=labels[test],seeds=seeds[test],null_accuracy=null,alpha=alpha)
        if feature=='both':
            shuffled=paired_label_swap(labels,seeds,np.random.default_rng(5130))
            pp,alpha2=probe.fit_select(shuffled,classification=True)
            for b in range(labels.shape[1]):
                rows.append(dict(cohort=cohort,bit=b,age_updates=int(updates[-1]-(b+1)*2*235),features='both',model='label_shuffle',alpha=float(alpha2[b]),**event_metrics(shuffled[test,b],pp['test'][:,b],draws)))
            mean,basis,_=randomized_pca(x[train],32,seed=337)
            embedding=(x-mean)@basis
            pp,epoch=fit_mlp(embedding,labels,split,classification=True)
            for b in range(labels.shape[1]):
                rows.append(dict(cohort=cohort,bit=b,age_updates=int(updates[-1]-(b+1)*2*235),features='both',model='mlp',epoch=epoch,**event_metrics(labels[test,b],pp[test,b],draws)))
    # Holm over ALL linear feature/bit endpoints within this cohort; primary stays explicit.
    idx=[i for i,r in enumerate(rows) if r['model']=='ridge_classifier']
    for i,p in zip(idx,holm([rows[i]['p_value'] for i in idx])): rows[i]['p_holm']=float(p)
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--cohort',choices=['exploratory','one_bit','multi_bit','all'],default='all')
    parser.add_argument('--keep-aligned',action='store_true'); args=parser.parse_args()
    cohorts=['exploratory','one_bit','multi_bit'] if args.cohort=='all' else [args.cohort]
    with threadpool_limits(limits=3):
        for cohort in cohorts:
            started=time.monotonic(); print(f'Preparing {cohort}',flush=True)
            records,seeds,split,aligned,diagnostics,updates,summary=prepare(cohort)
            summary.update(cohort=cohort,runs=len(records),independent_seeds=len(np.unique(seeds)),
                           split_runs={k:len(v) for k,v in split.items()},split_seeds={k:len(np.unique(seeds[v])) for k,v in split.items()},
                           final_C_accuracy_mean=float(diagnostics[:,0].mean()),final_C_accuracy_range=[float(diagnostics[:,0].min()),float(diagnostics[:,0].max())],
                           final_C_loss_mean=float(diagnostics[:,1].mean()),source_sha256=sha256_file(__file__))
            print(f'Regression {cohort}',flush=True)
            reg=regression_analysis(cohort,records,seeds,split,aligned,updates,summary)
            print(f'Events {cohort}',flush=True)
            events=event_analysis(cohort,records,seeds,split,aligned,diagnostics,updates)
            for name,rows in [('trajectory_regression_results.csv',reg),('event_decoding_results.csv',events)]:
                if not rows: continue
                path=ROOT/name
                frame=pd.read_csv(path) if path.exists() else pd.DataFrame()
                if len(frame): frame=frame[frame.cohort!=cohort]
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',RuntimeWarning)
                    pd.concat([frame,pd.DataFrame(rows)],ignore_index=True).to_csv(path,index=False)
            summary['elapsed_seconds']=time.monotonic()-started
            write_json(ROOT/'analysis'/cohort/'summary.json',summary)
            del aligned
            if not args.keep_aligned: (ROOT/'analysis'/cohort/'aligned_trajectories.npy').unlink()
            print(json.dumps(summary),flush=True)

if __name__=='__main__': main()
