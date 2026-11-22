"""Cross-time initialization retrieval using representations on a fixed probe.

The raw hidden-representation endpoint is fixed before computing results. No
weight distance, neuron assignment, model training or parent identity enters CKA.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
DATASETS=("mnist","fashion_mnist","kmnist")
PRIMARY_KEY="one_minus_cka|hidden|none|raw"
METRIC_KEYS=(PRIMARY_KEY,"one_minus_cka|hidden|none|standardized","one_minus_cka|logits|none|raw")
INFERENCE_SEED=2026112000


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _module(path,name):
    spec=importlib.util.spec_from_file_location(name,Path(path))
    if spec is None or spec.loader is None:raise ImportError(path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def _json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def centered_features(representation,standardized=False):
    """Center probe rows; optionally set each nonconstant feature's population SD to 1."""
    features=np.asarray(representation,dtype=np.float64)
    if features.ndim!=2 or features.shape[0]<2 or not np.isfinite(features).all():
        raise ValueError('Expected finite [probe examples, features] representation')
    centered=features-features.mean(axis=0,keepdims=True)
    # Exact constant columns can acquire a tiny constant residual from summing
    # non-binary decimals. They must remain zero, especially before SD scaling.
    centered[:,np.ptp(features,axis=0)==0]=0
    sd=np.sqrt(np.mean(centered**2,axis=0))
    nonzero=sd>0
    if standardized:
        centered=np.divide(centered,sd[None,:],out=np.zeros_like(centered),where=nonzero[None,:])
    gram=centered.T@centered
    normalizer=float(np.linalg.norm(gram,'fro'))
    if not np.isfinite(normalizer) or normalizer<=0:
        raise ValueError('CKA undefined: the entire centered representation has zero variance; no fallback')
    return centered,normalizer,int(np.count_nonzero(~nonzero))


def cka_cross(queries,candidates,query_normalizers=None,candidate_normalizers=None):
    """Biased centered linear CKA from centered features, without row/feature matching.

    The common probe example order is fixed. Hidden features need no correspondence:
    ||X.T@Y||_F^2 / (||X.T@X||_F ||Y.T@Y||_F). Biased HSIC factors
    cancel in this ratio. All query/candidate entries use this same computation.
    """
    q=[np.asarray(a,dtype=np.float64) for a in queries]
    c=[np.asarray(a,dtype=np.float64) for a in candidates]
    if not q or not c:raise ValueError('Both representation ensembles must be nonempty')
    if any(a.shape!=q[0].shape for a in q) or any(b.shape!=c[0].shape for b in c) or q[0].shape[0]!=c[0].shape[0]:
        raise ValueError('Consistent probe/feature dimensions within each ensemble required')
    qnorm=np.asarray(query_normalizers if query_normalizers is not None else [np.linalg.norm(a.T@a,'fro') for a in q])
    cnorm=np.asarray(candidate_normalizers if candidate_normalizers is not None else [np.linalg.norm(a.T@a,'fro') for a in c])
    if qnorm.shape!=(len(q),) or cnorm.shape!=(len(c),) or not np.isfinite(qnorm).all() or not np.isfinite(cnorm).all() or np.any(qnorm<=0) or np.any(cnorm<=0):
        raise ValueError('Positive finite CKA normalizers required; zero-variance logits are not replaced')
    cross=np.concatenate(q,axis=1).T@np.concatenate(c,axis=1)
    blocks=cross.reshape(len(q),q[0].shape[1],len(c),c[0].shape[1])
    numerators=np.sum(blocks**2,axis=(1,3))
    raw=numerators/(qnorm[:,None]*cnorm[None,:])
    if not np.isfinite(raw).all() or raw.min() < -1e-12 or raw.max()>1+1e-10:
        raise ValueError('CKA outside its numerical admissible range')
    clipped=np.clip(raw,0,1)
    return 1-clipped,raw


def definitions():
    common={
        'metric':'one_minus_cka','alignment':'none','formula':'1 - ||Xc.T @ Yc||_F^2 / (||Xc.T @ Xc||_F * ||Yc.T @ Yc||_F)',
        'estimator':'biased centered linear CKA; rows are the same 3000 fixed probe examples in the same order',
        'orientation':'distances[k,t,i,j]: representation of query run i at E=t versus candidate run j at E=0',
        'identity_policy':'same label-blind calculation for all 100 entries; no diagonal substitution or preference',
        'zero_variance':'Individual constant columns contribute zero; an entirely constant centered representation raises an error. No replacement experiment.',
        'neuron_assignment':'none','training_performed':False,
    }
    return {
        PRIMARY_KEY:{**common,'layer':'hidden','gauge':'raw','features':'saved ReLU hidden activations H from original activation archive, mean-centered per feature',
                     'inferential_role':'prespecified_primary','scaling_caution':'Raw hidden CKA is invariant to whole-matrix scaling and orthogonal feature rotations, but not arbitrary independent positive neuron rescaling.'},
        METRIC_KEYS[1]:{**common,'layer':'hidden','gauge':'standardized','features':'saved H; center each feature then divide nonconstant columns by their population SD; exactly zero-variance columns are set to zero',
                        'inferential_role':'prespecified_exploratory_robustness','scaling_caution':'Standardization removes positive per-neuron scaling on the fixed probe; no candidate-dependent fitting.'},
        METRIC_KEYS[2]:{**common,'layer':'logits','gauge':'raw','features':'saved H @ corresponding saved W1; original biases are verified to be fixed zeros; center each logit feature across the probe',
                        'inferential_role':'prespecified_exploratory_robustness','scaling_caution':'Uses logits, not softmax probabilities; includes no neuron correspondence or weight-space distance.'},
    }


def build_representations(root:Path=ROOT,out:Path|None=None):
    root=Path(root).resolve();out=(root/'04_Representation_Retrieval') if out is None else Path(out).resolve()
    if out!=root/'04_Representation_Retrieval':raise ValueError('All outputs must be isolated in 04_Representation_Retrieval')
    source_path=root/'01_RSA/code/distances.py'
    source=_module(source_path,'_representation_retrieval_manifest_source_'+sha256(source_path)[:16])
    source_validation=root/'01_RSA/data/validation.json'
    validated=json.loads(source_validation.read_text())
    if not validated.get('passed') or validated['code_sha256']!=sha256(source_path):
        raise ValueError('Original manifest-loading source does not match its successful validation')
    groups,manifest,run_audit=source.load_manifest_runs(root)
    _,pipeline_audit=source.load_pipeline_artifacts(root,manifest)
    probe_path=root/'outputs/probes/main.npz';probe_hash=sha256(probe_path)
    with np.load(probe_path,allow_pickle=False) as probe:
        x=probe['X'].astype(np.float64)
        if x.shape!=(3000,784) or not np.isfinite(x).all():raise ValueError('Invalid common probe')
        if any(np.count_nonzero(probe['datasets'].astype(str)==d)!=1000 for d in DATASETS):
            raise ValueError('Expected 1000 fixed probe examples from each source dataset')
    data=out/'data';data.mkdir(parents=True,exist_ok=True)
    plan={'primary_key':PRIMARY_KEY,'metric_keys':list(METRIC_KEYS),'base_seed':INFERENCE_SEED,'primary_checkpoint':20,
          'test_family':'three dataset-specific raw-hidden E20 endpoints, Holm correction; all other endpoints exploratory',
          'candidate_checkpoint':0,'fixed_probe_path':str(probe_path),'fixed_probe_sha256':probe_hash,'probe_size':3000,
          'within_dataset_only':True,'new_training':False,'weight_distances':False,'neuron_assignments':False,
          'definition_fixed_before_results':True}
    _json(out/'analysis_plan.json',plan)
    _json(data/'distance_definitions.json',definitions())
    _json(data/'source_manifest.json',manifest)
    values={d:np.empty((3,21,10,10)) for d in DATASETS}
    cka_unclipped={d:np.empty((3,21,10,10)) for d in DATASETS}
    normalizers={d:np.empty((3,21,10)) for d in DATASETS}
    zero_columns={d:np.empty((3,21,10),dtype=np.int16) for d in DATASETS}
    initial={};initial_norms={};activation_audit=[];recheck_errors=[]
    all_run_ids={r['run_id'] for runs in groups.values() for r in runs}
    with threadpool_limits(limits=1):
        for checkpoint in range(21):
            receipt_path=root/f'outputs/pairs/main/checkpoint_{checkpoint:04d}.json'
            receipt=json.loads(receipt_path.read_text())
            activation_path=Path(receipt['activation_path'])
            if not activation_path.is_absolute():activation_path=root/activation_path
            digest=sha256(activation_path)
            if digest!=receipt['activation_sha256']:raise ValueError(f'Activation hash mismatch: {activation_path}')
            with np.load(activation_path,allow_pickle=False) as archive:
                ids=archive['run_ids'].astype(str).tolist()
                h_all=archive['H'].astype(np.float64)
            if len(ids)!=30 or len(set(ids))!=30 or set(ids)!=all_run_ids or h_all.shape!=(30,3000,64):
                raise ValueError(f'Activation identities/dimensions mismatch: {activation_path}')
            if not np.isfinite(h_all).all() or np.any(h_all<0):raise ValueError('Invalid saved ReLU activations')
            for dataset in DATASETS:
                prepared=[[],[],[]]
                for i,run in enumerate(groups[dataset]):
                    h=h_all[ids.index(run['run_id'])]
                    if checkpoint in (0,20) and i==0:
                        expected=np.maximum(x@run['W0'][checkpoint],0)
                        error=float(np.max(np.abs(h-expected)))
                        if error>1e-10:raise ValueError('Saved H differs from probe @ manifest-bound W0')
                        recheck_errors.append({'dataset':dataset,'run_id':run['run_id'],'checkpoint':checkpoint,'max_abs_error':error})
                    logits=h@run['W1'][checkpoint]
                    for k,(features,standardized) in enumerate(((h,False),(h,True),(logits,False))):
                        centered,norm,zeros=centered_features(features,standardized)
                        prepared[k].append(centered)
                        normalizers[dataset][k,checkpoint,i]=norm
                        zero_columns[dataset][k,checkpoint,i]=zeros
                if checkpoint==0:
                    initial[dataset]=prepared
                    initial_norms[dataset]=normalizers[dataset][:,0].copy()
                for k in range(3):
                    values[dataset][k,checkpoint],cka_unclipped[dataset][k,checkpoint]=cka_cross(
                        prepared[k],initial[dataset][k],normalizers[dataset][k,checkpoint],initial_norms[dataset][k])
            activation_audit.append({'checkpoint':checkpoint,'path':str(activation_path),'sha256':digest,
                                     'original_receipt_path':str(receipt_path),'original_receipt_sha256':sha256(receipt_path),
                                     'run_ids':ids,'shape':[30,3000,64]})
            print(f'Representation retrieval: E={checkpoint}, three datasets x 100 pairs x 3 CKA variants',flush=True)
    outputs=[]
    for dataset in DATASETS:
        runs=groups[dataset];run_ids=np.asarray([r['run_id'] for r in runs]);path=data/f'{dataset}_retrieval.npz'
        diagonal=float(np.max(np.abs(np.diagonal(values[dataset][:,0],axis1=-2,axis2=-1))))
        symmetry=float(np.max(np.abs(values[dataset][:,0]-values[dataset][:,0].swapaxes(-1,-2))))
        if diagonal>1e-10 or symmetry>1e-10:raise AssertionError('E0 CKA self/symmetry sanity failed')
        np.savez_compressed(path,distances=values[dataset],cka_unclipped=cka_unclipped[dataset],metric_keys=np.asarray(METRIC_KEYS),
                            checkpoints=np.arange(21),run_ids=run_ids,query_run_ids=run_ids,candidate_run_ids=run_ids,
                            seeds=np.asarray([r['seed'] for r in runs]),candidate_checkpoint=np.asarray(0),
                            gram_frobenius_normalizers=normalizers[dataset],zero_variance_feature_counts=zero_columns[dataset],
                            feature_dimensions=np.asarray([64,64,10]),probe_sha256=np.asarray(probe_hash),
                            activation_sha256=np.asarray([r['sha256'] for r in activation_audit]))
        outputs.append({'dataset':dataset,'path':str(path),'sha256':sha256(path),'shape':list(values[dataset].shape),
                        'query_run_ids':run_ids.tolist(),'candidate_run_ids':run_ids.tolist(),'E0_max_diagonal_distance':diagonal,
                        'E0_max_symmetry_error':symmetry,'zero_variance_feature_counts_max':zero_columns[dataset].max(axis=(1,2)).tolist()})
    validation={'passed':True,'code_sha256':sha256(__file__),'primary_key':PRIMARY_KEY,'n_metric_keys':3,'n_distances':3*3*21*100,
                'outputs':outputs,'runs':run_audit,'original_pipeline_sources':pipeline_audit,'activation_archives':activation_audit,
                'source_manifest_path':str(root/'outputs/manifests/main.json'),'source_manifest_sha256':sha256(root/'outputs/manifests/main.json'),
                'probe_path':str(probe_path),'probe_sha256':probe_hash,'source_loader_sha256':sha256(source_path),
                'H_recomputed_from_manifest_weights_checks':recheck_errors,'candidate_checkpoint':0,'biases_verified_zero':True,
                'all_entries_computed_identically':True,'diagonal_overridden':False,'neuron_matching_performed':False,
                'normalization':'Feature centering for all; featurewise population SD only for standardized hidden robustness',
                'zero_variance_policy':'Individual constant features retained as zero columns; entire zero-variance representation raises an error',
                'previous_experiments_modified':False,'training_performed':False}
    _json(data/'validation.json',validation)
    return validation


def run_statistics(root:Path=ROOT,out:Path|None=None):
    root=Path(root).resolve();out=root/'04_Representation_Retrieval' if out is None else Path(out).resolve()
    if out!=root/'04_Representation_Retrieval':raise ValueError('Inference output must be isolated in 04')
    validation=json.loads((out/'data/validation.json').read_text())
    if not validation.get('passed') or validation['code_sha256']!=sha256(__file__):
        raise ValueError('Representational distances have stale or failed validation')
    if any(sha256(Path(row['path']))!=row['sha256'] for row in validation['outputs']):
        raise ValueError('Representation distance archive hash mismatch')
    path=root/'02_Initialization_Retrieval/code/inference.py'
    before=sha256(path)
    frozen=json.loads((root/'02_Initialization_Retrieval/artifact_manifest.json').read_text())
    expected=next(row['sha256'] for row in frozen['files'] if row['path']=='code/inference.py')
    if before!=expected:raise ValueError('Point 1 inference source differs from its frozen artifact manifest')
    inference=_module(path,'_representation_retrieval_frozen_inference_'+before[:16])
    result=inference.run_inference(out,base_seed=INFERENCE_SEED,primary_key=PRIMARY_KEY,controls=False)
    if sha256(path)!=before:raise AssertionError('Original inference source changed')
    _json(out/'provenance/inference_source.json',{'path':str(path),'sha256':before,'unmodified':True,
                                               'base_seed':INFERENCE_SEED,'primary_key':PRIMARY_KEY,'controls':False})
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    parser.add_argument('--stage',choices=('all','distances','inference'),default='all')
    args=parser.parse_args()
    if args.stage in ('all','distances'):build_representations(args.root)
    if args.stage in ('all','inference'):run_statistics(args.root)
