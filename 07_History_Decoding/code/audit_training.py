"""Independent read-only integrity audit of the complete paired-history cohort.

No import of experiment training implementation: initialization/order and archive
consistency are recomputed independently. Only the audit receipt is written.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from learning_trajectory.training import build_model
from learning_trajectory.data import sha256_file, _atomic_json
DEFAULT_OUTPUT=ROOT/'07_History_Decoding'


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def ahash(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def whash(w0,w1):
    return hashlib.sha256(np.ascontiguousarray(w0).tobytes()+np.ascontiguousarray(w1).tobytes()).hexdigest()


def equal_json(a,b):
    return json.dumps(a,sort_keys=True)==json.dumps(b,sort_keys=True)


def audit(output=DEFAULT_OUTPUT, seeds=None, modes=None, require_manifest=True):
    output=Path(output)
    torch.set_num_threads(1)
    start=time.monotonic()
    plan=json.loads((output/'plan.json').read_text())
    freeze=json.loads((output/'provenance/freeze_receipt.json').read_text())
    expected_seeds=plan['seeds']
    seeds=expected_seeds if seeds is None else list(seeds)
    modes=list(plan['conditions']) if modes is None else list(modes)
    require(set(seeds)<=set(expected_seeds),'Unknown seeds')
    require(set(modes)<=set(plan['conditions']),'Unknown momentum condition')
    full=set(seeds)==set(expected_seeds) and set(modes)==set(plan['conditions'])
    plan_hash=sha256_file(output/'plan.json')
    probe_hash=sha256_file(output/'probe_inputs.npz')
    require(plan_hash==freeze['plan_sha256'],'Frozen amended plan hash mismatch')
    require(probe_hash==freeze['probe_inputs_sha256']==plan['probe_inputs_sha256'],'Probe hash mismatch')
    require(sha256_file(output/'probe_splits.json')==freeze['splits_sha256']==plan['probe_splits_sha256'],'Split hash mismatch')
    with np.load(output/'probe_inputs.npz',allow_pickle=False) as p:
        probe_indices=p['indices'].copy()
    require(probe_indices.shape==(256,) and len(np.unique(probe_indices))==256,'Probe input count/uniqueness')
    sources={'training_source':output/'code/training.py','original_training_source':ROOT/'src/learning_trajectory/training.py',
             'original_data_source':ROOT/'src/learning_trajectory/data.py','original_metrics_source':ROOT/'src/learning_trajectory/metrics.py',
             'plan':output/'plan.json','probe_inputs':output/'probe_inputs.npz'}
    source_hashes={k:sha256_file(p) for k,p in sources.items()}
    dm=json.loads((ROOT/'data/mnist/manifest.json').read_text())
    data_hashes={r['filename']:sha256_file(ROOT/'data/mnist'/r['filename']) for r in dm['files']}
    require(all(data_hashes[r['filename']]==r['sha256'] for r in dm['files']),'Original dataset hash mismatch')
    hcount=len(plan['config']['horizons']); width=plan['config']['hidden_width']
    n_epochs=2*plan['config']['ab_epochs']+plan['config']['c_epochs']
    histories=[['A','B','C'],['B','A','C']]
    expected_diag=['accuracy','loss','raw_W0_frob','raw_W1_frob','canonical_hidden_rms','logit_rms']
    records=[]; manifest_lookup={}; all_contexts=set(); timings=[]
    for seed in seeds:
        model=build_model(seed,10,width)
        init0=model[0].weight.detach().numpy().T.copy()
        init1=model[2].weight.detach().numpy().T.copy()
        init_hash=whash(init0,init1)
        order_seeds=[900000+seed*32+e for e in range(n_epochs)]
        orders=[torch.randperm(plan['n_training_images'],generator=torch.Generator().manual_seed(s)) for s in order_seeds]
        order_hashes=[ahash(o.numpy()) for o in orders]
        condition_initials=[]
        for mode in modes:
            path=output/'checkpoints'/mode/f'seed_{seed}.npz'
            require(path.exists(),f'Missing checkpoint {path}')
            with np.load(path,allow_pickle=False) as a:
                m=json.loads(str(a['metadata']))
                require(int(a['seed'])==seed and m['seed']==seed and m['mode']==mode,f'Archive identity {path}')
                require(a['horizons'].tolist()==plan['config']['horizons'],f'Horizons {path}')
                shapes={'W0':(2,hcount,784,width),'W1':(2,hcount,width,10),'accuracy':(2,hcount),'loss':(2,hcount),
                        'diagnostics':(2,hcount,6),'logits':(2,hcount,256,10),'bias0':(2,hcount,width),'bias1':(2,hcount,10)}
                for key,shape in shapes.items():
                    require(a[key].shape==shape and np.isfinite(a[key]).all(),f'Shape/finite {key}: {path}')
                require(not a['bias0'].any() and not a['bias1'].any(),f'Nonzero bias {path}')
                require(np.all((a['accuracy']>=0)&(a['accuracy']<=1)) and np.all(a['loss']>=0),f'Invalid performance range {path}')
                require(np.array_equal(a['probe_indices'],probe_indices),f'Probe input identity {path}')
                require(a['diagnostic_names'].tolist()==expected_diag,f'Diagnostic names {path}')
                require(np.array_equal(a['diagnostics'][...,:2],np.stack([a['accuracy'],a['loss']],axis=-1)),f'Diagnostic performance consistency {path}')
                require(np.array_equal(a['initial_W0'],init0) and np.array_equal(a['initial_W1'],init1),f'Regenerated initialization {path}')
                condition_initials.append(whash(a['initial_W0'],a['initial_W1']))
                require(m['initialization_sha256']==init_hash and m['paired_initialization_sha256']==[init_hash,init_hash] and m['paired_initialization_equal'],f'Paired initialization hashes {path}')
                require(m['order_seeds']==order_seeds and m['order_sha256']==order_hashes and m['identical_orders_both_histories'],f'Regenerated shuffle orders {path}')
                require(m['histories']==histories and m['label_offsets']=={'A':0,'B':1,'C':3},f'Task definitions {path}')
                require(m['sample_counts']=={'train':60000,'test':10000,'probe':256},f'Full sample count {path}')
                require(m['updates_per_epoch']==235,f'Batch count {path}')
                cfg=m['config']
                for key in ('hidden_width','batch_size','learning_rate','momentum','c_epochs','horizons','threads'):
                    require(cfg[key]==plan['config'][key],f'Config {key}: {path}')
                require(cfg['pre_epochs']==plan['config']['ab_epochs'],f'A/B phase length {path}')
                ctx=m['context']; fingerprint=ctx['fingerprint']
                require(ctx['source_hashes']==source_hashes and ctx['dataset_sha256']==data_hashes,f'Source/data fingerprints {path}')
                require(equal_json(ctx['config'],cfg),f'Context config {path}')
                require(ctx['torch_version']==str(torch.__version__),f'PyTorch provenance {path}')
                digest=hashlib.sha256(json.dumps({k:v for k,v in ctx.items() if k!='fingerprint'},sort_keys=True).encode()).hexdigest()
                require(fingerprint==digest,f'Context fingerprint integrity {path}')
                all_contexts.add(fingerprint)
                transitions=m['transition_audit']
                require(len(transitions)==6,f'Transition audit count {path}')
                for h in range(2):
                    for phase in range(3):
                        t=transitions[h*3+phase]
                        require((t['history'],t['phase'],t['task'])==(h,phase,histories[h][phase]),f'Transition identity {path}')
                        before,after=t['momentum_norm_before'],t['momentum_norm_start']
                        if phase==0:
                            require(before==after==0 and not t['reset'],f'Initial optimizer state {path}')
                        elif mode=='reset':
                            require(before>0 and after==0 and t['reset'],f'Momentum reset {path}')
                        else:
                            require(before>0 and before==after and not t['reset'],f'Momentum carried {path}')
                epochs=m['epoch_training']
                require(len(epochs)==2*n_epochs,f'Epoch log count {path}')
                for h in range(2):
                    for e in range(n_epochs):
                        ep=epochs[h*n_epochs+e]
                        task=histories[h][0] if e<cfg['pre_epochs'] else histories[h][1] if e<2*cfg['pre_epochs'] else 'C'
                        phase_ep=e+1 if e<cfg['pre_epochs'] else e-cfg['pre_epochs']+1 if e<2*cfg['pre_epochs'] else e-2*cfg['pre_epochs']+1
                        require(ep['history']==h and ep['global_epoch']==e and ep['task']==task and ep['phase_epoch']==phase_ep,f'Epoch identity {path}')
                        require(ep['updates']==(e+1)*235 and ep['order_seed']==order_seeds[e] and ep['order_sha256']==order_hashes[e],f'Epoch updates/orders {path}')
                        require(np.isfinite(ep['train_loss']) and ep['train_loss']>=0,f'Training loss range {path}')
                pt_path=path.with_suffix('.pt'); pt_hash=sha256_file(pt_path)
                require(pt_hash==m['full_state_sha256'],f'Full-state file integrity {path}')
                pt=torch.load(pt_path,map_location='cpu',weights_only=False)
                require(equal_json(pt['metadata'],{k:v for k,v in m.items() if k!='full_state_sha256'}),f'PT/NPZ metadata equality {path}')
                require(np.array_equal(pt['initial_W0'],init0) and np.array_equal(pt['initial_W1'],init1),f'Full-state initialization {path}')
                require(len(pt['checkpoints'])==2 and all(len(arm)==hcount for arm in pt['checkpoints']),f'Full-state count {path}')
                for h in range(2):
                    for j,c in enumerate(plan['config']['horizons']):
                        s=pt['checkpoints'][h][j]; g=2*cfg['pre_epochs']+c-1
                        require(s['history']==h and s['c_epoch']==c and s['global_epoch_completed']==g and s['updates']==plan['c_checkpoint_updates'][j],f'Full-state position {path}')
                        state=s['model_state_dict']
                        for key,matrix in [('0.weight','W0'),('2.weight','W1')]:
                            require(np.array_equal(state[key].numpy().T,a[matrix][h,j]),f'Full-state weights {path}')
                        require(not torch.count_nonzero(state['0.bias']) and not torch.count_nonzero(state['2.bias']),f'Full-state biases {path}')
                        opt=s['optimizer_state_dict']
                        require(len(opt['param_groups'])==1 and set(opt['state'])=={0,1},f'Optimizer parameter state {path}')
                        group=opt['param_groups'][0]
                        require(group['params']==[0,1] and group['lr']==cfg['learning_rate'] and group['momentum']==cfg['momentum'],f'Optimizer settings {path}')
                        require(group['weight_decay']==0 and group['dampening']==0 and not group['nesterov'] and not group['maximize'],f'Unexpected SGD variant {path}')
                        for k,shape in [(0,(width,784)),(1,(10,width))]:
                            buf=opt['state'][k]['momentum_buffer']
                            require(tuple(buf.shape)==shape and bool(torch.isfinite(buf).all()),f'Momentum array {path}')
                        require(torch.equal(s['last_epoch_order'],orders[g]),f'Full-state last order {path}')
                        require(s['next_order_seed']==900000+seed*32+g+1,f'Next shuffle seed {path}')
                        require(s['torch_rng_state'].dtype==torch.uint8 and len(s['numpy_rng_state'])==5 and len(s['python_rng_state'])==3,f'RNG state presence {path}')
                npz_hash=sha256_file(path)
                for h in range(2):
                    for j,c in enumerate(plan['config']['horizons']):
                        manifest_lookup[(seed,mode,h,c)]={'path':str(path.relative_to(output)),'checkpoint_sha256':npz_hash,
                            'full_state_sha256':pt_hash,'checkpoint_index':j,'accuracy':float(a['accuracy'][h,j]),'loss':float(a['loss'][h,j]),
                            'initialization_sha256':init_hash,'context_fingerprint':fingerprint}
                timings.append(m['training_seconds'])
                records.append({'seed':seed,'mode':mode,'passed':True,'npz_sha256':npz_hash,'pt_sha256':pt_hash,
                    'initialization_sha256':init_hash,'training_seconds':m['training_seconds']})
        require(len(set(condition_initials))==1,f'Cross-condition initialization identity seed={seed}')
    require(len(all_contexts)==1,'Multiple training context fingerprints')
    manifest_rows=None
    if require_manifest:
        with (output/'history_run_manifest.csv').open(newline='') as f: manifest=list(csv.DictReader(f))
        require(len(manifest)==len(manifest_lookup),'Run manifest row count')
        keys=set()
        for r in manifest:
            key=(int(r['seed']),r['momentum_mode'],int(r['history']),int(r['C_horizon']))
            require(key in manifest_lookup and key not in keys,'Run manifest unique identity')
            keys.add(key); expected=manifest_lookup[key]
            require(r['final_checkpoint']==expected['path'],'Run manifest checkpoint path')
            for field in ('checkpoint_sha256','full_state_sha256','initialization_sha256','context_fingerprint'):
                require(r[field]==expected[field],f'Manifest {field}')
            for field in ('accuracy','loss'): require(float(r[field])==expected[field],f'Manifest {field}')
            require(int(r['checkpoint_index'])==expected['checkpoint_index'],'Manifest checkpoint index')
            require(r['momentum_reset']==str(key[1]=='reset'),'Manifest momentum reset')
            require(r['task_order']=='→'.join(histories[key[2]]),'Manifest history order')
            require(json.loads(r['phase_epochs'])==[1,1,key[3]],'Manifest phase lengths')
            require(float(r['learning_rate'])==.1 and float(r['momentum'])==.9 and int(r['batch_size'])==256 and int(r['updates_per_epoch'])==235,'Manifest optimizer schedule')
            require(json.loads(r['label_permutations'])==plan['task_permutations'],'Manifest task permutations')
        manifest_rows=len(manifest)
    if full:
        actual=set((output/'checkpoints').glob('*/seed_*.npz'))
        expected={output/'checkpoints'/m/f'seed_{s}.npz' for m in modes for s in seeds}
        require(actual==expected,'Unexpected or missing cohort archives')
        require(len(records)==400 and (manifest_rows==2400 or not require_manifest),'Frozen full cohort counts')
    return {'passed':True,'complete_cohort':full,'audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'independent_of_training_implementation':True,'audit_source_sha256':sha256_file(__file__),
        'plan_sha256':plan_hash,'probe_inputs_sha256':probe_hash,'context_fingerprint':next(iter(all_contexts)),
        'seed_blocks':len(seeds),'conditions':modes,'pair_archives':len(records),'training_histories':len(records)*2,
        'full_checkpoints':len(records)*2*hcount,'manifest_rows':manifest_rows,
        'checks':['read-only original dataset hashes','amended frozen plan/splits/probe hashes','exact expected seeds and dimensions',
            'baseline-generated initialization and cross-arm/cross-condition identity','independently regenerated twelve shuffle orders',
            'all transition momentum reset or exact carry','zero frozen biases','NPZ/PT weights and metadata equality',
            'all optimizer momentum tensors and RNG states','full-state byte hashes','all manifest rows'],
        'training_seconds_mean':float(np.mean(timings)),'training_seconds_median':float(np.median(timings)),
        'wall_seconds':time.monotonic()-start,'archives':records}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=p.parse_args()
    result=audit(args.output)
    _atomic_json(args.output/'provenance/training_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='archives'},indent=2),flush=True)
if __name__=='__main__': main()
