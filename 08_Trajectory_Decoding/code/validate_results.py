#!/usr/bin/env python3
"""Independent archive/split/provenance audit and paired performance controls.

This deliberately does not import train_cohorts.py or analyze.py. Its expectations
come from the frozen plan and independently reconstructed phase/update schedules.
"""
from __future__ import annotations
import argparse
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import shutil
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parent

def digest_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def digest_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

def initial_digest(w0,w1):
    h=hashlib.sha256();h.update(np.ascontiguousarray(w0).tobytes());h.update(np.ascontiguousarray(w1).tobytes());return h.hexdigest()

def require(condition,message):
    if not condition:raise AssertionError(message)

def bootstrap_difference(differences,reps=1000,seed=2026111608):
    d=np.asarray(differences,dtype=np.float64)
    require(len(d)>0,'No paired differences available')
    rng=np.random.default_rng(seed)
    draws=d[rng.integers(0,len(d),size=(reps,len(d)))].mean(axis=1)
    return float(d.mean()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975))

def expected_tasks(bits,config):
    windows=[]
    for bit in bits:windows.extend(['A','B'] if bit==0 else ['B','A'])
    tasks=[task for task in windows for _ in range(config['window_epochs'])]+['C']*config['c_epochs']
    return windows,tasks

def audit_run(row,root,plan,sources,order_cache):
    cohort=row['cohort'];seed=int(row['seed']);arm=int(row['arm']);config=plan['config'];n_bits=1 if cohort=='one_bit' else 4
    run_id=f'{cohort}_s{seed}_h{arm}'
    require(row['run_id']==run_id,'Run ID differs from cohort/seed/arm')
    require(arm in (0,1),'Invalid arm')
    bits=tuple(int(b) for b in row['bits'])
    require(len(bits)==n_bits and set(bits)<={0,1},'Invalid bit vector')
    expected=(arm,) if cohort=='one_bit' else tuple((int(x)^arm) for x in np.random.default_rng(810000+seed).integers(0,2,size=4))
    require(bits==expected,'Bit vector differs from preregistered RNG')
    path=root/'checkpoints'/cohort/(run_id+'.npz');pt_path=path.with_suffix('.pt')
    require(Path(row['path']).resolve()==path.resolve(),'Manifest NPZ path not canonical')
    require(Path(row['final_state_path']).resolve()==pt_path.resolve(),'Manifest PT path not canonical')
    npz_hash=digest_file(path);pt_hash=digest_file(pt_path)
    require(npz_hash==row['npz_sha256'],'Manifest NPZ checksum differs')
    require(pt_hash==row['final_state_sha256'],'Manifest PT checksum differs')
    windows,tasks=expected_tasks(bits,config);batches=(60000+config['batch_size']-1)//config['batch_size']
    updates=np.asarray([0]+[e*batches+v for e in range(len(tasks)) for v in ((batches+1)//2,batches)],dtype=np.int64)
    reset_updates=list(range(0,(len(windows)*config['window_epochs']+1)*batches,config['window_epochs']*batches))
    expected_phase=['initial']+[task for task in tasks for _ in range(2)]
    shuffle_seeds=[5000000+seed*100+e for e in range(len(tasks))]
    if seed not in order_cache:
        hashes=[]
        for shuffle_seed in shuffle_seeds:
            order=torch.randperm(60000,generator=torch.Generator().manual_seed(shuffle_seed))
            hashes.append(digest_array(order.numpy()))
        order_cache[seed]=hashes
    with np.load(path,allow_pickle=False) as archive:
        arrays={key:archive[key] for key in archive.files if key!='metadata'}
        meta=json.loads(str(archive['metadata']))
    require(set(arrays)==set(meta['array_sha256']),'Stored array hash inventory differs')
    for key,value in arrays.items():require(digest_array(value)==meta['array_sha256'][key],f'Payload checksum differs: {key}')
    for key,value in sources.items():require(meta[key]==value,f'Frozen source differs: {key}')
    require(meta['final_state_sha256']==pt_hash,'Metadata final state checksum differs')
    require(meta['config']==config,'Archive configuration differs from frozen plan')
    require(meta['cohort']==cohort and meta['seed']==seed and meta['arm']==arm and tuple(meta['bits'])==bits,'Archive identity differs from manifest')
    require(meta['windows']==windows and meta['epoch_tasks']==tasks,'Curriculum differs')
    require(meta['shuffle_seeds']==shuffle_seeds,'Shuffle RNG schedule differs')
    require(meta['order_sha256']==order_cache[seed],'Reconstructed epoch image order differs')
    require(meta['momentum_reset_updates']==reset_updates,'Momentum reset schedule differs')
    require(meta['train_samples']==60000 and meta['test_samples']==10000,'Incomplete MNIST sample set')
    require(np.array_equal(arrays['updates'],updates),'Checkpoint update grid differs')
    require(np.array_equal(arrays['checkpoints'],np.arange(len(updates))),'Checkpoint IDs differ')
    require(np.array_equal(arrays['phase'],expected_phase),'Checkpoint tasks differ')
    require(np.all(arrays['learning_rate']==config['learning_rate']),'Learning rate differs')
    require(json.loads(row['updates'])==updates.tolist(),'Manifest updates differ')
    require(json.loads(row['checkpoint_ids'])==list(range(len(updates))),'Manifest checkpoint IDs differ')
    require(json.loads(row['rng'])==shuffle_seeds,'Manifest RNG differs')
    require(row['phase_sequence']=='|'.join(windows)+'|C','Manifest curriculum differs')
    require(json.loads(row['phase_lengths'])==[config['window_epochs']]*len(windows)+[config['c_epochs']],'Manifest phase lengths differ')
    require(float(row['learning_rate'])==config['learning_rate'] and float(row['momentum'])==config['momentum'],'Manifest optimizer differs')
    require(row['momentum_reset']=='True' and row['final_task']=='C=(y+3)%10','Manifest final task/reset differs')
    n=len(updates);width=config['hidden_width']
    shapes={'W0':(n,784,width),'W1':(n,width,10),'momentum_W0':(n,784,width),'momentum_W1':(n,width,10),
            'bias0':(n,width),'bias1':(n,10),'accuracy':(n,),'loss':(n,),'learning_rate':(n,)}
    for key,shape in shapes.items():
        require(arrays[key].shape==shape and np.isfinite(arrays[key]).all(),f'Wrong shape/nonfinite {key}')
    for key in ('W0','W1','momentum_W0','momentum_W1','bias0','bias1'):require(arrays[key].dtype==np.float32,f'Wrong precision {key}')
    require(not np.any(arrays['bias0']) and not np.any(arrays['bias1']),'Fixed biases changed')
    require(not np.any(arrays['momentum_W0'][0]) and not np.any(arrays['momentum_W1'][0]),'Nonzero initial momentum')
    require(np.all((arrays['accuracy']>=0)&(arrays['accuracy']<=1)) and np.all(arrays['loss']>=0),'Invalid C metrics')
    init_hash=initial_digest(arrays['W0'][0],arrays['W1'][0])
    require(init_hash==meta['init_hash']==row['init_hash'],'Initial-state hash differs')
    accuracy,loss=float(arrays['accuracy'][-1]),float(arrays['loss'][-1])
    require(accuracy==float(row['final_accuracy']) and loss==float(row['final_loss']),'Manifest C performance differs')
    # Files originate from this controlled local run; full RNG records require weights_only=False.
    state=torch.load(pt_path,map_location='cpu',weights_only=False)
    require(state['update']==int(updates[-1]) and state['epoch']==len(tasks)-1 and state['training_complete'],'Final state position differs')
    require(state['batch_position']==batches and state['next_epoch']==len(tasks),'Final batch position differs')
    require(state['config']==config and state['spec']['seed']==seed and tuple(state['spec']['bits'])==bits,'Final state identity differs')
    for label,key in (('W0','0.weight'),('W1','2.weight')):
        require(np.array_equal(state['model_state_dict'][key].numpy().T,arrays[label][-1]),f'Final model differs: {label}')
    require(not torch.any(state['model_state_dict']['0.bias']) and not torch.any(state['model_state_dict']['2.bias']),'Final state biases nonzero')
    param_ids=state['optimizer_state_dict']['param_groups'][0]['params']
    for pid,label in zip(param_ids,('W0','W1')):
        require(np.array_equal(state['optimizer_state_dict']['state'][pid]['momentum_buffer'].numpy().T,arrays['momentum_'+label][-1]),f'Final momentum differs: {label}')
    require(digest_array(state['epoch_order'].numpy())==order_cache[seed][-1],'Final saved image order differs')
    require(state['details']['order_sha256']==meta['order_sha256'],'Final state RNG metadata differs')
    return {'run_id':run_id,'cohort':cohort,'seed':seed,'arm':arm,'bits':bits,'init_hash':init_hash,
            'final_accuracy':accuracy,'final_loss':loss,'order_sha256':meta['order_sha256'],
            'checkpoints':n,'npz_sha256':npz_hash,'final_state_sha256':pt_hash}

def audit_pairs(records,plan,require_complete):
    out={};all_initial={}
    for cohort in ('one_bit','multi_bit'):
        rows=[r for r in records if r['cohort']==cohort];groups={}
        for r in rows:groups.setdefault(r['seed'],[]).append(r)
        if require_complete:
            expected=set(range(plan['seed_bases'][cohort],plan['seed_bases'][cohort]+plan['config']['blocks']))
            require(set(groups)==expected,f'{cohort}: expected 200 independent seeds')
        initials=[]
        for seed,pair in groups.items():
            require(len(pair)==2 and {r['arm'] for r in pair}=={0,1},f'Incomplete seed pair {seed}')
            a,b=pair
            require(a['init_hash']==b['init_hash'],f'Paired initial weights differ {seed}')
            require(a['order_sha256']==b['order_sha256'],f'Paired image order differs {seed}')
            require(np.all(np.asarray(a['bits'])+b['bits']==1),f'Bits not complementary {seed}')
            initials.append(a['init_hash']);all_initial[seed]=a['init_hash']
        require(len(set(initials))==len(initials),f'Duplicate initialization across seeds in {cohort}')
        out[cohort]={'independent_seeds':len(groups),'runs':len(rows),'checkpoints_per_run':15 if cohort=='one_bit' else 27,
                     'paired_initialization_and_order':True,'complementary_bits':True}
    require(len(set(all_initial.values()))==len(all_initial),'Initial states duplicate across cohorts')
    return out

def audit_splits(records,splits,require_complete):
    output={}
    for cohort in ('one_bit','multi_bit'):
        rows=[r for r in records if r['cohort']==cohort]
        if cohort not in splits:
            require(not require_complete,f'Missing controlled splits for {cohort}');continue
        spec=splits[cohort];groups=spec['splits'];seen=set();details={}
        for group,count in (('train',120),('validation',40),('test',40)):
            seeds=set(map(int,groups[group]['seeds']));run_ids=set(groups[group]['run_ids'])
            require(not seeds&seen,f'Seed leakage in {cohort}/{group}');seen|=seeds
            if require_complete:require(len(seeds)==count,f'Wrong seed count {cohort}/{group}')
            selected=[r for r in rows if r['seed'] in seeds]
            if require_complete:require({r['run_id'] for r in selected}==run_ids,f'Split run IDs differ {cohort}/{group}')
            if not selected:continue
            bits=np.array([r['bits'] for r in selected])
            require(np.all(bits.sum(axis=0)*2==len(selected)),f'Unbalanced bit in {cohort}/{group}')
            for seed in seeds:
                pair=[r for r in selected if r['seed']==seed]
                if require_complete:require(len(pair)==2,f'Incomplete seed in split {seed}')
            details[group]={'independent_seeds':len(seeds),'runs':len(selected),'bit_ones':bits.sum(axis=0).tolist(),
                            'bit_zeros':(len(selected)-bits.sum(axis=0)).tolist()}
        if require_complete:
            require(seen=={r['seed'] for r in rows},f'Splits do not cover complete cohort {cohort}')
            require(spec['reference_run_id'] in groups['train']['run_ids'],f'Alignment reference outside training split {cohort}')
            ref=next(r for r in rows if r['run_id']==spec['reference_run_id'])
            if 'reference_source_sha256' in spec:require(ref['npz_sha256']==spec['reference_source_sha256'],'Reference source hash differs')
        output[cohort]=details
    return output

def performance_controls(records,splits):
    rows=[]
    for ci,cohort in enumerate(('one_bit','multi_bit')):
        cohort_rows=[r for r in records if r['cohort']==cohort]
        if not cohort_rows:continue
        groups={'all':{r['seed'] for r in cohort_rows}}
        if cohort in splits:
            groups.update({key:set(value['seeds']) for key,value in splits[cohort]['splits'].items()})
        for gi,(group,seeds) in enumerate(groups.items()):
            selected=[r for r in cohort_rows if r['seed'] in seeds]
            if not selected:continue
            for bit in range(1 if cohort=='one_bit' else 4):
                for mi,metric in enumerate(('final_accuracy','final_loss')):
                    zero={r['seed']:r[metric] for r in selected if r['bits'][bit]==0}
                    one={r['seed']:r[metric] for r in selected if r['bits'][bit]==1}
                    require(set(zero)==set(one),'Unpaired performance group')
                    d=[one[s]-zero[s] for s in sorted(zero)]
                    mean,low,high=bootstrap_difference(d,seed=2026111608+ci*100+gi*10+bit*2+mi)
                    rows.append({'cohort':cohort,'split':group,'bit_index':bit,'metric':metric,'seed_blocks':len(zero),
                                 'bit0_mean':float(np.mean(list(zero.values()))),'bit1_mean':float(np.mean(list(one.values()))),
                                 'paired_difference_bit1_minus_bit0':mean,'ci_low':low,'ci_high':high,
                                 'bootstrap_replicates':1000,'inference_unit':'whole seed block',
                                 'interpretation':'descriptive difference; no equivalence test or equivalence claim'})
    return rows

def audit_originals(root):
    inventory=json.loads((root/'provenance/source_inventory_before.json').read_text());changed=[]
    for row in inventory['files']:
        path=PROJECT/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes'] or digest_file(path)!=row['sha256']:changed.append(row['path'])
    return {'checked_files':len(inventory['files']),'changed_files':changed,'passed':not changed,'scope':inventory['scope']}

def source_capture(root,smoke):
    inventory=[];folder=root/'provenance'/('smoke_source_snapshot' if smoke else 'validated_source_snapshot')
    folder.mkdir(parents=True,exist_ok=True)
    for path in sorted((root/'code').glob('*.py')):
        inventory.append({'path':str(path.relative_to(root)),'bytes':path.stat().st_size,'sha256':digest_file(path)})
        shutil.copy2(path,folder/path.name)
    return {'python':platform.python_version(),'packages':{p:version(p) for p in ('numpy','scipy','torch','pandas','matplotlib')},
            'code_files':inventory,'snapshot':str(folder.relative_to(root))}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--smoke-pairs',type=int,default=0,help='Audit up to this many complete pairs per available cohort; no complete-cohort claim')
    args=p.parse_args();root=args.root.resolve();smoke=args.smoke_pairs>0;started=time.monotonic()
    torch.set_num_threads(1)
    plan=json.loads((root/'training_plan.json').read_text());sources=json.loads((root/'provenance/training_sources.json').read_text())
    result={'utc':datetime.now(timezone.utc).isoformat(),'mode':'partial_smoke' if smoke else 'complete_audit','passed':False,
            'errors':[],'notes':['Performance differences are descriptive; intervals containing zero do not prove performance equivalence.']}
    try:
        current={'training_source_sha256':digest_file(root/'code/train_cohorts.py'),
                 'shared_training_source_sha256':digest_file(PROJECT/'src/learning_trajectory/training.py'),
                 'shared_data_source_sha256':digest_file(PROJECT/'src/learning_trajectory/data.py'),
                 'dataset_manifest_sha256':digest_file(PROJECT/'data/mnist/manifest.json')}
        require(current==sources,'Frozen training source fingerprints no longer match')
        with (root/'trajectory_run_manifest.csv').open(newline='') as f:manifest=list(csv.DictReader(f))
        require(len({r['run_id'] for r in manifest})==len(manifest),'Duplicate run IDs in manifest')
        require({r['cohort'] for r in manifest}<={'one_bit','multi_bit'},'Unexpected manifest cohort')
        if smoke:
            selected=[]
            for cohort in ('one_bit','multi_bit'):
                rows=[r for r in manifest if r['cohort']==cohort];seeds=sorted({int(r['seed']) for r in rows})[:args.smoke_pairs]
                selected.extend(r for r in rows if int(r['seed']) in seeds)
            manifest=selected
        else:
            require(len(manifest)==800,'Expected exactly 800 training runs')
            for cohort in ('one_bit','multi_bit'):
                expected={r['run_id'] for r in manifest if r['cohort']==cohort}
                folder=root/'checkpoints'/cohort
                require({p.stem for p in folder.glob('*.npz')}==expected,f'NPZ file inventory differs in {cohort}')
                require({p.stem for p in folder.glob('*.pt')}==expected,f'PT file inventory differs in {cohort}')
        records=[];order_cache={}
        for i,row in enumerate(manifest):
            try:records.append(audit_run(row,root,plan,sources,order_cache))
            except Exception as error:raise AssertionError(f'{row["run_id"]}: {error}') from error
            if (i+1)%100==0:print(json.dumps({'audited_runs':i+1,'total':len(manifest)}),flush=True)
        result['runs_audited']=len(records);result['cohort_checks']=audit_pairs(records,plan,not smoke)
        splits=json.loads((root/'trajectory_probe_splits.json').read_text())
        result['split_checks']=audit_splits(records,splits,not smoke)
        result['array_hashes_and_npz_pt_hashes_passed']=True
        result['source_provenance']=source_capture(root,smoke)
        if not smoke:
            controls=performance_controls(records,splits)
            with (root/'final_performance_controls.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(controls[0]));writer.writeheader();writer.writerows(controls)
            result['performance_control_rows']=len(controls)
            result['original_inventory']=audit_originals(root)
            require(result['original_inventory']['passed'],'Original protected files changed')
        result['passed']=True
    except Exception as error:
        result['errors'].append(f'{type(error).__name__}: {error}')
    result['elapsed_seconds']=time.monotonic()-started
    target=root/('provenance/validation_smoke.json' if smoke else 'validation.json')
    target.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'mode':result['mode'],'runs_audited':result.get('runs_audited'),
                      'errors':result['errors'],'path':str(target),'elapsed_seconds':result['elapsed_seconds']},indent=2))
    if not result['passed']:sys.exit(1)

if __name__=='__main__':main()
