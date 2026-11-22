#!/usr/bin/env python3
"""Controlled AB/BA training; every result is isolated from reproduction outputs."""
from __future__ import annotations
import argparse
import copy
import csv
from dataclasses import asdict, dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import hashlib
import io
import json
import math
import multiprocessing
import os
from pathlib import Path
import platform
import random
import sys
import tempfile
import time
import numpy as np
import torch
from torch.nn import functional as F

PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT/'src'))
from learning_trajectory.data import NORMALIZATION, IDX_FILES, read_idx, sha256_file, _atomic_json, _atomic_npz, _atomic_bytes
from learning_trajectory.training import build_model, evaluate, _configure_threads
ROOT=Path(__file__).resolve().parents[1]
VERSION='trajectory-decoding-training-v1'

@dataclass(frozen=True)
class Config:
    blocks:int=200
    hidden_width:int=64
    batch_size:int=256
    learning_rate:float=.1
    momentum:float=.9
    window_epochs:int=1
    c_epochs:int=5
    test_batch_size:int=2048
    threads:int=1

@dataclass(frozen=True)
class Spec:
    cohort:str
    seed:int
    arm:int
    bits:tuple[int,...]
    @property
    def run_id(self):return f'{self.cohort}_s{self.seed}_h{self.arm}'

def specs(cohort,config=Config()):
    result=[]
    for name in (('one_bit','multi_bit') if cohort=='all' else (cohort,)):
        if name not in ('one_bit','multi_bit'):raise ValueError(name)
        base=10000 if name=='one_bit' else 20000
        for i in range(config.blocks):
            seed=base+i
            bits=(0,) if name=='one_bit' else tuple(int(x) for x in np.random.default_rng(810000+seed).integers(0,2,size=4))
            result.extend(Spec(name,seed,arm,tuple(b^arm for b in bits)) for arm in (0,1))
    return result

def schedule(spec,config):
    windows=[task for bit in spec.bits for task in (('A','B') if bit==0 else ('B','A'))]
    return windows,[task for task in windows for _ in range(config.window_epochs)]+['C']*config.c_epochs

def checkpoint_updates(spec,config,n_train):
    batches=math.ceil(n_train/config.batch_size)
    if batches<2:raise ValueError('At least two batches per epoch required')
    return np.asarray([0]+[e*batches+u for e in range(len(schedule(spec,config)[1])) for u in (math.ceil(batches/2),batches)],dtype=np.int64)

def plan(config):
    return {'version':VERSION,'config':asdict(config),'dataset':'mnist',
            'sample_sizes':{'training':60000,'test':10000},'architecture':[784,config.hidden_width,10],
            'label_offsets':{'A':0,'B':1,'C':3},'seed_bases':{'one_bit':10000,'multi_bit':20000},
            'multi_bit_rng':'numpy.default_rng(810000 + init_seed).integers(0,2,size=4); arm 1 complements arm 0',
            'shuffle_rng':'torch.Generator().manual_seed(5000000 + init_seed * 100 + zero_based_epoch)',
            'paired_controls':'same initialization and global-epoch image order in both arms',
            'momentum':'reset at each A/B window boundary and start of C; preserve within C',
            'checkpoint':'0, ceil(batches/2) and end of every epoch; accuracy/loss always evaluated on C',
            'biases':'zero and frozen','training_dtype':'float32','cpu_deterministic':True,'early_stopping':False,
            'checkpoint_momentum_semantics':'stored after update, before any following phase reset',
            'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__}

def source_fingerprints(data_dir):
    return {'training_source_sha256':sha256_file(__file__),
            'shared_training_source_sha256':sha256_file(PROJECT/'src/learning_trajectory/training.py'),
            'shared_data_source_sha256':sha256_file(PROJECT/'src/learning_trajectory/data.py'),
            'dataset_manifest_sha256':sha256_file(Path(data_dir)/'mnist/manifest.json')}

def freeze_plan(output,config):
    path=Path(output)/'training_plan.json';value=plan(config)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError(f'Frozen training plan differs: {path}; use a separate output directory')
    else:_atomic_json(path,value)

def verify_data(data_dir):
    folder=Path(data_dir)/'mnist';manifest=json.loads((folder/'manifest.json').read_text())
    for filename,shape in IDX_FILES:
        row=next(x for x in manifest['files'] if x['filename']==filename)
        if sha256_file(folder/filename)!=row['sha256']:raise ValueError(f'Dataset checksum mismatch: {filename}')
        read_idx(folder/filename,shape)

@lru_cache(maxsize=1)
def load_readonly(data_dir):
    folder=Path(data_dir)/'mnist';mean,std=NORMALIZATION['mnist'];arrays=[]
    for filename,shape in IDX_FILES:
        array=read_idx(folder/filename,shape)
        if len(shape)==3:array=(array.reshape(len(array),-1).astype(np.float32)/255.0-mean)/std
        else:array=array.astype(np.int64)
        arrays.append(torch.from_numpy(array.copy()))
    return tuple(arrays)

def hash_initial(w0,w1):
    h=hashlib.sha256()
    for w in (w0,w1):h.update(np.ascontiguousarray(w).tobytes())
    return h.hexdigest()

def train_arrays(spec,arrays,config=Config()):
    """All stochastic choices are tied to the seed, never the history arm."""
    _configure_threads(config.threads);random.seed(spec.seed);np.random.seed(spec.seed)
    x_train,y_train,x_test,y_test=arrays
    model=build_model(spec.seed,10,config.hidden_width)
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=config.learning_rate,momentum=config.momentum)
    windows,tasks=schedule(spec,config);updates=checkpoint_updates(spec,config,len(x_train));targets=set(updates.tolist())
    values={k:[] for k in ('W0','W1','momentum_W0','momentum_W1','accuracy','loss','phase','learning_rate')}
    update=0;epoch=-1;batch_position=0;reset_updates=[];shuffle_seeds=[];order_hashes=[]
    def save(task):
        accuracy,loss=evaluate(model,x_test,(y_test+3)%10,config.test_batch_size)
        for key,layer in (('W0',model[0]),('W1',model[2])):
            values[key].append(layer.weight.detach().numpy().T.copy())
            buffer=optimizer.state.get(layer.weight,{}).get('momentum_buffer',torch.zeros_like(layer.weight))
            values['momentum_'+key].append(buffer.detach().numpy().T.copy())
        values['accuracy'].append(accuracy);values['loss'].append(loss)
        values['phase'].append(task);values['learning_rate'].append(config.learning_rate)
        if torch.count_nonzero(model[0].bias) or torch.count_nonzero(model[2].bias):raise AssertionError('Frozen biases changed')
    save('initial')
    for epoch,task in enumerate(tasks):
        if epoch%config.window_epochs==0 and epoch<=len(windows)*config.window_epochs:
            optimizer.state.clear();reset_updates.append(update)
        shuffle_seed=5000000+spec.seed*100+epoch
        generator=torch.Generator().manual_seed(shuffle_seed);order=torch.randperm(len(x_train),generator=generator)
        shuffle_seeds.append(shuffle_seed);order_hashes.append(hashlib.sha256(order.numpy().tobytes()).hexdigest())
        labels=(y_train+{'A':0,'B':1,'C':3}[task])%10;batch_position=0
        for start in range(0,len(x_train),config.batch_size):
            selection=order[start:start+config.batch_size];optimizer.zero_grad(set_to_none=True)
            loss=F.cross_entropy(model(x_train[selection]),labels[selection])
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite training loss {spec.run_id} update {update}')
            loss.backward();optimizer.step();update+=1;batch_position+=1
            if update in targets:save(task)
    if len(values['W0'])!=len(updates):raise AssertionError('Incomplete trajectory')
    result={key:np.asarray(value) for key,value in values.items()};result['checkpoints']=np.arange(len(updates),dtype=np.int64);result['updates']=updates
    result['bias0']=np.zeros((len(updates),config.hidden_width),dtype=np.float32);result['bias1']=np.zeros((len(updates),10),dtype=np.float32)
    details={'windows':windows,'epoch_tasks':tasks,'shuffle_seeds':shuffle_seeds,'order_sha256':order_hashes,
             'momentum_reset_updates':reset_updates,'init_hash':hash_initial(result['W0'][0],result['W1'][0]),
             'train_samples':len(x_train),'test_samples':len(x_test)}
    state={'model_state_dict':copy.deepcopy(model.state_dict()),'optimizer_state_dict':copy.deepcopy(optimizer.state_dict()),
           'torch_rng_state':torch.get_rng_state().clone(),'numpy_rng_state':np.random.get_state(),'python_rng_state':random.getstate(),
           'epoch':epoch,'batch_position':batch_position,'update':update,'epoch_order':order.clone(),
           'order_generator_state':generator.get_state().clone(),'spec':asdict(spec),'config':asdict(config),
           'details':details,'training_complete':True,'next_epoch':epoch+1}
    return result,details,state

def _atomic_torch(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:torch.save(value,f)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)

def expected_config(spec,config,provenance):
    return hashlib.sha256(json.dumps({'spec':asdict(spec),'config':asdict(config),'provenance':provenance,'version':VERSION},sort_keys=True).encode()).hexdigest()

def validate_run(path,spec,config,provenance):
    path=Path(path)
    with np.load(path,allow_pickle=False) as a:
        m=json.loads(str(a['metadata']))
        if m['config_fingerprint']!=expected_config(spec,config,provenance):raise ValueError(f'Run source/config mismatch: {path}')
        if not np.array_equal(a['updates'],checkpoint_updates(spec,config,60000)):raise ValueError(f'Bad update schedule: {path}')
        for key,digest in m['array_sha256'].items():
            if hashlib.sha256(np.ascontiguousarray(a[key]).tobytes()).hexdigest()!=digest:
                raise ValueError(f'Checkpoint payload checksum mismatch: {key} in {path}')
        n=len(a['updates'])
        for key,shape in (('W0',(n,784,config.hidden_width)),('W1',(n,config.hidden_width,10)),('momentum_W0',(n,784,config.hidden_width)),('momentum_W1',(n,config.hidden_width,10)),('accuracy',(n,)),('loss',(n,))):
            if a[key].shape!=shape or not np.isfinite(a[key]).all():raise ValueError(f'Invalid {key}: {path}')
        if np.any(a['bias0']) or np.any(a['bias1']):raise ValueError('Nonzero biases')
        if hash_initial(a['W0'][0],a['W1'][0])!=m['init_hash']:raise ValueError('Initial hash mismatch')
        if not np.array_equal(a['phase'],['initial']+[t for t in m['epoch_tasks'] for _ in range(2)]):raise ValueError('Phase schedule mismatch')
        accuracy,loss=float(a['accuracy'][-1]),float(a['loss'][-1])
    state_path=path.with_suffix('.pt')
    if sha256_file(state_path)!=m['final_state_sha256']:raise ValueError('Final-state checksum mismatch')
    return {'cohort':spec.cohort,'run_id':spec.run_id,'seed':spec.seed,'arm':spec.arm,'bits':''.join(map(str,spec.bits)),
            'phase_sequence':'|'.join(m['windows'])+'|C','phase_lengths':json.dumps([config.window_epochs]*len(m['windows'])+[config.c_epochs]),
            'init_hash':m['init_hash'],'final_accuracy':accuracy,'final_loss':loss,'path':str(path.resolve()),
            'npz_sha256':sha256_file(path),'final_state_path':str(state_path.resolve()),'final_state_sha256':m['final_state_sha256'],
            'learning_rate':config.learning_rate,'momentum':config.momentum,'momentum_reset':True,'final_task':'C=(y+3)%10',
            'rng':json.dumps(m['shuffle_seeds']),'checkpoint_ids':json.dumps(list(range(len(m['updates'])))),
            'updates':json.dumps(m['updates']),'elapsed_seconds':m['elapsed_seconds']}

def train_run(spec,data_dir,output,config,provenance):
    path=Path(output)/'checkpoints'/spec.cohort/(spec.run_id+'.npz')
    if path.exists():return validate_run(path,spec,config,provenance)
    started=time.monotonic();arrays=load_readonly(str(Path(data_dir).resolve()))
    trajectory,details,state=train_arrays(spec,arrays,config);state_path=path.with_suffix('.pt');_atomic_torch(state_path,state)
    metadata={**asdict(spec),'run_id':spec.run_id,'config':asdict(config),'version':VERSION,**provenance,**details,
              'updates':trajectory['updates'].tolist(),'config_fingerprint':expected_config(spec,config,provenance),
              'normalization':NORMALIZATION['mnist'],'final_state_sha256':sha256_file(state_path),'elapsed_seconds':time.monotonic()-started}
    metadata['array_sha256']={key:hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest() for key,value in trajectory.items()}
    trajectory['metadata']=np.asarray(json.dumps(metadata,sort_keys=True));_atomic_npz(path,**trajectory)
    return validate_run(path,spec,config,provenance)

def pair_worker(arguments):
    pair,data_dir,output,config,provenance=arguments;rows=[train_run(spec,data_dir,output,config,provenance) for spec in pair]
    if rows[0]['init_hash']!=rows[1]['init_hash']:raise AssertionError('Paired initialization differs')
    return rows

def write_manifest(output,records):
    path=Path(output)/'trajectory_run_manifest.csv';known={}
    if path.exists():
        with path.open(newline='') as f:known={r['run_id']:r for r in csv.DictReader(f)}
    known.update({r['run_id']:r for r in records});rows=sorted(known.values(),key=lambda r:r['run_id'])
    text=io.StringIO();writer=csv.DictWriter(text,fieldnames=list(records[0]));writer.writeheader();writer.writerows(rows)
    _atomic_bytes(path,text.getvalue().encode())

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cohort',choices=('one_bit','multi_bit','all'),default='all')
    p.add_argument('--workers',type=int,default=6);p.add_argument('--limit-blocks',type=int)
    p.add_argument('--output',type=Path,default=ROOT);p.add_argument('--data-dir',type=Path,default=PROJECT/'data')
    args=p.parse_args();config=Config();args.output.mkdir(parents=True,exist_ok=True)
    freeze_plan(args.output,config);verify_data(args.data_dir);provenance=source_fingerprints(args.data_dir)
    _atomic_json(args.output/'provenance/training_sources.json',provenance);candidates=specs(args.cohort,config)
    if args.limit_blocks:
        if args.limit_blocks<1:raise ValueError('limit-blocks must be positive')
        bases={'one_bit':10000,'multi_bit':20000};candidates=[s for s in candidates if s.seed-bases[s.cohort]<args.limit_blocks]
    pairs=[candidates[i:i+2] for i in range(0,len(candidates),2)]
    arguments=[(pair,str(args.data_dir),str(args.output),config,provenance) for pair in pairs];complete=0;started=time.monotonic()
    def consume(rows):
        nonlocal complete
        write_manifest(args.output,rows);complete+=len(rows)
        print(json.dumps({'completed_runs':complete,'requested_runs':len(candidates),'last_seed':rows[0]['seed'],
                          'cohort':rows[0]['cohort'],'final_accuracy':[r['final_accuracy'] for r in rows],
                          'elapsed_seconds':round(time.monotonic()-started,2)}),flush=True)
    if args.workers==1:
        for a in arguments:consume(pair_worker(a))
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures=[pool.submit(pair_worker,a) for a in arguments]
            for future in as_completed(futures):consume(future.result())
    _atomic_json(args.output/'provenance'/f'training_completion_{args.cohort}.json',
                 {'completed_runs':complete,'expected_runs':len(specs(args.cohort,config)),
                  'complete':complete==len(specs(args.cohort,config)),'elapsed_seconds':time.monotonic()-started,'provenance':provenance})

if __name__=='__main__':main()
