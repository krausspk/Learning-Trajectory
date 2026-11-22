"""Deterministic paired-history training. Inputs are read-only; archives atomic."""
from __future__ import annotations
import argparse, copy, csv, hashlib, io, json, math, multiprocessing, os, random, sys, tempfile, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('MKL_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from learning_trajectory.data import IDX_FILES,NORMALIZATION,read_idx,sha256_file,_atomic_npz,_atomic_json,_atomic_bytes
from learning_trajectory.training import build_model,_configure_threads,evaluate
from learning_trajectory.metrics import canonicalize
DEFAULT_OUTPUT=ROOT/'07_History_Decoding'
VERSION='paired-task-history-v1'
HISTORIES=(('A','B','C'),('B','A','C'))
OFFSETS={'A':0,'B':1,'C':3}
DIAGNOSTIC_NAMES=('accuracy','loss','raw_W0_frob','raw_W1_frob','canonical_hidden_rms','logit_rms')

@dataclass(frozen=True)
class Config:
    hidden_width:int=64
    batch_size:int=256
    learning_rate:float=.1
    momentum:float=.9
    pre_epochs:int=1
    c_epochs:int=10
    horizons:tuple[int,...]=(1,5,10)
    test_batch_size:int=2048
    threads:int=1

def array_hash(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def weights_hash(w0,w1): return hashlib.sha256(np.ascontiguousarray(w0).tobytes()+np.ascontiguousarray(w1).tobytes()).hexdigest()
def json_hash(value): return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def order_seed(seed,epoch): return 900000+seed*32+epoch
def epoch_order(seed,epoch,n): return torch.randperm(n,generator=torch.Generator().manual_seed(order_seed(seed,epoch)))

def read_mnist(data_dir=ROOT/'data'):
    folder=Path(data_dir)/'mnist'
    manifest=json.loads((folder/'manifest.json').read_text())
    records={r['filename']:r for r in manifest['files']}
    tensors,hashes=[],{}
    mean,std=NORMALIZATION['mnist']
    for i,(name,shape) in enumerate(IDX_FILES):
        path=folder/name
        digest=sha256_file(path)
        if records[name]['sha256']!=digest: raise ValueError(f'MNIST hash mismatch: {path}')
        hashes[name]=digest
        a=read_idx(path,shape)
        if i%2==0:
            t=torch.from_numpy(a.copy()).reshape(len(a),784).to(torch.float32)
            t.div_(255.).sub_(mean).div_(std)
        else: t=torch.from_numpy(a.astype(np.int64))
        tensors.append(t)
    return tuple(tensors),hashes

@lru_cache(maxsize=1)
def cached_mnist():
    _configure_threads(1)
    return read_mnist()

def optimizer_for(model,cfg):
    return torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=cfg.learning_rate,momentum=cfg.momentum)
def momentum_norm(opt): return sum(float(v.get('momentum_buffer',torch.zeros(1)).norm()) for v in opt.state.values())

def train_pair(seed,arrays,probe_indices,mode='reset',cfg=Config(),collect_states=True):
    if mode not in ('reset','continued'): raise ValueError('Invalid momentum mode')
    if not cfg.horizons or max(cfg.horizons)!=cfg.c_epochs or min(cfg.horizons)<1: raise ValueError('Invalid horizons')
    if 2*cfg.pre_epochs+cfg.c_epochs>32: raise ValueError('Too many epochs for shuffle seed spacing')
    _configure_threads(cfg.threads)
    random.seed(seed); np.random.seed(seed)
    xtr,ytr,xte,yte=arrays
    if xtr.dtype!=torch.float32 or xte.dtype!=torch.float32: raise ValueError('Images must use float32')
    probe_indices=np.asarray(probe_indices,dtype=np.int64)
    if len(np.unique(probe_indices))!=len(probe_indices) or np.any(probe_indices<0) or np.any(probe_indices>=len(xte)): raise ValueError('Invalid probe indices')
    xp=xte[torch.from_numpy(probe_indices)]
    initial=copy.deepcopy(build_model(seed,10,cfg.hidden_width).state_dict())
    iw0=initial['0.weight'].numpy().T.copy(); iw1=initial['2.weight'].numpy().T.copy()
    initial_hash=weights_hash(iw0,iw1)
    total_epochs=2*cfg.pre_epochs+cfg.c_epochs
    orders=[epoch_order(seed,e,len(xtr)) for e in range(total_epochs)]
    order_hashes=[array_hash(o.numpy()) for o in orders]
    keys=('W0','W1','accuracy','loss','diagnostics','logits','bias0','bias1')
    collected={k:[] for k in keys}
    transitions,epochs,states,init_hashes=[],[],[],[]
    for h,history in enumerate(HISTORIES):
        model=build_model(seed,10,cfg.hidden_width); model.load_state_dict(initial)
        init_hashes.append(weights_hash(model[0].weight.detach().numpy().T,model[2].weight.detach().numpy().T))
        if init_hashes[-1]!=initial_hash: raise AssertionError('Pair initialization mismatch')
        opt=optimizer_for(model,cfg); arm={k:[] for k in keys}; arm_states=[]; global_epoch=0; updates=0
        for phase,task in enumerate(history):
            before=momentum_norm(opt)
            if phase and mode=='reset': opt=optimizer_for(model,cfg)
            transitions.append({'history':h,'phase':phase,'task':task,'momentum_norm_before':before,
                'momentum_norm_start':momentum_norm(opt),'reset':bool(phase and mode=='reset')})
            labels=(ytr+OFFSETS[task]).remainder(10)
            for phase_epoch in range(1,(cfg.c_epochs if task=='C' else cfg.pre_epochs)+1):
                order=orders[global_epoch]; epoch_loss=0.
                for start in range(0,len(order),cfg.batch_size):
                    ix=order[start:start+cfg.batch_size]; opt.zero_grad(set_to_none=True)
                    loss=F.cross_entropy(model(xtr[ix]),labels[ix])
                    if not torch.isfinite(loss): raise FloatingPointError(f'Nonfinite loss seed={seed} h={h}')
                    loss.backward(); opt.step(); epoch_loss+=float(loss.detach())*len(ix); updates+=1
                if torch.count_nonzero(model[0].bias) or torch.count_nonzero(model[2].bias): raise AssertionError('Bias changed')
                epochs.append({'history':h,'global_epoch':global_epoch,'task':task,'phase_epoch':phase_epoch,
                    'updates':updates,'train_loss':epoch_loss/len(xtr),'order_seed':order_seed(seed,global_epoch),'order_sha256':order_hashes[global_epoch]})
                if task=='C' and phase_epoch in cfg.horizons:
                    acc,tloss=evaluate(model,xte,(yte+3).remainder(10),cfg.test_batch_size)
                    w0=model[0].weight.detach().numpy().T.copy(); w1=model[2].weight.detach().numpy().T.copy()
                    c0,_=canonicalize(w0,w1)
                    with torch.inference_mode(): logits=model(xp).numpy().copy()
                    hidden=np.maximum(xp.numpy().astype(np.float64)@c0,0.)
                    diagnostics=[acc,tloss,np.linalg.norm(w0.astype(np.float64)),np.linalg.norm(w1.astype(np.float64)),
                        np.sqrt(np.mean(hidden**2)),np.sqrt(np.mean(logits.astype(np.float64)**2))]
                    values={'W0':w0,'W1':w1,'accuracy':acc,'loss':tloss,'diagnostics':diagnostics,'logits':logits,
                        'bias0':model[0].bias.detach().numpy().copy(),'bias1':model[2].bias.detach().numpy().copy()}
                    for k in keys: arm[k].append(values[k])
                    if collect_states:
                        arm_states.append({'history':h,'c_epoch':phase_epoch,'global_epoch_completed':global_epoch,'updates':updates,
                            'model_state_dict':copy.deepcopy(model.state_dict()),'optimizer_state_dict':copy.deepcopy(opt.state_dict()),
                            'torch_rng_state':torch.get_rng_state().clone(),'numpy_rng_state':copy.deepcopy(np.random.get_state()),
                            'python_rng_state':random.getstate(),'last_epoch_order':order.clone(),'next_order_seed':order_seed(seed,global_epoch+1)})
                global_epoch+=1
        for k in keys: collected[k].append(np.asarray(arm[k]))
        states.append(arm_states)
    result={k:np.asarray(v) for k,v in collected.items()}
    result.update(seed=np.asarray(seed),horizons=np.asarray(cfg.horizons),initial_W0=iw0,initial_W1=iw1,
        probe_indices=probe_indices,diagnostic_names=np.asarray(DIAGNOSTIC_NAMES))
    metadata={'seed':seed,'mode':mode,'config':asdict(cfg),'version':VERSION,'histories':HISTORIES,'label_offsets':OFFSETS,
        'initialization_sha256':initial_hash,'paired_initialization_sha256':init_hashes,'paired_initialization_equal':len(set(init_hashes))==1,
        'order_seeds':[order_seed(seed,e) for e in range(total_epochs)],'order_sha256':order_hashes,'identical_orders_both_histories':True,
        'transition_audit':transitions,'epoch_training':epochs,'updates_per_epoch':math.ceil(len(xtr)/cfg.batch_size),
        'sample_counts':{'train':len(xtr),'test':len(xte),'probe':len(probe_indices)},'precision':'CPU float32 training; canonical diagnostics float64',
        'initialization':'PyTorch default 784-64-10; zero/frozen biases; cloned full state',
        'shuffle':'torch.randperm; seed=900000+initialization_seed*32+global_epoch'}
    return result,metadata,states

def atomic_pt(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.stem+'.',suffix='.pt',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f: torch.save(obj,f)
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def context(output,cfg,data_hashes):
    output=Path(output)
    paths={'training_source':Path(__file__),'original_training_source':ROOT/'src/learning_trajectory/training.py',
        'original_data_source':ROOT/'src/learning_trajectory/data.py','original_metrics_source':ROOT/'src/learning_trajectory/metrics.py',
        'plan':output/'plan.json','probe_inputs':output/'probe_inputs.npz'}
    plan=json.loads(paths['plan'].read_text())
    expected=plan.get('config',{})
    for name,value in [('batch_size',cfg.batch_size),('learning_rate',cfg.learning_rate),('momentum',cfg.momentum),('ab_epochs',cfg.pre_epochs),('c_epochs',cfg.c_epochs),('horizons',list(cfg.horizons))]:
        if name in expected and expected[name]!=value: raise ValueError(f'Plan/config mismatch: {name}')
    value={'version':VERSION,'config':asdict(cfg),'source_hashes':{k:sha256_file(p) for k,p in paths.items()},
        'dataset_sha256':data_hashes,'torch_version':torch.__version__}
    value['fingerprint']=json_hash(value)
    return value

def validate_archive(path,expected):
    with np.load(path,allow_pickle=False) as a:
        m=json.loads(str(a['metadata']))
        if m['context']['fingerprint']!=expected['fingerprint']: raise ValueError(f'Archive fingerprint mismatch; refusing overwrite: {path}')
        n=len(expected['config']['horizons']); w=expected['config']['hidden_width']
        for k,shape in [('W0',(2,n,784,w)),('W1',(2,n,w,10)),('accuracy',(2,n)),('loss',(2,n)),('diagnostics',(2,n,6)),('logits',(2,n,256,10))]:
            if a[k].shape!=shape or not np.isfinite(a[k]).all(): raise ValueError(f'Invalid {k}: {path}')
        if not m['paired_initialization_equal'] or np.any(a['bias0']) or np.any(a['bias1']): raise ValueError('Pair/bias validation failed')
        pt=Path(path).with_suffix('.pt')
        if not pt.exists() or sha256_file(pt)!=m['full_state_sha256']: raise ValueError(f'Invalid full-state checkpoint: {pt}')
    return m

def train_seed(seed,mode,output,cfg=Config()):
    output=Path(output); arrays,hashes=cached_mnist(); ctx=context(output,cfg,hashes)
    path=output/'checkpoints'/mode/f'seed_{seed}.npz'
    if path.exists():
        m=validate_archive(path,ctx)
        return {'seed':seed,'mode':mode,'cached':True,'seconds':m['training_seconds']}
    with np.load(output/'probe_inputs.npz',allow_pickle=False) as p: indices=p['indices'].copy()
    start=time.monotonic(); result,m,states=train_pair(seed,arrays,indices,mode,cfg)
    m.update(context=ctx,training_seconds=time.monotonic()-start)
    atomic_pt(path.with_suffix('.pt'),{'metadata':m,'initial_W0':result['initial_W0'],'initial_W1':result['initial_W1'],'checkpoints':states})
    m['full_state_sha256']=sha256_file(path.with_suffix('.pt'))
    _atomic_npz(path,**result,metadata=np.asarray(json.dumps(m,sort_keys=True)))
    validate_archive(path,ctx)
    return {'seed':seed,'mode':mode,'cached':False,'seconds':m['training_seconds']}

def build_manifest(output):
    output=Path(output); rows=[]
    for path in sorted((output/'checkpoints').glob('*/seed_*.npz')):
        with np.load(path,allow_pickle=False) as a:
            m=json.loads(str(a['metadata'])); digest=sha256_file(path)
            for h in range(2):
                for j,c in enumerate(a['horizons'].tolist()):
                    rows.append({'seed':int(a['seed']),'history':h,'task_order':'→'.join(HISTORIES[h]),'momentum_mode':m['mode'],
                        'momentum_reset':m['mode']=='reset','C_horizon':c,'initialization_sha256':m['initialization_sha256'],
                        'label_permutations':json.dumps({k:[(y+v)%10 for y in range(10)] for k,v in OFFSETS.items()}),
                        'phase_epochs':json.dumps([m['config']['pre_epochs'],m['config']['pre_epochs'],c]),'updates_per_epoch':m['updates_per_epoch'],
                        'learning_rate':m['config']['learning_rate'],'momentum':m['config']['momentum'],'batch_size':m['config']['batch_size'],
                        'rng':m['shuffle'],'final_checkpoint':str(path.relative_to(output)),'checkpoint_index':j,'checkpoint_sha256':digest,
                        'full_state_sha256':m['full_state_sha256'],'accuracy':float(a['accuracy'][h,j]),'loss':float(a['loss'][h,j]),
                        'context_fingerprint':m['context']['fingerprint']})
    if rows:
        s=io.StringIO(); writer=csv.DictWriter(s,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        _atomic_bytes(output/'history_run_manifest.csv',s.getvalue().encode())
    return len(rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); p.add_argument('--workers',type=int,default=6)
    p.add_argument('--mode',choices=('reset','continued','all'),default='all')
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(61001,61201))); p.add_argument('--manifest-only',action='store_true')
    args=p.parse_args()
    if args.manifest_only: print(json.dumps({'manifest_rows':build_manifest(args.output)}),flush=True); return
    modes=('reset','continued') if args.mode=='all' else (args.mode,)
    jobs=[(s,m) for m in modes for s in args.seeds]; start=time.monotonic()
    if args.workers==1:
        for i,(s,m) in enumerate(jobs,1):
            record=train_seed(s,m,args.output); print(json.dumps({'completed':i,'total':len(jobs),**record}),flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=multiprocessing.get_context('spawn')) as pool:
            futures={pool.submit(train_seed,s,m,args.output):(s,m) for s,m in jobs}
            for i,f in enumerate(as_completed(futures),1): print(json.dumps({'completed':i,'total':len(jobs),**f.result()}),flush=True)
    receipt={'completed':True,'seed_blocks_per_mode':len(args.seeds),'modes':modes,'manifest_rows':build_manifest(args.output),'wall_seconds':time.monotonic()-start}
    _atomic_json(args.output/'provenance'/f'training_{args.mode}_receipt.json',receipt); print(json.dumps(receipt),flush=True)
if __name__=='__main__': main()
