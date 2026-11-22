import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
spec=importlib.util.spec_from_file_location('history_training',Path(__file__).resolve().parents[1]/'code'/'training.py')
training=importlib.util.module_from_spec(spec); sys.modules[spec.name]=training; spec.loader.exec_module(training)

def arrays():
    x=torch.randn(40,784,generator=torch.Generator().manual_seed(81)); y=torch.arange(40).remainder(10)
    return x,y,x[:12].clone(),y[:12].clone()
def small(mode='reset'):
    cfg=training.Config(hidden_width=8,batch_size=16,c_epochs=2,horizons=(1,2),test_batch_size=6)
    return training.train_pair(61001,arrays(),np.arange(8),mode,cfg)

def test_paired_initialization_order_and_reset():
    d,m,states=small()
    assert m['paired_initialization_equal'] and len(set(m['paired_initialization_sha256']))==1
    assert d['W0'].shape==(2,2,784,8) and d['W1'].shape==(2,2,8,10) and d['logits'].shape==(2,2,8,10)
    for t in m['transition_audit']:
        assert t['momentum_norm_start']==0
        if t['phase']: assert t['momentum_norm_before']>0 and t['reset']
    es=m['epoch_training']
    assert [e['order_sha256'] for e in es[:4]]==[e['order_sha256'] for e in es[4:]]
    assert [e['task'] for e in es[:4]]==['A','B','C','C'] and [e['task'] for e in es[4:]]==['B','A','C','C']
    assert not d['bias0'].any() and not d['bias1'].any()
    assert all(s['updates'] in (9,12) for arm in states for s in arm)

def test_determinism_and_distinct_history():
    a,ma,_=small(); b,mb,_=small()
    for k in a: assert np.array_equal(a[k],b[k]),k
    assert ma==mb
    assert not np.array_equal(a['W0'][0],a['W0'][1])

def test_continued_momentum_and_pairing():
    a,ma,_=small(); b,mb,_=small('continued')
    assert ma['initialization_sha256']==mb['initialization_sha256'] and ma['order_sha256']==mb['order_sha256']
    for t in mb['transition_audit']:
        assert not t['reset'] and t['momentum_norm_start']==t['momentum_norm_before']
        if t['phase']: assert t['momentum_norm_start']>0
    assert not np.array_equal(a['W0'],b['W0'])

def test_diagnostics_and_checkpoint_match_final_weights():
    d,m,states=small(); _,_,xt,yt=arrays()
    for h in range(2):
        for j in range(2):
            s=states[h][j]
            assert np.array_equal(s['model_state_dict']['0.weight'].numpy().T,d['W0'][h,j])
            assert np.array_equal(s['model_state_dict']['2.weight'].numpy().T,d['W1'][h,j])
            logits=torch.relu(xt@torch.from_numpy(d['W0'][h,j]))@torch.from_numpy(d['W1'][h,j])
            assert np.allclose(logits[:8].numpy(),d['logits'][h,j],atol=2e-6)
            acc=float((logits.argmax(1)==(yt+3).remainder(10)).float().mean())
            assert abs(acc-d['accuracy'][h,j])<1e-7
            assert np.allclose(d['diagnostics'][h,j,:2],[d['accuracy'][h,j],d['loss'][h,j]])

def test_orders_distinct_and_inputs_unchanged():
    a=arrays(); old=[x.clone() for x in a]
    cfg=training.Config(hidden_width=8,batch_size=16,c_epochs=1,horizons=(1,))
    training.train_pair(61001,a,np.arange(8),'reset',cfg)
    for x,y in zip(a,old): assert torch.equal(x,y)
    hashes={training.array_hash(training.epoch_order(s,e,100).numpy()) for s in (61001,61002) for e in range(12)}
    assert len(hashes)==24

def test_checkpoint_resumes_bitwise_to_next_c_horizon():
    """Optimizer buffers and the specified next order reconstruct the trajectory."""
    import copy
    from torch.nn import functional as F
    d,m,states=small()
    x,y,_,_=arrays()
    cfg=training.Config(hidden_width=8,batch_size=16,c_epochs=2,horizons=(1,2),test_batch_size=6)
    for h in range(2):
        s=states[h][0]
        model=training.build_model(61001,10,8)
        model.load_state_dict(s['model_state_dict'])
        opt=training.optimizer_for(model,cfg)
        opt.load_state_dict(copy.deepcopy(s['optimizer_state_dict']))
        order=training.epoch_order(61001,s['global_epoch_completed']+1,len(x))
        assert s['next_order_seed']==training.order_seed(61001,s['global_epoch_completed']+1)
        for start in range(0,len(x),cfg.batch_size):
            ix=order[start:start+cfg.batch_size]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(model(x[ix]),(y[ix]+3).remainder(10)).backward()
            opt.step()
        assert np.array_equal(model[0].weight.detach().numpy().T,d['W0'][h,1])
        assert np.array_equal(model[2].weight.detach().numpy().T,d['W1'][h,1])
