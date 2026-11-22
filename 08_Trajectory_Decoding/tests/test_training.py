import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
path=Path(__file__).resolve().parents[1]/'code/train_cohorts.py'
spec=importlib.util.spec_from_file_location('controlled_train',path)
m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)

def tiny_arrays():
    g=torch.Generator().manual_seed(17)
    return torch.randn(24,784,generator=g),torch.arange(24)%10,torch.randn(12,784,generator=g),torch.arange(12)%10

def test_balance_and_disjoint_seeds():
    runs=m.specs('all');seeds={name:{s.seed for s in runs if s.cohort==name} for name in ('one_bit','multi_bit')}
    assert not seeds['one_bit']&seeds['multi_bit']
    assert len(seeds['one_bit'])==len(seeds['multi_bit'])==200
    for a,b in zip(runs[::2],runs[1::2]):
        assert a.seed==b.seed and np.all(np.array(a.bits)+b.bits==1)
        for run in (a,b):
            tasks=m.schedule(run,m.Config())[0]
            assert tasks.count('A')==tasks.count('B')==len(run.bits)
            assert all(set(tasks[i:i+2])=={'A','B'} for i in range(0,len(tasks),2))

def test_paired_initialization_orders_and_momentum_reset():
    c=m.Config(hidden_width=6,batch_size=8,c_epochs=2,test_batch_size=12)
    a=m.Spec('one_bit',10000,0,(0,));b=m.Spec('one_bit',10000,1,(1,))
    ta,da,sa=m.train_arrays(a,tiny_arrays(),c);tb,db,sb=m.train_arrays(b,tiny_arrays(),c)
    assert da['init_hash']==db['init_hash'] and da['order_sha256']==db['order_sha256']
    assert np.array_equal(ta['W0'][0],tb['W0'][0]) and np.array_equal(ta['W1'][0],tb['W1'][0])
    assert da['momentum_reset_updates']==[0,3,6]
    assert np.array_equal(ta['updates'],[0,2,3,5,6,8,9,11,12])
    assert not np.array_equal(ta['W0'][-1],tb['W0'][-1])
    assert np.count_nonzero(ta['momentum_W0'][0])==0
    assert not np.any(ta['bias0']) and not np.any(ta['bias1'])
    # Replay the first two C updates from boundary weights with reset momentum.
    model=m.build_model(a.seed,10,c.hidden_width)
    with torch.no_grad():
        model[0].weight.copy_(torch.from_numpy(ta['W0'][4].T));model[2].weight.copy_(torch.from_numpy(ta['W1'][4].T))
    optim=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=.1,momentum=.9)
    x,y,_,_=tiny_arrays();order=torch.randperm(len(x),generator=torch.Generator().manual_seed(da['shuffle_seeds'][2]))
    for start in (0,8):
        sel=order[start:start+8];optim.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(model(x[sel]),(y[sel]+3)%10).backward();optim.step()
    np.testing.assert_array_equal(model[0].weight.detach().numpy().T,ta['W0'][5])
    again,_,_=m.train_arrays(a,tiny_arrays(),c)
    np.testing.assert_array_equal(ta['W0'],again['W0'])
    np.testing.assert_array_equal(sa['model_state_dict']['0.weight'].numpy().T,ta['W0'][-1])

def test_phase_schedule_and_endpoint_counts():
    c=m.Config()
    for name,n in [('one_bit',15),('multi_bit',27)]:
        run=m.specs(name)[0];updates=m.checkpoint_updates(run,c,60000)
        assert len(updates)==n and updates[1]==118 and updates[2]==235
        assert updates[-1]==(len(run.bits)*2+5)*235
