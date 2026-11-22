"""Load a stored state for a later continuation or controlled intervention."""
from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch
from .training import build_model, _configure_threads


def restore_checkpoint(path, checkpoint, restore_rng=True):
    """Restore a trusted local .pt archive and return (model, optimizer, state).

    ``state`` includes the current epoch permutation and consumed batch position.
    Continue its remaining minibatches, then construct the next epoch order with
    seed + epoch. The end-of-epoch checkpoint stores the completed epoch number.
    The source archive is never modified. Only load trusted local torch archives.
    """
    archive=torch.load(Path(path),map_location='cpu',weights_only=False)
    found=[s for s in archive['checkpoints'] if s['checkpoint']==checkpoint]
    if len(found)!=1: raise KeyError(f'Checkpoint {checkpoint} not uniquely found in {path}')
    state=found[0];config=state['config'];spec=state['spec']
    _configure_threads(config['threads'])
    model=build_model(spec['seed'],2 if spec['task']=='parity' else 10,config['hidden_width'])
    optimizer=torch.optim.SGD([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],momentum=config['momentum'])
    model.load_state_dict(state['model_state_dict'])
    optimizer.load_state_dict(state['optimizer_state_dict'])
    if restore_rng:
        torch.set_rng_state(state['torch_rng_state'])
        np.random.set_state(state['numpy_rng_state'])
        random.setstate(state['python_rng_state'])
    return model,optimizer,state


def verify_continuation(output='outputs',data_dir='data'):
    """Replay the second half of main/M1 epoch one from its saved E1 state."""
    from .data import load_dataset
    from .analysis import write_json
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    path=Path(output)/'checkpoints/main/M1.pt'
    model,optimizer,state=restore_checkpoint(path,1)
    images,labels,_,_=load_dataset('mnist',data_dir)
    order=state['epoch_order'];batch=state['config']['batch_size']
    count=0
    for start in range(state['samples_consumed_this_epoch'],len(images),batch):
        selected=order[start:start+batch]
        optimizer.zero_grad(set_to_none=True)
        loss=torch.nn.functional.cross_entropy(model(images[selected]),labels[selected])
        loss.backward();optimizer.step();count+=1
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as saved:
        errors=[float(np.max(np.abs(model[j].weight.detach().numpy().T-saved[name][2]))) for j,name in [(0,'W0'),(2,'W1')]]
    result={'run':'main/M1','from_checkpoint':1,'to_checkpoint':2,'remaining_updates':count,
            'W0_max_error':errors[0],'W1_max_error':errors[1],'bitwise_equal':max(errors)==0}
    write_json(Path(output)/'validation/continuation_check.json',result)
    if not result['bitwise_equal']:raise AssertionError(f'Saved continuation differs: {result}')
    return result
