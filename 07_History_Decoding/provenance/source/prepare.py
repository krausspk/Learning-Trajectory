"""Freeze the history experiment before fitting or test-set inspection."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import platform
import shutil
import sys
import numpy as np
import torch

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'src'))
from learning_trajectory.data import read_idx, sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def inventory():
    files = [ROOT / '03_Aufgabenhistorie_aus_Endgewichten_dekodieren.md', ROOT / 'README.md']
    for directory in ('outputs', 'data', 'src', 'configs', 'references', 'figures', 'docs'):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(ROOT)): {'sha256': sha256_file(p), 'bytes': p.stat().st_size} for p in sorted(files)}


def main():
    if (OUT / 'plan.json').exists():
        raise RuntimeError('The frozen plan already exists; do not overwrite it.')
    checks = json.loads((ROOT / 'outputs/validation/checks.json').read_text())
    rerun = json.loads((ROOT / 'outputs/validation/independent_rerun_check.json').read_text())
    assert checks['passed'] and rerun['all_bitwise_equal'] and rerun['runs'] == 65
    seeds = list(range(61001, 61201))
    split_seed = 7312026
    shuffled = np.random.RandomState(split_seed).permutation(seeds).tolist()
    splits = {'train_seeds': sorted(shuffled[:120]), 'val_seeds': sorted(shuffled[120:160]),
              'test_seeds': sorted(shuffled[160:]), 'split_seed': split_seed,
              'probe_seed': 7312027, 'permutation_seed': 7312028, 'bootstrap_seed': 7312029,
              'negative_control_seed': 7312030, 'unit': 'initialization seed; both histories always together'}
    write_json(OUT / 'probe_splits.json', splits)
    # Fixed functional inputs, selected without any trained-network outputs.
    labels = read_idx(ROOT / 'data/mnist/t10k-labels-idx1-ubyte', (10000,))
    rng = np.random.RandomState(7312031)
    indices = np.concatenate([rng.choice(np.flatnonzero(labels == k), 26 if k < 6 else 25,
                                        replace=False) for k in range(10)]).astype(np.int64)
    np.savez_compressed(OUT / 'probe_inputs.npz', indices=indices, labels=labels[indices])
    plan = {
        'version': 'history-decoding-v1', 'frozen_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'frozen before any experiment training, fitting or test inspection',
        'source_specification': '03_Aufgabenhistorie_aus_Endgewichten_dekodieren.md',
        'seeds': seeds, 'independent_seed_blocks': 200, 'histories': {'0': ['A','B','C'], '1': ['B','A','C']},
        'task_permutations': {k: [(i + shift) % 10 for i in range(10)] for k, shift in [('A',0),('B',1),('C',3)]},
        'architecture': [784,64,10], 'biases': 'exactly zero, frozen', 'dataset': 'MNIST',
        'n_training_images': 60000, 'n_test_images': 10000,
        'config': {'hidden_width':64, 'batch_size':256, 'learning_rate':0.1, 'momentum':0.9,
                   'ab_epochs':1, 'c_epochs':10, 'horizons':[1,5,10], 'threads':1},
        'updates_per_epoch':235, 'updates_per_history':2820, 'c_checkpoint_updates':[705,1645,2820],
        'shuffle': 'torch.randperm(60000, Generator.manual_seed(900000 + seed*32 + zero_based_global_epoch)); same order in both histories and momentum conditions',
        'conditions': {'reset':'new SGD at A/B and B/C phase boundaries, both arms',
                       'continued':'same SGD state across all phases; robustness only'},
        'runs_per_condition':400, 'total_training_runs':800,
        'primary': {'condition':'reset','features':'both','c_epochs':10,'metric':'balanced_accuracy',
                    'one_sided_alpha':0.05,'chance':0.5},
        'secondary': ['W0','W1','both at C1/C5','diagnostics','functional_logits','covariate_adjusted_both',
                      'continued both C10','random-label negative control'],
        'probe': {'type':'linear L2 logistic regression', 'pca':False,
                  'lambda_grid':[0.0001,0.001,0.01,0.1],
                  'selection':'validation balanced accuracy, then validation log loss, then grid order',
                  'fit':'train only; no train+validation refit',
                  'alignment':'canonical ReLU gauge then Git Re-Basin to frozen train-only reference',
                  'reference':'minimum SHA256 canonical C10 weight bytes among training samples; separate per momentum condition; no label use',
                  'scaling':'train-only coordinate centering and scalar RMS per layer, divided by sqrt(layer dimension); combined divided by sqrt(2)',
                  'covariates':['accuracy','loss','raw_W0_frob','raw_W1_frob','canonical_hidden_rms','logit_rms'],
                  'covariate_adjustment':'train-only OLS weights on standardized diagnostics; logistic probe of residuals',
                  'forbidden_features':['seed','run_id','filename','earlier_weights','optimizer','history_metadata','batch_order']},
        'inference': {'primary_permutations':999,'secondary_permutations':199,'bootstrap_replicates':2000,
                      'permutation':'independent paired swaps within every seed in train, validation and test; refit classifier and reselect lambda each time; unsupervised transforms fixed',
                      'p_value':'(1 + count(null BA >= observed BA))/(1 + permutations)',
                      'confidence_intervals':'95% percentile bootstrap of test seed pairs, conditional on trained probe',
                      'multiplicity':'primary unadjusted; Holm adjustment across all secondary BA endpoints',
                      'equivalence':'report paired C performance differences and CIs; no formal equivalence claim',
                      'independent_confirmation':'pilot only; a separate larger confirmation is not part of this frozen pilot'},
        'numeric_precision': 'CPU float32 training; float64 weight alignment/probe/statistics',
        'probe_inputs_sha256':sha256_file(OUT / 'probe_inputs.npz'),
        'probe_splits_sha256':sha256_file(OUT / 'probe_splits.json'),
        'original_reproduction_prerequisite':{'validated':True,'independently_recomputed_runs':65,'all_bitwise_equal':True},
    }
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'provenance/source_inventory_before.json', inventory())
    write_json(OUT / 'provenance/freeze_receipt.json', {'plan_sha256':sha256_file(OUT/'plan.json'),
               'splits_sha256':sha256_file(OUT/'probe_splits.json'),'probe_inputs_sha256':sha256_file(OUT/'probe_inputs.npz'),
               'python':sys.version,'torch':torch.__version__,'numpy':np.__version__,'platform':platform.platform()})
    print(json.dumps({'plan_frozen':True,'seeds':len(seeds),'train':120,'validation':40,'test':40,
                      'training_runs':800,'original_files_inventoried':len(inventory())}))


if __name__ == '__main__':
    main()
