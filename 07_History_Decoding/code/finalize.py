"""Read-only audit of the completed experiment and original-artifact preservation."""
from pathlib import Path
from datetime import datetime, timezone
import csv
import json
import sys
import numpy as np
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
sys.path.insert(0, str(ROOT / 'src'))
from learning_trajectory.data import sha256_file


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def main():
    plan = json.loads((OUT / 'plan.json').read_text())
    frozen = json.loads((OUT / 'provenance/freeze_receipt.json').read_text())
    assert sha256_file(OUT / 'plan.json') == frozen['plan_sha256']
    assert sha256_file(OUT / 'probe_splits.json') == frozen['splits_sha256']
    assert sha256_file(OUT / 'probe_inputs.npz') == frozen['probe_inputs_sha256']
    splits = json.loads((OUT / 'probe_splits.json').read_text())
    parts = [set(splits[k]) for k in ('train_seeds','val_seeds','test_seeds')]
    assert [len(p) for p in parts] == [120,40,40]
    assert len(set.union(*parts)) == 200 and sum(map(len,parts)) == 200
    archives = []
    for condition in ('reset','continued'):
        for seed in plan['seeds']:
            path = OUT / 'checkpoints' / condition / f'seed_{seed}.npz'
            with np.load(path, allow_pickle=False) as data:
                assert int(data['seed']) == seed
                metadata = json.loads(str(data['metadata']))
                assert metadata['mode'] == condition
                assert metadata['context']['source_hashes']['plan'] == frozen['plan_sha256']
                expected = {'W0':(2,3,784,64),'W1':(2,3,64,10),'accuracy':(2,3),
                            'loss':(2,3),'diagnostics':(2,3,6),'logits':(2,3,256,10)}
                for key, shape in expected.items():
                    assert data[key].shape == shape, (path,key,data[key].shape)
                    assert np.isfinite(data[key]).all(), (path,key)
                assert np.all((data['accuracy'] >= 0) & (data['accuracy'] <= 1))
                assert np.all(data['loss'] >= 0)
                assert np.array_equal(data['diagnostics'][...,0],data['accuracy'])
                assert np.array_equal(data['diagnostics'][...,1],data['loss'])
                assert np.array_equal(data['horizons'],[1,5,10])
            archives.append({'path':str(path.relative_to(OUT)),'sha256':sha256_file(path)})
    with (OUT / 'history_probe_results.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 20
    independent_endpoints = []
    for row in rows:
        negative = row['negative_control'].lower() in ('true','1')
        endpoint = f"{row['condition']}_{'negative_labels' if negative else row['features']}_C{row['c_epochs']}"
        with np.load(OUT / 'analysis' / f'{endpoint}.npz', allow_pickle=False) as values:
            test = values['test_indices']
            assert set(values['seed'][test]) == parts[2]
            for key, expected in zip(('train_indices','val_indices','test_indices'),parts):
                assert set(values['seed'][values[key]]) == expected
            y, score = values['target'][test], values['score'][test]
            assert np.array_equal(y.reshape(-1,2).sum(axis=1),np.ones(40))
            predicted = score >= 0
            counts = {'tn':np.sum((y==0)&~predicted),'fp':np.sum((y==0)&predicted),
                      'fn':np.sum((y==1)&~predicted),'tp':np.sum((y==1)&predicted)}
            assert all(int(row[key]) == value for key,value in counts.items())
            accuracy = float(np.mean(predicted==y))
            auc = float((rankdata(score)[y==1].sum()-40*41/2)/(40*40))
            assert abs(accuracy-float(row['balanced_accuracy'])) < 1e-12
            assert abs(auc-float(row['auc'])) < 1e-12
            null = values['permutation_balanced_accuracy']
            assert len(null)==999 and int(row['n_permutations'])==999
            p_value = (1+int(np.sum(null>=accuracy-1e-12)))/1000
            assert abs(p_value-float(row['permutation_p'])) < 1e-12
            independent_endpoints.append({'endpoint':endpoint,'recomputed_accuracy':accuracy,
                                         'recomputed_auc':auc,'recomputed_p':p_value})
    secondary = sorted([r for r in rows if r['primary'].lower() not in ('true','1')],
                       key=lambda r:float(r['permutation_p']))
    assert len(secondary)==19
    previous = 0.
    for i,row in enumerate(secondary):
        previous = max(previous,min(1.,(19-i)*float(row['permutation_p'])))
        assert abs(previous-float(row['secondary_holm_p']))<1e-12
    write_json(OUT / 'provenance/independent_result_audit.json',
               {'passed':True,'endpoints':independent_endpoints,'holm_family_size':19})
    primary = [r for r in rows if r['condition']=='reset' and r['features']=='both' and int(r['c_epochs'])==10
               and r.get('negative_control','False').lower() not in ('true','1')]
    assert len(primary)==1
    p = primary[0]
    assert int(p['n_test_blocks']) == 40 and int(p['n_permutations']) == 999
    assert int(p['tn'])+int(p['fp'])+int(p['fn'])+int(p['tp']) == 80
    ba = .5*(int(p['tn'])/(int(p['tn'])+int(p['fp'])) + int(p['tp'])/(int(p['tp'])+int(p['fn'])))
    assert abs(ba-float(p['balanced_accuracy'])) < 1e-12
    for name in ('history_run_manifest.csv','history_decoding_report.md',
                 'history_decoding_by_C_duration.png','history_decoding_by_C_duration.pdf',
                 'history_probe_ablation.png','history_probe_ablation.pdf'):
        assert (OUT / name).stat().st_size > 0, name
    with (OUT / 'history_run_manifest.csv').open() as f:
        manifest = list(csv.DictReader(f))
    assert len(manifest) == 2400
    assert len({(r['seed'],r['history'],r['momentum_mode'],r['C_horizon']) for r in manifest}) == 2400
    before = json.loads((OUT / 'provenance/source_inventory_before.json').read_text())
    changed = [name for name,value in before.items()
               if not (ROOT/name).is_file() or sha256_file(ROOT/name) != value['sha256']]
    write_json(OUT / 'provenance/preservation_check.json',
               {'checked_files':len(before),'changed_files':changed,'passed':not changed})
    assert not changed, changed
    write_json(OUT / 'provenance/completion_receipt.json',
               {'completed_at_utc':datetime.now(timezone.utc).isoformat(),'passed':True,
                'seed_archives':len(archives),'training_runs':2*len(archives),
                'final_weight_samples':6*len(archives),'probe_result_rows':len(rows),
                'primary_result':p,'frozen_plan_unchanged':True,'original_files_unchanged':len(before)})
    files = [p for p in OUT.rglob('*') if p.is_file() and '__pycache__' not in p.parts
             and '.pytest_cache' not in p.parts and p.name != 'artifact_manifest.json']
    write_json(OUT / 'artifact_manifest.json',
               {str(p.relative_to(OUT)):{'sha256':sha256_file(p),'bytes':p.stat().st_size} for p in sorted(files)})
    print(json.dumps({'passed':True,'training_runs':2*len(archives),'result_rows':len(rows),
                      'original_files_unchanged':len(before)}))


if __name__ == '__main__':
    main()
