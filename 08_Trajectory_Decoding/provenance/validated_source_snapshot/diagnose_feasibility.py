#!/usr/bin/env python3
"""Descriptive post-hoc C-training feasibility; never supplies probe features.

The original protocol and all training/probe fits remain untouched. A neuron
called sample-inactive here has zero ReLU activation on a fixed 1024-image
sample; this is not proof of zero activation on every possible input.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys

# Keep this diagnostic small beside concurrent training/analysis processes.
for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd
from scipy.stats import beta, spearmanr
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT / 'src'))
from learning_trajectory.data import read_idx, NORMALIZATION


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values, float)
    if not len(values):
        return {'count': 0}
    return {'count': int(len(values)), 'mean': float(values.mean()),
            'std': float(values.std(ddof=1)) if len(values) > 1 else 0.,
            'min': float(values.min()), 'q25': float(np.quantile(values, .25)),
            'median': float(np.median(values)), 'q75': float(np.quantile(values, .75)),
            'max': float(values.max())}


def activation_diagnostics(images, w0):
    hidden = np.maximum(images @ w0, 0)
    return {'sample_dead_units': int(np.count_nonzero(np.all(hidden == 0, axis=0))),
            'fraction_active': float(np.mean(hidden > 0)),
            'hidden_rms': float(np.sqrt(np.mean(hidden.astype(np.float64) ** 2)))}


def before_c_index(archive, metadata):
    first_c_epoch = metadata['epoch_tasks'].index('C')
    update = first_c_epoch * math.ceil(metadata['train_samples'] / metadata['config']['batch_size'])
    indices = np.flatnonzero(archive['updates'] == update)
    if len(indices) != 1:
        raise ValueError(f'Cannot uniquely identify checkpoint immediately before C: {metadata["run_id"]}')
    return int(indices[0]), int(update)


def pair_success_intervals(path, confidence=.95):
    """Exact intervals for full seed-pair success, NOT individual-run accuracy."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as z:
        scores, labels, seeds = z['scores'], z['labels'], z['seeds']
    if scores.ndim == 1:
        scores, labels = scores[:, None], labels[:, None]
    unique = np.unique(seeds)
    groups = [np.flatnonzero(seeds == seed) for seed in unique]
    if any(len(group) != 2 for group in groups):
        raise ValueError('Pair-success interval requires exactly two histories per seed')
    correct = (scores >= .5) == labels
    rows = []
    targets = [(str(bit), correct[:, bit]) for bit in range(labels.shape[1])]
    if labels.shape[1] > 1:
        targets.append(('joint_all_bits', np.all(correct, axis=1)))
    for target, row_correct in targets:
        count = sum(bool(np.all(row_correct[group])) for group in groups)
        n = len(groups); tail = (1 - confidence) / 2
        rows.append({'cohort': path.parent.name, 'score_source': str(path.resolve()),
                     'score_sha256': file_digest(path), 'decoder': path.stem, 'bit': target,
                     'independent_seed_pairs': n, 'fully_correct_pairs': count,
                     'pair_success_rate': count / n, 'confidence': confidence,
                     'pair_success_ci_low': float(beta.ppf(tail, count, n - count + 1)) if count else 0.,
                     'pair_success_ci_high': float(beta.ppf(1 - tail, count + 1, n - count)) if count < n else 1.,
                     'metric_definition': 'both histories of a new seed correctly classified; conditional on fitted probe',
                     'interval_method': 'two-sided Clopper-Pearson for independent seed-pair successes',
                     'caution': 'separate pair metric, not exact per-run accuracy interval; joint_all_bits requires every bit in both histories correct'})
    return rows


def diagnose(manifest, splits, data_dir, output_prefix, cohort='multi_bit', selection_csv=None, score_files=()):
    manifest, splits, data_dir, output_prefix = map(Path, (manifest, splits, data_dir, output_prefix))
    manifest_bytes = manifest.read_bytes()
    frame = pd.read_csv(io.StringIO(manifest_bytes.decode()), dtype={'bits': str})
    frame = frame[frame.cohort == cohort].sort_values(['seed', 'arm'])
    selection_digest = None
    if selection_csv:
        selection_bytes = Path(selection_csv).read_bytes(); selection_digest = digest(selection_bytes)
        ids = set(pd.read_csv(io.StringIO(selection_bytes.decode())).run_id)
        frame = frame[frame.run_id.isin(ids)]
        if set(frame.run_id) != ids:
            raise ValueError('Selection contains IDs missing from the current cohort manifest')
    if frame.empty or frame.run_id.duplicated().any():
        raise ValueError('Expected nonempty unique manifest run IDs')
    split_bytes = splits.read_bytes(); split_doc = json.loads(split_bytes)
    split_info = split_doc.get(cohort, {}).get('splits', {})
    split_map = {}
    for key, info in split_info.items():
        for run_id in info['run_ids']:
            if run_id in split_map:
                raise ValueError('Run occurs in multiple splits')
            split_map[run_id] = key
    if split_info and not set(frame.run_id).issubset(split_map):
        raise ValueError('Some selected runs have no split assignment')

    image_path = data_dir / 'mnist' / 't10k-images-idx3-ubyte'
    data_manifest = data_dir / 'mnist' / 'manifest.json'
    data_manifest_bytes = data_manifest.read_bytes()
    dataset = json.loads(data_manifest_bytes)
    expected = next(row['sha256'] for row in dataset['files'] if row['filename'] == image_path.name)
    actual = file_digest(image_path)
    if actual != expected:
        raise ValueError('Original MNIST test image hash mismatch')
    indices = np.arange(1024, dtype='<i8')
    x = read_idx(image_path, (10000, 28, 28))[indices].reshape(1024, -1).astype(np.float32)
    mean, std = NORMALIZATION['mnist']; x = (x / 255.0 - mean) / std
    rows = []; configs = {}; source_rows = []
    for record in frame.to_dict('records'):
        path = Path(record['path'])
        if not path.is_absolute():
            path = manifest.parent / path
        archive_hash = file_digest(path)
        if archive_hash != record['npz_sha256']:
            raise ValueError(f'Checkpoint does not match frozen manifest: {path}')
        with np.load(path, allow_pickle=False) as z:
            metadata = json.loads(str(z['metadata']))
            if metadata['run_id'] != record['run_id'] or metadata['cohort'] != cohort:
                raise ValueError('Manifest/metadata identity mismatch')
            if metadata['normalization'] != [mean, std] or metadata['dataset_manifest_sha256'] != digest(data_manifest_bytes):
                raise ValueError('Diagnostic data preprocessing differs from training data provenance')
            if np.any(z['bias0']) or np.any(z['bias1']):
                raise ValueError('Expected frozen zero biases')
            pre_c, pre_c_update = before_c_index(z, metadata)
            w0, w1 = z['W0'], z['W1']
            before = activation_diagnostics(x, w0[pre_c]); final = activation_diagnostics(x, w0[-1])
            configs[digest(json.dumps(metadata['config'], sort_keys=True).encode())] = metadata['config']
            if not np.isclose(float(z['accuracy'][-1]), record['final_accuracy'], atol=1e-14, rtol=0):
                raise ValueError('Final accuracy differs between checkpoint and manifest')
            if not np.isclose(float(z['loss'][-1]), record['final_loss'], atol=1e-12, rtol=0):
                raise ValueError('Final loss differs between checkpoint and manifest')
            rows.append({'run_id': metadata['run_id'], 'seed': metadata['seed'], 'arm': metadata['arm'],
                         'bits': ''.join(str(b) for b in metadata['bits']), 'split': split_map.get(metadata['run_id'], 'not_available'),
                         'C_accuracy': float(z['accuracy'][-1]), 'C_loss': float(z['loss'][-1]),
                         'hidden_width': w0.shape[2], 'before_C_checkpoint_index': pre_c,
                         'before_C_update': pre_c_update, 'final_update': int(z['updates'][-1]),
                         **{f'final_{key}': value for key, value in final.items()},
                         **{f'before_C_{key}': value for key, value in before.items()},
                         'W0_norm': float(np.linalg.norm(w0[-1])), 'W1_norm': float(np.linalg.norm(w1[-1])),
                         'max_abs_W0': float(np.max(np.abs(w0[-1]))), 'max_abs_W1': float(np.max(np.abs(w1[-1]))),
                         'source_sha256': archive_hash, 'config_fingerprint': metadata['config_fingerprint']})
        source_rows.append({'run_id': record['run_id'], 'path': str(path.resolve()), 'sha256': archive_hash,
                            'training_source_sha256': metadata['training_source_sha256'],
                            'shared_training_source_sha256': metadata['shared_training_source_sha256'],
                            'shared_data_source_sha256': metadata['shared_data_source_sha256']})
    if len(configs) != 1:
        raise ValueError('Diagnostic expects one common frozen training configuration')
    config = next(iter(configs.values()))
    if selection_csv is None and len(frame) != 2 * config['blocks']:
        raise ValueError(f'Full-cohort mode requires {2 * config["blocks"]} runs, got {len(frame)}; use --selection-csv for a frozen validation subset')
    result = pd.DataFrame(rows)
    complete = result.groupby('seed').filter(lambda g: len(g) == 2 and set(g.arm) == {0, 1})
    pair_rows = []
    for seed, group in complete.groupby('seed'):
        first, second = group.set_index('arm').loc[0], group.set_index('arm').loc[1]
        if first['split'] != second['split']:
            raise ValueError('Paired histories split across probe partitions')
        pair_rows.append({'seed': seed, 'split': first['split'], 'C_accuracy_arm0': first.C_accuracy,
                          'C_accuracy_arm1': second.C_accuracy, 'C_accuracy_arm1_minus_arm0': second.C_accuracy-first.C_accuracy,
                          'C_accuracy_absolute_gap': abs(second.C_accuracy-first.C_accuracy),
                          'C_loss_arm0': first.C_loss, 'C_loss_arm1': second.C_loss,
                          'C_loss_arm1_minus_arm0': second.C_loss-first.C_loss,
                          'C_loss_absolute_gap': abs(second.C_loss-first.C_loss),
                          'final_sample_dead_units_arm0': first.final_sample_dead_units,
                          'final_sample_dead_units_arm1': second.final_sample_dead_units})
    pairs = pd.DataFrame(pair_rows)
    if selection_csv is None and len(pairs) != config['blocks']:
        raise ValueError('Full cohort contains incomplete seed pairs')
    rho = spearmanr(result.final_sample_dead_units, result.C_accuracy).statistic
    summary = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), 'cohort': cohort,
               'posthoc_descriptive_only': True, 'excluded_from_probe_features': True,
               'protocol_or_training_changed': False, 'selection': 'entire frozen cohort' if selection_csv is None else 'fixed existing snapshot run IDs',
               'selection_csv': str(Path(selection_csv).resolve()) if selection_csv else None,
               'selection_sha256': selection_digest, 'runs': len(result), 'complete_seed_pairs': len(pairs),
               'split_information_available': bool(split_info), 'config': config,
               'sample_dead_definition': 'exactly zero ReLU activation on each of the fixed1024 test images; not proof of global neuron death',
               'C_accuracy': distribution(result.C_accuracy), 'C_loss': distribution(result.C_loss),
               'posthoc_descriptive_accuracy_counts': {f'below_{threshold:g}': int((result.C_accuracy < threshold).sum()) for threshold in (.5,.8,.9,.95)},
               'final_sample_dead_units': distribution(result.final_sample_dead_units),
               'before_C_sample_dead_units': distribution(result.before_C_sample_dead_units),
               'paired_accuracy_absolute_gap': distribution(pairs.C_accuracy_absolute_gap) if len(pairs) else {'count':0},
               'paired_loss_absolute_gap': distribution(pairs.C_loss_absolute_gap) if len(pairs) else {'count':0},
               'spearman_sample_inactivity_vs_accuracy': float(rho) if np.isfinite(rho) else None,
               'descriptive_counts_not_preregistered_exclusion_rules': True,
               'interpretation': 'Assess common-final-performance feasibility without removing runs, retuning training or changing probes. High sample inactivity is descriptive, not causal proof. Unequal or failed C learning limits clean retention claims.',
               'source_provenance': {'script': str(Path(__file__).resolve()), 'script_sha256': file_digest(__file__),
                   'manifest': str(manifest.resolve()), 'manifest_sha256': digest(manifest_bytes),
                   'training_plan_sha256': file_digest(manifest.parent/'training_plan.json'),
                   'splits': str(splits.resolve()), 'splits_sha256': digest(split_bytes),
                   'original_data_module_sha256': file_digest(PROJECT/'src/learning_trajectory/data.py'),
                   'dataset_manifest_sha256': digest(data_manifest_bytes), 'test_image_path': str(image_path.resolve()),
                   'test_image_sha256': actual, 'normalization': [mean,std], 'sample_indices': '0..1023 inclusive',
                   'sample_indices_dtype': 'little-endian int64', 'sample_indices_sha256': digest(indices.tobytes()),
                   'normalized_images_shape': list(x.shape), 'normalized_images_dtype': 'little-endian float32',
                   'normalized_images_sha256': digest(np.ascontiguousarray(x,dtype='<f4').tobytes()),
                   'numpy_version': np.__version__, 'checkpoint_sources': source_rows}}
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_prefix.with_suffix('.csv'), index=False)
    pairs.to_csv(output_prefix.parent / (output_prefix.name + '_pairs.csv'), index=False)
    intervals = [row for score in score_files for row in pair_success_intervals(score)]
    if intervals:
        interval_path = output_prefix.parent / (output_prefix.name + '_pair_success_intervals.csv')
        pd.DataFrame(intervals).to_csv(interval_path, index=False)
        summary['separate_pair_success_intervals'] = str(interval_path.resolve())
    output_prefix.with_suffix('.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'trajectory_run_manifest.csv')
    parser.add_argument('--splits', type=Path, default=ROOT/'trajectory_probe_splits.json')
    parser.add_argument('--data-dir', type=Path, default=PROJECT/'data')
    parser.add_argument('--cohort', choices=['one_bit','multi_bit'], default='multi_bit')
    parser.add_argument('--output-prefix', type=Path, default=ROOT/'provenance/multi_bit_feasibility_final')
    parser.add_argument('--selection-csv', type=Path, help='Validate only IDs in a fixed existing snapshot CSV; full-cohort mode otherwise')
    parser.add_argument('--pair-score', type=Path, action='append', default=[], help='Optional event score NPZ; emit separately labeled exact seed-pair success intervals')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        summary=diagnose(args.manifest,args.splits,args.data_dir,args.output_prefix,args.cohort,args.selection_csv,args.pair_score)
    print(json.dumps({key:summary[key] for key in ['cohort','runs','complete_seed_pairs','C_accuracy','final_sample_dead_units','before_C_sample_dead_units']},indent=2))

if __name__=='__main__':
    main()
