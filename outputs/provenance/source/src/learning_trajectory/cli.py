"""One-command complete reproduction with inspectable intermediate artifacts."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys


def artifact_manifest(output):
    from .analysis import file_hash,write_json
    output=Path(output)
    entries=[]
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.name!='artifact_manifest.json' and 'logs' not in path.relative_to(output).parts and not path.name.endswith('.tmp'):
            entries.append({'path':str(path.relative_to(output)),'bytes':path.stat().st_size,'sha256':file_hash(path)})
    write_json(output/'artifact_manifest.json',{'created_utc':datetime.now(timezone.utc).isoformat(),'excluded':['logs (may still be streaming)','artifact_manifest.json (self)','*.tmp'],'files':entries,'total_bytes':sum(x['bytes'] for x in entries)})
    return len(entries)


def snapshot(output):
    from .analysis import file_hash,write_json
    output=Path(output);folder=output/'provenance';folder.mkdir(parents=True,exist_ok=True)
    names=['numpy','scipy','torch','pandas','matplotlib','pytest','threadpoolctl']
    env={'python':sys.version,'platform':platform.platform(),'packages':{n:importlib.metadata.version(n) for n in names},
         'thread_environment':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}}
    write_json(folder/'environment.json',env)
    source=[]
    for root in ['src','configs','tests','docs','scripts']:
        for path in sorted(Path(root).rglob('*')):
            if path.is_file() and '__pycache__' not in path.parts:
                target=folder/'source'/path;target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(path,target)
                source.append({'path':str(path),'sha256':file_hash(path)})
    for name in ['pyproject.toml','requirements.lock','README.md']:
        if Path(name).exists():shutil.copy2(name,folder/name)
    write_json(folder/'source_manifest.json',source)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['all','download','train','analyze','validate','sensitivity','figures','report','manifest','status'])
    parser.add_argument('--output',default='outputs');parser.add_argument('--data-dir',default='data')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--cohort',choices=['all','main','task','early'],default='all')
    args=parser.parse_args()
    if args.workers<1:parser.error('--workers must be positive')
    # Limits are set before importing NumPy/PyTorch to avoid nested BLAS pools.
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:
        os.environ[key]='1'
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        stages=['download','train','analyze','validate','sensitivity','figures','report','manifest'] if args.stage=='all' else [args.stage]
        for stage in stages:
            print(f'=== {stage} ===',flush=True)
            if stage=='download':
                from .data import build_probes
                print(build_probes(args.data_dir,args.output),flush=True)
            elif stage=='train':
                from .training import train_cohort
                train_cohort(args.cohort,args.data_dir,args.output,args.workers)
            elif stage=='analyze':
                from .analysis import analyze
                analyze(args.output,('main','task','early') if args.cohort=='all' else (args.cohort,),args.workers)
            elif stage=='validate':
                from .validation import run_validation
                from .checkpoints import verify_continuation
                result=run_validation(args.output,args.workers)
                if not result['passed']:raise RuntimeError('Scientific implementation validation failed; inspect validation/checks.json')
                verify_continuation(args.output,args.data_dir)
            elif stage=='sensitivity':
                from .sensitivity import run_sensitivity
                run_sensitivity(args.output)
            elif stage=='figures':
                from .figures import make_figures
                make_figures(args.output)
            elif stage=='report':
                from .report import make_report
                print(json.dumps(make_report(args.output),indent=2),flush=True)
            elif stage=='manifest':
                snapshot(args.output)
                print({'manifest_files':artifact_manifest(args.output)},flush=True)
            elif stage=='status':
                output=Path(args.output)
                for cohort in ['main','task','early']:
                    archives=list((output/'checkpoints'/cohort).glob('*.npz'))
                    pairs=list((output/'pairs'/cohort).glob('*.csv'))
                    print(f'{cohort}: {len(archives)} runs, {len(pairs)} analyzed checkpoints')
                for file in ['validation/checks.json','summary/qualitative_checks.json']:
                    p=output/file
                    if p.exists():print(p.read_text())


if __name__=='__main__':main()
