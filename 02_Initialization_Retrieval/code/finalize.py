"""Provenance and preservation checks, shared read-only by experiment 03."""
from pathlib import Path
import hashlib,json,platform
from datetime import datetime,timezone


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finalize(out:Path, extra_sources=()):
    import numpy, scipy, pandas, matplotlib
    out=Path(out).resolve();root=out.parent
    config=json.loads((out/'analysis_plan.json').read_text())
    config.update({'version':'1.0.0','completed_utc':datetime.now(timezone.utc).isoformat(),
                   'analysis_plan_sha256':sha(out/'analysis_plan.json'),
                   'software':{'python':platform.python_version(),'numpy':numpy.__version__,'scipy':scipy.__version__,'pandas':pandas.__version__,'matplotlib':matplotlib.__version__},
                   'code_sha256':{str(p.relative_to(root)):sha(p) for p in sorted((out/'code').glob('*.py'))},
                   'reused_code_sha256':{str(Path(p).resolve().relative_to(root)):sha(p) for p in extra_sources},
                   'rank_score_definition':'(n-mean_correct_rank)/(n-1); chance=.5, perfect=1',
                   'top1_ties':'Fractional credit 1/m for a true candidate tied among m minima',
                   'rank_ties':'Average ranks; exact numerical tie tolerance documented in inference_diagnostics.json',
                   'uncertainty':'Paired delete-one-network row+column; chance-adjusted scores, then back-transform to full candidate-set scale',
                   'jackknife_boundary_warning':'A perfect observed score can have zero jackknife variance; this is not population certainty',
                   'null_storage':'all label permutations and null statistics saved for reconstruction',
                   'reproduction_command':f'.venv/bin/python {out.name}/code/run.py'})
    (out/'retrieval_config.json').write_text(json.dumps(config,indent=2,ensure_ascii=False)+'\n')
    inventories=[out/'provenance/source_inventory_before.json']
    if (out/'provenance/point1_inventory_before.json').exists():inventories.append(out/'provenance/point1_inventory_before.json')
    rows=[r for p in inventories for r in json.loads(p.read_text())['files']]
    changed=[r['path'] for r in rows if not (root/r['path']).is_file() or sha(root/r['path'])!=r['sha256']]
    preservation={'all_unchanged':not changed,'files_checked':len(rows),'changed':changed,'checked_utc':datetime.now(timezone.utc).isoformat()}
    (out/'provenance/preservation_check.json').write_text(json.dumps(preservation,indent=2)+'\n')
    if changed:raise AssertionError(f'Previous experiment files changed: {changed}')
    manifest(out)
    return preservation


def manifest(out):
    out=Path(out)
    paths=sorted(p for p in out.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and 'logs' not in p.relative_to(out).parts)
    rows=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in paths]
    (out/'artifact_manifest.json').write_text(json.dumps({'files':rows},indent=2)+'\n')
    return len(rows)
