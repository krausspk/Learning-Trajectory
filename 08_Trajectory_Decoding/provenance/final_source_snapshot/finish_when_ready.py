#!/usr/bin/env python3
"""Finish analyses as complete cohorts arrive; does not change trained data."""
import csv
import subprocess
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def count(cohort):
    path=ROOT/'trajectory_run_manifest.csv'
    if not path.exists():return 0
    with path.open() as f:return sum(r['cohort']==cohort for r in csv.DictReader(f))

for cohort in ('one_bit','multi_bit'):
    while count(cohort)<400:time.sleep(10)
    print(f'Complete training cohort {cohort}; starting analysis',flush=True)
    with (ROOT/'logs'/f'analysis_{cohort}.log').open('w') as log:
        subprocess.run([sys.executable,str(ROOT/'code/analyze.py'),'--cohort',cohort],stdout=log,stderr=subprocess.STDOUT,check=True)
    print(f'Completed analysis {cohort}',flush=True)
print('Both controlled analyses complete',flush=True)
