"""Rebuild the isolated point-1 experiment from saved training checkpoints."""
from pathlib import Path
import os,sys,argparse
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'02_Initialization_Retrieval'
sys.path.insert(0,str(ROOT/'src'))

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','distances','inference','plots','report','finalize'],default='all')
    stage=parser.parse_args().stage
    if stage in ('all','distances'):
        from distances import build_distances
        build_distances(ROOT,OUT)
    if stage in ('all','inference'):
        from inference import run_inference
        run_inference(OUT)
    if stage in ('all','plots'):
        from plots import make_plots
        make_plots(OUT)
    if stage in ('all','report'):
        from reporting import make_report
        make_report(OUT)
    if stage in ('all','finalize'):
        from finalize import finalize
        print(finalize(OUT,[ROOT/'01_RSA/code/distances.py',ROOT/'src/learning_trajectory/metrics.py']))
    print('Completed',stage,flush=True)

if __name__=='__main__':main()
