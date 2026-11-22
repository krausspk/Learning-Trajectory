"""Reproduce the isolated representation-retrieval experiment."""
from pathlib import Path
import os,sys,argparse,importlib.util,subprocess
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'04_Representation_Retrieval'

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all','distances','inference','plots','report','pdf','finalize'],default='all');stage=p.parse_args().stage
    if stage in ('all','distances'):
        from representations import build_representations
        build_representations(ROOT,OUT)
    if stage in ('all','inference'):
        from representations import run_statistics
        run_statistics(ROOT,OUT)
    if stage in ('all','plots'):
        from plots import make_plots
        make_plots(OUT)
    if stage in ('all','report'):
        from reporting import make_reports
        make_reports(OUT)
    if stage in ('all','pdf'):
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_results_content.py'),'04'],check=True,cwd=ROOT)
        subprocess.run([sys.executable,str(ROOT/'scripts/build_results_pdf.py'),str(OUT)],check=True,cwd=ROOT)
    if stage in ('all','finalize'):
        path=ROOT/'02_Initialization_Retrieval/code/finalize.py';spec=importlib.util.spec_from_file_location('_repr_finalizer',path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        print(m.finalize(OUT,[ROOT/'01_RSA/code/distances.py',ROOT/'02_Initialization_Retrieval/code/inference.py']))
    print('Completed',stage)

if __name__=='__main__':main()
