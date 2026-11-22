"""Reproduce point 2 while preserving the frozen point-1 experiment."""
from pathlib import Path
import os,sys,argparse,importlib.util
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'03_Initialization_Controls'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['all','distances','inference','norms','plots','report','finalize'],default='all')
    stage=parser.parse_args().stage
    if stage in ('all','distances'):
        from distances import build_distances
        build_distances(ROOT,OUT)
    if stage in ('all','inference'):
        inference=load(ROOT/'02_Initialization_Retrieval/code/inference.py','_controls_inference')
        inference.run_inference(OUT,base_seed=2026111800,controls=True)
    if stage in ('all','norms'):
        from norm_rank_control import build_norm_rank_controls
        build_norm_rank_controls(ROOT)
        wrapper=load(OUT/'code/norm_rank_inference.py','_norm_rank_inference')
        wrapper.run_inference(OUT/'norm_rank_controls')
    if stage in ('all','plots'):
        from plots import make_plots
        make_plots(OUT)
    if stage in ('all','report'):
        from reporting import make_report
        make_report(OUT)
    if stage in ('all','finalize'):
        finalizer=load(ROOT/'02_Initialization_Retrieval/code/finalize.py','_controls_finalizer')
        sources=[ROOT/'01_RSA/code/distances.py',ROOT/'02_Initialization_Retrieval/code/distances.py',ROOT/'02_Initialization_Retrieval/code/inference.py',ROOT/'02_Initialization_Retrieval/code/finalize.py',ROOT/'src/learning_trajectory/metrics.py']
        print(finalizer.finalize(OUT,sources))
        import json
        path=OUT/'retrieval_config.json'
        cfg=json.loads(path.read_text())
        cfg.update({'planned_variants':42,'posthoc_norm_rank_variants':6,'posthoc_norm_rank_seeds':{'mnist':2026111901,'fashion_mnist':2026111902,'kmnist':2026111903},'posthoc_reason':'Uncalibrated initial-versus-final norm differences give identical candidate ordering when all trained norms exceed all initial norms; norm ranks test order retention separately.','norm_rank_note':'norm_rank_controls/analysis_note.json','norm_rank_jackknife':'Ranks recalibrated on the remaining nine queries and candidates for each paired deletion, then retrieval reranked.','norm_rank_inference_source':'code/norm_rank_inference.py'})
        path.write_text(json.dumps(cfg,indent=2,ensure_ascii=False)+'\n')
        finalizer.manifest(OUT)
    print('Completed',stage,flush=True)

if __name__=='__main__':main()
