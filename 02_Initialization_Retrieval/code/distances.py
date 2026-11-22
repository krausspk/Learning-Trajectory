"""Blind cross-time initialization retrieval distances, isolated from prior work.

Rows are current query networks; columns are candidate initializations. All 100
query/candidate pairs use exactly the same independent matching computation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
METRIC_KEY = "normalized_frobenius|total|optimal|canonical"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_module(root=ROOT):
    """Load the unchanged manifest/gauge implementation by an explicit hashed name."""
    path = Path(root)/"01_RSA/code/distances.py"
    module_name = "_initialization_retrieval_rsa01_source_"+sha256(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name,path)
    if spec is None or spec.loader is None: raise ImportError(f"Cannot import verified RSA source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


SOURCE = source_module()


def canonical_unit_network(W0,W1):
    """Canonical neuron gauge, then one shared total-weight L2 normalization."""
    a,b,counts = SOURCE.canonical_gauge(W0,W1)
    total_norm = float(np.sqrt(np.sum(a*a)+np.sum(b*b)))
    if not np.isfinite(total_norm) or total_norm <= 0:
        raise ValueError("The complete canonical weight vector must have a positive finite norm")
    return (a/total_norm,b/total_norm),counts


def match_unit_networks(query,candidate):
    """Map candidate neurons onto query neurons using a joint Hungarian objective."""
    a0,a1 = query
    b0,b1 = candidate
    if a0.shape != b0.shape or a1.shape != b1.shape or a0.shape[1] != a1.shape[0]:
        raise ValueError("Query and candidate must have the same valid architecture")
    rows,cols = linear_sum_assignment(-(a0.T@b0+a1@b1.T))
    permutation = np.empty(a0.shape[1],dtype=np.int64)
    permutation[rows] = cols
    # Direct squared differences avoid cancellation near the E0 identity match.
    value = float(np.sqrt(np.sum((a0-b0[:,permutation])**2)+np.sum((a1-b1[permutation,:])**2)))
    return value,permutation


def retrieval_distance(query,candidate):
    query_unit,_ = canonical_unit_network(*query)
    candidate_unit,_ = canonical_unit_network(*candidate)
    return match_unit_networks(query_unit,candidate_unit)


def cross_distances(queries,candidates,already_normalized=False):
    """Return a rectangular cost matrix with no identity/diagonal information."""
    if not queries or not candidates: raise ValueError("Both network ensembles must be nonempty")
    if not already_normalized:
        queries = [canonical_unit_network(*network)[0] for network in queries]
        candidates = [canonical_unit_network(*network)[0] for network in candidates]
    values = np.empty((len(queries),len(candidates)),dtype=np.float64)
    permutations = np.empty((len(queries),len(candidates),queries[0][0].shape[1]),dtype=np.int64)
    for query_index,query in enumerate(queries):
        for candidate_index,candidate in enumerate(candidates):
            values[query_index,candidate_index],permutations[query_index,candidate_index] = match_unit_networks(query,candidate)
    return values,permutations


def build_distances(root:Path,out:Path):
    root,out = Path(root).resolve(),Path(out).resolve()
    if out != root/"02_Initialization_Retrieval":
        raise ValueError("Retrieval outputs must be isolated in 02_Initialization_Retrieval")
    source = source_module(root)
    prior_validation_path = root/"01_RSA/data/validation.json"
    prior_validation = json.loads(prior_validation_path.read_text())
    if not prior_validation.get("passed") or prior_validation.get("code_sha256") != sha256(root/"01_RSA/code/distances.py"):
        raise ValueError("Prior RSA distance code must match its successfully validated source fingerprint")
    groups,manifest,run_checks = source.load_manifest_runs(root)
    # This also verifies association with the original reproduction's figure data.
    _,pipeline_checks = source.load_pipeline_artifacts(root,manifest)
    data_dir = out/"data"
    data_dir.mkdir(parents=True,exist_ok=True)
    source.save_json(data_dir/"source_manifest.json",manifest)
    source.save_json(data_dir/"distance_definitions.json",{
        METRIC_KEY:{
            "formula":"min_P ||u(W_i(t)) - P u(W_j(0))||_2, with u the flattened joint W0/W1 vector after canonical neuron gauge and total L2 normalization",
            "normalization":"Equalize incoming/outgoing neuron norms first; divide BOTH W0 and W1 by sqrt(||W0||_F^2+||W1||_F^2)",
            "matching":"Maximize the joint normalized W0/W1 inner product with a global Hungarian assignment for one hidden layer",
            "equivalence":"Distance squared = 2*(1-cosine), because both full vectors have norm one",
            "orientation":"distances[0,t,i,j]: current query run i at checkpoint t versus candidate run j at checkpoint E0",
            "permutation_orientation":"p[k] is candidate hidden neuron assigned to query hidden neuron k; aligned candidate=(B0[:,p],B1[p,:])",
            "identity_policy":"The matching functions receive no run identity. All 100 entries, including the diagonal, use the same computation; no diagonal replacement or preference",
            "zero_norm_convention":"Reuse the deterministic 01_RSA neuron-level convention. Reject an entirely zero full weight vector, for which unit normalization is undefined.",
            "norms_arrays":"norms_raw/canonical retain PRE-total-normalization W0,W1,total norms in norm_layers order",
        }
    })
    records = []
    zero_counts = {"both_positive":0,"incoming_only":0,"outgoing_only":0,"both_zero":0}
    max_unit_norm_error,max_cosine_squared_error,max_gauge_error = 0.0,0.0,0.0
    with threadpool_limits(limits=1):
        for dataset,runs in groups.items():
            print(f"Initialization retrieval: {dataset}, 21 checkpoints x 100 blind query/candidate matches",flush=True)
            normalized = []
            norms_raw,norms_canonical = np.empty((21,10,3)),np.empty((21,10,3))
            for checkpoint in range(21):
                row = []
                for run_index,run in enumerate(runs):
                    raw = (run["W0"][checkpoint],run["W1"][checkpoint])
                    a,b,counts = source.canonical_gauge(*raw)
                    for key,value in counts.items(): zero_counts[key]+=value
                    if counts["both_positive"]==64:
                        expected = source.pipeline_metrics.canonicalize(*raw)
                        max_gauge_error = max(max_gauge_error,max(float(np.max(np.abs(x-y))) for x,y in zip((a,b),expected)))
                    for weights,target in ((raw,norms_raw),((a,b),norms_canonical)):
                        target[checkpoint,run_index] = [np.linalg.norm(weights[0]),np.linalg.norm(weights[1]),np.sqrt(np.sum(weights[0]**2)+np.sum(weights[1]**2))]
                    full_norm = norms_canonical[checkpoint,run_index,2]
                    if not np.isfinite(full_norm) or full_norm <= 0: raise ValueError("Cannot normalize an entirely zero network")
                    unit = (a/full_norm,b/full_norm)
                    row.append(unit)
                    max_unit_norm_error = max(max_unit_norm_error,abs(float(np.sum(unit[0]**2)+np.sum(unit[1]**2))-1))
                normalized.append(row)
            values = np.empty((1,21,10,10),dtype=np.float64)
            permutations = np.empty((21,10,10,64),dtype=np.int64)
            initial_candidates = normalized[0]
            for checkpoint in range(21):
                values[0,checkpoint],permutations[checkpoint] = cross_distances(normalized[checkpoint],initial_candidates,already_normalized=True)
                for i in range(10):
                    for j in range(10):
                        query,candidate = normalized[checkpoint][i],initial_candidates[j]
                        p = permutations[checkpoint,i,j]
                        cosine = float(np.sum(query[0]*candidate[0][:,p])+np.sum(query[1]*candidate[1][p,:]))
                        max_cosine_squared_error = max(max_cosine_squared_error,abs(values[0,checkpoint,i,j]**2-2*(1-cosine)))
            if not np.isfinite(values).all() or np.any(values<0) or np.any(values>2+1e-12): raise AssertionError("Invalid normalized retrieval distance")
            zero_diagonal_error = float(np.max(np.abs(np.diag(values[0,0]))))
            e0_symmetry_error = float(np.max(np.abs(values[0,0]-values[0,0].T)))
            run_ids = np.asarray([run["run_id"] for run in runs])
            path = data_dir/f"{dataset}_retrieval.npz"
            np.savez_compressed(path,distances=values,metric_keys=np.asarray([METRIC_KEY]),checkpoints=np.arange(21),
                                query_run_ids=run_ids,run_ids=run_ids,candidate_run_ids=run_ids,
                                seeds=np.asarray([run["seed"] for run in runs]),permutations=permutations,
                                norms_raw=norms_raw,norms_canonical=norms_canonical,norm_layers=np.asarray(["W0","W1","total"]),
                                candidate_checkpoint=np.asarray(0),source_manifest_sha256=np.asarray(sha256(root/"outputs/manifests/main.json")))
            records.append({"dataset":dataset,"path":str(path),"sha256":sha256(path),"shape":list(values.shape),
                            "query_run_ids":run_ids.tolist(),"candidate_run_ids":run_ids.tolist(),
                            "seeds":[run["seed"] for run in runs],"E0_max_diagonal_distance":zero_diagonal_error,
                            "E0_max_symmetry_error":e0_symmetry_error,"matches_per_checkpoint":100})
    passed = (max_unit_norm_error<1e-12 and max_cosine_squared_error<1e-12 and max_gauge_error<1e-12
              and all(record["E0_max_diagonal_distance"]<1e-12 and record["E0_max_symmetry_error"]<1e-12 for record in records))
    validation = {"passed":bool(passed),"primary_key":METRIC_KEY,"code_sha256":sha256(__file__),
                  "source_code_path":str(root/"01_RSA/code/distances.py"),"source_code_sha256":sha256(root/"01_RSA/code/distances.py"),
                  "source_validation_path":str(prior_validation_path),"source_validation_sha256":sha256(prior_validation_path),
                  "source_manifest_path":str(root/"outputs/manifests/main.json"),"source_manifest_sha256":sha256(root/"outputs/manifests/main.json"),
                  "runs":run_checks,"original_pipeline_sources":pipeline_checks,"outputs":records,"zero_norm_counts":zero_counts,
                  "max_unit_squared_norm_error":max_unit_norm_error,"max_cosine_squared_distance_equivalence_error":max_cosine_squared_error,
                  "max_pipeline_gauge_difference":max_gauge_error,"n_distances":3*21*100,
                  "all_entries_computed_identically":True,"diagonal_overridden":False,"candidate_checkpoint":0,
                  "previous_experiments_modified":False,"training_performed":False}
    source.save_json(data_dir/"validation.json",validation)
    if not passed: raise AssertionError("Retrieval distance validation failed; inspect data/validation.json")
    return validation


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=ROOT/"02_Initialization_Retrieval")
    args = parser.parse_args()
    result = build_distances(args.root,args.out)
    print(json.dumps({"passed":result["passed"],"n_distances":result["n_distances"]}))
