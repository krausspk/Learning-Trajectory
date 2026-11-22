"""Manifest-bound RDMs for the isolated initial-condition RSA experiment.

This module reads the completed reproduction and writes exclusively below 01_RSA.
It never trains, rewrites a checkpoint, or changes the original analysis pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
from learning_trajectory import metrics as pipeline_metrics


DATASETS = {"mnist": ("M", 100), "fashion_mnist": ("F", 200), "kmnist": ("K", 300)}
PRIMARY_KEY = "frobenius|total|optimal|canonical"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def encode(obj):
        if isinstance(obj,np.ndarray): return obj.tolist()
        if isinstance(obj,np.generic): return obj.item()
        if isinstance(obj,Path): return str(obj)
        raise TypeError(type(obj).__name__)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,default=encode,allow_nan=False)+"\n")
    temporary.replace(path)


def _resolved(root, recorded):
    path = Path(recorded)
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def load_manifest_runs(root):
    """Resolve every run solely through the completed, hashed main manifest."""
    root = Path(root).resolve()
    manifest_path = root/"outputs/manifests/main.json"
    manifest = json.loads(manifest_path.read_text())
    if not (manifest.get("complete") and manifest.get("cohort") == "main" and manifest.get("expected_runs") == 30):
        raise ValueError("The reproduced main cohort manifest is not complete")
    records = manifest.get("runs",[])
    if len(records) != 30 or len({r["run_id"] for r in records}) != 30:
        raise ValueError("The main manifest must contain exactly 30 distinct run identities")
    validation_path = root/"outputs/validation/checks.json"
    if not json.loads(validation_path.read_text()).get("passed"):
        raise ValueError("The original reproduction's validation has not passed")
    source_hashes = {"training_source_sha256":sha256(root/"src/learning_trajectory/training.py"),
                     "data_source_sha256":sha256(root/"src/learning_trajectory/data.py")}
    runs = {name:[] for name in DATASETS}
    audit = []
    for record in records:
        dataset = record["dataset"]
        if dataset not in DATASETS: raise ValueError(f"Unexpected dataset: {dataset}")
        prefix,seed_base = DATASETS[dataset]
        if not (seed_base+1 <= record["seed"] <= seed_base+10 and record["run_id"] == f"{prefix}{record['seed']-seed_base}"):
            raise ValueError(f"Seed/dataset/run identity mismatch: {record}")
        path = _resolved(root,record["path"])
        if not path.is_file() or sha256(path) != record["sha256"]:
            raise ValueError(f"Missing or hash-mismatched checkpoint: {path}")
        state_path = _resolved(root,record["full_state_path"])
        if not state_path.is_file() or sha256(state_path) != record["full_state_sha256"]:
            raise ValueError(f"Missing or hash-mismatched full-state archive: {state_path}")
        with np.load(path,allow_pickle=False) as archive:
            meta = json.loads(str(archive["metadata"]))
            for key in ("run_id","cohort","dataset","task","seed","block","config_hash"):
                if meta.get(key) != record.get(key): raise ValueError(f"Manifest/metadata mismatch for {key}: {path}")
            if meta["cohort"] != "main" or meta["task"] != "identity": raise ValueError(f"Wrong cohort/task: {path}")
            for key,expected in source_hashes.items():
                if meta.get(key) != expected: raise ValueError(f"Source provenance mismatch for {key}: {path}")
            dataset_manifest = root/"data"/dataset/"manifest.json"
            if meta.get("dataset_manifest_sha256") != sha256(dataset_manifest): raise ValueError(f"Dataset provenance mismatch: {path}")
            if meta.get("full_state_sha256") != record["full_state_sha256"]: raise ValueError(f"State hash metadata mismatch: {path}")
            expected_config = {"hidden_width":64,"batch_size":256,"epochs":10,"learning_rate":0.1,"momentum":0.9}
            if any(meta.get("config",{}).get(k) != v for k,v in expected_config.items()): raise ValueError(f"Training configuration mismatch: {path}")
            if not np.array_equal(archive["checkpoints"],np.arange(21)): raise ValueError(f"Expected checkpoints E0..E20: {path}")
            expected_updates = np.asarray([0]+[update for epoch in range(10) for update in (epoch*235+118,(epoch+1)*235)])
            if not np.array_equal(archive["updates"],expected_updates): raise ValueError(f"Unexpected half-epoch update schedule: {path}")
            for key,shape in (("W0",(21,784,64)),("W1",(21,64,10)),("bias0",(21,64)),("bias1",(21,10)),("accuracy",(21,)),("loss",(21,))):
                if archive[key].shape != shape or not np.isfinite(archive[key]).all(): raise ValueError(f"Bad shape/nonfinite {key}: {path}")
            if np.any(archive["bias0"]) or np.any(archive["bias1"]): raise ValueError(f"Nonzero fixed bias: {path}")
            runs[dataset].append({"run_id":record["run_id"],"seed":record["seed"],"dataset":dataset,"path":path,
                                  "W0":archive["W0"].astype(np.float64),"W1":archive["W1"].astype(np.float64),"meta":meta})
        audit.append({"run_id":record["run_id"],"dataset":dataset,"seed":record["seed"],"path":str(path),
                      "sha256":record["sha256"],"full_state_path":str(state_path),"full_state_sha256":record["full_state_sha256"],
                      "source_manifest":str(manifest_path),"source_manifest_sha256":sha256(manifest_path),"all_checks_passed":True})
    for dataset,group in runs.items():
        group.sort(key=lambda r:r["seed"])
        if len(group) != 10 or len({r["seed"] for r in group}) != 10: raise ValueError(f"Expected ten distinct seeds: {dataset}")
    return runs,manifest,audit


def canonical_gauge(W0,W1):
    """Equal incoming/outgoing norms with deterministic zero-norm conventions."""
    a,b = np.asarray(W0,dtype=np.float64),np.asarray(W1,dtype=np.float64)
    incoming,outgoing = np.linalg.norm(a,axis=0),np.linalg.norm(b,axis=1)
    both = (incoming > 0) & (outgoing > 0)
    only_in = (incoming > 0) & (outgoing == 0)
    only_out = (incoming == 0) & (outgoing > 0)
    neither = (incoming == 0) & (outgoing == 0)
    scale = np.ones_like(incoming)
    scale[both] = np.sqrt(outgoing[both]/incoming[both])
    scale[only_in] = 1/incoming[only_in]
    scale[only_out] = outgoing[only_out]
    if not np.isfinite(scale).all() or np.any(scale <= 0): raise ValueError("Canonicalization produced an invalid positive scale")
    result = a*scale[None,:],b/scale[:,None]
    counts = {"both_positive":int(both.sum()),"incoming_only":int(only_in.sum()),"outgoing_only":int(only_out.sum()),"both_zero":int(neither.sum())}
    return *result,counts


def git_permutation(a0,a1,b0,b1):
    """Shared optimal assignment of incoming plus outgoing weight inner products."""
    rows,cols = linear_sum_assignment(-(a0.T@b0+a1@b1.T))
    result = np.empty(a0.shape[1],dtype=np.int64)
    result[rows] = cols
    return result


def geometric_distances(a0,a1,b0,b1,p):
    b0,b1 = b0[:,p],b1[p,:]
    sq0,sq1 = float(np.sum((a0-b0)**2)),float(np.sum((a1-b1)**2))
    values = {("frobenius","W0"):np.sqrt(sq0),("frobenius","W1"):np.sqrt(sq1),
              ("frobenius","total"):np.sqrt(sq0+sq1),
              ("frobenius","balanced"):np.sqrt(0.5*(sq0/a0.size+sq1/a1.size))}
    for layer,a,b in (("W0",a0,b0),("W1",a1,b1),("total",np.concatenate((a0.ravel(),a1.ravel())),np.concatenate((b0.ravel(),b1.ravel())))):
        na,nb = np.linalg.norm(a),np.linalg.norm(b)
        cosine = 0.0 if na == nb == 0 else (1.0 if na == 0 or nb == 0 else float(np.clip(1-np.sum(a*b)/(na*nb),0,2)))
        values[("cosine",layer)] = cosine
        values[("normalized_frobenius",layer)] = np.sqrt(2*cosine)
    return values


def definitions():
    result = {}
    for gauge,alignment in (("canonical","optimal"),("canonical","fixed_E0"),("canonical","fixed_E20"),("raw","optimal")):
        for metric,layer in (("frobenius","total"),("frobenius","W0"),("frobenius","W1"),("frobenius","balanced"),
                             ("cosine","total"),("cosine","W0"),("cosine","W1"),
                             ("normalized_frobenius","total"),("normalized_frobenius","W0"),("normalized_frobenius","W1")):
            key = "|".join((metric,layer,alignment,gauge))
            formula = {"frobenius":"sqrt(sum of squared elementwise differences)","cosine":"1 - <A,B>/(||A||_F ||B||_F)",
                       "normalized_frobenius":"sqrt(2 * cosine_dissimilarity), equal to distance between unit-normalized nonzero vectors"}[metric]
            if layer == "balanced": formula = "sqrt(0.5*(||delta W0||_F^2 / n0 + ||delta W1||_F^2 / n1)); RMS of layer RMS distances"
            result[key] = {"metric":metric,"layer":layer,"alignment":alignment,"gauge":gauge,"formula":formula,
                           "assignment_objective":"Maximize joint W0/W1 Frobenius inner product in the stated gauge (one Hungarian assignment)",
                           "assignment_timing":"recomputed separately for every pair/checkpoint" if alignment=="optimal" else f"pair assignment inferred at {alignment[6:]} and used unchanged for BOTH initial and comparison RDMs at all checkpoints",
                           "layer_convention":"Both layers share the same joint-network permutation; no layer-specific rematching",
                           "total_convention":"Flatten and concatenate W0,W1" if layer=="total" else "as specified by layer",
                           "zero_norm_convention":"cosine=0 if both vectors are zero; cosine=1 if exactly one is zero; normalized distance defined by sqrt(2*cosine)"}
    for metric,gauge,method in (("alpha2","canonical","alpha_profile"),("alpha2","raw","alpha_profile"),("snd","raw","snd_profile")):
        for layer in ("W0","W1","balanced","size_weighted"):
            key = "|".join((metric,layer,method,gauge))
            result[key] = {"metric":metric,"layer":layer,"alignment":method,"gauge":gauge,
                           "formula":"sum |a*b|^2 * sign_disagreement / sum |a*b|^2" if metric=="alpha2" else "mean entrywise SND: 0 for both zero; |a-b|/|a+b| for equal nonzero signs (denominator>1e-12); 1 otherwise",
                           "assignment_objective":"Original reproduced pipeline's direct canonical-Alpha profile Hungarian" if gauge=="canonical" else "Original reproduced pipeline's PIF-initialized metric-specific profile Hungarian",
                           "aggregation":"(d0+d1)/2" if layer=="balanced" else ("(n0*d0+n1*d1)/(n0+n1)" if layer=="size_weighted" else "matrix-level measure"),
                           "reuse":"Original hash-verified per-checkpoint assignments; values independently reevaluated from stored weights and compared to original pair CSV",
                           "caution":"Alpha is a sign-disagreement dissimilarity, not a mathematical metric" if metric=="alpha2" else "SND defines a different geometry from Frobenius"}
    return result


def load_pipeline_artifacts(root,manifest):
    """Validate original figure-source receipts and retain their exact run mapping."""
    root = Path(root)
    files = [root/"src/learning_trajectory"/name for name in ("analysis.py","metrics.py","statistics.py")]
    files += [_resolved(root,r["path"]) for r in sorted(manifest["runs"],key=lambda r:r["run_id"])]
    files += [root/"outputs/probes/main.npz"]
    fingerprint = hashlib.sha256("".join(sha256(p) for p in files).encode()).hexdigest()
    artifacts,audit = {},[]
    for checkpoint in range(21):
        csv_path = root/f"outputs/pairs/main/checkpoint_{checkpoint:04d}.csv"
        receipt_path = csv_path.with_suffix(".json")
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("fingerprint") != fingerprint or receipt.get("sha256") != sha256(csv_path): raise ValueError(f"Stale or corrupt original pair-table receipt: {receipt_path}")
        if receipt.get("checkpoint") != checkpoint or receipt.get("cohort") != "main" or receipt.get("unordered_pairs") != 435: raise ValueError(f"Wrong original checkpoint mapping: {receipt_path}")
        alignment_path = _resolved(root,receipt["alignment_path"])
        if receipt["alignment_sha256"] != sha256(alignment_path): raise ValueError(f"Corrupt alignment archive: {alignment_path}")
        lookup = {}
        with np.load(alignment_path,allow_pickle=False) as archive:
            ids = archive["run_ids"].tolist()
            if set(ids) != {r["run_id"] for r in manifest["runs"]}: raise ValueError("Alignment run IDs differ from main manifest")
            for (a,b),method,p in zip(archive["pairs"],archive["methods"],archive["permutations"]):
                if not np.array_equal(np.sort(p),np.arange(64)): raise ValueError("Invalid original neuron assignment")
                lookup[(ids[a],ids[b],str(method))] = p.copy()
                lookup[(ids[b],ids[a],str(method))] = np.argsort(p)
        frame = pd.read_csv(csv_path)
        values = {}
        relevant = frame[(frame.alignment=="profile") & frame.metric.isin(["raw_alpha","canonical_alpha","snd"])]
        for row in relevant.itertuples(index=False):
            values[(row.run_a,row.run_b,row.metric,row.layer)] = float(row.value)
            values[(row.run_b,row.run_a,row.metric,row.layer)] = float(row.value)
        artifacts[checkpoint] = {"permutations":lookup,"values":values}
        audit.append({"checkpoint":checkpoint,"pair_table":str(csv_path),"pair_sha256":receipt["sha256"],"receipt_sha256":sha256(receipt_path),
                      "alignment_path":str(alignment_path),"alignment_sha256":receipt["alignment_sha256"],"fingerprint":fingerprint,
                      "figure_mapping":"Primary canonical-Alpha Figures 1/2; profile raw-Alpha/SND supplement S2/S3; no benchmark-candidate substitution"})
    return artifacts,audit


def assignment_diagnostics(dataset,run_ids,pairs,permutations):
    changes,cycles = [],[]
    index = {tuple(pair):k for k,pair in enumerate(pairs)}
    def oriented(checkpoint,a,b):
        if a < b:return permutations[checkpoint,index[(a,b)]]
        return np.argsort(permutations[checkpoint,index[(b,a)]])
    for checkpoint in range(21):
        for pair_index,(a,b) in enumerate(pairs):
            changes.append({"dataset":dataset,"checkpoint":checkpoint,"run_a":run_ids[a],"run_b":run_ids[b],
                            "change_fraction_vs_E0":float(np.mean(permutations[checkpoint,pair_index]!=permutations[0,pair_index])),
                            "change_fraction_vs_E20":float(np.mean(permutations[checkpoint,pair_index]!=permutations[20,pair_index]))})
    for checkpoint in (0,20):
        for a,b,c in itertools.combinations(range(10),3):
            ab,bc,ac = oriented(checkpoint,a,b),oriented(checkpoint,b,c),oriented(checkpoint,a,c)
            composed = bc[ab]
            cycles.append({"dataset":dataset,"checkpoint":checkpoint,"run_a":run_ids[a],"run_b":run_ids[b],"run_c":run_ids[c],
                           "fraction_consistent":float(np.mean(composed==ac)),"exactly_consistent":bool(np.array_equal(composed,ac))})
    return changes,cycles


def build_distances(root:Path,out:Path):
    root,out = Path(root).resolve(),Path(out).resolve()
    if out != root/"01_RSA": raise ValueError("RSA distance outputs must be isolated in the project's 01_RSA directory")
    out.mkdir(parents=True,exist_ok=True)
    definitions_map = definitions()
    metric_keys = list(definitions_map)
    key_index = {key:i for i,key in enumerate(metric_keys)}
    runs,manifest,run_audit = load_manifest_runs(root)
    artifacts,pipeline_audit = load_pipeline_artifacts(root,manifest)
    save_json(out/"data/source_manifest.json",manifest)
    save_json(out/"data/distance_definitions.json",definitions_map)
    norm_rows,zero_rows,changes,cycles,output_records = [],[],[],[],[]
    max_pipeline_error,max_gauge_error,max_function_error,max_transpose_error = 0.0,0.0,0.0,0.0
    probe = np.load(root/"outputs/probes/main.npz",allow_pickle=False)["X"][:32].astype(np.float64)
    with threadpool_limits(limits=1):
        for dataset,group in runs.items():
            print(f"RSA distances: {dataset}, 10 manifest-bound runs, 21 checkpoints, 45 pairs",flush=True)
            pairs = np.asarray(list(itertools.combinations(range(10),2)),dtype=np.int64)
            distances = np.zeros((len(metric_keys),21,10,10),dtype=np.float64)
            permutations = np.empty((21,45,64),dtype=np.int64)
            raw_permutations = np.empty_like(permutations)
            alpha_permutations = np.empty_like(permutations)
            raw_alpha_permutations = np.empty_like(permutations)
            snd_permutations = np.empty_like(permutations)
            norms_raw,norms_canonical = np.empty((21,10,3)),np.empty((21,10,3))
            canonical = []
            for run_index,run in enumerate(group):
                trajectory = []
                for checkpoint in range(21):
                    a0,a1 = run["W0"][checkpoint],run["W1"][checkpoint]
                    c0,c1,counts = canonical_gauge(a0,a1)
                    trajectory.append((c0,c1))
                    zero_rows.append({"dataset":dataset,"run_id":run["run_id"],"checkpoint":checkpoint,**counts})
                    if counts["both_positive"] == 64:
                        expected = pipeline_metrics.canonicalize(a0,a1)
                        max_gauge_error = max(max_gauge_error,max(float(np.max(np.abs(a-b))) for a,b in zip((c0,c1),expected)))
                    max_function_error = max(max_function_error,float(np.max(np.abs(np.maximum(probe@a0,0)@a1-np.maximum(probe@c0,0)@c1))))
                    for gauge,weights,target in (("raw",(a0,a1),norms_raw),("canonical",(c0,c1),norms_canonical)):
                        values = [np.linalg.norm(weights[0]),np.linalg.norm(weights[1]),np.sqrt(np.sum(weights[0]**2)+np.sum(weights[1]**2))]
                        target[checkpoint,run_index] = values
                        norm_rows.extend({"dataset":dataset,"run_id":run["run_id"],"checkpoint":checkpoint,"gauge":gauge,"layer":layer,"norm":float(value)} for layer,value in zip(("W0","W1","total"),values))
                canonical.append(trajectory)
            for checkpoint in range(21):
                for pair_index,(ia,ib) in enumerate(pairs):
                    a0,a1 = canonical[ia][checkpoint]
                    b0,b1 = canonical[ib][checkpoint]
                    permutations[checkpoint,pair_index] = git_permutation(a0,a1,b0,b1)
                    if checkpoint in (0,20):
                        reverse = git_permutation(b0,b1,a0,a1)
                        forward_value = geometric_distances(a0,a1,b0,b1,permutations[checkpoint,pair_index])[("frobenius","total")]
                        reverse_value = geometric_distances(b0,b1,a0,a1,reverse)[("frobenius","total")]
                        max_transpose_error = max(max_transpose_error,abs(forward_value-reverse_value))
            for checkpoint in range(21):
                for pair_index,(ia,ib) in enumerate(pairs):
                    ra,rb = group[ia],group[ib]
                    ca,cb = canonical[ia][checkpoint],canonical[ib][checkpoint]
                    raw_a = (ra["W0"][checkpoint],ra["W1"][checkpoint])
                    raw_b = (rb["W0"][checkpoint],rb["W1"][checkpoint])
                    raw_p = git_permutation(*raw_a,*raw_b)
                    raw_permutations[checkpoint,pair_index] = raw_p
                    for gauge,alignment,a,b,p in (("canonical","optimal",ca,cb,permutations[checkpoint,pair_index]),
                                                  ("canonical","fixed_E0",ca,cb,permutations[0,pair_index]),
                                                  ("canonical","fixed_E20",ca,cb,permutations[20,pair_index]),
                                                  ("raw","optimal",raw_a,raw_b,raw_p)):
                        for (metric,layer),value in geometric_distances(*a,*b,p).items():
                            index = key_index["|".join((metric,layer,alignment,gauge))]
                            distances[index,checkpoint,ia,ib] = distances[index,checkpoint,ib,ia] = value
                    for method,metric,gauge,label,p_store in (("canonical_alpha","alpha2","canonical","alpha_profile",alpha_permutations),
                                                              ("raw_alpha","alpha2","raw","alpha_profile",raw_alpha_permutations),
                                                              ("snd","snd","raw","snd_profile",snd_permutations)):
                        p = artifacts[checkpoint]["permutations"][(ra["run_id"],rb["run_id"],method)]
                        p_store[checkpoint,pair_index] = p
                        a,b = (ca,cb) if gauge=="canonical" else (raw_a,raw_b)
                        values = pipeline_metrics.layers(*a,*b,p,metric="alpha" if metric=="alpha2" else "snd")
                        for layer,value in values.items():
                            expected = artifacts[checkpoint]["values"][(ra["run_id"],rb["run_id"],method,layer)]
                            max_pipeline_error = max(max_pipeline_error,abs(value-expected))
                            index = key_index["|".join((metric,layer,label,gauge))]
                            distances[index,checkpoint,ia,ib] = distances[index,checkpoint,ib,ia] = value
            if not np.isfinite(distances).all() or np.any(distances < 0): raise ValueError(f"Invalid RDM values: {dataset}")
            if not np.array_equal(distances,distances.swapaxes(-1,-2)) or np.any(np.diagonal(distances,axis1=-2,axis2=-1)):
                raise AssertionError("RDM symmetry/zero-diagonal invariant failed")
            run_ids = np.asarray([r["run_id"] for r in group])
            changes_here,cycles_here = assignment_diagnostics(dataset,run_ids,pairs,permutations)
            changes.extend(changes_here); cycles.extend(cycles_here)
            path = out/"data"/f"{dataset}_distances.npz"
            np.savez_compressed(path,distances=distances,metric_keys=np.asarray(metric_keys),checkpoints=np.arange(21),run_ids=run_ids,
                                seeds=np.asarray([r["seed"] for r in group]),pairs=pairs,permutations=permutations,
                                raw_permutations=raw_permutations,alpha_permutations=alpha_permutations,
                                raw_alpha_permutations=raw_alpha_permutations,snd_permutations=snd_permutations,
                                norms_raw=norms_raw,norms_canonical=norms_canonical,norm_layers=np.asarray(["W0","W1","total"]),
                                source_manifest_sha256=np.asarray(sha256(root/"outputs/manifests/main.json")))
            output_records.append({"dataset":dataset,"path":str(path),"sha256":sha256(path),"shape":list(distances.shape),
                                   "run_ids":run_ids.tolist(),"seeds":[r["seed"] for r in group],"pairs_per_checkpoint":45})
    for name,rows in (("norms",norm_rows),("zero_norm_counts",zero_rows),("assignment_changes",changes),("cycle_consistency",cycles)):
        pd.DataFrame(rows).to_csv(out/"data"/f"{name}.csv",index=False,float_format="%.17g")
    change_summary = pd.DataFrame(changes).groupby(["dataset","checkpoint"])[["change_fraction_vs_E0","change_fraction_vs_E20"]].agg(["mean","min","max"])
    change_summary.columns = ["_".join(columns) for columns in change_summary.columns]
    change_summary.reset_index().to_csv(out/"data/assignment_change_summary.csv",index=False)
    cycle_summary = pd.DataFrame(cycles).groupby(["dataset","checkpoint"]).agg(mean_fraction_consistent=("fraction_consistent","mean"),exactly_consistent_triplets=("exactly_consistent","sum"),n_triplets=("exactly_consistent","size")).reset_index()
    cycle_summary.to_csv(out/"data/cycle_consistency_summary.csv",index=False)
    validation = {"passed": bool(max_pipeline_error < 1e-12 and max_gauge_error < 1e-12 and max_function_error < 1e-10 and max_transpose_error < 1e-12),
                  "source_manifest":str(root/"outputs/manifests/main.json"),"source_manifest_sha256":sha256(root/"outputs/manifests/main.json"),
                  "source_reproduction_validation_sha256":sha256(root/"outputs/validation/checks.json"),
                  "code_sha256":sha256(__file__),"pipeline_metrics_sha256":sha256(pipeline_metrics.__file__),
                  "primary_key":PRIMARY_KEY,"n_metrics":len(metric_keys),"runs":run_audit,"pipeline_sources":pipeline_audit,"outputs":output_records,
                  "max_pipeline_dissimilarity_recompute_error":max_pipeline_error,"max_nondegenerate_gauge_difference":max_gauge_error,
                  "max_gauge_logit_error":max_function_error,"max_primary_reverse_assignment_distance_error":max_transpose_error,
                  "zero_norm_totals":{key:sum(row[key] for row in zero_rows) for key in ("both_positive","incoming_only","outgoing_only","both_zero")},
                  "zero_norm_convention":"Both norms positive: s=sqrt(out/in). Incoming only: s=1/in. Outgoing only: s=out. Both zero: s=1. Transform W0*=s and W1/=s.",
                  "pair_orientation":"p[k] is the neuron in B matched to neuron k in A; B_aligned=(B0[:,p],B1[p,:]); opposite orientation uses inverse permutation",
                  "metric_consistency":"Primary gauge, objective and metric are unchanged over E0..E20. Fixed_E0/E20 reuse each pair's same permutation for BOTH compared RDMs.",
                  "original_results_modified":False}
    save_json(out/"data/validation.json",validation)
    if not validation["passed"]: raise AssertionError("RSA distance validation failed; see 01_RSA/data/validation.json")
    return validation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--out",type=Path,default=ROOT/"01_RSA")
    arguments = parser.parse_args()
    result = build_distances(arguments.root,arguments.out)
    print(json.dumps({"passed":result["passed"],"n_metrics":result["n_metrics"],"datasets":len(result["outputs"])}))
