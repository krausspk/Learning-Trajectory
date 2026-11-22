"""Frozen, seed-blocked decoding of final weights only.

Linear logistic probes use an exact train-Gram feature map; this is a change of
coordinates, not PCA truncation. No earlier weights, optimizer state, seed or
history metadata enter any model. All preprocessing fits on training blocks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.special import expit
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from learning_trajectory.metrics import canonicalize, align, apply_permutation

HORIZONS = (1, 5, 10)
LAMBDAS = (0.0001, 0.001, 0.01, 0.1)
DIAGNOSTIC_NAMES = ("accuracy", "loss", "raw_W0_frob", "raw_W1_frob", "canonical_hidden_rms", "logit_rms")


def digest_array(*arrays):
    h = hashlib.sha256()
    for array in arrays:
        h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def binary_metrics(y, score):
    y = np.asarray(y, dtype=int)
    score = np.asarray(score, dtype=float)
    prediction = (score >= 0).astype(int)
    tn = int(np.sum((y == 0) & (prediction == 0)))
    fp = int(np.sum((y == 0) & (prediction == 1)))
    fn = int(np.sum((y == 1) & (prediction == 0)))
    tp = int(np.sum((y == 1) & (prediction == 1)))
    n0, n1 = tn + fp, tp + fn
    auc = (rankdata(score)[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)
    return {"balanced_accuracy": float((tn / n0 + tp / n1) / 2),
            "accuracy": float(np.mean(y == prediction)), "auc": float(auc),
            "logloss": float(np.mean(np.logaddexp(0, score) - y * score)),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def paired_bootstrap(y, score, n_bootstrap=2000, seed=62005):
    """Resample complete pairs; each bootstrap always retains balanced labels."""
    y, score = np.asarray(y).reshape(-1, 2), np.asarray(score).reshape(-1, 2)
    rng = np.random.RandomState(seed)
    values = np.empty((n_bootstrap, 2))
    for b in range(n_bootstrap):
        selected = rng.randint(0, len(y), len(y))
        result = binary_metrics(y[selected].ravel(), score[selected].ravel())
        values[b] = result["balanced_accuracy"], result["auc"]
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {"ba_ci_low": float(bounds[0, 0]), "ba_ci_high": float(bounds[1, 0]),
            "auc_ci_low": float(bounds[0, 1]), "auc_ci_high": float(bounds[1, 1])}


def swapped_labels(n_blocks, rng):
    flips = rng.randint(0, 2, n_blocks)
    return np.column_stack((flips, 1 - flips)).ravel()


def center_scale_block(x, train_indices):
    """Equalize layer energy using one train-only scale, not per-column scales."""
    x = np.asarray(x, dtype=np.float64)
    mean = x[train_indices].mean(axis=0)
    centered = x - mean
    rms = float(np.sqrt(np.mean(centered[train_indices] ** 2)))
    if rms <= 1e-15:
        rms = 1.0
    result = centered / (rms * np.sqrt(x.shape[1]))
    return result, {"n_features": x.shape[1], "train_rms": rms,
                    "train_mean_sha256": digest_array(mean)}


def standardized_diagnostics(x, train_indices):
    x = np.asarray(x, dtype=np.float64)
    mean = x[train_indices].mean(axis=0)
    scale = x[train_indices].std(axis=0)
    scale[scale <= 1e-15] = 1.0
    return (x - mean) / scale, {"mean": mean.tolist(), "scale": scale.tolist()}


def residualize(x, covariates, train_indices):
    design = np.column_stack((np.ones(len(covariates)), covariates))
    coef, _, rank, _ = np.linalg.lstsq(design[train_indices], x[train_indices], rcond=None)
    return x - design @ coef, {"design_rank": int(rank), "coefficient_sha256": digest_array(coef)}


def train_gram_map(x, train_indices):
    """Exact linear map onto span of training rows, dropping numeric nullspace."""
    train = x[train_indices]
    gram = train @ train.T
    values, vectors = eigh((gram + gram.T) / 2, check_finite=False)
    keep = values > max(float(values[-1]) * 1e-12, 1e-14)
    values, vectors = values[keep], vectors[:, keep]
    mapped = (x @ train.T) @ (vectors / np.sqrt(values))
    error = float(np.max(np.abs(mapped[train_indices] @ mapped[train_indices].T - gram)))
    return mapped, {"rank": len(values), "gram_reconstruction_max_error": error,
                    "n_original_features": x.shape[1]}


def fit_logistic(x, y, regularization, start=None):
    """Mean logistic loss + lambda/2 * ||w||^2; intercept unpenalized."""
    n, d = x.shape
    if start is None:
        start = np.zeros(d + 1)
    def objective(theta):
        score = x @ theta[:-1] + theta[-1]
        residual = (expit(score) - y) / n
        value = np.mean(np.logaddexp(0, score) - y * score) + regularization * np.dot(theta[:-1], theta[:-1]) / 2
        gradient = np.r_[x.T @ residual + regularization * theta[:-1], residual.sum()]
        return value, gradient
    fitted = minimize(objective, start, method="L-BFGS-B", jac=True,
                      options={"maxiter": 600, "ftol": 1e-11, "gtol": 1e-7, "maxls": 30})
    # Precision-limit stops are harmless only when the gradient is small.
    if not fitted.success and np.max(np.abs(fitted.jac)) > 2e-5:
        raise RuntimeError(f"Logistic probe failed to converge: {fitted.message}")
    return fitted.x, {"iterations": int(fitted.nit), "success": bool(fitted.success),
                      "gradient_max": float(np.max(np.abs(fitted.jac)))}


def select_probe(mapped, y, train_indices, val_indices, test_indices, lambdas=LAMBDAS):
    candidates = []
    # Strong-to-weak warm starts improve speed without changing selection.
    start = None
    for regularization in sorted(lambdas, reverse=True):
        theta, diagnostic = fit_logistic(mapped[train_indices], y[train_indices], regularization, start)
        start = theta
        scores = mapped @ theta[:-1] + theta[-1]
        val = binary_metrics(y[val_indices], scores[val_indices])
        candidates.append(((-val["balanced_accuracy"], val["logloss"], list(lambdas).index(regularization)),
                           regularization, theta, scores, val, diagnostic))
    chosen = min(candidates, key=lambda item: item[0])
    return {"lambda": chosen[1], "theta": chosen[2], "scores": chosen[3],
            "validation_ba": chosen[4]["balanced_accuracy"], "validation_logloss": chosen[4]["logloss"],
            "optimizer": chosen[5], "test": binary_metrics(y[test_indices], chosen[3][test_indices]),
            "candidates": [{"lambda": item[1], "validation_ba": item[4]["balanced_accuracy"],
                             "validation_logloss": item[4]["logloss"], **item[5]} for item in candidates]}


def permutation_test(mapped, observed, indices, n_blocks, n_permutations, seed, lambdas=LAMBDAS):
    """Shuffle labels independently within every seed, then refit+reselect."""
    rng = np.random.RandomState(seed)
    null = np.empty(n_permutations)
    for b in range(n_permutations):
        y = swapped_labels(n_blocks, rng)
        fitted = select_probe(mapped, y, *indices, lambdas=lambdas)
        null[b] = fitted["test"]["balanced_accuracy"]
    p = (1 + np.count_nonzero(null >= observed - 1e-12)) / (n_permutations + 1)
    return float(p), null


def load_condition(folder, condition, seeds):
    arrays = {k: [] for k in ("W0", "W1", "accuracy", "loss", "diagnostics", "logits")}
    source = []
    for seed in seeds:
        path = folder / "checkpoints" / condition / f"seed_{seed}.npz"
        with np.load(path, allow_pickle=False) as archive:
            for key in arrays:
                arrays[key].append(archive[key].copy())
        source.append({"seed": int(seed), "file": str(path.relative_to(folder)),
                       "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    for key in arrays:
        value = np.stack(arrays[key])
        arrays[key] = value.reshape((-1,) + value.shape[2:])
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite checkpoint {condition}/{key}")
    assert arrays["W0"].shape == (len(seeds) * 2, 3, 784, 64)
    assert arrays["W1"].shape == (len(seeds) * 2, 3, 64, 10)
    return arrays, source


def wait_for_checkpoints(folder, condition, seeds, timeout=7200.):
    """Wait for atomically published archives; never inspect partial outcomes."""
    if timeout <= 0:
        raise ValueError("Checkpoint wait timeout must be positive")
    started = time.monotonic()
    expected = [folder / "checkpoints" / condition / f"seed_{seed}.npz" for seed in seeds]
    while True:
        missing = [path for path in expected if not path.is_file()]
        elapsed = time.monotonic() - started
        if not missing:
            return {"wait_seconds": elapsed, "expected_files": len(expected)}
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        print(f"{now} Waiting for {condition}: {len(missing)}/{len(expected)} checkpoint files missing; elapsed {elapsed:.0f}s", flush=True)
        if elapsed >= timeout:
            raise TimeoutError(f"Timed out waiting for {condition} after {elapsed:.1f}s; {len(missing)} checkpoint files missing")
        time.sleep(min(30., timeout - elapsed))


def aligned_features(arrays, train_indices, condition, folder, symmetry_seed=62007):
    """The reference is selected by weights alone, without looking at labels."""
    canonical = []
    hashes = []
    for index in train_indices:
        c0, c1 = canonicalize(arrays["W0"][index, 2], arrays["W1"][index, 2])
        canonical.append((c0, c1))
        hashes.append(digest_array(c0, c1))
    selected = int(np.argmin(np.asarray(hashes)))
    reference = canonical[selected]
    reference_index = int(train_indices[selected])
    np.savez_compressed(folder / "analysis" / f"reference_{condition}.npz",
                        W0=reference[0], W1=reference[1], row_index=reference_index,
                        canonical_weights_sha256=hashes[selected])
    result = []
    permutations = np.empty((len(arrays["W0"]), 3, 64), dtype=np.int16)
    rng = np.random.RandomState(symmetry_seed)
    checks = {"permutation_max_weight_error": 0., "gauge_max_weight_error": 0.,
              "aligned_rows_checked": len(arrays["W0"]) * 3}
    for h in range(3):
        f0 = np.empty((len(arrays["W0"]), 784 * 64), dtype=np.float64)
        f1 = np.empty((len(arrays["W0"]), 64 * 10), dtype=np.float64)
        for index in range(len(f0)):
            c0, c1 = canonicalize(arrays["W0"][index, h], arrays["W1"][index, h])
            p = align(*reference, c0, c1, "git_rebasin")
            a0, a1 = apply_permutation(c0, c1, p)
            permutations[index, h] = p
            f0[index], f1[index] = a0.ravel(), a1.ravel()
            # Verify every row under a new hidden reindexing. The check is
            # stronger than testing predictions for one chosen fitted probe.
            q = rng.permutation(64)
            q0, q1 = apply_permutation(c0, c1, q)
            pq = align(*reference, q0, q1, "git_rebasin")
            b0, b1 = apply_permutation(q0, q1, pq)
            err = max(np.max(np.abs(a0 - b0)), np.max(np.abs(a1 - b1)))
            checks["permutation_max_weight_error"] = max(checks["permutation_max_weight_error"], float(err))
            scales = np.exp(rng.uniform(-1, 1, 64))
            g0, g1 = canonicalize(c0 * scales, c1 / scales[:, None])
            pg = align(*reference, g0, g1, "git_rebasin")
            g0, g1 = apply_permutation(g0, g1, pg)
            err = max(np.max(np.abs(a0 - g0)), np.max(np.abs(a1 - g1)))
            checks["gauge_max_weight_error"] = max(checks["gauge_max_weight_error"], float(err))
        result.append((f0, f1))
    if checks["permutation_max_weight_error"] > 1e-9 or checks["gauge_max_weight_error"] > 1e-9:
        raise AssertionError(f"Alignment symmetry check failed: {checks}")
    np.savez_compressed(folder / "analysis" / f"alignments_{condition}.npz", permutations=permutations)
    return result, {"reference_row_index": reference_index,
                    "reference_selection": "minimum SHA256 of canonical C10 final weights among training rows; no labels used",
                    "reference_sha256": hashes[selected], "symmetry_checks": checks}


def performance_tables(arrays, condition, split_indices, bootstrap_seed=62005, n_bootstrap=2000):
    summaries, paired = [], []
    rng = np.random.RandomState(bootstrap_seed)
    for h, horizon in enumerate(HORIZONS):
        for split, indices in {"all": np.arange(len(arrays["accuracy"])), **split_indices}.items():
            for history in (0, 1):
                chosen = indices[indices % 2 == history]
                summaries.append({"condition": condition, "c_epochs": horizon, "split": split,
                                  "history": history, "n": len(chosen),
                                  "accuracy_mean": float(arrays["accuracy"][chosen, h].mean()),
                                  "accuracy_sd": float(arrays["accuracy"][chosen, h].std(ddof=1)),
                                  "loss_mean": float(arrays["loss"][chosen, h].mean()),
                                  "loss_sd": float(arrays["loss"][chosen, h].std(ddof=1))})
            for metric in ("accuracy", "loss"):
                values = arrays[metric][indices, h].reshape(-1, 2)
                differences = values[:, 1] - values[:, 0]
                resamples = rng.randint(0, len(differences), (n_bootstrap, len(differences)))
                lower, upper = np.quantile(differences[resamples].mean(axis=1), [0.025, 0.975])
                paired.append({"condition": condition, "c_epochs": horizon, "split": split,
                               "metric": metric, "n_blocks": len(differences),
                               "diff_H1_minus_H0": float(differences.mean()),
                               "ci_low": float(lower), "ci_high": float(upper)})
    return summaries, paired


def holm_adjust(rows):
    chosen = [i for i, row in enumerate(rows) if not row["primary"]]
    order = sorted(chosen, key=lambda i: rows[i]["permutation_p"])
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * rows[index]["permutation_p"]))
        rows[index]["secondary_holm_p"] = running
    for row in rows:
        row.setdefault("secondary_holm_p", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--conditions", nargs="+", default=["reset", "continued"])
    parser.add_argument("--primary-permutations", type=int, default=999)
    parser.add_argument("--secondary-permutations", type=int, default=999)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--wait-for-checkpoints", action="store_true",
                        help="Wait before each condition until every expected atomic NPZ exists")
    parser.add_argument("--checkpoint-wait-timeout", type=float, default=7200.,
                        help="Maximum checkpoint wait per condition, in seconds (default 7200)")
    args = parser.parse_args()
    folder = args.output.resolve()
    (folder / "analysis").mkdir(exist_ok=True)
    splits = json.loads((folder / "probe_splits.json").read_text())
    plan = json.loads((folder / "plan.json").read_text())
    if tuple(plan["probe"]["lambda_grid"]) != LAMBDAS:
        raise ValueError("Frozen lambda grid differs from implementation")
    if (args.primary_permutations != plan["inference"]["primary_permutations"] or
            args.secondary_permutations != plan["inference"]["secondary_permutations"] or
            args.bootstrap != plan["inference"]["bootstrap_replicates"]):
        raise ValueError("Inference counts must match the frozen analysis plan")
    permutation_seed = int(splits["permutation_seed"])
    bootstrap_seed = int(splits["bootstrap_seed"])
    negative_seed = int(splits["negative_control_seed"])
    probe_seed = int(splits["probe_seed"])
    split_seeds = {key: splits[f"{key}_seeds"] for key in ("train", "val", "test")}
    seeds = sorted(sum(split_seeds.values(), []))
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seed splits overlap")
    seed_index = {seed: i for i, seed in enumerate(seeds)}
    split_indices = {key: np.array([2 * seed_index[seed] + h for seed in sorted(values) for h in (0, 1)])
                     for key, values in split_seeds.items()}
    indices = tuple(split_indices[key] for key in ("train", "val", "test"))
    train, val, test = indices
    true_y = np.tile([0, 1], len(seeds))
    settings = {"lambdas": list(LAMBDAS), "n_primary_permutations": args.primary_permutations,
                "n_secondary_permutations": args.secondary_permutations, "n_bootstrap": args.bootstrap,
                "permutation_seed": permutation_seed, "bootstrap_seed": bootstrap_seed, "negative_label_seed": negative_seed,
                "bootstrap_scope": "test seed pairs conditional on frozen fitted probe",
                "hyperparameter_selection": "validation BA, then validation logloss, then lambda list order; no refit on validation",
                "primary": "reset/both/C10/balanced_accuracy", "feature_input": "final endpoint weights only; diagnostics/logits explicit secondary baselines",
                "covariate_adjustment": "train-only OLS residualization of aligned standardized weights against six standardized diagnostics and intercept",
                "permutation_method": "independent within-pair swaps across all train/validation/test blocks, complete refit and validation lambda reselection",
                "multiple_testing": "one unadjusted primary; Holm across all 19 secondary endpoints including negative control",
                "split_sha256": hashlib.sha256((folder / "probe_splits.json").read_bytes()).hexdigest(),
                "plan_sha256": hashlib.sha256((folder / "plan.json").read_bytes()).hexdigest(),
                "probe_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    settings["orchestration"] = {"wait_for_checkpoints": args.wait_for_checkpoints,
                                  "checkpoint_wait_timeout_seconds": args.checkpoint_wait_timeout}
    write_json(folder / "analysis" / "frozen_probe_settings.json", settings)
    rows, predictions, summary_rows, paired_rows, details = [], [], [], [], {}
    for condition in args.conditions:
        started = time.monotonic()
        if args.wait_for_checkpoints:
            wait_for_checkpoints(folder, condition, seeds, args.checkpoint_wait_timeout)
        print(f"Loading {condition}", flush=True)
        arrays, sources = load_condition(folder, condition, seeds)
        aligned, alignment_meta = aligned_features(arrays, train, condition, folder, probe_seed)
        details[condition] = {**alignment_meta, "source_files": sources, "endpoints": []}
        summary, paired = performance_tables(arrays, condition, split_indices, bootstrap_seed, args.bootstrap)
        summary_rows.extend(summary)
        paired_rows.extend(paired)
        for h, horizon in enumerate(HORIZONS):
            if condition == "continued" and horizon != 10:
                continue
            x0, scale0 = center_scale_block(aligned[h][0], train)
            x1, scale1 = center_scale_block(aligned[h][1], train)
            both = np.column_stack((x0, x1)) / np.sqrt(2)
            diag, diag_meta = standardized_diagnostics(arrays["diagnostics"][:, h], train)
            residual, residual_meta = residualize(both, diag, train)
            functional, function_meta = center_scale_block(arrays["logits"][:, h].reshape(len(seeds) * 2, -1), train)
            features = {"W0": x0, "W1": x1, "both": both,
                        "diagnostics": diag / np.sqrt(diag.shape[1]),
                        "functional_logits": functional, "covariate_adjusted_both": residual}
            endpoint_names = ["both"] if condition == "continued" else list(features)
            if condition == "reset" and horizon == 10:
                endpoint_names.append("negative_labels")
            mapped_cache = {}
            for feature in endpoint_names:
                negative = feature == "negative_labels"
                actual_feature = "both" if negative else feature
                primary = condition == "reset" and feature == "both" and horizon == 10
                if actual_feature not in mapped_cache:
                    mapped_cache[actual_feature] = train_gram_map(features[actual_feature], train)
                mapped, gram_meta = mapped_cache[actual_feature]
                y = swapped_labels(len(seeds), np.random.RandomState(negative_seed)) if negative else true_y
                fitted = select_probe(mapped, y, *indices)
                result = fitted["test"]
                n_perm = args.primary_permutations if primary else args.secondary_permutations
                p, null = permutation_test(mapped, result["balanced_accuracy"], indices, len(seeds), n_perm, permutation_seed)
                confidence = paired_bootstrap(y[test], fitted["scores"][test], args.bootstrap, bootstrap_seed)
                endpoint = f"{condition}_{feature}_C{horizon}"
                np.savez_compressed(folder / "analysis" / f"{endpoint}.npz",
                                    seed=np.repeat(seeds, 2), history=true_y, target=y,
                                    score=fitted["scores"], train_indices=train, val_indices=val, test_indices=test,
                                    permutation_balanced_accuracy=null, coefficient=fitted["theta"])
                row = {"condition": condition, "features": actual_feature,
                       "layer": actual_feature if actual_feature in ("W0", "W1") else "both" if "both" in actual_feature else "none",
                       "c_epochs": horizon, "primary": primary,
                       "n_train_blocks": len(train) // 2, "n_val_blocks": len(val) // 2, "n_test_blocks": len(test) // 2,
                       "balanced_accuracy": result["balanced_accuracy"], "accuracy": result["accuracy"], "auc": result["auc"],
                       **confidence, "effect_above_chance": result["balanced_accuracy"] - 0.5,
                       "permutation_p": p, "n_permutations": n_perm, "lambda": fitted["lambda"],
                       "validation_ba": fitted["validation_ba"], "validation_logloss": fitted["validation_logloss"],
                       "tn": result["tn"], "fp": result["fp"], "fn": result["fn"], "tp": result["tp"],
                       "negative_control": negative}
                rows.append(row)
                for split, selected in split_indices.items():
                    for index in selected:
                        predictions.append({"endpoint": endpoint, "condition": condition, "features": actual_feature,
                                            "c_epochs": horizon, "negative_control": negative, "split": split,
                                            "seed": seeds[index // 2], "history": int(true_y[index]), "target": int(y[index]),
                                            "score": float(fitted["scores"][index]),
                                            "probability_H1": float(expit(fitted["scores"][index])),
                                            "prediction": int(fitted["scores"][index] >= 0)})
                details[condition]["endpoints"].append({"endpoint": endpoint, "gram": gram_meta,
                    "layer_scales": {"W0": scale0, "W1": scale1}, "diagnostic_scaling": diag_meta,
                    "functional_scale": function_meta, "residualization": residual_meta,
                    "selected_optimizer": fitted["optimizer"], "validation_candidates": fitted["candidates"],
                    "permutation_null_mean": float(null.mean()), "permutation_null_sd": float(null.std(ddof=1))})
                print(f"{endpoint}: BA={result['balanced_accuracy']:.4f} AUC={result['auc']:.4f} p={p:.4g}", flush=True)
                holm_adjust(rows)
                write_csv(folder / "history_probe_results.csv", rows)
                write_csv(folder / "history_probe_predictions.csv", predictions)
                write_csv(folder / "performance_summary.csv", summary_rows)
                write_csv(folder / "performance_paired.csv", paired_rows)
                write_json(folder / "probe_diagnostics.json", {"settings": settings, "conditions": details})
        print(f"{condition} finished in {time.monotonic() - started:.1f}s", flush=True)
        del arrays, aligned
    print("All probes complete", flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
