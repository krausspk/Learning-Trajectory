"""Analysis-only gauge/permutation comparisons for perturbation forks.

The primary quantity is the Frobenius norm of the concatenated, canonically
balanced matrices after a single shared hidden-neuron assignment.  In this
one-hidden-layer architecture Git Re-Basin's assignment exactly minimizes
this canonical Euclidean objective.  No function here modifies input arrays.

Canonical balancing defines a reproducible representative of positive ReLU
rescalings; we do not claim optimization over all possible gauges in the raw
Euclidean metric.  Thresholds are empirical reference-cohort summaries, not
formal population coverage guarantees.
"""
from __future__ import annotations

import numpy as np

from learning_trajectory.metrics import align, apply_permutation, canonicalize, dissimilarity


def canonical_pair(w0, w1):
    """Return fresh float64 canonical matrices in paper orientation."""
    return canonicalize(w0, w1)


def _aligned_canonical(a, b, method):
    a0, a1 = canonical_pair(*a)
    b0, b1 = canonical_pair(*b)
    permutation = align(a0, a1, b0, b1, method=method)
    b0, b1 = apply_permutation(b0, b1, permutation)
    return (a0, a1), (b0, b1)


def _frobenius(a, b):
    squared = sum(float(np.square(x - y).sum()) for x, y in zip(a, b))
    return float(np.sqrt(squared))


def distance(a, b, method="git_rebasin"):
    """Canonical, jointly permutation-aligned whole-network Frobenius norm.

    ``pif`` requests the pooled-z-score PIF/Hungarian robustness assignment;
    ``canonical_alpha`` requests the profile-Alpha assignment.  These alternate
    assignments need not minimize Euclidean distance and must not be called
    the exact Euclidean quotient optimum.
    """
    ca, cb = _aligned_canonical(a, b, method)
    return _frobenius(ca, cb)


def paired_metrics(a, b):
    """Primary distance and prespecified PIF/Alpha robustness measurements."""
    ca, cb = _aligned_canonical(a, b, "canonical_alpha")
    vector_a = np.concatenate([w.ravel() for w in ca])
    vector_b = np.concatenate([w.ravel() for w in cb])
    return {
        "distance": distance(a, b, "git_rebasin"),
        "pif_distance": distance(a, b, "pif"),
        "alpha": dissimilarity(vector_a, vector_b, metric="alpha"),
        "alpha_distance": _frobenius(ca, cb),
    }


def threshold_calibration(networks, accuracies, losses):
    """Calibrate reference-only region criteria before observing perturbations.

    Each reference contributes its median distance to all other references;
    diagonal/self distances are excluded.  The primary structural threshold
    is the 95th percentile of those medians; the 90th percentile is a declared
    sensitivity analysis.  Functional thresholds use the same independent
    reference cohort.  NumPy's linear empirical quantile convention is fixed.
    """
    networks = list(networks)
    count = len(networks)
    accuracy = np.asarray(accuracies, dtype=np.float64)
    loss = np.asarray(losses, dtype=np.float64)
    if count < 3:
        raise ValueError("Calibration requires at least three independent references")
    if accuracy.shape != (count,) or loss.shape != (count,):
        raise ValueError("Each reference requires one accuracy and one loss")
    if not np.isfinite(accuracy).all() or not np.isfinite(loss).all():
        raise ValueError("Reference performance must be finite")
    if np.any((accuracy < 0) | (accuracy > 1)) or np.any(loss < 0):
        raise ValueError("Invalid reference accuracy or loss")
    pairwise = np.zeros((count, count), dtype=np.float64)
    for i in range(count):
        for j in range(i + 1, count):
            pairwise[i, j] = pairwise[j, i] = distance(networks[i], networks[j])
    medians = np.array([np.median(np.delete(pairwise[i], i)) for i in range(count)])
    return {
        "distance_q95": float(np.quantile(medians, .95, method="linear")),
        "distance_q90": float(np.quantile(medians, .90, method="linear")),
        "accuracy_q05": float(np.quantile(accuracy, .05, method="linear")),
        "loss_q95": float(np.quantile(loss, .95, method="linear")),
        "reference_count": count,
        "structural_statistic": "median distance to independent reference cohort",
        "calibration_statistic": "leave-one-reference-out median distance",
        "quantile_method": "linear",
        "threshold_scope": "empirical reference-cohort criterion; no population coverage guarantee",
        "per_reference": [
            {"index": i, "loo_median_distance": float(medians[i]),
             "accuracy": float(accuracy[i]), "loss": float(loss[i])}
            for i in range(count)
        ],
        "pairwise_distances": pairwise.tolist(),
    }


def region_membership(network, references, accuracy, loss, thresholds, quantile="q95"):
    """Return structural, functional and conjunctive reference-region membership.

    Membership alone is not evidence of return: callers must also retain
    membership immediately after perturbation and distinguish retention from
    leaving followed by re-entry.  ``network`` must be independent of the
    held-out references.  Invalid/nonfinite candidate states fail membership.
    """
    if quantile not in ("q90", "q95"):
        raise ValueError("quantile must be q90 or q95")
    references = list(references)
    if not references:
        raise ValueError("At least one held-out reference is required")
    if len(references) != thresholds["reference_count"]:
        raise ValueError("Reference cohort size differs from threshold calibration")
    distance_threshold = float(thresholds["distance_" + quantile])
    accuracy_threshold = float(thresholds["accuracy_q05"])
    loss_threshold = float(thresholds["loss_q95"])
    if not np.isfinite([distance_threshold, accuracy_threshold, loss_threshold]).all():
        raise ValueError("Region thresholds must be finite")
    finite_network = all(np.isfinite(np.asarray(w)).all() for w in network)
    if finite_network:
        # Degenerate zero-norm neurons have no defined gauge in this protocol.
        try:
            median_distance = float(np.median([distance(network, ref) for ref in references]))
        except ValueError as error:
            if "Canonical gauge requires" not in str(error):
                raise
            median_distance = float("nan")
    else:
        median_distance = float("nan")
    structural = bool(np.isfinite(median_distance) and median_distance <= distance_threshold)
    functional = bool(np.isfinite(accuracy) and np.isfinite(loss)
                      and accuracy >= accuracy_threshold and accuracy <= 1
                      and loss >= 0 and loss <= loss_threshold)
    return {"median_distance": median_distance, "structural": structural,
            "functional": functional, "member": structural and functional,
            "distance_threshold": distance_threshold, "quantile": quantile}
