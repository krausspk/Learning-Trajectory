"""Permutation-aware comparisons from Krauss et al. (2026), §§2.5–2.13.

Weights use the paper's orientation: W0[input, hidden], W1[hidden, output].
A permutation p maps B onto A: compare A with (B0[:, p], B1[p, :]).
All analysis arithmetic is float64; the training checkpoints remain float32.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

EPS = 1e-12


def _matrix(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or not np.isfinite(a).all():
        raise ValueError("Expected a finite two-dimensional array")
    return a


def _network(w0, w1):
    w0, w1 = _matrix(w0), _matrix(w1)
    if w0.shape[1] != w1.shape[0]:
        raise ValueError("Inconsistent hidden-layer widths")
    return w0, w1


def canonicalize(W0, W1):
    """Equalize each hidden neuron's incoming/outgoing L2 norms (Eq. 19).

    Degenerate neurons are rejected: the paper's canonical convention is
    defined only when both norms are nonzero (all reported neurons qualified).
    """
    W0, W1 = _network(W0, W1)
    a, b = np.linalg.norm(W0, axis=0), np.linalg.norm(W1, axis=1)
    if np.any(a == 0) or np.any(b == 0):
        raise ValueError("Canonical gauge requires nonzero incoming and outgoing norms")
    s = np.sqrt(b / a)
    return W0 * s[None, :], W1 / s[:, None]


def apply_permutation(W0, W1, p):
    W0, W1 = _network(W0, W1)
    p = np.asarray(p, dtype=int)
    if p.shape != (W0.shape[1],) or not np.array_equal(np.sort(p), np.arange(len(p))):
        raise ValueError("p must be a permutation of all hidden-neuron indices")
    return W0[:, p], W1[p, :]


def _metric_name(metric):
    name = str(metric).lower()
    aliases = {"raw_alpha": "alpha", "canonical_alpha": "alpha", "rms": "euclidean", "rmse": "euclidean"}
    name = aliases.get(name, name)
    if name not in ("alpha", "euclidean", "snd"):
        raise ValueError(f"Unknown dissimilarity: {metric}")
    return name


def dissimilarity(A, B, metric="alpha", alpha=2):
    """Matrix-level Alpha, RMS Euclidean, or SND (Eqs. 11–14)."""
    A, B = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
    if A.shape != B.shape or A.size == 0 or not (np.isfinite(A).all() and np.isfinite(B).all()):
        raise ValueError("Arrays must have equal nonempty shapes and finite values")
    metric = _metric_name(metric)
    if metric == "euclidean":
        return float(np.sqrt(np.mean(np.square(A - B))))
    if metric == "snd":
        denominator = np.abs(A + B)
        same = (np.sign(A) == np.sign(B)) & (A != 0) & (denominator > EPS)
        contribution = np.ones(A.shape, dtype=np.float64)
        np.divide(np.abs(A - B), denominator, out=contribution, where=same)
        contribution[(A == 0) & (B == 0)] = 0
        return float(np.mean(contribution))
    if alpha <= 0:
        raise ValueError("Alpha exponent must be positive")
    importance = np.abs(A * B) ** alpha
    total = importance.sum()
    if total <= EPS:
        return 0.0
    return float(importance[np.sign(A) != np.sign(B)].sum() / total)


def _profile_cost(A, B, metric="alpha", alpha=2):
    """All pair costs for neuron profiles stored as rows; memory is bounded."""
    A, B = _matrix(A), _matrix(B)
    if A.shape[1] != B.shape[1] or A.shape[1] == 0:
        raise ValueError("Neuron profiles must share their nonempty feature dimension")
    metric = _metric_name(metric)
    if metric == "euclidean":
        return cdist(A, B, metric="euclidean") / np.sqrt(A.shape[1])
    if metric == "alpha":
        if alpha <= 0:
            raise ValueError("Alpha exponent must be positive")
        aa, bb = np.abs(A) ** alpha, np.abs(B) ** alpha
        denominator = aa @ bb.T
        ap, an = aa * (A > 0), aa * (A < 0)
        bp, bn = bb * (B > 0), bb * (B < 0)
        numerator = ap @ bn.T + an @ bp.T
        result = np.zeros_like(denominator)
        np.divide(numerator, denominator, out=result, where=denominator > EPS)
        return np.clip(result, 0, 1)
    # Only one A row is broadcast against B, avoiding an n*n*features tensor.
    result = np.empty((len(A), len(B)))
    for i, a in enumerate(A):
        denominator = np.abs(a[None, :] + B)
        same = (np.sign(a)[None, :] == np.sign(B)) & (a[None, :] != 0) & (denominator > EPS)
        contribution = np.ones_like(B)
        np.divide(np.abs(a[None, :] - B), denominator, out=contribution, where=same)
        contribution[(a[None, :] == 0) & (B == 0)] = 0
        result[i] = contribution.mean(axis=1)
    return result


def profile_cost(W0a, W1a, W0b, W1b, metric="alpha", alpha=2):
    """Full incoming-plus-outgoing neuron profile cost (Eq. 9)."""
    W0a, W1a = _network(W0a, W1a)
    W0b, W1b = _network(W0b, W1b)
    return _profile_cost(np.concatenate((W0a.T, W1a), axis=1),
                         np.concatenate((W0b.T, W1b), axis=1), metric, alpha)


def incoming_cost(W0a, W0b, metric="alpha", alpha=2):
    """W0-only profile costs for the identity/parity task control."""
    return _profile_cost(_matrix(W0a).T, _matrix(W0b).T, metric, alpha)


def _assignment(cost):
    if cost.shape[0] != cost.shape[1]:
        raise ValueError("Matching requires equal hidden widths")
    rows, cols = linear_sum_assignment(cost)
    p = np.empty(len(rows), dtype=np.int64)
    p[rows] = cols
    return p


def align_incoming(W0a, W0b, metric="alpha", alpha=2):
    return _assignment(incoming_cost(W0a, W0b, metric, alpha))


def pif_features(W0, W1):
    """Eight permutation-invariant features in the order of Eq. 8."""
    W0, W1 = _network(W0, W1)
    return np.column_stack((W0.sum(axis=0), np.count_nonzero(W0, axis=0),
                            (W0 > 0).sum(axis=0), np.linalg.norm(W0, axis=0),
                            W1.sum(axis=1), np.count_nonzero(W1, axis=1),
                            (W1 > 0).sum(axis=1), np.linalg.norm(W1, axis=1)))


def pif_assignment(W0a, W1a, W0b, W1b):
    """Coarse PIF-only assignment, before profile refinement."""
    a, b = pif_features(W0a, W1a), pif_features(W0b, W1b)
    pooled = np.concatenate((a, b))
    mean, std = pooled.mean(axis=0), pooled.std(axis=0)
    std[std == 0] = 1.0
    return _assignment(cdist((a - mean) / std, (b - mean) / std))


def align(W0a, W1a, W0b, W1b, method="canonical_alpha"):
    """Infer the one-hidden-layer assignment specified in the paper.

    ``pif`` denotes the coarse PIF candidate alone. ``raw_alpha``, ``snd``
    and ``euclidean`` include PIF presorting followed by exact profile
    assignment. Presorting cannot change the optimum, but can resolve ties
    differently. ``benchmark`` compares coarse canonical-PIF and direct
    canonical-Alpha candidates using size-weighted canonical Alpha.
    """
    W0a, W1a = _network(W0a, W1a)
    W0b, W1b = _network(W0b, W1b)
    if W0a.shape != W0b.shape or W1a.shape != W1b.shape:
        raise ValueError("Whole-network matching requires identical architectures")
    if method in ("identity", "unaligned", "unmatched"):
        return np.arange(W0a.shape[1])
    if method == "git_rebasin":
        return _assignment(-(W0a.T @ W0b + W1a @ W1b.T))
    if method == "pif":
        return pif_assignment(W0a, W1a, W0b, W1b)
    if method in ("canonical_alpha", "benchmark"):
        a0, a1 = canonicalize(W0a, W1a)
        b0, b1 = canonicalize(W0b, W1b)
        direct = _assignment(profile_cost(a0, a1, b0, b1, "alpha"))
        if method == "canonical_alpha":
            return direct
        pif = pif_assignment(a0, a1, b0, b1)
        candidates = (pif, direct)
        values = [layers(a0, a1, b0, b1, p, "alpha")["size_weighted"] for p in candidates]
        return candidates[int(np.argmin(values))]
    if method in ("raw_alpha", "alpha", "euclidean", "snd"):
        pif = pif_assignment(W0a, W1a, W0b, W1b)
        b0, b1 = apply_permutation(W0b, W1b, pif)
        refined = _assignment(profile_cost(W0a, W1a, b0, b1, _metric_name(method)))
        return pif[refined]
    raise ValueError(f"Unknown alignment method: {method}")


def layers(W0a, W1a, W0b, W1b, p=None, metric="alpha", canonical=False):
    """Layerwise and two aggregation conventions (Eqs. 15 and 16)."""
    W0a, W1a = _network(W0a, W1a)
    W0b, W1b = _network(W0b, W1b)
    if canonical:
        W0a, W1a = canonicalize(W0a, W1a)
        W0b, W1b = canonicalize(W0b, W1b)
    if p is not None:
        W0b, W1b = apply_permutation(W0b, W1b, p)
    d0, d1 = dissimilarity(W0a, W0b, metric), dissimilarity(W1a, W1b, metric)
    return {"W0": d0, "W1": d1, "balanced": (d0 + d1) / 2,
            "size_weighted": (W0a.size * d0 + W1a.size * d1) / (W0a.size + W1a.size)}


def weight_agreement(W0a, W1a, W0b, W1b, p=None):
    """Raw total inner product, cosine, and RMS for the alignment benchmark."""
    W0a, W1a = _network(W0a, W1a)
    W0b, W1b = _network(W0b, W1b)
    if p is not None:
        W0b, W1b = apply_permutation(W0b, W1b, p)
    a, b = np.concatenate((W0a.ravel(), W1a.ravel())), np.concatenate((W0b.ravel(), W1b.ravel()))
    inner = float(np.dot(a, b))
    norm = float(np.linalg.norm(a) * np.linalg.norm(b))
    return {"inner_product": inner, "cosine": inner / norm if norm else float("nan"),
            "rmse": dissimilarity(a, b, "euclidean")}


def linear_cka(Ha, Hb):
    """Biased linear CKA with sample-centered features (Eq. 21)."""
    Ha, Hb = _matrix(Ha), _matrix(Hb)
    if Ha.shape[0] != Hb.shape[0]:
        raise ValueError("CKA requires identical probe samples")
    a, b = Ha - Ha.mean(axis=0), Hb - Hb.mean(axis=0)
    numerator = np.square(a.T @ b).sum()
    denominator = np.linalg.norm(a.T @ a) * np.linalg.norm(b.T @ b)
    if denominator == 0:
        raise ValueError("CKA is undefined for a representation with zero centered variance")
    return float(np.clip(numerator / denominator, 0, 1))


cka = linear_cka


def normalized_activations(H):
    """Center/unit-normalize neurons; dead or constant columns become zero.

    Zero correlation for a constant neuron is an explicit numerical convention
    because Pearson correlation itself is undefined in that case.
    """
    H = _matrix(H)
    centered = H - H.mean(axis=0)
    norms = np.linalg.norm(centered, axis=0)
    return np.divide(centered, norms, out=np.zeros_like(centered), where=norms > 0)


def activation_correspondence(Ha, Hb, p=None):
    """Mean matched-neuron Pearson correlation (Eq. 22)."""
    a, b = normalized_activations(Ha), normalized_activations(Hb)
    if a.shape != b.shape:
        raise ValueError("Activation correspondence requires equal matrix shapes")
    if p is not None:
        p = np.asarray(p, dtype=int)
        if not np.array_equal(np.sort(p), np.arange(a.shape[1])):
            raise ValueError("p must permute the hidden-neuron indices")
        b = b[:, p]
    return float(np.sum(a * b, axis=0).mean())
