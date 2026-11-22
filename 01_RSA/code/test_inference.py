"""Independent checks of network-identity RSA inference and partial ranks."""
import importlib.util
import itertools
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

spec = importlib.util.spec_from_file_location("rsa_inference", Path(__file__).with_name("inference.py"))
inference = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inference)


def test_identity_permutations_match_simultaneous_rdm_rows_columns():
    pairs = np.array(list(itertools.combinations(range(4), 2)))
    permutations = np.array(list(itertools.permutations(range(4))))
    # Ties ensure rank averaging, rather than argsort-based ordinal ranks.
    a = np.array([0.2, 0.8, 0.8, 0.4, 1.1, 0.5])
    b = np.array([0.7, 0.3, 0.3, 1.2, 0.4, 0.8])
    rdm = np.zeros((4, 4)); rdm[pairs[:, 0], pairs[:, 1]] = b; rdm += rdm.T
    edges = inference.identity_edge_indices(permutations, pairs, 4)
    assert np.array_equal(edges[0], np.arange(6))
    fast = inference.normalized_ranks(b)[edges] @ inference.normalized_ranks(a)
    slow = []
    for p in permutations:
        permuted = rdm[np.ix_(p, p)]
        slow.append(spearmanr(a, permuted[pairs[:, 0], pairs[:, 1]]).statistic)
    np.testing.assert_allclose(fast, slow, atol=1e-14)
    assert abs(fast.mean()) < 1e-14


def test_jackknife_deletes_entire_network_and_reranks():
    n = 6
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    rng = np.random.RandomState(18)
    a = rng.uniform(size=(n, n)); a = (a+a.T)/2
    b = rng.uniform(size=(n, n)); b = (b+b.T)/2
    initial, final = a[pairs[:, 0], pairs[:, 1]], b[pairs[:, 0], pairs[:, 1]]
    result = inference.jackknife_rsa(initial, final, pairs, n)
    independent = []
    for deleted in range(n):
        keep = np.arange(n) != deleted
        ai, bi = a[np.ix_(keep, keep)], b[np.ix_(keep, keep)]
        iu = np.triu_indices(n-1, 1)
        independent.append(spearmanr(ai[iu], bi[iu]).statistic)
    np.testing.assert_allclose(result, independent, atol=1e-14)
    rho = spearmanr(initial, final).statistic
    se, lo, hi = inference.jackknife_interval(rho, result)
    expected_se = np.sqrt((n-1)/n*np.sum((result-result.mean())**2))
    np.testing.assert_allclose(se, expected_se)
    np.testing.assert_allclose([lo, hi], [rho-inference.Z95*se, rho+inference.Z95*se])


def test_partial_spearman_and_permutations_match_explicit_rank_regression():
    n = 6; rng = np.random.RandomState(19)
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    x, y = rng.normal(size=(2, len(pairs)))
    cov = rng.normal(size=(len(pairs), 2))
    mappings = np.array([rng.permutation(n) for _ in range(12)])
    edges = inference.identity_edge_indices(mappings, pairs, n)
    def independent(x, y, cov):
        design = np.column_stack([np.ones(len(x)), rankdata(cov[:, 0]), rankdata(cov[:, 1])])
        xr, yr = rankdata(x), rankdata(y)
        rx = xr - design @ np.linalg.lstsq(design, xr, rcond=None)[0]
        ry = yr - design @ np.linalg.lstsq(design, yr, rcond=None)[0]
        return np.corrcoef(rx, ry)[0, 1]
    np.testing.assert_allclose(inference.partial_spearman(x, y, cov), independent(x, y, cov), atol=1e-13)
    null, rank = inference._partial_null(x, y, cov, edges)
    assert rank == 3
    np.testing.assert_allclose(null, [independent(x, y[e], cov) for e in edges], atol=1e-12)


def test_monte_carlo_add_one_and_holm():
    null = np.array([-.5, -.1, 0., .1, .5])
    assert inference.permutation_pvalues(.5, null) == (2/6, 3/6)
    np.testing.assert_allclose(inference.holm_adjust([.01, .04, .03]), [.03, .06, .06])
