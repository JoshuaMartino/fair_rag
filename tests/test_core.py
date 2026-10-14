import numpy as np, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fairrag.rerank import fgr, det_const_sort
from fairrag.rsbe import ndkl, signed_scores
from fairrag.cglc import CGLC, Features
from fairrag.metrics import rd_one, retrieval_utility

def test_fgr_balances_prefixes():
    g = np.array([0]*80 + [1]*20)                      # skewed pool, relevance order = index
    q = np.array([.5, .5]); s = fgr(g, q, 10)
    assert len(set(s)) == 10
    assert abs(g[s].mean() - 0.5) <= 0.1
    assert all(abs(g[s[:k]].mean() - .5) <= .5 for k in range(1, 11))

def test_fgr_improves_ndkl_over_vanilla():
    rng = np.random.default_rng(0); g = (rng.random(100) < .2).astype(int)
    q = np.array([.5, .5])
    assert ndkl(g[fgr(g, q, 15)], q) < ndkl(g[:15], q)

def test_detconst_min_counts():
    g = np.array([0]*60 + [1]*40); q = np.array([.5, .5]); s = det_const_sort(g, q, 10)
    assert len(s) == 10 and g[s].sum() >= 4

def test_signed_sign():
    q = np.array([.5, .5]); S = signed_scores(np.zeros(10, int), q)
    assert S[0] > 0 > S[1]

def test_cglc_reduces_rd():
    rng = np.random.default_rng(0); N = 2000
    g = rng.integers(0, 2, N); y = rng.integers(0, 2, N); occ = np.zeros(N, int)
    lt = rng.normal(0, 1, N) + 1.0 * y + 0.8 * g           # target logit biased toward g=1
    logits = np.stack([lt, rng.normal(0, 1, N), rng.normal(0, 1, N)], 1)
    S = rng.normal(0, 1, (N, 2)); f = Features(logits, g, y, occ, S)
    c = CGLC(1, seed=0); before = rd_one(logits.argmax(1) == 0, g)
    c.fit(f, n_random=200, verbose=False)
    assert rd_one(c.predict(f), g) < before

def test_retrieval_utility():
    m = retrieval_utility(np.array([[0, 1, 0], [0, 0, 0]]))
    assert abs(m["hit"] - .5) < 1e-9 and abs(m["mrr"] - .25) < 1e-9
