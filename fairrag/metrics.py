"""Retriever / generator utility + fairness metrics (App. C)."""
import numpy as np
from .rsbe import prefix_shares, ndkl


def retrieval_utility(rel_matrix):
    """rel_matrix: (N, K) binary relevance of each ranked slate item. Returns Hit@K, MRR, NDCG@K."""
    R = np.asarray(rel_matrix, float)
    N, K = R.shape
    hit = (R.max(1) > 0).mean()
    first = np.where(R.max(1) > 0, R.argmax(1) + 1, np.inf)
    mrr = np.where(np.isinf(first), 0.0, 1.0 / first).mean()
    disc = 1.0 / np.log2(np.arange(1, K + 1) + 1)
    dcg = (R * disc).sum(1)
    idcg = (-np.sort(-R, axis=1) * disc).sum(1)
    ndcg = np.where(idcg > 0, dcg / np.maximum(idcg, 1e-12), 0.0).mean()
    return {"hit": hit, "mrr": mrr, "ndcg": ndcg}


def skew_min_max(groups, eps=1e-3):
    """Appendix C.2: min / max over prefixes t<=K of log((p_t(a0)+eps)/(p_t(a1)+eps)) for binary groups."""
    P = prefix_shares(groups, 2)
    s = np.log((P[:, 0] + eps) / (P[:, 1] + eps))
    return s.min(), s.max()


def retrieval_fairness(slate_groups, q, eps=1e-3):
    mm = np.array([skew_min_max(g, eps) for g in slate_groups])
    nd = np.mean([ndkl(g, q, eps) for g in slate_groups])
    return {"min_skew": mm[:, 0].mean(), "max_skew": mm[:, 1].mean(), "ndkl": nd}


def _prf(pred, y):
    tp = (pred & (y == 1)).sum(); fp = (pred & (y == 0)).sum(); fn = (~pred & (y == 1)).sum()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def rd_one(pred, g):
    a, b = pred[g == 0], pred[g == 1]
    return abs((a.mean() if len(a) else 0.0) - (b.mean() if len(b) else 0.0))


def tpr_gap_one(pred, g, y):
    m = y == 1
    return rd_one(pred[m], g[m])


def generator_metrics(pred, y, g, occ):
    """Per-occupation one-vs-rest binary metrics, macro-averaged (Sec. 4.2/C.3/C.4)."""
    pred = np.asarray(pred, bool); y = np.asarray(y); g = np.asarray(g); occ = np.asarray(occ)
    rows = []
    for o in np.unique(occ):
        m = occ == o
        p, r, f = _prf(pred[m], y[m])
        rows.append((float((pred[m] == (y[m] == 1)).mean()), p, r, f, rd_one(pred[m], g[m]),
                     tpr_gap_one(pred[m], g[m], y[m])))
    a = np.mean(rows, axis=0)
    return dict(zip(["acc", "prec", "rec", "f1", "rd", "tpr_gap"], a.tolist()))
