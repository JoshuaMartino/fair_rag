"""Residual Slate Bias Estimator (paper Sec. 3.2, Eqs. 2-4)."""
import numpy as np


def prefix_shares(groups, n_groups):
    """p_k(a_m): share of group m among the first k items, k=1..K. Returns (K, M)."""
    groups = np.asarray(groups)
    onehot = np.eye(n_groups)[groups]                      # (K, M)
    return np.cumsum(onehot, axis=0) / np.arange(1, len(groups) + 1)[:, None]


def dcg_weights(K):
    return 1.0 / np.log2(np.arange(1, K + 1) + 1.0)        # w_k = 1/log2(k+1)


def ndkl(groups, q, eps=1e-3):
    """Eq. 2-3, implemented exactly as printed (smoothed, un-normalised KL per prefix)."""
    q = np.asarray(q, dtype=float)
    P = prefix_shares(groups, len(q))
    kl = ((P + eps) * np.log((P + eps) / (q[None, :] + eps))).sum(1)
    w = dcg_weights(len(groups))
    return float((w * kl).sum() / w.sum())


def signed_scores(groups, q, eps=1e-3):
    """Eq. 4: S_K[a_m] = (p_K(a_m) - q_m) * NDKL(K). Returns (M,)."""
    q = np.asarray(q, dtype=float)
    pK = prefix_shares(groups, len(q))[-1]
    return (pK - q) * ndkl(groups, q, eps)


def skew_scores(groups, q):
    """Coarse 'skew proxy' used in the Table 5 'Skew' ablation: p_K - q (no NDKL, no prefix sensitivity)."""
    q = np.asarray(q, dtype=float)
    return prefix_shares(groups, len(q))[-1] - q
