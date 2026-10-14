"""Fair Greedy Reranker (Alg. 1, Sec. 3.1) and DetConstSort (Geyik et al. 2019) baseline."""
import math
import numpy as np


def fgr(groups, q, K, fallback="paper"):
    """Fair Greedy Reranker.

    groups : group id of each candidate in the pool S_T, ordered by relevance (best first)
    q      : target distribution over groups
    returns: indices into the pool (length min(K, T)) forming the fair slate S_K.

    U_k[a] = q_a*k - #{j<=k-1 : g_j = a}            (Eq. 1)
    paper    : take first remaining doc whose group is in argmax_a U_k[a]; if none, first remaining doc.
    next_deficit (variant): walk groups by descending deficit until one has a remaining doc.
    """
    groups = np.asarray(groups); q = np.asarray(q, float); M = len(q)
    remaining = list(range(len(groups)))
    count = np.zeros(M); picked = []
    for k in range(1, min(K, len(groups)) + 1):
        U = q * k - count
        chosen = None
        if fallback == "paper":
            top = set(np.flatnonzero(U >= U.max() - 1e-12).tolist())
            for pos, idx in enumerate(remaining):
                if groups[idx] in top:
                    chosen = pos; break
        else:
            for a in np.argsort(-U, kind="stable"):
                for pos, idx in enumerate(remaining):
                    if groups[idx] == a:
                        chosen = pos; break
                if chosen is not None:
                    break
        if chosen is None:                       # "context not found" branch (line 5-7)
            chosen = 0
        idx = remaining.pop(chosen)
        picked.append(idx); count[groups[idx]] += 1
    return picked


def det_const_sort(groups, q, K):
    """Deterministic constrained sorting (Geyik et al., 2019, Alg. DetConstSort) with min-count
    constraints floor(k*q_a). Candidates are indices in the pool (rank == index == relevance order)."""
    groups = np.asarray(groups); q = np.asarray(q, float); M = len(q)
    K = min(K, len(groups))
    cand = {a: [i for i in range(len(groups)) if groups[i] == a] for a in range(M)}   # best-first
    counts = np.zeros(M, int); min_counts = np.zeros(M, int)
    ranked, max_idx = [], []
    last_empty = 0; k = 0
    while last_empty < K and k < 10 * K + len(groups):
        k += 1
        tmp_min = np.array([math.floor(k * q[a] + 1e-9) for a in range(M)])
        changed = [a for a in range(M) if tmp_min[a] > min_counts[a]]
        if changed:
            changed.sort(key=lambda a: cand[a][counts[a]] if counts[a] < len(cand[a]) else 1e18)
            for a in changed:
                if counts[a] >= len(cand[a]) or last_empty >= K:
                    continue
                ranked.append(cand[a][counts[a]]); max_idx.append(k); last_empty += 1
                start = last_empty - 1
                while start > 0 and max_idx[start - 1] >= start + 1 and ranked[start - 1] > ranked[start]:
                    ranked[start - 1], ranked[start] = ranked[start], ranked[start - 1]
                    max_idx[start - 1], max_idx[start] = max_idx[start], max_idx[start - 1]
                    start -= 1
                counts[a] += 1
            min_counts = tmp_min
    if len(ranked) < K:                           # constraints exhausted -> fill by relevance
        rest = [i for i in range(len(groups)) if i not in set(ranked)]
        ranked += rest[: K - len(ranked)]
    return ranked[:K]


def vanilla(groups, q, K):
    return list(range(min(K, len(groups))))


RERANKERS = {"vanilla": vanilla, "fgr": fgr, "detconst": det_const_sort}
