import math
from collections import Counter
def hit_at_k(pool, query_occupations, K):
    """
    query_occupations:
        {bio_id: occupation}
    """

    hits = 0

    for bio_id, docs in pool.items():

        target = query_occupations[bio_id]

        found = any(
            doc.meta["occupation"] == target
            for doc in docs[:K]
        )

        hits += int(found)

    return hits / len(pool)

def mrr(pool, query_occupations):

    total = 0.0

    for bio_id, docs in pool.items():

        target = query_occupations[bio_id]

        reciprocal_rank = 0.0

        for rank, doc in enumerate(docs, start=1):

            if doc.meta["occupation"] == target:
                reciprocal_rank = 1.0 / rank
                break

        total += reciprocal_rank

    return total / len(pool)

def ndcg_at_k(pool, query_occupations, K, ideal_relevant=None):
    """NDCG@K (Appendix C.1).

    The paper only says "DCG@K normalized by the ideal DCG@K". By default the
    ideal ranking is built from the relevant documents that are in the slate
    (the reported numbers have NDCG slightly above MRR in every row of
    Table 1, which is what this normalisation produces).

    ideal_relevant : dict, optional
        {bio_id: number of relevant documents available}, e.g. counted over
        the retrieved pool of T documents. When given, the ideal ranking uses
        min(K, that count) relevant documents instead.
    """

    ndcgs = []

    for bio_id, docs in pool.items():

        target = query_occupations[bio_id]

        dcg = 0.0

        relevant_count = sum(
            doc.meta["occupation"] == target
            for doc in docs[:K]
        )

        if ideal_relevant is not None:
            relevant_count = min(K, int(ideal_relevant[bio_id]))

        for rank, doc in enumerate(docs[:K], start=1):

            rel = int(
                doc.meta["occupation"] == target
            )

            dcg += rel / math.log2(rank + 1)

        idcg = sum(
            1 / math.log2(rank + 1)
            for rank in range(
                1,
                relevant_count + 1
            )
        )

        ndcgs.append(
            0.0 if idcg == 0 else dcg / idcg
        )

    return sum(ndcgs) / len(ndcgs)

def mean_ndkl(rsbe_results):

    return sum(
        r["ndkl"]
        for r in rsbe_results.values()
    ) / len(rsbe_results)

def prefix_skew(
    pool,
    group_a0="male",
    group_a1="female",
    group_field="gender",
    epsilon=1e-12,
):
    """Min/Max log-odds skew over every prefix (Appendix C.2, read literally).

    Note: this definition cannot produce the Min/Max Skew columns of Table 1.
    At prefix 1 one group always has share 0, so every query contributes
    +/- log((1 + eps) / eps) to its max or min. No single epsilon gives both
    FGR's (-0.381, 0.161) and Vanilla's -3.441. Use skew_at_k() for the
    numbers that are comparable with Table 1.
    """

    results = {}

    for bio_id, slate in pool.items():

        counts = Counter()

        skews = []

        for k, doc in enumerate(
            slate,
            start=1,
        ):

            counts[
                doc.meta[group_field]
            ] += 1

            p0 = counts[group_a0] / k
            p1 = counts[group_a1] / k

            skew = math.log(
                (p0 + epsilon)
                /
                (p1 + epsilon)
            )

            skews.append(skew)

        # An empty slate (nothing retrieved or everything filtered out)
        # exposes no group, so it contributes no skew.
        results[bio_id] = {
            "min_skew": min(skews) if skews else 0.0,
            "max_skew": max(skews) if skews else 0.0,
        }

    return results

def group_exposure(
    pool,
    K,
    group_field="gender",
):

    counts = Counter()

    total = 0

    for docs in pool.values():

        for doc in docs[:K]:

            counts[
                doc.meta[group_field]
            ] += 1

            total += 1

    return {
        g: counts[g] / total
        for g in counts
    }


def skew_at_k(
    pool,
    q,
    K=None,
    group_field="gender",
    epsilon=1e-12,
):
    """MinSkew@K / MaxSkew@K of Geyik et al. (2019), per query.

    Skew_a@K = log((p_K(a) + eps) / (q[a] + eps)), where p_K(a) is the share
    of group a in the top-K slate. MinSkew is the minimum over groups (the
    most under-represented group, <= 0) and MaxSkew the maximum over groups
    (the most over-represented group, >= 0, at most log(1 / min q)).

    This is the reading that is consistent with Table 1 of the paper: the
    reported Max Skew never exceeds log(2) = 0.693 while Min Skew goes to
    -3.4, and a perfectly alternating K=15 slate gives (-0.069, 0.065).

    Returns
    -------
    dict
        {bio_id: {"min_skew": float, "max_skew": float, "skews": {group: float}}}
    """

    results = {}

    for bio_id, slate in pool.items():

        docs = slate if K is None else slate[:K]
        n = len(docs)

        counts = Counter(
            doc.meta[group_field]
            for doc in docs
        )

        skews = {}

        for group, target_share in q.items():

            # An empty slate exposes no group: treat it as on target.
            share = counts[group] / n if n else target_share

            skews[group] = math.log(
                (share + epsilon)
                /
                (target_share + epsilon)
            )

        results[bio_id] = {
            "min_skew": min(skews.values()),
            "max_skew": max(skews.values()),
            "skews": skews,
        }

    return results

