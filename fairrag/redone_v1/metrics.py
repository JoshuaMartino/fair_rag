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

def ndcg_at_k(pool, query_occupations, K):

    ndcgs = []

    for bio_id, docs in pool.items():

        target = query_occupations[bio_id]

        dcg = 0.0

        relevant_count = sum(
            doc.meta["occupation"] == target
            for doc in docs[:K]
        )

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

        results[bio_id] = {
            "min_skew": min(skews),
            "max_skew": max(skews),
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