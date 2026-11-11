from collections import defaultdict


def fgr(pool, q, K, group_field="gender"):
    """
    Fair Greedy Reranker (FGR)

    Parameters
    ----------
    pool : dict
        Output of dense_retrieve() or sparse_retrieve().

        {
            bio_id: [Document, Document, ...]
        }

        Documents must already be ordered by retrieval
        relevance (highest score first).

    q : dict
        Target group distribution.

        Example:
        {
            "male": 0.5,
            "female": 0.5
        }

    K : int
        Final slate size.

    group_field : str
        Metadata field containing the protected group.

    Returns
    -------
    dict
        {
            bio_id: [Document, ..., Document]
        }

        Each list contains K reranked documents.
    """

    reranked_pool = {}

    for bio_id, retrieved_docs in pool.items():

        remaining = list(retrieved_docs)
        picked = []

        counts = defaultdict(int)

        for k in range(1, K + 1):

            if not remaining:
                break

            #
            # Line 3:
            # U_k[a] = q[a] * k - count[a]
            #
            deficits = {
                group: q[group] * k - counts[group]
                for group in q
            }

            #
            # Group(s) with largest deficit
            #
            max_deficit = max(deficits.values())

            target_groups = {
                group
                for group, deficit in deficits.items()
                if deficit == max_deficit
            }

            #
            # Line 4:
            # first doc in remaining whose group is
            # in argmax U_k[a]
            #
            selected = None

            for doc in remaining:

                group = doc.meta[group_field]

                if group in target_groups:
                    selected = doc
                    break

            #
            # Lines 5-6 fallback
            #
            if selected is None:
                selected = remaining[0]

            #
            # Lines 8-9
            #
            picked.append(selected)
            remaining.remove(selected)

            counts[selected.meta[group_field]] += 1

        reranked_pool[bio_id] = picked

    return reranked_pool


''' example usage: dense_pool = dense_retrieve(
    dense_store,
    test_df[["bio_id", "bio"]]
)

dense_fgr = fgr(
    pool=dense_pool,
    q={
        "male": 0.5,
        "female": 0.5,
    },
    K=10,
)'''

import math
from collections import Counter


def rsbe(reranked_pool, q, group_field="gender", epsilon=1e-12):
    """
    Residual Slate Bias Estimator (RSBE)

    Parameters
    ----------
    reranked_pool : dict
        Output from FGR / Vanilla / DetConst

        {
            bio_id: [Document, ..., Document]
        }

    q : dict
        Target distribution.

        Example:
        {
            "male": 0.5,
            "female": 0.5
        }

    group_field : str
        Metadata field containing the protected group.

    epsilon : float
        KL smoothing constant.

    Returns
    -------
    dict

        {
            bio_id: {
                "ndkl": float,
                "signed_scores": {
                    group: score
                }
            }
        }
    """

    results = {}

    groups = list(q.keys())

    for bio_id, slate in reranked_pool.items():

        K = len(slate)

        if K == 0:
            # No retrieved documents: there is no slate to measure.
            results[bio_id] = {
                "ndkl": 0.0,
                "signed_scores": {g: 0.0 for g in groups},
            }
            continue

        prefix_counts = Counter()

        weighted_kl_sum = 0.0
        B_K = 0.0

        #
        # Compute NDKL(K)
        #
        for k in range(1, K + 1):

            doc = slate[k - 1]

            group = doc.meta[group_field]
            prefix_counts[group] += 1

            kl = 0.0

            for g in groups:

                p_k = prefix_counts[g] / k

                kl += (
                    (p_k + epsilon)
                    * math.log(
                        (p_k + epsilon)
                        / (q[g] + epsilon)
                    )
                )

            w_k = 1.0 / math.log2(k + 1)

            weighted_kl_sum += w_k * kl
            B_K += w_k

        ndkl = weighted_kl_sum / B_K

        #
        # Final slate distribution p_K
        #
        final_counts = Counter(
            doc.meta[group_field]
            for doc in slate
        )

        signed_scores = {}

        for g in groups:

            p_K = final_counts[g] / K

            #
            # Equation (4)
            #
            signed_scores[g] = (
                (p_K - q[g])
                * ndkl
            )

        results[bio_id] = {
            "ndkl": ndkl,
            "signed_scores": signed_scores,
        }

    return results


def skew_signal(reranked_pool, q, group_field="gender", kind="share_gap", epsilon=1e-12):
    """Coarse skew proxy used by the "Skew" ablation row of Table 5.

    The paper describes this ablation as driving CGLC "with a coarse skew
    proxy (instead of signed, prefix-sensitive NDKL)" and gives no formula.
    Two readings are implemented:

    kind="share_gap"  : p_K(a) - q[a]  (Equation 4 without the NDKL factor)
    kind="log_skew"   : log((p_K(a) + eps) / (q[a] + eps))  (Skew@K)

    The result has the same structure as rsbe(), so it can be passed to the
    calibrator in place of the signed NDKL scores.
    """

    if kind not in {"share_gap", "log_skew"}:
        raise ValueError("kind must be 'share_gap' or 'log_skew'")

    results = {}

    for bio_id, slate in reranked_pool.items():

        K = len(slate)

        counts = Counter(
            doc.meta[group_field]
            for doc in slate
        )

        signed_scores = {}

        for g in q:

            p_K = counts[g] / K if K else q[g]

            if kind == "share_gap":
                signed_scores[g] = p_K - q[g]
            else:
                signed_scores[g] = math.log(
                    (p_K + epsilon) / (q[g] + epsilon)
                )

        results[bio_id] = {
            "ndkl": 0.0,
            "signed_scores": signed_scores,
        }

    return results


def det_const_sort(pool, q, K, group_field="gender", strict_feasibility=True):
    """DetConstSort baseline (Geyik et al., 2019, Algorithm 3).

    Parameters
    ----------
    pool : dict
        {bio_id: [Document, ...]} ordered by relevance, best first.

    q : dict
        Target group distribution.

    K : int
        Final slate size.

    Returns
    -------
    dict
        {bio_id: [Document, ...]} with up to K documents.

    Notes
    -----
    Relevance is taken from the position in the retrieved list, not from
    doc.score, because the FAISS HNSW store returns a distance (lower is
    better) while BM25 returns a score (higher is better).

    Documents whose group is not in q can never satisfy a minimum-count
    requirement, so they are only used to fill the slate when every group
    in q has run out of candidates.

    strict_feasibility : bool
        Geyik et al. write the swap test as ``maxIndices[start-1] >= start``
        with 0-based positions, which lets an item slip one place past the
        prefix it was inserted for. True (default) uses ``>= start + 1`` so
        the minimum-count requirement holds at every prefix; False follows
        the published pseudocode. The default is the variant that agrees
        with Table 1, where DetConstSort and FGR have near-identical NDKL
        (0.147 vs 0.146): the published test gives a visibly higher NDKL.
    """

    slack = 1 if strict_feasibility else 0

    reranked_pool = {}

    for bio_id, retrieved_docs in pool.items():

        # Per-group candidate queues: (relevance, document), best first.
        total = len(retrieved_docs)

        candidates = {group: [] for group in q}
        leftovers = []

        for position, doc in enumerate(retrieved_docs):

            group = doc.meta[group_field]
            relevance = total - position

            if group in candidates:
                candidates[group].append((relevance, doc))
            else:
                leftovers.append(doc)

        counts = {group: 0 for group in q}
        min_counts = {group: 0 for group in q}

        ranked_docs = []
        ranked_scores = []
        max_indices = []

        available = sum(len(items) for items in candidates.values())
        slate_size = min(K, available)

        k = 0

        while len(ranked_docs) < slate_size:

            k += 1

            temp_min_counts = {
                group: math.floor(k * q[group])
                for group in q
            }

            changed = [
                group
                for group in q
                if min_counts[group] < temp_min_counts[group]
                and counts[group] < len(candidates[group])
            ]

            # Highest-scoring next candidate first.
            changed.sort(
                key=lambda group: candidates[group][counts[group]][0],
                reverse=True,
            )

            for group in changed:

                if len(ranked_docs) >= slate_size:
                    break

                relevance, doc = candidates[group][counts[group]]

                ranked_docs.append(doc)
                ranked_scores.append(relevance)
                max_indices.append(k)

                start = len(ranked_docs) - 1

                # Bubble the new item up while it is more relevant than its
                # predecessor and the predecessor may still move down.
                while (
                    start > 0
                    and max_indices[start - 1] >= start + slack
                    and ranked_scores[start - 1] < ranked_scores[start]
                ):
                    for values in (ranked_docs, ranked_scores, max_indices):
                        values[start - 1], values[start] = (
                            values[start],
                            values[start - 1],
                        )
                    start -= 1

                counts[group] += 1

            min_counts = temp_min_counts

        # Only reached when the groups in q cannot fill the slate.
        if len(ranked_docs) < K:
            ranked_docs.extend(leftovers[: K - len(ranked_docs)])

        reranked_pool[bio_id] = ranked_docs

    return reranked_pool

