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