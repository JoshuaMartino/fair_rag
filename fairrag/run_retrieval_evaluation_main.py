"""Evaluate Vanilla retrieval and FGR using the project's implemented methods.

Example:
    python run_retrieval_evaluation.py \
        --test-folder datasets/test \
        --dense-index wiki_people \
        --sparse-index bm25_store.pkl \
        --retriever both \
        --pool-k 100 \
        --final-k 10 \
        --output-dir results/retrieval
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from index import load_dense_index, load_sparse_index
from metrics import group_exposure, hit_at_k, mean_ndkl, mrr, ndcg_at_k, prefix_skew
from rerankers import fgr, rsbe
from retrieval import dense_retrieve, sparse_retrieve


def load_test_collections(test_folder=None, test_file=None):
    if test_file:
        paths = [Path(test_file)]
    else:
        paths = sorted(Path(test_folder).glob("*.csv"))

    if not paths:
        raise FileNotFoundError("No test CSV files were found.")

    collections = []
    for path in paths:
        frame = pd.read_csv(path)
        frame = frame.head(10)
        if "target" not in frame.columns:
            frame["target"] = path.stem
        collections.append(frame)
    return collections


def prepare_queries(test_collections, positive_only=False):
    """Combine per-target test frames without allowing bio_id key collisions."""
    prepared = []

    for frame_number, frame in enumerate(test_collections):
        frame = frame.copy()

        if "bio" not in frame.columns:
            if "hard_text" in frame.columns:
                frame = frame.rename(columns={"hard_text": "bio"})
            else:
                raise KeyError("Each test collection needs a 'bio' or 'hard_text' column.")

        if "bio_id" not in frame.columns:
            if "id" in frame.columns:
                frame["bio_id"] = frame["id"]
            else:
                raise KeyError("Each test collection needs an 'id' or 'bio_id' column.")

        if "source_profession" not in frame.columns:
            if "profession" not in frame.columns:
                raise KeyError("Each test collection needs 'profession' or 'source_profession'.")
            frame["source_profession"] = frame["profession"]

        if "target" not in frame.columns:
            raise KeyError("Each test collection needs a 'target' column.")

        if positive_only:
            if "y" not in frame.columns:
                raise KeyError("--positive-only requires a 'y' column.")
            frame = frame[frame["y"] == 1].copy()

        # The same biography may occur in multiple occupation files. The retrievers
        # return a dict keyed by bio_id, so make each query key experiment-specific.
        frame["original_bio_id"] = frame["bio_id"]
        frame["bio_id"] = (
            frame["target"].astype(str)
            + "::"
            + frame["bio_id"].astype(str)
            + "::"
            + str(frame_number)
        )
        prepared.append(frame)

    queries = pd.concat(prepared, ignore_index=True)
    if queries.empty:
        raise ValueError("There are no queries to evaluate after filtering.")
    if queries["bio_id"].duplicated().any():
        raise ValueError("Evaluation query keys are not unique.")
    return queries


def vanilla_pool(retrieved_pool, final_k):
    """No-reranker baseline: retain the retriever's original first K documents."""
    return {
        bio_id: docs[:final_k]
        for bio_id, docs in retrieved_pool.items()
    }


def average_signed_ndkl(rsbe_results, group):
    return sum(
        result["signed_scores"][group]
        for result in rsbe_results.values()
    ) / len(rsbe_results)


def average_prefix_skew(skew_results, key):
    return sum(result[key] for result in skew_results.values()) / len(skew_results)


def evaluate_slate(name, retriever_name, pool, query_occupations, final_k, q):
    rsbe_results = rsbe(pool, q=q, group_field="gender")
    skew_results = prefix_skew(
        pool,
        group_a0=list(q)[0],
        group_a1=list(q)[1],
        group_field="gender",
    )
    exposure = group_exposure(pool, K=final_k, group_field="gender")

    result = {
        "retriever": retriever_name,
        "reranker": name,
        "queries": len(pool),
        "k": final_k,
        "hit_at_k": hit_at_k(pool, query_occupations, final_k),
        # mrr() has no K parameter, but pool is already truncated to final_k.
        "mrr_at_k": mrr(pool, query_occupations),
        "ndcg_at_k": ndcg_at_k(pool, query_occupations, final_k),
        "mean_ndkl": mean_ndkl(rsbe_results),
        "mean_min_skew": average_prefix_skew(skew_results, "min_skew"),
        "mean_max_skew": average_prefix_skew(skew_results, "max_skew"),
    }

    for group in q:
        result[f"mean_signed_ndkl_group_{group}"] = average_signed_ndkl(
            rsbe_results, group
        )
        result[f"exposure_group_{group}"] = exposure.get(group, 0.0)

    return result, rsbe_results, skew_results


def documents_to_rows(pool, retriever_name, reranker_name):
    rows = []
    for bio_id, docs in pool.items():
        for rank, doc in enumerate(docs, start=1):
            rows.append(
                {
                    "bio_id": bio_id,
                    "retriever": retriever_name,
                    "reranker": reranker_name,
                    "rank": rank,
                    "score": doc.score,
                    "content": doc.content,
                    "gender": doc.meta.get("gender"),
                    "occupation": doc.meta.get("occupation"),
                    "title": doc.meta.get("title", doc.meta.get("name")),
                }
            )
    return pd.DataFrame(rows)


def run_retriever(
    retriever_name,
    document_store,
    queries,
    query_occupations,
    pool_k,
    final_k,
    q,
    output_dir,
):
    query_frame = queries[["bio_id", "bio"]]
    print("here")

    if retriever_name == "dense":
        retrieved_pool = dense_retrieve(document_store, query_frame, top_k=pool_k)
    else:
        retrieved_pool = sparse_retrieve(document_store, query_frame, top_k=pool_k)

    missing = set(query_frame["bio_id"]) - set(retrieved_pool)
    if missing:
        raise ValueError(f"{retriever_name} retrieval omitted {len(missing)} queries.")
    print("target distrabution:", q)
    observed_groups = {doc.meta.get("gender") for docs in retrieved_pool.values() for doc in docs}
    expected_groups = set(q)
    print("obsereved documents groups:", observed_groups)
    print("expected document groups:", expected_groups)
    
    print({
    repr(doc.meta.get("gender"))
    for docs in retrieved_pool.values()
    for doc in docs
})
    vanilla = vanilla_pool(retrieved_pool, final_k)
    fair = fgr(retrieved_pool, q=q, K=final_k, group_field="gender")

    summaries = []
    for reranker_name, slate in [("vanilla", vanilla), ("fgr", fair)]:
        summary, rsbe_results, skew_results = evaluate_slate(
            reranker_name,
            retriever_name,
            slate,
            query_occupations,
            final_k,
            q,
        )
        summaries.append(summary)

        documents_to_rows(slate, retriever_name, reranker_name).to_csv(
            output_dir / f"{retriever_name}_{reranker_name}_slates.csv",
            index=False,
        )

        fairness_rows = []
        for bio_id in slate:
            fairness_rows.append(
                {
                    "bio_id": bio_id,
                    "ndkl": rsbe_results[bio_id]["ndkl"],
                    **{
                        f"signed_ndkl_group_{group}": rsbe_results[bio_id][
                            "signed_scores"
                        ][group]
                        for group in q
                    },
                    "min_skew": skew_results[bio_id]["min_skew"],
                    "max_skew": skew_results[bio_id]["max_skew"],
                }
            )
        pd.DataFrame(fairness_rows).to_csv(
            output_dir / f"{retriever_name}_{reranker_name}_fairness_per_query.csv",
            index=False,
        )

    return summaries


def main():
    # ------------------------------------------------------------------
    # Experiment configuration: edit these values before running.
    # ------------------------------------------------------------------
    test_folder = Path("datasets/occupations/")
    test_file = None  # e.g. Path("datasets/test/rapper.csv"); overrides folder

    retriever_to_run = "sparse"  # "dense", "sparse", or "both"
    dense_index_path = "wiki_people"
    sparse_index_path = "datasets/wiki_name_sparse.json"

    pool_k = 100
    final_k = 15

    # Use "source_profession" for the biography's true occupation.
    # Use "target" for the occupation associated with the test collection.
    relevance_column = "source_profession"

    # Set True to evaluate only y=1 biographies.
    positive_only = False

    # gender metadata: 0=male, 1=female
    q = {'Male': 0.5, 'Female': 0.5}

    output_dir = Path("results/retrieval")
    output_dir.mkdir(parents=True, exist_ok=True)

    if pool_k < final_k:
        raise ValueError("pool_k must be greater than or equal to final_k.")
    if any(value < 0 for value in q.values()) or abs(sum(q.values()) - 1.0) > 1e-12:
        raise ValueError("The target distribution q must be non-negative and sum to 1.")
    if retriever_to_run not in {"dense", "sparse", "both"}:
        raise ValueError("retriever_to_run must be 'dense', 'sparse', or 'both'.")
    if relevance_column not in {"source_profession", "target"}:
        raise ValueError("relevance_column must be 'source_profession' or 'target'.")

    test_collections = load_test_collections(
        test_folder=None if test_file is not None else test_folder,
        test_file=test_file,
    )
    queries = prepare_queries(test_collections, positive_only=positive_only)
    query_occupations = dict(zip(queries["bio_id"], queries[relevance_column]))

    summaries = []

    if retriever_to_run in {"dense", "both"}:
        print("Loading dense index...")
        dense_store = load_dense_index(dense_index_path)
        print(f"Running dense retrieval for {len(queries)} queries...")
        summaries.extend(
            run_retriever(
                "dense",
                dense_store,
                queries,
                query_occupations,
                pool_k,
                final_k,
                q,
                output_dir,
            )
        )

    if retriever_to_run in {"sparse", "both"}:
        print("Loading sparse index...")
        sparse_store = load_sparse_index(sparse_index_path)
        print(f"Running sparse retrieval for {len(queries)} queries...")
        summaries.extend(
            run_retriever(
                "sparse",
                sparse_store,
                queries,
                query_occupations,
                pool_k,
                final_k,
                q,
                output_dir,
            )
        )

    summary = pd.DataFrame(summaries)
    summary_path = output_dir / "retrieval_summary.csv"
    summary.to_csv(summary_path, index=False)

    config = {
        "test_folder": str(test_folder) if test_file is None else None,
        "test_file": str(test_file) if test_file is not None else None,
        "retriever": retriever_to_run,
        "dense_index": dense_index_path,
        "sparse_index": sparse_index_path,
        "pool_k": pool_k,
        "final_k": final_k,
        "relevance_column": relevance_column,
        "positive_only": positive_only,
        "q": q,
    }
    with open(output_dir / "evaluation_config.json", "w", encoding="utf-8") as file:
        json.dump(config, file, indent=2)

    display_columns = [
        "retriever",
        "reranker",
        "hit_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "mean_ndkl",
        "mean_min_skew",
        "mean_max_skew",
        "exposure_group_Male",
        "exposure_group_Female",
    ]
    display_columns = [column for column in display_columns if column in summary.columns]

    print()
    print("Vanilla versus FGR")
    print()
    print(summary[display_columns].to_string(index=False))
    print()
    print(f"Saved results to: {summary_path.resolve()}")

    return summary


if __name__ == "__main__":
    main()
