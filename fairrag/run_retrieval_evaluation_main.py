"""Retrieval stage: Vanilla, DetConstSort and FGR (Table 1 of the paper).

Run from inside ``fairrag/`` (edit the configuration block in ``main`` first):

    python run_retrieval_evaluation_main.py

For every query set (test and, if present, the CGLC training set) and every
retriever this writes, under ``results/retrieval/<split>/``:

    <retriever>_<reranker>_slates.csv              one row per ranked document
    <retriever>_<reranker>_fairness_per_query.csv  NDKL, signed NDKL, skew
    retrieval_summary.csv                          Table 1 rows
    skew_sweep.csv                                 skew definition x epsilon
    evaluation_config.json

The slates files are the input of the generation stage
(``run_generation_main.py``), so the LLMs never need the indexes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from metrics import (
    group_exposure,
    hit_at_k,
    mean_ndkl,
    mrr,
    ndcg_at_k,
    prefix_skew,
    skew_at_k,
)
from rerankers import det_const_sort, fgr, rsbe
from slates import save_slates


# Table 1 of the paper, for a side-by-side print at the end of a run.
PAPER_TABLE_1 = {
    ("dense", "vanilla"): (0.438, 0.281, 0.310, -3.441, 0.525, 0.488),
    ("dense", "det_const_sort"): (0.461, 0.263, 0.301, -0.375, 0.156, 0.147),
    ("dense", "fgr"): (0.462, 0.263, 0.301, -0.381, 0.161, 0.146),
    ("sparse", "vanilla"): (0.474, 0.284, 0.318, -3.781, 0.515, 0.488),
    ("sparse", "det_const_sort"): (0.480, 0.241, 0.289, -1.515, 0.263, 0.227),
    ("sparse", "fgr"): (0.479, 0.238, 0.288, -1.510, 0.268, 0.227),
}

SKEW_SWEEP_EPSILONS = (1e-12, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)


def load_test_collections(test_folder=None, test_file=None, max_rows_per_file=None):
    """Load the per-occupation query files.

    max_rows_per_file : int, optional
        Debugging aid. Leave as None for a real run: the files are written
        male-positive rows first, so ``head(10)`` keeps only male, y=1
        biographies and the result is not comparable with the paper.
    """
    if test_file:
        paths = [Path(test_file)]
    else:
        paths = sorted(Path(test_folder).glob("*.csv"))

    if not paths:
        raise FileNotFoundError("No test CSV files were found.")

    collections = []
    for path in paths:
        frame = pd.read_csv(path)
        if max_rows_per_file is not None:
            # Sample across the whole file so every gender/label cell is kept.
            frame = frame.sample(
                n=min(max_rows_per_file, len(frame)), random_state=42
            )
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

        # The same biography may occur in multiple occupation files (and, in
        # train.csv, under several targets). The retrievers return a dict
        # keyed by bio_id, so make each query key experiment-specific.
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


def utility_metrics(pool, relevance, final_k, suffix):
    return {
        f"hit_at_k{suffix}": hit_at_k(pool, relevance, final_k),
        # mrr() has no K parameter, but pool is already truncated to final_k.
        f"mrr_at_k{suffix}": mrr(pool, relevance),
        f"ndcg_at_k{suffix}": ndcg_at_k(pool, relevance, final_k),
    }


def evaluate_slate(
    name,
    retriever_name,
    pool,
    query_targets,
    query_sources,
    final_k,
    q,
    skew_epsilon=1e-12,
    ndkl_epsilon=1e-12,
):
    """Utility and fairness metrics for one (retriever, reranker) slate set.

    Utility is reported under both relevance definitions:

    *_target : a passage is relevant when its occupation equals the occupation
               of the test file the biography sits in (the task label o). All
               400 rows of a file are scored, including the 200 negatives.
    *_source : a passage is relevant when its occupation equals the
               biography's own occupation.
    """
    groups = list(q)
    rsbe_results = rsbe(pool, q=q, group_field="gender", epsilon=ndkl_epsilon)
    skew_results = skew_at_k(
        pool, q=q, K=final_k, group_field="gender", epsilon=skew_epsilon
    )
    prefix_results = prefix_skew(
        pool,
        group_a0=groups[0],
        group_a1=groups[1],
        group_field="gender",
        epsilon=skew_epsilon,
    )
    exposure = group_exposure(pool, K=final_k, group_field="gender")

    result = {
        "retriever": retriever_name,
        "reranker": name,
        "queries": len(pool),
        "k": final_k,
    }
    result.update(utility_metrics(pool, query_targets, final_k, "_target"))
    result.update(utility_metrics(pool, query_sources, final_k, "_source"))
    result.update(
        {
            "mean_ndkl": mean_ndkl(rsbe_results),
            # Skew@K of Geyik et al.: comparable with Table 1.
            "mean_min_skew": average_prefix_skew(skew_results, "min_skew"),
            "mean_max_skew": average_prefix_skew(skew_results, "max_skew"),
            # Appendix C.2 read literally (log-odds over every prefix).
            "mean_prefix_min_skew": average_prefix_skew(prefix_results, "min_skew"),
            "mean_prefix_max_skew": average_prefix_skew(prefix_results, "max_skew"),
        }
    )

    for group in q:
        result[f"mean_signed_ndkl_group_{group}"] = average_signed_ndkl(
            rsbe_results, group
        )
        result[f"exposure_group_{group}"] = exposure.get(group, 0.0)

    return result, rsbe_results, skew_results, prefix_results


def skew_definition_sweep(pool, retriever_name, reranker_name, final_k, q):
    """Mean Min/Max skew under both definitions for a range of epsilons.

    The paper reports neither the epsilon nor (unambiguously) the definition,
    so this table is what to compare against the Min/Max Skew columns of
    Table 1.
    """
    groups = list(q)
    rows = []
    for epsilon in SKEW_SWEEP_EPSILONS:
        at_k = skew_at_k(pool, q=q, K=final_k, group_field="gender", epsilon=epsilon)
        prefix = prefix_skew(
            pool,
            group_a0=groups[0],
            group_a1=groups[1],
            group_field="gender",
            epsilon=epsilon,
        )
        rows.append(
            {
                "retriever": retriever_name,
                "reranker": reranker_name,
                "epsilon": epsilon,
                "skew_at_k_min": average_prefix_skew(at_k, "min_skew"),
                "skew_at_k_max": average_prefix_skew(at_k, "max_skew"),
                "prefix_log_odds_min": average_prefix_skew(prefix, "min_skew"),
                "prefix_log_odds_max": average_prefix_skew(prefix, "max_skew"),
            }
        )
    return rows


def rerank_all(retrieved_pool, final_k, q):
    """All retriever-side systems of Table 1 that need no LLM."""
    return {
        "vanilla": vanilla_pool(retrieved_pool, final_k),
        "det_const_sort": det_const_sort(
            retrieved_pool, q=q, K=final_k, group_field="gender"
        ),
        "fgr": fgr(retrieved_pool, q=q, K=final_k, group_field="gender"),
    }


def evaluate_and_save(
    slates_by_reranker,
    retriever_name,
    query_targets,
    query_sources,
    final_k,
    q,
    output_dir,
    skew_epsilon=1e-12,
):
    """Evaluate each slate set, write its CSVs, and return summary rows."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    sweep_rows = []

    for reranker_name, slate in slates_by_reranker.items():
        summary, rsbe_results, skew_results, prefix_results = evaluate_slate(
            reranker_name,
            retriever_name,
            slate,
            query_targets,
            query_sources,
            final_k,
            q,
            skew_epsilon=skew_epsilon,
        )
        summaries.append(summary)
        sweep_rows.extend(
            skew_definition_sweep(slate, retriever_name, reranker_name, final_k, q)
        )

        save_slates(
            slate,
            retriever_name,
            reranker_name,
            output_dir / f"{retriever_name}_{reranker_name}_slates.csv",
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
                    "prefix_min_skew": prefix_results[bio_id]["min_skew"],
                    "prefix_max_skew": prefix_results[bio_id]["max_skew"],
                }
            )
        pd.DataFrame(fairness_rows).to_csv(
            output_dir / f"{retriever_name}_{reranker_name}_fairness_per_query.csv",
            index=False,
        )

    return summaries, sweep_rows


def run_retriever(
    retriever_name,
    document_store,
    queries,
    query_targets,
    query_sources,
    pool_k,
    final_k,
    q,
    output_dir,
    skew_epsilon=1e-12,
    dense_model_name="BAAI/bge-large-en",
):
    # Imported here so the metric code can be used without Haystack installed.
    from retrieval import dense_retrieve, sparse_retrieve

    query_frame = queries[["bio_id", "bio"]]

    if retriever_name == "dense":
        retrieved_pool = dense_retrieve(
            document_store, query_frame, top_k=pool_k, model_name=dense_model_name
        )
    else:
        retrieved_pool = sparse_retrieve(document_store, query_frame, top_k=pool_k)

    missing = set(query_frame["bio_id"]) - set(retrieved_pool)
    if missing:
        raise ValueError(f"{retriever_name} retrieval omitted {len(missing)} queries.")

    observed_groups = {
        doc.meta.get("gender") for docs in retrieved_pool.values() for doc in docs
    }
    unexpected = observed_groups - set(q)
    print("Target distribution:", q)
    print("Document groups in the retrieved pools:", sorted(map(repr, observed_groups)))
    if unexpected:
        print(
            "Note: documents with groups outside q "
            f"({sorted(map(repr, unexpected))}) count towards slate length "
            "but towards no group's share."
        )
    if not observed_groups & set(q):
        raise ValueError(
            "No retrieved document has a gender label that appears in q. "
            "Check that the keys of q match the index metadata exactly."
        )

    slates_by_reranker = rerank_all(retrieved_pool, final_k, q)

    return evaluate_and_save(
        slates_by_reranker,
        retriever_name,
        query_targets,
        query_sources,
        final_k,
        q,
        output_dir,
        skew_epsilon=skew_epsilon,
    )


def print_summary(summary, relevance_suffix):
    display_columns = [
        "retriever",
        "reranker",
        f"hit_at_k{relevance_suffix}",
        f"mrr_at_k{relevance_suffix}",
        f"ndcg_at_k{relevance_suffix}",
        "mean_min_skew",
        "mean_max_skew",
        "mean_ndkl",
    ]
    display_columns = [c for c in display_columns if c in summary.columns]
    print(summary[display_columns].round(3).to_string(index=False))


def print_paper_table(summary):
    rows = []
    for row in summary.itertuples(index=False):
        reference = PAPER_TABLE_1.get((row.retriever, row.reranker))
        if reference is None:
            continue
        rows.append(
            dict(
                zip(
                    ["retriever", "reranker", "hit", "mrr", "ndcg",
                     "min_skew", "max_skew", "ndkl"],
                    (row.retriever, row.reranker) + reference,
                )
            )
        )
    if rows:
        print(pd.DataFrame(rows).to_string(index=False))


def main():
    # ------------------------------------------------------------------
    # Experiment configuration: edit these values before running.
    # ------------------------------------------------------------------
    # Query sets. "test" is the 22 x 400 inference set; "train" is the CGLC
    # optimisation set (only needed for the generation stage). A value can be
    # a folder of per-occupation CSVs or a single CSV. Missing paths are
    # skipped.
    query_sets = {
        "test": Path("datasets/occupations/"),
        "train": Path("datasets/train/train.csv"),
    }

    retriever_to_run = "sparse"  # "dense", "sparse", or "both"
    dense_index_path = "wiki_people"
    sparse_index_path = "datasets/wiki_name_sparse.json"
    dense_model_name = "BAAI/bge-large-en"

    pool_k = 100   # T, Appendix A.5 (TOP_K=100)
    final_k = 15   # K, Appendix A.5 (DISPLAY_K=15)

    # Which relevance definition the printed table uses. Both are always
    # computed and saved (columns *_target and *_source).
    #   "target"            passage occupation == occupation of the test file
    #   "source_profession" passage occupation == the biography's occupation
    relevance_column = "target"

    # Set True to evaluate only y=1 biographies.
    positive_only = False

    # Debugging only: number of rows sampled from each query file.
    max_rows_per_file = None

    # Keys must match the gender metadata stored in the indexes.
    q = {'Male': 0.5, 'Female': 0.5}

    # Smoothing for the skew columns of the summary. Not reported in the
    # paper. 1e-4 is the value at which the dense slates of this reproduction
    # give Table 1's Min/Max Skew for both Vanilla and FGR (within 0.05);
    # skew_sweep.csv shows the columns for a range of values.
    skew_epsilon = 1e-4

    output_root = Path("results/retrieval")

    if pool_k < final_k:
        raise ValueError("pool_k must be greater than or equal to final_k.")
    if any(value < 0 for value in q.values()) or abs(sum(q.values()) - 1.0) > 1e-12:
        raise ValueError("The target distribution q must be non-negative and sum to 1.")
    if retriever_to_run not in {"dense", "sparse", "both"}:
        raise ValueError("retriever_to_run must be 'dense', 'sparse', or 'both'.")
    if relevance_column not in {"source_profession", "target"}:
        raise ValueError("relevance_column must be 'source_profession' or 'target'.")

    from index import load_dense_index, load_sparse_index

    stores = {}
    if retriever_to_run in {"dense", "both"}:
        print("Loading dense index...")
        stores["dense"] = load_dense_index(dense_index_path)
    if retriever_to_run in {"sparse", "both"}:
        print("Loading sparse index...")
        stores["sparse"] = load_sparse_index(sparse_index_path)

    summaries_by_split = {}

    for split, query_path in query_sets.items():
        query_path = Path(query_path)
        if not query_path.exists():
            print(f"Skipping '{split}': {query_path} does not exist.")
            continue

        output_dir = output_root / split
        output_dir.mkdir(parents=True, exist_ok=True)

        if query_path.is_dir():
            collections = load_test_collections(
                test_folder=query_path, max_rows_per_file=max_rows_per_file
            )
        else:
            collections = load_test_collections(
                test_file=query_path, max_rows_per_file=max_rows_per_file
            )
        queries = prepare_queries(collections, positive_only=positive_only)
        query_targets = dict(zip(queries["bio_id"], queries["target"]))
        query_sources = dict(zip(queries["bio_id"], queries["source_profession"]))

        summaries = []
        sweep_rows = []

        for retriever_name, store in stores.items():
            print(f"[{split}] Running {retriever_name} retrieval for {len(queries)} queries...")
            retriever_summaries, retriever_sweep = run_retriever(
                retriever_name,
                store,
                queries,
                query_targets,
                query_sources,
                pool_k,
                final_k,
                q,
                output_dir,
                skew_epsilon=skew_epsilon,
                dense_model_name=dense_model_name,
            )
            summaries.extend(retriever_summaries)
            sweep_rows.extend(retriever_sweep)

        summary = pd.DataFrame(summaries)
        # Unsuffixed columns follow relevance_column (analysis.py reads these).
        chosen = "_target" if relevance_column == "target" else "_source"
        for metric in ("hit_at_k", "mrr_at_k", "ndcg_at_k"):
            summary[metric] = summary[f"{metric}{chosen}"]
        summary_path = output_dir / "retrieval_summary.csv"
        summary.to_csv(summary_path, index=False)
        pd.DataFrame(sweep_rows).to_csv(output_dir / "skew_sweep.csv", index=False)

        config = {
            "split": split,
            "query_path": str(query_path),
            "retriever": retriever_to_run,
            "dense_index": dense_index_path,
            "sparse_index": sparse_index_path,
            "dense_model_name": dense_model_name,
            "pool_k": pool_k,
            "final_k": final_k,
            "relevance_column": relevance_column,
            "positive_only": positive_only,
            "max_rows_per_file": max_rows_per_file,
            "skew_epsilon": skew_epsilon,
            "q": q,
        }
        with open(output_dir / "evaluation_config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        suffix = "_target" if relevance_column == "target" else "_source"
        other = "_source" if suffix == "_target" else "_target"

        print()
        print(f"[{split}] Results, relevance = {relevance_column}")
        print_summary(summary, suffix)
        print()
        print(f"[{split}] Utility under the other relevance definition")
        print_summary(summary, other)
        if split == "test":
            print()
            print("Paper, Table 1")
            print_paper_table(summary)
        print()
        print(f"Saved results to: {summary_path.resolve()}")

        summaries_by_split[split] = summary

    return summaries_by_split


if __name__ == "__main__":
    main()
