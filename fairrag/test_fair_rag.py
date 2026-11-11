"""Pytest test suite for the Fair-RAG project.

Run from the project root with:
    pytest -q test_fair_rag.py

The tests mock Haystack, FAISS, SentenceTransformers, and disk I/O, so they do
not download models or build real indexes.
"""

from __future__ import annotations

import importlib
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Lightweight test doubles
# ---------------------------------------------------------------------------

@dataclass(eq=False)
class FakeDocument:
    content: str
    meta: dict = field(default_factory=dict)
    score: float | None = None
    embedding: list[float] | None = None
    id: str | None = None


@pytest.fixture
def bios_df():
    return pd.DataFrame(
        {
            "bio_id": ["b1", "b2"],
            "bio": ["A biography about music", "A biography about medicine"],
            "gender": [0, 1],
            "occupation": ["rapper", "physician"],
            "y": [1, 0],
            "target_profession": ["rapper", "rapper"],
        }
    )


@pytest.fixture
def ranked_docs():
    return [
        FakeDocument("male rapper passage", {"gender": 0, "occupation": "rapper", "title":"A rapper"}, 0.90, id="d1"),
        FakeDocument("male actor passage", {"gender": 0, "occupation": "actor", "title":"An actor"}, 0.80, id="d2"),
        FakeDocument("female rapper passage", {"gender": 1, "occupation": "rapper", "title":"Another rapper"}, 0.70, id="d3"),
        FakeDocument("female doctor passage", {"gender": 1, "occupation": "physician", "title":"the greatest mind"}, 0.60, id="d4"),
    ]


@pytest.fixture
def pool(ranked_docs):
    return {"b1": list(ranked_docs), "b2": list(ranked_docs)}


def _module(name: str):
    """Import a project module and produce a useful error if it is unavailable."""
    try:
        return importlib.import_module(name)
    except Exception as exc:
        pytest.fail(f"Could not import {name}.py: {exc!r}")


def _function(module, *candidate_names):
    for name in candidate_names:
        fn = getattr(module, name, None)
        if callable(fn):
            return fn
    pytest.fail(
        f"{module.__name__}.py must define one of: {', '.join(candidate_names)}"
    )


# ---------------------------------------------------------------------------
# retrieval.py
# ---------------------------------------------------------------------------


def test_sparse_retrieve_returns_pool_and_preserves_document_scores(bios_df):
    retrieval = _module("retrieval")
    returned_docs = [FakeDocument("passage", {"gender": 0, "title":"word", "occupation":"rapper"}, score=0.72)]
    retriever_instance = MagicMock()
    retriever_instance.run.side_effect = [
        {"documents": returned_docs},
        {"documents": returned_docs},
    ]

    with patch.object(
        retrieval, "InMemoryBM25Retriever", return_value=retriever_instance
    ) as retriever_cls:
        store = object()
        result = retrieval.sparse_retrieve(store, bios_df[["bio_id", "bio"]], top_k=25)

    retriever_cls.assert_called_once_with(document_store=store, top_k=25)
    assert set(result) == {"b1", "b2"}
    assert result["b1"][0].score == pytest.approx(0.72)
    assert retriever_instance.run.call_args_list[0].kwargs == {
        "query": "A biography about music"
    }


def test_dense_retrieve_normalizes_queries_and_returns_scored_pool(bios_df):
    retrieval = _module("retrieval")
    embedder = MagicMock()
    embedder.encode.side_effect = [
        np.array([0.6, 0.8], dtype=np.float32),
        np.array([1.0, 0.0], dtype=np.float32),
    ]
    returned_docs = [FakeDocument("passage", {"gender": 1}, score=0.93)]
    retriever_instance = MagicMock()
    retriever_instance.run.return_value = {"documents": returned_docs}

    with (
        patch.object(retrieval, "SentenceTransformer", return_value=embedder) as model_cls,
        patch.object(
            retrieval, "FAISSEmbeddingRetriever", return_value=retriever_instance
        ) as retriever_cls,
    ):
        store = object()
        result = retrieval.dense_retrieve(store, bios_df[["bio_id", "bio"]], top_k=100)

    model_cls.assert_called_once_with("BAAI/bge-large-en")
    retriever_cls.assert_called_once_with(document_store=store, top_k=100)
    assert embedder.encode.call_count == 2
    assert all(c.kwargs["normalize_embeddings"] is True for c in embedder.encode.call_args_list)
    assert retriever_instance.run.call_args_list[0].kwargs["query_embedding"] == pytest.approx([0.6, 0.8])
    assert result["b1"][0].score == pytest.approx(0.93)


def test_retrievers_reject_missing_required_columns():
    retrieval = _module("retrieval")
    bad_df = pd.DataFrame({"text": ["missing bio and bio_id"]})

    with patch.object(retrieval, "InMemoryBM25Retriever"):
        with pytest.raises(KeyError):
            retrieval.sparse_retrieve(object(), bad_df)


# ---------------------------------------------------------------------------
# rerankers.py: FGR and RSBE
# ---------------------------------------------------------------------------


def test_fgr_returns_k_documents_per_query_without_reuse(pool):
    rerankers = _module("rerankers")
    fgr = _function(rerankers, "fgr", "FGR")
    result = fgr(pool, q={0: 0.5, 1: 0.5}, K=4, group_field="gender")

    assert set(result) == set(pool)
    for bio_id, docs in result.items():
        assert len(docs) == 4
        assert len({id(doc) for doc in docs}) == 4
        assert [doc.meta["gender"] for doc in docs] == [0, 1, 0, 1]


def test_fgr_preserves_highest_relevance_within_selected_group(ranked_docs):
    rerankers = _module("rerankers")
    fgr = _function(rerankers, "fgr", "FGR")
    result = fgr({"b1": ranked_docs}, q={0: 0.5, 1: 0.5}, K=2, group_field="gender")

    selected = result["b1"]
    assert selected[0].id == "d1"  # highest ranked group-0 item
    assert selected[1].id == "d3"  # highest ranked group-1 item


def test_fgr_falls_back_when_target_group_has_no_candidate():
    rerankers = _module("rerankers")
    fgr = _function(rerankers, "fgr", "FGR")
    docs = [
        FakeDocument("a", {"gender": 0}, 0.9, id="a"),
        FakeDocument("b", {"gender": 0}, 0.8, id="b"),
    ]
    result = fgr({"q": docs}, q={0: 0.5, 1: 0.5}, K=2, group_field="gender")
    assert [d.id for d in result["q"]] == ["a", "b"]


def test_rsbe_is_zero_for_exactly_balanced_prefixes():
    rerankers = _module("rerankers")
    rsbe = _function(rerankers, "rsbe", "RSBE")
    docs = [
        FakeDocument("m", {"gender": 0}),
        FakeDocument("f", {"gender": 1}),
    ]
    result = rsbe({"b1": docs}, q={0: 0.5, 1: 0.5}, group_field="gender")

    assert result["b1"]["ndkl"] >= 0
    # Final shares match q, so signed scores must be zero even though prefix 1 is skewed.
    assert result["b1"]["signed_scores"][0] == pytest.approx(0.0)
    assert result["b1"]["signed_scores"][1] == pytest.approx(0.0)


def test_rsbe_sign_identifies_over_and_under_exposure():
    rerankers = _module("rerankers")
    rsbe = _function(rerankers, "rsbe", "RSBE")
    docs = [FakeDocument(str(i), {"gender": 0}) for i in range(4)]
    result = rsbe({"b1": docs}, q={0: 0.5, 1: 0.5}, group_field="gender")["b1"]

    assert result["ndkl"] > 0
    assert result["signed_scores"][0] > 0
    assert result["signed_scores"][1] < 0


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------


def test_utility_metrics_with_known_ranking():
    metrics = _module("metrics")
    hit = _function(metrics, "hit_at_k", "hit_rate_at_k")
    mrr = _function(metrics, "mrr", "mean_reciprocal_rank")
    ndcg = _function(metrics, "ndcg_at_k", "ndcg")

    pool = {
        "b1": [
            FakeDocument("x", {"occupation": "actor"}),
            FakeDocument("y", {"occupation": "rapper"}),
        ],
        "b2": [
            FakeDocument("z", {"occupation": "physician"}),
            FakeDocument("w", {"occupation": "actor"}),
        ],
    }
    targets = {"b1": "rapper", "b2": "rapper"}

    assert hit(pool, targets, 2) == pytest.approx(0.5)
    assert mrr(pool, targets) == pytest.approx(0.25)
    # Query 1 has relevance at rank 2: DCG=1/log2(3), IDCG=1. Query 2=0.
    assert ndcg(pool, targets, 2) == pytest.approx((1 / math.log2(3)) / 2)


def test_group_exposure_sums_to_one(pool):
    metrics = _module("metrics")
    exposure = _function(metrics, "group_exposure", "group_exposure_stats")
    result = exposure(pool, K=4, group_field="gender")
    assert sum(result.values()) == pytest.approx(1.0)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.5)


def test_prefix_skew_returns_finite_min_and_max(pool):
    metrics = _module("metrics")
    prefix_skew = _function(metrics, "prefix_skew", "min_max_skew")
    result = prefix_skew(
        pool,
        group_a0=0,
        group_a1=1,
        group_field="gender",
        epsilon=1e-6,
    )
    for values in result.values():
        assert math.isfinite(values["min_skew"])
        assert math.isfinite(values["max_skew"])
        assert values["min_skew"] <= values["max_skew"]


# ---------------------------------------------------------------------------
# prompt.py
# ---------------------------------------------------------------------------


def test_prompt_renderer_includes_options_bio_context_and_gender(ranked_docs):
    prompt_module = _module("prompt")
    build_prompt = _function(prompt_module, "build_prompt")
    text = build_prompt(
        bio="Biography text",
        contexts=ranked_docs[:2],
        op1="actor",
        target="rapper",
        op2="physician",
    )

    assert isinstance(text, str)
    assert "Biography text" in text
    assert "actor" in text and "rapper" in text and "physician" in text
    assert ranked_docs[0].content in text
    assert "gender" in text.lower()
    assert text.rstrip().endswith("Answer:")


def test_sample_distractors_excludes_forbidden_labels():
    prompt_module = _module("prompt")
    sample = _function(prompt_module, "sample_distractors")
    occupations = ["actor", "rapper", "physician", "teacher"]
    selected = sample(occupations, exclude=["rapper"], n=2, seed=42)

    assert len(selected) == 2
    assert len(set(selected)) == 2
    assert "rapper" not in selected


def test_sample_distractors_fails_if_candidate_pool_too_small():
    prompt_module = _module("prompt")
    sample = _function(prompt_module, "sample_distractors")
    with pytest.raises((ValueError, IndexError)):
        sample(["rapper"], exclude=["rapper"], n=2, seed=42)


# ---------------------------------------------------------------------------
# data_prepro.py
# ---------------------------------------------------------------------------


def test_wikipedia_preprocessing_filters_chunks_and_preserves_metadata(tmp_path):
    module = _module("data_prepro")
    preprocess = _function(module, "preprocess_people_in_wikipedia")

    wiki = pd.DataFrame(
        {
            "text": [" ".join(f"w{i}" for i in range(151)), "ignore me"],
            "gender": [0, 1],
            "title": ["TITLE A", "title B"],
            "occupation_level_3": ["rapper", "unmapped"],
        }
    )
    mapping = pd.DataFrame({"profession_name": ["rapper"]})
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    with patch.object(module.pd, "read_pickle", return_value=wiki):
        result = preprocess("unused.pkl", str(mapping_path), str(tmp_path) + "/")

    assert len(result) == 2
    assert set(result["occupation"]) == {"rapper"}
    assert result.iloc[0]["title"] == "TITLE A"
    assert max(result["text"].str.split().str.len()) <= 150
    assert (tmp_path / "wiki_24.csv").exists()


def test_wikipedia_preprocessing_skips_missing_text(tmp_path):
    module = _module("data_prepro")
    preprocess = _function(module, "preprocess_people_in_wikipedia")
    wiki = pd.DataFrame(
        {
            "text": [None],
            "gender": [0],
            "title": ["title A"],
            "occupation_level_3": ["rapper"],
        }
    )
    mapping = pd.DataFrame({"profession_name": ["rapper"]})
    mapping_path = tmp_path / "mapping.csv"
    mapping.to_csv(mapping_path, index=False)

    with patch.object(module.pd, "read_pickle", return_value=wiki):
        # Either skip missing text cleanly, or explicitly raise a clear ValueError.
        try:
            result = preprocess("unused.pkl", str(mapping_path), str(tmp_path) + "/")
        except ValueError as exc:
            assert "text" in str(exc).lower()
        else:
            assert len(result) == 0


# ---------------------------------------------------------------------------
# index.py
# ---------------------------------------------------------------------------


def test_create_dense_index_uses_bge_normalization_hnsw_and_persistence():
    module = _module("index")
    create = _function(module, "create_dense_index")
    wiki_df = pd.DataFrame(
        {
            "text": ["one", "two"],
            "gender": [0, 1],
            "title": ["A", "B"],
            "occupation": ["rapper", "actor"],
        }
    )
    model = MagicMock()
    model.encode.return_value = np.ones((2, 1024), dtype=np.float32)
    store = MagicMock()

    with (
        patch.object(module, "SentenceTransformer", return_value=model) as model_cls,
        patch.object(module, "FAISSDocumentStore", return_value=store) as store_cls,
    ):
        result = create(wiki_df, index_path="wiki_people")

    model_cls.assert_called_once_with("BAAI/bge-large-en")
    assert model.encode.call_args.kwargs["batch_size"] == 256
    assert model.encode.call_args.kwargs["normalize_embeddings"] is True
    assert store_cls.call_args.kwargs["index_string"] == "HNSW32"
    assert store_cls.call_args.kwargs["embedding_dim"] == 1024
    store.write_documents.assert_called_once()
    store.save.assert_called()
    assert result is store or (isinstance(result, tuple) and store in result)


def test_create_sparse_index_writes_same_docs_and_returns_store(tmp_path):
    module = _module("index")
    create = _function(module, "create_sparse_index")

    wiki_df = pd.DataFrame(
        {
            "text": ["one"],
            "gender": [0],
            "title": ["A"],
            "occupation": ["rapper"],
        }
    )

    store = MagicMock()
    save_path = tmp_path / "bm25_store.pkl"

    with (
        patch.object(
            module,
            "InMemoryDocumentStore",
            return_value=store,
        ) as store_class,
        patch.object(
            module.pickle,
            "dump",
        ) as pickle_dump,
    ):
        result = create(
            wiki_df,
            str(save_path),
        )

    # Verify BM25 store configuration
    store_class.assert_called_once_with(
        bm25_algorithm="BM25Okapi"
    )

    # Verify documents were written
    store.write_documents.assert_called_once()

    written_docs = store.write_documents.call_args.args[0]

    assert len(written_docs) == 1
    assert written_docs[0].content == "one"
    assert written_docs[0].meta == {
        "gender": 0,
        "title": "A",
        "occupation": "rapper",
    }

    # Verify persistence was requested (Appendix A.4: save_to_disk)
    store.save_to_disk.assert_called_once_with(str(save_path))
    pickle_dump.assert_not_called()

    # Verify function return
    assert result is store

# ---------------------------------------------------------------------------
# data.py: balanced mini-set construction
# ---------------------------------------------------------------------------


def test_training_builder_has_100_per_profession_balance_and_unique_ids(
    tmp_path
):
    module = _module("data")

    rows = []
    uid = 0

    for occupation in [
        "rapper",
        "actor",
        "physician",
    ]:
        for gender in [0, 1]:
            for _ in range(120):
                rows.append(
                    {
                        "id": f"id_{uid}",
                        "bio_id": f"id_{uid}",
                        "profession": occupation,
                        "gender": gender,
                        "bio": f"Biography {uid}",
                    }
                )
                uid += 1

    bios_df = pd.DataFrame(rows)

    result = module.populate_train_collection(
        b_in_b=bios_df,
        folder_path=tmp_path,
    )

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 300

    # No row reuse anywhere in the complete collection.
    assert result["id"].is_unique

    # 100 examples per target profession.
    assert (
        result.groupby("target")
        .size()
        .eq(100)
        .all()
    )

    # 50 male and 50 female examples per target.
    assert (
        result.groupby(["target", "gender"])
        .size()
        .eq(50)
        .all()
    )

    # 50 positive and 50 negative examples per target.
    assert (
        result.groupby(["target", "y"])
        .size()
        .eq(50)
        .all()
    )

    # Exactly 25 rows in each gender-label cell.
    assert (
        result.groupby(
            ["target", "gender", "y"]
        )
        .size()
        .eq(25)
        .all()
    )

    positives = result[result["y"] == 1]
    negatives = result[result["y"] == 0]

    assert (
        positives["source_profession"]
        == positives["target"]
    ).all()

    assert (
        negatives["source_profession"]
        != negatives["target"]
    ).all()

    output_path = tmp_path / "train.csv"

    assert output_path.exists()

    saved_result = pd.read_csv(output_path)

    assert len(saved_result) == len(result)
    assert "Unnamed: 0" not in saved_result.columns
def test_populate_test_collection_creates_balanced_files(
    tmp_path
):
    module = _module("data")

    rows = []
    uid = 0

    # rapper and actor should be retained.
    for occupation in ["rapper", "actor"]:
        for gender in [0, 1]:
            for _ in range(120):
                rows.append(
                    {
                        "id": f"id_{uid}",
                        "bio_id": f"id_{uid}",
                        "profession": occupation,
                        "gender": gender,
                        "bio": f"Biography {uid}",
                    }
                )
                uid += 1

    # paralegal has enough rows but is excluded.
    for gender in [0, 1]:
        for _ in range(120):
            rows.append(
                {
                    "id": f"id_{uid}",
                    "bio_id": f"id_{uid}",
                    "profession": "paralegal",
                    "gender": gender,
                    "bio": f"Biography {uid}",
                }
            )
            uid += 1

    # teacher is below the required threshold.
    for gender in [0, 1]:
        for _ in range(50):
            rows.append(
                {
                    "id": f"id_{uid}",
                    "bio_id": f"id_{uid}",
                    "profession": "teacher",
                    "gender": gender,
                    "bio": f"Biography {uid}",
                }
            )
            uid += 1

    bios_df = pd.DataFrame(rows)

    test_collections = module.populate_test_collection(
        b_in_b=bios_df,
        folder_path=tmp_path,
    )

    # The return value is a list of DataFrames.
    assert isinstance(test_collections, list)
    assert len(test_collections) == 2

    assert all(
        isinstance(collection, pd.DataFrame)
        for collection in test_collections
    )

    collections_by_target = {
        collection["target"].iloc[0]: collection
        for collection in test_collections
    }

    assert set(collections_by_target) == {
        "rapper",
        "actor",
    }

    # CSV files should still be created.
    assert (tmp_path / "rapper.csv").exists()
    assert (tmp_path / "actor.csv").exists()

    # Excluded occupations should not have files.
    assert not (tmp_path / "paralegal.csv").exists()
    assert not (tmp_path / "teacher.csv").exists()

    for target, test_df in collections_by_target.items():
        # Four groups of 100.
        assert len(test_df) == 400

        # Every row belongs to this target experiment.
        assert test_df["target"].eq(target).all()

        # 200 male and 200 female biographies.
        assert (
            test_df.groupby("gender")
            .size()
            .eq(200)
            .all()
        )

        # 200 positives and 200 negatives.
        assert (
            test_df.groupby("y")
            .size()
            .eq(200)
            .all()
        )

        # 100 in each gender-label cell.
        assert (
            test_df.groupby(["gender", "y"])
            .size()
            .eq(100)
            .all()
        )

        positives = test_df[test_df["y"] == 1]
        negatives = test_df[test_df["y"] == 0]

        assert positives["profession"].eq(target).all()
        assert negatives["profession"].ne(target).all()

        assert (
            test_df["source_profession"]
            == test_df["profession"]
        ).all()

        # No biography should repeat within one test file.
        assert test_df["id"].is_unique

        # Confirm the saved CSV matches the returned DataFrame.
        saved_df = pd.read_csv(
            tmp_path / f"{target}.csv"
        )

        assert len(saved_df) == len(test_df)
        assert "Unnamed: 0" not in saved_df.columns