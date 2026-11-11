"""Reading and writing retrieved slates.

The retrieval runner saves every slate to CSV so that the (expensive) LLM
stage can be run later, and on another machine, without the indexes. This
module holds the small helpers shared by both stages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass(eq=False)
class SlateDocument:
    """Minimal stand-in for a Haystack Document (content, meta, score)."""

    content: str
    meta: dict = field(default_factory=dict)
    score: float | None = None


def documents_to_rows(pool, retriever_name, reranker_name):
    """Flatten {bio_id: [Document, ...]} into one row per ranked document."""
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
    return pd.DataFrame(
        rows,
        columns=[
            "bio_id", "retriever", "reranker", "rank", "score",
            "content", "gender", "occupation", "title",
        ],
    )


def save_slates(pool, retriever_name, reranker_name, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    documents_to_rows(pool, retriever_name, reranker_name).to_csv(path, index=False)
    return path


def load_slates(path, bio_ids=None):
    """Load a slates CSV back into {bio_id: [SlateDocument, ...]}.

    bio_ids : iterable, optional
        Query keys that must be present in the result. Queries with no rows in
        the file (nothing retrieved, or everything filtered out) get an empty
        slate.
    """
    frame = pd.read_csv(path, keep_default_na=False, na_values=[""])
    frame = frame.sort_values(["bio_id", "rank"], kind="stable")

    pool = {}
    for bio_id, group in frame.groupby("bio_id", sort=False):
        docs = []
        for row in group.itertuples(index=False):
            gender = None if pd.isna(row.gender) else row.gender
            occupation = None if pd.isna(row.occupation) else row.occupation
            title = None if pd.isna(row.title) else row.title
            docs.append(
                SlateDocument(
                    content="" if pd.isna(row.content) else str(row.content),
                    meta={
                        "gender": gender,
                        "occupation": occupation,
                        "title": title,
                    },
                    score=None if pd.isna(row.score) else float(row.score),
                )
            )
        pool[str(bio_id)] = docs

    if bio_ids is not None:
        for bio_id in bio_ids:
            pool.setdefault(str(bio_id), [])

    return pool
