from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
import pickle

DENSE_MODEL = "BAAI/bge-large-en"
EF_SEARCH = 1024  # Appendix A.3: query search width


def set_ef_search(document_store, ef_search=EF_SEARCH):
    """Set the HNSW query search width (Appendix A.3: ef_search=1024).

    FAISS defaults to efSearch=16, and uses max(efSearch, top_k) at query
    time, so without this the top-100 pool comes from a much narrower search
    than the paper's. Returns True if the value was set.
    """
    try:
        import faiss

        index = document_store.index
        inner = faiss.downcast_index(getattr(index, "index", index))
        inner.hnsw.efSearch = int(ef_search)
        return True
    except Exception:
        return False


def create_dense_index(wiki_df, index_path="wiki_people", model_name=DENSE_MODEL):
    model = SentenceTransformer(model_name)
    texts = wiki_df["text"].tolist()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar = True, convert_to_numpy=True, normalize_embeddings=True)
    docs = []
    for (_,row), embedding in zip(wiki_df.iterrows(), embeddings):
        docs.append(Document(content=row["text"],embedding=embedding, meta={"gender":row["gender"], "title":row["title"], "occupation":row["occupation"]}))
    document_store = FAISSDocumentStore(index_path=index_path, index_string ="HNSW32", embedding_dim=embeddings.shape[1])
    document_store.write_documents(docs, duplicate_policy=DuplicatePolicy.OVERWRITE)
    document_store.save(index_path)
    return document_store

def load_dense_index(index_path="wiki_people", ef_search=EF_SEARCH):
    # The constructor loads <index_path>.faiss / .json when they exist. In
    # faiss-haystack 3.x ``load`` is an instance method, so calling
    # FAISSDocumentStore.load(index_path=...) on the class fails.
    document_store = FAISSDocumentStore(
        index_path=index_path,
        index_string="HNSW32",
        embedding_dim=1024,
    )
    if not set_ef_search(document_store, ef_search):
        print("Warning: could not set HNSW efSearch; using the FAISS default.")
    return document_store


def create_sparse_index(wiki_df, save_path):
    # wiki_24.csv has a "name" column; the merged file has "title" as well.
    title_column = "title" if "title" in wiki_df.columns else "name"
    docs = [Document(content=row["text"], meta = {"gender":row["gender"], "title":row[title_column], "occupation":row["occupation"]}) for _, row in wiki_df.iterrows()]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")

    document_store.write_documents(docs)
    document_store.save_to_disk(save_path)
    return document_store

def load_sparse_index(save_path):
    docs = InMemoryDocumentStore.load_from_disk(save_path)
    return docs

