from haystack import Document
from haystack.document_stores.types import DuplicatePolicy
from haystack_integrations.document_stores.faiss import FAISSDocumentStore
from sentence_transformers import SentenceTransformer
import json
import pandas as pd
from haystack import Document
from haystack.document_stores.in_memory import InMemoryDocumentStore
import pickle

def create_dense_index(wiki_df, index_path="wiki_people"):
    model = SentenceTransformer("BAAI/bge-large-en")
    texts = wiki_df["text"].tolist()
    embeddings = model.encode(texts, batch_size=256, show_progress_bar = True, convert_to_numpy=True, normalize_embeddings=True)
    docs = []
    for (_,row), embedding in zip(wiki_df.iterrows(), embeddings):
        docs.append(Document(content=row["text"],embedding=embedding, meta={"gender":row["gender"], "title":row["title"], "occupation":row["occupation"]}))
    document_store = FAISSDocumentStore(index_path=index_path, index_string ="HNSW32", embedding_dim=embeddings.shape[1])
    document_store.write_documents(docs, duplicate_policy=DuplicatePolicy.OVERWRITE)
    document_store.save(index_path)
    return document_store

def load_dense_index(index_path="wiki_people"):
    document_store = FAISSDocumentStore.load(index_path=index_path)
    return document_store


def create_sparse_index(wiki_df, save_path):
    docs = [Document(content=row["text"], meta = {"gender":row["gender"], "name":row["name"], "occupation":row["occupation"]}) for _, row in wiki_df.iterrows()]
    document_store = InMemoryDocumentStore(bm25_algorithm="BM25Okapi")

    document_store.write_documents(docs)
    document_store.save_to_disk(save_path)
    return document_store

def load_sparse_index(save_path):
    docs = InMemoryDocumentStore.load_from_disk(save_path)
    return docs

