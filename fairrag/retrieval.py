"""Dense (FAISS HNSW32 + BAAI/bge-large-en) and sparse (BM25) retrievers (App. A.3-A.5)."""
import os
import numpy as np


class DenseRetriever:
    def __init__(self, corpus_texts, index_dir="cache/dense", model_name="BAAI/bge-large-en", device=None, bs=256):
        import faiss, torch
        from sentence_transformers import SentenceTransformer
        self.faiss = faiss
        self.model = SentenceTransformer(model_name, device=device)
        self.model.half() if torch.cuda.is_available() else None
        os.makedirs(index_dir, exist_ok=True)
        ip, ep = f"{index_dir}/wiki_people.faiss", f"{index_dir}/emb.npy"
        if os.path.exists(ip):
            self.index = faiss.read_index(ip)
            self.emb = np.load(ep, mmap_mode="r")
        else:
            self.emb = self.encode(corpus_texts, bs)
            np.save(ep, self.emb)
            self.index = faiss.IndexHNSWFlat(self.emb.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = 200
            self.index.add(self.emb)
            faiss.write_index(self.index, ip)
        self.index.hnsw.efSearch = 1024

    def encode(self, texts, bs=256):
        return np.asarray(self.model.encode(list(texts), batch_size=bs, normalize_embeddings=True,
                                            show_progress_bar=True), dtype="float32")

    def search(self, queries, T=100, bs=256):
        q = self.encode(queries, bs)
        scores, ids = self.index.search(q, T)
        return ids, scores


class BM25Retriever:
    def __init__(self, corpus_texts):
        import bm25s
        self.bm25s = bm25s
        self.ret = bm25s.BM25()
        self.ret.index(bm25s.tokenize(list(corpus_texts), stopwords="en"))

    def search(self, queries, T=100, bs=None):
        toks = self.bm25s.tokenize(list(queries), stopwords="en")
        ids, scores = self.ret.retrieve(toks, k=T)
        return np.asarray(ids), np.asarray(scores)
