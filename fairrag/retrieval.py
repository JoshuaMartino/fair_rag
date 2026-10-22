from haystack.components.retrievers.in_memory import InMemoryBM25Retriever
from sentence_transformers import SentenceTransformer
from haystack_integrations.components.retrievers.faiss import (
    FAISSEmbeddingRetriever,
)
from tqdm import tqdm

def sparse_retrieve_fast(document_store, bios_df, top_k=100):
    retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k)
    pool = {}
    bios_df_it = bios_df[['bio_id','bio']].itertuples(index=False,name=None)
    for bio_id, bio in tqdm(bios_df_it, total=len(bios_df), desc='Retrieval'):
        result = retriever.run(query=bio)
        pool[bio_id] = result["documents"]

    return pool
def sparse_retrieve(document_store, bios_df, top_k=100):
    retriever = InMemoryBM25Retriever(document_store=document_store, top_k=top_k)
    pool = {}
    male_count = 0
    female_count = 0
    
    for _,row in bios_df.iterrows():
        result = retriever.run(query=row["bio"])
        
        pool[row["bio_id"]] = result["documents"]
        for doc in result["documents"]:
            if doc.meta['gender'] == 'Male':
                male_count+=1
            else: female_count+=1
    print("Male docs in pool:", str(male_count))
    print("female docs in pool:", str(female_count))
    return pool
def dense_retrieve(document_store, bios_df, top_k=100):
    embedder = SentenceTransformer("BAAI/bge-large-en")
    retriever = FAISSEmbeddingRetriever(document_store = document_store, top_k=top_k)
    pool = {}
    for _, row in bios_df.iterrows():
        query_embedding = embedder.encode(row["bio"], normalize_embeddings=True)
        result = retriever.run(query_embedding=query_embedding.tolist())
        pool[row["bio_id"]] = result["documents"]
    return pool


'''usage dense_pool = dense_retrieve(
    dense_store,
    test_df[["bio_id", "bio"]]
)

sparse_pool = sparse_retrieve(
    sparse_store,
    test_df[["bio_id", "bio"]]
)"""'''