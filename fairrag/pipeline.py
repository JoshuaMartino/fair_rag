"""End-to-end glue: retrieve -> rerank -> prompt LLM -> features for CGLC / evaluation."""
import os, pickle
import numpy as np
from tqdm import tqdm
from .rerank import RERANKERS
from .rsbe import signed_scores, skew_scores
from .generator import build_prompt


def retrieve_pool(retriever, bios_df, T, cache_path=None):
    """Retrieve top-T once per unique biography. Returns {bio_id: (ids, scores)}."""
    if cache_path and os.path.exists(cache_path):
        return pickle.load(open(cache_path, "rb"))
    u = bios_df.drop_duplicates("bio_id")
    ids, scores = retriever.search(u["bio"].tolist(), T)
    pool = {b: (ids[i], scores[i]) for i, b in enumerate(u["bio_id"])}
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True); pickle.dump(pool, open(cache_path, "wb"))
    return pool


def make_slates(pool, corpus_gender, name, q, K, bio_texts=None, filter_fn=None):
    """name in {vanilla, fgr, detconst, fairfilter}. FairFilter filters the top-K (as in a post-retrieval filter)."""
    slates = {}
    for b, (ids, _) in tqdm(pool.items(), desc=f"rerank[{name}]"):
        if name == "fairfilter":
            top = list(ids[:K]); keep = filter_fn(bio_texts[b], top)
            slates[b] = [top[i] for i in keep]
        else:
            order = RERANKERS[name](corpus_gender[ids], q, K)
            slates[b] = [int(ids[i]) for i in order]
    return slates


def retrieval_eval_arrays(slates, bios_df, corpus_gender, corpus_occ):
    """Relevance = passage occupation == bio profession; groups of the slate by metadata gender."""
    u = bios_df.drop_duplicates("bio_id")
    rel, grp = [], []
    for b, prof in zip(u.bio_id, u.profession):
        s = np.asarray(slates[b]); rel.append((corpus_occ[s] == prof).astype(int)); grp.append(corpus_gender[s])
    K = min(len(r) for r in rel)
    return np.array([r[:K] for r in rel]), [g[:K] for g in grp]


def llm_features(df, slates, corpus, scorer, q, use_context=True, preamble=None, eps=1e-3, cache=None):
    """Score every (bio, target) row. Returns dict of arrays (logits target-first, g, y, occ, S)."""
    if cache and os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))
    L, S, SK = [], [], []
    for r in tqdm(df.itertuples(), total=len(df), desc="LLM scoring"):
        sl = slates[r.bio_id]
        passages = [(corpus.text[i], corpus.gender[i]) for i in sl] if use_context else None
        prompt = build_prompt(r.bio, r.target, r.d1, r.d2, passages, preamble)
        L.append(scorer.option_logits(prompt, r.target, r.d1, r.d2))
        gr = corpus.gender.values[sl] if len(sl) else np.zeros(1, int)
        S.append(signed_scores(gr, q, eps)); SK.append(skew_scores(gr, q))
    out = dict(logits=np.array(L), g=df.gender.values, y=df.y.values, occ=df.occ.values, S=np.array(S), S_skew=np.array(SK))
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True); np.savez(cache, **out)
    return out
