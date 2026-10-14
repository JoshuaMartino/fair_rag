"""Table 1: retriever-side utility + fairness for Vanilla / DetConstSort / FairFilter / FairFT / FGR."""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fairrag.data import load_corpus, load_bias_in_bios, build_inference_set
from fairrag.retrieval import DenseRetriever, BM25Retriever
from fairrag.pipeline import retrieve_pool, make_slates, retrieval_eval_arrays
from fairrag.metrics import retrieval_utility, retrieval_fairness

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_csv", required=True, help="wiki_people_subset_24.csv (text, gender, title, occupation_level_3)")
ap.add_argument("--retriever", choices=["dense", "bm25"], default="dense")
ap.add_argument("--rerankers", default="vanilla,detconst,fgr")
ap.add_argument("--T", type=int, default=100); ap.add_argument("--K", type=int, default=15)
ap.add_argument("--eps", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--filter_llm", default=None, help="HF model id for the FairFilter baseline (optional)")
ap.add_argument("--out", default="results/table1.csv")
a = ap.parse_args()

corpus = load_corpus(a.corpus_csv, seed=a.seed)
bios = load_bias_in_bios("train")
ev, occs = build_inference_set(bios, corpus.occupation.unique(), a.seed)
print(f"{len(occs)} occupations, {len(ev)} eval rows, {len(corpus)} passages")
q = np.array([0.5, 0.5])
ret = DenseRetriever(corpus.text, f"cache/dense") if a.retriever == "dense" else BM25Retriever(corpus.text)
pool = retrieve_pool(ret, ev, a.T, f"cache/{a.retriever}_pool_T{a.T}.pkl")
cg, co = corpus.gender.values, corpus.occupation.values
bio_texts = dict(zip(ev.bio_id, ev.bio))
rows = []
for name in a.rerankers.split(","):
    filt = None
    if name == "fairfilter":
        from fairrag.generator import OptionScorer
        from fairrag.baselines import fair_filter
        sc = OptionScorer(a.filter_llm); filt = lambda bio, ids: fair_filter(sc, bio, [corpus.text[i] for i in ids], a.K)
    sl = make_slates(pool, cg, name, q, a.K, bio_texts, filt)
    rel, grp = retrieval_eval_arrays(sl, ev, cg, co)
    rows.append({"retriever": a.retriever, "reranker": name, **retrieval_utility(rel), **retrieval_fairness(grp, q, a.eps)})
    print(rows[-1])
os.makedirs(os.path.dirname(a.out), exist_ok=True)
pd.DataFrame(rows).to_csv(a.out, index=False); print(pd.DataFrame(rows).round(3).to_string())
