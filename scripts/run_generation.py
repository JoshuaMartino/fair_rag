"""Tables 2-5: generator-side utility/fairness for one LLM.
  configs: no_retriever, vanilla, detconst, fairfilter, in_context, fgr (FGR only), and +CGLC variants
  --criterion tpr  -> Table 4 (equal opportunity);  --ablations -> Table 5"""
import argparse, json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from fairrag.data import load_corpus, load_bias_in_bios, build_inference_set, build_cglc_train_set
from fairrag.retrieval import DenseRetriever, BM25Retriever
from fairrag.pipeline import retrieve_pool, make_slates, llm_features
from fairrag.generator import OptionScorer
from fairrag.baselines import IN_CONTEXT_PREAMBLE, fair_filter
from fairrag.cglc import CGLC, Features
from fairrag.metrics import generator_metrics

ap = argparse.ArgumentParser()
ap.add_argument("--corpus_csv", required=True)
ap.add_argument("--llm", required=True, help="e.g. google/gemma-2b, meta-llama/Llama-2-7b-hf, google/flan-t5-xl, ...")
ap.add_argument("--seq2seq", action="store_true", help="for Flan-T5")
ap.add_argument("--retriever", choices=["dense", "bm25"], default="dense")
ap.add_argument("--T", type=int, default=100); ap.add_argument("--K", type=int, default=15)
ap.add_argument("--criterion", choices=["rd", "tpr"], default="rd")
ap.add_argument("--acc_penalty", type=float, default=0.0, help="optional utility regulariser (NOT in the paper)")
ap.add_argument("--ablations", action="store_true"); ap.add_argument("--max_rows", type=int, default=None)
ap.add_argument("--skip", default="", help="comma list of configs to skip (e.g. fairfilter)")
ap.add_argument("--seed", type=int, default=42); ap.add_argument("--eps", type=float, default=1e-3)
ap.add_argument("--out", default=None)
a = ap.parse_args()
tag = a.llm.split("/")[-1] + f"_{a.retriever}"
out = a.out or f"results/gen_{tag}_{a.criterion}.csv"
q = np.array([0.5, 0.5])

corpus = load_corpus(a.corpus_csv, seed=a.seed); cg = corpus.gender.values
bios_tr, bios_te = load_bias_in_bios("train"), load_bias_in_bios("test")
test, occs = build_inference_set(bios_tr, corpus.occupation.unique(), a.seed)
train, _ = build_cglc_train_set(bios_te, occs, a.seed)
if a.max_rows: test, train = test.sample(a.max_rows, random_state=0), train.sample(min(a.max_rows, len(train)), random_state=0)
print(f"test rows {len(test)}  CGLC-train rows {len(train)}  occupations {len(occs)}")

ret = DenseRetriever(corpus.text, "cache/dense") if a.retriever == "dense" else BM25Retriever(corpus.text)
both = pd.concat([test, train]).drop_duplicates("bio_id")
pool = retrieve_pool(ret, both, a.T, f"cache/{a.retriever}_pool_gen_T{a.T}_{len(both)}.pkl")
scorer = OptionScorer(a.llm, seq2seq=a.seq2seq)
bio_texts = dict(zip(both.bio_id, both.bio))
filt = lambda bio, ids: fair_filter(scorer, bio, [corpus.text[i] for i in ids], a.K)

def feats(split_name, df, rr, ctx=True, pre=None):
    key = f"cache/feat_{tag}_{split_name}_{rr}_{int(ctx)}_{int(bool(pre))}_{len(df)}.npz"
    if os.path.exists(key): return dict(np.load(key, allow_pickle=True))
    sl = make_slates({b: pool[b] for b in df.bio_id.unique()}, cg, rr if rr != "none" else "vanilla", q, a.K, bio_texts,
                     filt if rr == "fairfilter" else None)
    return llm_features(df, sl, corpus, scorer, q, ctx, pre, a.eps, key)

def F(d, sig="S"): return Features(d["logits"], d["g"], d["y"], d["occ"], d[sig])
def evaluate(f, pred): return generator_metrics(pred, f.y, f.g, f.occ)
rows = []
def add(name, m): rows.append({"llm": a.llm, "model": name, **m}); print(rows[-1])

skip = set(a.skip.split(","))
plain = {"no_retriever": ("none", False, None), "vanilla": ("vanilla", True, None), "detconst": ("detconst", True, None),
         "fairfilter": ("fairfilter", True, None), "in_context": ("vanilla", True, IN_CONTEXT_PREAMBLE),
         "fgr_only": ("fgr", True, None)}
for name, (rr, ctx, pre) in plain.items():
    if name in skip: continue
    ft = F(feats("test", test, rr, ctx, pre)); add(name, evaluate(ft, ft.logits.argmax(1) == 0))

def cglc_run(name, rr, **kw):
    sig = kw.pop("sig", "S")
    tr, te = F(feats("train", train, rr), sig), F(feats("test", test, rr), sig)
    c = CGLC(len(occs), criterion=a.criterion, acc_penalty=a.acc_penalty, seed=a.seed, **kw)
    c.fit(tr); add(name, evaluate(te, c.predict(te)))
    os.makedirs("results", exist_ok=True); json.dump(c.state(), open(f"results/cglc_{tag}_{name}.json", "w"))

for name, rr in [("fgr+cglc (Our)", "fgr"), ("detconst+cglc", "detconst"), ("fairfilter+cglc", "fairfilter")]:
    if rr in skip: continue
    cglc_run(name, rr)
if a.ablations:   # Table 5
    cglc_run("abl: skew-only signal", "fgr", sig="S_skew")
    cglc_run("abl: no gate eta", "fgr", use_gate=False)
    cglc_run("abl: no delta_o", "fgr", use_delta_o=False)
    cglc_run("abl: no b_o", "fgr", use_b_o=False)
os.makedirs(os.path.dirname(out), exist_ok=True)
pd.DataFrame(rows).to_csv(out, index=False); print(pd.DataFrame(rows).round(3).to_string())
