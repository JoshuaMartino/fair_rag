# Fair RAG (FGR + RSBE + CGLC) — replication code

Re-implementation of *Fair RAG: End-to-End Fairness Across Retrieval and Generation* (Haque et al., Findings of ACL 2026).

## Layout
| file | paper section |
|---|---|
| `fairrag/rerank.py` | FGR (Alg. 1, Eq. 1) + DetConstSort baseline |
| `fairrag/rsbe.py` | RSBE: prefix shares, NDKL, signed NDKL (Eqs. 2-4) |
| `fairrag/cglc.py` | CGLC: z-scoring, gate, logit shift, 3-stage search (Eqs. 5-11, App. A.6) |
| `fairrag/data.py` | Humans-of-Wikipedia corpus, Bias-in-Bios test / CGLC-train sets (App. A.2) |
| `fairrag/retrieval.py` | FAISS-HNSW32 + BGE-large-en, BM25 (App. A.3-A.5) |
| `fairrag/generator.py` | App. B prompt, option log-prob scoring |
| `fairrag/baselines.py` | In-Context, FairFilter, FairFT |
| `fairrag/metrics.py` | Hit/MRR/NDCG, min/max skew, NDKL, accuracy/P/R/F1, RD, TPR gap |
| `scripts/run_retrieval.py` | Table 1 |
| `scripts/run_generation.py` | Tables 2, 3, 4 (`--criterion tpr`), 5 (`--ablations`) |

## Run
```bash
pip install -r requirements.txt
python -m pytest tests
python scripts/run_retrieval.py --corpus_csv wiki_people_subset_24.csv --retriever dense --rerankers vanilla,detconst,fgr
python scripts/run_retrieval.py --corpus_csv wiki_people_subset_24.csv --retriever bm25  --rerankers vanilla,detconst,fgr
python scripts/run_generation.py --corpus_csv wiki_people_subset_24.csv --llm google/gemma-2b --ablations
python scripts/run_generation.py --corpus_csv wiki_people_subset_24.csv --llm google/flan-t5-xl --seq2seq
python scripts/run_generation.py --corpus_csv wiki_people_subset_24.csv --llm google/gemma-2b --criterion tpr   # Table 4
```
Add `--skip fairfilter` to avoid the (slow) LLM-judge baseline; `--max_rows 500` for a smoke test.
The corpus CSV (`text, gender, title, occupation_level_3`) is not bundled: build it from Humans of Wikipedia
(24 occupations, 126,609 pages). Bias in Bios is pulled from HF (`LabHC/bias_in_bios`).

## Under-specified points — my interpretation (check against the authors' code if released)
1. **Binary task / logits.** Per occupation o, the LLM scores 3 options (target + 2 random distractors); `y_hat=1` iff target wins. Logits = option sequence log-probs, log-softmaxed over the 3 options. The shift is applied to the target logit only.
2. **Eq. 7 / App. A.6 terms.** `Delta = delta + delta_o + (b + b_o)*s(g) + beta_{sign z} * z_o[g]`, `s(g)=+1` for female, `-1` for male; `beta` has separate +/- values. The paper never says how `b`, `b_o` enter or whether `delta_o` and `b_o` differ in role; this is my reading. `|Delta|` hard-clipped at 3.
3. **Objective.** Mean RD (or TPR gap) only, as in Eq. 11. This has a degenerate "predict all negative" optimum; the bounded search ranges/clip limit it. `--acc_penalty` adds an optional accuracy term (off by default, not in the paper).
4. **Standardisation (Eq. 5).** mu_o / sigma_o computed over training slates of positive examples of class o (fallback: all examples of o).
5. **Target q** = (0.5, 0.5); smoothing `eps=1e-3` (not given; affects Min/Max skew magnitudes). Eq. 2 is implemented exactly as printed (un-normalised smoothed KL).
6. **Slate groups** use the passage's metadata gender (as in the prompt, which shows gender per passage).
7. **FGR** follows Alg. 1 literally (argmax-deficit set, else top remaining); `fallback="next_deficit"` is a variant. Note the Appendix E proof is loose: `sum_a U_k(a) = 1` always, so the "no group underrepresented" branch can never fire, and step 2 indexes `N_{k-1}` where `N_k` is meant.
8. **Retrieval relevance** for Table 1 = passage occupation equals the bio's profession.
9. **FairFilter / FairFT** are reconstructed from App. D's two-line descriptions (FairFT updates only the query tower so the HNSW index stays valid). Expect differences from the original. FairFT is implemented in `baselines.py` (`fair_ft_train`) but not wired into `run_generation.py`.
10. **Stack differences:** `bm25s` instead of Haystack's BM25; direct FAISS/sentence-transformers instead of Haystack stores. The paper's Gemma-2B, Flan-T5 "3B vs 250M" and DeepSeek 6.7B model sizes are inconsistent; pick checkpoints explicitly with `--llm`.
11. Training set for CGLC comes from the Bias-in-Bios *test* split and inference from the *train* split, as written in App. A.2.
