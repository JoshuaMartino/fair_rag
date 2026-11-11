# Fair RAG reproduction

Reproduction of Haque et al., "Fair RAG: End-to-End Fairness Across Retrieval
and Generation" (Findings of ACL 2026): the Fair Greedy Reranker (FGR), the
Residual Slate Bias Estimator (RSBE) and Confidence-Gated Logit Calibration
(CGLC), with the baselines and tables of the paper.

## Layout

All code is in `fairrag/` and is run from inside that folder.

| File | Purpose |
| --- | --- |
| `data_prepro.py`, `data.py` | Download and preprocess the corpora; build the 22 x 400 test files and the CGLC training set |
| `index.py`, `retrieval.py` | FAISS + BGE dense index, BM25 sparse index, retrieval |
| `rerankers.py` | FGR, RSBE, DetConstSort, coarse skew signal |
| `metrics.py` | Hit@K, MRR, NDCG@K, NDKL, Skew@K |
| `prompt.py` | The three-way prompt of Appendix B |
| `generator.py` | Candidate scoring for decoder-only and encoder-decoder LLMs |
| `calibration.py` | CGLC: fit, save, load, apply; RD and TPR-gap objectives; ablations |
| `baselines.py`, `fairft.py` | In Context, FairFilter, FairFT |
| `generation_metrics.py` | Accuracy, precision, recall, F1, RD, TPR gap |
| `run_retrieval_evaluation_main.py` | Stage 1: Table 1 and the slates |
| `run_generation_main.py` | Stage 2: LLM scores for every system |
| `run_calibration_main.py` | Stage 3: Tables 2 to 5 |
| `analysis.py` | Plots and tables comparing Vanilla and FGR |

## Running

Each runner has a configuration block at the top of `main()`.

```
cd fairrag
python data_prepro.py                     # preprocess Bias in Bios / Wikipedia
python data.py                            # test files + datasets/train/train.csv
python run_retrieval_evaluation_main.py   # needs the indexes; CPU
python run_generation_main.py             # one LLM per run; GPU
python run_calibration_main.py            # CPU, seconds
pytest -q                                 # no downloads, no GPU
```

Stage 1 writes every slate to CSV, stage 2 writes the raw candidate scores,
and stage 3 only reads those files, so calibration settings can be changed
without touching the indexes or the LLM. Stage 2 can be interrupted and
resumed.

To fill the FairFT rows, follow the workflow at the top of `fairft.py`.

## Where the paper is silent, and what this code does

| Topic | Paper | Here |
| --- | --- | --- |
| Min/Max Skew | Appendix C.2 gives log-odds over every prefix; no epsilon | Skew@K of Geyik et al. (log of share over target, min and max over groups), which is the only reading that can give Table 1. Epsilon is set to 1e-4, the value at which this reproduction's dense slates give Table 1's skew columns. The literal definition is reported alongside, and `skew_sweep.csv` gives both for a range of epsilons |
| Relevance for Hit/MRR/NDCG | "occupation-relevant retrieval" | Both: passage occupation equals the test file's target (`*_target`, printed by default) or the biography's own occupation (`*_source`) |
| Ideal DCG | "normalized by the ideal DCG@K" | Ideal built from the relevant passages in the slate; `ideal_relevant` switches to pool counts |
| DetConstSort swap test | Cites Geyik et al. | The variant that never violates a minimum count; `strict_feasibility=False` gives the published pseudocode |
| Class logits | "per-class logits" | Mean log-probability of the candidate string (`score_mode`) |
| Passage budget in the prompt | "truncated to a fixed budget" | `max_passage_words = 64` |
| Distractors for negatives | Two sampled from other occupations | The same; `negative_mode="include_true"` forces the true occupation in |
| LLM checkpoints | Family and size only | Listed in `run_generation_main.py` |
| CGLC search ranges, clip | Not reported | Defaults in `calibration.py`, overridable from `run_calibration_main.py` |
| "Skew" ablation signal | "coarse skew proxy" | `p_K(a) - q_a` (Equation 4 without the NDKL factor) |
| delta_o and b_o | Both added to the shift | Implemented as written. They are then interchangeable, so the "No delta_o" and "No b_o" rows coincide with the full model whenever the grid lets one stand in for the other |
| Baseline prompts and settings | Not given | Written from the cited papers; see `baselines.py` and `fairft.py` |
