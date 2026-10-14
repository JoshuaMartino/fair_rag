"""Baselines (App. D): In-Context preambles, FairFilter (LLM filter), FairFT (retriever alignment).
DetConstSort lives in rerank.py. FairFilter/FairFT are re-implementations from the short description in this
paper's App. D; the original method (Zhang et al., 2025b) may differ in detail."""
import numpy as np

# ---- In Context (Oba et al., 2024): counterfactual + gender-neutral preambles
IN_CONTEXT_PREAMBLE = (
    "Despite being a female, Alex became a software engineer. Despite being a male, Sam became a nurse. "
    "Occupations are held by people of all genders in proportions unrelated to a person's gender; "
    "do not use gender to decide the profession.")


# ---- FairFilter: (i) fairness screening, (ii) utility screening, using the generator LLM as judge
def fair_filter(scorer, query_bio, passages, keep_k=15):
    kept = []
    for i, text in enumerate(passages):
        t = " ".join(text.split()[:120])
        bias = scorer.yes_no(f"Is the following passage biased, stereotypical or harmful toward any gender? "
                             f"Answer Yes or No.\nPassage: {t}\nAnswer:")
        if bias > 0:
            continue
        rel = scorer.yes_no(f"Biography: {' '.join(query_bio.split()[:120])}\nPassage: {t}\n"
                            f"Is the passage relevant for identifying this person's profession? Answer Yes or No.\nAnswer:")
        if rel > 0:
            kept.append(i)
    if len(kept) < keep_k:                                  # never return an empty/short slate
        kept += [i for i in range(len(passages)) if i not in set(kept)][: keep_k - len(kept)]
    return kept[:keep_k]


# ---- FairFT: align (query side of) the dense retriever with an LLM-derived fairness distribution.
def fair_ft_train(dense, scorer, corpus, train_queries, k=10, steps_per_epoch=200, lr=1e-5, tau=0.05, epochs=1):
    """Minimise KL(P_R || Q_LM) over top-k docs, updating only the query tower (doc embeddings stay fixed, so the
    existing HNSW index remains valid). Q_LM(d|q) = softmax of the LLM's 'fair & useful evidence' yes/no margin."""
    import torch
    model = dense.model.float(); model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for q in train_queries[: steps_per_epoch]:
            ids, _ = dense.search([q], k)
            ids = ids[0]
            with torch.no_grad():
                m = [scorer.yes_no(f"Passage: {' '.join(corpus.text[i].split()[:100])}\n"
                                   f"Would using this passage as evidence avoid gender bias while staying informative? "
                                   f"Answer Yes or No.\nAnswer:") for i in ids]
                Q = torch.softmax(torch.tensor(m), 0)
            feats = model.tokenize([q])
            feats = {a: b.to(model.device) for a, b in feats.items()}
            qe = torch.nn.functional.normalize(model(feats)["sentence_embedding"], dim=-1)
            D = torch.tensor(np.asarray(dense.emb[ids]), device=qe.device).float()
            P = torch.softmax((qe @ D.T) / tau, -1).squeeze(0)
            loss = (P * (P.clamp_min(1e-9).log() - Q.to(P.device).clamp_min(1e-9).log())).sum()
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return dense
