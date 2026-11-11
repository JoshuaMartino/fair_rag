"""Generator-side baselines of Appendix D that need an LLM or a prompt change.

In Context   Oba et al. (2024): bias-suppression preambles placed before the
             prompt.
FairFilter   Zhang et al. (2025b): an LLM screens every retrieved passage for
             fairness, then re-checks the survivors for relevance.

DetConstSort lives in rerankers.py and FairFT in fairft.py.

The Fair RAG paper names these baselines but gives no prompts or settings
for them, so the wording and choices below are this reproduction's, written
from the descriptions in the cited papers.
"""

from __future__ import annotations

from typing import Callable, Sequence


# ----------------------------------------------------------------------
# In Context (Oba et al., 2024)
# ----------------------------------------------------------------------

# Counterfactual ("CF-detailed") preambles: a person in an occupation that is
# stereotyped for the other gender. Oba et al. report that a small number of
# preambles (three or fewer of a kind) works best.
COUNTERFACTUAL_PREAMBLES = (
    "Despite being a female, Mary became a surgeon.",
    "Despite being a male, James became a nurse.",
    "Despite being a female, Linda became an architect.",
    "Despite being a male, Robert became a dietitian.",
)

# Gender-neutral descriptions of stereotyped occupations ("Desc").
DESCRIPTIVE_PREAMBLES = (
    "A surgeon is a trained, careful and dedicated medical professional.",
    "A nurse is a skilled, attentive and knowledgeable health professional.",
)


def in_context_preamble(kind: str = "counterfactual", n: int | None = None) -> str:
    """Preamble text for the In Context baseline.

    kind : "counterfactual", "descriptive" or "both"
    n    : number of sentences to use (default: all of the chosen kind)
    """
    if kind == "counterfactual":
        sentences = COUNTERFACTUAL_PREAMBLES
    elif kind == "descriptive":
        sentences = DESCRIPTIVE_PREAMBLES
    elif kind == "both":
        sentences = COUNTERFACTUAL_PREAMBLES + DESCRIPTIVE_PREAMBLES
    else:
        raise ValueError("kind must be 'counterfactual', 'descriptive' or 'both'")

    if n is not None:
        sentences = sentences[:n]

    return " ".join(sentences)


# ----------------------------------------------------------------------
# FairFilter (Zhang et al., 2025b)
# ----------------------------------------------------------------------

FAIRNESS_SCREEN_TEMPLATE = """You are a fairness evaluation model.

Decide whether the document below could introduce bias or unfair treatment into a language model's answer to the query. Fairness means avoiding bias or discrimination concerning gender, race, religion, age, nationality and similar attributes.

Query:
{query}

Document:
{document}

Reply True if the document is fair to use, or False if it could introduce bias.

Pass:"""

UTILITY_SCREEN_TEMPLATE = """You are a relevance evaluation model.

Decide whether the document below is critical for answering the query. A document is critical if it contains the answer or is highly relevant to it.

Query:
{query}

Document:
{document}

Reply True if the document is critical, or False if it is not.

Pass:"""

CLASSIFICATION_QUERY_TEMPLATE = (
    "Which profession best matches the person described in this biography?\n"
    "{bio}"
)


def make_llm_judge(model, tokenizer, device) -> Callable[[str], bool]:
    """Return judge(prompt) -> bool that compares the scores of True and False."""
    # Imported here so the preamble helpers do not need torch.
    from generator import score_candidates

    def judge(prompt: str) -> bool:
        scores = score_candidates(
            prompt=prompt,
            candidates=["True", "False"],
            model=model,
            tokenizer=tokenizer,
            device=device,
            score_mode="first_token",
        )
        return scores["True"] >= scores["False"]

    return judge


def fair_filter(
    docs: Sequence,
    bio: str,
    judge: Callable[[str], bool],
    utility_screen: bool = True,
    max_passage_words: int | None = None,
) -> list:
    """Two-step FairFilter on one slate. Order of the kept passages is preserved.

    1. Fairness screening drops passages the LLM flags as potentially biased.
    2. Utility screening keeps, among the remaining passages, those the LLM
       judges relevant (this is how the Fair RAG paper describes the step).
       If it would remove every passage, the fairness-screened slate is kept.

    The slate can come back shorter than K, or empty.
    """
    query = CLASSIFICATION_QUERY_TEMPLATE.format(bio=bio)

    def text_of(doc) -> str:
        words = str(doc.content).split()
        if max_passage_words is not None:
            words = words[:max_passage_words]
        return " ".join(words)

    fair_docs = [
        doc
        for doc in docs
        if judge(FAIRNESS_SCREEN_TEMPLATE.format(query=query, document=text_of(doc)))
    ]

    if not utility_screen or not fair_docs:
        return fair_docs

    useful_docs = [
        doc
        for doc in fair_docs
        if judge(UTILITY_SCREEN_TEMPLATE.format(query=query, document=text_of(doc)))
    ]

    return useful_docs if useful_docs else fair_docs
