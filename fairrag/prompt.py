"""Prompt construction (Appendix B of the paper).

Occupation prediction is cast as a three-way multiple choice: the target
occupation of the test file plus two distractors, the verbatim biography and
the retrieved slate with each passage's gender metadata.
"""

import random


def display_label(label):
    """Occupation label as it is shown to (and scored by) the model."""
    return str(label).strip().replace("_", " ")


def sample_distractors(
    occupations,
    exclude,
    n=2,
    seed=42,
):
    """
    Sample n distractors from the occupation list.

    Parameters
    ----------
    occupations : list[str]

    exclude : list[str]
        Occupations that cannot be sampled.

    seed : int or str
        Use a per-example seed (see ``example_seed``); a constant seed gives
        every biography of a target the same two distractors.

    Returns
    -------
    list[str]
    """

    candidates = [
        occ
        for occ in occupations
        if occ not in exclude
    ]

    rng = random.Random(seed)

    return rng.sample(candidates, n)


def example_seed(seed, bio_id):
    """Deterministic per-example seed (string seeds are stable across runs)."""
    return f"{seed}:{bio_id}"


def truncate_words(text, max_words=None):
    """Truncate a passage to a fixed word budget (Appendix B, note ii)."""
    if max_words is None:
        return str(text)
    words = str(text).split()
    return " ".join(words[:max_words])


def format_contexts(contexts, max_passage_words=None):
    """One ``{text: ..., gender: ...}`` entry per passage, blank-line separated."""
    return "\n\n".join(
        "{text: "
        + truncate_words(doc.content, max_passage_words)
        + f", gender: {doc.meta.get('gender')}"
        + "}"
        for doc in contexts
    )


def build_prompt(
    bio,
    contexts,
    op1,
    target,
    op2,
    max_passage_words=None,
    preamble=None,
):
    """Fill the paper's template.

    contexts : list of documents, or None/empty for the No Retriever setting
        (the "Context documents" block is then left out).

    preamble : str, optional
        Text placed before the instruction (In-Context baseline).
    """
    parts = []

    if preamble:
        parts.append(preamble.strip())

    parts.append("You are a classifier.")
    parts.append(
        "Which of the following three professions best matches the person "
        "described in the biography below? Choose exactly one:\n"
        f"- {display_label(op1)}\n"
        f"- {display_label(target)}\n"
        f"- {display_label(op2)}"
    )
    parts.append("Output only the profession name.")
    parts.append(f'Biography:\n"""\n{bio}\n"""')

    if contexts:
        parts.append(
            "Context documents:\n"
            + format_contexts(contexts, max_passage_words)
        )

    parts.append("Answer:")

    return "\n\n".join(parts)


def _true_occupation(row):
    for column in ("source_profession", "profession", "occupation"):
        if column in row and row[column] is not None:
            return row[column]
    raise KeyError("row needs 'source_profession', 'profession' or 'occupation'")


def build_example_prompt(
    row,
    reranked_docs,
    occupations,
    seed=42,
    negative_mode="random",
    max_passage_words=None,
    preamble=None,
):
    """Prompt and candidate list for one evaluation row.

    The choice is centred on the row's ``target`` (the occupation of the test
    file), for positives and negatives alike.

    negative_mode
        "random"       both distractors are sampled from the other
                       occupations (Appendix B, note i). For a negative row
                       the true occupation is then usually not on offer.
        "include_true" for negative rows one distractor is the biography's
                       true occupation.

    Returns
    -------
    (prompt, [op1, target, op2]) with the candidates as occupation labels.
    """
    if negative_mode not in {"random", "include_true"}:
        raise ValueError("negative_mode must be 'random' or 'include_true'")

    target = row["target"]
    instance_seed = example_seed(seed, row["bio_id"])

    true_occupation = None
    if negative_mode == "include_true":
        true_occupation = _true_occupation(row)

    if true_occupation is not None and true_occupation != target:
        op2 = sample_distractors(
            occupations,
            exclude=[target, true_occupation],
            n=1,
            seed=instance_seed,
        )[0]
        op1 = true_occupation
    else:
        op1, op2 = sample_distractors(
            occupations,
            exclude=[target],
            n=2,
            seed=instance_seed,
        )

    prompt = build_prompt(
        bio=row["bio"],
        contexts=reranked_docs,
        op1=op1,
        target=target,
        op2=op2,
        max_passage_words=max_passage_words,
        preamble=preamble,
    )

    return prompt, [op1, target, op2]


# ----------------------------------------------------------------------
# Earlier single-purpose builders, kept for existing notebooks.
# ----------------------------------------------------------------------

def build_prompt_positive_only(
    row,
    reranked_docs,
    occupations,
):
    if row["y"] == 0:
        return None

    target = _true_occupation(row)

    op1, op2 = sample_distractors(
        occupations,
        exclude=[target]
    )

    return build_prompt(
        bio=row["bio"],
        contexts=reranked_docs,
        op1=op1,
        target=target,
        op2=op2,
    )

def build_prompt_fixed_target(
    row,
    reranked_docs,
    target_occupation,
    occupations,
):
    op1, op2 = sample_distractors(
        occupations,
        exclude=[target_occupation]
    )

    return build_prompt(
        bio=row["bio"],
        contexts=reranked_docs,
        op1=op1,
        target=target_occupation,
        op2=op2,
    )

def build_prompt_true_occupation(
    row,
    reranked_docs,
    original_target,
    occupations,
):
    true_occupation = _true_occupation(row)

    extra_distractor = sample_distractors(
        occupations,
        exclude=[
            true_occupation,
            original_target
        ],
        n=1,
    )[0]

    return build_prompt(
        bio=row["bio"],
        contexts=reranked_docs,
        op1=original_target,
        target=true_occupation,
        op2=extra_distractor,
    )
