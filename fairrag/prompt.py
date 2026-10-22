import random


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

def build_prompt(
    bio,
    contexts,
    op1,
    target,
    op2,
):
    context_text = "\n".join(
        f"{{text: [gender={doc.meta['gender']}] {doc.content}}}"
        for doc in contexts
    )

    return f"""You are a classifier.
Which of the following three professions
best matches the person described in the
biography below? Choose exactly one:
- {op1}
- {target}
- {op2}
Output only the profession name.

Biography:
\"\"\"
{bio}
\"\"\"

Context documents:
{context_text}

Answer:"""

def build_prompt_positive_only(
    row,
    reranked_docs,
    occupations,
):
    if row["y"] == 0:
        return None

    target = row["occupation"]

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
    true_occupation = row["occupation"]

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