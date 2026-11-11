"""Generation stage: score the three candidate occupations with an LLM.

Run from inside ``fairrag/`` after ``run_retrieval_evaluation_main.py``
(edit the configuration block in ``main`` first):

    python run_generation_main.py

For one LLM, every system and every split this writes

    results/generation/<model>/<split>/<system>.jsonl

with one line per biography: the three candidates, their raw scores, and the
slate's bias signals (signed NDKL from RSBE and the coarse skew proxy). The
files can be resumed: biographies already in a file are skipped.

No calibration happens here. ``run_calibration_main.py`` reads these files,
fits CGLC on the train split and builds Tables 2 to 5, so the calibration
experiments can be repeated without running the LLM again.

Systems
-------
no_retriever    biography only
vanilla         retriever's top K
det_const_sort  DetConstSort slate
fgr             Fair Greedy Reranker slate
in_context      vanilla slate + bias-suppression preamble
fair_filter     vanilla slate filtered by the LLM (also saved as a slates CSV)
fairft          vanilla slate of the FairFT-tuned retriever (needs fairft.py
                and a second retrieval run; skipped when its slates are absent)
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

from baselines import fair_filter, in_context_preamble, make_llm_judge
from prompt import build_example_prompt
from rerankers import rsbe, skew_signal
from run_retrieval_evaluation_main import (
    evaluate_and_save,
    load_test_collections,
    prepare_queries,
)
from slates import load_slates


# System -> reranker whose slates CSV it reads (None: no context).
SYSTEM_SLATES = {
    "no_retriever": None,
    "vanilla": "vanilla",
    "det_const_sort": "det_const_sort",
    "fgr": "fgr",
    "in_context": "vanilla",
    "fair_filter": "vanilla",
    "fairft": "vanilla",
}

DEFAULT_GENDER_MAP = {
    "0": "Male", "1": "Female",
    "0.0": "Male", "1.0": "Female",
    "male": "Male", "female": "Female",
    "m": "Male", "f": "Female",
}


def model_tag(model_name: str) -> str:
    """File-system friendly name, e.g. google/gemma-2b -> google__gemma-2b."""
    return re.sub(r"[^A-Za-z0-9._-]+", "__", str(model_name)).strip("_")


def normalise_gender(values, gender_map=None):
    """Map the query gender column (0/1 or strings) onto the group labels."""
    gender_map = {**DEFAULT_GENDER_MAP, **(gender_map or {})}
    normalised = []
    for value in values:
        key = str(value).strip().lower()
        if key not in gender_map:
            raise ValueError(
                f"Unrecognised query gender value {value!r}; add it to gender_map."
            )
        normalised.append(gender_map[key])
    return normalised


def load_queries(query_path, max_rows_per_file=None, gender_map=None):
    """Load a query set exactly as the retrieval stage does (same bio_id keys)."""
    query_path = Path(query_path)
    if query_path.is_dir():
        collections = load_test_collections(
            test_folder=query_path, max_rows_per_file=max_rows_per_file
        )
    else:
        collections = load_test_collections(
            test_file=query_path, max_rows_per_file=max_rows_per_file
        )
    queries = prepare_queries(collections)

    if "y" not in queries.columns:
        queries["y"] = (queries["source_profession"] == queries["target"]).astype(int)

    gender_column = "gender"
    if gender_column not in queries.columns:
        raise KeyError("The query files need a 'gender' column.")
    queries["subject_group"] = normalise_gender(queries[gender_column], gender_map)
    return queries


def read_done(path):
    """bio_ids already written to a (possibly partial) output file."""
    path = Path(path)
    if not path.exists():
        return set()
    done = set()
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["bio_id"])
            except (json.JSONDecodeError, KeyError):
                # A run interrupted mid-write leaves a broken last line.
                continue
    return done


def load_records(path) -> pd.DataFrame:
    """Read a generation output file into a DataFrame (one row per biography)."""
    rows = []
    with open(path, encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("bio_id", keep="last").reset_index(drop=True)
    return frame


def run_system(
    system,
    queries,
    slates,
    occupations,
    scorer,
    output_path,
    q,
    seed=42,
    negative_mode="random",
    max_passage_words=None,
    preamble=None,
    judge=None,
    progress_every=200,
):
    """Score every query of one system and append the results to a JSONL file.

    scorer(prompt, candidates) -> {candidate: score}

    Returns the slates actually shown to the model ({bio_id: [docs]}), which
    differ from the input for fair_filter.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done = read_done(output_path)
    groups = list(q)

    # For a resumed fair_filter run, rebuild the filtered slates of the rows
    # already written instead of asking the LLM again.
    kept_ranks = {}
    if system == "fair_filter" and done:
        previous = load_records(output_path)
        kept_ranks = dict(zip(previous["bio_id"], previous["context_ranks"]))

    shown = {}

    with open(output_path, "a", encoding="utf-8") as file:
        for number, row in enumerate(queries.to_dict("records"), start=1):
            bio_id = row["bio_id"]

            retrieved = [] if slates is None else slates.get(bio_id, [])
            docs = retrieved

            if system == "fair_filter":
                if bio_id in done:
                    docs = [retrieved[rank - 1] for rank in kept_ranks[bio_id]]
                else:
                    docs = fair_filter(
                        retrieved,
                        bio=row["bio"],
                        judge=judge,
                        max_passage_words=max_passage_words,
                    )
            shown[bio_id] = docs

            if bio_id in done:
                continue

            prompt, candidates = build_example_prompt(
                row,
                docs,
                occupations,
                seed=seed,
                negative_mode=negative_mode,
                max_passage_words=max_passage_words,
                preamble=preamble,
            )
            scores = scorer(prompt, candidates)

            signed = rsbe({bio_id: docs}, q=q, group_field="gender")[bio_id]
            coarse = skew_signal({bio_id: docs}, q=q, group_field="gender")[bio_id]

            record = {
                "bio_id": bio_id,
                "system": system,
                "target": row["target"],
                "y": int(row["y"]),
                "subject_group": row["subject_group"],
                "source_profession": row["source_profession"],
                "candidates": list(candidates),
                "scores": [float(scores[c]) for c in candidates],
                "ndkl": float(signed["ndkl"]),
                "signed_ndkl": [float(signed["signed_scores"][g]) for g in groups],
                "skew_signal": [float(coarse["signed_scores"][g]) for g in groups],
                "groups": groups,
                "n_context": len(docs),
                # 1-based positions, in the input slate, of the passages shown.
                "context_ranks": [
                    rank
                    for rank, doc in enumerate(retrieved, start=1)
                    if any(doc is kept for kept in docs)
                ],
            }
            file.write(json.dumps(record) + "\n")
            file.flush()

            if progress_every and number % progress_every == 0:
                print(f"  {system}: {number}/{len(queries)}")

    return shown


def main():
    # ------------------------------------------------------------------
    # Experiment configuration: edit these values before running.
    # ------------------------------------------------------------------
    # The paper names the model families and sizes but not the checkpoints.
    # These are the public checkpoints that match those descriptions:
    #   "google/gemma-2b"
    #   "meta-llama/Llama-2-7b-hf"
    #   "deepseek-ai/deepseek-coder-6.7b-base"
    #   "tiiuae/falcon-7b"
    #   "google/flan-t5-xl"   (the "3B version"; Section 5.2 also calls
    #                          Flan-T5 a 250M model, which is flan-t5-base)
    model_name = "google/gemma-2b"
    torch_dtype = "float16"      # None on CPU
    device = None                # None: cuda if available
    trust_remote_code = False

    # Generation in the paper is paired with the dense retriever.
    retriever = "dense"

    query_sets = {
        "train": Path("datasets/train/train.csv"),
        "test": Path("datasets/occupations/"),
    }
    systems = [
        "no_retriever",
        "vanilla",
        "det_const_sort",
        "fgr",
        "in_context",
        "fair_filter",
        "fairft",
    ]

    retrieval_root = Path("results/retrieval")
    # Slates retrieved with the FairFT-tuned encoder (see fairft.py).
    fairft_retrieval_root = Path("results/retrieval_fairft")
    output_root = Path("results/generation")

    # Must match the retrieval run.
    q = {'Male': 0.5, 'Female': 0.5}
    final_k = 15

    # "mean_logprob", "sum_logprob" or "first_token". Train and test must use
    # the same mode.
    score_mode = "mean_logprob"

    # Appendix B: each passage is "truncated to a fixed budget" (not given).
    # 15 passages of 150 words do not fit Falcon-7B's 2,048-token window, so
    # a budget is needed; 64 words keeps the whole prompt under ~1,700 tokens.
    max_passage_words = 64

    # "random": both distractors are sampled from the other occupations.
    # "include_true": negatives always see their true occupation as an option.
    negative_mode = "random"
    seed = 42

    # Debugging only: rows sampled per query file (must match retrieval run).
    max_rows_per_file = None

    # ------------------------------------------------------------------
    from generator import load_model, score_candidates

    print(f"Loading {model_name}...")
    model, tokenizer, resolved_device = load_model(
        model_name,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    def scorer(prompt, candidates):
        return score_candidates(
            prompt=prompt,
            candidates=candidates,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            score_mode=score_mode,
        )

    judge = make_llm_judge(model, tokenizer, resolved_device)
    preamble = in_context_preamble()

    for split, query_path in query_sets.items():
        if not Path(query_path).exists():
            print(f"Skipping '{split}': {query_path} does not exist.")
            continue

        queries = load_queries(query_path, max_rows_per_file=max_rows_per_file)
        occupations = sorted(queries["target"].astype(str).unique())
        output_dir = output_root / model_tag(model_name) / split
        output_dir.mkdir(parents=True, exist_ok=True)

        for system in systems:
            reranker = SYSTEM_SLATES[system]
            slates = None

            if reranker is not None:
                root = fairft_retrieval_root if system == "fairft" else retrieval_root
                slates_path = root / split / f"{retriever}_{reranker}_slates.csv"
                if not slates_path.exists():
                    print(f"[{split}] Skipping {system}: {slates_path} not found.")
                    continue
                slates = load_slates(slates_path, bio_ids=queries["bio_id"])

            print(f"[{split}] {system}: {len(queries)} biographies")
            shown = run_system(
                system,
                queries,
                slates,
                occupations,
                scorer,
                output_dir / f"{system}.jsonl",
                q,
                seed=seed,
                negative_mode=negative_mode,
                max_passage_words=max_passage_words,
                preamble=preamble if system == "in_context" else None,
                judge=judge,
            )

            # FairFilter changes the slate, so it gets its own Table 1 row.
            if system == "fair_filter" and split == "test":
                summaries, _ = evaluate_and_save(
                    {"fair_filter": shown},
                    retriever,
                    dict(zip(queries["bio_id"], queries["target"])),
                    dict(zip(queries["bio_id"], queries["source_profession"])),
                    final_k,
                    q,
                    output_dir,
                )
                pd.DataFrame(summaries).to_csv(
                    output_dir / "fair_filter_retrieval_summary.csv", index=False
                )

        config = {
            "model_name": model_name,
            "retriever": retriever,
            "score_mode": score_mode,
            "max_passage_words": max_passage_words,
            "negative_mode": negative_mode,
            "seed": seed,
            "q": q,
            "occupations": occupations,
            "systems": systems,
        }
        with open(output_dir / "generation_config.json", "w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

    print("Done. Next: python run_calibration_main.py")


if __name__ == "__main__":
    main()
