"""Tests for the parts of the paper added after the retrieval prototype.

Run from inside ``fairrag/`` with:
    pytest -q test_paper_pipeline.py

Nothing is downloaded: the language models are tiny randomly initialised
networks built in tiny_models.py.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
import torch

from baselines import fair_filter, in_context_preamble
from calibration import (
    ConfidenceGatedLogitCalibrator,
    one_vs_rest_gap,
    one_vs_rest_selected,
)
from fairft import answer_probability, fairft_loss, train_fairft
from generation_metrics import per_occupation_report, summarize
from generator import score_candidates
from metrics import ndcg_at_k, prefix_skew, skew_at_k
from prompt import build_example_prompt, build_prompt
from rerankers import det_const_sort, fgr, rsbe, skew_signal
from run_calibration_main import fit_and_apply, uncalibrated_selected
from run_generation_main import load_queries, load_records, normalise_gender, run_system
from run_retrieval_evaluation_main import (
    evaluate_slate,
    load_test_collections,
    prepare_queries,
)
from slates import SlateDocument, load_slates, save_slates
from tiny_models import tiny_causal, tiny_seq2seq

Q = {"Male": 0.5, "Female": 0.5}


def doc(gender, occupation="nurse", content="text", score=None):
    return SlateDocument(content, {"gender": gender, "occupation": occupation}, score)


def alternating(n, first="Female"):
    other = "Male" if first == "Female" else "Female"
    return [doc(first if i % 2 == 0 else other) for i in range(n)]


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def test_ndkl_of_perfectly_alternating_k15_slate():
    # Lower bound for FGR at K=15 with a 50/50 target. Table 1 reports 0.146
    # for FGR (dense), just above it; K=10 could not go below 0.1617.
    assert rsbe({"q": alternating(15)}, Q)["q"]["ndkl"] == pytest.approx(0.1257, abs=1e-4)
    assert rsbe({"q": alternating(10)}, Q)["q"]["ndkl"] == pytest.approx(0.1617, abs=1e-4)


def test_skew_at_k_matches_geyik_definition():
    result = skew_at_k({"q": alternating(15)}, Q)["q"]
    assert result["max_skew"] == pytest.approx(math.log((8 / 15) / 0.5))
    assert result["min_skew"] == pytest.approx(math.log((7 / 15) / 0.5))

    all_male = skew_at_k({"q": [doc("Male")] * 15}, Q, epsilon=1e-6)["q"]
    assert all_male["max_skew"] == pytest.approx(math.log(2), abs=1e-5)
    assert all_male["min_skew"] == pytest.approx(math.log(1e-6 / 0.5), abs=1e-3)


def test_prefix_log_odds_is_dominated_by_the_first_position():
    # Appendix C.2 read literally: the first prefix always has one empty group.
    result = prefix_skew({"q": alternating(15)}, "Male", "Female", epsilon=1e-12)["q"]
    assert result["min_skew"] == pytest.approx(math.log(1e-12 / (1 + 1e-12)), rel=1e-6)


def test_skew_handles_empty_slates():
    assert skew_at_k({"q": []}, Q)["q"]["min_skew"] == 0.0
    assert prefix_skew({"q": []}, "Male", "Female")["q"]["max_skew"] == 0.0
    assert rsbe({"q": []}, Q)["q"]["ndkl"] == 0.0


def test_ndcg_ideal_can_use_pool_counts():
    slate = [doc("Male", "poet"), doc("Male", "nurse")]
    relevance = {"q": "nurse"}
    assert ndcg_at_k({"q": slate}, relevance, 2) == pytest.approx(1 / math.log2(3))
    with_pool = ndcg_at_k({"q": slate}, relevance, 2, ideal_relevant={"q": 5})
    assert with_pool == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_evaluate_slate_reports_both_relevance_definitions():
    slate = {"q": [doc("Male", "nurse"), doc("Female", "poet")]}
    summary, *_ = evaluate_slate(
        "vanilla", "dense", slate, {"q": "poet"}, {"q": "nurse"}, 2, Q
    )
    assert summary["mrr_at_k_target"] == pytest.approx(0.5)
    assert summary["mrr_at_k_source"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Rerankers
# ---------------------------------------------------------------------------

def male_heavy_pool(n=100, female_every=5):
    return [
        SlateDocument(str(i), {"gender": "Female" if i % female_every == 0 else "Male"}, -i)
        for i in range(n)
    ]


def test_det_const_sort_meets_minimum_counts_at_every_prefix():
    slate = det_const_sort({"q": male_heavy_pool()}, Q, 15)["q"]
    assert len(slate) == 15
    assert len({d.content for d in slate}) == 15
    for k in range(1, 16):
        prefix = [d.meta["gender"] for d in slate[:k]]
        for group, share in Q.items():
            assert prefix.count(group) >= math.floor(k * share)


def test_det_const_sort_published_variant_is_close_to_fgr():
    pool = {"q": male_heavy_pool()}
    published = det_const_sort(pool, Q, 15, strict_feasibility=False)["q"]
    fair = fgr(pool, Q, 15)["q"]
    assert sorted(d.content for d in published) == sorted(d.content for d in fair)
    counts = [d.meta["gender"] for d in published]
    assert abs(counts.count("Male") - counts.count("Female")) <= 1


def test_det_const_sort_falls_back_when_a_group_is_missing():
    pool = {"q": [doc("Male", content=str(i)) for i in range(30)]}
    slate = det_const_sort(pool, Q, 15)["q"]
    assert [d.content for d in slate] == [str(i) for i in range(15)]


def test_skew_signal_is_equation_4_without_the_ndkl_factor():
    slate = {"q": [doc("Male")] * 3 + [doc("Female")]}
    signal = skew_signal(slate, Q)["q"]["signed_scores"]
    assert signal["Male"] == pytest.approx(0.25)
    assert signal["Female"] == pytest.approx(-0.25)
    signed = rsbe(slate, Q)["q"]
    assert signed["signed_scores"]["Male"] == pytest.approx(0.25 * signed["ndkl"])


# ---------------------------------------------------------------------------
# Query loading and slates
# ---------------------------------------------------------------------------

def write_query_file(folder, target, n_per_cell=3, start=0):
    rows = []
    uid = start
    for gender in (0, 1):
        for y in (1, 0):
            for _ in range(n_per_cell):
                profession = target if y else "poet"
                rows.append(
                    {
                        "bio_id": uid, "bio": f"bio {uid}", "profession": profession,
                        "gender": gender, "y": y, "target": target,
                        "source_profession": profession,
                    }
                )
                uid += 1
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(folder / f"{target}.csv", index=False)


def test_all_rows_of_every_query_file_are_loaded(tmp_path):
    write_query_file(tmp_path, "nurse", n_per_cell=5)
    queries = prepare_queries(load_test_collections(test_folder=tmp_path))
    assert len(queries) == 20
    assert set(queries["gender"]) == {0, 1}
    assert set(queries["y"]) == {0, 1}

    capped = prepare_queries(
        load_test_collections(test_folder=tmp_path, max_rows_per_file=8)
    )
    assert len(capped) == 8


def test_query_gender_is_mapped_to_group_labels(tmp_path):
    assert normalise_gender([0, 1, "female", "Male"]) == ["Male", "Female", "Female", "Male"]
    with pytest.raises(ValueError):
        normalise_gender(["unknown"])
    write_query_file(tmp_path, "nurse")
    queries = load_queries(tmp_path)
    assert set(queries["subject_group"]) == {"Male", "Female"}


def test_slates_round_trip_through_csv(tmp_path):
    pool = {
        "nurse::1::0": [doc("Male", "nurse", "first", 0.9), doc(None, "poet", "second", 0.5)],
    }
    path = save_slates(pool, "dense", "fgr", tmp_path / "slates.csv")
    loaded = load_slates(path, bio_ids=["nurse::1::0", "nurse::2::0"])
    assert [d.content for d in loaded["nurse::1::0"]] == ["first", "second"]
    assert loaded["nurse::1::0"][0].meta["gender"] == "Male"
    assert loaded["nurse::1::0"][1].meta["gender"] is None
    assert loaded["nurse::2::0"] == []


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

OCCUPATIONS = ["nurse", "surgeon", "teacher", "interior_designer", "dj", "poet"]


def test_prompt_follows_the_paper_template():
    text = build_prompt(
        "A biography.", [doc("Female", content="one two dropped words")],
        "interior_designer", "nurse", "dj", max_passage_words=2,
    )
    lines = text.split("\n")
    assert lines[0] == "You are a classifier."
    assert "- interior designer\n- nurse\n- dj" in text
    assert "Output only the profession name." in text
    assert '"""\nA biography.\n"""' in text
    assert "{text: one two, gender: Female}" in text
    assert "dropped" not in text
    assert text.endswith("Answer:")


def test_no_retriever_prompt_has_no_context_block():
    text = build_prompt("A biography.", [], "dj", "nurse", "poet")
    assert "Context documents" not in text


def test_distractors_vary_per_example_and_are_reproducible():
    row = {"bio_id": "a", "bio": "x", "target": "nurse", "source_profession": "poet"}
    _, first = build_example_prompt(row, [], OCCUPATIONS)
    _, again = build_example_prompt(row, [], OCCUPATIONS)
    assert first == again
    assert first[1] == "nurse" and "nurse" not in (first[0], first[2])

    seen = {
        tuple(build_example_prompt({**row, "bio_id": i}, [], OCCUPATIONS)[1])
        for i in range(30)
    }
    assert len(seen) > 1

    _, with_true = build_example_prompt(
        row, [], OCCUPATIONS, negative_mode="include_true"
    )
    assert with_true[0] == "poet" and with_true[1] == "nurse"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

PROMPT = build_prompt(
    "she is a nurse at the hospital",
    [doc("Female", content="she works at the hospital")],
    "interior_designer", "nurse", "dj",
)
CANDIDATES = ["interior_designer", "nurse", "dj"]


def brute_force_causal(model, tokenizer, prompt, text):
    prompt_ids = tokenizer(prompt)["input_ids"]
    full_ids = tokenizer(prompt + " " + text)["input_ids"]
    with torch.no_grad():
        log_probs = torch.log_softmax(model(input_ids=torch.tensor([full_ids])).logits[0], -1)
    return torch.stack(
        [log_probs[i - 1, full_ids[i]] for i in range(len(prompt_ids), len(full_ids))]
    )


@pytest.mark.parametrize("mode", ["first_token", "sum_logprob", "mean_logprob"])
def test_cached_causal_scoring_equals_full_forward_pass(mode):
    model, tokenizer = tiny_causal()
    scores = score_candidates(PROMPT, CANDIDATES, model, tokenizer, torch.device("cpu"), mode)
    for label, text in zip(CANDIDATES, ["interior designer", "nurse", "dj"]):
        tokens = brute_force_causal(model, tokenizer, PROMPT, text)
        expected = {
            "first_token": tokens[0], "sum_logprob": tokens.sum(), "mean_logprob": tokens.mean(),
        }[mode]
        assert scores[label] == pytest.approx(float(expected), abs=1e-4)


def test_seq2seq_scoring_equals_teacher_forced_loss():
    model, tokenizer = tiny_seq2seq()
    scores = score_candidates(
        PROMPT, CANDIDATES, model, tokenizer, torch.device("cpu"), "sum_logprob"
    )
    encoded = tokenizer(PROMPT, return_tensors="pt")
    for label, text in zip(CANDIDATES, ["interior designer", "nurse", "dj"]):
        labels = torch.tensor([tokenizer(text, add_special_tokens=False)["input_ids"]])
        with torch.no_grad():
            loss = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                labels=labels,
            ).loss
        assert scores[label] == pytest.approx(-float(loss) * labels.shape[1], abs=1e-4)


# ---------------------------------------------------------------------------
# Generator metrics
# ---------------------------------------------------------------------------

def test_generation_metrics_on_a_hand_worked_example():
    #            male: 2 positives, 2 negatives    female: 2 positives, 2 negatives
    labels = [1, 1, 0, 0, 1, 1, 0, 0]
    groups = ["Male"] * 4 + ["Female"] * 4
    selected = [1, 1, 1, 0, 1, 0, 0, 0]
    targets = ["nurse"] * 8

    result = summarize(selected, labels, groups, targets)
    assert result["acc"] == pytest.approx(6 / 8)
    assert result["prec"] == pytest.approx(3 / 4)
    assert result["rec"] == pytest.approx(3 / 4)
    assert result["rd"] == pytest.approx(abs(3 / 4 - 1 / 4))
    assert result["tpr_gap"] == pytest.approx(abs(1.0 - 0.5))

    report = per_occupation_report(selected, labels, groups, targets)
    assert report.loc[0, "f1"] == pytest.approx(0.75)

    gap, _ = one_vs_rest_gap(np.array(selected, bool), groups, targets, ("Male", "Female"))
    assert gap == pytest.approx(result["rd"])
    tpr, _ = one_vs_rest_gap(
        np.array(selected, bool), groups, targets, ("Male", "Female"), labels=labels
    )
    assert tpr == pytest.approx(result["tpr_gap"])


# ---------------------------------------------------------------------------
# CGLC
# ---------------------------------------------------------------------------

def biased_scores(n_per_cell=40, seed=0):
    """Synthetic logits in which male subjects get a boost on the target."""
    rng = np.random.default_rng(seed)
    occupations = ["nurse", "poet", "dj", "teacher"]
    rows = []
    for target in occupations:
        others = [o for o in occupations if o != target]
        for group in ("Male", "Female"):
            for y in (1, 0):
                for _ in range(n_per_cell):
                    distractors = list(rng.choice(others, 2, replace=False))
                    base = 0.6 if y else -0.2
                    boost = 0.5 if group == "Male" else 0.0
                    lean = rng.normal(0.2 if group == "Male" else -0.2, 0.1)
                    rows.append(
                        {
                            "bio_id": len(rows),
                            "target": target,
                            "y": y,
                            "subject_group": group,
                            "candidates": [distractors[0], target, distractors[1]],
                            "scores": [
                                float(rng.normal(0, 0.4)),
                                float(base + boost + rng.normal(0, 0.4)),
                                float(rng.normal(0, 0.4)),
                            ],
                            "signed_ndkl": [lean, -lean],
                            "skew_signal": [lean * 2, -lean * 2],
                            "groups": ["Male", "Female"],
                        }
                    )
    return pd.DataFrame(rows), occupations


FAST = {"stage_a_iterations": 60, "stage_b_passes": 2, "stage_c_passes": 2}


def test_cglc_lowers_risk_difference_on_train_and_held_out_data():
    train, occupations = biased_scores(seed=0)
    test, _ = biased_scores(seed=1)
    groups = ["Male", "Female"]

    before = summarize(
        uncalibrated_selected(test), test["y"], test["subject_group"], test["target"]
    )
    selected, calibrator, fit = fit_and_apply(
        train, test, occupations, groups, fit_options=FAST
    )
    after = summarize(selected, test["y"], test["subject_group"], test["target"])

    assert fit["stage_c_value"] <= fit["stage_b_value"] <= fit["initial_value"]
    assert fit["stage_c_value"] < fit["initial_value"]
    assert after["rd"] < before["rd"]


def test_cglc_can_target_the_tpr_gap():
    train, occupations = biased_scores(seed=0)
    _, calibrator, fit = fit_and_apply(
        train, train, occupations, ["Male", "Female"],
        objective="tpr_gap", fit_options=FAST,
    )
    assert calibrator.objective == "tpr_gap"
    assert fit["stage_c_value"] < fit["initial_value"]


def test_tpr_objective_requires_labels():
    calibrator = ConfidenceGatedLogitCalibrator(["nurse", "poet"])
    with pytest.raises(ValueError):
        calibrator.fit(
            logits=np.zeros((2, 2)), signed_rsbe=np.zeros((2, 2)),
            subject_groups=["Male", "Female"], target_occupations=["nurse", "nurse"],
            objective="tpr_gap",
        )


def test_ablation_switches_are_trained_saved_and_applied(tmp_path):
    train, occupations = biased_scores(seed=0)
    _, calibrator, _ = fit_and_apply(
        train, train, occupations, ["Male", "Female"],
        options={"use_gate": False, "use_delta_class": False}, fit_options=FAST,
    )
    assert all(value == 0.0 for value in calibrator.parameters.delta_class.values())

    from run_calibration_main import records_to_arrays

    arrays = records_to_arrays(train, occupations)
    arguments = dict(
        logits=arrays["logits"], signed_rsbe=arrays["signed_rsbe"],
        subject_groups=arrays["subject_groups"], candidate_mask=arrays["candidate_mask"],
    )
    output = calibrator.predict(**arguments)
    assert np.all(output["gate"][arrays["candidate_mask"]] == 1.0)

    calibrator.save(tmp_path / "cglc.json")
    loaded = ConfidenceGatedLogitCalibrator.load(tmp_path / "cglc.json")
    assert loaded.options == calibrator.options
    assert np.array_equal(loaded.predict(**arguments)["predictions"], output["predictions"])


def test_accuracy_guard_rejects_degenerate_solutions():
    train, occupations = biased_scores(seed=0)
    before = uncalibrated_selected(train)
    base_accuracy = float(np.mean(before == (train["y"] == 1)))
    selected, _, _ = fit_and_apply(
        train, train, occupations, ["Male", "Female"],
        fit_options={**FAST, "max_accuracy_drop": 0.02},
    )
    assert float(np.mean(selected == (train["y"] == 1))) >= base_accuracy - 0.02 - 1e-9


def test_one_vs_rest_selected():
    assert one_vs_rest_selected(["nurse", "poet"], ["nurse", "nurse"]).tolist() == [True, False]


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def test_fair_filter_applies_both_screens_and_keeps_order():
    docs = [doc("Male", content=name) for name in ["keep", "biased", "irrelevant", "keep too"]]

    def judge(prompt):
        if "fairness evaluation" in prompt:
            return "biased" not in prompt.split("Document:")[1]
        return "irrelevant" not in prompt.split("Document:")[1]

    assert [d.content for d in fair_filter(docs, "bio", judge)] == ["keep", "keep too"]
    assert [d.content for d in fair_filter(docs, "bio", judge, utility_screen=False)] == [
        "keep", "irrelevant", "keep too",
    ]
    # Utility screen removing everything falls back to the fairness-screened slate.
    assert len(fair_filter(docs, "bio", lambda p: "fairness evaluation" in p)) == 4
    assert fair_filter(docs, "bio", lambda p: False) == []


def test_in_context_preamble_goes_before_the_instruction():
    preamble = in_context_preamble(n=2)
    text = build_prompt("bio", [], "dj", "nurse", "poet", preamble=preamble)
    assert text.startswith(preamble)
    assert text.index(preamble) < text.index("You are a classifier.")


# ---------------------------------------------------------------------------
# FairFT
# ---------------------------------------------------------------------------

def test_fairft_loss_is_zero_when_the_distributions_agree():
    probabilities = torch.tensor([[0.1, 0.5, 0.9]])
    assert float(fairft_loss(probabilities.clone(), probabilities)) == pytest.approx(0.0, abs=1e-7)
    assert float(fairft_loss(torch.tensor([[0.9, 0.5, 0.1]]), probabilities)) > 0.1


def test_answer_probability_is_a_softmax_over_candidates():
    assert answer_probability({"a": 0.0, "b": 0.0}, "b") == pytest.approx(0.5)


class BagOfWordsEncoder(torch.nn.Module):
    """Stand-in for a SentenceTransformer: mean of learned word vectors."""

    def __init__(self, vocabulary):
        super().__init__()
        self.index = {word: i for i, word in enumerate(vocabulary)}
        self.embedding = torch.nn.EmbeddingBag(len(vocabulary), 8, mode="mean")

    def tokenize(self, texts):
        ids = [[self.index[w] for w in text.split()] for text in texts]
        width = max(len(row) for row in ids)
        padded = [row + [row[-1]] * (width - len(row)) for row in ids]
        return {"input_ids": torch.tensor(padded)}

    def forward(self, features):
        return {"sentence_embedding": self.embedding(features["input_ids"])}


def test_fairft_training_reduces_the_loss():
    torch.manual_seed(0)
    encoder = BagOfWordsEncoder(["nurse", "hospital", "poem", "music", "ward"])
    queries = pd.DataFrame({"bio_id": ["a", "b"], "bio": ["nurse ward", "nurse hospital"]})
    passages = [doc("Male", content=t) for t in ["hospital ward", "poem music", "music"]]
    pool = {"a": passages, "b": passages}
    probabilities = {"a": [0.9, 0.2, 0.1], "b": [0.8, 0.3, 0.1]}

    history = train_fairft(
        encoder, queries, pool, probabilities,
        epochs=60, batch_size=2, learning_rate=0.05, device="cpu", log_every=0,
    )
    assert history[-1] < history[0] * 0.5


# ---------------------------------------------------------------------------
# Generation runner
# ---------------------------------------------------------------------------

def test_run_system_writes_records_and_resumes(tmp_path):
    write_query_file(tmp_path / "queries", "nurse", n_per_cell=1)
    queries = load_queries(tmp_path / "queries")
    slates = {
        bio_id: [doc("Male", "nurse", "one"), doc("Female", "poet", "two")]
        for bio_id in queries["bio_id"]
    }
    calls = []

    def scorer(prompt, candidates):
        calls.append(prompt)
        return {c: float(i) for i, c in enumerate(candidates)}

    path = tmp_path / "out" / "fair_filter.jsonl"
    judge = lambda prompt: "two" not in prompt.split("Document:")[1]
    shown = run_system(
        "fair_filter", queries.iloc[:2], slates, OCCUPATIONS, scorer, path, Q,
        judge=judge, progress_every=0,
    )
    assert len(calls) == 2
    assert all([d.content for d in docs] == ["one"] for docs in shown.values())

    # Second call covers all four rows; the first two are not scored again.
    resumed = run_system(
        "fair_filter", queries, slates, OCCUPATIONS, scorer, path, Q,
        judge=judge, progress_every=0,
    )
    assert len(calls) == 4
    assert all([d.content for d in docs] == ["one"] for docs in resumed.values())

    records = load_records(path)
    assert len(records) == 4
    assert records["context_ranks"].tolist() == [[1]] * 4
    assert records["n_context"].tolist() == [1] * 4
    first = json.loads(path.read_text().splitlines()[0])
    assert first["groups"] == ["Male", "Female"]
    assert len(first["scores"]) == 3 and first["candidates"][1] == "nurse"
    # One male passage out of one: over-exposed relative to 50/50.
    assert first["signed_ndkl"][0] > 0 > first["signed_ndkl"][1]
