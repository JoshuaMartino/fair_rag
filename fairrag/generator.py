"""Generator integration for Fair-RAG.

This module connects:
    prompt.py -> Hugging Face model -> raw candidate scores -> CGLC

Both decoder-only models (Gemma, Llama 2, DeepSeek, Falcon) and
encoder-decoder models (Flan-T5) are supported. The class "logits" are the
log-probabilities the model assigns to each of the three candidate
occupation strings.

``score_candidates`` is the method the experiment runner uses. ``generate``
scores one prompt and applies a CGLC calibrator.
"""

from __future__ import annotations

import copy
import re
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)

from calibration import ConfidenceGatedLogitCalibrator
from prompt import display_label


_OPTION_PATTERN = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)

VALID_SCORE_MODES = {"first_token", "sum_logprob", "mean_logprob"}


def extract_candidates_from_prompt(prompt: str) -> tuple[str, str, str]:
    """Extract the three bullet-point occupations from a prompt.py prompt."""
    candidates = tuple(_OPTION_PATTERN.findall(prompt))

    if len(candidates) != 3:
        raise ValueError(
            "Expected exactly three '- occupation' lines in the prompt, "
            f"but found {len(candidates)}: {candidates}"
        )

    if len(set(candidates)) != 3:
        raise ValueError(f"Prompt candidates must be distinct: {candidates}")

    return candidates


def is_encoder_decoder(model: Any) -> bool:
    return bool(getattr(getattr(model, "config", None), "is_encoder_decoder", False))


def _resolve_model_and_tokenizer(
    model: str | Any,
    tokenizer: Any | None,
    device: str | None,
    torch_dtype: torch.dtype | str | None,
    trust_remote_code: bool,
) -> tuple[Any, Any, torch.device]:
    """Accept a model name or an already-loaded language model."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved_device = torch.device(device)

    dtype_lookup = {
        None: None,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }

    if isinstance(torch_dtype, str):
        if torch_dtype not in dtype_lookup:
            raise ValueError(
                "torch_dtype must be float16, bfloat16, float32, or None"
            )
        torch_dtype = dtype_lookup[torch_dtype]

    if isinstance(model, str):
        model_name = model

        if tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code,
            )

        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        model_class = (
            AutoModelForSeq2SeqLM
            if getattr(config, "is_encoder_decoder", False)
            else AutoModelForCausalLM
        )

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = model_class.from_pretrained(
            model_name,
            **model_kwargs,
        )
    elif tokenizer is None:
        raise ValueError(
            "When passing an already-loaded model, also pass its tokenizer."
        )

    if tokenizer.pad_token_id is None:
        fallback = tokenizer.eos_token or tokenizer.unk_token
        if fallback is None:
            raise ValueError("Tokenizer needs a pad, EOS, or unknown token.")
        tokenizer.pad_token = fallback

    model.to(resolved_device)
    model.eval()

    return model, tokenizer, resolved_device


def load_model(
    model: str | Any,
    tokenizer: Any | None = None,
    device: str | None = None,
    torch_dtype: torch.dtype | str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, Any, torch.device]:
    """Load (or pass through) a model and tokenizer, moved to the device."""
    return _resolve_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )


def _reduce(token_scores: torch.Tensor, score_mode: str) -> float:
    if score_mode == "first_token":
        return float(token_scores[0].item())
    if score_mode == "sum_logprob":
        return float(token_scores.sum().item())
    return float(token_scores.mean().item())


_warned_lengths: set[int] = set()


def _warn_if_too_long(length: int, tokenizer: Any) -> None:
    limit = getattr(tokenizer, "model_max_length", None)
    if limit is None or limit > 1_000_000 or length <= limit:
        return
    if limit not in _warned_lengths:
        _warned_lengths.add(limit)
        warnings.warn(
            f"A prompt has {length} tokens but the tokenizer reports a maximum "
            f"of {limit}. Lower max_passage_words so the slate fits.",
            stacklevel=3,
        )


def _continuation_ids(
    tokenizer: Any,
    prompt: str,
    prompt_ids: list[int],
    answer: str,
) -> tuple[int, list[int]]:
    """Token ids of ``answer`` as they occur right after ``prompt``.

    Returns (shared_prefix_length, full_ids). Tokenising prompt + answer
    together keeps the answer tokens exactly as the model would see them;
    the shared prefix is where the cached prompt can be reused.
    """
    full_ids = tokenizer(prompt + answer, add_special_tokens=True)["input_ids"]
    full_ids = [int(value) for value in full_ids]

    shared = 0
    for a, b in zip(prompt_ids, full_ids):
        if a != b:
            break
        shared += 1

    if shared == len(full_ids):
        raise ValueError(f"Candidate {answer!r} produced no tokens.")

    return shared, full_ids


@torch.inference_mode()
def _score_causal(
    prompt: str,
    answers: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    score_mode: str,
) -> list[float]:
    prompt_ids = [
        int(value)
        for value in tokenizer(prompt, add_special_tokens=True)["input_ids"]
    ]

    if not prompt_ids:
        start_token = tokenizer.bos_token_id
        if start_token is None:
            start_token = tokenizer.eos_token_id
        if start_token is None:
            raise ValueError("The model requires at least one prompt token.")
        prompt_ids = [int(start_token)]
        continuations = []
        for answer in answers:
            ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
            if not ids:
                raise ValueError(f"Candidate {answer!r} produced no tokens.")
            continuations.append((1, prompt_ids + [int(v) for v in ids]))
    else:
        continuations = [
            _continuation_ids(tokenizer, prompt, prompt_ids, answer)
            for answer in answers
        ]

    _warn_if_too_long(max(len(ids) for _, ids in continuations), tokenizer)

    # Run the shared prefix once and reuse its key/value cache for the three
    # candidates. The prefix must leave at least one token to predict from.
    shared = min(length for length, _ in continuations)
    shared = max(1, shared)

    prefix = torch.tensor([continuations[0][1][:shared]], device=device)
    prefix_output = model(
        input_ids=prefix,
        attention_mask=torch.ones_like(prefix),
        use_cache=True,
    )
    last_log_probs = torch.log_softmax(prefix_output.logits[0, -1].float(), dim=-1)
    cache = prefix_output.past_key_values

    scores = []
    for _, full_ids in continuations:
        answer_ids = torch.tensor(full_ids[shared:], device=device)
        token_scores = [last_log_probs[answer_ids[0]]]

        if answer_ids.numel() > 1:
            step_input = answer_ids[:-1].unsqueeze(0)
            attention_mask = torch.ones(
                (1, shared + step_input.shape[1]), dtype=torch.long, device=device
            )
            step_output = model(
                input_ids=step_input,
                attention_mask=attention_mask,
                # Cache objects are updated in place, so each candidate needs
                # its own copy of the prompt cache.
                past_key_values=copy.deepcopy(cache),
                use_cache=True,
            )
            step_log_probs = torch.log_softmax(step_output.logits[0].float(), dim=-1)
            positions = torch.arange(step_input.shape[1], device=device)
            token_scores.extend(step_log_probs[positions, answer_ids[1:]])

        scores.append(_reduce(torch.stack(token_scores), score_mode))

    return scores


@torch.inference_mode()
def _score_seq2seq(
    prompt: str,
    answers: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    score_mode: str,
) -> list[float]:
    encoded = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    _warn_if_too_long(int(input_ids.shape[1]), tokenizer)

    # Encode the long prompt once; only the short decoder pass is repeated.
    encoder_outputs = model.get_encoder()(
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    start_token = model.config.decoder_start_token_id
    if start_token is None:
        start_token = tokenizer.pad_token_id

    scores = []
    for answer in answers:
        answer_ids = tokenizer(answer.strip(), add_special_tokens=False)["input_ids"]
        if not answer_ids:
            raise ValueError(f"Candidate {answer!r} produced no tokens.")
        answer_ids = torch.tensor(answer_ids, device=device)

        decoder_input_ids = torch.cat(
            [torch.tensor([start_token], device=device), answer_ids[:-1]]
        ).unsqueeze(0)

        output = model(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
        )
        log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
        positions = torch.arange(answer_ids.numel(), device=device)
        scores.append(_reduce(log_probs[positions, answer_ids], score_mode))

    return scores


def score_candidates(
    prompt: str,
    candidates: Sequence[str],
    model: Any,
    tokenizer: Any,
    device: torch.device,
    score_mode: str = "mean_logprob",
    answer_prefix: str = " ",
) -> dict[str, float]:
    """Score every candidate occupation as the answer to ``prompt``.

    Candidates are occupation labels; they are scored in the form shown in
    the prompt (underscores replaced by spaces).

    ``mean_logprob`` is the default because occupation labels have different
    numbers of tokens and several share a first token. CGLC must be trained
    and tested with the same score mode.

    Returns
    -------
    dict
        {candidate label: score}, in the order given.
    """
    if score_mode not in VALID_SCORE_MODES:
        raise ValueError(f"score_mode must be one of {sorted(VALID_SCORE_MODES)}")

    answers = [answer_prefix + display_label(candidate) for candidate in candidates]

    if is_encoder_decoder(model):
        values = _score_seq2seq(prompt, answers, model, tokenizer, device, score_mode)
    else:
        values = _score_causal(prompt, answers, model, tokenizer, device, score_mode)

    return {candidate: value for candidate, value in zip(candidates, values)}


def score_candidate(
    prompt: str,
    candidate: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    score_mode: str = "mean_logprob",
    answer_prefix: str = " ",
) -> float:
    """Score a single candidate string (see ``score_candidates``)."""
    return score_candidates(
        prompt=prompt,
        candidates=[candidate],
        model=model,
        tokenizer=tokenizer,
        device=device,
        score_mode=score_mode,
        answer_prefix=answer_prefix,
    )[candidate]


def _load_calibrator(
    cglc: ConfidenceGatedLogitCalibrator | str | Path | None,
) -> ConfidenceGatedLogitCalibrator | None:
    if cglc is None:
        return None
    if isinstance(cglc, ConfidenceGatedLogitCalibrator):
        return cglc
    return ConfidenceGatedLogitCalibrator.load(cglc)


def _signed_rsbe_vector(
    signed_rsbe: Sequence[float] | dict[str, float] | dict[str, Any],
    groups: Sequence[str],
) -> np.ndarray:
    """Accept either a vector, signed-score dict, or full rsbe() result."""
    if isinstance(signed_rsbe, dict):
        if "signed_scores" in signed_rsbe:
            signed_rsbe = signed_rsbe["signed_scores"]
        values = [float(signed_rsbe[group]) for group in groups]
    else:
        values = [float(value) for value in signed_rsbe]

    if len(values) != len(groups):
        raise ValueError(
            f"Expected {len(groups)} signed RSBE values, found {len(values)}."
        )

    return np.asarray([values], dtype=np.float64)


def generate(
    prompt: str,
    model: str | Any,
    cglc: ConfidenceGatedLogitCalibrator | str | Path | None,
    signed_rsbe: Sequence[float] | dict[str, float] | dict[str, Any],
    subject_group: str,
    candidates: Sequence[str] | None = None,
    tokenizer: Any | None = None,
    device: str | None = None,
    torch_dtype: torch.dtype | str | None = None,
    score_mode: str = "mean_logprob",
    trust_remote_code: bool = False,
    **calibration_options: Any,
) -> dict[str, Any]:
    """Send one prompt to the model, score its choices, and apply CGLC.

    Parameters
    ----------
    prompt:
        Complete prompt returned by a method in prompt.py.
    model:
        Hugging Face model name/path or an already-loaded model.
    cglc:
        Loaded calibrator, path to cglc.json, or None for an uncalibrated run.
    signed_rsbe:
        The ``rsbe_results[bio_id]`` object, its ``signed_scores`` dictionary,
        or a vector ordered like ``cglc.groups``.
    subject_group:
        Biography subject group, e.g. ``"Male"`` or ``"Female"``.
    candidates:
        Optional explicit candidate list (occupation labels). If omitted, the
        three bullet-point professions are extracted from the prompt.

    Returns
    -------
    dict
        Base prediction, calibrated prediction, raw/calibrated scores, gate,
        shift, and the candidate order.
    """
    calibrator = _load_calibrator(cglc)

    if candidates is None:
        candidates = extract_candidates_from_prompt(prompt)
        if calibrator is not None:
            # The prompt shows "interior designer"; the calibrator knows the
            # label "interior_designer".
            by_display = {display_label(o): o for o in calibrator.occupations}
            candidates = tuple(by_display.get(c, c) for c in candidates)
    else:
        candidates = tuple(str(value).strip() for value in candidates)
        if len(candidates) != 3 or len(set(candidates)) != 3:
            raise ValueError("candidates must contain exactly three distinct labels")

    model, tokenizer, resolved_device = _resolve_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
    )

    raw_scores = score_candidates(
        prompt=prompt,
        candidates=candidates,
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        score_mode=score_mode,
    )
    base_prediction = max(raw_scores, key=raw_scores.get)

    if calibrator is None:
        return {
            "base_prediction": base_prediction,
            "prediction": base_prediction,
            "prediction_changed": False,
            "candidates": list(candidates),
            "raw_scores": raw_scores,
            "calibrated_scores": dict(raw_scores),
            "gate": {candidate: 1.0 for candidate in candidates},
            "shift": {candidate: 0.0 for candidate in candidates},
            "score_mode": score_mode,
        }

    if subject_group not in calibrator.groups:
        raise ValueError(
            f"subject_group {subject_group!r} is not in {calibrator.groups}"
        )

    unknown_candidates = set(candidates) - set(calibrator.occupations)
    if unknown_candidates:
        raise KeyError(
            f"CGLC does not contain occupations: {sorted(unknown_candidates)}"
        )

    logits = np.zeros((1, len(calibrator.occupations)), dtype=np.float64)
    candidate_mask = np.zeros_like(logits, dtype=bool)

    for candidate, score in raw_scores.items():
        index = calibrator.occupation_to_index[candidate]
        logits[0, index] = score
        candidate_mask[0, index] = True

    signed_rsbe_array = _signed_rsbe_vector(
        signed_rsbe=signed_rsbe,
        groups=calibrator.groups,
    )

    calibrated = calibrator.predict(
        logits=logits,
        signed_rsbe=signed_rsbe_array,
        subject_groups=np.asarray([subject_group]),
        candidate_mask=candidate_mask,
        **calibration_options,
    )

    prediction = str(calibrated["predictions"][0])
    calibrated_scores = {}
    gate = {}
    shift = {}

    for candidate in candidates:
        index = calibrator.occupation_to_index[candidate]
        calibrated_scores[candidate] = float(
            calibrated["calibrated_logits"][0, index]
        )
        gate[candidate] = float(calibrated["gate"][0, index])
        shift[candidate] = float(calibrated["shift"][0, index])

    return {
        "base_prediction": base_prediction,
        "prediction": prediction,
        "prediction_changed": prediction != base_prediction,
        "candidates": list(candidates),
        "raw_scores": raw_scores,
        "calibrated_scores": calibrated_scores,
        "gate": gate,
        "shift": shift,
        "signed_rsbe": {
            group: float(signed_rsbe_array[0, index])
            for index, group in enumerate(calibrator.groups)
        },
        "score_mode": score_mode,
    }


class FairRAGGenerator:
    """Load the model and CGLC once, then call ``generate`` repeatedly."""

    def __init__(
        self,
        model: str | Any,
        cglc: ConfidenceGatedLogitCalibrator | str | Path | None,
        tokenizer: Any | None = None,
        device: str | None = None,
        torch_dtype: torch.dtype | str | None = None,
        score_mode: str = "mean_logprob",
        trust_remote_code: bool = False,
    ) -> None:
        self.model, self.tokenizer, self.device = _resolve_model_and_tokenizer(
            model=model,
            tokenizer=tokenizer,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        self.cglc = _load_calibrator(cglc)
        self.score_mode = score_mode

    def score(self, prompt: str, candidates: Sequence[str]) -> dict[str, float]:
        """Raw candidate scores for one prompt (no calibration)."""
        return score_candidates(
            prompt=prompt,
            candidates=candidates,
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            score_mode=self.score_mode,
        )

    def generate(
        self,
        prompt: str,
        signed_rsbe: Sequence[float] | dict[str, float] | dict[str, Any],
        subject_group: str,
        candidates: Sequence[str] | None = None,
        **calibration_options: Any,
    ) -> dict[str, Any]:
        return generate(
            prompt=prompt,
            model=self.model,
            tokenizer=self.tokenizer,
            cglc=self.cglc,
            signed_rsbe=signed_rsbe,
            subject_group=subject_group,
            candidates=candidates,
            device=str(self.device),
            score_mode=self.score_mode,
            **calibration_options,
        )
