"""Generator integration for Fair-RAG.

This module connects:
    prompt.py -> Hugging Face model -> raw candidate scores -> CGLC

The public method is ``generate``. It accepts the already-built prompt, a
loaded or saved CGLC calibrator, and a model. The three occupation choices are
read from the prompt unless supplied explicitly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from calibration import ConfidenceGatedLogitCalibrator


_OPTION_PATTERN = re.compile(r"^\s*-\s*(.+?)\s*$", re.MULTILINE)


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


def _resolve_model_and_tokenizer(
    model: str | Any,
    tokenizer: Any | None,
    device: str | None,
    torch_dtype: torch.dtype | str | None,
    trust_remote_code: bool,
) -> tuple[Any, Any, torch.device]:
    """Accept a model name or an already-loaded causal language model."""
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

        model_kwargs = {
            "trust_remote_code": trust_remote_code,
        }
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

        model = AutoModelForCausalLM.from_pretrained(
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


@torch.inference_mode()
def score_candidate(
    prompt: str,
    candidate: str,
    model: Any,
    tokenizer: Any,
    device: torch.device,
    score_mode: str = "mean_logprob",
    answer_prefix: str = " ",
) -> float:
    """Score a complete candidate string with a causal language model.

    ``mean_logprob`` is the default because occupation labels may contain
    different numbers of tokens. CGLC must be trained and tested with the same
    score mode.
    """
    valid_modes = {"first_token", "sum_logprob", "mean_logprob"}
    if score_mode not in valid_modes:
        raise ValueError(f"score_mode must be one of {sorted(valid_modes)}")

    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        return_tensors="pt",
    )["input_ids"][0]

    candidate_ids = tokenizer(
        answer_prefix + candidate,
        add_special_tokens=False,
        return_tensors="pt",
    )["input_ids"][0]

    if candidate_ids.numel() == 0:
        raise ValueError(f"Candidate {candidate!r} produced no tokens.")

    if prompt_ids.numel() == 0:
        start_token = tokenizer.bos_token_id or tokenizer.eos_token_id
        if start_token is None:
            raise ValueError("The model requires at least one prompt token.")
        prompt_ids = torch.tensor([start_token], dtype=torch.long)

    input_ids = torch.cat([prompt_ids, candidate_ids]).unsqueeze(0).to(device)
    attention_mask = torch.ones_like(input_ids)

    output = model(input_ids=input_ids, attention_mask=attention_mask)
    log_probabilities = torch.log_softmax(output.logits[0], dim=-1)

    prompt_length = int(prompt_ids.numel())
    token_positions = torch.arange(
        prompt_length - 1,
        prompt_length + int(candidate_ids.numel()) - 1,
        device=device,
    )
    candidate_ids = candidate_ids.to(device)
    token_scores = log_probabilities[token_positions, candidate_ids]

    if score_mode == "first_token":
        return float(token_scores[0].item())
    if score_mode == "sum_logprob":
        return float(token_scores.sum().item())
    return float(token_scores.mean().item())


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
        Hugging Face model name/path or an already-loaded causal LM.
    cglc:
        Loaded calibrator, path to cglc.json, or None for an uncalibrated run.
    signed_rsbe:
        The ``rsbe_results[bio_id]`` object, its ``signed_scores`` dictionary,
        or a vector ordered like ``cglc.groups``.
    subject_group:
        Biography subject group, e.g. ``"Male"`` or ``"Female"``.
    candidates:
        Optional explicit candidate list. If omitted, the three bullet-point
        professions are extracted from the prompt.

    Returns
    -------
    dict
        Base prediction, calibrated prediction, raw/calibrated scores, gate,
        shift, and the candidate order.
    """
    calibrator = _load_calibrator(cglc)

    if candidates is None:
        candidates = extract_candidates_from_prompt(prompt)
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

    raw_scores = {
        candidate: score_candidate(
            prompt=prompt,
            candidate=candidate,
            model=model,
            tokenizer=tokenizer,
            device=resolved_device,
            score_mode=score_mode,
        )
        for candidate in candidates
    }
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
