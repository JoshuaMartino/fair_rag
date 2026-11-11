"""Confidence-Gated Logit Calibration (CGLC).

Public API
----------
Training::

    calibrator = ConfidenceGatedLogitCalibrator(occupations)
    fit_result = calibrator.fit(
        logits=train_logits,
        signed_rsbe=train_signed_rsbe,
        subject_groups=train_groups,
        target_occupations=train_targets,
        candidate_mask=train_candidate_mask,
    )
    calibrator.save("results/calibration/cglc.json")

Inference::

    calibrator = ConfidenceGatedLogitCalibrator.load(
        "results/calibration/cglc.json"
    )
    result = calibrator.predict(
        logits=test_logits,
        signed_rsbe=test_signed_rsbe,
        subject_groups=test_groups,
        candidate_mask=test_candidate_mask,
    )

The objective is the paper's one-vs-rest task: every example belongs to a
target occupation o (its test file) and counts as selected when the
calibrated argmax over its three candidates is o. ``objective="rd"`` minimises
the mean risk difference (Equation 11); ``objective="tpr_gap"`` minimises the
mean TPR gap (equality of opportunity, Section 5.4) and needs ``labels``.

Ablations (Table 5) are trained and applied with the same switches::

    calibrator.fit(..., use_gate=False)          # "No eta"
    calibrator.fit(..., use_delta_class=False)   # "No delta_o"
    calibrator.fit(..., use_b_class=False)       # "No b_o"

Notes
-----
The appendix describes six global scalars: delta, gamma, two group offsets,
and separate beta weights for positive and negative standardized residuals.
Stage C learns the per-class corrections delta_o and b_o. Search ranges are
not reported in the paper, so the defaults below are explicit reproduction
hyperparameters and can be changed through ``fit``.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass
class CGLCParameters:
    """Learned CGLC parameters."""

    delta: float = 0.0
    gamma: float = 0.0
    b_group: dict[str, float] = field(default_factory=dict)
    beta_positive: float = 0.0
    beta_negative: float = 0.0
    delta_class: dict[str, float] = field(default_factory=dict)
    b_class: dict[str, float] = field(default_factory=dict)
    max_shift: float = 1.0


DEFAULT_OPTIONS = {
    "use_gate": True,
    "use_residual": True,
    "use_delta_class": True,
    "use_b_class": True,
}

VALID_OBJECTIVES = {"rd", "tpr_gap"}


def one_vs_rest_selected(predictions: Any, example_targets: Any) -> np.ndarray:
    """1 where the predicted occupation is the example's target occupation."""
    predictions = np.asarray(predictions).astype(str)
    example_targets = np.asarray(example_targets).astype(str)
    return predictions == example_targets


def one_vs_rest_gap(
    selected: Any,
    subject_groups: Any,
    example_targets: Any,
    groups: Sequence[str],
    labels: Any | None = None,
) -> tuple[float, dict[str, float]]:
    """Mean and per-occupation |selection-rate gap| between the two groups.

    With ``labels`` omitted this is the risk difference RD(o) of Equation 11,
    computed over all examples of occupation o's task. With ``labels`` given,
    only positives (label 1) are used, which gives the TPR gap of Appendix C.4.
    Occupations in which a group has no examples are left out of the mean.
    """
    selected = np.asarray(selected, dtype=bool)
    subject_groups = np.asarray(subject_groups).astype(str)
    example_targets = np.asarray(example_targets).astype(str)
    group_a, group_b = [str(group) for group in groups]

    keep = np.ones(len(selected), dtype=bool)
    if labels is not None:
        keep = np.asarray(labels).astype(int) == 1

    per_class: dict[str, float] = {}
    for occupation in dict.fromkeys(example_targets.tolist()):
        rows = keep & (example_targets == occupation)
        rows_a = rows & (subject_groups == group_a)
        rows_b = rows & (subject_groups == group_b)
        if not rows_a.any() or not rows_b.any():
            continue
        per_class[occupation] = abs(
            float(selected[rows_a].mean()) - float(selected[rows_b].mean())
        )

    if not per_class:
        raise ValueError("no occupation has examples from both protected groups")

    return float(np.mean(list(per_class.values()))), per_class


class ConfidenceGatedLogitCalibrator:
    """Fit, save, load, and apply Confidence-Gated Logit Calibration."""

    def __init__(
        self,
        occupations: Sequence[str],
        groups: Sequence[str] = ("Male", "Female"),
        epsilon: float = 1e-8,
        random_state: int = 42,
    ) -> None:
        self.occupations = [str(value) for value in occupations]
        self.groups = [str(value) for value in groups]

        if not self.occupations:
            raise ValueError("occupations cannot be empty")
        if len(set(self.occupations)) != len(self.occupations):
            raise ValueError("occupations must be unique")
        if len(self.groups) != 2 or len(set(self.groups)) != 2:
            raise ValueError("groups must contain exactly two distinct labels")
        if epsilon <= 0:
            raise ValueError("epsilon must be greater than zero")

        self.occupation_to_index = {
            occupation: index
            for index, occupation in enumerate(self.occupations)
        }
        self.group_to_index = {
            group: index
            for index, group in enumerate(self.groups)
        }
        self.epsilon = float(epsilon)
        self.random_state = int(random_state)
        self.rng = np.random.default_rng(self.random_state)

        self.rsbe_means: np.ndarray | None = None
        self.rsbe_stds: np.ndarray | None = None
        self.parameters = self._zero_parameters()

        # Component switches and objective used at fit time. They are saved
        # with the parameters and reused at inference (Appendix A.6, iv-v).
        self.options = dict(DEFAULT_OPTIONS)
        self.objective = "rd"
        self.rd_mode = "one_vs_rest"

    def _zero_parameters(self, max_shift: float = 1.0) -> CGLCParameters:
        return CGLCParameters(
            b_group={group: 0.0 for group in self.groups},
            delta_class={occupation: 0.0 for occupation in self.occupations},
            b_class={occupation: 0.0 for occupation in self.occupations},
            max_shift=float(max_shift),
        )

    def _validate_inputs(
        self,
        logits: Any,
        signed_rsbe: Any,
        subject_groups: Any,
        candidate_mask: Any | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        logits = np.asarray(logits, dtype=np.float64)
        signed_rsbe = np.asarray(signed_rsbe, dtype=np.float64)
        subject_groups = np.asarray(subject_groups).astype(str)

        if logits.ndim != 2:
            raise ValueError("logits must be a two-dimensional array")
        expected_logits_shape = (len(subject_groups), len(self.occupations))
        if logits.shape != expected_logits_shape:
            raise ValueError(
                f"logits has shape {logits.shape}; expected {expected_logits_shape}"
            )

        expected_rsbe_shape = (len(subject_groups), len(self.groups))
        if signed_rsbe.shape != expected_rsbe_shape:
            raise ValueError(
                f"signed_rsbe has shape {signed_rsbe.shape}; "
                f"expected {expected_rsbe_shape}"
            )

        unexpected = set(subject_groups) - set(self.groups)
        if unexpected:
            raise ValueError(f"unexpected protected-group labels: {sorted(unexpected)}")

        if candidate_mask is None:
            candidate_mask = np.ones_like(logits, dtype=bool)
        else:
            candidate_mask = np.asarray(candidate_mask, dtype=bool)
            if candidate_mask.shape != logits.shape:
                raise ValueError("candidate_mask must have the same shape as logits")

        if np.any(candidate_mask.sum(axis=1) == 0):
            raise ValueError("every example must contain at least one candidate")
        if not np.isfinite(logits[candidate_mask]).all():
            raise ValueError("candidate logits must all be finite")
        if not np.isfinite(signed_rsbe).all():
            raise ValueError("signed_rsbe must contain only finite values")

        return logits, signed_rsbe, subject_groups, candidate_mask

    def fit_rsbe_statistics(
        self,
        signed_rsbe: Any,
        target_occupations: Any,
    ) -> "ConfidenceGatedLogitCalibrator":
        """Fit class-conditional RSBE means and standard deviations."""
        signed_rsbe = np.asarray(signed_rsbe, dtype=np.float64)
        target_occupations = np.asarray(target_occupations).astype(str)
        expected_shape = (len(target_occupations), len(self.groups))

        if signed_rsbe.shape != expected_shape:
            raise ValueError(
                f"signed_rsbe has shape {signed_rsbe.shape}; expected {expected_shape}"
            )
        unknown = set(target_occupations) - set(self.occupations)
        if unknown:
            raise ValueError(f"unknown target occupations: {sorted(unknown)}")

        global_mean = signed_rsbe.mean(axis=0)
        global_std = signed_rsbe.std(axis=0)
        global_std = np.where(global_std < self.epsilon, 1.0, global_std)

        self.rsbe_means = np.zeros(
            (len(self.occupations), len(self.groups)), dtype=np.float64
        )
        self.rsbe_stds = np.ones_like(self.rsbe_means)

        for occupation, index in self.occupation_to_index.items():
            rows = target_occupations == occupation
            if not np.any(rows):
                self.rsbe_means[index] = global_mean
                self.rsbe_stds[index] = global_std
                continue

            values = signed_rsbe[rows]
            self.rsbe_means[index] = values.mean(axis=0)
            occupation_std = values.std(axis=0)
            self.rsbe_stds[index] = np.where(
                occupation_std < self.epsilon, 1.0, occupation_std
            )

        return self

    def standardize_rsbe(self, signed_rsbe: Any) -> np.ndarray:
        """Return z_o[a] with shape examples x occupations x groups."""
        if self.rsbe_means is None or self.rsbe_stds is None:
            raise RuntimeError("RSBE statistics have not been fitted")

        signed_rsbe = np.asarray(signed_rsbe, dtype=np.float64)
        if signed_rsbe.ndim != 2 or signed_rsbe.shape[1] != len(self.groups):
            raise ValueError(
                "signed_rsbe must have shape (examples, number_of_groups)"
            )

        return (
            signed_rsbe[:, None, :] - self.rsbe_means[None, :, :]
        ) / (self.rsbe_stds[None, :, :] + self.epsilon)

    def calculate_shift(
        self,
        subject_residual: Any,
        subject_groups: Any,
        parameters: CGLCParameters | None = None,
    ) -> np.ndarray:
        """Calculate clipped Delta for every example and occupation."""
        if parameters is None:
            parameters = self.parameters

        subject_residual = np.asarray(subject_residual, dtype=np.float64)
        subject_groups = np.asarray(subject_groups).astype(str)
        expected_shape = (len(subject_groups), len(self.occupations))
        if subject_residual.shape != expected_shape:
            raise ValueError(
                f"subject_residual has shape {subject_residual.shape}; "
                f"expected {expected_shape}"
            )

        missing_group_parameters = set(subject_groups) - set(parameters.b_group)
        if missing_group_parameters:
            raise ValueError(
                "parameters do not contain offsets for groups: "
                f"{sorted(missing_group_parameters)}"
            )

        group_offsets = np.asarray(
            [parameters.b_group[group] for group in subject_groups],
            dtype=np.float64,
        )[:, None]
        delta_class = np.asarray(
            [parameters.delta_class.get(o, 0.0) for o in self.occupations],
            dtype=np.float64,
        )
        b_class = np.asarray(
            [parameters.b_class.get(o, 0.0) for o in self.occupations],
            dtype=np.float64,
        )
        beta = np.where(
            subject_residual >= 0,
            parameters.beta_positive,
            parameters.beta_negative,
        )

        shift = (
            parameters.delta
            + group_offsets
            + delta_class[None, :]
            + b_class[None, :]
            + beta * subject_residual
        )
        return np.clip(shift, -parameters.max_shift, parameters.max_shift)

    def calibrate_logits(
        self,
        logits: Any,
        signed_rsbe: Any,
        subject_groups: Any,
        candidate_mask: Any | None = None,
        parameters: CGLCParameters | None = None,
        use_gate: bool | None = None,
        use_residual: bool | None = None,
        use_delta_class: bool | None = None,
        use_b_class: bool | None = None,
    ) -> dict[str, np.ndarray]:
        """Apply CGLC to raw pre-softmax candidate logits.

        The ``use_*`` switches default to the ones the calibrator was fitted
        with.
        """
        logits, signed_rsbe, subject_groups, candidate_mask = self._validate_inputs(
            logits, signed_rsbe, subject_groups, candidate_mask
        )
        if parameters is None:
            parameters = self.parameters
        if use_gate is None:
            use_gate = self.options["use_gate"]
        if use_residual is None:
            use_residual = self.options["use_residual"]
        if use_delta_class is None:
            use_delta_class = self.options["use_delta_class"]
        if use_b_class is None:
            use_b_class = self.options["use_b_class"]

        parameters = deepcopy(parameters)
        if not use_residual:
            parameters.beta_positive = 0.0
            parameters.beta_negative = 0.0
        if not use_delta_class:
            parameters.delta_class = {o: 0.0 for o in self.occupations}
        if not use_b_class:
            parameters.b_class = {o: 0.0 for o in self.occupations}

        z_scores = self.standardize_rsbe(signed_rsbe)
        group_indices = np.asarray(
            [self.group_to_index[group] for group in subject_groups], dtype=int
        )
        n_examples, n_classes = logits.shape
        subject_residual = z_scores[
            np.arange(n_examples)[:, None],
            np.arange(n_classes)[None, :],
            group_indices[:, None],
        ]
        shift = self.calculate_shift(subject_residual, subject_groups, parameters)

        candidate_counts = candidate_mask.sum(axis=1, keepdims=True)
        mean_candidate_logits = (
            np.where(candidate_mask, logits, 0.0).sum(axis=1, keepdims=True)
            / candidate_counts
        )
        if use_gate:
            gate = np.exp(
                -parameters.gamma * np.abs(logits - mean_candidate_logits)
            )
        else:
            gate = np.ones_like(logits, dtype=np.float64)
        gate = np.where(candidate_mask, gate, 0.0)

        calibrated_logits = logits - gate * shift
        calibrated_logits = np.where(candidate_mask, calibrated_logits, -np.inf)

        return {
            "calibrated_logits": calibrated_logits,
            "gate": gate,
            "shift": shift,
            "subject_residual": subject_residual,
        }

    def predict(
        self,
        logits: Any,
        signed_rsbe: Any,
        subject_groups: Any,
        candidate_mask: Any | None = None,
        **calibration_options: Any,
    ) -> dict[str, np.ndarray]:
        """Return calibrated logits, diagnostics, and occupation predictions."""
        output = self.calibrate_logits(
            logits=logits,
            signed_rsbe=signed_rsbe,
            subject_groups=subject_groups,
            candidate_mask=candidate_mask,
            **calibration_options,
        )
        predicted_indices = np.argmax(output["calibrated_logits"], axis=1)
        output["predicted_indices"] = predicted_indices
        output["predictions"] = np.asarray(
            [self.occupations[index] for index in predicted_indices]
        )
        return output

    def mean_risk_difference(
        self,
        predictions: Any,
        subject_groups: Any,
    ) -> tuple[float, dict[str, float]]:
        """Macro-average absolute predicted-class selection-rate gaps."""
        predictions = np.asarray(predictions).astype(str)
        subject_groups = np.asarray(subject_groups).astype(str)
        group_a, group_b = self.groups
        rows_a = subject_groups == group_a
        rows_b = subject_groups == group_b

        if not np.any(rows_a) or not np.any(rows_b):
            raise ValueError("both protected groups are required to calculate RD")

        per_class = {}
        for occupation in self.occupations:
            rate_a = float(np.mean(predictions[rows_a] == occupation))
            rate_b = float(np.mean(predictions[rows_b] == occupation))
            per_class[occupation] = abs(rate_a - rate_b)

        return float(np.mean(list(per_class.values()))), per_class

    def _objective(
        self,
        parameters: CGLCParameters,
        data: Mapping[str, Any],
    ) -> float:
        """Value to minimise for one candidate parameter set."""
        predictions = self.predict(
            logits=data["logits"],
            signed_rsbe=data["signed_rsbe"],
            subject_groups=data["subject_groups"],
            candidate_mask=data["candidate_mask"],
            parameters=parameters,
        )["predictions"]

        if self.objective == "rd" and self.rd_mode == "multiclass":
            value, _ = self.mean_risk_difference(
                predictions, data["subject_groups"]
            )
            return value

        selected = one_vs_rest_selected(predictions, data["example_targets"])

        accuracy_floor = data.get("accuracy_floor")
        if accuracy_floor is not None:
            accuracy = float(np.mean(selected == (data["labels"] == 1)))
            if accuracy < accuracy_floor:
                return float("inf")

        value, _ = one_vs_rest_gap(
            selected,
            data["subject_groups"],
            data["example_targets"],
            self.groups,
            labels=data["labels"] if self.objective == "tpr_gap" else None,
        )
        return value

    @staticmethod
    def _subset(data: Mapping[str, Any], indices: np.ndarray) -> dict[str, Any]:
        subset = dict(data)
        for key in (
            "logits", "signed_rsbe", "subject_groups", "candidate_mask",
            "example_targets", "labels",
        ):
            if data.get(key) is not None:
                subset[key] = data[key][indices]
        return subset

    def _stage_a_random_search(
        self,
        data: Mapping[str, Any],
        iterations: int,
        subset_fraction: float,
        max_shift: float,
        ranges: Mapping[str, tuple[float, float]],
    ) -> tuple[CGLCParameters, float]:
        """Stage A: coarse global search on a random development subset."""
        if iterations < 1:
            raise ValueError("iterations must be at least one")
        if not 0 < subset_fraction <= 1:
            raise ValueError("subset_fraction must be in (0, 1]")

        size = len(data["subject_groups"])
        subset_size = max(1, int(round(size * subset_fraction)))
        indices = self.rng.choice(size, subset_size, replace=False)
        subset = self._subset(data, indices)

        best = self._zero_parameters(max_shift=max_shift)
        best_objective = self._objective(best, subset)

        for _ in range(iterations):
            candidate = self._zero_parameters(max_shift=max_shift)
            candidate.delta = float(self.rng.uniform(*ranges["delta"]))
            candidate.gamma = float(self.rng.uniform(*ranges["gamma"]))
            candidate.beta_positive = float(
                self.rng.uniform(*ranges["beta_positive"])
            )
            candidate.beta_negative = float(
                self.rng.uniform(*ranges["beta_negative"])
            )
            candidate.b_group = {
                group: float(self.rng.uniform(*ranges["b_group"]))
                for group in self.groups
            }

            objective = self._objective(candidate, subset)
            if objective < best_objective:
                best, best_objective = candidate, objective

        return best, best_objective

    @staticmethod
    def _get_global(parameters: CGLCParameters, name: str) -> float:
        if name.startswith("b_group::"):
            return parameters.b_group[name.split("::", 1)[1]]
        return float(getattr(parameters, name))

    @staticmethod
    def _set_global(
        parameters: CGLCParameters,
        name: str,
        value: float,
    ) -> CGLCParameters:
        result = deepcopy(parameters)
        if name.startswith("b_group::"):
            result.b_group[name.split("::", 1)[1]] = float(value)
        else:
            setattr(result, name, float(value))
        return result

    def _stage_b_coordinate_descent(
        self,
        initial_parameters: CGLCParameters,
        data: Mapping[str, Any],
        coordinate_offsets: Mapping[str, Sequence[float]],
        max_passes: int,
    ) -> tuple[CGLCParameters, float]:
        """Stage B: coordinate descent over the six global scalars."""
        best = deepcopy(initial_parameters)
        best_objective = self._objective(best, data)

        coordinates = ["delta"]
        if self.options["use_gate"]:
            coordinates.append("gamma")
        coordinates.extend(f"b_group::{group}" for group in self.groups)
        if self.options["use_residual"]:
            coordinates.extend(["beta_positive", "beta_negative"])

        for _ in range(max_passes):
            pass_improved = False
            for name in coordinates:
                base = self._get_global(best, name)
                offsets = coordinate_offsets[
                    "gamma" if name == "gamma" else
                    "beta" if name.startswith("beta_") else
                    "default"
                ]
                coordinate_best = best
                coordinate_objective = best_objective

                for offset in offsets:
                    value = base + float(offset)
                    if name == "gamma":
                        value = max(0.0, value)
                    candidate = self._set_global(best, name, value)
                    objective = self._objective(candidate, data)
                    if objective < coordinate_objective:
                        coordinate_best = candidate
                        coordinate_objective = objective

                if coordinate_objective < best_objective:
                    best = coordinate_best
                    best_objective = coordinate_objective
                    pass_improved = True

            if not pass_improved:
                break

        return best, best_objective

    def _stage_c_class_tuning(
        self,
        initial_parameters: CGLCParameters,
        data: Mapping[str, Any],
        class_grid: Sequence[float],
        max_passes: int,
    ) -> tuple[CGLCParameters, float]:
        """Stage C: discrete tuning of delta_o and b_o for each class."""
        best = deepcopy(initial_parameters)
        best_objective = self._objective(best, data)

        attributes = []
        if self.options["use_delta_class"]:
            attributes.append("delta_class")
        if self.options["use_b_class"]:
            attributes.append("b_class")

        for _ in range(max_passes):
            pass_improved = False
            for occupation in self.occupations:
                for attribute in attributes:
                    base = getattr(best, attribute).get(occupation, 0.0)
                    coordinate_best = best
                    coordinate_objective = best_objective

                    for offset in class_grid:
                        candidate = deepcopy(best)
                        getattr(candidate, attribute)[occupation] = base + float(offset)
                        objective = self._objective(candidate, data)
                        if objective < coordinate_objective:
                            coordinate_best = candidate
                            coordinate_objective = objective

                    if coordinate_objective < best_objective:
                        best = coordinate_best
                        best_objective = coordinate_objective
                        pass_improved = True

            if not pass_improved:
                break

        return best, best_objective

    def fit(
        self,
        logits: Any,
        signed_rsbe: Any,
        subject_groups: Any,
        target_occupations: Any,
        candidate_mask: Any | None = None,
        labels: Any | None = None,
        objective: str = "rd",
        rd_mode: str = "one_vs_rest",
        use_gate: bool = True,
        use_residual: bool = True,
        use_delta_class: bool = True,
        use_b_class: bool = True,
        max_accuracy_drop: float | None = None,
        stage_a_iterations: int = 500,
        stage_a_subset_fraction: float = 0.4,
        stage_b_passes: int = 5,
        stage_c_passes: int = 5,
        max_shift: float = 1.0,
        search_ranges: Mapping[str, tuple[float, float]] | None = None,
        coordinate_offsets: Mapping[str, Sequence[float]] | None = None,
        class_grid: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Fit CGLC with random search, coordinate descent, and class tuning.

        target_occupations
            The target occupation o of each example's one-vs-rest task (the
            ``target`` column). Used for the per-class RSBE statistics and
            for the objective.
        labels
            1 where the biography's true occupation is the target, else 0.
            Required for ``objective="tpr_gap"`` and ``max_accuracy_drop``.
        objective
            "rd" (demographic parity, Equation 11) or "tpr_gap" (equality of
            opportunity, Section 5.4).
        rd_mode
            "one_vs_rest" (default, the paper's task) or "multiclass" (the
            earlier behaviour: selection-rate gap of each occupation over all
            examples).
        use_gate, use_residual, use_delta_class, use_b_class
            Component switches for the ablations of Table 5. A switched-off
            component is neither searched nor applied.
        max_accuracy_drop
            Optional guard that is not in the paper: reject parameter sets
            whose one-vs-rest accuracy falls more than this far below the
            uncalibrated accuracy. Equation 11 alone can be lowered by simply
            never selecting the target.
        """
        logits, signed_rsbe, subject_groups, candidate_mask = self._validate_inputs(
            logits, signed_rsbe, subject_groups, candidate_mask
        )
        target_occupations = np.asarray(target_occupations).astype(str)
        if len(target_occupations) != len(subject_groups):
            raise ValueError("target_occupations must match the number of examples")
        if objective not in VALID_OBJECTIVES:
            raise ValueError(f"objective must be one of {sorted(VALID_OBJECTIVES)}")
        if rd_mode not in {"one_vs_rest", "multiclass"}:
            raise ValueError("rd_mode must be 'one_vs_rest' or 'multiclass'")
        if labels is not None:
            labels = np.asarray(labels).astype(int)
            if len(labels) != len(subject_groups):
                raise ValueError("labels must match the number of examples")
        if labels is None and (objective == "tpr_gap" or max_accuracy_drop is not None):
            raise ValueError("labels are required for this objective or guard")

        self.options = {
            "use_gate": bool(use_gate),
            "use_residual": bool(use_residual),
            "use_delta_class": bool(use_delta_class),
            "use_b_class": bool(use_b_class),
        }
        self.objective = objective
        self.rd_mode = rd_mode

        self.fit_rsbe_statistics(signed_rsbe, target_occupations)

        data: dict[str, Any] = {
            "logits": logits,
            "signed_rsbe": signed_rsbe,
            "subject_groups": subject_groups,
            "candidate_mask": candidate_mask,
            "example_targets": target_occupations,
            "labels": labels,
            "accuracy_floor": None,
        }

        zero = self._zero_parameters(max_shift=max_shift)
        initial_objective = self._objective(zero, data)

        if max_accuracy_drop is not None:
            base_predictions = self.predict(
                logits=logits,
                signed_rsbe=signed_rsbe,
                subject_groups=subject_groups,
                candidate_mask=candidate_mask,
                parameters=zero,
            )["predictions"]
            base_selected = one_vs_rest_selected(base_predictions, target_occupations)
            base_accuracy = float(np.mean(base_selected == (labels == 1)))
            data["accuracy_floor"] = base_accuracy - float(max_accuracy_drop)

        if search_ranges is None:
            search_ranges = {
                "delta": (-0.5, 0.5),
                "gamma": (0.0, 5.0),
                "b_group": (-0.5, 0.5),
                "beta_positive": (-1.0, 1.0),
                "beta_negative": (-1.0, 1.0),
            }
        if coordinate_offsets is None:
            coordinate_offsets = {
                "default": (-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20),
                "gamma": (-1.0, -0.50, -0.25, 0.0, 0.25, 0.50, 1.0),
                "beta": (-0.40, -0.20, -0.10, 0.0, 0.10, 0.20, 0.40),
            }
        if class_grid is None:
            class_grid = (-0.30, -0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20, 0.30)

        stage_a_parameters, stage_a_value = self._stage_a_random_search(
            data,
            stage_a_iterations,
            stage_a_subset_fraction,
            max_shift,
            search_ranges,
        )
        # Stage A is scored on a subset. Only keep its result as the warm
        # start if it is no worse than doing nothing on the full set.
        if self._objective(stage_a_parameters, data) > initial_objective:
            stage_a_parameters = zero
        stage_b_parameters, stage_b_value = self._stage_b_coordinate_descent(
            stage_a_parameters,
            data,
            coordinate_offsets,
            stage_b_passes,
        )
        stage_c_parameters, stage_c_value = self._stage_c_class_tuning(
            stage_b_parameters,
            data,
            class_grid,
            stage_c_passes,
        )
        self.parameters = stage_c_parameters

        return {
            "parameters": deepcopy(self.parameters),
            "objective": objective,
            "initial_value": float(initial_objective),
            "stage_a_value": float(stage_a_value),
            "stage_b_value": float(stage_b_value),
            "stage_c_value": float(stage_c_value),
            # Earlier key names, kept for existing code.
            "stage_a_rd": float(stage_a_value),
            "stage_b_rd": float(stage_b_value),
            "stage_c_rd": float(stage_c_value),
        }

    def save(self, file_path: str | Path) -> None:
        """Serialize frozen parameters and RSBE statistics as JSON."""
        if self.rsbe_means is None or self.rsbe_stds is None:
            raise RuntimeError("cannot save an unfitted calibrator")

        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "occupations": self.occupations,
            "groups": self.groups,
            "epsilon": self.epsilon,
            "random_state": self.random_state,
            "parameters": asdict(self.parameters),
            "options": dict(self.options),
            "objective": self.objective,
            "rd_mode": self.rd_mode,
            "rsbe_means": self.rsbe_means.tolist(),
            "rsbe_stds": self.rsbe_stds.tolist(),
        }
        file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, file_path: str | Path) -> "ConfidenceGatedLogitCalibrator":
        """Load a frozen calibrator from JSON."""
        payload = json.loads(Path(file_path).read_text(encoding="utf-8"))
        calibrator = cls(
            occupations=payload["occupations"],
            groups=payload["groups"],
            epsilon=payload["epsilon"],
            random_state=payload.get("random_state", 42),
        )
        calibrator.parameters = CGLCParameters(**payload["parameters"])
        calibrator.options = {**DEFAULT_OPTIONS, **payload.get("options", {})}
        calibrator.objective = payload.get("objective", "rd")
        calibrator.rd_mode = payload.get("rd_mode", "one_vs_rest")
        calibrator.rsbe_means = np.asarray(payload["rsbe_means"], dtype=np.float64)
        calibrator.rsbe_stds = np.asarray(payload["rsbe_stds"], dtype=np.float64)
        return calibrator


def rsbe_results_to_array(
    bio_ids: Iterable[Any],
    rsbe_results: Mapping[Any, Mapping[str, Any]],
    groups: Sequence[str] = ("Male", "Female"),
) -> np.ndarray:
    """Convert the existing RSBE result dictionary into a numeric matrix."""
    rows = []
    for bio_id in bio_ids:
        if bio_id not in rsbe_results:
            raise KeyError(f"RSBE results do not contain bio_id {bio_id!r}")
        signed_scores = rsbe_results[bio_id]["signed_scores"]
        rows.append([float(signed_scores[group]) for group in groups])
    return np.asarray(rows, dtype=np.float64)


def choice_logits_to_matrix(
    examples: Sequence[Mapping[str, Any]],
    occupations: Sequence[str],
    logits_key: str = "candidate_logits",
) -> tuple[np.ndarray, np.ndarray]:
    """Convert per-example three-choice logit dictionaries into matrix form."""
    occupations = [str(value) for value in occupations]
    occupation_to_index = {
        occupation: index for index, occupation in enumerate(occupations)
    }
    logits = np.zeros((len(examples), len(occupations)), dtype=np.float64)
    candidate_mask = np.zeros_like(logits, dtype=bool)

    for row_index, example in enumerate(examples):
        candidate_logits = example[logits_key]
        if not candidate_logits:
            raise ValueError(f"example {row_index} contains no candidate logits")
        for occupation, value in candidate_logits.items():
            if occupation not in occupation_to_index:
                raise KeyError(f"unknown candidate occupation: {occupation!r}")
            column_index = occupation_to_index[occupation]
            logits[row_index, column_index] = float(value)
            candidate_mask[row_index, column_index] = True

    return logits, candidate_mask


def train_cglc(
    occupations: Sequence[str],
    training_logits: Any,
    training_candidate_mask: Any,
    training_signed_rsbe: Any,
    training_groups: Any,
    training_targets: Any,
    output_path: str | Path,
    random_state: int = 42,
    **fit_options: Any,
) -> tuple[ConfidenceGatedLogitCalibrator, dict[str, Any]]:
    """Convenience entry point for training and saving CGLC."""
    calibrator = ConfidenceGatedLogitCalibrator(
        occupations=occupations,
        groups=("Male", "Female"),
        random_state=random_state,
    )
    fit_result = calibrator.fit(
        logits=training_logits,
        signed_rsbe=training_signed_rsbe,
        subject_groups=training_groups,
        target_occupations=training_targets,
        candidate_mask=training_candidate_mask,
        **fit_options,
    )
    calibrator.save(output_path)
    return calibrator, fit_result


def apply_cglc(
    calibration_path: str | Path,
    logits: Any,
    candidate_mask: Any,
    signed_rsbe: Any,
    subject_groups: Any,
    **calibration_options: Any,
) -> dict[str, np.ndarray]:
    """Convenience entry point for loading and applying frozen CGLC."""
    calibrator = ConfidenceGatedLogitCalibrator.load(calibration_path)
    return calibrator.predict(
        logits=logits,
        signed_rsbe=signed_rsbe,
        subject_groups=subject_groups,
        candidate_mask=candidate_mask,
        **calibration_options,
    )
