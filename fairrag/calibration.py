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
        use_gate: bool = True,
        use_residual: bool = True,
        use_delta_class: bool = True,
        use_b_class: bool = True,
    ) -> dict[str, np.ndarray]:
        """Apply CGLC to raw pre-softmax candidate logits."""
        logits, signed_rsbe, subject_groups, candidate_mask = self._validate_inputs(
            logits, signed_rsbe, subject_groups, candidate_mask
        )
        if parameters is None:
            parameters = self.parameters

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
        logits: np.ndarray,
        signed_rsbe: np.ndarray,
        subject_groups: np.ndarray,
        candidate_mask: np.ndarray,
    ) -> float:
        predictions = self.predict(
            logits=logits,
            signed_rsbe=signed_rsbe,
            subject_groups=subject_groups,
            candidate_mask=candidate_mask,
            parameters=parameters,
        )["predictions"]
        value, _ = self.mean_risk_difference(predictions, subject_groups)
        return value

    def _stage_a_random_search(
        self,
        logits: np.ndarray,
        signed_rsbe: np.ndarray,
        subject_groups: np.ndarray,
        candidate_mask: np.ndarray,
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

        subset_size = max(1, int(round(len(subject_groups) * subset_fraction)))
        indices = self.rng.choice(len(subject_groups), subset_size, replace=False)
        sub_logits = logits[indices]
        sub_rsbe = signed_rsbe[indices]
        sub_groups = subject_groups[indices]
        sub_mask = candidate_mask[indices]

        best = self._zero_parameters(max_shift=max_shift)
        best_objective = self._objective(
            best, sub_logits, sub_rsbe, sub_groups, sub_mask
        )

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

            objective = self._objective(
                candidate, sub_logits, sub_rsbe, sub_groups, sub_mask
            )
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
        logits: np.ndarray,
        signed_rsbe: np.ndarray,
        subject_groups: np.ndarray,
        candidate_mask: np.ndarray,
        coordinate_offsets: Mapping[str, Sequence[float]],
        max_passes: int,
    ) -> tuple[CGLCParameters, float]:
        """Stage B: coordinate descent over the six global scalars."""
        best = deepcopy(initial_parameters)
        best_objective = self._objective(
            best, logits, signed_rsbe, subject_groups, candidate_mask
        )

        coordinates = ["delta", "gamma"]
        coordinates.extend(f"b_group::{group}" for group in self.groups)
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
                    objective = self._objective(
                        candidate, logits, signed_rsbe, subject_groups, candidate_mask
                    )
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
        logits: np.ndarray,
        signed_rsbe: np.ndarray,
        subject_groups: np.ndarray,
        candidate_mask: np.ndarray,
        class_grid: Sequence[float],
        max_passes: int,
    ) -> tuple[CGLCParameters, float]:
        """Stage C: discrete tuning of delta_o and b_o for each class."""
        best = deepcopy(initial_parameters)
        best_objective = self._objective(
            best, logits, signed_rsbe, subject_groups, candidate_mask
        )

        for _ in range(max_passes):
            pass_improved = False
            for occupation in self.occupations:
                for attribute in ("delta_class", "b_class"):
                    base = getattr(best, attribute).get(occupation, 0.0)
                    coordinate_best = best
                    coordinate_objective = best_objective

                    for offset in class_grid:
                        candidate = deepcopy(best)
                        getattr(candidate, attribute)[occupation] = base + float(offset)
                        objective = self._objective(
                            candidate,
                            logits,
                            signed_rsbe,
                            subject_groups,
                            candidate_mask,
                        )
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
        stage_a_iterations: int = 500,
        stage_a_subset_fraction: float = 0.4,
        stage_b_passes: int = 5,
        stage_c_passes: int = 5,
        max_shift: float = 1.0,
        search_ranges: Mapping[str, tuple[float, float]] | None = None,
        coordinate_offsets: Mapping[str, Sequence[float]] | None = None,
        class_grid: Sequence[float] | None = None,
    ) -> dict[str, Any]:
        """Fit CGLC with random search, coordinate descent, and class tuning."""
        logits, signed_rsbe, subject_groups, candidate_mask = self._validate_inputs(
            logits, signed_rsbe, subject_groups, candidate_mask
        )
        target_occupations = np.asarray(target_occupations).astype(str)
        if len(target_occupations) != len(subject_groups):
            raise ValueError("target_occupations must match the number of examples")

        self.fit_rsbe_statistics(signed_rsbe, target_occupations)

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

        stage_a_parameters, stage_a_rd = self._stage_a_random_search(
            logits,
            signed_rsbe,
            subject_groups,
            candidate_mask,
            stage_a_iterations,
            stage_a_subset_fraction,
            max_shift,
            search_ranges,
        )
        stage_b_parameters, stage_b_rd = self._stage_b_coordinate_descent(
            stage_a_parameters,
            logits,
            signed_rsbe,
            subject_groups,
            candidate_mask,
            coordinate_offsets,
            stage_b_passes,
        )
        stage_c_parameters, stage_c_rd = self._stage_c_class_tuning(
            stage_b_parameters,
            logits,
            signed_rsbe,
            subject_groups,
            candidate_mask,
            class_grid,
            stage_c_passes,
        )
        self.parameters = stage_c_parameters

        return {
            "parameters": deepcopy(self.parameters),
            "stage_a_rd": float(stage_a_rd),
            "stage_b_rd": float(stage_b_rd),
            "stage_c_rd": float(stage_c_rd),
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
