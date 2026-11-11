"""Generator-side utility and fairness metrics (Appendix C.3 and C.4).

The task is one-vs-rest per occupation: every example belongs to the task of
a target occupation o and has a label y (1 if the biography's occupation is
o). The model *selects* o when its (calibrated) argmax over the three
candidates is o.

Utility   accuracy over all examples; precision, recall and F1 of the
          "selected" decision per occupation, then macro-averaged.
Fairness  risk difference RD(o) = |P(selected | g=a0) - P(selected | g=a1)|
          and TPR gap (the same gap among the positives), macro-averaged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd


def _rate(mask: np.ndarray, selected: np.ndarray) -> float:
    return float(selected[mask].mean()) if mask.any() else float("nan")


def per_occupation_report(
    selected,
    labels,
    subject_groups,
    example_targets,
    groups: Sequence[str] = ("Male", "Female"),
) -> pd.DataFrame:
    """One row of utility and fairness metrics per target occupation."""
    selected = np.asarray(selected, dtype=bool)
    labels = np.asarray(labels).astype(int) == 1
    subject_groups = np.asarray(subject_groups).astype(str)
    example_targets = np.asarray(example_targets).astype(str)
    group_a, group_b = [str(group) for group in groups]

    rows = []
    for occupation in dict.fromkeys(example_targets.tolist()):
        in_task = example_targets == occupation
        sel = selected[in_task]
        lab = labels[in_task]
        grp = subject_groups[in_task]

        tp = int(np.sum(sel & lab))
        fp = int(np.sum(sel & ~lab))
        fn = int(np.sum(~sel & lab))
        tn = int(np.sum(~sel & ~lab))

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        is_a = grp == group_a
        is_b = grp == group_b

        rate_a = _rate(is_a, sel)
        rate_b = _rate(is_b, sel)
        tpr_a = _rate(is_a & lab, sel)
        tpr_b = _rate(is_b & lab, sel)

        rows.append(
            {
                "occupation": occupation,
                "n": int(in_task.sum()),
                "accuracy": (tp + tn) / max(1, int(in_task.sum())),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                f"selection_rate_{group_a}": rate_a,
                f"selection_rate_{group_b}": rate_b,
                "rd": abs(rate_a - rate_b),
                f"tpr_{group_a}": tpr_a,
                f"tpr_{group_b}": tpr_b,
                "tpr_gap": abs(tpr_a - tpr_b),
            }
        )

    return pd.DataFrame(rows)


def summarize(
    selected,
    labels,
    subject_groups,
    example_targets,
    groups: Sequence[str] = ("Male", "Female"),
) -> dict[str, float]:
    """Acc / Prec / Rec / F1 / RD / TPR-Gap as reported in Tables 2 to 5."""
    report = per_occupation_report(
        selected, labels, subject_groups, example_targets, groups
    )
    selected = np.asarray(selected, dtype=bool)
    labels = np.asarray(labels).astype(int) == 1

    return {
        "acc": float(np.mean(selected == labels)),
        "prec": float(report["precision"].mean()),
        "rec": float(report["recall"].mean()),
        "f1": float(report["f1"].mean()),
        "rd": float(np.nanmean(report["rd"])),
        "tpr_gap": float(np.nanmean(report["tpr_gap"])),
    }
