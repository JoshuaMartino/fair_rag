"""Calibration stage: fit CGLC and build Tables 2 to 5 of the paper.

Run from inside ``fairrag/`` after ``run_generation_main.py``
(edit the configuration block in ``main`` first):

    python run_calibration_main.py

Reads   results/generation/<model>/{train,test}/<system>.jsonl
Writes  results/calibration/<model>/
            table2_generator.csv         baselines vs. FGR + CGLC
            table3_rerankers_cglc.csv    each reranker paired with CGLC
            table4_equal_opportunity.csv CGLC trained on the TPR gap
            table5_ablation.csv          CGLC components
            per_occupation_<row>.csv     per-occupation breakdown of each row
            cglc_<name>.json             the frozen calibrators

Everything here runs on CPU in seconds to minutes, so search ranges and
other choices the paper leaves open can be varied without touching the LLM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from calibration import ConfidenceGatedLogitCalibrator
from generation_metrics import per_occupation_report, summarize
from run_generation_main import load_records, model_tag


SYSTEM_LABELS = {
    "no_retriever": "No Retriever",
    "vanilla": "Vanilla",
    "det_const_sort": "Det const sort",
    "fair_filter": "FairFilter",
    "fairft": "FairFT",
    "in_context": "In Context",
    "fgr": "FGR",
}

# Table 5 rows: (label, signal column, component switches).
ABLATIONS = [
    ("Skew", "skew_signal", {}),
    ("No eta", "signed_ndkl", {"use_gate": False}),
    ("No delta_o", "signed_ndkl", {"use_delta_class": False}),
    ("No b_o", "signed_ndkl", {"use_b_class": False}),
    ("NDKL + eta + delta_o + b_o", "signed_ndkl", {}),
]


def records_to_arrays(frame, occupations, signal="signed_ndkl"):
    """Turn generation records into the matrices the calibrator expects."""
    index = {occupation: i for i, occupation in enumerate(occupations)}

    logits = np.zeros((len(frame), len(occupations)), dtype=np.float64)
    mask = np.zeros_like(logits, dtype=bool)

    for row_number, (candidates, scores) in enumerate(
        zip(frame["candidates"], frame["scores"])
    ):
        for candidate, score in zip(candidates, scores):
            if candidate not in index:
                raise KeyError(f"Unknown candidate occupation {candidate!r}")
            logits[row_number, index[candidate]] = float(score)
            mask[row_number, index[candidate]] = True

    return {
        "logits": logits,
        "candidate_mask": mask,
        "signed_rsbe": np.asarray(frame[signal].tolist(), dtype=np.float64),
        "subject_groups": frame["subject_group"].astype(str).to_numpy(),
        "target_occupations": frame["target"].astype(str).to_numpy(),
        "labels": frame["y"].astype(int).to_numpy(),
    }


def uncalibrated_selected(frame):
    """True where the raw argmax over the three candidates is the target."""
    selected = []
    for candidates, scores, target in zip(
        frame["candidates"], frame["scores"], frame["target"]
    ):
        best = candidates[int(np.argmax(scores))]
        selected.append(best == target)
    return np.asarray(selected, dtype=bool)


def collect_occupations(frames):
    occupations = set()
    for frame in frames:
        for candidates in frame["candidates"]:
            occupations.update(candidates)
        occupations.update(frame["target"].astype(str))
    return sorted(occupations)


def fit_and_apply(
    train_frame,
    test_frame,
    occupations,
    groups,
    objective="rd",
    signal="signed_ndkl",
    options=None,
    fit_options=None,
    random_state=42,
):
    """Fit CGLC on the train records and apply it, frozen, to the test records.

    Returns (selected_test, calibrator, fit_result).
    """
    train = records_to_arrays(train_frame, occupations, signal)
    test = records_to_arrays(test_frame, occupations, signal)

    calibrator = ConfidenceGatedLogitCalibrator(
        occupations=occupations,
        groups=groups,
        random_state=random_state,
    )
    fit_result = calibrator.fit(
        logits=train["logits"],
        signed_rsbe=train["signed_rsbe"],
        subject_groups=train["subject_groups"],
        target_occupations=train["target_occupations"],
        candidate_mask=train["candidate_mask"],
        labels=train["labels"],
        objective=objective,
        **(options or {}),
        **(fit_options or {}),
    )

    predictions = calibrator.predict(
        logits=test["logits"],
        signed_rsbe=test["signed_rsbe"],
        subject_groups=test["subject_groups"],
        candidate_mask=test["candidate_mask"],
    )["predictions"]
    selected = predictions == test["target_occupations"]

    return selected, calibrator, fit_result


def metric_row(label, frame, selected, groups, extra=None):
    row = {"model": label, "n": len(frame)}
    row.update(
        summarize(
            selected,
            frame["y"],
            frame["subject_group"],
            frame["target"],
            groups,
        )
    )
    row.update(extra or {})
    return row


def save_per_occupation(output_dir, name, frame, selected, groups):
    safe = "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")
    per_occupation_report(
        selected, frame["y"], frame["subject_group"], frame["target"], groups
    ).to_csv(output_dir / f"per_occupation_{safe}.csv", index=False)


def load_split(generation_dir, split, systems):
    frames = {}
    for system in systems:
        path = generation_dir / split / f"{system}.jsonl"
        if path.exists():
            frame = load_records(path)
            if not frame.empty:
                frames[system] = frame
    return frames


def main():
    # ------------------------------------------------------------------
    # Experiment configuration: edit these values before running.
    # ------------------------------------------------------------------
    model_name = "google/gemma-2b"

    generation_root = Path("results/generation")
    output_root = Path("results/calibration")

    # Rows of Table 2 (uncalibrated) and Table 3 (each paired with CGLC).
    baseline_systems = [
        "no_retriever", "vanilla", "det_const_sort",
        "fair_filter", "fairft", "in_context",
    ]
    cglc_systems = ["det_const_sort", "fair_filter", "fgr"]

    # Search settings. The paper does not report ranges, so these are the
    # calibrator's defaults; pass search_ranges / coordinate_offsets /
    # class_grid / max_shift here to change them.
    fit_options = {
        "stage_a_iterations": 500,
        "stage_a_subset_fraction": 0.4,
        "max_shift": 1.0,
        # Not in the paper. Set e.g. 0.03 to stop the search trading more
        # than three points of accuracy for a lower gap.
        "max_accuracy_drop": None,
    }
    random_state = 42

    # ------------------------------------------------------------------
    generation_dir = generation_root / model_tag(model_name)
    output_dir = output_root / model_tag(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_systems = sorted(set(baseline_systems) | set(cglc_systems))
    train = load_split(generation_dir, "train", all_systems)
    test = load_split(generation_dir, "test", all_systems)

    if not test:
        raise FileNotFoundError(
            f"No test records found under {generation_dir / 'test'}. "
            "Run run_generation_main.py first."
        )

    occupations = collect_occupations(list(train.values()) + list(test.values()))
    groups = list(next(iter(test.values()))["groups"].iloc[0])
    print(f"{model_name}: {len(occupations)} occupations, groups {groups}")

    def calibrated(system, objective="rd", signal="signed_ndkl", options=None, name=None):
        if system not in train or system not in test:
            print(f"  (no train/test records for {system}; row skipped)")
            return None
        selected, calibrator, fit_result = fit_and_apply(
            train[system],
            test[system],
            occupations,
            groups,
            objective=objective,
            signal=signal,
            options=options,
            fit_options=fit_options,
            random_state=random_state,
        )
        if name:
            calibrator.save(output_dir / f"cglc_{name}.json")
        print(
            f"  {name or system}: train {objective} "
            f"{fit_result['initial_value']:.4f} -> {fit_result['stage_c_value']:.4f}"
        )
        return selected

    # ---------------- Table 2: baselines vs. ours -----------------------
    print("Table 2")
    table2 = []
    for system in baseline_systems:
        if system not in test:
            continue
        selected = uncalibrated_selected(test[system])
        table2.append(metric_row(SYSTEM_LABELS[system], test[system], selected, groups))
        save_per_occupation(output_dir, system, test[system], selected, groups)

    ours = calibrated("fgr", name="fgr_rd")
    if ours is not None:
        table2.append(metric_row("Our (FGR + CGLC)", test["fgr"], ours, groups))
        save_per_occupation(output_dir, "fgr_cglc", test["fgr"], ours, groups)
    table2 = pd.DataFrame(table2)
    table2.to_csv(output_dir / "table2_generator.csv", index=False)

    # ---------------- Table 3: rerankers + CGLC -------------------------
    print("Table 3")
    table3 = []
    for system in cglc_systems:
        selected = ours if system == "fgr" else calibrated(system, name=f"{system}_rd")
        if selected is None:
            continue
        table3.append(
            metric_row(f"{SYSTEM_LABELS[system]} + CGLC", test[system], selected, groups)
        )
    table3 = pd.DataFrame(table3)
    table3.to_csv(output_dir / "table3_rerankers_cglc.csv", index=False)

    # ---------------- Table 4: equality of opportunity ------------------
    print("Table 4")
    table4 = []
    for system in baseline_systems:
        if system not in test or system == "no_retriever":
            continue
        selected = uncalibrated_selected(test[system])
        table4.append(metric_row(SYSTEM_LABELS[system], test[system], selected, groups))
    ours_eo = calibrated("fgr", objective="tpr_gap", name="fgr_tpr_gap")
    if ours_eo is not None:
        table4.append(metric_row("Our (FGR + CGLC, TPR gap)", test["fgr"], ours_eo, groups))
    table4 = pd.DataFrame(table4)
    table4.to_csv(output_dir / "table4_equal_opportunity.csv", index=False)

    # ---------------- Table 5: ablation ---------------------------------
    print("Table 5")
    table5 = []
    if "fgr" in test:
        fgr_only = uncalibrated_selected(test["fgr"])
        table5.append(metric_row("FGR Only", test["fgr"], fgr_only, groups))
        for label, signal, options in ABLATIONS:
            is_full = signal == "signed_ndkl" and not options
            selected = (
                ours
                if is_full
                else calibrated("fgr", signal=signal, options=options, name=None)
            )
            if selected is None:
                continue
            table5.append(metric_row(label, test["fgr"], selected, groups))
    table5 = pd.DataFrame(table5)
    table5.to_csv(output_dir / "table5_ablation.csv", index=False)

    # ---------------- Print ---------------------------------------------
    def show(title, table, columns):
        print()
        print(title)
        if table.empty:
            print("  (no rows)")
            return
        print(table[["model"] + columns].round(3).to_string(index=False))

    utility = ["acc", "prec", "rec", "f1"]
    show("Table 2: generator side", table2, utility + ["rd"])
    show("Table 3: rerankers paired with CGLC", table3, utility + ["rd"])
    show("Table 4: equality of opportunity", table4, utility + ["tpr_gap"])
    show("Table 5: CGLC ablation", table5, utility + ["rd"])
    print()
    print(f"Saved tables to: {output_dir.resolve()}")

    return {"table2": table2, "table3": table3, "table4": table4, "table5": table5}


if __name__ == "__main__":
    main()
