"""Analysis methods for comparing Vanilla and FGR retrieval results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def validate_columns(
    dataframe,
    required_columns,
    file_label,
):
    """Check that a dataframe contains all required columns."""
    missing_columns = (
        set(required_columns)
        - set(dataframe.columns)
    )

    if missing_columns:
        raise KeyError(
            f"{file_label} is missing required columns: "
            f"{sorted(missing_columns)}. "
            f"Available columns: "
            f"{dataframe.columns.tolist()}"
        )


def extract_query_occupation(bio_ids):
    """
    Extract the query occupation from an experiment-specific bio ID.

    Example:
        accountant::12345::0 -> accountant
    """
    return (
        bio_ids.astype(str)
        .str.split("::", n=1)
        .str[0]
    )


def prepare_slates(dataframe, system):
    """Add analysis columns to a slate dataframe."""
    dataframe = dataframe.copy()

    dataframe["query_occupation"] = (
        extract_query_occupation(
            dataframe["bio_id"]
        )
    )

    dataframe["system"] = system

    dataframe["gender"] = (
        dataframe["gender"]
        .astype("string")
        .str.strip()
    )

    dataframe["relevant"] = (
        dataframe["occupation"].astype(str)
        ==
        dataframe["query_occupation"].astype(str)
    )

    return dataframe


def save_figure(output_path):
    """Save and close the current matplotlib figure."""
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()
def analyse_retrieval_results(
    retrieval_summary_path,
    vanilla_fairness_path,
    fgr_fairness_path,
    vanilla_slates_path,
    fgr_slates_path,
    output_dir="results/analysis",
    groups=("Male", "Female"),):
    """
    Generate tables and graphs comparing Vanilla and FGR.

    Parameters
    ----------
    retrieval_summary_path : str or Path
        Path to retrieval_summary.csv.

    vanilla_fairness_path : str or Path
        Path to sparse_vanilla_fairness_per_query.csv.

    fgr_fairness_path : str or Path
        Path to sparse_fgr_fairness_per_query.csv.

    vanilla_slates_path : str or Path
        Path to sparse_vanilla_slates.csv.

    fgr_slates_path : str or Path
        Path to sparse_fgr_slates.csv.

    output_dir : str or Path
        Directory for generated tables and figures.

    groups : tuple
        Protected-group labels as stored in document metadata.

    Returns
    -------
    dict
        Dictionary containing the generated DataFrames.
    """
    if len(groups) != 2:
        raise ValueError(
            "This analysis currently requires exactly two groups."
        )

    output_dir = Path(output_dir)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    tables_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Load files
    # ---------------------------------------------------------
    summary = pd.read_csv(retrieval_summary_path)

    vanilla_fairness = pd.read_csv(
        vanilla_fairness_path
    )

    fgr_fairness = pd.read_csv(
        fgr_fairness_path
    )

    vanilla_slates = pd.read_csv(
        vanilla_slates_path
    )

    fgr_slates = pd.read_csv(
        fgr_slates_path
    )

    # ---------------------------------------------------------
    # Validate required columns
    # ---------------------------------------------------------
    required_summary_columns = {
        "reranker",
        "hit_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "mean_ndkl",
    }

    required_fairness_columns = {
        "bio_id",
        "ndkl",
        "min_skew",
        "max_skew",
    }

    required_slate_columns = {
        "bio_id",
        "rank",
        "gender",
        "occupation",
        "score",
    }

    validate_columns(
        summary,
        required_summary_columns,
        "retrieval summary",
    )

    validate_columns(
        vanilla_fairness,
        required_fairness_columns,
        "Vanilla fairness",
    )

    validate_columns(
        fgr_fairness,
        required_fairness_columns,
        "FGR fairness",
    )

    validate_columns(
        vanilla_slates,
        required_slate_columns,
        "Vanilla slates",
    )

    validate_columns(
        fgr_slates,
        required_slate_columns,
        "FGR slates",
    )

    # ---------------------------------------------------------
    # Aggregate comparison table
    # ---------------------------------------------------------
    vanilla_summary = summary[
        summary["reranker"].str.lower() == "vanilla"
    ]

    fgr_summary = summary[
        summary["reranker"].str.lower() == "fgr"
    ]

    if len(vanilla_summary) != 1:
        raise ValueError(
            "Expected exactly one Vanilla row in retrieval_summary.csv."
        )

    if len(fgr_summary) != 1:
        raise ValueError(
            "Expected exactly one FGR row in retrieval_summary.csv."
        )

    vanilla_row = vanilla_summary.iloc[0]
    fgr_row = fgr_summary.iloc[0]

    metrics = [
        "hit_at_k",
        "mrr_at_k",
        "ndcg_at_k",
        "mean_ndkl",
        "mean_min_skew",
        "mean_max_skew",
    ]

    for group in groups:
        metrics.append(
            f"mean_signed_ndkl_group_{group}"
        )

        metrics.append(
            f"exposure_group_{group}"
        )

    metrics = [
        metric
        for metric in metrics
        if metric in summary.columns
    ]

    aggregate_rows = []

    for metric in metrics:
        vanilla_value = float(vanilla_row[metric])
        fgr_value = float(fgr_row[metric])

        absolute_change = (
            fgr_value - vanilla_value
        )

        if np.isclose(vanilla_value, 0):
            relative_change = np.nan
        else:
            relative_change = (
                absolute_change
                / abs(vanilla_value)
            )

        aggregate_rows.append({
            "metric": metric,
            "vanilla": vanilla_value,
            "fgr": fgr_value,
            "absolute_change": absolute_change,
            "relative_change": relative_change,
            "relative_change_percent": (
                relative_change * 100
                if pd.notna(relative_change)
                else np.nan
            ),
        })

    aggregate_comparison = pd.DataFrame(
        aggregate_rows
    )

    aggregate_comparison.to_csv(
        tables_dir / "aggregate_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Query occupation extraction
    # ---------------------------------------------------------
    vanilla_fairness = vanilla_fairness.copy()
    fgr_fairness = fgr_fairness.copy()

    vanilla_fairness["query_occupation"] = (
        extract_query_occupation(
            vanilla_fairness["bio_id"]
        )
    )

    fgr_fairness["query_occupation"] = (
        extract_query_occupation(
            fgr_fairness["bio_id"]
        )
    )

    # ---------------------------------------------------------
    # Per-query comparison
    # ---------------------------------------------------------
    fairness_per_query = vanilla_fairness.merge(
        fgr_fairness,
        on=[
            "bio_id",
            "query_occupation",
        ],
        suffixes=(
            "_vanilla",
            "_fgr",
        ),
        validate="one_to_one",)
        # ---------------------------------------------------------
    # Calculate per-query fairness changes
    # ---------------------------------------------------------
    fairness_per_query["ndkl_reduction"] = (
        fairness_per_query["ndkl_vanilla"]
        - fairness_per_query["ndkl_fgr"]
    )

    fairness_per_query[
        "ndkl_reduction_percent"
    ] = np.where(
        np.isclose(
            fairness_per_query["ndkl_vanilla"],
            0,
        ),
        np.nan,
        (
            100
            * fairness_per_query["ndkl_reduction"]
            / fairness_per_query["ndkl_vanilla"]
        ),
    )

    fairness_per_query["fairness_outcome"] = (
        np.select(
            [
                fairness_per_query[
                    "ndkl_reduction"
                ] > 0,

                fairness_per_query[
                    "ndkl_reduction"
                ] < 0,
            ],
            [
                "improved",
                "worsened",
            ],
            default="unchanged",
        )
    )

    fairness_per_query.to_csv(
        tables_dir
        / "fairness_per_query_comparison.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Fairness by occupation
    # ---------------------------------------------------------
    fairness_by_occupation = (
        fairness_per_query
        .groupby(
            "query_occupation",
            as_index=False,
        )
        .agg(
            queries=(
                "bio_id",
                "size",
            ),

            vanilla_mean_ndkl=(
                "ndkl_vanilla",
                "mean",
            ),

            fgr_mean_ndkl=(
                "ndkl_fgr",
                "mean",
            ),

            mean_ndkl_reduction=(
                "ndkl_reduction",
                "mean",
            ),

            median_ndkl_reduction=(
                "ndkl_reduction",
                "median",
            ),

            improved_queries=(
                "fairness_outcome",
                lambda values: int(
                    (
                        values == "improved"
                    ).sum()
                ),
            ),

            unchanged_queries=(
                "fairness_outcome",
                lambda values: int(
                    (
                        values == "unchanged"
                    ).sum()
                ),
            ),

            worsened_queries=(
                "fairness_outcome",
                lambda values: int(
                    (
                        values == "worsened"
                    ).sum()
                ),
            ),
        )
        .sort_values(
            "mean_ndkl_reduction",
            ascending=False,
        )
    )

    fairness_by_occupation.to_csv(
        tables_dir
        / "fairness_by_occupation.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Prepare the slate data
    # ---------------------------------------------------------
    vanilla_slates = prepare_slates(
        vanilla_slates,
        system="Vanilla",
    )

    fgr_slates = prepare_slates(
        fgr_slates,
        system="FGR",
    )

    combined_slates = pd.concat(
        [
            vanilla_slates,
            fgr_slates,
        ],
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # Overall exposure
    # ---------------------------------------------------------
    overall_exposure = (
        combined_slates
        .groupby(
            [
                "system",
                "gender",
            ],
            dropna=False,
        )
        .size()
        .rename("documents")
        .reset_index()
    )

    overall_exposure["exposure"] = (
        overall_exposure
        .groupby("system")["documents"]
        .transform(
            lambda values: (
                values / values.sum()
            )
        )
    )

    overall_exposure.to_csv(
        tables_dir / "overall_exposure.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Exposure by query occupation
    # ---------------------------------------------------------
    exposure_by_occupation = (
        combined_slates
        .groupby(
            [
                "query_occupation",
                "system",
                "gender",
            ],
            dropna=False,
        )
        .size()
        .rename("documents")
        .reset_index()
    )

    exposure_by_occupation[
        "exposure"
    ] = (
        exposure_by_occupation
        .groupby(
            [
                "query_occupation",
                "system",
            ]
        )["documents"]
        .transform(
            lambda values: (
                values / values.sum()
            )
        )
    )

    exposure_by_occupation.to_csv(
        tables_dir
        / "exposure_by_occupation.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Exposure by rank
    # ---------------------------------------------------------
    exposure_by_rank = (
        combined_slates
        .groupby(
            [
                "system",
                "rank",
                "gender",
            ],
            dropna=False,
        )
        .size()
        .rename("documents")
        .reset_index()
    )

    exposure_by_rank[
        "share_at_rank"
    ] = (
        exposure_by_rank
        .groupby(
            [
                "system",
                "rank",
            ]
        )["documents"]
        .transform(
            lambda values: (
                values / values.sum()
            )
        )
    )

    exposure_by_rank.to_csv(
        tables_dir / "exposure_by_rank.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Group counts for every query
    # ---------------------------------------------------------
    group_counts_per_query = (
        combined_slates
        .groupby(
            [
                "system",
                "bio_id",
                "query_occupation",
                "gender",
            ],
            dropna=False,
        )
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    for group in groups:
        if group not in group_counts_per_query.columns:
            group_counts_per_query[group] = 0

    group_counts_per_query[
        "slate_size"
    ] = group_counts_per_query[
        list(groups)
    ].sum(axis=1)

    group_counts_per_query[
        "absolute_group_gap"
    ] = (
        group_counts_per_query[groups[0]]
        - group_counts_per_query[groups[1]]
    ).abs()

    # For K=15, the closest possible balance is 8/7.
    group_counts_per_query[
        "approximately_balanced"
    ] = (
        group_counts_per_query[
            "absolute_group_gap"
        ] <= 1
    )

    group_counts_per_query.to_csv(
        tables_dir
        / "group_counts_per_query.csv",
        index=False,
    )

    balance_summary = (
        group_counts_per_query
        .groupby(
            "system",
            as_index=False,
        )
        .agg(
            queries=(
                "bio_id",
                "size",
            ),

            approximately_balanced_queries=(
                "approximately_balanced",
                "sum",
            ),

            mean_absolute_group_gap=(
                "absolute_group_gap",
                "mean",
            ),
        )
    )

    balance_summary[
        "approximately_balanced_rate"
    ] = (
        balance_summary[
            "approximately_balanced_queries"
        ]
        / balance_summary["queries"]
    )

    balance_summary.to_csv(
        tables_dir / "balance_summary.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Metadata audit
    # ---------------------------------------------------------
    observed_groups = set(
        combined_slates["gender"]
        .dropna()
        .astype(str)
        .unique()
    )

    expected_groups = set(
        map(str, groups)
    )

    metadata_audit = pd.DataFrame([
        {
            "expected_groups": ", ".join(
                map(str, groups)
            ),

            "observed_groups": ", ".join(
                sorted(observed_groups)
            ),

            "unexpected_groups": ", ".join(
                sorted(
                    observed_groups
                    - expected_groups
                )
            ),

            "missing_gender_rows": int(
                combined_slates[
                    "gender"
                ].isna().sum()
            ),
        }
    ])

    metadata_audit.to_csv(
        tables_dir / "metadata_audit.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Figure 1: utility comparison
    # ---------------------------------------------------------
    utility_metrics = [
        "hit_at_k",
        "mrr_at_k",
        "ndcg_at_k",
    ]

    utility_data = aggregate_comparison[
        aggregate_comparison[
            "metric"
        ].isin(utility_metrics)
    ].set_index("metric")

    utility_data[
        [
            "vanilla",
            "fgr",
        ]
    ].plot(kind="bar")

    plt.title(
        "Retrieval utility: Vanilla vs FGR"
    )

    plt.xlabel("Metric")
    plt.ylabel("Score")
    plt.xticks(rotation=0)
    plt.legend(title="System")

    save_figure(
        figures_dir
        / "utility_comparison.png"
    )

    # ---------------------------------------------------------
    # Figure 2: overall exposure
    # ---------------------------------------------------------
    exposure_plot = overall_exposure.pivot(
        index="gender",
        columns="system",
        values="exposure",
    ).fillna(0)

    exposure_plot.plot(kind="bar")

    plt.axhline(
        0.5,
        linestyle="--",
        label="Target 0.5",
    )

    plt.title("Overall group exposure")
    plt.xlabel("Gender")
    plt.ylabel("Exposure share")
    plt.xticks(rotation=0)
    plt.legend()

    save_figure(
        figures_dir
        / "overall_exposure.png"
    )

    # ---------------------------------------------------------
    # Figure 3: NDKL by occupation
    # ---------------------------------------------------------
    occupation_plot = (
        fairness_by_occupation
        .sort_values(
            "vanilla_mean_ndkl"
        )
    )

    x_positions = np.arange(
        len(occupation_plot)
    )

    width = 0.4

    plt.figure(
        figsize=(
            max(
                9,
                len(occupation_plot) * 0.45,
            ),
            6,
        )
    )

    plt.bar(
        x_positions - width / 2,
        occupation_plot[
            "vanilla_mean_ndkl"
        ],
        width,
        label="Vanilla",
    )

    plt.bar(
        x_positions + width / 2,
        occupation_plot[
            "fgr_mean_ndkl"
        ],
        width,
        label="FGR",
    )

    plt.title(
        "Mean NDKL by query occupation"
    )

    plt.xlabel("Occupation")
    plt.ylabel("Mean NDKL")

    plt.xticks(
        x_positions,
        occupation_plot[
            "query_occupation"
        ],
        rotation=60,
        ha="right",
    )

    plt.legend()

    save_figure(
        figures_dir
        / "ndkl_by_occupation.png"
    )

    # ---------------------------------------------------------
    # Figure 4: per-query NDKL scatter
    # ---------------------------------------------------------
    plt.figure()

    plt.scatter(
        fairness_per_query[
            "ndkl_vanilla"
        ],
        fairness_per_query[
            "ndkl_fgr"
        ],
        alpha=0.65,
    )

    maximum_ndkl = max(
        fairness_per_query[
            "ndkl_vanilla"
        ].max(),

        fairness_per_query[
            "ndkl_fgr"
        ].max(),
    )

    plt.plot(
        [
            0,
            maximum_ndkl,
        ],
        [
            0,
            maximum_ndkl,
        ],
        linestyle="--",
    )

    plt.title(
        "Per-query NDKL: Vanilla vs FGR"
    )

    plt.xlabel("Vanilla NDKL")
    plt.ylabel("FGR NDKL")

    save_figure(
        figures_dir
        / "ndkl_per_query_scatter.png"
    )

    # ---------------------------------------------------------
    # Figure 5: NDKL reduction distribution
    # ---------------------------------------------------------
    plt.figure()

    plt.hist(
        fairness_per_query[
            "ndkl_reduction"
        ],
        bins=20,
    )

    plt.axvline(
        0,
        linestyle="--",
    )

    plt.title(
        "Distribution of per-query NDKL reduction"
    )

    plt.xlabel(
        "Vanilla NDKL - FGR NDKL"
    )

    plt.ylabel("Queries")

    save_figure(
        figures_dir
        / "ndkl_reduction_distribution.png"
    )

    # ---------------------------------------------------------
    # Figure 6: Female exposure by rank
    # ---------------------------------------------------------
    female_rank_data = exposure_by_rank[
        exposure_by_rank["gender"]
        == groups[1]
    ]

    plt.figure()

    for system, system_data in (
        female_rank_data.groupby("system")
    ):
        plt.plot(
            system_data["rank"],
            system_data["share_at_rank"],
            marker="o",
            label=system,
        )

    plt.axhline(
        0.5,
        linestyle="--",
        label="Target 0.5",
    )

    plt.title(
        f"{groups[1]} exposure by rank"
    )

    plt.xlabel("Rank")
    plt.ylabel("Share at rank")
    plt.legend()

    save_figure(
        figures_dir
        / "female_exposure_by_rank.png"
    )

    # ---------------------------------------------------------
    # Return everything main() expects
    # ---------------------------------------------------------
    return {
        "aggregate_comparison": (
            aggregate_comparison
        ),

        "fairness_per_query": (
            fairness_per_query
        ),

        "fairness_by_occupation": (
            fairness_by_occupation
        ),

        "overall_exposure": (
            overall_exposure
        ),

        "exposure_by_occupation": (
            exposure_by_occupation
        ),

        "exposure_by_rank": (
            exposure_by_rank
        ),

        "group_counts_per_query": (
            group_counts_per_query
        ),

        "balance_summary": (
            balance_summary
        ),

        "metadata_audit": (
            metadata_audit
        ),
    }

def main():
    # ---------------------------------------------------------
    # Configure experiment file paths
    # ---------------------------------------------------------
    results_dir = Path("datasets/10each")
    output_dir = Path("results/analysis")
    print(results_dir)
    print(output_dir)

    retrieval_summary_path = (
        results_dir
        / "retrieval_summary.csv"
    )

    vanilla_fairness_path = (
        results_dir
        / "sparse_vanilla_fairness_per_query.csv"
    )

    fgr_fairness_path = (
        results_dir
        / "sparse_fgr_fairness_per_query.csv"
    )

    vanilla_slates_path = (
        results_dir
        / "sparse_vanilla_slates.csv"
    )

    fgr_slates_path = (
        results_dir
        / "sparse_fgr_slates.csv"
    )

    # The labels must match the gender metadata stored
    # in the retrieved Haystack documents.
    groups = (
        "Male",
        "Female",
    )
    print("here")
    # ---------------------------------------------------------
    # Run the complete analysis
    # ---------------------------------------------------------
    tables = analyse_retrieval_results(
        retrieval_summary_path=(
            retrieval_summary_path
        ),

        vanilla_fairness_path=(
            vanilla_fairness_path
        ),

        fgr_fairness_path=(
            fgr_fairness_path
        ),

        vanilla_slates_path=(
            vanilla_slates_path
        ),

        fgr_slates_path=(
            fgr_slates_path
        ),

        output_dir=output_dir,

        groups=groups,
    )

    # ---------------------------------------------------------
    # Display the main results
    # ---------------------------------------------------------
    print()
    print("Analysis complete")
    print("=" * 80)

    print()
    print("Aggregate comparison")
    print("-" * 80)

    print(
        tables[
            "aggregate_comparison"
        ].to_string(index=False)
    )

    print()
    print("Fairness by occupation")
    print("-" * 80)

    print(
        tables[
            "fairness_by_occupation"
        ].to_string(index=False)
    )

    print()
    print("Overall exposure")
    print("-" * 80)

    print(
        tables[
            "overall_exposure"
        ].to_string(index=False)
    )

    print()
    print("Balance summary")
    print("-" * 80)

    print(
        tables[
            "balance_summary"
        ].to_string(index=False)
    )

    print()
    print("Metadata audit")
    print("-" * 80)

    print(
        tables[
            "metadata_audit"
        ].to_string(index=False)
    )

    print()
    print(
        "Tables saved to:",
        (output_dir / "tables").resolve(),
    )

    print(
        "Figures saved to:",
        (output_dir / "figures").resolve(),
    )

if __name__ == '__main__':
    main()