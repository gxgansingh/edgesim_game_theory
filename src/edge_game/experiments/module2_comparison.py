"""Module-1 versus Module-2 comparison analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METRICS = [
    "utility_mean",
    "response_time_mean",
    "throughput",
    "success_ratio",
    "rejected_tasks",
    "resource_utilization",
    "load_variance",
    "jains_fairness_index",
    "average_queue_length",
    "priority_success_ratio",
]


def generate_module2_comparison(
    output_directory: str | Path = "outputs/module2",
) -> pd.DataFrame:
    """Compare Module-1 baseline metrics against Module-2 MFG metrics."""
    output_path = Path(output_directory)

    module1_summary_path = (
        Path("outputs")
        / "module1"
        / "aggregated"
        / "module1_summary.csv"
    )

    module2_summary_path = (
        output_path
        / "aggregated"
        / "module2_summary.csv"
    )

    comparison_path = (
        output_path
        / "aggregated"
        / "module1_vs_module2.csv"
    )

    figure_path = (
        output_path
        / "figures"
        / "module1_vs_module2.png"
    )

    module1 = pd.read_csv(
        module1_summary_path
    )

    module2 = pd.read_csv(
        module2_summary_path
    )

    module1 = module1[
        module1["metric"].isin(METRICS)
    ].copy()

    module2 = module2[
        module2["metric"].isin(METRICS)
    ].copy()

    module1 = module1.rename(
        columns={
            "mean": "module1_mean",
            "std": "module1_std",
            "min": "module1_min",
            "max": "module1_max",
        }
    )

    module2 = module2.rename(
        columns={
            "mean": "module2_mean",
            "std": "module2_std",
            "min": "module2_min",
            "max": "module2_max",
        }
    )

    comparison = module1[
        [
            "metric",
            "module1_mean",
            "module1_std",
            "module1_min",
            "module1_max",
        ]
    ].merge(
        module2[
            [
                "metric",
                "module2_mean",
                "module2_std",
                "module2_min",
                "module2_max",
            ]
        ],
        on="metric",
        how="inner",
    )

    comparison["absolute_change"] = (
        comparison["module2_mean"]
        - comparison["module1_mean"]
    )

    comparison["relative_change_percent"] = (
        comparison["absolute_change"]
        / comparison["module1_mean"].replace(
            0,
            float("nan"),
        )
        * 100.0
    )

    comparison.loc[
        comparison["module1_mean"] == 0,
        "relative_change_percent",
    ] = 0.0

    comparison["module2_improvement"] = (
        comparison["absolute_change"]
        > 0
    )

    lower_is_better = {
        "response_time_mean",
        "rejected_tasks",
        "resource_utilization",
        "load_variance",
        "average_queue_length",
    }

    for index, row in comparison.iterrows():
        metric = row["metric"]
        change = row["absolute_change"]

        if abs(change) <= 1e-12:
            improvement = True
        elif metric in lower_is_better:
            improvement = change < 0
        elif metric == "jains_fairness_index":
            improvement = change > 0
        else:
            improvement = change > 0

        comparison.loc[
            index,
            "module2_improvement",
        ] = improvement

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    _generate_comparison_figure(
        comparison=comparison,
        output_path=figure_path,
    )

    _print_comparison(
        comparison=comparison,
    )

    return comparison


def _generate_comparison_figure(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a normalized Module-1 versus Module-2 comparison figure."""
    metrics = comparison["metric"].tolist()

    module1_values = (
        comparison["module1_mean"]
        .to_numpy()
    )

    module2_values = (
        comparison["module2_mean"]
        .to_numpy()
    )

    normalized_module1 = []
    normalized_module2 = []

    for module1_value, module2_value in zip(
        module1_values,
        module2_values,
    ):
        scale = max(
            abs(module1_value),
            abs(module2_value),
            1e-12,
        )

        normalized_module1.append(
            module1_value / scale
        )

        normalized_module2.append(
            module2_value / scale
        )

    positions = range(len(metrics))

    figure, axis = plt.subplots(
        figsize=(12, 7)
    )

    width = 0.38

    axis.bar(
        [
            position - width / 2
            for position in positions
        ],
        normalized_module1,
        width=width,
        label="Module-1 Baseline",
    )

    axis.bar(
        [
            position + width / 2
            for position in positions
        ],
        normalized_module2,
        width=width,
        label="Module-2 Mean-Field",
    )

    axis.set_title(
        "Module-1 Baseline vs Module-2 Mean-Field"
    )

    axis.set_ylabel(
        "Normalized Metric Value"
    )

    axis.set_xticks(
        list(positions)
    )

    axis.set_xticklabels(
        metrics,
        rotation=45,
        ha="right",
    )

    axis.legend()

    axis.grid(
        axis="y",
        alpha=0.3,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=180,
    )

    plt.close(figure)


def _print_comparison(
    comparison: pd.DataFrame,
) -> None:
    """Print the Module-1 versus Module-2 comparison."""
    print(
        "\n=== Module-1 vs Module-2 ==="
    )

    print(
        "Metric | Module-1 | Module-2 | "
        "Change | Relative | Improved"
    )

    print("-" * 95)

    for _, row in comparison.iterrows():
        print(
            f"{row['metric']} | "
            f"{row['module1_mean']:.6f} | "
            f"{row['module2_mean']:.6f} | "
            f"{row['absolute_change']:.6f} | "
            f"{row['relative_change_percent']:.2f}% | "
            f"{bool(row['module2_improvement'])}"
        )


if __name__ == "__main__":
    generate_module2_comparison()