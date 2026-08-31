"""Module-1 baseline experiment for a simple IoT-Edge architecture."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import BaselinePolicy
from ..config import SimulationConfig


def _run_single_experiment(
    config: SimulationConfig,
    seed: int,
) -> dict:
    """Run one no-priority baseline experiment."""
    seed_config = replace(
        config,
        seed=seed,
    )

    policy = BaselinePolicy(
        config=seed_config,
    )

    result = run_policy_experiment(
        config=seed_config,
        policy_name="module1_no_priority",
        policy=policy,
    )

    metrics = dict(result.metrics)
    metrics["seed"] = seed
    metrics["policy"] = "no_priority_baseline"

    return metrics


def run_module1_experiment(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Run Module-1 using a simple no-priority load-balancing policy."""
    output_path = Path(output_directory)
    raw_path = output_path / "raw"
    aggregated_path = output_path / "aggregated"
    figures_path = output_path / "figures"

    raw_path.mkdir(parents=True, exist_ok=True)
    aggregated_path.mkdir(parents=True, exist_ok=True)
    figures_path.mkdir(parents=True, exist_ok=True)

    rows = [
        _run_single_experiment(
            config=config,
            seed=seed,
        )
        for seed in seeds
    ]

    raw_results = pd.DataFrame(rows)

    raw_results.to_csv(
        raw_path / "module1_raw.csv",
        index=False,
    )

    numeric_columns = raw_results.select_dtypes(
        include="number"
    ).columns

    summary_rows = []

    for column in numeric_columns:
        if column == "seed":
            continue

        summary_rows.append(
            {
                "metric": column,
                "mean": raw_results[column].mean(),
                "std": raw_results[column].std(
                    ddof=1
                ),
                "min": raw_results[column].min(),
                "max": raw_results[column].max(),
            }
        )

    summary = pd.DataFrame(summary_rows)

    summary.to_csv(
        aggregated_path / "module1_summary.csv",
        index=False,
    )

    _generate_figures(
        raw_results=raw_results,
        figures_directory=figures_path,
    )

    _generate_report(
        config=config,
        seeds=seeds,
        summary=summary,
        output_path=output_path / "module1_report.md",
    )

    return raw_results


def _generate_figures(
    raw_results: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate Module-1 metric figures."""
    figure_metrics = [
        (
            "response_time_mean",
            "Response Time",
            "Response time",
        ),
        (
            "throughput",
            "Throughput",
            "Completed tasks",
        ),
        (
            "success_ratio",
            "Success Ratio",
            "Success ratio",
        ),
        (
            "resource_utilization",
            "Resource Utilization",
            "Utilization",
        ),
        (
            "load_variance",
            "Load Variance",
            "Load variance",
        ),
        (
            "jains_fairness_index",
            "Jain's Fairness Index",
            "Fairness index",
        ),
        (
            "average_queue_length",
            "Average Queue Length",
            "Queue length",
        ),
        (
            "rejected_tasks",
            "Rejected Tasks",
            "Rejected tasks",
        ),
        (
            "utility_mean",
            "Mean Utility",
            "Utility",
        ),
    ]

    for metric, title, ylabel in figure_metrics:
        if metric not in raw_results.columns:
            continue

        figure, axis = plt.subplots(
            figsize=(8, 5)
        )

        axis.plot(
            raw_results["seed"],
            raw_results[metric],
            marker="o",
        )

        axis.set_title(
            f"Module-1: {title}"
        )
        axis.set_xlabel("Seed")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)

        figure.tight_layout()

        figure.savefig(
            figures_directory
            / f"module1_{metric}.png",
            dpi=180,
        )

        plt.close(figure)


def _generate_report(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate a concise Module-1 experiment report."""
    architecture = """
IoT Devices
    |
    v
Edge Gateway
    |
    v
Edge Server
    |
    v
Edge Nodes
    |
    +-- Node 1
    +-- Node 2
    +-- Node 3
    +-- ...
"""

    lines = [
        "# Module-1 Experiment Report",
        "",
        "## Architecture",
        "",
        "Simple IoT-to-Edge architecture with no priority-based scheduling.",
        "",
        "```text",
        architecture.strip(),
        "```",
        "",
        "## Load-Balancing Policy",
        "",
        "The experiment uses the baseline least-loaded allocation policy.",
        "Tasks are assigned to the feasible edge node with the lowest",
        "current load ratio, with queue length used as a secondary",
        "tie-breaking criterion.",
        "",
        "## Configuration",
        "",
        f"- Simulation steps: {config.simulation_steps}",
        f"- Tasks per step: {config.tasks_per_step}",
        f"- Servers: {config.number_of_servers}",
        f"- Nodes per server: {config.nodes_per_server}",
        f"- Seeds: {len(seeds)}",
        "",
        "## Metrics",
        "",
        "The experiment records:",
        "",
        "- Mean utility",
        "- Mean response time",
        "- Throughput",
        "- Success ratio",
        "- Rejected tasks",
        "- Resource utilization",
        "- Load variance",
        "- Jain's fairness index",
        "- Average queue length",
        "",
        "## Aggregated Results",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['metric']} | "
            f"{row['mean']:.6f} | "
            f"{row['std']:.6f} | "
            f"{row['min']:.6f} | "
            f"{row['max']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Module-1 establishes the baseline load-balancing behavior",
            "for the simplified IoT-Edge network without priority-aware",
            "decision making.",
            "",
            "The resulting metrics provide the reference point for",
            "later modules that introduce additional decision logic.",
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )