"""Module-2 Mean-Field Game experiment with no-priority load balancing."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import MeanFieldPolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy


def _run_single_experiment(
    config: SimulationConfig,
    seed: int,
) -> dict:
    """Run one no-priority Mean-Field Game experiment."""
    seed_config = replace(
        config,
        seed=seed,
    )

    policy, equilibrium_diagnostics = (
        build_mean_field_policy(
            config=seed_config,
            ablation_variant="no_priority",
        )
    )

    result = run_policy_experiment(
        config=seed_config,
        policy_name="module2_no_priority_mean_field",
        policy=policy,
    )

    metrics = dict(result.metrics)

    metrics["seed"] = seed
    metrics["policy"] = (
        "no_priority_mean_field"
    )

    metrics["equilibrium_converged"] = (
        equilibrium_diagnostics["converged"]
    )

    metrics["equilibrium_iterations"] = (
        equilibrium_diagnostics["iterations"]
    )

    metrics["equilibrium_residual"] = (
        equilibrium_diagnostics[
            "distribution_residual"
        ]
    )

    metrics["equilibrium_policy_residual"] = (
        equilibrium_diagnostics[
            "policy_residual"
        ]
    )

    return metrics


def run_module2_experiment(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Run Module-2 using the no-priority Mean-Field policy."""
    output_path = Path(output_directory)

    raw_path = (
        output_path / "raw"
    )

    aggregated_path = (
        output_path / "aggregated"
    )

    figures_path = (
        output_path / "figures"
    )

    raw_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = [
        _run_single_experiment(
            config=config,
            seed=seed,
        )
        for seed in seeds
    ]

    raw_results = pd.DataFrame(rows)

    raw_results.to_csv(
        raw_path / "module2_raw.csv",
        index=False,
    )

    numeric_columns = (
        raw_results.select_dtypes(
            include="number"
        ).columns
    )

    summary_rows = []

    for column in numeric_columns:
        if column == "seed":
            continue

        summary_rows.append(
            {
                "metric": column,
                "mean": raw_results[
                    column
                ].mean(),
                "std": raw_results[
                    column
                ].std(
                    ddof=1
                ),
                "min": raw_results[
                    column
                ].min(),
                "max": raw_results[
                    column
                ].max(),
            }
        )

    summary = pd.DataFrame(
        summary_rows
    )

    summary.to_csv(
        aggregated_path
        / "module2_summary.csv",
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
        output_path=(
            output_path
            / "module2_report.md"
        ),
    )

    return raw_results


def _generate_figures(
    raw_results: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate Module-2 metric figures."""
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
            f"Module-2: {title}"
        )

        axis.set_xlabel(
            "Seed"
        )

        axis.set_ylabel(
            ylabel
        )

        axis.grid(
            True,
            alpha=0.3,
        )

        figure.tight_layout()

        figure.savefig(
            figures_directory
            / f"module2_{metric}.png",
            dpi=180,
        )

        plt.close(
            figure
        )


def _generate_report(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the Module-2 experiment report."""
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
        "# Module-2 Experiment Report",
        "",
        "## Architecture",
        "",
        (
            "Simple IoT-to-Edge architecture "
            "with no-priority load balancing."
        ),
        "",
        "```text",
        architecture.strip(),
        "```",
        "",
        "## Load-Balancing Policy",
        "",
        (
            "The experiment uses the "
            "Mean-Field Game allocation policy "
            "with the no-priority ablation."
        ),
        "",
        (
            "The policy evaluates each feasible "
            "edge node using its Mean-Field state, "
            "control and aggregate mean-field state."
        ),
        "",
        (
            "The node with the minimum Mean-Field "
            "running cost is selected."
        ),
        "",
        "## Configuration",
        "",
        f"- Simulation steps: {config.simulation_steps}",
        f"- Tasks per step: {config.tasks_per_step}",
        f"- Servers: {config.number_of_servers}",
        f"- Nodes per server: {config.nodes_per_server}",
        f"- Seeds: {len(seeds)}",
        "- Ablation variant: no_priority",
        "",
        "## Mean-Field Formulation",
        "",
        (
            "The no-priority formulation removes "
            "priority-dependent reward from the "
            "decision objective."
        ),
        "",
        (
            "Priority-specific service preferences "
            "are replaced by population-weighted "
            "neutral parameters."
        ),
        "",
        "## Metrics",
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
            (
                "Module-2 evaluates Mean-Field Game "
                "based load balancing without using "
                "priority as a decision preference."
            ),
            "",
            (
                "The results provide a second "
                "reference point against the "
                "Module-1 least-loaded baseline."
            ),
            "",
            (
                "The primary comparison focuses on "
                "response time, throughput, resource "
                "utilization, load variance and "
                "Jain's fairness index."
            ),
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )