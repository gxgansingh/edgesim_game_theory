"""Module-3 hierarchical server-then-node experiment."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..algorithms.hierarchical_experiment import (
    run_hierarchical_policy_experiment,
)
from ..algorithms.policy import BaselinePolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy


def _run_single_experiment(
    config: SimulationConfig,
    seed: int,
) -> dict:
    """Run one hierarchical Mean-Field experiment."""
    seed_config = replace(
        config,
        seed=seed,
    )

    node_policy, equilibrium_diagnostics = (
        build_mean_field_policy(
            config=seed_config,
            ablation_variant="no_priority",
        )
    )

    policy = __import__(
        "src.edge_game.algorithms.policy",
        fromlist=["HierarchicalPolicy"],
    ).HierarchicalPolicy(
        node_policy=node_policy,
        config=seed_config,
    )

    result = run_hierarchical_policy_experiment(
        config=seed_config,
        policy_name="module3_hierarchical_no_priority",
        policy=policy,
    )

    metrics = dict(result.metrics)

    metrics["seed"] = seed
    metrics["policy"] = (
        "hierarchical_no_priority"
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

    metrics["server_decisions"] = len(
        result.server_decision_records
    )

    metrics["average_selected_server_rank"] = (
        pd.DataFrame(
            result.server_decision_records
        )["selected_server_rank"].mean()
        if result.server_decision_records
        else 0.0
    )

    return metrics


def run_module3_experiment(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Run Module-3 hierarchical load balancing."""
    output_path = Path(output_directory)

    raw_path = output_path / "raw"
    aggregated_path = output_path / "aggregated"
    figures_path = output_path / "figures"

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
        raw_path / "module3_raw.csv",
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
                ].std(ddof=1),
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
        / "module3_summary.csv",
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
            / "module3_report.md"
        ),
    )

    return raw_results


def _generate_figures(
    raw_results: pd.DataFrame,
    figures_directory: Path,
) -> None:
    """Generate Module-3 metric figures."""
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
        (
            "average_selected_server_rank",
            "Selected Server Rank",
            "Average rank",
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
            f"Module-3: {title}"
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
            / f"module3_{metric}.png",
            dpi=180,
        )

        plt.close(figure)


def _generate_report(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the Module-3 experiment report."""
    architecture = """
IoT Devices
    |
    v
Edge Gateway
    |
    v
Edge Servers
    |
    +-- Server 1
    |     +-- Node 1
    |     +-- Node 2
    |     +-- ...
    |
    +-- Server 2
    |     +-- Node 1
    |     +-- Node 2
    |     +-- ...
    |
    +-- Server 3
          +-- Node 1
          +-- Node 2
          +-- ...
"""

    lines = [
        "# Module-3 Experiment Report",
        "",
        "## Architecture",
        "",
        (
            "Hierarchical IoT-to-Edge architecture "
            "with server-level and node-level "
            "load balancing."
        ),
        "",
        "```text",
        architecture.strip(),
        "```",
        "",
        "## Hierarchical Decision Process",
        "",
        "1. Generate an IoT task.",
        "2. Identify feasible edge servers.",
        "3. Select the least-loaded feasible server.",
        "4. Identify feasible nodes inside the selected server.",
        "5. Apply the no-priority Mean-Field node policy.",
        "6. Allocate the task to the selected node.",
        "",
        "## Server-Level Policy",
        "",
        (
            "Servers are ranked using the mean CPU load "
            "of their managed edge nodes."
        ),
        "",
        (
            "Total queue length is used as a secondary "
            "tie-breaking criterion."
        ),
        "",
        "## Node-Level Policy",
        "",
        (
            "The existing no-priority Mean-Field policy "
            "is retained at the edge-node layer."
        ),
        "",
        "## Configuration",
        "",
        f"- Simulation steps: {config.simulation_steps}",
        f"- Tasks per step: {config.tasks_per_step}",
        f"- Servers: {config.number_of_servers}",
        f"- Nodes per server: {config.nodes_per_server}",
        f"- Seeds: {len(seeds)}",
        "- Server policy: least-loaded feasible server",
        "- Node policy: no-priority Mean-Field",
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
        "- Priority success ratio",
        "- Average selected server rank",
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
                "Module-3 extends node-level allocation "
                "into a hierarchical server-then-node "
                "decision process."
            ),
            "",
            (
                "The server layer first reduces the "
                "node search space to one selected "
                "edge-server cluster."
            ),
            "",
            (
                "The existing no-priority Mean-Field "
                "decision logic is then applied inside "
                "the selected server."
            ),
            "",
        ]
    )

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )