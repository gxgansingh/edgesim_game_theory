"""Benchmark the edge load balancer on single and repeated runs."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..algorithms.experiment import ExperimentResult, run_policy_experiment
from ..algorithms.policy import BaselinePolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy
from .module2 import _generate_resource_filtering_figure, _summarize_filtering_audit


BENCHMARK_METRICS = (
    "utility_mean",
    "response_time_mean",
    "throughput",
    "success_ratio",
    "resource_utilization",
    "load_variance",
    "jains_fairness_index",
    "average_queue_length",
    "rejected_tasks",
)

POLICY_LABELS = {
    "mfg_priority": "MFG Load Balancer",
    "least_loaded": "Least-Loaded Baseline",
}


def _run_policy(
    config: SimulationConfig,
    seed: int,
    policy_key: str,
    mfg_policy=None,
) -> ExperimentResult:
    """Run one benchmark policy for one seed."""
    seed_config = replace(config, seed=seed)

    if policy_key == "mfg_priority":
        if mfg_policy is None:
            mfg_policy, _ = build_mean_field_policy(
                config=seed_config,
                ablation_variant="full",
            )
        policy = mfg_policy
    elif policy_key == "least_loaded":
        policy = BaselinePolicy(config=seed_config)
    else:
        raise ValueError(f"Unknown benchmark policy: {policy_key}")

    return run_policy_experiment(
        config=seed_config,
        policy_name=POLICY_LABELS[policy_key],
        policy=policy,
    )


def _metric_row(
    seed: int,
    policy_key: str,
    result: ExperimentResult,
    equilibrium_diagnostics: dict | None = None,
) -> dict:
    """Convert one experiment result into a benchmark row."""
    row = {
        "seed": int(seed),
        "policy": policy_key,
        "policy_label": POLICY_LABELS[policy_key],
    }
    row.update({metric: float(result.metrics.get(metric, 0.0)) for metric in BENCHMARK_METRICS})
    diagnostics = equilibrium_diagnostics or {}
    row["equilibrium_converged"] = bool(
        diagnostics.get("converged", result.metrics.get("equilibrium_converged", True))
    )
    row["equilibrium_iterations"] = int(
        diagnostics.get("iterations", result.metrics.get("equilibrium_iterations", 0))
    )
    row["equilibrium_distribution_residual"] = float(
        diagnostics.get("distribution_residual", result.metrics.get("equilibrium_distribution_residual", 0.0))
    )
    row["equilibrium_policy_residual"] = float(
        diagnostics.get("policy_residual", result.metrics.get("equilibrium_policy_residual", 0.0))
    )
    return row


def _single_node_rows(
    result: ExperimentResult,
    simulation_steps: int,
) -> pd.DataFrame:
    """Build per-node time-series data for one run."""
    frame = pd.DataFrame(result.node_state_records)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "tick",
                "node_id",
                "server_id",
                "cpu_utilization",
                "memory_utilization",
                "bandwidth_utilization",
                "queue_length",
            ]
        )
    return frame.loc[frame["tick"] < simulation_steps].copy()


def _single_node_summary(
    node_history: pd.DataFrame,
    selection_records: list[dict],
) -> pd.DataFrame:
    """Summarize one-run node utilization and selection distribution."""
    if node_history.empty:
        return pd.DataFrame()

    selection_counts = pd.Series(dtype=float)
    if selection_records:
        selection_counts = pd.Series(
            [int(record["node_id"]) for record in selection_records]
        ).value_counts()

    rows = []
    for node_id, group in node_history.groupby("node_id", sort=True):
        rows.append(
            {
                "node_id": int(node_id),
                "server_id": int(group["server_id"].iloc[0]),
                "average_cpu_utilization": float(group["cpu_utilization"].mean()),
                "peak_cpu_utilization": float(group["cpu_utilization"].max()),
                "final_cpu_utilization": float(group["cpu_utilization"].iloc[-1]),
                "average_memory_utilization": float(group["memory_utilization"].mean()),
                "average_bandwidth_utilization": float(group["bandwidth_utilization"].mean()),
                "average_queue_length": float(group["queue_length"].mean()),
                "tasks_selected": int(selection_counts.get(node_id, 0)),
            }
        )
    return pd.DataFrame(rows)


def _summarize_repeated_runs(raw: pd.DataFrame) -> pd.DataFrame:
    """Calculate mean, standard deviation, and 95% t confidence interval."""
    rows = []
    for (policy, label), group in raw.groupby(["policy", "policy_label"], sort=False):
        n = len(group)
        for metric in BENCHMARK_METRICS:
            values = group[metric].astype(float)
            mean = float(values.mean())
            std = float(values.std(ddof=1)) if n > 1 else 0.0
            if n > 1:
                margin = float(stats.t.ppf(0.975, n - 1) * std / np.sqrt(n))
            else:
                margin = 0.0
            rows.append(
                {
                    "policy": policy,
                    "policy_label": label,
                    "metric": metric,
                    "runs": int(n),
                    "mean": mean,
                    "std": std,
                    "min": float(values.min()),
                    "max": float(values.max()),
                    "ci95_lower": mean - margin,
                    "ci95_upper": mean + margin,
                }
            )
    return pd.DataFrame(rows)


def _summarize_repeated_nodes(node_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate average CPU utilization by node across repeated runs."""
    if node_rows.empty:
        return pd.DataFrame()
    return (
        node_rows.groupby(["policy", "policy_label", "node_id", "server_id"], as_index=False)
        .agg(
            average_cpu_utilization_mean=("average_cpu_utilization", "mean"),
            average_cpu_utilization_std=("average_cpu_utilization", "std"),
            peak_cpu_utilization_mean=("peak_cpu_utilization", "mean"),
            final_cpu_utilization_mean=("final_cpu_utilization", "mean"),
            average_queue_length_mean=("average_queue_length", "mean"),
            tasks_selected_mean=("tasks_selected", "mean"),
        )
        .fillna(0.0)
    )


def _generate_single_run_figure(
    node_history: pd.DataFrame,
    output_path: Path,
) -> None:
    """Generate the attached-figure-style CPU utilization benchmark plot."""
    figure, axis = plt.subplots(figsize=(12, 6.5))

    for node_id, group in node_history.groupby("node_id", sort=True):
        axis.plot(
            group["tick"],
            group["cpu_utilization"] * 100.0,
            marker="o",
            markersize=2.5,
            linewidth=1.2,
            label=f"Edge Node {int(node_id) + 1}",
        )

    axis.set_title("MFG Load Balancer: Single-Run Edge CPU Utilization")
    axis.set_xlabel("Simulation Tick")
    axis.set_ylabel("CPU Utilization (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        ncol=1,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _generate_repeated_time_series_figure(
    time_series_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot per-node mean CPU utilization over ticks across repeated runs."""
    mfg = time_series_summary.loc[
        time_series_summary["policy"] == "mfg_priority"
    ]
    if mfg.empty:
        return

    figure, axis = plt.subplots(figsize=(12, 6.5))
    for node_id, group in mfg.groupby("node_id", sort=True):
        axis.plot(
            group["tick"],
            group["mean_cpu_utilization"] * 100.0,
            linewidth=1.2,
            label=f"Edge Node {int(node_id) + 1}",
        )

    axis.set_title("MFG Load Balancer: Mean Edge CPU Utilization Across 10 Runs")
    axis.set_xlabel("Simulation Tick")
    axis.set_ylabel("Mean CPU Utilization (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, alpha=0.3)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _generate_repeated_node_figure(
    node_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot mean CPU utilization across ten runs for every edge node."""
    mfg = node_summary.loc[node_summary["policy"] == "mfg_priority"]
    if mfg.empty:
        return

    figure, axis = plt.subplots(figsize=(12, 6))
    axis.bar(
        mfg["node_id"].astype(str),
        mfg["average_cpu_utilization_mean"] * 100.0,
        yerr=mfg["average_cpu_utilization_std"].fillna(0.0) * 100.0,
        capsize=3,
    )
    axis.set_title("MFG Load Balancer: Mean Edge CPU Utilization Across Repeated Runs")
    axis.set_xlabel("Edge Node ID")
    axis.set_ylabel("Mean CPU Utilization (%)")
    axis.set_ylim(0, 100)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _generate_metric_comparison_figure(
    raw: pd.DataFrame,
    metric: str,
    output_path: Path,
) -> None:
    """Plot a repeated-run metric for the load balancer and baseline."""
    groups = []
    labels = []
    means = []
    errors = []
    for policy in ("mfg_priority", "least_loaded"):
        values = raw.loc[raw["policy"] == policy, metric].astype(float)
        if values.empty:
            continue
        groups.append(policy)
        labels.append(POLICY_LABELS[policy])
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    if not groups:
        return

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(labels, means, yerr=errors, capsize=4)
    axis.set_title(f"10-Run Benchmark: {metric.replace('_', ' ').title()}")
    axis.set_ylabel(metric.replace("_", " ").title())
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _write_report(
    output_path: Path,
    config: SimulationConfig,
    seeds: tuple[int, ...],
    single_metrics: dict,
    repeated_summary: pd.DataFrame,
) -> None:
    """Write a reproducible benchmark report."""
    lines = [
        "# Edge Load Balancer Benchmark",
        "",
        "## Benchmark Design",
        "",
        f"- Single-run seed: `{seeds[0]}`",
        f"- Repeated runs: `{len(seeds)}`",
        f"- Simulation steps per run: `{config.simulation_steps}`",
        f"- Tasks per step: `{config.tasks_per_step}`",
        f"- Edge servers: `{config.number_of_servers}`",
        f"- Edge nodes per server: `{config.nodes_per_server}`",
        f"- Total edge nodes: `{config.number_of_servers * config.nodes_per_server}`",
        "- Primary policy: MFG Load Balancer",
        "- Reference policy: Least-Loaded Baseline",
        f"- Benchmark MFG state points: `{config.mean_field_state_points}`",
        f"- Benchmark MFG iteration limit: `{config.mean_field_max_iterations}`",
        f"- Benchmark MFG tolerance: `{config.mean_field_tolerance}`",
        f"- Benchmark FPK iteration limit: `{config.fpk_max_iterations}`",
        f"- Benchmark FPK tolerance: `{config.fpk_tolerance}`",
        "",
        "## Single-Run Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for metric in BENCHMARK_METRICS:
        lines.append(f"| {metric} | {single_metrics.get(metric, 0.0):.6f} |")

    lines.extend(
        [
            "",
            "## 10-Run Results",
            "",
            "| Policy | Metric | Mean | Std | 95% CI |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for _, row in repeated_summary.iterrows():
        lines.append(
            f"| {row['policy_label']} | {row['metric']} | "
            f"{row['mean']:.6f} | {row['std']:.6f} | "
            f"[{row['ci95_lower']:.6f}, {row['ci95_upper']:.6f}] |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The single-run time-series figure shows how the load balancer distributes CPU utilization across edge nodes over simulation ticks.",
            "The repeated-run tables report mean, standard deviation, and 95% Student-t confidence intervals so that performance is not judged from one lucky random seed.",
            "The least-loaded policy is included as a reference baseline. It is not the proposed load balancer.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_load_balancer_benchmark(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> dict[str, Path]:
    """Run single-run and repeated-run load-balancer benchmarks."""
    if not seeds:
        raise ValueError("At least one benchmark seed is required.")

    output = Path(output_directory)
    raw_dir = output / "raw"
    aggregated_dir = output / "aggregated"
    figures_dir = output / "figures"
    for directory in (raw_dir, aggregated_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    primary_seed = seeds[0]
    benchmark_config = replace(
        config,
        simulation_steps=config.benchmark_simulation_steps,
        mean_field_state_points=config.benchmark_mean_field_state_points,
        mean_field_max_iterations=config.benchmark_mean_field_max_iterations,
        mean_field_tolerance=config.benchmark_mean_field_tolerance,
        mean_field_policy_tolerance=config.benchmark_mean_field_policy_tolerance,
        mean_field_raw_policy_tolerance=config.benchmark_mean_field_raw_policy_tolerance,
        fpk_max_iterations=config.benchmark_fpk_max_iterations,
        fpk_tolerance=config.benchmark_fpk_tolerance,
    )
    mfg_policy, equilibrium_diagnostics = build_mean_field_policy(
        config=replace(benchmark_config, seed=primary_seed),
        ablation_variant="full",
    )
    single_result = _run_policy(
        benchmark_config,
        primary_seed,
        "mfg_priority",
        mfg_policy=mfg_policy,
    )
    single_history = _single_node_rows(single_result, benchmark_config.simulation_steps)
    single_summary = _single_node_summary(
        single_history,
        single_result.selection_records,
    )

    single_metrics = pd.DataFrame(
        [{
            "seed": primary_seed,
            **{metric: float(single_result.metrics.get(metric, 0.0)) for metric in BENCHMARK_METRICS},
            "equilibrium_converged": bool(equilibrium_diagnostics["converged"]),
            "equilibrium_iterations": int(equilibrium_diagnostics["iterations"]),
            "equilibrium_distribution_residual": float(equilibrium_diagnostics["distribution_residual"]),
            "equilibrium_policy_residual": float(equilibrium_diagnostics["policy_residual"]),
        }]
    )
    single_metrics.to_csv(raw_dir / "single_run_metrics.csv", index=False)
    single_history.to_csv(raw_dir / "single_run_node_utilization.csv", index=False)
    single_summary.to_csv(aggregated_dir / "single_run_node_summary.csv", index=False)

    single_filtering = pd.DataFrame(
        [
            {
                "seed": int(primary_seed),
                "policy": "mfg_priority",
                **record,
            }
            for record in single_result.filtering_records
        ]
    )
    single_selection = pd.DataFrame(
        [
            {
                "seed": int(primary_seed),
                "policy": "mfg_priority",
                **record,
            }
            for record in single_result.selection_records
        ]
    )
    single_filtering.to_csv(raw_dir / "resource_filtering_audit.csv", index=False)
    single_filtering_summary = _summarize_filtering_audit(single_filtering)
    single_filtering_summary.to_csv(aggregated_dir / "resource_filtering_summary.csv", index=False)
    _generate_resource_filtering_figure(
        filtering_audit=single_filtering,
        selection_audit=single_selection,
        figures_directory=figures_dir,
    )

    repeated_rows = []
    repeated_node_rows = []
    repeated_time_rows = []
    for seed in seeds:
        for policy_key in ("mfg_priority", "least_loaded"):
            result = _run_policy(
                benchmark_config,
                seed,
                policy_key,
                mfg_policy=mfg_policy,
            )
            repeated_rows.append(
                _metric_row(
                    seed,
                    policy_key,
                    result,
                    equilibrium_diagnostics if policy_key == "mfg_priority" else None,
                )
            )

            history = _single_node_rows(result, benchmark_config.simulation_steps)
            if not history.empty:
                history = history.copy()
                history.insert(0, "policy_label", POLICY_LABELS[policy_key])
                history.insert(0, "policy", policy_key)
                history.insert(0, "seed", int(seed))
                repeated_time_rows.append(history)
            node_summary = _single_node_summary(history, result.selection_records)
            if not node_summary.empty:
                node_summary.insert(0, "policy_label", POLICY_LABELS[policy_key])
                node_summary.insert(0, "policy", policy_key)
                node_summary.insert(0, "seed", int(seed))
                repeated_node_rows.append(node_summary)

    repeated_raw = pd.DataFrame(repeated_rows)
    repeated_summary = _summarize_repeated_runs(repeated_raw)
    repeated_nodes_raw = pd.concat(repeated_node_rows, ignore_index=True) if repeated_node_rows else pd.DataFrame()
    repeated_nodes_summary = _summarize_repeated_nodes(repeated_nodes_raw)
    repeated_time_raw = pd.concat(repeated_time_rows, ignore_index=True) if repeated_time_rows else pd.DataFrame()
    if not repeated_time_raw.empty:
        repeated_time_summary = (
            repeated_time_raw.groupby(
                ["policy", "policy_label", "node_id", "server_id", "tick"],
                as_index=False,
            )
            .agg(
                mean_cpu_utilization=("cpu_utilization", "mean"),
                std_cpu_utilization=("cpu_utilization", "std"),
            )
            .fillna(0.0)
        )
    else:
        repeated_time_summary = pd.DataFrame()

    repeated_raw.to_csv(raw_dir / "ten_run_benchmark_raw.csv", index=False)
    repeated_summary.to_csv(aggregated_dir / "ten_run_benchmark_summary.csv", index=False)
    repeated_nodes_raw.to_csv(raw_dir / "ten_run_node_utilization_raw.csv", index=False)
    repeated_nodes_summary.to_csv(aggregated_dir / "ten_run_node_utilization_summary.csv", index=False)
    repeated_time_raw.to_csv(raw_dir / "ten_run_node_utilization_time_series_raw.csv", index=False)
    repeated_time_summary.to_csv(aggregated_dir / "ten_run_node_utilization_time_series_summary.csv", index=False)

    _generate_single_run_figure(
        single_history,
        figures_dir / "single_run_node_utilization.png",
    )
    _generate_repeated_node_figure(
        repeated_nodes_summary,
        figures_dir / "ten_run_node_utilization.png",
    )
    _generate_repeated_time_series_figure(
        repeated_time_summary,
        figures_dir / "ten_run_node_utilization_time_series.png",
    )
    for metric in BENCHMARK_METRICS:
        _generate_metric_comparison_figure(
            repeated_raw,
            metric,
            figures_dir / f"ten_run_{metric}.png",
        )

    report_path = output / "load_balancer_benchmark_report.md"
    _write_report(
        report_path,
        benchmark_config,
        seeds,
        single_result.metrics,
        repeated_summary,
    )

    return {
        "single_metrics": raw_dir / "single_run_metrics.csv",
        "single_node_utilization": raw_dir / "single_run_node_utilization.csv",
        "single_node_summary": aggregated_dir / "single_run_node_summary.csv",
        "resource_filtering_audit": raw_dir / "resource_filtering_audit.csv",
        "resource_filtering_summary": aggregated_dir / "resource_filtering_summary.csv",
        "resource_filtering_figure": figures_dir / "resource_filtering_selection_audit.png",
        "ten_run_raw": raw_dir / "ten_run_benchmark_raw.csv",
        "ten_run_summary": aggregated_dir / "ten_run_benchmark_summary.csv",
        "ten_run_node_raw": raw_dir / "ten_run_node_utilization_raw.csv",
        "ten_run_node_summary": aggregated_dir / "ten_run_node_utilization_summary.csv",
        "ten_run_time_series_raw": raw_dir / "ten_run_node_utilization_time_series_raw.csv",
        "ten_run_time_series_summary": aggregated_dir / "ten_run_node_utilization_time_series_summary.csv",
        "single_figure": figures_dir / "single_run_node_utilization.png",
        "ten_run_node_figure": figures_dir / "ten_run_node_utilization.png",
        "ten_run_time_series_figure": figures_dir / "ten_run_node_utilization_time_series.png",
        "report": report_path,
    }
