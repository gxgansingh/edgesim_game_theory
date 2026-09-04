"""Network-load performance matrix generation."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..config import SimulationConfig
from .runner import build_mean_field_policy
from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import BaselinePolicy
from .statistics import _confidence_interval


PERFORMANCE_METRICS = [
    "utility_mean",
    "response_time_mean",
    "throughput",
    "success_ratio",
    "resource_utilization",
    "load_variance",
    "jains_fairness_index",
    "average_queue_length",
    "rejected_tasks",
]


def _validate_load_levels(levels: tuple[float, ...]) -> None:
    """Validate network-load values before running experiments."""
    if not levels:
        raise ValueError("At least one network-load level is required.")
    if any(level < 0.0 or level > 1.0 for level in levels):
        raise ValueError("Network-load levels must be between 0 and 1.")


def run_performance_matrix(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    output_directory: str | Path,
) -> pd.DataFrame:
    """Run baseline and MFG experiments across network-load levels."""
    _validate_load_levels(config.network_load_levels)
    if not seeds:
        raise ValueError("At least one experiment seed is required.")

    output_path = Path(output_directory)
    raw_path = output_path / "raw"
    aggregated_path = output_path / "aggregated"
    figures_path = output_path / "figures"

    for directory in (raw_path, aggregated_path, figures_path):
        directory.mkdir(parents=True, exist_ok=True)

    # The equilibrium does not directly depend on the externally varied
    # network-load scalar, so solve it once and reuse it for every load level.
    mean_field_policy, equilibrium_diagnostics = build_mean_field_policy(config)

    rows: list[dict] = []

    for network_load in config.network_load_levels:
        load_config = replace(
            config,
            network_load=float(network_load),
        )

        for seed in seeds:
            seed_config = replace(load_config, seed=int(seed))

            baseline = run_policy_experiment(
                config=seed_config,
                policy_name="least_loaded_baseline",
                policy=BaselinePolicy(config=seed_config),
            )

            mfg = run_policy_experiment(
                config=seed_config,
                policy_name="priority_aware_mean_field",
                policy=mean_field_policy,
            )

            for result in (baseline, mfg):
                rows.append(
                    {
                        "network_load": float(network_load),
                        "seed": int(seed),
                        "policy_name": result.policy_name,
                        **result.metrics,
                    }
                )

    raw_results = pd.DataFrame(rows).sort_values(
        ["network_load", "seed", "policy_name"]
    ).reset_index(drop=True)

    raw_results.to_csv(
        raw_path / "performance_matrix_raw.csv",
        index=False,
    )

    aggregated_rows: list[dict] = []

    for (network_load, policy_name), frame in raw_results.groupby(
        ["network_load", "policy_name"],
        sort=True,
    ):
        for metric in PERFORMANCE_METRICS:
            values = frame[metric].to_numpy(dtype=float)
            ci_low, ci_high = _confidence_interval(values)
            mean = float(values.mean())
            baseline_frame = raw_results.loc[
                raw_results["network_load"] == network_load
            ]
            baseline_values = baseline_frame.loc[
                baseline_frame["policy_name"] == "least_loaded_baseline",
                metric,
            ].to_numpy(dtype=float)

            aggregated_rows.append(
                {
                    "network_load": float(network_load),
                    "policy_name": policy_name,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": mean,
                    "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                    "baseline_mean": float(baseline_values.mean()) if len(baseline_values) else 0.0,
                    "absolute_change_vs_baseline": (
                        mean - float(baseline_values.mean())
                        if len(baseline_values)
                        else 0.0
                    ),
                    "relative_change_percent_vs_baseline": (
                        100.0
                        * (mean - float(baseline_values.mean()))
                        / abs(float(baseline_values.mean()))
                        if len(baseline_values) and abs(float(baseline_values.mean())) > 1e-12
                        else 0.0
                    ),
                }
            )

    aggregated = pd.DataFrame(aggregated_rows)
    aggregated.to_csv(
        aggregated_path / "performance_matrix.csv",
        index=False,
    )

    pd.DataFrame([equilibrium_diagnostics]).to_csv(
        aggregated_path / "equilibrium_diagnostics.csv",
        index=False,
    )

    _generate_figures(aggregated, figures_path)
    _generate_report(
        config=config,
        seeds=seeds,
        aggregated=aggregated,
        equilibrium_diagnostics=equilibrium_diagnostics,
        output_path=output_path / "performance_matrix_report.md",
    )

    return aggregated


def _generate_figures(aggregated: pd.DataFrame, directory: Path) -> None:
    """Generate one network-load figure for each evaluation metric."""
    for metric in PERFORMANCE_METRICS:
        frame = aggregated.loc[aggregated["metric"] == metric].copy()
        if frame.empty:
            continue

        figure, axis = plt.subplots(figsize=(8, 5))

        for policy_name, policy_frame in frame.groupby("policy_name", sort=True):
            policy_frame = policy_frame.sort_values("network_load")
            axis.plot(
                policy_frame["network_load"],
                policy_frame["mean"],
                marker="o",
                label=policy_name,
            )

        axis.set_xlabel("Network Load")
        axis.set_ylabel(metric.replace("_", " ").title())
        axis.set_title(f"{metric.replace('_', ' ').title()} vs Network Load")
        axis.set_ylim(bottom=0.0) if metric in {
            "success_ratio",
            "resource_utilization",
            "jains_fairness_index",
        } else None
        axis.legend()
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(directory / f"network_load_{metric}.png", dpi=180)
        plt.close(figure)


def _generate_report(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    aggregated: pd.DataFrame,
    equilibrium_diagnostics: dict,
    output_path: Path,
) -> None:
    """Generate a concise machine-produced performance report."""
    lines = [
        "# Network-Load Performance Matrix",
        "",
        "## Experiment Design",
        "",
        f"- Network-load levels: {', '.join(f'{v:.2f}' for v in config.network_load_levels)}",
        f"- Repeated seeds: {len(seeds)}",
        "- Policies: least-loaded baseline and priority-aware Mean-Field policy",
        "- Confidence interval: 95% Student-t interval over repeated seeds",
        "",
        "## MFG Equilibrium",
        "",
        f"- Converged: {equilibrium_diagnostics['converged']}",
        f"- Iterations: {equilibrium_diagnostics['iterations']}",
        f"- Distribution residual: {equilibrium_diagnostics['distribution_residual']:.6g}",
        f"- Policy residual: {equilibrium_diagnostics['policy_residual']:.6g}",
        "",
        "## Performance Matrix",
        "",
    ]

    matrix = aggregated[
        [
            "network_load",
            "policy_name",
            "metric",
            "mean",
            "std",
            "ci95_low",
            "ci95_high",
            "relative_change_percent_vs_baseline",
        ]
    ].copy()

    lines.append(matrix.to_markdown(index=False, floatfmt=".4f"))
    lines.extend([
        "",
        "## Interpretation",
        "",
        "The network-load parameter changes the latency pressure used by the resource-feasibility stage. Infeasible nodes are removed before policy selection. The MFG policy then selects among the remaining feasible nodes.",
        "",
        "The CSV files are the authoritative experiment outputs; this report is generated directly from those results.",
    ])

    output_path.write_text("\n".join(lines), encoding="utf-8")
