"""Visualization utilities for saved edge-game experiment results."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


POLICY_LABELS = {
    "least_loaded_baseline": "Least-Loaded Baseline",
    "priority_aware_mean_field": "Priority-Aware Mean-Field",
}

METRIC_LABELS = {
    "utility_mean": "Utility",
    "response_time_mean": "Response Time",
    "throughput": "Throughput",
    "success_ratio": "Success Ratio",
    "rejected_tasks": "Rejected Tasks",
    "resource_utilization": "Resource Utilization",
    "load_variance": "Load Variance",
    "jains_fairness_index": "Jain's Fairness",
    "average_queue_length": "Average Queue Length",
    "priority_success_ratio": "Priority Success Ratio",
}


def _require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: Path,
) -> None:
    """Validate that a result file contains required columns."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {missing}."
        )


def _save_figure(
    figure: plt.Figure,
    output_path: Path,
) -> Path:
    """Save one figure and close it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)
    return output_path



def _select_profile(
    frame: pd.DataFrame,
    preferred: str = "balanced",
) -> pd.DataFrame:
    """Select one utility profile when a result contains multiple profiles."""
    if "profile" not in frame.columns:
        return frame.copy()

    profiles = list(frame["profile"].dropna().unique())
    if not profiles:
        return frame.copy()

    selected = preferred if preferred in profiles else profiles[0]
    return frame.loc[frame["profile"] == selected].copy()


def plot_policy_performance(
    paired_comparison: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot paired Mean-Field deltas across workload scenarios."""
    source = Path(output_path)
    required = [
        "scenario",
        "metric",
        "mean_difference",
    ]
    _require_columns(
        paired_comparison,
        required,
        source,
    )

    selected_metrics = [
        "utility_mean",
        "response_time_mean",
        "throughput",
        "success_ratio",
        "resource_utilization",
        "load_variance",
        "jains_fairness_index",
        "average_queue_length",
        "priority_success_ratio",
    ]

    frame = _select_profile(paired_comparison)
    frame = frame.loc[
        frame["metric"].isin(selected_metrics)
    ].copy()

    scenarios = list(frame["scenario"].drop_duplicates())
    metrics = [
        metric
        for metric in selected_metrics
        if metric in set(frame["metric"])
    ]

    figure, axes = plt.subplots(
        len(metrics),
        1,
        figsize=(10, max(8, len(metrics) * 1.6)),
        squeeze=False,
    )

    for index, metric in enumerate(metrics):
        axis = axes[index, 0]
        metric_frame = frame.loc[
            frame["metric"] == metric
        ]

        values = []
        for scenario in scenarios:
            row = metric_frame.loc[
                metric_frame["scenario"] == scenario
            ]
            values.append(
                float(row["mean_difference"].iloc[0])
                if not row.empty
                else np.nan
            )

        axis.bar(
            np.arange(len(scenarios)),
            values,
        )
        axis.axhline(
            0.0,
            linewidth=0.8,
        )
        axis.set_ylabel(
            METRIC_LABELS.get(metric, metric)
        )
        axis.set_xticks(
            np.arange(len(scenarios)),
            scenarios,
        )
        axis.grid(
            axis="y",
            alpha=0.25,
        )

    return _save_figure(
        figure,
        source,
    )


def plot_utility_weight_sensitivity(
    paired_comparison: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot Mean-Field utility improvement for every utility profile."""
    source = Path(output_path)
    required = [
        "profile",
        "scenario",
        "metric",
        "mean_difference",
    ]
    _require_columns(
        paired_comparison,
        required,
        source,
    )

    frame = paired_comparison.loc[
        paired_comparison["metric"] == "utility_mean"
    ].copy()

    pivot = frame.pivot_table(
        index="profile",
        columns="scenario",
        values="mean_difference",
        aggfunc="mean",
    )

    figure, axis = plt.subplots(
        figsize=(10, 5.5)
    )

    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(len(pivot.columns)),
        pivot.columns,
    )
    axis.set_yticks(
        np.arange(len(pivot.index)),
        pivot.index,
    )
    axis.set_xlabel("Workload Scenario")
    axis.set_ylabel("Utility Profile")
    axis.set_title("Mean-Field Utility Improvement")

    for row_index in range(pivot.shape[0]):
        for column_index in range(pivot.shape[1]):
            value = pivot.iloc[
                row_index,
                column_index,
            ]
            if pd.notna(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.4f}",
                    ha="center",
                    va="center",
                )

    figure.colorbar(
        image,
        ax=axis,
        label="Mean-Field minus Baseline",
    )

    return _save_figure(
        figure,
        source,
    )


def plot_selection_divergence(
    paired_selection: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot node-selection divergence by workload scenario."""
    source = Path(output_path)

    required = [
        "scenario",
        "metric",
        "comparison_mean",
        "ci95_low",
        "ci95_high",
    ]

    _require_columns(
        paired_selection,
        required,
        source,
    )

    frame = _select_profile(paired_selection)

    frame = frame.loc[
        frame["metric"] == "selection_divergence_rate"
    ].copy()

    frame = frame.reset_index(drop=True)

    figure, axis = plt.subplots(
        figsize=(9, 5)
    )

    scenarios = frame["scenario"].astype(str).tolist()

    means = (
        pd.to_numeric(
            frame["comparison_mean"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        .reshape(-1)
    )

    lower = (
        pd.to_numeric(
            frame["ci95_low"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        .reshape(-1)
    )

    upper = (
        pd.to_numeric(
            frame["ci95_high"],
            errors="coerce",
        )
        .to_numpy(dtype=float)
        .reshape(-1)
    )

    if not (
        len(scenarios)
        == len(means)
        == len(lower)
        == len(upper)
    ):
        raise ValueError(
            "Selection divergence data has inconsistent lengths: "
            f"scenarios={len(scenarios)}, "
            f"means={len(means)}, "
            f"ci95_low={len(lower)}, "
            f"ci95_high={len(upper)}."
        )

    if np.isnan(means).any():
        raise ValueError(
            "Selection divergence contains invalid comparison_mean values."
        )

    if np.isnan(lower).any() or np.isnan(upper).any():
        raise ValueError(
            "Selection divergence contains invalid confidence interval values."
        )

    lower_errors = means - lower
    upper_errors = upper - means

    if np.any(lower_errors < 0) or np.any(upper_errors < 0):
        raise ValueError(
            "Selection divergence confidence intervals are invalid: "
            "ci95_low must be <= comparison_mean <= ci95_high."
        )

    x = np.arange(len(scenarios))

    axis.bar(
        x,
        means,
    )

    # Draw confidence intervals manually instead of passing yerr
    # through Matplotlib's bar/errorbar machinery.
    for position, mean, lower_error, upper_error in zip(
        x,
        means,
        lower_errors,
        upper_errors,
    ):
        lower_value = mean - lower_error
        upper_value = mean + upper_error

        axis.vlines(
            position,
            lower_value,
            upper_value,
            linewidth=1.0,
        )

        cap_width = 0.08

        axis.hlines(
            lower_value,
            position - cap_width,
            position + cap_width,
            linewidth=1.0,
        )

        axis.hlines(
            upper_value,
            position - cap_width,
            position + cap_width,
            linewidth=1.0,
        )

    axis.set_xticks(
        x,
        scenarios,
    )

    axis.set_ylabel(
        "Selection Divergence Rate"
    )

    axis.set_ylim(
        0.0,
        1.0,
    )

    axis.set_title(
        "Mean-Field vs Baseline Node Selection"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    return _save_figure(
        figure,
        source,
    )


def plot_decision_audit(
    decision_comparison: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot decision-audit identity metrics by workload scenario."""
    source = Path(output_path)
    required = [
        "scenario",
        "metric",
        "comparison_mean",
    ]
    _require_columns(
        decision_comparison,
        required,
        source,
    )

    metrics = [
        "candidate_set_identity_rate",
        "selected_node_identity_rate",
    ]

    frame = _select_profile(decision_comparison)
    frame = frame.loc[
        frame["metric"].isin(metrics)
    ].copy()

    scenarios = list(frame["scenario"].drop_duplicates())
    x = np.arange(len(scenarios))
    width = 0.35

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    for offset, metric in enumerate(metrics):
        values = []
        for scenario in scenarios:
            row = frame.loc[
                (frame["scenario"] == scenario)
                & (frame["metric"] == metric)
            ]
            values.append(
                float(row["comparison_mean"].iloc[0])
                if not row.empty
                else np.nan
            )

        axis.bar(
            x + (offset - 0.5) * width,
            values,
            width,
            label=metric.replace("_", " ").title(),
        )

    axis.set_xticks(x, scenarios)
    axis.set_ylabel("Identity Rate")
    axis.set_ylim(0.0, 1.0)
    axis.set_title("Mean-Field Decision Identity Audit")
    axis.legend()
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    return _save_figure(
        figure,
        source,
    )


def plot_equilibrium_diagnostics(
    diagnostics: pd.DataFrame,
    output_path: str | Path,
) -> Path:
    """Plot equilibrium build time and iteration count by profile."""
    source = Path(output_path)
    required = [
        "profile",
        "iterations",
        "equilibrium_build_seconds",
    ]
    _require_columns(
        diagnostics,
        required,
        source,
    )

    frame = diagnostics.sort_values("profile").copy()
    x = np.arange(len(frame))

    figure, axis = plt.subplots(
        figsize=(10, 5)
    )

    axis.bar(
        x,
        frame["equilibrium_build_seconds"].to_numpy(dtype=float),
    )
    axis.set_xticks(
        x,
        frame["profile"],
        rotation=20,
        ha="right",
    )
    axis.set_ylabel("Build Time (seconds)")
    axis.set_title("Mean-Field Equilibrium Build Cost")
    axis.grid(
        axis="y",
        alpha=0.25,
    )

    for position, (_, row) in enumerate(frame.iterrows()):
        axis.text(
            position,
            float(row["equilibrium_build_seconds"]),
            f"{int(row['iterations'])} iterations",
            ha="center",
            va="bottom",
        )

    return _save_figure(
        figure,
        source,
    )


def generate_utility_sensitivity_figures(
    results_directory: str | Path,
) -> list[Path]:
    """Generate all figures from saved utility sensitivity CSV files."""
    results_path = Path(results_directory)
    paired_path = (
        results_path
        / "utility_sensitivity_paired_comparison.csv"
    )
    diagnostics_path = (
        results_path / "equilibrium_diagnostics.csv"
    )

    selection_path = results_path / "selection_comparison.csv"
    decision_path = results_path / "decision_comparison.csv"

    for path in (
        paired_path,
        diagnostics_path,
        selection_path,
        decision_path,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required result file was not found: {path}"
            )

    paired = pd.read_csv(paired_path)
    diagnostics = pd.read_csv(diagnostics_path)
    selection = pd.read_csv(selection_path)
    decision = pd.read_csv(decision_path)

    if selection.empty:
        raise ValueError(
            f"Selection comparison file is empty: {selection_path}"
        )
    if decision.empty:
        raise ValueError(
            f"Decision comparison file is empty: {decision_path}"
        )

    output_directory = results_path / "figures"

    return [
        plot_policy_performance(
            paired,
            output_directory / "policy_performance.png",
        ),
        plot_utility_weight_sensitivity(
            paired,
            output_directory / "utility_weight_sensitivity.png",
        ),
        plot_selection_divergence(
            selection,
            output_directory / "selection_divergence.png",
        ),
        plot_decision_audit(
            decision,
            output_directory / "decision_audit.png",
        ),
        plot_equilibrium_diagnostics(
            diagnostics,
            output_directory / "equilibrium_build_cost.png",
        ),
    ]


def main() -> None:
    """Generate figures from existing experiment outputs."""
    parser = argparse.ArgumentParser(
        description="Generate figures from saved utility sensitivity results."
    )
    parser.add_argument(
        "results_directory",
        nargs="?",
        default="outputs/utility_sensitivity",
        help="Directory containing saved experiment CSV files.",
    )

    args = parser.parse_args()

    figures = generate_utility_sensitivity_figures(
        args.results_directory
    )

    print("Visualization completed.")
    for figure in figures:
        print(f"Figure saved to: {figure}")


if __name__ == "__main__":
    main()



def generate_robustness_figures(
    results_directory: str | Path,
) -> list[str]:
    """Generate figures for robustness experiment outputs."""
    results_path = Path(results_directory)
    figures_path = results_path / "figures"
    figures_path.mkdir(parents=True, exist_ok=True)

    raw_path = results_path / "robustness_raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Robustness raw results were not found: {raw_path}"
        )

    raw = pd.read_csv(raw_path)

    _require_columns(
        raw,
        [
            "scenario",
            "policy_name",
            "response_time_mean",
            "success_ratio",
            "rejected_tasks",
            "resource_utilization",
        ],
        raw_path,
    )

    generated: list[str] = []

    plot_definitions = [
        (
            "response_time_mean",
            "Mean Response Time",
            "Response Time",
        ),
        (
            "success_ratio",
            "Success Ratio",
            "Success Ratio",
        ),
        (
            "rejected_tasks",
            "Rejected Tasks",
            "Rejected Tasks",
        ),
        (
            "resource_utilization",
            "Resource Utilization",
            "Resource Utilization",
        ),
    ]

    for metric, title, ylabel in plot_definitions:
        figure, axis = plt.subplots(
            figsize=(10, 6)
        )

        for policy_name, frame in raw.groupby(
            "policy_name",
            sort=True,
        ):
            grouped = (
                frame.groupby(
                    "scenario",
                    sort=False,
                )[metric]
                .mean()
            )

            scenarios = grouped.index.to_numpy(
                dtype=object
            )

            values = grouped.to_numpy(
                dtype=float
            )

            axis.plot(
                scenarios,
                values,
                marker="o",
                label=POLICY_LABELS.get(
                    policy_name,
                    policy_name,
                ),
            )

        axis.set_title(
            f"Robustness: {title}"
        )
        axis.set_xlabel("Scenario")
        axis.set_ylabel(ylabel)

        axis.tick_params(
            axis="x",
            rotation=25,
        )

        axis.legend()

        axis.grid(
            axis="y",
            alpha=0.25,
        )

        figure.tight_layout()

        path = (
            figures_path
            / f"robustness_{metric}.png"
        )

        figure.savefig(
            path,
            dpi=180,
            bbox_inches="tight",
        )

        plt.close(figure)

        generated.append(str(path))

    return generated
