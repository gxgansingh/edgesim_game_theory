"""Tests for saved-result visualization utilities."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.edge_game.experiments.visualization import (
    generate_utility_sensitivity_figures,
    plot_decision_audit,
    plot_equilibrium_diagnostics,
    plot_policy_performance,
    plot_selection_divergence,
    plot_utility_weight_sensitivity,
)


def _paired_frame() -> pd.DataFrame:
    """Create a deterministic paired-comparison frame."""
    rows = []
    for profile in ("balanced", "priority_latency"):
        for scenario in ("default", "high_congestion"):
            for metric, value in (
                ("utility_mean", 0.1),
                ("response_time_mean", -0.2),
                ("throughput", 1.0),
                ("success_ratio", 0.01),
                ("resource_utilization", 0.02),
                ("load_variance", -0.01),
                ("jains_fairness_index", 0.01),
                ("average_queue_length", -0.1),
                ("priority_success_ratio", 0.02),
            ):
                rows.append(
                    {
                        "profile": profile,
                        "scenario": scenario,
                        "metric": metric,
                        "mean_difference": value,
                    }
                )
    return pd.DataFrame(rows)


def test_policy_performance_figure_is_created(tmp_path: Path) -> None:
    """Verify policy performance visualization is written."""
    output = tmp_path / "performance.png"
    result = plot_policy_performance(
        _paired_frame(),
        output,
    )
    assert result == output
    assert output.exists()
    assert output.stat().st_size > 0


def test_utility_weight_figure_is_created(tmp_path: Path) -> None:
    """Verify utility profile sensitivity visualization is written."""
    output = tmp_path / "utility.png"
    result = plot_utility_weight_sensitivity(
        _paired_frame(),
        output,
    )
    assert result == output
    assert output.exists()


def test_selection_divergence_figure_is_created(tmp_path: Path) -> None:
    """Verify selection divergence visualization is written."""
    frame = pd.DataFrame(
        [
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "selection_divergence_rate",
                "comparison_mean": 0.5,
                "ci95_low": 0.4,
                "ci95_high": 0.6,
            },
            {
                "profile": "balanced",
                "scenario": "high_congestion",
                "metric": "selection_divergence_rate",
                "comparison_mean": 0.8,
                "ci95_low": 0.7,
                "ci95_high": 0.9,
            },
        ]
    )
    output = tmp_path / "selection.png"
    result = plot_selection_divergence(frame, output)
    assert result == output
    assert output.exists()


def test_decision_audit_figure_is_created(tmp_path: Path) -> None:
    """Verify decision identity visualization is written."""
    frame = pd.DataFrame(
        [
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "candidate_set_identity_rate",
                "comparison_mean": 0.6,
            },
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "selected_node_identity_rate",
                "comparison_mean": 0.5,
            },
            {
                "profile": "balanced",
                "scenario": "high_congestion",
                "metric": "candidate_set_identity_rate",
                "comparison_mean": 0.2,
            },
            {
                "profile": "balanced",
                "scenario": "high_congestion",
                "metric": "selected_node_identity_rate",
                "comparison_mean": 0.1,
            },
        ]
    )
    output = tmp_path / "decision.png"
    result = plot_decision_audit(frame, output)
    assert result == output
    assert output.exists()


def test_equilibrium_diagnostics_figure_is_created(tmp_path: Path) -> None:
    """Verify equilibrium build-cost visualization is written."""
    frame = pd.DataFrame(
        [
            {
                "profile": "balanced",
                "iterations": 10,
                "equilibrium_build_seconds": 2.5,
            },
            {
                "profile": "priority_latency",
                "iterations": 12,
                "equilibrium_build_seconds": 3.0,
            },
        ]
    )
    output = tmp_path / "diagnostics.png"
    result = plot_equilibrium_diagnostics(frame, output)
    assert result == output
    assert output.exists()


def test_generation_reads_saved_csv_files(tmp_path: Path) -> None:
    """Verify the end-to-end generator uses saved CSV results."""
    result_dir = tmp_path / "utility_sensitivity"
    result_dir.mkdir()

    _paired_frame().to_csv(
        result_dir / "utility_sensitivity_paired_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "profile": "balanced",
                "iterations": 10,
                "equilibrium_build_seconds": 2.5,
            },
        ]
    ).to_csv(
        result_dir / "equilibrium_diagnostics.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "selection_divergence_rate",
                "comparison_mean": 0.5,
                "ci95_low": 0.4,
                "ci95_high": 0.6,
            },
        ]
    ).to_csv(
        result_dir / "selection_comparison.csv",
        index=False,
    )

    pd.DataFrame(
        [
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "candidate_set_identity_rate",
                "comparison_mean": 0.6,
            },
            {
                "profile": "balanced",
                "scenario": "default",
                "metric": "selected_node_identity_rate",
                "comparison_mean": 0.5,
            },
        ]
    ).to_csv(
        result_dir / "decision_comparison.csv",
        index=False,
    )

    figures = generate_utility_sensitivity_figures(result_dir)

    assert len(figures) == 5
    assert all(path.exists() for path in figures)
    assert (result_dir / "figures" / "selection_divergence.png").exists()
    assert (result_dir / "figures" / "decision_audit.png").exists()
