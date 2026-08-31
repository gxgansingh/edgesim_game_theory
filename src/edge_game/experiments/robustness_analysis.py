"""Post-process robustness experiment results into research-ready findings."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.edge_game.experiments.statistics import benjamini_hochberg


METRIC_DIRECTIONS = {
    "utility_mean": "higher",
    "response_time_mean": "lower",
    "throughput": "higher",
    "success_ratio": "higher",
    "rejected_tasks": "lower",
    "resource_utilization": "higher",
    "load_variance": "lower",
    "jains_fairness_index": "higher",
    "average_queue_length": "lower",
    "priority_success_ratio": "higher",
}


def _effect_label(
    metric: str,
    delta: float,
    p_value: float,
    alpha: float,
) -> str:
    """Classify an effect using metric direction and statistical significance."""

    if not np.isfinite(delta):
        return "not_estimable"

    if not np.isfinite(p_value) or p_value >= alpha:
        return "not_significant"

    direction = METRIC_DIRECTIONS[metric]
    improved = delta > 0 if direction == "higher" else delta < 0

    if improved:
        return "significant_improvement"

    return "significant_regression"


def build_robustness_analysis(
    paired_comparison: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Build a metric-by-scenario interpretation table using FDR correction."""

    required = {
        "scenario",
        "metric",
        "mean_difference",
        "relative_change_percent",
        "p_value",
    }

    missing = required.difference(paired_comparison.columns)

    if missing:
        raise ValueError(
            "Robustness paired comparison is missing columns: "
            + ", ".join(sorted(missing))
        )

    frame = paired_comparison.loc[
        paired_comparison["metric"].isin(METRIC_DIRECTIONS)
    ].copy()

    frame["metric_direction"] = frame["metric"].map(METRIC_DIRECTIONS)

    finite_mask = (
        frame["p_value"].notna()
        & np.isfinite(frame["p_value"])
    )

    finite_p_values = (
        frame.loc[finite_mask, "p_value"]
        .astype(float)
        .tolist()
    )

    adjusted_p_values, fdr_significant = benjamini_hochberg(
        finite_p_values,
        alpha=alpha,
    )

    frame["adjusted_p_value"] = np.nan

    frame.loc[
        finite_mask,
        "adjusted_p_value",
    ] = adjusted_p_values

    frame["raw_significant"] = (
        frame["p_value"].notna()
        & (frame["p_value"] < alpha)
    )

    frame["significant"] = False

    frame.loc[
        finite_mask,
        "significant",
    ] = fdr_significant

    frame["effect"] = [
        _effect_label(
            metric,
            delta,
            adjusted_p_value,
            alpha,
        )
        for metric, delta, adjusted_p_value in zip(
            frame["metric"],
            frame["mean_difference"],
            frame["adjusted_p_value"],
        )
    ]

    frame["alpha"] = float(alpha)

    frame["favorable"] = (
        frame["effect"] == "significant_improvement"
    )

    frame["unfavorable"] = (
        frame["effect"] == "significant_regression"
    )

    columns = [
        "scenario",
        "metric",
        "metric_direction",
        "mean_difference",
        "relative_change_percent",
        "p_value",
        "adjusted_p_value",
        "alpha",
        "raw_significant",
        "significant",
        "favorable",
        "unfavorable",
        "effect",
    ]

    return (
        frame[columns]
        .sort_values(["scenario", "metric"])
        .reset_index(drop=True)
    )


def summarize_robustness_tradeoffs(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize FDR-significant improvements and regressions per scenario."""

    rows: list[dict[str, object]] = []

    for scenario, frame in analysis.groupby(
        "scenario",
        sort=True,
    ):
        improvements = frame.loc[
            frame["favorable"]
        ]

        regressions = frame.loc[
            frame["unfavorable"]
        ]

        rows.append(
            {
                "scenario": scenario,
                "significant_improvements": int(
                    len(improvements)
                ),
                "significant_regressions": int(
                    len(regressions)
                ),
                "net_significant_effects": int(
                    len(improvements)
                    - len(regressions)
                ),
                "improved_metrics": ", ".join(
                    improvements["metric"].tolist()
                ),
                "regressed_metrics": ", ".join(
                    regressions["metric"].tolist()
                ),
            }
        )

    return pd.DataFrame(rows)


def _classify_robustness(
    improvements: int,
    regressions: int,
) -> str:
    """Classify a scenario using only FDR-significant effects."""

    if improvements > 0 and regressions == 0:
        return "improvement"

    if improvements == 0 and regressions > 0:
        return "regression_risk"

    if improvements > 0 and regressions > 0:
        return "mixed"

    return "neutral"


def _strongest_metric(
    frame: pd.DataFrame,
    favorable: bool,
) -> str:
    """Return the metric with the largest significant relative effect."""

    if favorable:
        selected = frame.loc[
            frame["effect"] == "significant_improvement"
        ]
    else:
        selected = frame.loc[
            frame["effect"] == "significant_regression"
        ]

    if selected.empty:
        return ""

    selected = selected.copy()

    selected["_effect_magnitude"] = (
        selected["relative_change_percent"]
        .abs()
    )

    selected = selected.sort_values(
        "_effect_magnitude",
        ascending=False,
    )

    return str(
        selected.iloc[0]["metric"]
    )


def build_robustness_interpretation(
    analysis: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a scenario-level interpretation from FDR-corrected effects.

    Only rows marked as FDR-significant through the ``significant``
    column are used to determine improvements, regressions, and
    scenario classifications.
    """

    required = {
        "scenario",
        "metric",
        "relative_change_percent",
        "significant",
        "favorable",
        "unfavorable",
        "effect",
    }

    missing = required.difference(analysis.columns)

    if missing:
        raise ValueError(
            "Robustness analysis is missing columns: "
            + ", ".join(sorted(missing))
        )

    rows: list[dict[str, object]] = []

    for scenario, frame in analysis.groupby(
        "scenario",
        sort=True,
    ):
        significant = frame.loc[
            frame["significant"].astype(bool)
        ]

        improvements = significant.loc[
            significant["favorable"].astype(bool)
        ]

        regressions = significant.loc[
            significant["unfavorable"].astype(bool)
        ]

        improvement_count = int(
            len(improvements)
        )

        regression_count = int(
            len(regressions)
        )

        net_effects = (
            improvement_count
            - regression_count
        )

        rows.append(
            {
                "scenario": scenario,
                "significant_improvements": improvement_count,
                "significant_regressions": regression_count,
                "net_significant_effects": net_effects,
                "strongest_improvement_metric": (
                    _strongest_metric(
                        frame,
                        favorable=True,
                    )
                ),
                "strongest_regression_metric": (
                    _strongest_metric(
                        frame,
                        favorable=False,
                    )
                ),
                "overall_classification": (
                    _classify_robustness(
                        improvement_count,
                        regression_count,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "scenario",
            "significant_improvements",
            "significant_regressions",
            "net_significant_effects",
            "strongest_improvement_metric",
            "strongest_regression_metric",
            "overall_classification",
        ],
    )


def generate_robustness_analysis_outputs(
    results_directory: str | Path,
    alpha: float = 0.05,
) -> tuple[str, str, str]:
    """Create FDR-corrected analysis CSVs and robustness figures."""

    results_path = Path(results_directory)

    paired_path = (
        results_path
        / "robustness_paired_comparison.csv"
    )

    if not paired_path.exists():
        raise FileNotFoundError(
            f"Missing robustness results: {paired_path}"
        )

    paired = pd.read_csv(paired_path)

    analysis = build_robustness_analysis(
        paired,
        alpha=alpha,
    )

    tradeoffs = summarize_robustness_tradeoffs(
        analysis
    )

    interpretation = build_robustness_interpretation(
        analysis
    )

    analysis_path = (
        results_path
        / "robustness_effect_analysis.csv"
    )

    tradeoff_path = (
        results_path
        / "robustness_tradeoff_summary.csv"
    )

    interpretation_path = (
        results_path
        / "robustness_interpretation.csv"
    )

    analysis.to_csv(
        analysis_path,
        index=False,
    )

    tradeoffs.to_csv(
        tradeoff_path,
        index=False,
    )

    interpretation.to_csv(
        interpretation_path,
        index=False,
    )

    figures_path = (
        results_path
        / "figures"
    )

    figures_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    scenarios = sorted(
        analysis["scenario"].unique()
    )

    metrics = [
        metric
        for metric in METRIC_DIRECTIONS
        if metric in analysis["metric"].unique()
    ]

    matrix = np.zeros(
        (
            len(scenarios),
            len(metrics),
        )
    )

    effect_values = {
        "significant_improvement": 1.0,
        "not_significant": 0.0,
        "significant_regression": -1.0,
        "not_estimable": 0.0,
    }

    for row_index, scenario in enumerate(
        scenarios
    ):
        for column_index, metric in enumerate(
            metrics
        ):
            match = analysis.loc[
                (analysis["scenario"] == scenario)
                & (analysis["metric"] == metric),
                "effect",
            ]

            if not match.empty:
                matrix[
                    row_index,
                    column_index,
                ] = effect_values.get(
                    match.iloc[0],
                    0.0,
                )

    figure, axis = plt.subplots(
        figsize=(
            max(
                12,
                len(metrics) * 1.25,
            ),
            max(
                4.5,
                len(scenarios) * 1.0,
            ),
        )
    )

    image = axis.imshow(
        matrix,
        vmin=-1,
        vmax=1,
        aspect="auto",
    )

    axis.set_xticks(
        range(len(metrics))
    )

    axis.set_xticklabels(
        metrics,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        range(len(scenarios))
    )

    axis.set_yticklabels(
        scenarios
    )

    axis.set_title(
        "Robustness Effects: Mean-Field vs Baseline"
    )

    axis.set_xlabel("Metric")
    axis.set_ylabel("Scenario")

    for row_index in range(
        len(scenarios)
    ):
        for column_index in range(
            len(metrics)
        ):
            value = matrix[
                row_index,
                column_index,
            ]

            label = (
                "+"
                if value > 0
                else "-"
                if value < 0
                else "·"
            )

            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
            )

    figure.colorbar(
        image,
        ax=axis,
        ticks=[-1, 0, 1],
        label="Significant effect",
    )

    figure.tight_layout()

    figure_path = (
        figures_path
        / "robustness_effect_heatmap.png"
    )

    figure.savefig(
        figure_path,
        dpi=180,
    )

    plt.close(figure)

    return (
        str(analysis_path),
        str(tradeoff_path),
        str(figure_path),
    )