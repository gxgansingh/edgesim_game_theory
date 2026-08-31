"""Ablation experiments for the Priority-aware Mean-Field formulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import MeanFieldPolicy
from ..config import SimulationConfig
from ..models.mean_field_model import ABLATION_VARIANTS
from .runner import build_mean_field_policy
from .statistics import METRIC_COLUMNS, benjamini_hochberg


ABLATION_ORDER = (
    "full",
    "no_priority",
    "no_priority_reward",
    "no_latency",
    "no_resource",
    "no_queue",
    "no_energy",
)

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


@dataclass
class AblationExperimentResult:
    """Store ablation experiment outputs."""

    raw_results: pd.DataFrame
    summary: pd.DataFrame
    comparison: pd.DataFrame
    equilibrium_diagnostics: pd.DataFrame


def _validate_variants(
    variants: tuple[str, ...],
) -> None:
    """Validate requested ablation variants."""
    invalid = [
        variant
        for variant in variants
        if variant not in ABLATION_VARIANTS
    ]

    if invalid:
        valid = ", ".join(ABLATION_ORDER)
        raise ValueError(
            "Unsupported ablation variant(s): "
            f"{', '.join(invalid)}. "
            f"Expected one of: {valid}."
        )

    if "full" not in variants:
        raise ValueError(
            "The ablation experiment requires the 'full' variant "
            "as the reference formulation."
        )


def _validate_seeds(
    seeds: tuple[int, ...],
) -> None:
    """Validate experiment seeds."""
    if not seeds:
        raise ValueError(
            "At least one experiment seed is required."
        )

    if len(set(seeds)) != len(seeds):
        raise ValueError(
            "Experiment seeds must be unique."
        )


def _build_policy(
    config: SimulationConfig,
    variant: str,
) -> tuple[MeanFieldPolicy, dict]:
    """Build one Mean-Field policy for an ablation variant."""
    return build_mean_field_policy(
        config=config,
        ablation_variant=variant,
    )


def _run_variant(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    scenario: str,
    variant: str,
    policy: MeanFieldPolicy,
) -> list[dict]:
    """Run one ablation variant for one workload scenario."""
    from dataclasses import replace

    scenario_config = replace(
        config,
        tasks_per_step=_resolve_scenario_tasks(
            config=config,
            scenario=scenario,
        ),
    )

    rows: list[dict] = []

    for seed in seeds:
        seed_config = replace(
            scenario_config,
            seed=seed,
        )

        result = run_policy_experiment(
            config=seed_config,
            policy_name="priority_aware_mean_field",
            policy=policy,
        )

        row = {
            "scenario": scenario,
            "variant": variant,
            "seed": int(seed),
            "policy_name": result.policy_name,
        }

        row.update(result.metrics)
        rows.append(row)

    return rows


def _resolve_scenario_tasks(
    config: SimulationConfig,
    scenario: str,
) -> int:
    """Resolve the configured workload intensity for a scenario."""
    scenario_tasks = {
        "default": 3,
        "moderate_congestion": 8,
        "high_congestion": 15,
    }

    if scenario not in scenario_tasks:
        valid = ", ".join(sorted(scenario_tasks))
        raise ValueError(
            f"Unknown workload scenario '{scenario}'. "
            f"Valid scenarios: {valid}."
        )

    return scenario_tasks[scenario]


def _confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a two-sided Student-t confidence interval."""
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) == 0:
        return np.nan, np.nan

    mean = float(np.mean(values))

    if len(values) < 2:
        return mean, mean

    standard_error = float(
        stats.sem(values)
    )

    alpha = 1.0 - confidence

    critical = float(
        stats.t.ppf(
            1.0 - alpha / 2.0,
            df=len(values) - 1,
        )
    )

    margin = critical * standard_error

    return (
        mean - margin,
        mean + margin,
    )


def _relative_change(
    baseline_mean: float,
    comparison_mean: float,
) -> float:
    """Calculate percentage change relative to the full formulation."""
    if abs(baseline_mean) <= 1e-12:
        return np.nan

    return (
        (comparison_mean - baseline_mean)
        / abs(baseline_mean)
    ) * 100.0


def _effect_direction(
    metric: str,
    relative_change: float,
) -> str:
    """Classify whether a metric change is favorable or unfavorable."""
    if not np.isfinite(relative_change):
        return "indeterminate"

    direction = METRIC_DIRECTIONS[metric]

    if abs(relative_change) <= 1e-12:
        return "neutral"

    if direction == "higher":
        return (
            "improvement"
            if relative_change > 0
            else "regression"
        )

    return (
        "improvement"
        if relative_change < 0
        else "regression"
    )


def build_ablation_comparison(
    raw_results: pd.DataFrame,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Compare every ablation variant against the full formulation."""
    required = {
        "scenario",
        "variant",
        "seed",
        *METRIC_COLUMNS,
    }

    missing = required.difference(
        raw_results.columns
    )

    if missing:
        raise ValueError(
            "Ablation results are missing columns: "
            + ", ".join(sorted(missing))
        )

    full = raw_results.loc[
        raw_results["variant"] == "full"
    ].copy()

    ablations = raw_results.loc[
        raw_results["variant"] != "full"
    ].copy()

    if full.empty:
        raise ValueError(
            "Ablation comparison requires full-variant results."
        )

    rows: list[dict] = []

    for (
        scenario,
        variant,
    ), variant_frame in ablations.groupby(
        ["scenario", "variant"],
        sort=True,
    ):
        full_frame = full.loc[
            full["scenario"] == scenario
        ].copy()

        full_frame = full_frame.set_index(
            "seed"
        )
        variant_frame = variant_frame.set_index(
            "seed"
        )

        common_seeds = full_frame.index.intersection(
            variant_frame.index
        )

        if len(common_seeds) == 0:
            continue

        for metric in METRIC_COLUMNS:
            full_values = full_frame.loc[
                common_seeds,
                metric,
            ].to_numpy(dtype=float)

            variant_values = variant_frame.loc[
                common_seeds,
                metric,
            ].to_numpy(dtype=float)

            differences = (
                variant_values
                - full_values
            )

            finite = (
                np.isfinite(differences)
                & np.isfinite(full_values)
                & np.isfinite(variant_values)
            )

            full_values = full_values[finite]
            variant_values = variant_values[finite]
            differences = differences[finite]

            if len(differences) == 0:
                continue

            baseline_mean = float(
                np.mean(full_values)
            )

            comparison_mean = float(
                np.mean(variant_values)
            )

            mean_difference = float(
                np.mean(differences)
            )

            if len(differences) >= 2:
                difference_std = float(
                    np.std(
                        differences,
                        ddof=1,
                    )
                )

                if difference_std <= 1e-12:
                    p_value = (
                        1.0
                        if abs(mean_difference) <= 1e-12
                        else 0.0
                    )
                else:
                    p_value = float(
                        stats.ttest_1samp(
                            differences,
                            popmean=0.0,
                        ).pvalue
                    )
            else:
                difference_std = 0.0
                p_value = np.nan

            ci_low, ci_high = _confidence_interval(
                differences
            )

            relative_change = _relative_change(
                baseline_mean=baseline_mean,
                comparison_mean=comparison_mean,
            )

            rows.append(
                {
                    "scenario": scenario,
                    "variant": variant,
                    "metric": metric,
                    "n": int(len(differences)),
                    "baseline_variant": "full",
                    "baseline_mean": baseline_mean,
                    "comparison_mean": comparison_mean,
                    "mean_difference": mean_difference,
                    "relative_change_percent": relative_change,
                    "std_difference": difference_std,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "p_value": p_value,
                    "direction": _effect_direction(
                        metric,
                        relative_change,
                    ),
                }
            )

    comparison = pd.DataFrame(rows)

    if comparison.empty:
        return comparison

    valid_p_values = comparison["p_value"].notna()

    adjusted = np.full(
        len(comparison),
        np.nan,
        dtype=float,
    )

    significant = np.zeros(
        len(comparison),
        dtype=bool,
    )

    if valid_p_values.any():
        adjusted_values, significant_values = (
            benjamini_hochberg(
                comparison.loc[
                    valid_p_values,
                    "p_value",
                ].tolist(),
                alpha=alpha,
            )
        )

        adjusted[
            valid_p_values.to_numpy()
        ] = adjusted_values

        significant[
            valid_p_values.to_numpy()
        ] = significant_values

    comparison["adjusted_p_value"] = adjusted
    comparison["significant"] = significant

    comparison["effect"] = np.where(
        comparison["significant"]
        & (comparison["direction"] == "improvement"),
        "significant_improvement",
        np.where(
            comparison["significant"]
            & (comparison["direction"] == "regression"),
            "significant_regression",
            np.where(
                comparison["direction"] == "improvement",
                "directional_improvement",
                np.where(
                    comparison["direction"] == "regression",
                    "directional_regression",
                    comparison["direction"],
                ),
            ),
        ),
    )

    return comparison


def build_ablation_summary(
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize ablation effects by scenario and variant."""
    if comparison.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    for (
        scenario,
        variant,
    ), frame in comparison.groupby(
        ["scenario", "variant"],
        sort=True,
    ):
        significant_improvements = int(
            (
                frame["effect"]
                == "significant_improvement"
            ).sum()
        )

        significant_regressions = int(
            (
                frame["effect"]
                == "significant_regression"
            ).sum()
        )

        directional_improvements = int(
            frame["effect"].isin(
                [
                    "significant_improvement",
                    "directional_improvement",
                ]
            ).sum()
        )

        directional_regressions = int(
            frame["effect"].isin(
                [
                    "significant_regression",
                    "directional_regression",
                ]
            ).sum()
        )

        if significant_improvements > significant_regressions:
            classification = "improvement"
        elif significant_regressions > significant_improvements:
            classification = "regression_risk"
        elif (
            significant_improvements > 0
            and significant_regressions > 0
        ):
            classification = "mixed"
        elif (
            directional_improvements
            > directional_regressions
        ):
            classification = "directional_net_improvement"
        elif (
            directional_regressions
            > directional_improvements
        ):
            classification = "directional_net_regression"
        else:
            classification = "neutral"

        rows.append(
            {
                "scenario": scenario,
                "variant": variant,
                "significant_improvements": (
                    significant_improvements
                ),
                "significant_regressions": (
                    significant_regressions
                ),
                "directional_improvements": (
                    directional_improvements
                ),
                "directional_regressions": (
                    directional_regressions
                ),
                "net_significant_effects": (
                    significant_improvements
                    - significant_regressions
                ),
                "overall_classification": classification,
            }
        )

    return pd.DataFrame(rows)


def _save_heatmap(
    comparison: pd.DataFrame,
    output_path: Path,
) -> Path:
    """Save a heatmap of relative ablation effects."""
    frame = comparison.copy()

    frame = frame.loc[
        frame["variant"].isin(
            [
                variant
                for variant in ABLATION_ORDER
                if variant != "full"
            ]
        )
    ]

    if frame.empty:
        return output_path

    frame["label"] = (
        frame["variant"]
        + " | "
        + frame["scenario"]
    )

    pivot = frame.pivot_table(
        index="label",
        columns="metric",
        values="relative_change_percent",
        aggfunc="mean",
    )

    figure, axis = plt.subplots(
        figsize=(
            max(12, len(pivot.columns) * 1.4),
            max(6, len(pivot.index) * 0.45),
        )
    )

    image = axis.imshow(
        pivot.to_numpy(dtype=float),
        aspect="auto",
    )

    axis.set_xticks(
        np.arange(len(pivot.columns)),
        pivot.columns,
        rotation=45,
        ha="right",
    )

    axis.set_yticks(
        np.arange(len(pivot.index)),
        pivot.index,
    )

    axis.set_xlabel("Metric")
    axis.set_ylabel("Ablation | Scenario")
    axis.set_title(
        "Ablation Relative Change vs Full Formulation"
    )

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
                    f"{value:.2f}%",
                    ha="center",
                    va="center",
                )

    figure.colorbar(
        image,
        ax=axis,
        label="Relative Change (%)",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)

    return output_path


def build_ablation_report(
    comparison: pd.DataFrame,
    summary: pd.DataFrame,
) -> str:
    """Build a concise Markdown ablation report."""
    lines = [
        "# Ablation Analysis Report",
        "",
        "## 1. Experimental Overview",
        "",
        "The ablation experiment compares each reduced "
        "Mean-Field formulation against the full formulation.",
        "",
        f"- Variants evaluated: "
        f"{comparison['variant'].nunique() if not comparison.empty else 0}",
        f"- Scenarios evaluated: "
        f"{comparison['scenario'].nunique() if not comparison.empty else 0}",
        f"- Metrics evaluated: "
        f"{comparison['metric'].nunique() if not comparison.empty else 0}",
        "- Reference formulation: `full`",
        "- Significance correction: Benjamini-Hochberg FDR",
        "",
    ]

    if comparison.empty:
        lines.extend(
            [
                "No comparison results were generated.",
                "",
            ]
        )
        return "\n".join(lines)

    significant = comparison.loc[
        comparison["significant"]
    ]

    lines.extend(
        [
            "## 2. Significant Effects",
            "",
            f"- FDR-significant effects: {len(significant)}",
            "",
        ]
    )

    if significant.empty:
        lines.extend(
            [
                "No statistically significant effects were detected "
                "after FDR correction.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "| Variant | Scenario | Metric | Relative Change | Adjusted p-value | Effect |",
                "| --- | --- | --- | ---: | ---: | --- |",
            ]
        )

        for _, row in significant.iterrows():
            lines.append(
                "| "
                f"{row['variant']} | "
                f"{row['scenario']} | "
                f"{row['metric']} | "
                f"{row['relative_change_percent']:.2f}% | "
                f"{row['adjusted_p_value']:.4f} | "
                f"{row['effect']} |"
            )

        lines.append("")

    lines.extend(
        [
            "## 3. Scenario-Level Summary",
            "",
            "| Variant | Scenario | Significant Improvements | Significant Regressions | Classification |",
            "| --- | --- | ---: | ---: | --- |",
        ]
    )

    for _, row in summary.iterrows():
        lines.append(
            "| "
            f"{row['variant']} | "
            f"{row['scenario']} | "
            f"{row['significant_improvements']} | "
            f"{row['significant_regressions']} | "
            f"{row['overall_classification']} |"
        )

    lines.extend(
        [
            "",
            "## 4. Interpretation",
            "",
            "The ablation results indicate which components of the "
            "Priority-aware Mean-Field objective materially affect "
            "system performance relative to the full formulation.",
            "",
            "Statistical conclusions use FDR-adjusted p-values. "
            "Directional changes without statistical significance "
            "should not be interpreted as confirmed effects.",
            "",
        ]
    )

    return "\n".join(lines)


def save_ablation_outputs(
    result: AblationExperimentResult,
    output_directory: str | Path,
) -> list[Path]:
    """Save ablation CSV, figure, and report outputs."""
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

    paths: list[Path] = []

    raw_file = raw_path / "ablation_raw.csv"
    result.raw_results.to_csv(
        raw_file,
        index=False,
    )
    paths.append(raw_file)

    summary_file = (
        aggregated_path
        / "ablation_summary.csv"
    )
    result.summary.to_csv(
        summary_file,
        index=False,
    )
    paths.append(summary_file)

    comparison_file = (
        aggregated_path
        / "ablation_comparison.csv"
    )
    result.comparison.to_csv(
        comparison_file,
        index=False,
    )
    paths.append(comparison_file)

    diagnostics_file = (
        aggregated_path
        / "ablation_equilibrium_diagnostics.csv"
    )
    result.equilibrium_diagnostics.to_csv(
        diagnostics_file,
        index=False,
    )
    paths.append(diagnostics_file)

    figure_file = (
        figures_path
        / "ablation_effect_heatmap.png"
    )

    paths.append(
        _save_heatmap(
            result.comparison,
            figure_file,
        )
    )

    report_file = (
        output_path
        / "ablation_report.md"
    )

    report_file.write_text(
        build_ablation_report(
            result.comparison,
            result.summary,
        ),
        encoding="utf-8",
    )

    paths.append(report_file)

    return paths


def run_ablation_experiment(
    config: SimulationConfig,
    seeds: list[int] | tuple[int, ...],
    scenarios: list[str] | tuple[str, ...] | None = None,
    variants: list[str] | tuple[str, ...] | None = None,
    output_directory: str | Path | None = None,
) -> AblationExperimentResult:
    """Run the complete Mean-Field ablation experiment."""
    normalized_seeds = tuple(
        int(seed)
        for seed in seeds
    )

    _validate_seeds(
        normalized_seeds
    )

    normalized_variants = (
        tuple(variants)
        if variants is not None
        else ABLATION_ORDER
    )

    _validate_variants(
        normalized_variants
    )

    normalized_scenarios = (
        tuple(scenarios)
        if scenarios is not None
        else tuple(config.workload_scenarios)
    )

    if not normalized_scenarios:
        raise ValueError(
            "At least one workload scenario is required."
        )

    rows: list[dict] = []
    diagnostic_rows: list[dict] = []

    for variant in normalized_variants:
        policy, diagnostics = _build_policy(
            config=config,
            variant=variant,
        )

        diagnostic_row = {
            "variant": variant,
            **diagnostics,
        }

        diagnostic_rows.append(
            diagnostic_row
        )

        for scenario in normalized_scenarios:
            rows.extend(
                _run_variant(
                    config=config,
                    seeds=normalized_seeds,
                    scenario=scenario,
                    variant=variant,
                    policy=policy,
                )
            )

    raw_results = pd.DataFrame(
        rows
    ).sort_values(
        [
            "scenario",
            "variant",
            "seed",
        ]
    ).reset_index(drop=True)

    comparison = build_ablation_comparison(
        raw_results
    )

    summary = build_ablation_summary(
        comparison
    )

    result = AblationExperimentResult(
        raw_results=raw_results,
        summary=summary,
        comparison=comparison,
        equilibrium_diagnostics=pd.DataFrame(
            diagnostic_rows
        ),
    )

    if output_directory is not None:
        save_ablation_outputs(
            result=result,
            output_directory=output_directory,
        )

    return result