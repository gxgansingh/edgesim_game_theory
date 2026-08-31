from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RAW_COLUMNS = {
    "scenario",
    "scenario_description",
    "seed",
    "policy_name",
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
}

ROBUSTNESS_ANALYSIS_COLUMNS = {
    "scenario",
    "metric",
    "p_value",
    "adjusted_p_value",
    "raw_significant",
    "significant",
}

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

HIGHER_IS_BETTER = {
    "utility_mean": True,
    "response_time_mean": False,
    "throughput": True,
    "success_ratio": True,
    "rejected_tasks": False,
    "resource_utilization": True,
    "load_variance": False,
    "jains_fairness_index": True,
    "average_queue_length": False,
    "priority_success_ratio": True,
}


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    frame_name: str,
) -> None:
    required = set(required)
    missing = required.difference(frame.columns)

    if missing:
        raise ValueError(
            f"{frame_name} is missing required columns: "
            f"{', '.join(sorted(missing))}"
        )


def _relative_change(
    baseline: float,
    comparison: float,
) -> float:
    if pd.isna(baseline) or pd.isna(comparison):
        return float("nan")

    if abs(float(baseline)) <= 1e-12:
        return float("nan")

    return (
        (float(comparison) - float(baseline))
        / abs(float(baseline))
    ) * 100.0


def load_robustness_raw(
    output_directory: Path,
) -> pd.DataFrame:
    raw_path = output_directory / "robustness_raw.csv"

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Robustness raw results not found: {raw_path}"
        )

    frame = pd.read_csv(raw_path)

    _require_columns(
        frame,
        RAW_COLUMNS,
        "Robustness raw results",
    )

    return frame


def load_robustness_analysis(
    output_directory: Path,
) -> pd.DataFrame:
    """
    Load the statistical robustness analysis used to determine
    FDR-corrected significance.

    The diagnostic layer uses this file rather than recomputing
    statistical tests so that both reports share the same statistical
    decision.
    """

    analysis_path = (
        output_directory
        / "robustness_effect_analysis.csv"
    )

    if not analysis_path.exists():
        raise FileNotFoundError(
            "Robustness statistical analysis not found: "
            f"{analysis_path}. Run "
            "'python -m src.edge_game.experiments robustness-analysis' "
            "before running the diagnostic."
        )

    frame = pd.read_csv(analysis_path)

    _require_columns(
        frame,
        ROBUSTNESS_ANALYSIS_COLUMNS,
        "Robustness effect analysis",
    )

    return frame


def build_policy_pressure_summary(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a policy-level summary from the actual robustness metrics.

    The simulator records observable performance metrics rather than
    internal CPU, memory, bandwidth, queue, or energy pressure states.
    """

    _require_columns(
        raw,
        RAW_COLUMNS,
        "Robustness raw results",
    )

    records: list[dict] = []

    for (scenario, policy_name), group in raw.groupby(
        ["scenario", "policy_name"],
        sort=True,
    ):
        record = {
            "scenario": scenario,
            "policy_name": policy_name,
            "n": int(len(group)),
        }

        for metric in METRICS:
            values = pd.to_numeric(
                group[metric],
                errors="coerce",
            ).dropna()

            if values.empty:
                record[f"{metric}_mean"] = float("nan")
                record[f"{metric}_std"] = float("nan")
                record[f"{metric}_cv_percent"] = float("nan")
                continue

            mean_value = float(values.mean())

            std_value = (
                float(values.std(ddof=1))
                if len(values) > 1
                else 0.0
            )

            if abs(mean_value) > 1e-12:
                cv_percent = (
                    abs(std_value / mean_value) * 100.0
                )
            else:
                cv_percent = float("nan")

            record[f"{metric}_mean"] = mean_value
            record[f"{metric}_std"] = std_value
            record[f"{metric}_cv_percent"] = cv_percent

        records.append(record)

    return pd.DataFrame(records)


def build_pressure_differences(
    pressure_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare each non-baseline policy against the baseline policy.

    The function name is retained for compatibility with the existing
    diagnostic pipeline. The comparison operates on observed
    robustness metrics rather than unavailable pressure variables.
    """

    _require_columns(
        pressure_summary,
        {
            "scenario",
            "policy_name",
            "n",
        },
        "Policy summary",
    )

    records: list[dict] = []

    for scenario, scenario_frame in pressure_summary.groupby(
        "scenario",
        sort=True,
    ):
        policies = sorted(
            scenario_frame["policy_name"]
            .dropna()
            .unique()
        )

        if len(policies) < 2:
            continue

        baseline_policy = (
            "least_loaded_baseline"
            if "least_loaded_baseline" in policies
            else policies[0]
        )

        baseline_rows = scenario_frame[
            scenario_frame["policy_name"] == baseline_policy
        ]

        if baseline_rows.empty:
            continue

        baseline_row = baseline_rows.iloc[0]

        comparison_policies = [
            policy
            for policy in policies
            if policy != baseline_policy
        ]

        for comparison_policy in comparison_policies:
            comparison_rows = scenario_frame[
                scenario_frame["policy_name"]
                == comparison_policy
            ]

            if comparison_rows.empty:
                continue

            comparison_row = comparison_rows.iloc[0]

            for metric in METRICS:
                mean_column = f"{metric}_mean"

                baseline_value = float(
                    baseline_row[mean_column]
                )
                comparison_value = float(
                    comparison_row[mean_column]
                )

                difference = (
                    comparison_value
                    - baseline_value
                )

                relative_change = _relative_change(
                    baseline_value,
                    comparison_value,
                )

                records.append(
                    {
                        "scenario": scenario,
                        "baseline_policy": baseline_policy,
                        "comparison_policy": comparison_policy,
                        "metric": metric,
                        "baseline_mean": baseline_value,
                        "comparison_mean": comparison_value,
                        "difference": difference,
                        "relative_change_percent": (
                            relative_change
                        ),
                    }
                )

    return pd.DataFrame(records)


def _merge_statistical_significance(
    differences: pd.DataFrame,
    robustness_analysis: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Attach the existing statistical robustness results to the
    directional diagnostic.

    The statistical analysis contains scenario/metric-level results,
    while the diagnostic contains scenario/comparison-policy/metric
    records. The merge therefore uses scenario and metric.

    If no statistical analysis is supplied, the function preserves
    the directional diagnostic behavior and leaves significance fields
    unavailable.
    """

    result = differences.copy()

    if result.empty:
        for column in [
            "p_value",
            "adjusted_p_value",
            "raw_significant",
            "significant",
        ]:
            result[column] = pd.Series(dtype="float64")

        return result

    if robustness_analysis is None:
        result["p_value"] = float("nan")
        result["adjusted_p_value"] = float("nan")
        result["raw_significant"] = pd.NA
        result["significant"] = pd.NA
        return result

    analysis = robustness_analysis[
        [
            "scenario",
            "metric",
            "p_value",
            "adjusted_p_value",
            "raw_significant",
            "significant",
        ]
    ].copy()

    analysis = analysis.drop_duplicates(
        subset=["scenario", "metric"],
        keep="last",
    )

    result = result.merge(
        analysis,
        on=["scenario", "metric"],
        how="left",
        validate="many_to_one",
    )

    return result


def build_robustness_interpretation(
    differences: pd.DataFrame,
    robustness_analysis: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Classify observed policy changes using direction-aware metrics.

    When statistical robustness results are supplied, the output
    distinguishes between:

    - significant_improvement
    - directional_improvement
    - significant_regression
    - directional_regression
    - neutral
    - indeterminate

    Statistical significance is taken directly from the existing
    robustness analysis, including its FDR-corrected decision.
    """

    if differences.empty:
        return pd.DataFrame(
            columns=[
                "scenario",
                "comparison_policy",
                "metric",
                "relative_change_percent",
                "p_value",
                "adjusted_p_value",
                "raw_significant",
                "significant",
                "direction",
                "classification",
            ]
        )

    merged = _merge_statistical_significance(
        differences,
        robustness_analysis,
    )

    records: list[dict] = []

    for _, row in merged.iterrows():
        metric = row["metric"]
        relative_change = row["relative_change_percent"]

        p_value = row.get(
            "p_value",
            float("nan"),
        )

        adjusted_p_value = row.get(
            "adjusted_p_value",
            float("nan"),
        )

        raw_significant = row.get(
            "raw_significant",
            pd.NA,
        )

        significant = row.get(
            "significant",
            pd.NA,
        )

        if pd.isna(relative_change):
            direction = "not_available"
            classification = "indeterminate"

        elif abs(float(relative_change)) < 1e-12:
            direction = "no_change"
            classification = "neutral"

        else:
            improvement = (
                relative_change > 0
                if HIGHER_IS_BETTER[metric]
                else relative_change < 0
            )

            if improvement:
                direction = "favorable"

                if (
                    not pd.isna(significant)
                    and bool(significant)
                ):
                    classification = (
                        "significant_improvement"
                    )
                else:
                    classification = (
                        "directional_improvement"
                    )

            else:
                direction = "unfavorable"

                if (
                    not pd.isna(significant)
                    and bool(significant)
                ):
                    classification = (
                        "significant_regression"
                    )
                else:
                    classification = (
                        "directional_regression"
                    )

        records.append(
            {
                "scenario": row["scenario"],
                "comparison_policy": row[
                    "comparison_policy"
                ],
                "metric": metric,
                "relative_change_percent": (
                    relative_change
                ),
                "p_value": p_value,
                "adjusted_p_value": adjusted_p_value,
                "raw_significant": raw_significant,
                "significant": significant,
                "direction": direction,
                "classification": classification,
            }
        )

    return pd.DataFrame(records)


def build_scenario_summary(
    interpretation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a scenario-level summary that distinguishes statistical
    significance from directional effects.

    Significant effects take precedence when determining the primary
    scenario classification, but directional tradeoffs are preserved
    in the output so that statistically significant improvements are
    not incorrectly presented as uniformly positive outcomes.
    """

    columns = [
        "scenario",
        "improvements",
        "regressions",
        "neutral",
        "significant_improvements",
        "significant_regressions",
        "directional_improvements",
        "directional_regressions",
        "indeterminate",
        "overall_classification",
    ]

    if interpretation.empty:
        return pd.DataFrame(columns=columns)

    required_columns = {
        "scenario",
        "classification",
        "significant",
    }

    _require_columns(
        interpretation,
        required_columns,
        "Robustness interpretation",
    )

    records: list[dict] = []

    for scenario, group in interpretation.groupby(
        "scenario",
        sort=True,
    ):
        improvements = int(
            (
                group["classification"].isin(
                    [
                        "improvement",
                        "directional_improvement",
                        "significant_improvement",
                    ]
                )
            ).sum()
        )

        regressions = int(
            (
                group["classification"].isin(
                    [
                        "regression",
                        "directional_regression",
                        "significant_regression",
                    ]
                )
            ).sum()
        )

        neutral = int(
            (
                group["classification"] == "neutral"
            ).sum()
        )

        indeterminate = int(
            (
                group["classification"]
                == "indeterminate"
            ).sum()
        )

        significant_improvements = int(
            (
                (group["significant"] == True)
                & (
                    group["direction"]
                    == "favorable"
                )
            ).sum()
        )

        significant_regressions = int(
            (
                (group["significant"] == True)
                & (
                    group["direction"]
                    == "unfavorable"
                )
            ).sum()
        )

        directional_improvements = int(
            (
                (group["direction"] == "favorable")
                & (group["significant"] != True)
                & (
                    group["classification"]
                    != "indeterminate"
                )
            ).sum()
        )

        directional_regressions = int(
            (
                (group["direction"] == "unfavorable")
                & (group["significant"] != True)
                & (
                    group["classification"]
                    != "indeterminate"
                )
            ).sum()
        )

        if significant_improvements > 0:
            if (
                significant_regressions > 0
                or directional_regressions > 0
            ):
                overall = (
                    "significant_improvement_with_"
                    "directional_tradeoffs"
                )
            else:
                overall = "significant_net_improvement"

        elif significant_regressions > 0:
            if (
                significant_improvements > 0
                or directional_improvements > 0
            ):
                overall = (
                    "significant_regression_with_"
                    "directional_improvements"
                )
            else:
                overall = "significant_net_regression"

        elif directional_improvements > directional_regressions:
            overall = "directional_net_improvement"

        elif directional_regressions > directional_improvements:
            overall = "directional_net_regression"

        else:
            overall = "mixed_or_neutral"

        records.append(
            {
                "scenario": scenario,
                "improvements": improvements,
                "regressions": regressions,
                "neutral": neutral,
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
                "indeterminate": indeterminate,
                "overall_classification": overall,
            }
        )

    return pd.DataFrame(records, columns=columns)


def generate_effect_heatmap(
    differences: pd.DataFrame,
    output_path: Path,
) -> None:
    if differences.empty:
        return

    heatmap_data = differences.pivot_table(
        index="metric",
        columns="scenario",
        values="relative_change_percent",
        aggfunc="mean",
    )

    if heatmap_data.empty:
        return

    figure_height = max(
        5.0,
        0.45 * len(heatmap_data) + 2.0,
    )

    fig, ax = plt.subplots(
        figsize=(11, figure_height)
    )

    image = ax.imshow(
        heatmap_data.values,
        aspect="auto",
        cmap="coolwarm",
    )

    ax.set_xticks(
        range(len(heatmap_data.columns))
    )
    ax.set_xticklabels(
        heatmap_data.columns,
        rotation=30,
        ha="right",
    )

    ax.set_yticks(
        range(len(heatmap_data.index))
    )
    ax.set_yticklabels(
        heatmap_data.index
    )

    ax.set_title(
        "Robustness Relative Performance Change"
    )

    ax.set_xlabel("Scenario")
    ax.set_ylabel("Metric")

    colorbar = fig.colorbar(
        image,
        ax=ax,
    )
    colorbar.set_label(
        "Relative change (%)"
    )

    for row_index in range(
        heatmap_data.shape[0]
    ):
        for column_index in range(
            heatmap_data.shape[1]
        ):
            value = heatmap_data.iloc[
                row_index,
                column_index,
            ]

            if pd.isna(value):
                text = "NA"
            else:
                text = f"{value:.2f}%"

            ax.text(
                column_index,
                row_index,
                text,
                ha="center",
                va="center",
                fontsize=8,
            )

    fig.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def generate_diagnostic_outputs(
    output_directory: Path | str,
) -> list[Path]:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    figures_directory = (
        output_directory / "figures"
    )

    figures_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw = load_robustness_raw(
        output_directory
    )

    robustness_analysis = (
        load_robustness_analysis(
            output_directory
        )
    )

    pressure_summary = (
        build_policy_pressure_summary(raw)
    )

    pressure_differences = (
        build_pressure_differences(
            pressure_summary
        )
    )

    interpretation = (
        build_robustness_interpretation(
            pressure_differences,
            robustness_analysis,
        )
    )

    scenario_summary = (
        build_scenario_summary(
            interpretation
        )
    )

    summary_path = (
        output_directory
        / "robustness_diagnostic_policy_summary.csv"
    )

    differences_path = (
        output_directory
        / "robustness_diagnostic_differences.csv"
    )

    interpretation_path = (
        output_directory
        / "robustness_diagnostic_interpretation.csv"
    )

    scenario_path = (
        output_directory
        / "robustness_diagnostic_scenario_summary.csv"
    )

    heatmap_path = (
        figures_directory
        / "robustness_diagnostic_heatmap.png"
    )

    pressure_summary.to_csv(
        summary_path,
        index=False,
    )

    pressure_differences.to_csv(
        differences_path,
        index=False,
    )

    interpretation.to_csv(
        interpretation_path,
        index=False,
    )

    scenario_summary.to_csv(
        scenario_path,
        index=False,
    )

    generate_effect_heatmap(
        pressure_differences,
        heatmap_path,
    )

    return [
        summary_path,
        differences_path,
        interpretation_path,
        scenario_path,
        heatmap_path,
    ]


if __name__ == "__main__":
    output_directory = (
        Path("outputs")
        / "robustness"
    )

    paths = generate_diagnostic_outputs(
        output_directory
    )

    print(
        "Robustness diagnostic analysis completed."
    )

    for path in paths:
        print(
            f"Output saved to: {path}"
        )