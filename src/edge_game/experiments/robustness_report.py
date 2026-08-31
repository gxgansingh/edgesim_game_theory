from __future__ import annotations

from pathlib import Path

import pandas as pd


def _load_csv(
    output_directory: Path,
    filename: str,
) -> pd.DataFrame:
    path = output_directory / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required robustness output not found: {path}"
        )

    return pd.read_csv(path)


def _format_number(value: object) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.4f}"


def _format_percent(value: object) -> str:
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.2f}%"


def build_robustness_report(
    output_directory: Path | str,
) -> str:
    """
    Build a research-oriented Markdown report from the generated
    robustness experiment outputs.

    The report uses the existing experiment outputs and does not
    recompute statistical tests.
    """

    output_directory = Path(output_directory)

    effect_analysis = _load_csv(
        output_directory,
        "robustness_effect_analysis.csv",
    )

    tradeoff_summary = _load_csv(
        output_directory,
        "robustness_tradeoff_summary.csv",
    )

    diagnostic_summary = _load_csv(
        output_directory,
        "robustness_diagnostic_scenario_summary.csv",
    )

    diagnostic_interpretation = _load_csv(
        output_directory,
        "robustness_diagnostic_interpretation.csv",
    )

    raw = _load_csv(
        output_directory,
        "robustness_raw.csv",
    )

    scenarios = sorted(
        raw["scenario"].dropna().unique()
    )

    policies = sorted(
        raw["policy_name"].dropna().unique()
    )

    seed_count = int(
        raw["seed"].nunique()
    )

    metric_count = int(
        effect_analysis["metric"].nunique()
    )

    significant = effect_analysis[
        effect_analysis["significant"] == True
    ].copy()

    significant_improvements = significant[
        significant["favorable"] == True
    ].copy()

    significant_regressions = significant[
        significant["unfavorable"] == True
    ].copy()

    report: list[str] = []

    report.append(
        "# Robustness Analysis Report"
    )

    report.append(
        ""
    )

    report.append(
        "## 1. Experimental Overview"
    )

    report.append(
        ""
    )

    report.append(
        "The robustness experiment evaluates the "
        "priority-aware mean-field policy against "
        "the least-loaded baseline across multiple "
        "operating conditions."
    )

    report.append(
        ""
    )

    report.append(
        f"- Scenarios: {len(scenarios)}"
    )

    report.append(
        f"- Policies: {len(policies)}"
    )

    report.append(
        f"- Seeds: {seed_count}"
    )

    report.append(
        f"- Metrics: {metric_count}"
    )

    report.append(
        f"- Significance level: "
        f"{_format_number(effect_analysis['alpha'].iloc[0])}"
    )

    report.append(
        ""
    )

    report.append(
        "### Scenarios"
    )

    report.append(
        ""
    )

    for scenario in scenarios:
        report.append(
            f"- `{scenario}`"
        )

    report.append(
        ""
    )

    report.append(
        "### Policies"
    )

    report.append(
        ""
    )

    for policy in policies:
        report.append(
            f"- `{policy}`"
        )

    report.append(
        ""
    )

    report.append(
        "## 2. Multiple-Comparison Correction"
    )

    report.append(
        ""
    )

    report.append(
        "Statistical significance is determined using "
        "Benjamini-Hochberg false-discovery-rate correction. "
        "The `raw_significant` field represents the uncorrected "
        "p-value decision, while `significant` represents the "
        "FDR-corrected decision."
    )

    report.append(
        ""
    )

    report.append(
        f"- Raw significant effects: "
        f"{int(effect_analysis['raw_significant'].sum())}"
    )

    report.append(
        f"- FDR-significant effects: "
        f"{int(effect_analysis['significant'].sum())}"
    )

    report.append(
        ""
    )

    report.append(
        "## 3. Statistically Significant Improvements"
    )

    report.append(
        ""
    )

    if significant_improvements.empty:
        report.append(
            "No statistically significant improvements "
            "remain after FDR correction."
        )
    else:
        report.append(
            "| Scenario | Metric | Relative Change | Adjusted p-value |"
        )
        report.append(
            "|---|---|---:|---:|"
        )

        for _, row in significant_improvements.iterrows():
            report.append(
                "| "
                f"{row['scenario']} | "
                f"{row['metric']} | "
                f"{_format_percent(row['relative_change_percent'])} | "
                f"{_format_number(row['adjusted_p_value'])} |"
            )

    report.append(
        ""
    )

    report.append(
        "## 4. Statistically Significant Regressions"
    )

    report.append(
        ""
    )

    if significant_regressions.empty:
        report.append(
            "No statistically significant regressions "
            "remain after FDR correction."
        )
    else:
        report.append(
            "| Scenario | Metric | Relative Change | Adjusted p-value |"
        )
        report.append(
            "|---|---|---:|---:|"
        )

        for _, row in significant_regressions.iterrows():
            report.append(
                "| "
                f"{row['scenario']} | "
                f"{row['metric']} | "
                f"{_format_percent(row['relative_change_percent'])} | "
                f"{_format_number(row['adjusted_p_value'])} |"
            )

    report.append(
        ""
    )

    report.append(
        "## 5. Scenario-Level Interpretation"
    )

    report.append(
        ""
    )

    report.append(
        "| Scenario | Significant Improvements | "
        "Significant Regressions | "
        "Directional Improvements | "
        "Directional Regressions | Classification |"
    )

    report.append(
        "|---|---:|---:|---:|---:|---|"
    )

    for _, row in diagnostic_summary.iterrows():
        report.append(
            "| "
            f"{row['scenario']} | "
            f"{int(row['significant_improvements'])} | "
            f"{int(row['significant_regressions'])} | "
            f"{int(row['directional_improvements'])} | "
            f"{int(row['directional_regressions'])} | "
            f"{row['overall_classification']} |"
        )

    report.append(
        ""
    )

    report.append(
        "## 6. Detailed Scenario Findings"
    )

    report.append(
        ""
    )

    for scenario in scenarios:
        scenario_rows = diagnostic_interpretation[
            diagnostic_interpretation["scenario"]
            == scenario
        ]

        report.append(
            f"### {scenario}"
        )

        report.append(
            ""
        )

        for _, row in scenario_rows.iterrows():
            metric = row["metric"]
            change = row["relative_change_percent"]
            classification = row["classification"]

            if pd.isna(change):
                change_text = "N/A"
            else:
                change_text = (
                    f"{float(change):.2f}%"
                )

            report.append(
                f"- `{metric}`: "
                f"{change_text}, "
                f"{classification}"
            )

        report.append(
            ""
        )

    report.append(
        "## 7. Tradeoff Summary"
    )

    report.append(
        ""
    )

    report.append(
        "| Scenario | Significant Improvements | "
        "Significant Regressions | Net Significant Effects |"
    )

    report.append(
        "|---|---:|---:|---:|"
    )

    for _, row in tradeoff_summary.iterrows():
        report.append(
            "| "
            f"{row['scenario']} | "
            f"{int(row['significant_improvements'])} | "
            f"{int(row['significant_regressions'])} | "
            f"{int(row['net_significant_effects'])} |"
        )

    report.append(
        ""
    )

    report.append(
        "## 8. Overall Conclusion"
    )

    report.append(
        ""
    )

    if significant_improvements.empty and significant_regressions.empty:
        conclusion = (
            "No effects remain statistically significant after "
            "FDR correction. The observed changes should therefore "
            "be interpreted as directional rather than statistically "
            "confirmed effects."
        )
    elif (
        len(significant_improvements)
        > len(significant_regressions)
    ):
        conclusion = (
            "The priority-aware mean-field policy shows a net "
            "statistically significant improvement across the "
            "evaluated robustness scenarios, although individual "
            "tradeoffs remain."
        )
    elif (
        len(significant_regressions)
        > len(significant_improvements)
    ):
        conclusion = (
            "The priority-aware mean-field policy shows more "
            "statistically significant regressions than improvements "
            "after FDR correction. The policy therefore cannot be "
            "described as uniformly robust across all stress "
            "conditions."
        )
    else:
        conclusion = (
            "The policy exhibits a balanced mixture of statistically "
            "significant improvements and regressions. Robustness "
            "therefore depends on the operating condition and metric."
        )

    report.append(
        conclusion
    )

    report.append(
        ""
    )

    report.append(
        "## 9. Interpretation Guidance"
    )

    report.append(
        ""
    )

    report.append(
        "Directional improvements and regressions are not equivalent "
        "to statistically significant effects. Conclusions about "
        "statistical evidence should therefore use the FDR-corrected "
        "`significant` field."
    )

    report.append(
        ""
    )

    report.append(
        "Metrics with undefined percentage changes, such as a "
        "zero-to-zero rejected-task comparison, are reported as "
        "indeterminate rather than being assigned an artificial "
        "percentage change."
    )

    report.append(
        ""
    )

    return "\n".join(report)


def generate_robustness_report(
    output_directory: Path | str,
) -> Path:
    output_directory = Path(output_directory)

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    report = build_robustness_report(
        output_directory
    )

    report_path = (
        output_directory
        / "robustness_report.md"
    )

    report_path.write_text(
        report,
        encoding="utf-8",
    )

    return report_path


if __name__ == "__main__":
    output_directory = (
        Path("outputs")
        / "robustness"
    )

    report_path = generate_robustness_report(
        output_directory
    )

    print(
        "Robustness report generated."
    )

    print(
        f"Output saved to: {report_path}"
    )