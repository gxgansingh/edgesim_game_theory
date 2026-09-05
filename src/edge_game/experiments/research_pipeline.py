"""End-to-end research evaluation pipeline and master report generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..config import SimulationConfig
from .ablation import run_ablation_experiment
from .module1 import run_module1_experiment
from .module2 import run_module2_experiment
from .module3 import run_module3_experiment
from .performance_matrix import run_performance_matrix
from .robustness import run_robustness_experiment
from .robustness_analysis import generate_robustness_analysis_outputs
from .robustness_diagnostic import generate_diagnostic_outputs
from .robustness_report import generate_robustness_report
from .utility_sensitivity import run_utility_weight_sensitivity, save_utility_sensitivity_result
from .visualization import generate_robustness_figures, generate_utility_sensitivity_figures


@dataclass(frozen=True)
class ResearchPipelineResult:
    """Paths produced by the complete evaluation pipeline."""

    output_directory: Path
    master_report: Path
    manifest: Path


PIPELINE_STAGES = (
    "module1",
    "module2",
    "module3",
    "performance_matrix",
    "ablation",
    "robustness",
    "utility_sensitivity",
)


def _seed_sequence(config: SimulationConfig) -> tuple[int, ...]:
    """Build the deterministic repeated-experiment seed sequence."""
    return tuple(
        range(
            config.experiment_seed_start,
            config.experiment_seed_start + config.experiment_repetitions,
        )
    )


def _write_manifest(
    output_directory: Path,
    config: SimulationConfig,
    seeds: tuple[int, ...],
) -> Path:
    """Write the experiment design and generated-stage manifest."""
    manifest_path = output_directory / "pipeline_manifest.csv"
    rows = [
        {"category": "pipeline", "name": "stages", "value": ",".join(PIPELINE_STAGES)},
        {"category": "experiment", "name": "seed_count", "value": len(seeds)},
        {"category": "experiment", "name": "seed_start", "value": config.experiment_seed_start},
        {"category": "priority", "name": "P1 Critical", "value": config.critical_priority_share},
        {"category": "priority", "name": "P2 High", "value": config.high_priority_share},
        {"category": "priority", "name": "P3 Medium", "value": config.medium_priority_share},
        {"category": "priority", "name": "P4 Low", "value": config.low_priority_share},
        {"category": "network", "name": "network_load_levels", "value": ",".join(map(str, config.network_load_levels))},
        {"category": "resources", "name": "feasibility_dimensions", "value": "CPU,memory,bandwidth,latency,energy,queue"},
    ]
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


def _safe_read(path: Path) -> pd.DataFrame | None:
    """Read an optional CSV without failing master-report generation."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (OSError, ValueError, pd.errors.ParserError):
        return None


def _metric_snapshot(performance_matrix: pd.DataFrame | None) -> list[str]:
    """Extract compact headline findings from the performance matrix."""
    if performance_matrix is None or performance_matrix.empty:
        return ["Performance-matrix results were not available."]

    rows: list[str] = []
    mfg = performance_matrix.loc[
        performance_matrix["policy_name"] == "priority_aware_mean_field"
    ].copy()
    if mfg.empty:
        return ["No MFG performance rows were available."]

    for metric in ("utility_mean", "response_time_mean", "throughput", "success_ratio"):
        frame = mfg.loc[mfg["metric"] == metric]
        if frame.empty:
            continue
        best = frame.loc[frame["mean"].idxmax()]
        rows.append(
            f"- Best observed MFG {metric.replace('_', ' ')}: "
            f"{best['mean']:.4f} at network load {best['network_load']:.2f}."
        )
    return rows or ["No headline performance metrics were available."]


def _filtering_snapshot(output_directory: Path) -> list[str]:
    """Summarize the resource-filtering audit produced by the pipeline."""
    summary = _safe_read(
        output_directory
        / "module2"
        / "aggregated"
        / "resource_filtering_summary.csv"
    )
    if summary is None or summary.empty or "metric" not in summary.columns:
        return ["Resource-filtering summary was not available."]

    lines: list[str] = []
    for metric in ("total_candidate_checks", "feasible_candidate_checks", "filtered_candidate_checks", "filtering_ratio"):
        frame = summary.loc[summary["metric"] == metric]
        if frame.empty:
            continue
        value = frame.iloc[0]["value"]
        if metric == "filtering_ratio":
            lines.append(f"- Filtering ratio: {float(value):.2%}.")
        else:
            lines.append(f"- {metric.replace('_', ' ').title()}: {float(value):.2f}.")
    return lines or ["Resource-filtering summary was not available."]


def _write_master_report(
    output_directory: Path,
    config: SimulationConfig,
    seeds: tuple[int, ...],
) -> Path:
    """Generate the final report directly from generated experiment artifacts."""
    report_path = output_directory / "final_report.md"
    performance = _safe_read(
        output_directory
        / "performance_matrix"
        / "aggregated"
        / "performance_matrix.csv"
    )

    lines = [
        "# Edge Game Simulator: Final Research Evaluation",
        "",
        "## Experiment Design",
        "",
        f"- Repeated runs: **{len(seeds)}**",
        f"- Seeds: **{seeds[0]}–{seeds[-1]}**" if seeds else "- Seeds: none",
        f"- Network-load levels: **{', '.join(f'{x:.2f}' for x in config.network_load_levels)}**",
        "- Policies: **least-loaded baseline** and **priority-aware Mean-Field Game**",
        "- Priority classes: **P1 Critical 10%, P2 High 20%, P3 Medium 40%, P4 Low 30%**",
        "- Feasibility resources: **CPU, memory, bandwidth, latency, energy, queue**",
        "",
        "## Research Pipeline",
        "",
        "1. Module 1 builds the edge environment and workload population.",
        "2. Resource feasibility filtering removes infeasible task-node pairs.",
        "3. Module 2 solves the priority-aware HJB-FPK Mean-Field equilibrium.",
        "4. The policy selects a node only from the feasible candidate set.",
        "5. Module 3 evaluates hierarchical execution.",
        "6. Repeated experiments evaluate performance under varying conditions.",
        "7. Statistical and robustness analyses quantify uncertainty and stress behavior.",
        "",
        "## Priority Model",
        "",
        "| Class | Meaning | Share |",
        "|---|---|---:|",
        "| P1 | Critical | 10% |",
        "| P2 | High | 20% |",
        "| P3 | Medium | 40% |",
        "| P4 | Low | 30% |",
        "",
        "## Resource-Filtering Audit",
        "",
        *_filtering_snapshot(output_directory),
        "",
        "Filtering is a feasibility stage. It does not select the final node. "
        "The Mean-Field/game-theoretic policy performs selection from the remaining feasible candidates.",
        "",
        "## Network-Load Performance",
        "",
        *_metric_snapshot(performance),
        "",
        "The complete machine-generated performance matrix is stored in "
        "`performance_matrix/aggregated/performance_matrix.csv`.",
        "",
        "## Generated Evaluation Artifacts",
        "",
        "- `module1/`: environment and workload evaluation",
        "- `module2/`: priority-aware MFG and selection analysis",
        "- `module3/`: hierarchical execution evaluation",
        "- `performance_matrix/`: network-load matrix, confidence intervals, plots, and report",
        "- `ablation/`: component-removal analysis",
        "- `robustness/`: stress scenarios, diagnostics, statistical analysis, and figures",
        "- `utility_sensitivity/`: utility-weight sensitivity, selection analysis, diagnostics, and figures",
        "- `pipeline_manifest.csv`: reproducibility metadata",
        "",
        "## Reproducibility",
        "",
        "All results in this directory are generated by the simulator from the configured seed sequence. "
        "The CSV files are the authoritative numerical outputs; Markdown reports summarize those generated results.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def run_full_research_pipeline(
    config: SimulationConfig,
    output_directory: str | Path | None = None,
) -> ResearchPipelineResult:
    """Run the complete project evaluation and generate a master report."""
    base_output = Path(output_directory or config.output_directory) / "final_evaluation"
    base_output.mkdir(parents=True, exist_ok=True)
    seeds = _seed_sequence(config)
    manifest = _write_manifest(base_output, config, seeds)

    print("[1/7] Running Module 1...", flush=True)
    run_module1_experiment(config, seeds, base_output / "module1")

    print("[2/7] Running Module 2...", flush=True)
    run_module2_experiment(config, seeds, base_output / "module2")

    print("[3/7] Running Module 3...", flush=True)
    run_module3_experiment(config, seeds, base_output / "module3")

    print("[4/7] Running network-load performance matrix...", flush=True)
    run_performance_matrix(config, seeds, base_output / "performance_matrix")

    print("[5/7] Running ablation analysis...", flush=True)
    run_ablation_experiment(
        config=config,
        seeds=seeds,
        scenarios=config.workload_scenarios,
        output_directory=base_output / "ablation",
    )

    print("[6/7] Running robustness and utility-sensitivity analysis...", flush=True)
    robustness_dir = base_output / "robustness"
    run_robustness_experiment(config, seeds, robustness_dir)
    generate_robustness_analysis_outputs(robustness_dir)
    generate_diagnostic_outputs(robustness_dir)
    generate_robustness_report(robustness_dir)
    generate_robustness_figures(robustness_dir)

    utility_dir = base_output / "utility_sensitivity"
    utility_result = run_utility_weight_sensitivity(
        config=config,
        seeds=seeds,
        output_directory=utility_dir,
    )
    save_utility_sensitivity_result(utility_result, utility_dir)
    generate_utility_sensitivity_figures(utility_dir)

    print("[7/7] Generating final report...", flush=True)
    report = _write_master_report(base_output, config, seeds)

    return ResearchPipelineResult(
        output_directory=base_output,
        master_report=report,
        manifest=manifest,
    )
