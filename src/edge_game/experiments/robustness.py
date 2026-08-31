"""Robustness experiments under stressed workload and resource conditions."""
from __future__ import annotations
from dataclasses import dataclass, replace
from pathlib import Path
import pandas as pd
from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import BaselinePolicy
from ..config import SimulationConfig
from .runner import build_mean_field_policy
from .statistics import aggregate_policy_metrics, paired_policy_comparison

@dataclass(frozen=True)
class RobustnessScenario:
    name: str
    description: str
    tasks_per_step: int
    maximum_cpu_capacity: float
    maximum_memory_capacity: float
    maximum_bandwidth: float
    maximum_cpu_demand: float
    maximum_memory_demand: float
    maximum_bandwidth_demand: float

@dataclass
class RobustnessResult:
    raw_results: pd.DataFrame
    summary: pd.DataFrame
    paired_comparison: pd.DataFrame
    equilibrium_diagnostics: dict[str, float | int | bool]

ROBUSTNESS_SCENARIOS = (
    RobustnessScenario("nominal", "Reference operating conditions.", 3, 30.0, 32.0, 100.0, 8.0, 8.0, 15.0),
    RobustnessScenario("workload_stress", "High arrival intensity with unchanged node capacity.", 15, 30.0, 32.0, 100.0, 8.0, 8.0, 15.0),
    RobustnessScenario("cpu_scarcity", "Reduced CPU capacity with unchanged arrival intensity.", 8, 20.0, 32.0, 100.0, 8.0, 8.0, 15.0),
    RobustnessScenario("memory_bandwidth_stress", "Tighter memory and bandwidth capacity with higher demands.", 8, 30.0, 20.0, 60.0, 8.0, 12.0, 20.0),
    RobustnessScenario("mixed_stress", "Combined workload and multi-resource stress.", 15, 20.0, 20.0, 60.0, 8.0, 12.0, 20.0),
)

def _scenario_config(config: SimulationConfig, scenario: RobustnessScenario) -> SimulationConfig:
    """Apply one robustness perturbation to the base configuration."""
    return replace(
        config,
        tasks_per_step=scenario.tasks_per_step,
        maximum_cpu_capacity=scenario.maximum_cpu_capacity,
        maximum_memory_capacity=scenario.maximum_memory_capacity,
        maximum_bandwidth=scenario.maximum_bandwidth,
        maximum_cpu_demand=scenario.maximum_cpu_demand,
        maximum_memory_demand=scenario.maximum_memory_demand,
        maximum_bandwidth_demand=scenario.maximum_bandwidth_demand,
    )

def run_robustness_experiment(config: SimulationConfig, seeds: tuple[int, ...], output_directory: str | Path | None = None) -> RobustnessResult:
    """Evaluate both policies across controlled stress scenarios."""
    mean_field_policy, diagnostics = build_mean_field_policy(config)
    rows: list[dict] = []
    for scenario in ROBUSTNESS_SCENARIOS:
        scenario_config = _scenario_config(config, scenario)
        for seed in seeds:
            seed_config = replace(scenario_config, seed=seed)
            baseline_result = run_policy_experiment(config=seed_config, policy_name="least_loaded_baseline", policy=BaselinePolicy(config=seed_config))
            mean_field_result = run_policy_experiment(config=seed_config, policy_name="priority_aware_mean_field", policy=mean_field_policy)
            for result in (baseline_result, mean_field_result):
                rows.append({"scenario": scenario.name, "scenario_description": scenario.description, "seed": int(seed), "policy_name": result.policy_name, **result.metrics})
    raw_results = pd.DataFrame(rows).sort_values(["scenario", "seed", "policy_name"]).reset_index(drop=True)
    summary_parts: list[pd.DataFrame] = []
    paired_parts: list[pd.DataFrame] = []
    for scenario in raw_results["scenario"].unique():
        frame = raw_results.loc[raw_results["scenario"] == scenario].copy()
        scenario_summary = aggregate_policy_metrics(frame)
        scenario_summary.insert(0, "scenario", scenario)
        summary_parts.append(scenario_summary)
        scenario_paired = paired_policy_comparison(frame)
        scenario_paired.insert(0, "scenario", scenario)
        paired_parts.append(scenario_paired)
    summary = pd.concat(summary_parts, ignore_index=True)
    paired_comparison = pd.concat(paired_parts, ignore_index=True)
    if output_directory is not None:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)
        raw_results.to_csv(output_path / "robustness_raw.csv", index=False)
        summary.to_csv(output_path / "robustness_summary.csv", index=False)
        paired_comparison.to_csv(output_path / "robustness_paired_comparison.csv", index=False)
        pd.DataFrame([diagnostics]).to_csv(output_path / "equilibrium_diagnostics.csv", index=False)
    return RobustnessResult(raw_results, summary, paired_comparison, diagnostics)
