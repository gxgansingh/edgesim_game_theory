"""Utility-weight sensitivity experiments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
import time

import pandas as pd

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.policy import BaselinePolicy, MeanFieldPolicy
from ..config import SimulationConfig
from .runner import WORKLOAD_SCENARIO_TASKS, build_mean_field_policy
from .statistics import (
    aggregate_policy_metrics,
    paired_decision_audit,
    paired_policy_comparison,
    paired_selection_analysis,
    policy_selection_summary,
)


@dataclass(frozen=True)
class UtilityWeightProfile:
    """Named utility-weight configuration used in sensitivity analysis."""

    name: str
    priority_reward: float
    latency_cost: float
    resource_cost: float
    queue_cost: float
    energy_cost: float


@dataclass
class UtilitySensitivityResult:
    """Store utility-weight sensitivity outputs."""

    raw_results: pd.DataFrame
    summary: pd.DataFrame
    paired_comparison: pd.DataFrame
    selection_comparison: pd.DataFrame
    decision_comparison: pd.DataFrame


UTILITY_WEIGHT_PROFILES = (
    UtilityWeightProfile(
        name="balanced",
        priority_reward=1.0,
        latency_cost=1.0,
        resource_cost=1.0,
        queue_cost=1.0,
        energy_cost=1.0,
    ),
    UtilityWeightProfile(
        name="priority_latency",
        priority_reward=1.5,
        latency_cost=1.5,
        resource_cost=1.0,
        queue_cost=1.0,
        energy_cost=1.0,
    ),
    UtilityWeightProfile(
        name="priority_latency_queue",
        priority_reward=1.5,
        latency_cost=2.0,
        resource_cost=1.0,
        queue_cost=1.5,
        energy_cost=1.0,
    ),
    UtilityWeightProfile(
        name="priority_latency_queue_energy",
        priority_reward=1.5,
        latency_cost=2.0,
        resource_cost=1.0,
        queue_cost=1.5,
        energy_cost=1.5,
    ),
)


def _profile_config(
    config: SimulationConfig,
    profile: UtilityWeightProfile,
) -> SimulationConfig:
    """Apply one utility-weight profile to a simulation configuration."""
    return replace(
        config,
        utility_priority_reward_weight=profile.priority_reward,
        utility_latency_cost_weight=profile.latency_cost,
        utility_resource_cost_weight=profile.resource_cost,
        utility_queue_cost_weight=profile.queue_cost,
        utility_energy_cost_weight=profile.energy_cost,
    )


def run_utility_weight_sensitivity(
    config: SimulationConfig,
    seeds: list[int] | tuple[int, ...],
    scenarios: list[str] | tuple[str, ...] | None = None,
    profiles: tuple[UtilityWeightProfile, ...] = UTILITY_WEIGHT_PROFILES,
    output_directory: str | Path | None = None,
) -> UtilitySensitivityResult:
    """Evaluate policy performance across utility-weight profiles."""
    normalized_seeds = tuple(int(seed) for seed in seeds)

    if not normalized_seeds:
        raise ValueError("At least one experiment seed is required.")

    if len(set(normalized_seeds)) != len(normalized_seeds):
        raise ValueError("Experiment seeds must be unique.")

    selected_scenarios = tuple(
        scenarios if scenarios is not None else config.workload_scenarios
    )

    unknown = set(selected_scenarios) - set(WORKLOAD_SCENARIO_TASKS)
    if unknown:
        valid = ", ".join(sorted(WORKLOAD_SCENARIO_TASKS))
        raise ValueError(
            f"Unknown workload scenario(s): {sorted(unknown)}. "
            f"Valid scenarios: {valid}."
        )

    rows: list[dict] = []
    selection_rows: list[dict] = []
    decision_rows: list[dict] = []
    profile_diagnostics: dict[str, dict] = {}
    equilibrium_build_seconds: dict[str, float] = {}

    for profile in profiles:
        profile_config = _profile_config(config, profile)
        equilibrium_start = time.perf_counter()
        mean_field_policy, equilibrium_diagnostics = build_mean_field_policy(
            profile_config,
            fpk_time_step=0.01,
            fpk_max_iterations=1000,
        )
        equilibrium_build_seconds[profile.name] = (
            time.perf_counter() - equilibrium_start
        )
        equilibrium_diagnostics["equilibrium_build_seconds"] = (
            equilibrium_build_seconds[profile.name]
        )
        profile_diagnostics[profile.name] = equilibrium_diagnostics

        print(
            f"Equilibrium ready: profile={profile.name} "
            f"converged={equilibrium_diagnostics['converged']} "
            f"iterations={equilibrium_diagnostics['iterations']} "
            f"time={equilibrium_build_seconds[profile.name]:.2f}s",
            flush=True,
        )

        for scenario in selected_scenarios:
            print(
                f"Running profile={profile.name} scenario={scenario} "
                f"seeds={len(normalized_seeds)}...",
                flush=True,
            )
            scenario_config = replace(
                profile_config,
                tasks_per_step=WORKLOAD_SCENARIO_TASKS[scenario],
            )

            for seed in normalized_seeds:
                seed_config = replace(
                    scenario_config,
                    seed=seed,
                )

                baseline_result = run_policy_experiment(
                    config=seed_config,
                    policy_name="least_loaded_baseline",
                    policy=BaselinePolicy(config=seed_config),
                )

                mean_field_result = run_policy_experiment(
                    config=seed_config,
                    policy_name="priority_aware_mean_field",
                    policy=mean_field_policy,
                )

                for result in (
                    baseline_result,
                    mean_field_result,
                ):
                    rows.append(
                        {
                            "profile": profile.name,
                            "scenario": scenario,
                            "seed": int(seed),
                            "policy_name": result.policy_name,
                            "priority_reward_weight": profile.priority_reward,
                            "latency_cost_weight": profile.latency_cost,
                            "resource_cost_weight": profile.resource_cost,
                            "queue_cost_weight": profile.queue_cost,
                            "energy_cost_weight": profile.energy_cost,
                            **result.metrics,
                        }
                    )

                    for record in result.selection_records:
                        selection_rows.append(
                            {
                                "profile": profile.name,
                                "scenario": scenario,
                                "seed": int(seed),
                                "policy_name": result.policy_name,
                                **record,
                            }
                        )

                    for record in result.decision_records:
                        decision_rows.append(
                            {
                                "profile": profile.name,
                                "scenario": scenario,
                                "seed": int(seed),
                                **record,
                            }
                        )

    raw_results = pd.DataFrame(rows).sort_values(
        ["profile", "scenario", "seed", "policy_name"]
    ).reset_index(drop=True)

    summary_parts: list[pd.DataFrame] = []
    comparison_parts: list[pd.DataFrame] = []

    for (profile, scenario), frame in raw_results.groupby(
        ["profile", "scenario"],
        sort=True,
    ):
        summary = aggregate_policy_metrics(frame)
        summary.insert(0, "scenario", scenario)
        summary.insert(0, "profile", profile)
        summary_parts.append(summary)

        comparison = paired_policy_comparison(frame)
        comparison.insert(0, "scenario", scenario)
        comparison.insert(0, "profile", profile)
        comparison_parts.append(comparison)

    summary_frame = pd.concat(
        summary_parts,
        ignore_index=True,
    )
    comparison_frame = pd.concat(
        comparison_parts,
        ignore_index=True,
    )

    selection_records = pd.DataFrame(selection_rows)
    decision_records = pd.DataFrame(decision_rows)

    selection_comparison_parts: list[pd.DataFrame] = []
    decision_comparison_parts: list[pd.DataFrame] = []

    if not selection_records.empty:
        for (profile, scenario), frame in selection_records.groupby(
            ["profile", "scenario"],
            sort=True,
        ):
            selection_comparison, _ = paired_selection_analysis(frame)
            if not selection_comparison.empty:
                selection_comparison.insert(0, "scenario", scenario)
                selection_comparison.insert(0, "profile", profile)
                selection_comparison_parts.append(selection_comparison)

    if not decision_records.empty:
        for (profile, scenario), frame in decision_records.groupby(
            ["profile", "scenario"],
            sort=True,
        ):
            decision_comparison = paired_decision_audit(frame)
            if not decision_comparison.empty:
                decision_comparison.insert(0, "scenario", scenario)
                decision_comparison.insert(0, "profile", profile)
                decision_comparison_parts.append(decision_comparison)

    selection_comparison_frame = (
        pd.concat(selection_comparison_parts, ignore_index=True)
        if selection_comparison_parts
        else pd.DataFrame()
    )
    decision_comparison_frame = (
        pd.concat(decision_comparison_parts, ignore_index=True)
        if decision_comparison_parts
        else pd.DataFrame()
    )

    result = UtilitySensitivityResult(
        raw_results=raw_results,
        summary=summary_frame,
        paired_comparison=comparison_frame,
        selection_comparison=selection_comparison_frame,
        decision_comparison=decision_comparison_frame,
    )

    if output_directory is not None:
        save_utility_sensitivity_result(
            result=result,
            output_directory=output_directory,
            equilibrium_diagnostics=profile_diagnostics,
        )

    return result


def save_utility_sensitivity_result(
    result: UtilitySensitivityResult,
    output_directory: str | Path,
    equilibrium_diagnostics: dict[str, dict],
) -> None:
    """Save utility sensitivity outputs as CSV files."""
    output_path = Path(output_directory)
    output_path.mkdir(parents=True, exist_ok=True)

    result.raw_results.to_csv(
        output_path / "utility_sensitivity_raw.csv",
        index=False,
    )
    result.summary.to_csv(
        output_path / "utility_sensitivity_summary.csv",
        index=False,
    )
    result.paired_comparison.to_csv(
        output_path / "utility_sensitivity_paired_comparison.csv",
        index=False,
    )

    # Always write these analysis tables. The visualization layer depends on
    # their presence, and a missing file should never silently disable a figure.
    result.selection_comparison.to_csv(
        output_path / "selection_comparison.csv",
        index=False,
    )
    result.decision_comparison.to_csv(
        output_path / "decision_comparison.csv",
        index=False,
    )

    diagnostic_rows = []
    for profile, diagnostics in equilibrium_diagnostics.items():
        diagnostic_rows.append(
            {"profile": profile, **diagnostics}
        )

    pd.DataFrame(diagnostic_rows).to_csv(
        output_path / "equilibrium_diagnostics.csv",
        index=False,
    )
