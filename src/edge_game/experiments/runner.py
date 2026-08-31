"""Repeated policy experiment execution."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from ..algorithms.experiment import run_policy_experiment
from ..algorithms.mean_field import PriorityAwareMeanFieldSolver
from ..algorithms.policy import BaselinePolicy, MeanFieldPolicy
from ..config import SimulationConfig
from ..models.mean_field_model import (
    MeanFieldModel,
    MeanFieldParameters,
)
from .statistics import (
    aggregate_policy_metrics,
    paired_policy_comparison,
    paired_selection_analysis,
    policy_selection_summary,
    decision_audit_summary,
    paired_decision_audit,
)


WORKLOAD_SCENARIO_TASKS = {
    "default": 3,
    "moderate_congestion": 8,
    "high_congestion": 15,
}


@dataclass
class RepeatedExperimentResult:
    """Store repeated policy experiment outputs."""

    raw_results: pd.DataFrame
    summary: pd.DataFrame
    paired_comparison: pd.DataFrame
    selection_summary: pd.DataFrame
    selection_comparison: pd.DataFrame
    node_selection_frequency: pd.DataFrame
    decision_audit: pd.DataFrame
    decision_comparison: pd.DataFrame
    equilibrium_diagnostics: dict[
        str,
        float | int | bool | str,
    ]


def _extract_control_values(
    policy_entry,
) -> np.ndarray:
    """Return numerical control values from an equilibrium policy entry."""

    if hasattr(
        policy_entry,
        "control",
    ):
        values = policy_entry.control
    else:
        values = policy_entry

    return np.asarray(
        values,
        dtype=float,
    )


def _add_control_diagnostics(
    diagnostics: dict,
    equilibrium,
) -> None:
    """Add priority-specific equilibrium-control statistics."""

    for priority in sorted(
        equilibrium.policies
    ):
        policy_entry = (
            equilibrium.policies[
                priority
            ]
        )

        controls = (
            _extract_control_values(
                policy_entry
            )
        )

        if controls.size == 0:
            continue

        diagnostics[
            f"priority_{priority}_control_min"
        ] = float(
            np.min(controls)
        )

        diagnostics[
            f"priority_{priority}_control_max"
        ] = float(
            np.max(controls)
        )

        diagnostics[
            f"priority_{priority}_control_mean"
        ] = float(
            np.mean(controls)
        )

        diagnostics[
            f"priority_{priority}_control_std"
        ] = float(
            np.std(
                controls
            )
        )

        diagnostics[
            f"priority_{priority}_control_state_0"
        ] = float(
            controls[0]
        )

        diagnostics[
            f"priority_{priority}_control_state_05"
        ] = float(
            np.interp(
                0.5,
                np.linspace(
                    0.0,
                    1.0,
                    len(controls),
                ),
                controls,
            )
        )

        diagnostics[
            f"priority_{priority}_control_state_1"
        ] = float(
            controls[-1]
        )

        diagnostics[
            f"priority_{priority}_control_saturation_low"
        ] = float(
            np.mean(
                controls
                <= 1e-6
            )
        )

        diagnostics[
            f"priority_{priority}_control_saturation_high"
        ] = float(
            np.mean(
                controls
                >= 1.0 - 1e-6
            )
        )


def build_mean_field_policy(
    config: SimulationConfig,
    *,
    fpk_time_step: float | None = None,
    fpk_max_iterations: int | None = None,
    ablation_variant: str = "full",
) -> tuple[
    MeanFieldPolicy,
    dict[
        str,
        float | int | bool | str,
    ],
]:
    """Build the deterministic Mean-Field equilibrium policy."""

    parameters = MeanFieldParameters(
        diffusion=(
            config.mean_field_diffusion
        ),
        discount_factor=(
            config.mean_field_discount_factor
        ),
        utility_priority_reward_weight=(
            config.utility_priority_reward_weight
        ),
        utility_latency_cost_weight=(
            config.utility_latency_cost_weight
        ),
        utility_resource_cost_weight=(
            config.utility_resource_cost_weight
        ),
        utility_queue_cost_weight=(
            config.utility_queue_cost_weight
        ),
        utility_energy_cost_weight=(
            config.utility_energy_cost_weight
        ),
        ablation_variant=(
            ablation_variant
        ),
    )

    model = MeanFieldModel(
        parameters=parameters
    )

    state_grid = np.linspace(
        0.0,
        1.0,
        config.mean_field_state_points,
    )

    solver = (
        PriorityAwareMeanFieldSolver(
            model=model,
            state_grid=state_grid,
            tolerance=(
                config.mean_field_tolerance
            ),
            policy_tolerance=(
                config.mean_field_policy_tolerance
            ),
            raw_policy_tolerance=(
                config.mean_field_raw_policy_tolerance
            ),
            max_iterations=(
                config.mean_field_max_iterations
            ),
            damping=(
                config.mean_field_damping
            ),
            policy_damping=(
                config.mean_field_policy_damping
            ),
            fpk_time_step=(
                config.fpk_time_step
                if fpk_time_step is None
                else fpk_time_step
            ),
            fpk_tolerance=(
                config.fpk_tolerance
            ),
            fpk_max_iterations=(
                config.fpk_max_iterations
                if fpk_max_iterations is None
                else fpk_max_iterations
            ),
        )
    )

    equilibrium = (
        solver.solve()
    )

    diagnostics: dict[
        str,
        float | int | bool | str,
    ] = {
        "ablation_variant": (
            ablation_variant
        ),

        "converged": (
            equilibrium.converged
        ),

        "iterations": (
            equilibrium.iterations
        ),

        "distribution_residual": (
            equilibrium.residual
        ),

        "policy_residual": (
            equilibrium.policy_residual
        ),

        "raw_distribution_residual": (
            equilibrium.raw_residual
        ),

        "raw_policy_residual": (
            equilibrium.raw_policy_residual
        ),
    }

    for priority in sorted(
        equilibrium.fpk_iterations
    ):
        diagnostics[
            f"fpk_priority_{priority}_iterations"
        ] = (
            equilibrium.fpk_iterations[
                priority
            ]
        )

        diagnostics[
            f"fpk_priority_{priority}_residual"
        ] = (
            equilibrium.fpk_residuals[
                priority
            ]
        )

    _add_control_diagnostics(
        diagnostics=diagnostics,
        equilibrium=equilibrium,
    )

    return (
        MeanFieldPolicy(
            model=model,
            equilibrium=equilibrium,
            config=config,
        ),
        diagnostics,
    )


def _build_raw_row(
    seed: int,
    policy_name: str,
    metrics: dict,
) -> dict:
    """Build one normalized raw experiment row."""

    row = {
        "seed": int(
            seed
        ),
        "policy_name": (
            policy_name
        ),
    }

    row.update(
        metrics
    )

    return row


def _resolve_workload_scenario(
    config: SimulationConfig,
    scenario: str,
) -> SimulationConfig:
    """Return a configuration with the selected workload intensity."""

    if (
        scenario
        not in WORKLOAD_SCENARIO_TASKS
    ):
        valid = ", ".join(
            sorted(
                WORKLOAD_SCENARIO_TASKS
            )
        )

        raise ValueError(
            f"Unknown workload scenario "
            f"'{scenario}'. "
            f"Valid scenarios: {valid}."
        )

    return replace(
        config,
        tasks_per_step=(
            WORKLOAD_SCENARIO_TASKS[
                scenario
            ]
        ),
    )


def _run_repeated_with_policy(
    config: SimulationConfig,
    seeds: tuple[int, ...],
    scenario: str,
    mean_field_policy: MeanFieldPolicy,
    diagnostics: dict[
        str,
        float | int | bool | str,
    ],
    output_directory: (
        str | Path | None
    ),
) -> RepeatedExperimentResult:
    """Run paired experiments for one workload scenario."""

    scenario_config = (
        _resolve_workload_scenario(
            config=config,
            scenario=scenario,
        )
    )

    rows: list[dict] = []

    selection_rows: list[
        dict
    ] = []

    decision_rows: list[
        dict
    ] = []

    for seed in seeds:
        seed_config = replace(
            scenario_config,
            seed=seed,
        )

        baseline_result = (
            run_policy_experiment(
                config=seed_config,
                policy_name=(
                    "least_loaded_baseline"
                ),
                policy=BaselinePolicy(
                    config=seed_config
                ),
            )
        )

        mean_field_result = (
            run_policy_experiment(
                config=seed_config,
                policy_name=(
                    "priority_aware_mean_field"
                ),
                policy=mean_field_policy,
            )
        )

        rows.append(
            _build_raw_row(
                seed=seed,
                policy_name=(
                    baseline_result
                    .policy_name
                ),
                metrics=(
                    baseline_result
                    .metrics
                ),
            )
        )

        rows.append(
            _build_raw_row(
                seed=seed,
                policy_name=(
                    mean_field_result
                    .policy_name
                ),
                metrics=(
                    mean_field_result
                    .metrics
                ),
            )
        )

        for policy_result in (
            baseline_result,
            mean_field_result,
        ):
            for record in (
                policy_result
                .selection_records
            ):
                selection_rows.append(
                    {
                        "seed": int(
                            seed
                        ),

                        "policy_name": (
                            policy_result
                            .policy_name
                        ),

                        **record,
                    }
                )

            for record in (
                policy_result
                .decision_records
            ):
                decision_rows.append(
                    {
                        "seed": int(
                            seed
                        ),

                        **record,
                    }
                )

    raw_results = (
        pd.DataFrame(
            rows
        )
        .sort_values(
            [
                "seed",
                "policy_name",
            ]
        )
        .reset_index(
            drop=True
        )
    )

    summary = (
        aggregate_policy_metrics(
            raw_results
        )
    )

    paired = (
        paired_policy_comparison(
            raw_results
        )
    )

    selection_records = (
        pd.DataFrame(
            selection_rows,
            columns=[
                "seed",
                "policy_name",
                "task_id",
                "priority",
                "node_id",
                "cpu_capacity",
                "load_ratio",
                "queue_length",
            ],
        )
    )

    selection_summary = (
        policy_selection_summary(
            selection_records
        )
    )

    (
        selection_comparison,
        node_selection_frequency,
    ) = paired_selection_analysis(
        selection_records
    )

    decision_records = (
        pd.DataFrame(
            decision_rows
        )
    )

    decision_audit = (
        decision_audit_summary(
            decision_records
        )
    )

    decision_comparison = (
        paired_decision_audit(
            decision_records
        )
    )

    result = (
        RepeatedExperimentResult(
            raw_results=(
                raw_results
            ),

            summary=(
                summary
            ),

            paired_comparison=(
                paired
            ),

            selection_summary=(
                selection_summary
            ),

            selection_comparison=(
                selection_comparison
            ),

            node_selection_frequency=(
                node_selection_frequency
            ),

            decision_audit=(
                decision_audit
            ),

            decision_comparison=(
                decision_comparison
            ),

            equilibrium_diagnostics=(
                diagnostics
            ),
        )
    )

    if (
        output_directory
        is not None
    ):
        save_repeated_experiment_result(
            result=result,
            output_directory=(
                output_directory
            ),
        )

    return result


def run_repeated_experiments(
    config: SimulationConfig,
    seeds: (
        list[int]
        | tuple[int, ...]
    ),
    output_directory: (
        str
        | Path
        | None
    ) = None,
    scenario: str = "default",
) -> RepeatedExperimentResult:
    """Run paired baseline and Mean-Field experiments for one scenario."""

    normalized_seeds = tuple(
        int(seed)
        for seed in seeds
    )

    if not normalized_seeds:
        raise ValueError(
            "At least one experiment "
            "seed is required."
        )

    if (
        len(
            set(
                normalized_seeds
            )
        )
        != len(
            normalized_seeds
        )
    ):
        raise ValueError(
            "Experiment seeds "
            "must be unique."
        )

    (
        mean_field_policy,
        diagnostics,
    ) = build_mean_field_policy(
        config
    )

    return (
        _run_repeated_with_policy(
            config=config,
            seeds=(
                normalized_seeds
            ),
            scenario=scenario,
            mean_field_policy=(
                mean_field_policy
            ),
            diagnostics=(
                diagnostics
            ),
            output_directory=(
                output_directory
            ),
        )
    )


def run_workload_scenarios(
    config: SimulationConfig,
    seeds: (
        list[int]
        | tuple[int, ...]
    ),
    output_directory: (
        str
        | Path
        | None
    ) = None,
) -> dict[
    str,
    RepeatedExperimentResult,
]:
    """Run all configured workload scenarios using one MFG equilibrium."""

    normalized_seeds = tuple(
        int(seed)
        for seed in seeds
    )

    if not normalized_seeds:
        raise ValueError(
            "At least one experiment "
            "seed is required."
        )

    if (
        len(
            set(
                normalized_seeds
            )
        )
        != len(
            normalized_seeds
        )
    ):
        raise ValueError(
            "Experiment seeds "
            "must be unique."
        )

    (
        mean_field_policy,
        diagnostics,
    ) = build_mean_field_policy(
        config
    )

    results: dict[
        str,
        RepeatedExperimentResult,
    ] = {}

    for scenario in (
        config.workload_scenarios
    ):
        scenario_output = None

        if (
            output_directory
            is not None
        ):
            scenario_output = (
                Path(
                    output_directory
                )
                / scenario
            )

        results[
            scenario
        ] = (
            _run_repeated_with_policy(
                config=config,
                seeds=(
                    normalized_seeds
                ),
                scenario=scenario,
                mean_field_policy=(
                    mean_field_policy
                ),
                diagnostics=(
                    diagnostics
                ),
                output_directory=(
                    scenario_output
                ),
            )
        )

    return results


def save_repeated_experiment_result(
    result: RepeatedExperimentResult,
    output_directory: (
        str | Path
    ),
) -> None:
    """Save repeated experiment results to CSV files."""

    output_path = Path(
        output_directory
    )

    raw_path = (
        output_path
        / "raw"
    )

    aggregated_path = (
        output_path
        / "aggregated"
    )

    raw_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    aggregated_path.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.raw_results.to_csv(
        raw_path
        / "policy_comparison_raw.csv",
        index=False,
    )

    result.summary.to_csv(
        aggregated_path
        / "policy_summary.csv",
        index=False,
    )

    result.paired_comparison.to_csv(
        aggregated_path
        / "paired_policy_comparison.csv",
        index=False,
    )

    result.selection_summary.to_csv(
        aggregated_path
        / "policy_selection_summary.csv",
        index=False,
    )

    result.selection_comparison.to_csv(
        aggregated_path
        / "paired_selection_comparison.csv",
        index=False,
    )

    result.node_selection_frequency.to_csv(
        aggregated_path
        / "node_selection_frequency.csv",
        index=False,
    )

    result.decision_audit.to_csv(
        aggregated_path
        / "decision_audit_summary.csv",
        index=False,
    )

    result.decision_comparison.to_csv(
        aggregated_path
        / "paired_decision_audit.csv",
        index=False,
    )

    pd.DataFrame(
        [
            result.equilibrium_diagnostics
        ]
    ).to_csv(
        aggregated_path
        / "equilibrium_diagnostics.csv",
        index=False,
    )