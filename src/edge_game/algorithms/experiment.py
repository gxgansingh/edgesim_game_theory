"""Experiment execution utilities."""

from dataclasses import dataclass
from typing import Callable

from ..algorithms.candidate_filter import (
    filter_feasible_nodes,
)
from ..algorithms.utility import (
    calculate_utility,
)
from ..config import SimulationConfig
from ..environment import SimulationEnvironment
from ..metrics.collector import MetricCollector
from ..entities.task import Task


@dataclass
class ExperimentResult:
    """Store the result of one policy experiment."""

    policy_name: str
    metrics: dict
    selection_records: list[dict]
    decision_records: list[dict]


def run_policy_experiment(
    config: SimulationConfig,
    policy_name: str,
    policy,
) -> ExperimentResult:
    """Run the dynamic workload simulation using one policy."""
    environment = SimulationEnvironment(
        config=config
    )

    metrics = MetricCollector()

    nodes = environment.all_nodes()

    next_task_id = 0

    for _ in range(
        config.simulation_steps
    ):
        tasks = environment.create_tasks(
            starting_task_id=next_task_id
        )

        next_task_id += len(tasks)

        for task in tasks:
            metrics.record_arrival(
                task=task
            )

            candidates = (
                filter_feasible_nodes(
                    task=task,
                    nodes=nodes,
                )
            )

            if not candidates:
                metrics.record_rejection(
                    task=task
                )
                continue

            selected_node = (
                policy.select_node(
                    task=task,
                    candidates=candidates,
                )
            )

            if selected_node is None:
                metrics.record_rejection(
                    task=task
                )
                continue

            metrics.record_selection(
                task=task,
                node=selected_node,
            )

            metrics.record_selection_diagnostics(
                task=task,
                policy_name=policy_name,
                candidates=candidates,
                selected_node=selected_node,
                diagnostics=policy.selection_diagnostics(
                    task=task,
                    candidates=candidates,
                    selected_node=selected_node,
                ),
            )

            metrics.record_allocation(
                task=task,
                node=selected_node,
            )

            selected_node.allocate_task(
                task=task,
                current_time=(
                    environment.current_time
                ),
            )

        metrics.record_state(
            nodes=nodes
        )

        completed_tasks = (
            environment.advance_time()
        )

        for task in completed_tasks:
            metrics.record_completion(
                task=task,
                config=config,
            )

    return ExperimentResult(
        policy_name=policy_name,
        metrics=metrics.summary(
            nodes=nodes
        ),
        selection_records=list(
            metrics.selection_records
        ),
        decision_records=list(
            metrics.decision_records
        ),
    )