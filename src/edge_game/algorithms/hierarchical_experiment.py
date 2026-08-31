"""Hierarchical server-then-node experiment execution."""

from dataclasses import dataclass

from ..algorithms.candidate_filter import (
    filter_feasible_nodes,
    filter_feasible_servers,
)
from ..config import SimulationConfig
from ..environment import SimulationEnvironment
from ..metrics.collector import MetricCollector


@dataclass
class HierarchicalExperimentResult:
    """Store the result of one hierarchical experiment."""

    policy_name: str
    metrics: dict
    selection_records: list[dict]
    decision_records: list[dict]
    server_decision_records: list[dict]


def run_hierarchical_policy_experiment(
    config: SimulationConfig,
    policy_name: str,
    policy,
) -> HierarchicalExperimentResult:
    """Run hierarchical server-then-node workload allocation."""
    environment = SimulationEnvironment(
        config=config
    )

    metrics = MetricCollector()

    servers = environment.servers
    nodes = environment.all_nodes()

    next_task_id = 0
    server_decision_records: list[dict] = []

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

            server_candidates = (
                filter_feasible_servers(
                    task=task,
                    servers=servers,
                )
            )

            if not server_candidates:
                metrics.record_rejection(
                    task=task
                )
                continue

            selected_server = (
                policy.select_server(
                    task=task,
                    candidates=server_candidates,
                )
            )

            if selected_server is None:
                metrics.record_rejection(
                    task=task
                )
                continue

            server_diagnostics = (
                policy.server_selection_diagnostics(
                    task=task,
                    candidates=server_candidates,
                    selected_server=selected_server,
                )
            )

            server_decision_records.append(
                {
                    "task_id": int(task.task_id),
                    "priority": int(task.priority),
                    "policy_name": policy_name,
                    **server_diagnostics,
                }
            )

            node_candidates = (
                filter_feasible_nodes(
                    task=task,
                    nodes=selected_server.nodes,
                )
            )

            if not node_candidates:
                metrics.record_rejection(
                    task=task
                )
                continue

            selected_node = (
                policy.select_node(
                    task=task,
                    candidates=node_candidates,
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
                candidates=node_candidates,
                selected_node=selected_node,
                diagnostics=policy.selection_diagnostics(
                    task=task,
                    candidates=node_candidates,
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

    return HierarchicalExperimentResult(
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
        server_decision_records=(
            server_decision_records
        ),
    )