"""Utility functions for edge-node workload allocation."""

from dataclasses import dataclass

from ..config import SimulationConfig
from ..entities.edge_node import EdgeNode
from ..entities.task import Task


@dataclass(frozen=True)
class UtilityWeights:
    """Weights for the legacy implementation-level utility score."""

    resource: float = 1.0
    latency: float = 1.0
    energy: float = 1.0
    queue: float = 1.0
    priority: float = 1.0


@dataclass(frozen=True)
class OutcomeUtilityWeights:
    """Weights for the action-to-outcome research utility mapping."""

    priority_reward: float = 1.0
    latency_cost: float = 1.0
    resource_cost: float = 1.0
    queue_cost: float = 1.0
    energy_cost: float = 1.0


def calculate_utility(
    task: Task,
    node: EdgeNode,
    weights: UtilityWeights = UtilityWeights(),
) -> float:
    """Calculate the legacy pre-allocation utility score.

    This function is retained for backward compatibility with the foundation
    tests and existing simulator components. Final research experiments use
    ``calculate_outcome_utility`` so utility depends on the realized task
    outcome rather than only the pre-allocation node state.
    """
    cpu_headroom = node.available_cpu / max(node.cpu_capacity, 1e-12)
    memory_headroom = node.available_memory / max(
        node.memory_capacity,
        1e-12,
    )
    bandwidth_headroom = node.available_bandwidth / max(
        node.bandwidth_capacity,
        1e-12,
    )

    resource_score = (
        cpu_headroom
        + memory_headroom
        + bandwidth_headroom
    ) / 3.0

    queue_score = 1.0 / (1.0 + node.queue_length)
    priority_score = task.priority / 3.0

    estimated_latency = 1.0 + node.queue_length
    latency_score = 1.0 / max(estimated_latency, 1.0)

    energy_score = max(
        0.0,
        min(
            1.0,
            node.energy_capacity
            / max(
                node.energy_capacity + task.energy_budget,
                1e-12,
            ),
        ),
    )

    return (
        weights.resource * resource_score
        + weights.latency * latency_score
        + weights.energy * energy_score
        + weights.queue * queue_score
        + weights.priority * priority_score
    )


def calculate_outcome_utility(
    task: Task,
    response_time: float,
    node_cpu_capacity: float,
    node_memory_capacity: float,
    node_bandwidth_capacity: float,
    queue_length_at_assignment: int,
    node_energy_capacity: float,
    config: SimulationConfig,
) -> float:
    """Calculate utility from the realized task outcome.

    The mapping follows the project objective: reward higher-priority service
    while penalizing latency, resource consumption, queue pressure, and energy
    demand. Numerical weights are explicit configuration parameters so they can
    be calibrated to the final research formulation later.
    """
    weights = OutcomeUtilityWeights(
        priority_reward=(
            config.utility_priority_reward_weight
        ),
        latency_cost=(
            config.utility_latency_cost_weight
        ),
        resource_cost=(
            config.utility_resource_cost_weight
        ),
        queue_cost=(
            config.utility_queue_cost_weight
        ),
        energy_cost=(
            config.utility_energy_cost_weight
        ),
    )

    priority_reward = task.priority / 3.0

    latency_ratio = (
        response_time
        / max(task.latency_requirement, 1e-12)
    )
    latency_penalty = min(
        max(latency_ratio, 0.0),
        2.0,
    )

    resource_cost = (
        (
            task.cpu_demand
            / max(node_cpu_capacity, 1e-12)
        )
        + (
            task.memory_demand
            / max(node_memory_capacity, 1e-12)
        )
        + (
            task.bandwidth_demand
            / max(node_bandwidth_capacity, 1e-12)
        )
    ) / 3.0

    queue_cost = min(
        max(
            queue_length_at_assignment
            / max(
                config.utility_queue_normalization,
                1e-12,
            ),
            0.0,
        ),
        1.0,
    )

    energy_cost = min(
        max(
            task.energy_budget
            / max(node_energy_capacity, 1e-12),
            0.0,
        ),
        1.0,
    )

    return float(
        weights.priority_reward * priority_reward
        - weights.latency_cost * latency_penalty
        - weights.resource_cost * resource_cost
        - weights.queue_cost * queue_cost
        - weights.energy_cost * energy_cost
    )


def node_utility(
    node: EdgeNode,
    utility_max: float = 100.0,
) -> float:
    """Calculate load-based node utility.

    Higher utility represents lower current node utilization.
    """
    load_percentage = node.load_ratio() * 100.0

    return max(
        0.0,
        utility_max - load_percentage,
    )


def utility_probabilities(
    nodes: list[EdgeNode],
    utility_max: float = 100.0,
    epsilon: float = 1e-12,
) -> list[float]:
    """Convert node utilities into a normalized probability distribution."""
    if not nodes:
        return []

    utilities = [
        node_utility(
            node=node,
            utility_max=utility_max,
        )
        for node in nodes
    ]

    total_utility = sum(utilities)

    if total_utility <= epsilon:
        probability = 1.0 / len(nodes)
        return [probability] * len(nodes)

    return [
        utility / total_utility
        for utility in utilities
    ]
