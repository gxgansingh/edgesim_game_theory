"""Local-state mapping for the one-dimensional Mean-Field model."""

from dataclasses import dataclass

import numpy as np

from ..config import SimulationConfig
from ..entities.edge_node import EdgeNode


@dataclass(frozen=True)
class CompositeStateComponents:
    """Normalized local node-state components used by the MFG mapping."""

    cpu_load: float
    memory_load: float
    bandwidth_load: float
    queue_pressure: float
    energy_pressure: float

    def clipped(self) -> "CompositeStateComponents":
        """Return components clipped to the normalized state range."""
        return CompositeStateComponents(
            cpu_load=float(np.clip(self.cpu_load, 0.0, 1.0)),
            memory_load=float(np.clip(self.memory_load, 0.0, 1.0)),
            bandwidth_load=float(
                np.clip(self.bandwidth_load, 0.0, 1.0)
            ),
            queue_pressure=float(
                np.clip(self.queue_pressure, 0.0, 1.0)
            ),
            energy_pressure=float(
                np.clip(self.energy_pressure, 0.0, 1.0)
            ),
        )


def cpu_load(node: EdgeNode) -> float:
    """Return normalized CPU load."""
    return float(np.clip(node.load_ratio(), 0.0, 1.0))


def memory_load(node: EdgeNode) -> float:
    """Return normalized memory load."""
    return float(np.clip(node.memory_load_ratio(), 0.0, 1.0))


def bandwidth_load(node: EdgeNode) -> float:
    """Return normalized bandwidth load."""
    return float(np.clip(node.bandwidth_load_ratio(), 0.0, 1.0))


def queue_pressure(
    node: EdgeNode,
    config: SimulationConfig,
) -> float:
    """Normalize queue length against the configured queue scale."""
    if config.mean_field_queue_normalization <= 0.0:
        raise ValueError(
            "mean_field_queue_normalization must be positive."
        )

    return float(
        np.clip(
            node.queue_length
            / config.mean_field_queue_normalization,
            0.0,
            1.0,
        )
    )


def energy_pressure(node: EdgeNode) -> float:
    """Estimate normalized workload-related energy pressure.

    The simulator does not yet model physical energy consumption. Until an
    explicit energy accounting model is introduced, active-task energy
    budgets provide a deterministic workload-pressure proxy.
    """
    if node.energy_capacity <= 0.0:
        return 1.0

    active_energy = sum(
        max(task.energy_budget, 0.0)
        for task in node.active_tasks
    )

    return float(
        np.clip(
            active_energy / node.energy_capacity,
            0.0,
            1.0,
        )
    )


def composite_state_components(
    node: EdgeNode,
    config: SimulationConfig,
) -> CompositeStateComponents:
    """Build normalized local state components for one edge node."""
    return CompositeStateComponents(
        cpu_load=cpu_load(node),
        memory_load=memory_load(node),
        bandwidth_load=bandwidth_load(node),
        queue_pressure=queue_pressure(
            node=node,
            config=config,
        ),
        energy_pressure=energy_pressure(node),
    ).clipped()


def composite_state(
    node: EdgeNode,
    config: SimulationConfig,
) -> float:
    """Map heterogeneous local state to the existing 1D MFG state."""
    components = composite_state_components(
        node=node,
        config=config,
    )

    weights = np.asarray(
        [
            config.mean_field_cpu_state_weight,
            config.mean_field_memory_state_weight,
            config.mean_field_bandwidth_state_weight,
            config.mean_field_queue_state_weight,
            config.mean_field_energy_state_weight,
        ],
        dtype=float,
    )

    if np.any(weights < 0.0):
        raise ValueError(
            "Mean-Field state weights must be non-negative."
        )

    weight_sum = float(np.sum(weights))
    if weight_sum <= 1e-12:
        raise ValueError(
            "At least one Mean-Field state weight must be positive."
        )

    normalized_weights = weights / weight_sum
    values = np.asarray(
        [
            components.cpu_load,
            components.memory_load,
            components.bandwidth_load,
            components.queue_pressure,
            components.energy_pressure,
        ],
        dtype=float,
    )

    return float(
        np.clip(
            np.dot(normalized_weights, values),
            0.0,
            1.0,
        )
    )
