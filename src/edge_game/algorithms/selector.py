"""Utility-based workload selection policy."""

import random

from ..entities.edge_node import EdgeNode
from .utility import utility_probabilities


def select_node(
    nodes: list[EdgeNode],
    rng: random.Random | None = None,
) -> EdgeNode:
    """Select an edge node using utility-based probability."""
    if not nodes:
        raise ValueError("Cannot select a node from an empty collection.")

    generator = rng if rng is not None else random.Random()

    probabilities = utility_probabilities(nodes)

    return generator.choices(
        nodes,
        weights=probabilities,
        k=1,
    )[0]