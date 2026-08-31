"""Edge-server model."""

from dataclasses import dataclass, field
from typing import List

from .edge_node import EdgeNode


@dataclass
class EdgeServer:
    """Represents an edge server managing a cluster of edge nodes."""

    server_id: int
    nodes: List[EdgeNode] = field(default_factory=list)
    admission_capacity: float = 1.0

    def add_node(self, node: EdgeNode) -> None:
        """Add a node to the server-managed cluster."""
        self.nodes.append(node)

    def average_load(self) -> float:
        """Return the mean CPU load across managed nodes."""
        if not self.nodes:
            return 0.0
        return sum(node.load_ratio() for node in self.nodes) / len(self.nodes)
