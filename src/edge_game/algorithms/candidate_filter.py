"""Resource-aware feasibility filtering."""

from typing import Iterable, List

from ..entities.edge_node import EdgeNode
from ..entities.edge_server import EdgeServer
from ..entities.task import Task


def filter_feasible_nodes(
    task: Task,
    nodes: Iterable[EdgeNode],
) -> List[EdgeNode]:
    """Return nodes that satisfy the task's minimum resource requirements."""
    return [
        node
        for node in nodes
        if node.can_process(task)
    ]


def filter_feasible_servers(
    task: Task,
    servers: Iterable[EdgeServer],
) -> List[EdgeServer]:
    """Return servers containing at least one feasible node."""
    return [
        server
        for server in servers
        if any(
            node.can_process(task)
            for node in server.nodes
        )
    ]