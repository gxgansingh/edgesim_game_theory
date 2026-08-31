"""Edge ingress gateway model."""

from dataclasses import dataclass
from .task import Task


@dataclass
class EdgeGateway:
    """Represents the lightweight ingress point for IoT traffic."""

    gateway_id: int

    def forward(self, task: Task) -> Task:
        """Forward an incoming task to the associated edge server."""
        return task
