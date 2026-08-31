"""Simulation environment for the edge computing architecture."""

from dataclasses import dataclass
import random
from typing import List

from .config import SimulationConfig
from .entities.edge_node import EdgeNode
from .entities.edge_server import EdgeServer
from .entities.gateway import EdgeGateway
from .entities.task import Task


@dataclass
class SimulationEnvironment:
    """Create and manage the edge-computing environment."""

    config: SimulationConfig

    def __post_init__(
        self,
    ) -> None:
        """Initialize the simulation environment."""
        self.random = random.Random(
            self.config.seed
        )

        self.current_time = 0.0

        self.gateway = EdgeGateway(
            gateway_id=0
        )

        self.servers = (
            self._create_servers()
        )

    def _create_servers(
        self,
    ) -> List[EdgeServer]:
        """Create heterogeneous edge servers and nodes."""
        servers: List[EdgeServer] = []

        for server_id in range(
            self.config.number_of_servers
        ):
            server = EdgeServer(
                server_id=server_id
            )

            for local_node_id in range(
                self.config.nodes_per_server
            ):
                node_id = (
                    server_id
                    * self.config.nodes_per_server
                    + local_node_id
                )

                cpu = self.random.uniform(
                    self.config.minimum_cpu_capacity,
                    self.config.maximum_cpu_capacity,
                )

                memory = self.random.uniform(
                    self.config.minimum_memory_capacity,
                    self.config.maximum_memory_capacity,
                )

                bandwidth = self.random.uniform(
                    self.config.minimum_bandwidth,
                    self.config.maximum_bandwidth,
                )

                node = EdgeNode(
                    node_id=node_id,
                    server_id=server_id,
                    cpu_capacity=cpu,
                    memory_capacity=memory,
                    bandwidth_capacity=bandwidth,
                    energy_capacity=100.0,
                )

                server.add_node(
                    node
                )

            servers.append(
                server
            )

        return servers

    def all_nodes(
        self,
    ) -> List[EdgeNode]:
        """Return every edge node."""
        return [
            node
            for server in self.servers
            for node in server.nodes
        ]

    def create_task(
        self,
        task_id: int,
    ) -> Task:
        """Generate a synthetic task with four priority classes."""
        priority = self.random.choices(
            [4, 3, 2, 1],
            weights=[
                self.config.critical_priority_share,
                self.config.high_priority_share,
                self.config.medium_priority_share,
                self.config.low_priority_share,
            ],
            k=1,
        )[0]

        priority_metadata = {
            4: (
                "P1",
                4.0,
            ),
            3: (
                "P2",
                3.0,
            ),
            2: (
                "P3",
                2.0,
            ),
            1: (
                "P4",
                1.0,
            ),
        }

        priority_class, priority_score = (
            priority_metadata[
                priority
            ]
        )

        return Task(
            task_id=task_id,
            priority=priority,
            priority_class=priority_class,
            priority_score=priority_score,
            cpu_demand=self.random.uniform(
                self.config.minimum_cpu_demand,
                self.config.maximum_cpu_demand,
            ),
            memory_demand=self.random.uniform(
                self.config.minimum_memory_demand,
                self.config.maximum_memory_demand,
            ),
            bandwidth_demand=self.random.uniform(
                self.config.minimum_bandwidth_demand,
                self.config.maximum_bandwidth_demand,
            ),
            latency_requirement=self.random.uniform(
                self.config.minimum_latency_requirement,
                self.config.maximum_latency_requirement,
            ),
            energy_budget=self.random.uniform(
                self.config.minimum_energy_budget,
                self.config.maximum_energy_budget,
            ),
            workload_size=self.random.uniform(
                self.config.minimum_workload_size,
                self.config.maximum_workload_size,
            ),
            arrival_time=self.current_time,
        )

    def create_tasks(
        self,
        starting_task_id: int,
    ) -> list[Task]:
        """Generate multiple tasks for the current simulation step."""
        return [
            self.create_task(
                task_id=(
                    starting_task_id
                    + offset
                )
            )
            for offset in range(
                self.config.tasks_per_step
            )
        ]

    def advance_time(
        self,
    ) -> list[Task]:
        """Advance the simulation by one time step."""
        completed_tasks: list[Task] = []

        next_time = (
            self.current_time
            + self.config.simulation_time_step
        )

        for node in self.all_nodes():
            completed = (
                node.process_tasks(
                    time_step=(
                        self.config.simulation_time_step
                    ),
                    cpu_rate=(
                        self.config.task_cpu_rate
                    ),
                    reference_cpu_capacity=(
                        self.config.reference_cpu_capacity
                    ),
                    critical_priority_weight=(
                        self.config
                        .priority_service_weight_critical
                    ),
                    high_priority_weight=(
                        self.config
                        .priority_service_weight_high
                    ),
                    medium_priority_weight=(
                        self.config
                        .priority_service_weight_medium
                    ),
                    low_priority_weight=(
                        self.config
                        .priority_service_weight_low
                    ),
                )
            )

            for task in completed:
                node.release_task(
                    task=task,
                    current_time=next_time,
                )

            completed_tasks.extend(
                completed
            )

        self.current_time = (
            next_time
        )

        return completed_tasks