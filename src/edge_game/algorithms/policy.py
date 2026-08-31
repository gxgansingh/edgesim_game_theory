"""Task allocation policies."""

from dataclasses import dataclass, field

import numpy as np

from ..config import SimulationConfig
from ..entities.edge_node import EdgeNode
from ..entities.task import Task
from ..models.mean_field_model import MeanFieldModel
from .mean_field import MeanFieldEquilibriumResult
from .state import composite_state, composite_state_components


@dataclass
class BaselinePolicy:
    """Least-loaded baseline allocation policy."""

    config: SimulationConfig = field(
        default_factory=SimulationConfig
    )

    def _score(
        self,
        node: EdgeNode,
    ) -> float:
        """Return a scalar score consistent with the selection rule."""
        return float(
            node.load_ratio()
            + 1e-6 * node.queue_length
        )

    def select_node(
        self,
        task: Task,
        candidates: list[EdgeNode],
    ) -> EdgeNode | None:
        """Select the least-loaded feasible node."""
        if not candidates:
            return None

        return min(
            candidates,
            key=lambda node: (
                node.load_ratio(),
                node.queue_length,
                node.node_id,
            ),
        )

    def selection_diagnostics(
        self,
        task: Task,
        candidates: list[EdgeNode],
        selected_node: EdgeNode,
    ) -> dict:
        """Describe the baseline decision over the candidate set."""

        scores = {
            node.node_id: self._score(
                node
            )
            for node in candidates
        }

        ordered = sorted(
            candidates,
            key=lambda node: (
                node.load_ratio(),
                node.queue_length,
                node.node_id,
            ),
        )

        selected_score = scores[
            selected_node.node_id
        ]

        components = composite_state_components(
            node=selected_node,
            config=self.config,
        )

        return {
            "candidate_count": len(
                candidates
            ),

            "candidate_node_ids": ",".join(
                str(node.node_id)
                for node in candidates
            ),

            "candidate_scores": ",".join(
                (
                    f"{node.node_id}:"
                    f"{scores[node.node_id]:.12g}"
                )
                for node in candidates
            ),

            "selected_rank": (
                ordered.index(
                    selected_node
                )
                + 1
            ),

            "selected_score": (
                selected_score
            ),

            "best_score": min(
                scores.values()
            ),

            "worst_score": max(
                scores.values()
            ),

            "score_margin": (
                selected_score
                - min(
                    scores.values()
                )
            ),

            "score_tie_count": sum(
                abs(
                    score
                    - selected_score
                )
                <= 1e-12
                for score in scores.values()
            ),

            "state": float(
                selected_node.load_ratio()
            ),

            "control": np.nan,

            "mean_field_score": np.nan,

            "cpu_load": (
                components.cpu_load
            ),

            "memory_load": (
                components.memory_load
            ),

            "bandwidth_load": (
                components.bandwidth_load
            ),

            "queue_pressure": (
                components.queue_pressure
            ),

            "energy_pressure": (
                components.energy_pressure
            ),
        }


@dataclass
class MeanFieldPolicy:
    """Priority-aware Mean-Field allocation policy."""

    model: MeanFieldModel

    equilibrium: MeanFieldEquilibriumResult

    config: SimulationConfig = field(
        default_factory=SimulationConfig
    )

    def _state_value(
        self,
        node: EdgeNode,
    ) -> float:
        """Map heterogeneous node conditions into the MFG state."""

        return composite_state(
            node=node,
            config=self.config,
        )

    def _resolve_priority(
        self,
        priority: int,
    ) -> int:
        """Resolve a task priority to an available equilibrium policy."""

        available_priorities = sorted(
            self.equilibrium.policies
        )

        if not available_priorities:
            raise ValueError(
                "No Mean-Field policies are available."
            )

        if priority in (
            self.equilibrium.policies
        ):
            return priority

        return min(
            available_priorities,
            key=lambda available_priority: abs(
                available_priority
                - priority
            ),
        )

    def _policy_values(
        self,
        priority: int,
    ) -> np.ndarray:
        """Return the numerical control values for a priority."""

        resolved_priority = (
            self._resolve_priority(
                priority
            )
        )

        policy_entry = (
            self.equilibrium.policies[
                resolved_priority
            ]
        )

        if hasattr(
            policy_entry,
            "control",
        ):
            values = (
                policy_entry.control
            )
        else:
            values = (
                policy_entry
            )

        return np.asarray(
            values,
            dtype=float,
        )

    def _resolve_state_grid(
        self,
        priority: int,
        policy_values: np.ndarray,
    ) -> np.ndarray:
        """Resolve the state grid used by the equilibrium policy."""

        resolved_priority = (
            self._resolve_priority(
                priority
            )
        )

        if hasattr(
            self.equilibrium,
            "state_grid",
        ):
            state_grid = getattr(
                self.equilibrium,
                "state_grid",
            )

            if state_grid is not None:
                state_grid_array = np.asarray(
                    state_grid,
                    dtype=float,
                )

                if len(
                    state_grid_array
                ) == len(
                    policy_values
                ):
                    return state_grid_array

        if hasattr(
            self.equilibrium,
            "populations",
        ):
            populations = getattr(
                self.equilibrium,
                "populations",
            )

            if populations is not None:
                population = (
                    populations.get(
                        resolved_priority
                    )
                )

                if (
                    population is not None
                    and hasattr(
                        population,
                        "state_grid",
                    )
                ):
                    state_grid_array = np.asarray(
                        population.state_grid,
                        dtype=float,
                    )

                    if len(
                        state_grid_array
                    ) == len(
                        policy_values
                    ):
                        return state_grid_array

        return np.linspace(
            0.0,
            1.0,
            len(
                policy_values
            ),
        )

    def _control_value(
        self,
        priority: int,
        state: float,
    ) -> float:
        """Interpolate the equilibrium control for a state."""

        policy_values = (
            self._policy_values(
                priority
            )
        )

        if len(
            policy_values
        ) == 0:
            raise ValueError(
                "The Mean-Field policy contains no control values."
            )

        state_grid = (
            self._resolve_state_grid(
                priority=priority,
                policy_values=policy_values,
            )
        )

        clipped_state = float(
            np.clip(
                state,
                float(
                    np.min(
                        state_grid
                    )
                ),
                float(
                    np.max(
                        state_grid
                    )
                ),
            )
        )

        return float(
            np.interp(
                clipped_state,
                state_grid,
                policy_values,
            )
        )

    def _mean_field_state(
        self,
    ) -> float:
        """Return the aggregate equilibrium mean state."""

        distribution = getattr(
            self.equilibrium,
            "distribution",
            None,
        )

        if (
            distribution is not None
            and hasattr(
                distribution,
                "aggregate_mean_state",
            )
        ):
            return float(
                distribution.aggregate_mean_state()
            )

        return 0.0

    def _score(
        self,
        task: Task,
        node: EdgeNode,
    ) -> float:
        """Calculate the Mean-Field node-selection score."""

        state = self._state_value(
            node
        )

        control = self._control_value(
            priority=task.priority,
            state=state,
        )

        mean_field = (
            self._mean_field_state()
        )

        return float(
            self.model.running_cost(
                priority=task.priority,
                state=state,
                control=control,
                mean_field=mean_field,
            )
        )

    def select_node(
        self,
        task: Task,
        candidates: list[EdgeNode],
    ) -> EdgeNode | None:
        """Select the feasible node with minimum MFG cost."""

        if not candidates:
            return None

        return min(
            candidates,
            key=lambda node: (
                self._score(
                    task=task,
                    node=node,
                ),
                node.node_id,
            ),
        )

    def selection_diagnostics(
        self,
        task: Task,
        candidates: list[EdgeNode],
        selected_node: EdgeNode,
    ) -> dict:
        """Describe the MFG decision over the candidate set."""

        scores: dict[
            int,
            float,
        ] = {}

        states: dict[
            int,
            float,
        ] = {}

        controls: dict[
            int,
            float,
        ] = {}

        mean_field = (
            self._mean_field_state()
        )

        for node in candidates:
            state = self._state_value(
                node
            )

            control = (
                self._control_value(
                    priority=task.priority,
                    state=state,
                )
            )

            score = (
                self.model.running_cost(
                    priority=task.priority,
                    state=state,
                    control=control,
                    mean_field=mean_field,
                )
            )

            states[
                node.node_id
            ] = float(
                state
            )

            controls[
                node.node_id
            ] = float(
                control
            )

            scores[
                node.node_id
            ] = float(
                score
            )

        ordered = sorted(
            candidates,
            key=lambda node: (
                scores[
                    node.node_id
                ],
                node.node_id,
            ),
        )

        selected_score = scores[
            selected_node.node_id
        ]

        selected_state = states[
            selected_node.node_id
        ]

        selected_control = controls[
            selected_node.node_id
        ]

        components = (
            composite_state_components(
                node=selected_node,
                config=self.config,
            )
        )

        return {
            "candidate_count": len(
                candidates
            ),

            "candidate_node_ids": ",".join(
                str(
                    node.node_id
                )
                for node in candidates
            ),

            "candidate_scores": ",".join(
                (
                    f"{node.node_id}:"
                    f"{scores[node.node_id]:.12g}"
                )
                for node in candidates
            ),

            "candidate_states": ",".join(
                (
                    f"{node.node_id}:"
                    f"{states[node.node_id]:.12g}"
                )
                for node in candidates
            ),

            "candidate_controls": ",".join(
                (
                    f"{node.node_id}:"
                    f"{controls[node.node_id]:.12g}"
                )
                for node in candidates
            ),

            "candidate_mean_field_scores": ",".join(
                (
                    f"{node.node_id}:"
                    f"{scores[node.node_id]:.12g}"
                )
                for node in candidates
            ),

            "selected_rank": (
                ordered.index(
                    selected_node
                )
                + 1
            ),

            "selected_score": (
                selected_score
            ),

            "best_score": min(
                scores.values()
            ),

            "worst_score": max(
                scores.values()
            ),

            "score_margin": (
                selected_score
                - min(
                    scores.values()
                )
            ),

            "score_tie_count": sum(
                abs(
                    score
                    - selected_score
                )
                <= 1e-12
                for score in scores.values()
            ),

            "state": selected_state,

            "control": selected_control,

            "mean_field_score": (
                selected_score
            ),

            "cpu_load": (
                components.cpu_load
            ),

            "memory_load": (
                components.memory_load
            ),

            "bandwidth_load": (
                components.bandwidth_load
            ),

            "queue_pressure": (
                components.queue_pressure
            ),

            "energy_pressure": (
                components.energy_pressure
            ),
        }


@dataclass
class HierarchicalPolicy:
    """Hierarchical server-then-node allocation policy."""

    node_policy: object

    config: SimulationConfig = field(
        default_factory=SimulationConfig
    )

    def _server_score(
        self,
        server,
    ) -> tuple[
        float,
        int,
        int,
    ]:
        """Return the server-level load-balancing score."""

        average_load = (
            server.average_load()
        )

        total_queue = sum(
            node.queue_length
            for node in server.nodes
        )

        return (
            average_load,
            total_queue,
            server.server_id,
        )

    def select_server(
        self,
        task: Task,
        candidates,
    ):
        """Select the least-loaded feasible server."""

        if not candidates:
            return None

        return min(
            candidates,
            key=self._server_score,
        )

    def select_node(
        self,
        task: Task,
        candidates: list[EdgeNode],
    ) -> EdgeNode | None:
        """Select a node using the configured node-level policy."""

        if not candidates:
            return None

        return (
            self.node_policy.select_node(
                task=task,
                candidates=candidates,
            )
        )

    def selection_diagnostics(
        self,
        task: Task,
        candidates: list[EdgeNode],
        selected_node: EdgeNode,
    ) -> dict:
        """Return node-level diagnostics from the wrapped policy."""

        return (
            self.node_policy.selection_diagnostics(
                task=task,
                candidates=candidates,
                selected_node=selected_node,
            )
        )

    def server_selection_diagnostics(
        self,
        task: Task,
        candidates,
        selected_server,
    ) -> dict:
        """Return diagnostics for the server-level decision."""

        scores = {
            server.server_id: (
                self._server_score(
                    server
                )
            )
            for server in candidates
        }

        ordered = sorted(
            candidates,
            key=self._server_score,
        )

        selected_score = scores[
            selected_server.server_id
        ]

        return {
            "server_candidate_count": len(
                candidates
            ),

            "server_candidate_ids": ",".join(
                str(
                    server.server_id
                )
                for server in candidates
            ),

            "server_candidate_scores": ",".join(
                (
                    f"{server.server_id}:"
                    f"{scores[server.server_id][0]:.12g}"
                )
                for server in candidates
            ),

            "selected_server_id": int(
                selected_server.server_id
            ),

            "selected_server_rank": (
                ordered.index(
                    selected_server
                )
                + 1
            ),

            "selected_server_score": float(
                selected_score[0]
            ),

            "selected_server_queue": int(
                selected_score[1]
            ),

            "best_server_score": float(
                min(
                    score[0]
                    for score in scores.values()
                )
            ),

            "worst_server_score": float(
                max(
                    score[0]
                    for score in scores.values()
                )
            ),
        }