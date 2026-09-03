"""Priority-aware Mean-Field Game mathematical formulation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ABLATION_VARIANTS = {
    "full",
    "no_priority",
    "no_priority_reward",
    "no_latency",
    "no_resource",
    "no_queue",
    "no_energy",
}


@dataclass(frozen=True)
class PriorityParameters:
    """Parameters associated with one workload priority class."""

    arrival_rate: float
    congestion_weight: float
    processing_weight: float
    control_weight: float


@dataclass(frozen=True)
class MeanFieldParameters:
    """Global parameters for the Mean-Field Game."""

    congestion_coupling: float = 0.20
    processing_rate: float = 0.60
    congestion_decay: float = 0.30

    diffusion: float = 0.01
    discount_factor: float = 0.95

    priority_1: PriorityParameters = PriorityParameters(
        arrival_rate=0.15,
        congestion_weight=1.0,
        processing_weight=0.8,
        control_weight=0.20,
    )

    priority_2: PriorityParameters = PriorityParameters(
        arrival_rate=0.20,
        congestion_weight=1.5,
        processing_weight=1.0,
        control_weight=0.20,
    )

    priority_3: PriorityParameters = PriorityParameters(
        arrival_rate=0.25,
        congestion_weight=2.5,
        processing_weight=1.4,
        control_weight=0.15,
    )

    priority_4: PriorityParameters = PriorityParameters(
        arrival_rate=0.30,
        congestion_weight=3.5,
        processing_weight=1.8,
        control_weight=0.10,
    )

    population_weights: tuple[
        float,
        float,
        float,
        float,
    ] = (
        0.40,
        0.30,
        0.20,
        0.10,
    )

    # Research utility weights used directly by the HJB objective.
    utility_priority_reward_weight: float = 1.0
    utility_latency_cost_weight: float = 1.0
    utility_resource_cost_weight: float = 1.0
    utility_queue_cost_weight: float = 1.0
    utility_energy_cost_weight: float = 1.0

    # Ablation configuration.
    ablation_variant: str = "full"

    def __post_init__(self) -> None:
        """Validate the configured parameters."""

        if self.ablation_variant not in ABLATION_VARIANTS:
            valid_variants = ", ".join(
                sorted(ABLATION_VARIANTS)
            )

            raise ValueError(
                f"Unsupported ablation variant: "
                f"{self.ablation_variant!r}. "
                f"Expected one of: {valid_variants}."
            )

        if len(self.population_weights) != 4:
            raise ValueError(
                "population_weights must contain exactly "
                "four values for priorities 1 through 4."
            )

        if sum(self.population_weights) <= 0.0:
            raise ValueError(
                "Population weights must have a positive total."
            )

    def priority_parameters(
        self,
        priority: int,
    ) -> PriorityParameters:
        """Return parameters for a priority class."""

        mapping = {
            1: self.priority_1,
            2: self.priority_2,
            3: self.priority_3,
            4: self.priority_4,
        }

        if priority not in mapping:
            raise ValueError(
                f"Unsupported priority class: {priority}."
            )

        parameters = mapping[priority]

        if self.ablation_variant != "no_priority":
            return parameters

        return self._priority_neutral_parameters(
            parameters
        )

    def _priority_neutral_parameters(
        self,
        parameters: PriorityParameters,
    ) -> PriorityParameters:
        """
        Remove priority-dependent service preferences.

        Arrival rates remain priority-specific because they describe
        workload population characteristics rather than policy preference.
        Service-related parameters are replaced by their population-weighted
        averages.
        """

        weights = self.population_weights

        priority_parameters = (
            self.priority_1,
            self.priority_2,
            self.priority_3,
            self.priority_4,
        )

        total_weight = float(
            sum(weights)
        )

        if total_weight <= 1e-12:
            raise ValueError(
                "Population weights must have a positive total."
            )

        neutral_congestion_weight = (
            sum(
                weight * item.congestion_weight
                for weight, item in zip(
                    weights,
                    priority_parameters,
                )
            )
            / total_weight
        )

        neutral_processing_weight = (
            sum(
                weight * item.processing_weight
                for weight, item in zip(
                    weights,
                    priority_parameters,
                )
            )
            / total_weight
        )

        neutral_control_weight = (
            sum(
                weight * item.control_weight
                for weight, item in zip(
                    weights,
                    priority_parameters,
                )
            )
            / total_weight
        )

        return PriorityParameters(
            arrival_rate=parameters.arrival_rate,
            congestion_weight=neutral_congestion_weight,
            processing_weight=neutral_processing_weight,
            control_weight=neutral_control_weight,
        )

    def population_weight(
        self,
        priority: int,
    ) -> float:
        """Return the configured population weight."""

        if priority not in (1, 2, 3, 4):
            raise ValueError(
                f"Unsupported priority class: {priority}."
            )

        return self.population_weights[
            priority - 1
        ]


class MeanFieldModel:
    """Implement the proposed reduced-order Mean-Field formulation."""

    def __init__(
        self,
        parameters: MeanFieldParameters | None = None,
    ) -> None:
        """Initialize the mathematical model."""

        self.parameters = (
            parameters
            if parameters is not None
            else MeanFieldParameters()
        )

    def _utility_weight(
        self,
        component: str,
    ) -> float:
        """Return the effective utility weight for an objective component."""

        variant = self.parameters.ablation_variant

        if component == "priority_reward":

            if variant == "no_priority_reward":
                return 0.0

            if variant == "no_priority":
                return 0.0

            return (
                self.parameters.utility_priority_reward_weight
            )

        if component == "latency":

            if variant == "no_latency":
                return 0.0

            return (
                self.parameters.utility_latency_cost_weight
            )

        if component == "resource":

            if variant == "no_resource":
                return 0.0

            return (
                self.parameters.utility_resource_cost_weight
            )

        if component == "queue":

            if variant == "no_queue":
                return 0.0

            return (
                self.parameters.utility_queue_cost_weight
            )

        if component == "energy":

            if variant == "no_energy":
                return 0.0

            return (
                self.parameters.utility_energy_cost_weight
            )

        raise ValueError(
            f"Unsupported utility component: {component}."
        )

    def drift(
        self,
        priority: int,
        state: float,
        control: float,
        mean_field: float,
    ) -> float:
        """Calculate the HJB/FPK state drift."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        state = float(
            np.clip(
                state,
                0.0,
                1.0,
            )
        )

        control = float(
            np.clip(
                control,
                0.0,
                1.0,
            )
        )

        mean_field = float(
            np.clip(
                mean_field,
                0.0,
                1.0,
            )
        )

        return (
            parameters.arrival_rate
            + self.parameters.congestion_coupling
            * mean_field
            - self.parameters.processing_rate
            * control
            - self.parameters.congestion_decay
            * state
        )

    def drift_array(
        self,
        priority: int,
        state: np.ndarray,
        control: np.ndarray,
        mean_field: np.ndarray,
    ) -> np.ndarray:
        """Calculate the HJB/FPK drift for vectorized state inputs."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        state = np.clip(
            np.asarray(
                state,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        control = np.clip(
            np.asarray(
                control,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        mean_field = np.clip(
            np.asarray(
                mean_field,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        return (
            parameters.arrival_rate
            + self.parameters.congestion_coupling
            * mean_field
            - self.parameters.processing_rate
            * control
            - self.parameters.congestion_decay
            * state
        )

    def running_cost_array(
        self,
        priority: int,
        state: np.ndarray,
        control: np.ndarray,
        mean_field: np.ndarray,
    ) -> np.ndarray:
        """Calculate the running cost for vectorized inputs."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        state = np.clip(
            np.asarray(
                state,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        control = np.clip(
            np.asarray(
                control,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        mean_field = np.clip(
            np.asarray(
                mean_field,
                dtype=float,
            ),
            0.0,
            1.0,
        )

        priority_reward = (
            self._utility_weight(
                "priority_reward"
            )
            * (
                parameters.processing_weight
                * priority
                / 4.0
            )
        )

        latency_cost = (
            self._utility_weight(
                "latency"
            )
            * parameters.congestion_weight
            * state**2
        )

        resource_cost = (
            self._utility_weight(
                "resource"
            )
            * parameters.processing_weight
            * (1.0 - control) ** 2
        )

        queue_cost = (
            self._utility_weight(
                "queue"
            )
            * self.parameters.congestion_coupling
            * mean_field**2
        )

        energy_cost = (
            self._utility_weight(
                "energy"
            )
            * parameters.control_weight
            * control**2
        )

        return (
            latency_cost
            + resource_cost
            + queue_cost
            + energy_cost
            - priority_reward
        )

    def optimal_control_array(
        self,
        priority: int,
        value_gradient: np.ndarray,
    ) -> np.ndarray:
        """Calculate the continuous HJB optimal control."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        value_gradient = np.asarray(
            value_gradient,
            dtype=float,
        )

        numerator = (
            parameters.processing_weight
            + (
                self.parameters.processing_rate
                * value_gradient
                / 2.0
            )
        )

        denominator = (
            parameters.processing_weight
            + parameters.control_weight
        )

        if denominator <= 1e-12:
            return np.zeros_like(
                value_gradient
            )

        return np.clip(
            numerator / denominator,
            0.0,
            1.0,
        )

    def hjb_residual_array(
        self,
        priority: int,
        state: np.ndarray,
        control: np.ndarray,
        mean_field: np.ndarray,
        value_gradient: np.ndarray,
        value_second_derivative: np.ndarray,
        value: np.ndarray,
    ) -> np.ndarray:
        """Calculate the stationary HJB residual for vectorized inputs."""

        cost = self.running_cost_array(
            priority=priority,
            state=state,
            control=control,
            mean_field=mean_field,
        )

        drift = self.drift_array(
            priority=priority,
            state=state,
            control=control,
            mean_field=mean_field,
        )

        diffusion_term = (
            0.5
            * self.parameters.diffusion
            * np.asarray(
                value_second_derivative,
                dtype=float,
            )
        )

        return (
            cost
            + drift
            * np.asarray(
                value_gradient,
                dtype=float,
            )
            + diffusion_term
            - self.parameters.discount_factor
            * np.asarray(
                value,
                dtype=float,
            )
        )

    def running_cost(
        self,
        priority: int,
        state: float,
        control: float,
        mean_field: float,
    ) -> float:
        """Calculate the priority-aware running cost."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        state = float(
            np.clip(
                state,
                0.0,
                1.0,
            )
        )

        control = float(
            np.clip(
                control,
                0.0,
                1.0,
            )
        )

        mean_field = float(
            np.clip(
                mean_field,
                0.0,
                1.0,
            )
        )

        priority_reward = (
            self._utility_weight(
                "priority_reward"
            )
            * (
                parameters.processing_weight
                * priority
                / 4.0
            )
        )

        latency_cost = (
            self._utility_weight(
                "latency"
            )
            * parameters.congestion_weight
            * state**2
        )

        resource_cost = (
            self._utility_weight(
                "resource"
            )
            * parameters.processing_weight
            * (1.0 - control) ** 2
        )

        queue_cost = (
            self._utility_weight(
                "queue"
            )
            * self.parameters.congestion_coupling
            * mean_field**2
        )

        energy_cost = (
            self._utility_weight(
                "energy"
            )
            * parameters.control_weight
            * control**2
        )

        return float(
            latency_cost
            + resource_cost
            + queue_cost
            + energy_cost
            - priority_reward
        )

    def optimal_control(
        self,
        priority: int,
        value_gradient: float,
    ) -> float:
        """Calculate the continuous HJB optimal control."""

        parameters = (
            self.parameters.priority_parameters(
                priority
            )
        )

        numerator = (
            parameters.processing_weight
            + (
                self.parameters.processing_rate
                * value_gradient
                / 2.0
            )
        )

        denominator = (
            parameters.processing_weight
            + parameters.control_weight
        )

        if denominator <= 1e-12:
            return 0.0

        return float(
            np.clip(
                numerator / denominator,
                0.0,
                1.0,
            )
        )

    def hamiltonian(
        self,
        priority: int,
        state: float,
        control: float,
        mean_field: float,
        value_gradient: float,
    ) -> float:
        """Calculate the HJB Hamiltonian."""

        cost = self.running_cost(
            priority=priority,
            state=state,
            control=control,
            mean_field=mean_field,
        )

        drift = self.drift(
            priority=priority,
            state=state,
            control=control,
            mean_field=mean_field,
        )

        return (
            cost
            + drift
            * value_gradient
        )

    def hjb_residual(
        self,
        priority: int,
        state: float,
        control: float,
        mean_field: float,
        value_gradient: float,
        value_second_derivative: float,
        value: float,
    ) -> float:
        """Calculate the stationary HJB residual."""

        hamiltonian = self.hamiltonian(
            priority=priority,
            state=state,
            control=control,
            mean_field=mean_field,
            value_gradient=value_gradient,
        )

        diffusion_term = (
            0.5
            * self.parameters.diffusion
            * value_second_derivative
        )

        return (
            hamiltonian
            + diffusion_term
            - self.parameters.discount_factor
            * value
        )