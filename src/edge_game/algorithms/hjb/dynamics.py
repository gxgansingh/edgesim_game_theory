"""State dynamics for the HJB formulation."""

from dataclasses import dataclass

import numpy as np

from .state import MeanFieldState


@dataclass(frozen=True)
class StateDynamics:
    """Represent the drift of the Mean-Field Game state."""

    load_rate: float = 0.10
    queue_arrival_rate: float = 0.10
    queue_service_rate: float = 0.20
    energy_consumption_rate: float = 0.05
    communication_recovery_rate: float = 0.10

    def drift(
        self,
        state: MeanFieldState,
        control: float,
        mean_field_load: float,
    ) -> np.ndarray:
        """Calculate the continuous-state drift.

        The returned drift is an implementation-level state transition
        interface. The exact research drift terms must be replaced with the
        finalized mathematical formulation before research experiments.
        """
        state = state.clipped()

        bounded_control = float(
            np.clip(control, 0.0, 1.0)
        )

        bounded_mean_field = float(
            np.clip(mean_field_load, 0.0, 1.0)
        )

        load_drift = (
            self.load_rate
            * (
                bounded_control
                + bounded_mean_field
                - state.cpu_load
            )
        )

        queue_drift = (
            self.queue_arrival_rate
            * bounded_mean_field
            - self.queue_service_rate
            * bounded_control
        )

        energy_drift = -(
            self.energy_consumption_rate
            * bounded_control
        )

        communication_drift = (
            self.communication_recovery_rate
            * (
                1.0
                - state.communication_quality
            )
            - 0.05 * bounded_mean_field
        )

        return np.array(
            [
                load_drift,
                queue_drift,
                energy_drift,
                communication_drift,
            ],
            dtype=float,
        )