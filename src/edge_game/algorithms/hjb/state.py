"""State representation for the Mean-Field Game."""

from dataclasses import dataclass


@dataclass(frozen=True)
class MeanFieldState:
    """Represent the local state of an edge-node workload population."""

    priority: int
    cpu_load: float
    queue_length: float
    energy_level: float
    communication_quality: float

    def clipped(self) -> "MeanFieldState":
        """Return a state with normalized variables clipped to valid ranges."""
        return MeanFieldState(
            priority=self.priority,
            cpu_load=min(max(self.cpu_load, 0.0), 1.0),
            queue_length=max(self.queue_length, 0.0),
            energy_level=min(max(self.energy_level, 0.0), 1.0),
            communication_quality=min(
                max(self.communication_quality, 0.0),
                1.0,
            ),
        )

    def as_vector(self) -> tuple[float, ...]:
        """Return the continuous state variables as a tuple."""
        clipped_state = self.clipped()

        return (
            clipped_state.cpu_load,
            clipped_state.queue_length,
            clipped_state.energy_level,
            clipped_state.communication_quality,
        )