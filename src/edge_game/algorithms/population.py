"""Priority-specific Mean-Field population representation."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PriorityPopulation:
    """Represent one priority-specific workload population."""

    priority: int
    state_grid: np.ndarray
    density: np.ndarray
    population_mass: float

    def __post_init__(self) -> None:
        """Validate the population representation."""
        self.state_grid = np.asarray(
            self.state_grid,
            dtype=float,
        )

        self.density = np.asarray(
            self.density,
            dtype=float,
        )

        if self.state_grid.ndim != 1:
            raise ValueError(
                "state_grid must be one-dimensional."
            )

        if self.density.ndim != 1:
            raise ValueError(
                "density must be one-dimensional."
            )

        if len(self.state_grid) != len(
            self.density
        ):
            raise ValueError(
                "state_grid and density must have equal lengths."
            )

        if self.population_mass < 0.0:
            raise ValueError(
                "population_mass cannot be negative."
            )

        self.normalize()

    def normalize(self) -> None:
        """Normalize the population density."""
        self.density = np.maximum(
            self.density,
            0.0,
        )

        integral = float(
            np.trapezoid(
                self.density,
                self.state_grid,
            )
        )

        if integral <= 1e-12:
            self.density.fill(
                1.0 / len(self.density)
            )

            integral = float(
                np.trapezoid(
                    self.density,
                    self.state_grid,
                )
            )

        self.density /= integral

    def mean_state(self) -> float:
        """Return the population mean state."""
        return float(
            np.trapezoid(
                self.state_grid
                * self.density,
                self.state_grid,
            )
        )


@dataclass
class MultiPopulationMeanField:
    """Manage priority-specific workload populations."""

    populations: dict[
        int,
        PriorityPopulation
    ] = field(
        default_factory=dict
    )

    def add_population(
        self,
        population: PriorityPopulation,
    ) -> None:
        """Add a priority-specific population."""
        self.populations[
            population.priority
        ] = population

    def get_population(
        self,
        priority: int,
    ) -> PriorityPopulation:
        """Return a population by priority."""
        if priority not in self.populations:
            raise KeyError(
                f"Priority {priority} population not found."
            )

        return self.populations[
            priority
        ]

    def aggregate_density(self) -> np.ndarray:
        """Calculate the priority-weighted aggregate density."""
        if not self.populations:
            raise ValueError(
                "No populations are available."
            )

        priorities = sorted(
            self.populations
        )

        reference = self.populations[
            priorities[0]
        ]

        aggregate = np.zeros_like(
            reference.density
        )

        total_mass = sum(
            population.population_mass
            for population in self.populations.values()
        )

        if total_mass <= 1e-12:
            return aggregate

        for priority in priorities:
            population = self.populations[
                priority
            ]

            if not np.array_equal(
                population.state_grid,
                reference.state_grid,
            ):
                raise ValueError(
                    "All populations must share the same state grid."
                )

            weight = (
                population.population_mass
                / total_mass
            )

            aggregate += (
                weight
                * population.density
            )

        return aggregate

    def aggregate_mean_state(self) -> float:
        """Return the mean state of the aggregate population."""
        priorities = sorted(
            self.populations
        )

        if not priorities:
            return 0.0

        reference = self.populations[
            priorities[0]
        ]

        aggregate = self.aggregate_density()

        return float(
            np.trapezoid(
                reference.state_grid
                * aggregate,
                reference.state_grid,
            )
        )