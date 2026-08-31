"""Numerical Fokker-Planck-Kolmogorov solver."""

from dataclasses import dataclass

import numpy as np

from .distribution import MeanFieldDistribution


@dataclass
class FPKSolver:
    """One-dimensional finite-volume FPK transport solver."""

    grid: np.ndarray
    time_step: float = 0.01
    diffusion: float = 0.01

    def step(
        self,
        distribution: MeanFieldDistribution,
        drift_function,
    ) -> MeanFieldDistribution:
        """Advance the population distribution by one time step."""
        grid = np.asarray(
            self.grid,
            dtype=float,
        )

        density = distribution.density.copy()

        if len(grid) < 2:
            raise ValueError(
                "The FPK grid requires at least two points."
            )

        spacing = float(
            grid[1] - grid[0]
        )

        if spacing <= 0.0:
            raise ValueError(
                "The FPK grid must be strictly increasing."
            )

        drift = np.asarray(
            [
                drift_function(
                    float(state)
                )
                for state in grid
            ],
            dtype=float,
        )

        flux = drift * density

        next_density = density.copy()

        next_density[1:-1] -= (
            self.time_step
            / spacing
            * (
                flux[1:-1]
                - flux[:-2]
            )
        )

        if self.diffusion > 0.0:
            second_derivative = (
                density[2:]
                - 2.0 * density[1:-1]
                + density[:-2]
            ) / (
                spacing ** 2
            )

            next_density[1:-1] += (
                self.time_step
                * self.diffusion
                * second_derivative
            )

        next_density[0] = max(
            next_density[0],
            0.0,
        )

        next_density[-1] = max(
            next_density[-1],
            0.0,
        )

        next_density = np.maximum(
            next_density,
            0.0,
        )

        return MeanFieldDistribution(
            grid=grid,
            density=next_density,
        )