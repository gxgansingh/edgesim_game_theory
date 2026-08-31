"""Mean-field distribution representation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class MeanFieldDistribution:
    """Represent a normalized population distribution."""

    grid: np.ndarray
    density: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize the distribution."""
        self.grid = np.asarray(
            self.grid,
            dtype=float,
        )

        self.density = np.asarray(
            self.density,
            dtype=float,
        )

        if self.grid.ndim != 1:
            raise ValueError(
                "grid must be one-dimensional."
            )

        if self.density.ndim != 1:
            raise ValueError(
                "density must be one-dimensional."
            )

        if len(self.grid) != len(
            self.density
        ):
            raise ValueError(
                "grid and density must have equal lengths."
            )

        self.normalize()

    def normalize(self) -> None:
        """Normalize the density so its numerical integral equals one."""
        self.density = np.maximum(
            self.density,
            0.0,
        )

        total = float(
            np.trapezoid(
                self.density,
                self.grid,
            )
        )

        if total <= 1e-12:
            self.density.fill(
                1.0 / len(self.density)
            )
            total = float(
                np.trapezoid(
                    self.density,
                    self.grid,
                )
            )

        self.density /= total

    def mean(self) -> float:
        """Return the expected state value."""
        return float(
            np.trapezoid(
                self.grid * self.density,
                self.grid,
            )
        )

    def variance(self) -> float:
        """Return the state variance."""
        mean = self.mean()

        return float(
            np.trapezoid(
                (
                    self.grid - mean
                ) ** 2
                * self.density,
                self.grid,
            )
        )