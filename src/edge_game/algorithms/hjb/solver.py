"""Numerical Hamilton-Jacobi-Bellman solver."""

from dataclasses import dataclass

import numpy as np

from .dynamics import StateDynamics


@dataclass
class HJBSolution:
    """Store the result of an HJB solve."""

    value_function: np.ndarray
    optimal_control: np.ndarray
    iterations: int
    converged: bool
    residual: float


@dataclass
class HJBSolver:
    """Finite-difference HJB solver for a one-dimensional state grid."""

    state_grid: np.ndarray
    control_grid: np.ndarray
    time_step: float = 0.01
    discount_factor: float = 0.95
    tolerance: float = 1e-6
    max_iterations: int = 1000

    def solve(
        self,
        running_cost,
        drift_function,
    ) -> HJBSolution:
        """Solve the discrete HJB fixed-point problem.

        The supplied callbacks define the research-specific running cost and
        drift. This keeps the numerical solver independent of the final
        mathematical formulation.
        """
        state_grid = np.asarray(
            self.state_grid,
            dtype=float,
        )

        control_grid = np.asarray(
            self.control_grid,
            dtype=float,
        )

        if state_grid.ndim != 1:
            raise ValueError(
                "state_grid must be one-dimensional."
            )

        if control_grid.ndim != 1:
            raise ValueError(
                "control_grid must be one-dimensional."
            )

        if len(state_grid) < 2:
            raise ValueError(
                "state_grid must contain at least two points."
            )

        value_function = np.zeros(
            len(state_grid),
            dtype=float,
        )

        optimal_control = np.zeros(
            len(state_grid),
            dtype=float,
        )

        state_spacing = float(
            state_grid[1] - state_grid[0]
        )

        if state_spacing <= 0.0:
            raise ValueError(
                "state_grid must be strictly increasing."
            )

        converged = False
        residual = float("inf")

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            previous_value = value_function.copy()

            gradient = np.gradient(
                previous_value,
                state_spacing,
            )

            updated_value = np.empty_like(
                previous_value
            )

            for state_index, state in enumerate(
                state_grid
            ):
                candidate_values = []

                for control in control_grid:
                    cost = float(
                        running_cost(
                            state,
                            control,
                        )
                    )

                    drift = float(
                        drift_function(
                            state,
                            control,
                        )
                    )

                    hamiltonian = (
                        cost
                        + drift
                        * gradient[state_index]
                    )

                    candidate_values.append(
                        hamiltonian
                    )

                best_index = int(
                    np.argmin(
                        candidate_values
                    )
                )

                best_hamiltonian = (
                    candidate_values[best_index]
                )

                optimal_control[state_index] = (
                    control_grid[best_index]
                )

                updated_value[state_index] = (
                    best_hamiltonian
                    / max(
                        self.discount_factor,
                        1e-12,
                    )
                )

            value_function = (
                (1.0 - self.time_step)
                * previous_value
                + self.time_step
                * updated_value
            )

            residual = float(
                np.max(
                    np.abs(
                        value_function
                        - previous_value
                    )
                )
            )

            if residual < self.tolerance:
                converged = True
                break

        return HJBSolution(
            value_function=value_function,
            optimal_control=optimal_control,
            iterations=iteration,
            converged=converged,
            residual=residual,
        )