"""Priority-aware Mean-Field equilibrium solver."""

from dataclasses import dataclass

import numpy as np

from ..models.mean_field_model import MeanFieldModel
from .population import (
    MultiPopulationMeanField,
    PriorityPopulation,
)


@dataclass
class PriorityPolicy:
    """Store an equilibrium policy for one priority class."""

    priority: int
    control: np.ndarray


@dataclass
class MeanFieldEquilibriumResult:
    """Store a Mean-Field equilibrium result."""

    policies: dict[int, PriorityPolicy]
    distribution: MultiPopulationMeanField

    iterations: int
    converged: bool

    residual: float
    policy_residual: float

    raw_residual: float
    raw_policy_residual: float

    fpk_iterations: dict[int, int]
    fpk_residuals: dict[int, float]


class PriorityAwareMeanFieldSolver:
    """Solve the reduced priority-aware Mean-Field equilibrium."""

    def __init__(
        self,
        model: MeanFieldModel,
        state_grid: np.ndarray,
        control_grid: np.ndarray | None = None,
        tolerance: float = 1e-4,
        policy_tolerance: float = 1e-4,
        max_iterations: int = 1000,
        damping: float = 0.05,
        policy_damping: float = 0.05,
        fpk_time_step: float = 0.001,
        fpk_tolerance: float = 1e-5,
        fpk_max_iterations: int = 5000,
        raw_policy_tolerance: float = 1e-3,
    ) -> None:
        """Initialize the Mean-Field solver."""
        self.model = model

        self.raw_policy_tolerance = float(
            raw_policy_tolerance
        )

        self.state_grid = np.asarray(
            state_grid,
            dtype=float,
        )

        self.tolerance = float(
            tolerance
        )

        self.policy_tolerance = float(
            policy_tolerance
        )

        self.max_iterations = int(
            max_iterations
        )

        self.damping = float(
            np.clip(
                damping,
                1e-3,
                1.0,
            )
        )

        self.policy_damping = float(
            np.clip(
                policy_damping,
                1e-3,
                1.0,
            )
        )

        self.fpk_time_step = float(
            fpk_time_step
        )

        self.fpk_tolerance = float(
            fpk_tolerance
        )

        self.fpk_max_iterations = int(
            fpk_max_iterations
        )

        if len(self.state_grid) < 3:
            raise ValueError(
                "At least three state-grid points are required."
            )

        if np.any(
            np.diff(self.state_grid) <= 0.0
        ):
            raise ValueError(
                "state_grid must be strictly increasing."
            )

        if self.raw_policy_tolerance <= 0.0:
            raise ValueError(
                "raw_policy_tolerance must be positive."
            )

        if self.max_iterations < 1:
            raise ValueError(
                "max_iterations must be positive."
            )

        if self.fpk_time_step <= 0.0:
            raise ValueError(
                "fpk_time_step must be positive."
            )

        if self.fpk_tolerance <= 0.0:
            raise ValueError(
                "fpk_tolerance must be positive."
            )

        if self.fpk_max_iterations < 1:
            raise ValueError(
                "fpk_max_iterations must be positive."
            )

    def _initial_population(
        self,
        priority: int,
    ) -> PriorityPopulation:
        """Create an initial population density."""
        center = (
            0.20
            + 0.15 * (priority - 1)
        )

        density = np.exp(
            -(
                self.state_grid - center
            ) ** 2
            / 0.03
        )

        return PriorityPopulation(
            priority=priority,
            state_grid=self.state_grid.copy(),
            density=density,
            population_mass=(
                self.model.parameters.population_weight(
                    priority
                )
            ),
        )

    def _solve_policy(
        self,
        priority: int,
        mean_field: np.ndarray,
    ) -> PriorityPolicy:
        """Solve the stationary HJB best-response problem."""
        value = np.zeros_like(self.state_grid)
        policy = np.full_like(self.state_grid, 0.5)

        spacing = self.state_grid[1] - self.state_grid[0]

        for _ in range(500):
            gradient = np.gradient(value, spacing)
            second_derivative = np.gradient(gradient, spacing)

            updated_policy = self.model.optimal_control_array(
                priority=priority,
                value_gradient=gradient,
            )

            residual = self.model.hjb_residual_array(
                priority=priority,
                state=self.state_grid,
                control=updated_policy,
                mean_field=mean_field,
                value_gradient=gradient,
                value_second_derivative=second_derivative,
                value=value,
            )

            updated_value = value + 0.05 * residual

            value_difference = float(
                np.max(np.abs(updated_value - value))
            )
            policy_difference = float(
                np.max(np.abs(updated_policy - policy))
            )

            value = 0.90 * value + 0.10 * updated_value
            policy = 0.90 * policy + 0.10 * updated_policy

            if (
                value_difference < 1e-6
                and policy_difference < 1e-6
            ):
                break

        return PriorityPolicy(
            priority=priority,
            control=np.clip(policy, 0.0, 1.0),
        )

    def _fpk_step(
        self,
        population: PriorityPopulation,
        policy: PriorityPolicy,
        mean_field: np.ndarray,
    ) -> np.ndarray:
        """Perform one explicit FPK time step."""
        density = population.density.copy()

        spacing = (
            self.state_grid[1]
            - self.state_grid[0]
        )

        drift = self.model.drift_array(
            priority=population.priority,
            state=self.state_grid,
            control=policy.control,
            mean_field=mean_field,
        )

        flux = drift * density

        updated = density.copy()

        updated[1:-1] -= (
            self.fpk_time_step
            / spacing
            * (
                flux[1:-1]
                - flux[:-2]
            )
        )

        diffusion = (
            self.model.parameters.diffusion
        )

        second_derivative = (
            density[2:]
            - 2.0 * density[1:-1]
            + density[:-2]
        ) / (
            spacing ** 2
        )

        updated[1:-1] += (
            self.fpk_time_step
            * diffusion
            * second_derivative
        )

        updated = np.maximum(
            updated,
            0.0,
        )

        return updated

    def _solve_fpk(
        self,
        population: PriorityPopulation,
        policy: PriorityPolicy,
        mean_field: np.ndarray,
    ) -> tuple[
        PriorityPopulation,
        int,
        float,
    ]:
        """Evolve one population toward its stationary FPK state."""
        density = population.density.copy()

        residual = float("inf")

        for iteration in range(
            1,
            self.fpk_max_iterations + 1,
        ):
            previous_density = density.copy()

            temporary_population = PriorityPopulation(
                priority=population.priority,
                state_grid=self.state_grid.copy(),
                density=density,
                population_mass=population.population_mass,
            )

            density = self._fpk_step(
                population=temporary_population,
                policy=policy,
                mean_field=mean_field,
            )

            temporary_population.density = (
                density
            )

            temporary_population.normalize()

            density = (
                temporary_population.density
            )

            residual = float(
                np.max(
                    np.abs(
                        density
                        - previous_density
                    )
                )
            )

            if residual < self.fpk_tolerance:
                break

        result = PriorityPopulation(
            priority=population.priority,
            state_grid=self.state_grid.copy(),
            density=density,
            population_mass=population.population_mass,
        )

        return (
            result,
            iteration,
            residual,
        )

    @staticmethod
    def _policy_difference(
        previous: dict[int, PriorityPolicy],
        current: dict[int, PriorityPolicy],
    ) -> float:
        """Calculate the maximum policy difference."""
        if not previous:
            return float("inf")

        residual = 0.0

        for priority in current:
            difference = float(
                np.max(
                    np.abs(
                        current[priority].control
                        - previous[priority].control
                    )
                )
            )

            residual = max(
                residual,
                difference,
            )

        return residual

    def _relax_policies(
        self,
        previous: dict[int, PriorityPolicy],
        best_response: dict[int, PriorityPolicy],
    ) -> dict[int, PriorityPolicy]:
        """Apply fixed-point relaxation to the policy."""
        if not previous:
            return {
                priority: PriorityPolicy(
                    priority=policy.priority,
                    control=policy.control.copy(),
                )
                for priority, policy
                in best_response.items()
            }

        relaxed = {}

        for priority in best_response:
            previous_control = (
                previous[priority].control
            )

            response_control = (
                best_response[priority].control
            )

            control = (
                (1.0 - self.policy_damping)
                * previous_control
                + self.policy_damping
                * response_control
            )

            relaxed[priority] = (
                PriorityPolicy(
                    priority=priority,
                    control=np.clip(
                        control,
                        0.0,
                        1.0,
                    ),
                )
            )

        return relaxed

    def solve(self) -> MeanFieldEquilibriumResult:
        """Solve for the priority-aware Mean-Field equilibrium."""
        distribution = (
            MultiPopulationMeanField()
        )

        for priority in (1, 2, 3):
            distribution.add_population(
                self._initial_population(
                    priority
                )
            )

        policies: dict[
            int,
            PriorityPolicy
        ] = {}

        previous_policies: dict[
            int,
            PriorityPolicy
        ] = {}

        residual = float("inf")
        policy_residual = float("inf")

        raw_residual = float("inf")
        raw_policy_residual = float("inf")

        fpk_iterations: dict[
            int,
            int
        ] = {}

        fpk_residuals: dict[
            int,
            float
        ] = {}

        converged = False

        for iteration in range(
            1,
            self.max_iterations + 1,
        ):
            old_mean_field = (
                distribution.aggregate_density()
            )

            best_response = {}

            for priority in (1, 2, 3):
                best_response[priority] = (
                    self._solve_policy(
                        priority=priority,
                        mean_field=old_mean_field,
                    )
                )

            raw_policy_residual = (
                self._policy_difference(
                    previous=previous_policies,
                    current=best_response,
                )
            )

            policies = self._relax_policies(
                previous=previous_policies,
                best_response=best_response,
            )

            policy_residual = (
                self._policy_difference(
                    previous=previous_policies,
                    current=policies,
                )
            )

            updated_distribution = (
                MultiPopulationMeanField()
            )

            for priority in (1, 2, 3):
                population = (
                    distribution.get_population(
                        priority
                    )
                )

                updated_population, inner_iterations, inner_residual = (
                    self._solve_fpk(
                        population=population,
                        policy=policies[priority],
                        mean_field=old_mean_field,
                    )
                )

                fpk_iterations[priority] = (
                    inner_iterations
                )

                fpk_residuals[priority] = (
                    inner_residual
                )

                old_density = (
                    population.density
                )

                relaxed_density = (
                    (1.0 - self.damping)
                    * old_density
                    + self.damping
                    * updated_population.density
                )

                updated_population.density = (
                    relaxed_density
                )

                updated_population.normalize()

                updated_distribution.add_population(
                    updated_population
                )

            new_mean_field = (
                updated_distribution.aggregate_density()
            )

            raw_residual = float(
                np.max(
                    np.abs(
                        new_mean_field
                        - old_mean_field
                    )
                )
            )

            residual = raw_residual

            distribution = (
                updated_distribution
            )

            max_fpk_residual = max(
                fpk_residuals.values()
            )

            if (
                residual < self.tolerance
                and policy_residual
                < self.policy_tolerance
                and raw_policy_residual
                < self.raw_policy_tolerance
                and max_fpk_residual
                < self.fpk_tolerance
            ):
                converged = True
                break

            previous_policies = {
                priority: PriorityPolicy(
                    priority=policy.priority,
                    control=policy.control.copy(),
                )
                for priority, policy
                in policies.items()
            }

        return MeanFieldEquilibriumResult(
            policies=policies,
            distribution=distribution,
            iterations=iteration,
            converged=converged,
            residual=residual,
            policy_residual=policy_residual,
            raw_residual=raw_residual,
            raw_policy_residual=raw_policy_residual,
            fpk_iterations=fpk_iterations,
            fpk_residuals=fpk_residuals,
        )