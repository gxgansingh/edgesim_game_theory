"""Tests for the simulation and mathematical foundation."""

import numpy as np

from src.edge_game.algorithms.mean_field import (
    PriorityAwareMeanFieldSolver,
)
from src.edge_game.algorithms.fpk.distribution import (
    MeanFieldDistribution,
)
from src.edge_game.algorithms.fpk.solver import (
    FPKSolver,
)
from src.edge_game.algorithms.hjb.solver import (
    HJBSolver,
)
from src.edge_game.algorithms.hjb.state import (
    MeanFieldState,
)
from src.edge_game.algorithms.candidate_filter import (
    build_feasibility_audit,
    filter_feasible_nodes,
)
from src.edge_game.algorithms.utility import (
    calculate_utility,
)
from src.edge_game.config import SimulationConfig
from src.edge_game.environment import (
    SimulationEnvironment,
)
from src.edge_game.models.mean_field_model import (
    MeanFieldModel,
)
from src.edge_game.entities.edge_node import EdgeNode
from src.edge_game.entities.task import Task

def test_environment_creates_expected_node_count() -> None:
    """Verify the configured number of edge nodes is created."""
    config = SimulationConfig(
        number_of_servers=2,
        nodes_per_server=3,
    )

    environment = SimulationEnvironment(
        config=config
    )

    assert len(
        environment.all_nodes()
    ) == 6


def test_feasibility_filter_returns_valid_nodes() -> None:
    """Verify the feasibility filter excludes invalid nodes."""
    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
    )

    environment = SimulationEnvironment(
        config=config
    )

    task = environment.create_task(
        task_id=0
    )

    nodes = environment.all_nodes()

    feasible = filter_feasible_nodes(
        task=task,
        nodes=nodes,
    )

    assert all(
        node.can_process(task)
        for node in feasible
    )


def test_utility_returns_finite_value() -> None:
    """Verify the utility function produces a finite value."""
    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
    )

    environment = SimulationEnvironment(
        config=config
    )

    task = environment.create_task(
        task_id=0
    )

    node = environment.all_nodes()[0]

    utility = calculate_utility(
        task=task,
        node=node,
    )

    assert utility == utility


def test_task_allocation_changes_node_resources() -> None:
    """Verify task allocation consumes node resources."""
    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
    )

    environment = SimulationEnvironment(
        config=config
    )

    task = environment.create_task(
        task_id=0
    )

    node = environment.all_nodes()[0]

    initial_cpu = node.available_cpu

    node.allocate_task(
        task=task,
        current_time=environment.current_time,
    )

    assert node.available_cpu < initial_cpu
    assert task.assigned_node_id == node.node_id
    assert len(node.active_tasks) == 1


def test_task_processing_releases_resources() -> None:
    """Verify completed tasks release allocated resources."""
    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
        simulation_time_step=1.0,
        task_cpu_rate=100.0,
    )

    environment = SimulationEnvironment(
        config=config
    )

    node = environment.all_nodes()[0]

    task = environment.create_task(
        task_id=0
    )

    initial_cpu = node.available_cpu

    node.allocate_task(
        task=task,
        current_time=environment.current_time,
    )

    completed = environment.advance_time()

    assert task in completed
    assert len(node.active_tasks) == 0
    assert node.available_cpu == initial_cpu
    assert task.completion_time is not None


def test_mean_field_state_is_normalized() -> None:
    """Verify Mean-Field state values are clipped to valid ranges."""
    state = MeanFieldState(
        priority=3,
        cpu_load=1.5,
        queue_length=-2.0,
        energy_level=-0.5,
        communication_quality=1.5,
    )

    clipped = state.clipped()

    assert clipped.cpu_load == 1.0
    assert clipped.queue_length == 0.0
    assert clipped.energy_level == 0.0
    assert clipped.communication_quality == 1.0


def test_mean_field_distribution_is_normalized() -> None:
    """Verify a population density integrates to one."""
    grid = np.linspace(
        0.0,
        1.0,
        11,
    )

    density = np.ones(
        11,
        dtype=float,
    )

    distribution = MeanFieldDistribution(
        grid=grid,
        density=density,
    )

    integral = np.trapezoid(
        distribution.density,
        distribution.grid,
    )

    assert np.isclose(
        integral,
        1.0,
    )


def test_hjb_solver_produces_solution() -> None:
    """Verify the HJB numerical solver executes."""
    state_grid = np.linspace(
        0.0,
        1.0,
        11,
    )

    control_grid = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = HJBSolver(
        state_grid=state_grid,
        control_grid=control_grid,
        max_iterations=10,
    )

    solution = solver.solve(
        running_cost=lambda state, control: (
            state ** 2 + control ** 2
        ),
        drift_function=lambda state, control: (
            control - state
        ),
    )

    assert solution.value_function.shape == (
        len(state_grid),
    )

    assert solution.optimal_control.shape == (
        len(state_grid),
    )

    assert solution.iterations <= 10


def test_fpk_solver_preserves_distribution_normalization() -> None:
    """Verify the FPK step returns a normalized density."""
    grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    density = np.exp(
        -((grid - 0.5) ** 2) / 0.02
    )

    distribution = MeanFieldDistribution(
        grid=grid,
        density=density,
    )

    solver = FPKSolver(
        grid=grid,
        time_step=0.001,
        diffusion=0.001,
    )

    updated = solver.step(
        distribution=distribution,
        drift_function=lambda state: (
            0.1 * (0.5 - state)
        ),
    )

    integral = np.trapezoid(
        updated.density,
        updated.grid,
    )

    assert np.isclose(
        integral,
        1.0,
        atol=1e-6,
    )

from src.edge_game.algorithms.mean_field import (
    PriorityAwareMeanFieldSolver,
)
from src.edge_game.models.mean_field_model import (
    MeanFieldModel,
    MeanFieldParameters,
)


def test_resource_filtering_checks_all_required_dimensions() -> None:
    """Verify CPU, memory, bandwidth, latency, energy, and queue checks."""
    from src.edge_game.entities.edge_node import EdgeNode
    from src.edge_game.entities.task import Task

    config = SimulationConfig(
        base_network_latency=2.0,
        latency_load_penalty=8.0,
        latency_queue_penalty=0.75,
        latency_workload_penalty=0.50,
        energy_per_cpu_work_unit=0.25,
        maximum_queue_length=2,
    )

    task = Task(
        task_id=99,
        priority=4,
        cpu_demand=5.0,
        memory_demand=4.0,
        bandwidth_demand=5.0,
        latency_requirement=20.0,
        energy_budget=10.0,
        workload_size=4.0,
    )

    node = EdgeNode(
        node_id=1,
        server_id=0,
        cpu_capacity=10.0,
        memory_capacity=8.0,
        bandwidth_capacity=20.0,
        energy_capacity=100.0,
        queue_length=0,
    )

    audit = build_feasibility_audit(
        task=task,
        nodes=[node],
        config=config,
    )[0]

    assert audit.cpu_pass
    assert audit.memory_pass
    assert audit.bandwidth_pass
    assert audit.latency_pass
    assert audit.energy_pass
    assert audit.queue_pass
    assert audit.feasible


def test_resource_filtering_excludes_infeasible_nodes() -> None:
    """Verify an infeasible node is removed before policy selection."""
    from src.edge_game.entities.edge_node import EdgeNode
    from src.edge_game.entities.task import Task

    config = SimulationConfig()
    task = Task(
        task_id=100,
        priority=4,
        cpu_demand=8.0,
        memory_demand=8.0,
        bandwidth_demand=15.0,
        latency_requirement=50.0,
        energy_budget=20.0,
        workload_size=1.0,
    )

    feasible = EdgeNode(
        node_id=1,
        server_id=0,
        cpu_capacity=10.0,
        memory_capacity=16.0,
        bandwidth_capacity=30.0,
        energy_capacity=100.0,
    )
    infeasible = EdgeNode(
        node_id=2,
        server_id=0,
        cpu_capacity=4.0,
        memory_capacity=16.0,
        bandwidth_capacity=30.0,
        energy_capacity=100.0,
    )

    candidates = filter_feasible_nodes(
        task=task,
        nodes=[feasible, infeasible],
        config=config,
    )

    assert [node.node_id for node in candidates] == [1]


def test_priority_populations_are_created() -> None:
    """Verify all priority populations are represented."""
    model = MeanFieldModel()

    grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    controls = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=grid,
        control_grid=controls,
        max_iterations=2,
    )

    result = solver.solve()

    assert set(
        result.distribution.populations.keys()
    ) == {1, 2, 3, 4}


def test_mean_field_density_is_normalized() -> None:
    """Verify each priority population remains normalized."""
    model = MeanFieldModel()

    grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    controls = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=grid,
        control_grid=controls,
        max_iterations=2,
    )

    result = solver.solve()

    for population in (
        result.distribution.populations.values()
    ):
        integral = np.trapezoid(
            population.density,
            population.state_grid,
        )

        assert np.isclose(
            integral,
            1.0,
            atol=1e-6,
        )


def test_mean_field_controls_are_bounded() -> None:
    """Verify equilibrium controls remain within the action space."""
    model = MeanFieldModel()

    grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    controls = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=grid,
        control_grid=controls,
        max_iterations=2,
    )

    result = solver.solve()

    for policy in result.policies.values():
        assert np.all(
            policy.control >= 0.0
        )

        assert np.all(
            policy.control <= 1.0
        )

from src.edge_game.algorithms.policy import (
    BaselinePolicy,
    MeanFieldPolicy,
)
from src.edge_game.models.mean_field_model import (
    MeanFieldModel,
)
from src.edge_game.algorithms.mean_field import (
    PriorityAwareMeanFieldSolver,
)

def test_baseline_policy_selects_node() -> None:
    """Verify the baseline policy selects a feasible node."""
    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=2,
    )

    environment = SimulationEnvironment(
        config=config
    )

    task = environment.create_task(
        task_id=0
    )

    nodes = environment.all_nodes()

    candidates = filter_feasible_nodes(
        task=task,
        nodes=nodes,
    )

    policy = BaselinePolicy()

    selected = policy.select_node(
        task=task,
        candidates=candidates,
    )

    assert selected in candidates


def test_mean_field_policy_selects_node() -> None:
    """Verify the Mean-Field policy selects a feasible node."""
    model = MeanFieldModel()

    state_grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    control_grid = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=state_grid,
        control_grid=control_grid,
        max_iterations=2,
    )

    equilibrium = solver.solve()

    policy = MeanFieldPolicy(
        model=model,
        equilibrium=equilibrium,
    )

    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=2,
    )

    environment = SimulationEnvironment(
        config=config
    )

    task = environment.create_task(
        task_id=0
    )

    nodes = environment.all_nodes()

    candidates = filter_feasible_nodes(
        task=task,
        nodes=nodes,
    )

    selected = policy.select_node(
        task=task,
        candidates=candidates,
    )

    assert selected in candidates

def test_mean_field_damping_is_valid() -> None:
    """Verify the Mean-Field damping parameter is bounded."""
    config = SimulationConfig()

    assert 0.0 < config.mean_field_damping <= 1.0

def test_mean_field_policy_residual_is_reported() -> None:
    """Verify the equilibrium reports policy convergence."""
    model = MeanFieldModel()

    state_grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    control_grid = np.linspace(
        0.0,
        1.0,
        5,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=state_grid,
        control_grid=control_grid,
        tolerance=1e-3,
        policy_tolerance=1e-3,
        policy_damping=0.05,
        max_iterations=3,
        damping=0.20,
    )

    result = solver.solve()

    assert np.isfinite(
        result.residual
    )

    assert np.isfinite(
        result.policy_residual
    )


def test_mean_field_damping_is_valid() -> None:
    """Verify the configured damping factor is valid."""
    config = SimulationConfig()

    assert (
        0.0
        < config.mean_field_damping
        <= 1.0
    )

def test_mean_field_optimal_control_is_continuous() -> None:
    """Verify the analytical control remains within valid bounds."""
    model = MeanFieldModel()

    for priority in (1, 2, 3, 4):
        for gradient in (
            -10.0,
            -1.0,
            0.0,
            1.0,
            10.0,
        ):
            control = model.optimal_control(
                priority=priority,
                value_gradient=gradient,
            )

            assert 0.0 <= control <= 1.0

def test_mean_field_policy_is_not_restricted_to_control_grid() -> None:
    """Verify the policy can represent continuous control values."""
    model = MeanFieldModel()

    control = model.optimal_control(
        priority=2,
        value_gradient=0.37,
    )

    assert 0.0 <= control <= 1.0

def test_mean_field_policy_damping_is_valid() -> None:
    """Verify the Mean-Field policy damping is valid."""
    config = SimulationConfig()

    assert (
        0.0
        < config.mean_field_policy_damping
        <= 1.0
    )

def test_hjb_residual_is_finite() -> None:
    """Verify the HJB residual is numerically well-defined."""
    model = MeanFieldModel()

    residual = model.hjb_residual(
        priority=2,
        state=0.5,
        control=0.5,
        mean_field=0.4,
        value_gradient=0.1,
        value_second_derivative=0.05,
        value=0.2,
    )

    assert np.isfinite(
        residual
    )

def test_fpk_configuration_is_valid() -> None:
    """Verify FPK numerical configuration."""
    config = SimulationConfig()

    assert config.fpk_time_step > 0.0
    assert config.fpk_tolerance > 0.0
    assert config.fpk_max_iterations > 0

def test_fpk_solver_returns_finite_residual() -> None:
    """Verify the inner FPK solver returns a finite residual."""
    model = MeanFieldModel()

    state_grid = np.linspace(
        0.0,
        1.0,
        21,
    )

    solver = PriorityAwareMeanFieldSolver(
        model=model,
        state_grid=state_grid,
        tolerance=1e-3,
        policy_tolerance=1e-3,
        max_iterations=2,
        damping=0.05,
        policy_damping=0.05,
        fpk_time_step=0.001,
        fpk_tolerance=1e-4,
        fpk_max_iterations=10,
    )

    result = solver.solve()

    assert len(
        result.fpk_iterations
    ) == 4

    assert len(
        result.fpk_residuals
    ) == 4

    for residual in (
        result.fpk_residuals.values()
    ):
        assert np.isfinite(
            residual
        )

def test_raw_policy_tolerance_is_valid() -> None:
    """Verify the raw policy tolerance is valid."""
    config = SimulationConfig()

    assert (
        config.mean_field_raw_policy_tolerance
        > 0.0
    )

def test_higher_cpu_capacity_produces_higher_service_rate() -> None:
    """Verify heterogeneous CPU capacity affects service rate."""
    low_capacity_node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=10.0,
        memory_capacity=16.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
    )

    high_capacity_node = EdgeNode(
        node_id=1,
        server_id=0,
        cpu_capacity=30.0,
        memory_capacity=16.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
    )

    low_task = Task(
        task_id=0,
        priority=1,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    high_task = Task(
        task_id=1,
        priority=1,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    low_capacity_node.allocate_task(
        task=low_task,
        current_time=0.0,
    )

    high_capacity_node.allocate_task(
        task=high_task,
        current_time=0.0,
    )

    low_rate = (
        low_capacity_node.effective_service_rate(
            task=low_task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    high_rate = (
        high_capacity_node.effective_service_rate(
            task=high_task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    assert high_rate > low_rate


def test_priority_changes_service_share() -> None:
    """Verify higher priority receives a larger service share."""
    node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=32.0,
        bandwidth_capacity=100.0,
        energy_capacity=100.0,
    )

    low_priority_task = Task(
        task_id=0,
        priority=1,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    high_priority_task = Task(
        task_id=1,
        priority=3,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    node.allocate_task(
        task=low_priority_task,
        current_time=0.0,
    )

    node.allocate_task(
        task=high_priority_task,
        current_time=0.0,
    )

    low_rate = (
        node.effective_service_rate(
            task=low_priority_task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    high_rate = (
        node.effective_service_rate(
            task=high_priority_task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    assert high_rate > low_rate


def test_service_rate_is_shared_across_active_tasks() -> None:
    """Verify concurrent tasks share node service capacity."""
    node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=32.0,
        bandwidth_capacity=100.0,
        energy_capacity=100.0,
    )

    task = Task(
        task_id=0,
        priority=1,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    node.allocate_task(
        task=task,
        current_time=0.0,
    )

    single_task_rate = (
        node.effective_service_rate(
            task=task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    second_task = Task(
        task_id=1,
        priority=1,
        cpu_demand=1.0,
        memory_demand=1.0,
        bandwidth_demand=1.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=10.0,
    )

    node.allocate_task(
        task=second_task,
        current_time=0.0,
    )

    shared_task_rate = (
        node.effective_service_rate(
            task=task,
            base_cpu_rate=2.5,
            reference_cpu_capacity=20.0,
            low_priority_weight=1.0,
            medium_priority_weight=1.25,
            high_priority_weight=1.50,
        )
    )

    assert shared_task_rate < single_task_rate

def test_composite_state_changes_with_memory_load() -> None:
    """Verify memory pressure contributes to the one-dimensional MFG state."""
    from src.edge_game.algorithms.state import composite_state

    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
    )

    low_memory_node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=20.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
        available_cpu=10.0,
        available_memory=18.0,
        available_bandwidth=45.0,
    )

    high_memory_node = EdgeNode(
        node_id=1,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=20.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
        available_cpu=10.0,
        available_memory=8.0,
        available_bandwidth=45.0,
    )

    assert composite_state(
        low_memory_node,
        config,
    ) < composite_state(
        high_memory_node,
        config,
    )


def test_composite_state_changes_with_queue_pressure() -> None:
    """Verify queue pressure contributes to the one-dimensional MFG state."""
    from src.edge_game.algorithms.state import composite_state

    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
        mean_field_queue_normalization=10.0,
    )

    low_queue_node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=20.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
        queue_length=1,
    )

    high_queue_node = EdgeNode(
        node_id=1,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=20.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
        queue_length=8,
    )

    assert composite_state(
        low_queue_node,
        config,
    ) < composite_state(
        high_queue_node,
        config,
    )


def test_composite_state_is_bounded() -> None:
    """Verify the heterogeneous state mapping remains in the HJB domain."""
    from src.edge_game.algorithms.state import composite_state

    config = SimulationConfig(
        number_of_servers=1,
        nodes_per_server=1,
        mean_field_queue_normalization=2.0,
    )

    node = EdgeNode(
        node_id=0,
        server_id=0,
        cpu_capacity=20.0,
        memory_capacity=20.0,
        bandwidth_capacity=50.0,
        energy_capacity=100.0,
        queue_length=100,
        available_cpu=0.0,
        available_memory=0.0,
        available_bandwidth=0.0,
    )

    assert 0.0 <= composite_state(node, config) <= 1.0


def test_outcome_utility_penalizes_latency() -> None:
    """Verify realized utility decreases when response time increases."""
    from src.edge_game.algorithms.utility import (
        calculate_outcome_utility,
    )

    config = SimulationConfig()
    task = Task(
        task_id=0,
        priority=3,
        cpu_demand=2.0,
        memory_demand=2.0,
        bandwidth_demand=2.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=4.0,
    )

    fast = calculate_outcome_utility(
        task=task,
        response_time=2.0,
        node_cpu_capacity=20.0,
        node_memory_capacity=16.0,
        node_bandwidth_capacity=50.0,
        queue_length_at_assignment=0,
        node_energy_capacity=100.0,
        config=config,
    )

    slow = calculate_outcome_utility(
        task=task,
        response_time=8.0,
        node_cpu_capacity=20.0,
        node_memory_capacity=16.0,
        node_bandwidth_capacity=50.0,
        queue_length_at_assignment=0,
        node_energy_capacity=100.0,
        config=config,
    )

    assert fast > slow


def test_outcome_utility_penalizes_queue_pressure() -> None:
    """Verify realized utility decreases with assignment-time queue pressure."""
    from src.edge_game.algorithms.utility import (
        calculate_outcome_utility,
    )

    config = SimulationConfig()
    task = Task(
        task_id=0,
        priority=2,
        cpu_demand=2.0,
        memory_demand=2.0,
        bandwidth_demand=2.0,
        latency_requirement=10.0,
        energy_budget=5.0,
        workload_size=4.0,
    )

    low_queue = calculate_outcome_utility(
        task=task,
        response_time=2.0,
        node_cpu_capacity=20.0,
        node_memory_capacity=16.0,
        node_bandwidth_capacity=50.0,
        queue_length_at_assignment=0,
        node_energy_capacity=100.0,
        config=config,
    )

    high_queue = calculate_outcome_utility(
        task=task,
        response_time=2.0,
        node_cpu_capacity=20.0,
        node_memory_capacity=16.0,
        node_bandwidth_capacity=50.0,
        queue_length_at_assignment=8,
        node_energy_capacity=100.0,
        config=config,
    )

    assert low_queue > high_queue


def test_outcome_utility_penalizes_higher_response_time() -> None:
    """Verify realized utility decreases when response time increases."""
    from src.edge_game.algorithms.utility import calculate_outcome_utility
    from src.edge_game.entities.task import Task

    config = SimulationConfig()
    task = Task(
        task_id=999,
        priority=3,
        cpu_demand=2.0,
        memory_demand=2.0,
        bandwidth_demand=2.0,
        workload_size=2.0,
        latency_requirement=20.0,
        energy_budget=2.0,
    )

    common = {
        "task": task,
        "node_cpu_capacity": 20.0,
        "node_memory_capacity": 20.0,
        "node_bandwidth_capacity": 50.0,
        "queue_length_at_assignment": 1,
        "node_energy_capacity": 20.0,
        "config": config,
    }

    low_latency = calculate_outcome_utility(
        response_time=2.0,
        **common,
    )
    high_latency = calculate_outcome_utility(
        response_time=10.0,
        **common,
    )

    assert high_latency < low_latency


def test_vectorized_mean_field_drift_matches_scalar_drift() -> None:
    """Verify vectorized FPK drift matches scalar evaluations."""
    from src.edge_game.models.mean_field_model import MeanFieldModel

    model = MeanFieldModel()
    states = np.linspace(0.0, 1.0, 21)
    controls = np.linspace(0.1, 0.9, 21)
    mean_field = np.linspace(0.2, 0.8, 21)

    vectorized = model.drift_array(
        priority=2,
        state=states,
        control=controls,
        mean_field=mean_field,
    )

    scalar = np.array(
        [
            model.drift(
                priority=2,
                state=float(state),
                control=float(control),
                mean_field=float(mean),
            )
            for state, control, mean in zip(
                states,
                controls,
                mean_field,
            )
        ]
    )

    np.testing.assert_allclose(vectorized, scalar)


def test_vectorized_hjb_residual_matches_scalar_components() -> None:
    """Verify vectorized HJB residual matches scalar calculations."""
    from src.edge_game.models.mean_field_model import MeanFieldModel

    model = MeanFieldModel()
    states = np.linspace(0.0, 1.0, 21)
    controls = np.linspace(0.1, 0.9, 21)
    mean_field = np.linspace(0.2, 0.8, 21)
    value = np.linspace(0.0, 1.0, 21)
    gradient = np.gradient(value, states[1] - states[0])
    second_derivative = np.gradient(
        gradient,
        states[1] - states[0],
    )

    vectorized = model.hjb_residual_array(
        priority=3,
        state=states,
        control=controls,
        mean_field=mean_field,
        value_gradient=gradient,
        value_second_derivative=second_derivative,
        value=value,
    )

    scalar = np.array(
        [
            model.hjb_residual(
                priority=3,
                state=float(state),
                control=float(control),
                mean_field=float(mean),
                value_gradient=float(value_gradient),
                value_second_derivative=float(value_second),
                value=float(current_value),
            )
            for state, control, mean, value_gradient, value_second, current_value in zip(
                states,
                controls,
                mean_field,
                gradient,
                second_derivative,
                value,
            )
        ]
    )

    np.testing.assert_allclose(vectorized, scalar)
