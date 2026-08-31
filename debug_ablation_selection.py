from dataclasses import replace

from src.edge_game.config import SimulationConfig
from src.edge_game.experiments.runner import build_mean_field_policy
from src.edge_game.algorithms.experiment import run_policy_experiment


def main() -> None:
    config = SimulationConfig()
    seed = 42

    policies = {}

    for variant in ("full", "no_priority", "no_priority_reward"):
        policy, diagnostics = build_mean_field_policy(
            config=config,
            ablation_variant=variant,
        )

        policies[variant] = policy

        print(f"\n=== {variant} ===")
        print(f"Converged: {diagnostics['converged']}")
        print(f"Iterations: {diagnostics['iterations']}")

        for priority, equilibrium_policy in (
            policy.equilibrium.policies.items()
        ):
            print(
                f"Priority {priority}: "
                f"control_mean="
                f"{equilibrium_policy.control.mean():.9f}"
            )

    seed_config = replace(
        config,
        seed=seed,
        tasks_per_step=3,
    )

    results = {}

    for variant, policy in policies.items():
        result = run_policy_experiment(
            config=seed_config,
            policy_name=variant,
            policy=policy,
        )

        results[variant] = result

        print(
            f"\n{variant}: "
            f"utility={result.metrics['utility_mean']:.12f}, "
            f"response={result.metrics['response_time_mean']:.12f}, "
            f"throughput={result.metrics['throughput']:.12f}"
        )

    full_records = results["full"].decision_records
    no_priority_records = results["no_priority"].decision_records
    no_reward_records = results["no_priority_reward"].decision_records

    print("\n=== Decision comparison ===")

    full_by_task = {
        record["task_id"]: record
        for record in full_records
    }

    no_priority_by_task = {
        record["task_id"]: record
        for record in no_priority_records
    }

    no_reward_by_task = {
        record["task_id"]: record
        for record in no_reward_records
    }

    priority_differences = 0
    reward_differences = 0

    for task_id, full_record in full_by_task.items():
        no_priority_record = no_priority_by_task.get(task_id)
        no_reward_record = no_reward_by_task.get(task_id)

        if no_priority_record is not None:
            if (
                full_record["selected_node_id"]
                != no_priority_record["selected_node_id"]
            ):
                priority_differences += 1

        if no_reward_record is not None:
            if (
                full_record["selected_node_id"]
                != no_reward_record["selected_node_id"]
            ):
                reward_differences += 1

    print(
        "Full vs no_priority selected-node differences:",
        priority_differences,
    )

    print(
        "Full vs no_priority_reward selected-node differences:",
        reward_differences,
    )

    print("\n=== First 10 no_priority score comparisons ===")

    for task_id in sorted(full_by_task)[:10]:
        full_record = full_by_task[task_id]
        no_priority_record = no_priority_by_task[task_id]

        print(
            f"task={task_id} "
            f"priority={full_record['priority']} "
            f"full_node={full_record['selected_node_id']} "
            f"no_priority_node="
            f"{no_priority_record['selected_node_id']} "
            f"full_score="
            f"{full_record['selected_score']:.12f} "
            f"no_priority_score="
            f"{no_priority_record['selected_score']:.12f} "
            f"full_control="
            f"{full_record['control']:.12f} "
            f"no_priority_control="
            f"{no_priority_record['control']:.12f}"
        )


if __name__ == "__main__":
    main()