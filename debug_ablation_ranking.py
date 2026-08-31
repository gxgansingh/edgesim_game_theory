from dataclasses import replace

import numpy as np

from src.edge_game.config import SimulationConfig
from src.edge_game.experiments.runner import build_mean_field_policy
from src.edge_game.algorithms.experiment import run_policy_experiment


def main() -> None:
    config = SimulationConfig()
    seed_config = replace(
        config,
        seed=42,
        tasks_per_step=3,
    )

    policies = {}

    for variant in (
        "full",
        "no_priority",
        "no_priority_reward",
    ):
        policy, diagnostics = build_mean_field_policy(
            config=config,
            ablation_variant=variant,
        )

        policies[variant] = policy

        print(f"\n=== {variant} ===")
        print(
            f"Converged: {diagnostics['converged']}"
        )
        print(
            f"Iterations: {diagnostics['iterations']}"
        )

        for priority, equilibrium_policy in (
            policy.equilibrium.policies.items()
        ):
            print(
                f"Priority {priority}: "
                f"control_mean="
                f"{equilibrium_policy.control.mean():.9f}"
            )

    results = {}

    for variant, policy in policies.items():
        results[variant] = run_policy_experiment(
            config=seed_config,
            policy_name=variant,
            policy=policy,
        )

    full_records = results["full"].decision_records
    no_priority_records = (
        results["no_priority"].decision_records
    )
    no_reward_records = (
        results["no_priority_reward"].decision_records
    )

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

    print("\n=== Candidate Ranking Comparison ===")

    ranking_differences = 0
    score_differences = 0
    control_differences = 0

    for task_id in sorted(full_by_task):
        full = full_by_task[task_id]
        no_priority = no_priority_by_task[task_id]

        full_scores = _parse_scores(
            full["candidate_scores"]
        )

        no_priority_scores = _parse_scores(
            no_priority["candidate_scores"]
        )

        full_order = _ranking(full_scores)
        no_priority_order = _ranking(
            no_priority_scores
        )

        score_changed = any(
            not np.isclose(
                full_scores[node_id],
                no_priority_scores[node_id],
                atol=1e-12,
            )
            for node_id in full_scores
        )

        control_changed = not np.isclose(
            full["control"],
            no_priority["control"],
            atol=1e-12,
        )

        ranking_changed = (
            full_order != no_priority_order
        )

        if score_changed:
            score_differences += 1

        if control_changed:
            control_differences += 1

        if ranking_changed:
            ranking_differences += 1

        if task_id < 10:
            print(
                f"\nTask {task_id} "
                f"(priority={full['priority']})"
            )

            print(
                f"Full ranking:        {full_order}"
            )

            print(
                f"No-priority ranking: {no_priority_order}"
            )

            print(
                f"Selected full: "
                f"{full['selected_node_id']}"
            )

            print(
                f"Selected no-priority: "
                f"{no_priority['selected_node_id']}"
            )

            print(
                f"Score changed: "
                f"{score_changed}"
            )

            print(
                f"Control changed: "
                f"{control_changed}"
            )

            print(
                f"Ranking changed: "
                f"{ranking_changed}"
            )

            print("\nScores:")

            for node_id in sorted(full_scores):
                print(
                    f"  node={node_id} "
                    f"full={full_scores[node_id]:.12f} "
                    f"no_priority="
                    f"{no_priority_scores[node_id]:.12f}"
                )

    print("\n=== Summary ===")
    print(
        "Tasks with score differences:",
        score_differences,
    )
    print(
        "Tasks with control differences:",
        control_differences,
    )
    print(
        "Tasks with ranking differences:",
        ranking_differences,
    )

    print(
        "\n=== Full vs no_priority_reward ==="
    )

    reward_score_differences = 0
    reward_control_differences = 0
    reward_ranking_differences = 0

    for task_id in sorted(full_by_task):
        full = full_by_task[task_id]
        reward = no_reward_by_task[task_id]

        full_scores = _parse_scores(
            full["candidate_scores"]
        )

        reward_scores = _parse_scores(
            reward["candidate_scores"]
        )

        if any(
            not np.isclose(
                full_scores[node_id],
                reward_scores[node_id],
                atol=1e-12,
            )
            for node_id in full_scores
        ):
            reward_score_differences += 1

        if not np.isclose(
            full["control"],
            reward["control"],
            atol=1e-12,
        ):
            reward_control_differences += 1

        if _ranking(full_scores) != _ranking(
            reward_scores
        ):
            reward_ranking_differences += 1

    print(
        "Tasks with score differences:",
        reward_score_differences,
    )

    print(
        "Tasks with control differences:",
        reward_control_differences,
    )

    print(
        "Tasks with ranking differences:",
        reward_ranking_differences,
    )


def _parse_scores(value: str) -> dict[int, float]:
    scores = {}

    if not value:
        return scores

    for item in value.split(","):
        node_id, score = item.split(":", 1)

        scores[int(node_id)] = float(score)

    return scores


def _ranking(scores: dict[int, float]) -> list[int]:
    return [
        node_id
        for node_id, _ in sorted(
            scores.items(),
            key=lambda item: (
                item[1],
                item[0],
            ),
        )
    ]


if __name__ == "__main__":
    main()