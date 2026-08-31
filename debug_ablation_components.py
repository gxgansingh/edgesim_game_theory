from dataclasses import replace

from src.edge_game.config import SimulationConfig
from src.edge_game.experiments.runner import build_mean_field_policy
from src.edge_game.algorithms.experiment import run_policy_experiment


def parse_scores(value: str) -> dict[int, float]:
    """Parse node:value diagnostic strings into a dictionary."""
    scores: dict[int, float] = {}

    if not value:
        return scores

    for item in value.split(","):
        node_id, score = item.split(":", 1)
        scores[int(node_id)] = float(score)

    return scores


def parse_node_ids(value: str) -> list[int]:
    """Parse comma-separated node identifiers."""
    if not value:
        return []

    return [
        int(node_id)
        for node_id in value.split(",")
        if node_id
    ]


def main() -> None:
    """Compare candidate-level Mean-Field diagnostics."""
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
        policy, _ = build_mean_field_policy(
            config=config,
            ablation_variant=variant,
        )

        policies[variant] = policy

    results = {}

    for variant, policy in policies.items():
        results[variant] = run_policy_experiment(
            config=seed_config,
            policy_name=variant,
            policy=policy,
        )

    print("\n=== Candidate Component Analysis ===")

    full_records = results[
        "full"
    ].decision_records

    no_priority_records = results[
        "no_priority"
    ].decision_records

    full_by_task = {
        record["task_id"]: record
        for record in full_records
    }

    no_priority_by_task = {
        record["task_id"]: record
        for record in no_priority_records
    }

    common_task_ids = sorted(
        set(full_by_task)
        & set(no_priority_by_task)
    )

    for task_id in common_task_ids[:10]:
        full = full_by_task[task_id]
        no_priority = no_priority_by_task[task_id]

        print(
            f"\n{'=' * 80}"
        )

        print(
            f"Task {task_id} | "
            f"Priority {full['priority']}"
        )

        full_scores = parse_scores(
            full["candidate_scores"]
        )

        no_priority_scores = parse_scores(
            no_priority["candidate_scores"]
        )

        full_states = parse_scores(
            full["candidate_states"]
        )

        no_priority_states = parse_scores(
            no_priority["candidate_states"]
        )

        full_controls = parse_scores(
            full["candidate_controls"]
        )

        no_priority_controls = parse_scores(
            no_priority["candidate_controls"]
        )

        candidate_ids = parse_node_ids(
            full["candidate_node_ids"]
        )

        print(
            "node | "
            "state_full | "
            "state_no_priority | "
            "control_full | "
            "control_no_priority | "
            "score_full | "
            "score_no_priority | "
            "delta"
        )

        print("-" * 145)

        for node_id in candidate_ids:
            if node_id not in full_scores:
                continue

            if node_id not in no_priority_scores:
                continue

            full_state = full_states.get(
                node_id,
                float("nan"),
            )

            no_priority_state = no_priority_states.get(
                node_id,
                float("nan"),
            )

            full_control = full_controls.get(
                node_id,
                float("nan"),
            )

            no_priority_control = (
                no_priority_controls.get(
                    node_id,
                    float("nan"),
                )
            )

            full_score = full_scores[
                node_id
            ]

            no_priority_score = (
                no_priority_scores[node_id]
            )

            delta = (
                no_priority_score
                - full_score
            )

            print(
                f"{node_id:4d} | "
                f"{full_state:10.6f} | "
                f"{no_priority_state:17.6f} | "
                f"{full_control:13.6f} | "
                f"{no_priority_control:18.6f} | "
                f"{full_score:10.6f} | "
                f"{no_priority_score:17.6f} | "
                f"{delta:10.6f}"
            )

        print(
            f"\nSelected full: "
            f"{full.get('selected_node_id', 'n/a')}"
        )

        print(
            f"Selected no_priority: "
            f"{no_priority.get('selected_node_id', 'n/a')}"
        )

        print(
            f"Full selected state: "
            f"{full.get('state', 'n/a')}"
        )

        print(
            f"No-priority selected state: "
            f"{no_priority.get('state', 'n/a')}"
        )

        print(
            f"Full selected control: "
            f"{full.get('control', 'n/a')}"
        )

        print(
            f"No-priority selected control: "
            f"{no_priority.get('control', 'n/a')}"
        )

    print(
        "\n=== Interpretation ==="
    )

    print(
        "Candidate-level state, control, and score "
        "values are now exposed for direct comparison."
    )

    print(
        "The next step is to determine whether the "
        "priority ablation changes candidate ranking, "
        "candidate controls, or only absolute score values."
    )


if __name__ == "__main__":
    main()