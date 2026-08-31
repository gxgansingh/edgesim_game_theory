"""Entry point for utility-weight sensitivity experiments."""

from .config import SimulationConfig
from .experiments.utility_sensitivity import run_utility_weight_sensitivity
from .experiments.visualization import generate_utility_sensitivity_figures


def main() -> None:
    """Run the configured utility-weight sensitivity experiment."""
    config = SimulationConfig()

    seeds = tuple(
        range(
            config.experiment_seed_start,
            config.experiment_seed_start + config.experiment_repetitions,
        )
    )

    result = run_utility_weight_sensitivity(
        config=config,
        seeds=seeds,
        output_directory=(
            f"{config.output_directory}/utility_sensitivity"
        ),
    )

    print("Utility-weight sensitivity experiment completed.")
    print(f"Profiles: {result.raw_results['profile'].nunique()}")
    print(f"Scenarios: {result.raw_results['scenario'].nunique()}")
    print(f"Seeds per profile/scenario: {len(seeds)}")

    for profile in result.raw_results["profile"].unique():
        print(f"\n{'=' * 72}")
        print(f"Utility profile: {profile}")
        print(f"{'=' * 72}")

        weights = result.raw_results.loc[
            result.raw_results["profile"] == profile,
            [
                "priority_reward_weight",
                "latency_cost_weight",
                "resource_cost_weight",
                "queue_cost_weight",
                "energy_cost_weight",
            ],
        ].iloc[0]

        print("Weights:")
        print(
            f"priority={weights['priority_reward_weight']:.2f} | "
            f"latency={weights['latency_cost_weight']:.2f} | "
            f"resource={weights['resource_cost_weight']:.2f} | "
            f"queue={weights['queue_cost_weight']:.2f} | "
            f"energy={weights['energy_cost_weight']:.2f}"
        )

        profile_comparison = result.paired_comparison.loc[
            result.paired_comparison["profile"] == profile
        ]

        for scenario in profile_comparison["scenario"].unique():
            print(f"\nScenario: {scenario}")
            scenario_rows = profile_comparison.loc[
                profile_comparison["scenario"] == scenario
            ]

            for _, row in scenario_rows.iterrows():
                print(
                    f"{row['metric']} | "
                    f"delta={row['mean_difference']:.6f} | "
                    f"relative={row['relative_change_percent']:.2f}% | "
                    f"p={row['p_value']:.6f}"
                )

    results_directory = (
        f"{config.output_directory}/utility_sensitivity"
    )

    print(f"\nResults saved to: {results_directory}")

    figures = generate_utility_sensitivity_figures(
        results_directory
    )

    print("\nFigures generated:")
    for figure in figures:
        print(f"Figure saved to: {figure}")


if __name__ == "__main__":
    main()
