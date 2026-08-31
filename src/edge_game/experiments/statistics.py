"""Statistical analysis utilities for repeated policy experiments."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats


METRIC_COLUMNS = [
    "utility_mean",
    "response_time_mean",
    "throughput",
    "success_ratio",
    "rejected_tasks",
    "resource_utilization",
    "load_variance",
    "jains_fairness_index",
    "average_queue_length",
    "priority_success_ratio",
]


def benjamini_hochberg(
    p_values: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> tuple[list[float], list[bool]]:
    """Apply Benjamini-Hochberg false-discovery-rate correction."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1.")

    values = [float(value) for value in p_values]

    if not values:
        return [], []

    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("All p-values must be between 0 and 1.")

    indexed = sorted(
        enumerate(values),
        key=lambda item: item[1],
    )

    adjusted = [1.0] * len(values)
    running_minimum = 1.0
    count = len(values)

    for rank in range(count, 0, -1):
        original_index, p_value = indexed[rank - 1]
        adjusted_value = min(
            p_value * count / rank,
            running_minimum,
            1.0,
        )
        running_minimum = adjusted_value
        adjusted[original_index] = adjusted_value

    significant = [
        adjusted_value <= alpha
        for adjusted_value in adjusted
    ]

    return adjusted, significant


def _confidence_interval(
    values: np.ndarray,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a two-sided Student-t confidence interval."""
    values = np.asarray(
        values,
        dtype=float,
    )

    values = values[
        np.isfinite(values)
    ]

    if len(values) < 2:
        mean = (
            float(np.mean(values))
            if len(values)
            else 0.0
        )

        return mean, mean

    mean = float(
        np.mean(values)
    )

    standard_error = float(
        stats.sem(values)
    )

    alpha = 1.0 - confidence

    critical_value = float(
        stats.t.ppf(
            1.0 - alpha / 2.0,
            df=len(values) - 1,
        )
    )

    margin = (
        critical_value
        * standard_error
    )

    return (
        mean - margin,
        mean + margin,
    )


def aggregate_policy_metrics(
    results: pd.DataFrame,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Aggregate repeated policy metrics with uncertainty estimates."""
    rows: list[
        dict[str, float | int | str]
    ] = []

    for policy_name, policy_frame in results.groupby(
        "policy_name",
        sort=True,
    ):
        for metric in METRIC_COLUMNS:
            values = policy_frame[
                metric
            ].to_numpy(
                dtype=float
            )

            finite_values = values[
                np.isfinite(values)
            ]

            if len(finite_values) == 0:
                continue

            ci_low, ci_high = (
                _confidence_interval(
                    finite_values,
                    confidence=confidence,
                )
            )

            rows.append(
                {
                    "policy_name": policy_name,
                    "metric": metric,
                    "n": int(
                        len(finite_values)
                    ),
                    "mean": float(
                        np.mean(
                            finite_values
                        )
                    ),
                    "std": float(
                        np.std(
                            finite_values,
                            ddof=1,
                        )
                    )
                    if len(finite_values) > 1
                    else 0.0,
                    "median": float(
                        np.median(
                            finite_values
                        )
                    ),
                    "minimum": float(
                        np.min(
                            finite_values
                        )
                    ),
                    "maximum": float(
                        np.max(
                            finite_values
                        )
                    ),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

    return pd.DataFrame(rows)


def paired_policy_comparison(
    results: pd.DataFrame,
    baseline_policy: str = (
        "least_loaded_baseline"
    ),
    comparison_policy: str = (
        "priority_aware_mean_field"
    ),
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Calculate paired seed-level differences between two policies."""
    baseline = results.loc[
        results["policy_name"]
        == baseline_policy
    ].copy()

    comparison = results.loc[
        results["policy_name"]
        == comparison_policy
    ].copy()

    baseline = baseline.set_index(
        "seed"
    )

    comparison = comparison.set_index(
        "seed"
    )

    common_seeds = (
        baseline.index.intersection(
            comparison.index
        )
    )

    rows: list[
        dict[str, float | int | str]
    ] = []

    for metric in METRIC_COLUMNS:
        baseline_values = (
            baseline.loc[
                common_seeds,
                metric,
            ].to_numpy(
                dtype=float
            )
        )

        comparison_values = (
            comparison.loc[
                common_seeds,
                metric,
            ].to_numpy(
                dtype=float
            )
        )

        differences = (
            comparison_values
            - baseline_values
        )

        finite_mask = np.isfinite(
            differences
        )

        differences = differences[
            finite_mask
        ]

        if len(differences) == 0:
            continue

        ci_low, ci_high = (
            _confidence_interval(
                differences,
                confidence=confidence,
            )
        )

        if len(differences) >= 2:
            difference_std = float(
                np.std(
                    differences,
                    ddof=1,
                )
            )

            if difference_std <= 1e-12:
                p_value = (
                    1.0
                    if abs(
                        float(
                            np.mean(
                                differences
                            )
                        )
                    ) <= 1e-12
                    else 0.0
                )
            else:
                test = stats.ttest_1samp(
                    differences,
                    popmean=0.0,
                )

                p_value = float(
                    test.pvalue
                )
        else:
            p_value = math.nan

        baseline_mean = float(
            np.mean(
                baseline_values[
                    finite_mask
                ]
            )
        )

        comparison_mean = float(
            np.mean(
                comparison_values[
                    finite_mask
                ]
            )
        )

        if abs(
            baseline_mean
        ) > 1e-12:
            relative_change = (
                (
                    comparison_mean
                    - baseline_mean
                )
                / abs(baseline_mean)
            ) * 100.0
        else:
            relative_change = math.nan

        rows.append(
            {
                "baseline_policy": (
                    baseline_policy
                ),
                "comparison_policy": (
                    comparison_policy
                ),
                "metric": metric,
                "n": int(
                    len(differences)
                ),
                "baseline_mean": (
                    baseline_mean
                ),
                "comparison_mean": (
                    comparison_mean
                ),
                "mean_difference": float(
                    np.mean(
                        differences
                    )
                ),
                "relative_change_percent": (
                    relative_change
                ),
                "std_difference": float(
                    np.std(
                        differences,
                        ddof=1,
                    )
                )
                if len(differences) > 1
                else 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": p_value,
            }
        )

    return pd.DataFrame(rows)

SELECTION_METRIC_COLUMNS = [
    "selected_cpu_capacity",
    "selected_load_ratio",
    "selected_queue_length",
]



def _wilson_interval(
    successes: int,
    trials: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Calculate a Wilson confidence interval for a proportion."""
    if trials <= 0:
        return math.nan, math.nan

    z = float(
        stats.norm.ppf(
            1.0 - (1.0 - confidence) / 2.0
        )
    )

    proportion = successes / trials
    denominator = 1.0 + (z * z) / trials
    center = (
        proportion
        + (z * z) / (2.0 * trials)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (
                proportion * (1.0 - proportion)
                / trials
            )
            + (z * z) / (4.0 * trials * trials)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)

def policy_selection_summary(
    selection_records: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize node-selection characteristics by policy."""
    if selection_records.empty:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "metric",
                "n",
                "mean",
                "std",
                "median",
                "minimum",
                "maximum",
            ]
        )

    renamed = selection_records.rename(
        columns={
            "cpu_capacity": "selected_cpu_capacity",
            "load_ratio": "selected_load_ratio",
            "queue_length": "selected_queue_length",
        }
    )

    rows: list[dict] = []

    for policy_name, policy_frame in renamed.groupby(
        "policy_name",
        sort=True,
    ):
        for metric in SELECTION_METRIC_COLUMNS:
            values = policy_frame[metric].to_numpy(
                dtype=float
            )

            rows.append(
                {
                    "policy_name": policy_name,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(
                        np.std(values, ddof=1)
                    ) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def paired_selection_analysis(
    selection_records: pd.DataFrame,
    baseline_policy: str = "least_loaded_baseline",
    comparison_policy: str = "priority_aware_mean_field",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare paired node-selection decisions across policies."""
    if selection_records.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    baseline = selection_records.loc[
        selection_records["policy_name"] == baseline_policy
    ].copy()

    comparison = selection_records.loc[
        selection_records["policy_name"] == comparison_policy
    ].copy()

    merge_columns = [
        "seed",
        "task_id",
    ]

    baseline = baseline.rename(
        columns={
            "node_id": "baseline_node_id",
            "cpu_capacity": "baseline_cpu_capacity",
            "load_ratio": "baseline_load_ratio",
            "queue_length": "baseline_queue_length",
            "priority": "baseline_priority",
        }
    )

    comparison = comparison.rename(
        columns={
            "node_id": "comparison_node_id",
            "cpu_capacity": "comparison_cpu_capacity",
            "load_ratio": "comparison_load_ratio",
            "queue_length": "comparison_queue_length",
            "priority": "comparison_priority",
        }
    )

    merged = baseline.merge(
        comparison,
        on=merge_columns,
        how="inner",
    )

    if merged.empty:
        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )

    merged["same_node"] = (
        merged["baseline_node_id"]
        == merged["comparison_node_id"]
    )

    merged["diverged"] = (
        ~merged["same_node"]
    ).astype(float)

    rows: list[dict] = []

    def add_difference_metric(
        metric: str,
        baseline_values: np.ndarray,
        comparison_values: np.ndarray,
    ) -> None:
        differences = (
            comparison_values
            - baseline_values
        )

        if len(differences) == 0:
            return

        if len(differences) >= 2:
            std_difference = float(
                np.std(
                    differences,
                    ddof=1,
                )
            )

            if std_difference <= 1e-12:
                p_value = (
                    1.0
                    if abs(float(np.mean(differences))) <= 1e-12
                    else 0.0
                )
            else:
                p_value = float(
                    stats.ttest_1samp(
                        differences,
                        popmean=0.0,
                    ).pvalue
                )
        else:
            p_value = math.nan

        baseline_mean = float(
            np.mean(baseline_values)
        )
        comparison_mean = float(
            np.mean(comparison_values)
        )

        if abs(baseline_mean) > 1e-12:
            relative_change = (
                (comparison_mean - baseline_mean)
                / abs(baseline_mean)
            ) * 100.0
        else:
            relative_change = math.nan

        ci_low, ci_high = _confidence_interval(
            differences
        )

        rows.append(
            {
                "metric": metric,
                "n": int(len(differences)),
                "baseline_mean": baseline_mean,
                "comparison_mean": comparison_mean,
                "mean_difference": float(
                    np.mean(differences)
                ),
                "relative_change_percent": relative_change,
                "std_difference": (
                    float(
                        np.std(
                            differences,
                            ddof=1,
                        )
                    )
                    if len(differences) > 1
                    else 0.0
                ),
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": p_value,
            }
        )

    for metric, baseline_column, comparison_column in (
        (
            "selected_cpu_capacity",
            "baseline_cpu_capacity",
            "comparison_cpu_capacity",
        ),
        (
            "selected_load_ratio",
            "baseline_load_ratio",
            "comparison_load_ratio",
        ),
        (
            "selected_queue_length",
            "baseline_queue_length",
            "comparison_queue_length",
        ),
    ):
        add_difference_metric(
            metric=metric,
            baseline_values=merged[baseline_column].to_numpy(
                dtype=float
            ),
            comparison_values=merged[comparison_column].to_numpy(
                dtype=float
            ),
        )

    def add_divergence_metric(
        metric: str,
        frame: pd.DataFrame,
    ) -> None:
        """Add a divergence proportion with a Wilson interval."""
        trials = int(len(frame))
        successes = int(frame["diverged"].sum())

        if trials == 0:
            return

        rate = successes / trials
        ci_low, ci_high = _wilson_interval(
            successes=successes,
            trials=trials,
        )

        rows.append(
            {
                "metric": metric,
                "n": trials,
                "baseline_mean": 0.0,
                "comparison_mean": rate,
                "mean_difference": rate,
                "relative_change_percent": math.nan,
                "std_difference": 0.0,
                "ci95_low": ci_low,
                "ci95_high": ci_high,
                "p_value": math.nan,
                "statistical_test": "wilson_proportion_interval",
            }
        )

    add_divergence_metric(
        metric="selection_divergence_rate",
        frame=merged,
    )

    for priority in (1, 2, 3):
        priority_frame = merged.loc[
            merged["baseline_priority"] == priority
        ]

        if not priority_frame.empty:
            add_divergence_metric(
                metric=f"priority_{priority}_selection_divergence_rate",
                frame=priority_frame,
            )

    comparison = pd.DataFrame(rows)

    frequency = (
        selection_records.groupby(
            ["policy_name", "node_id"],
            as_index=False,
        )
        .agg(
            selection_count=("task_id", "count"),
            mean_selected_cpu_capacity=(
                "cpu_capacity",
                "mean",
            ),
            mean_selected_load_ratio=(
                "load_ratio",
                "mean",
            ),
            mean_selected_queue_length=(
                "queue_length",
                "mean",
            ),
        )
    )

    return comparison, frequency

DECISION_AUDIT_METRICS = [
    "candidate_count",
    "selected_rank",
    "selected_score",
    "score_margin",
    "score_tie_count",
    "state",
    "control",
    "cpu_load",
    "memory_load",
    "bandwidth_load",
    "queue_pressure",
    "energy_pressure",
]


def decision_audit_summary(
    decision_records: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize candidate-set and policy-decision diagnostics."""
    if decision_records.empty:
        return pd.DataFrame(
            columns=[
                "policy_name",
                "metric",
                "n",
                "mean",
                "std",
                "median",
                "minimum",
                "maximum",
            ]
        )

    rows: list[dict] = []

    for policy_name, frame in decision_records.groupby(
        "policy_name",
        sort=True,
    ):
        for metric in DECISION_AUDIT_METRICS:
            if metric not in frame.columns:
                continue

            values = frame[metric].to_numpy(
                dtype=float
            )
            values = values[np.isfinite(values)]

            if len(values) == 0:
                continue

            rows.append(
                {
                    "policy_name": policy_name,
                    "metric": metric,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(
                        np.std(values, ddof=1)
                    ) if len(values) > 1 else 0.0,
                    "median": float(np.median(values)),
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )

    return pd.DataFrame(rows)


def paired_decision_audit(
    decision_records: pd.DataFrame,
    baseline_policy: str = "least_loaded_baseline",
    comparison_policy: str = "priority_aware_mean_field",
) -> pd.DataFrame:
    """Compare candidate sets and decisions for matched task instances."""
    required_columns = {
        "seed",
        "task_id",
        "priority",
        "policy_name",
        "candidate_count",
        "candidate_node_ids",
        "selected_node_id",
        "selected_rank",
        "selected_score",
        "score_margin",
        "score_tie_count",
        "state",
        "control",
        "mean_field_score",
    }

    if decision_records.empty:
        return pd.DataFrame()

    missing = required_columns.difference(
        decision_records.columns
    )

    if missing:
        raise ValueError(
            "Decision audit is missing columns: "
            + ", ".join(sorted(missing))
        )

    baseline = decision_records.loc[
        decision_records["policy_name"] == baseline_policy
    ].copy()

    comparison = decision_records.loc[
        decision_records["policy_name"] == comparison_policy
    ].copy()

    baseline = baseline.rename(
        columns={
            column: f"baseline_{column}"
            for column in baseline.columns
            if column not in {"seed", "task_id"}
        }
    )

    comparison = comparison.rename(
        columns={
            column: f"comparison_{column}"
            for column in comparison.columns
            if column not in {"seed", "task_id"}
        }
    )

    merged = baseline.merge(
        comparison,
        on=["seed", "task_id"],
        how="inner",
    )

    if merged.empty:
        return pd.DataFrame()

    rows: list[dict] = []

    def add_numeric_difference(
        metric: str,
    ) -> None:
        baseline_values = merged[
            f"baseline_{metric}"
        ].to_numpy(dtype=float)
        comparison_values = merged[
            f"comparison_{metric}"
        ].to_numpy(dtype=float)

        differences = (
            comparison_values
            - baseline_values
        )

        finite = np.isfinite(differences)
        differences = differences[finite]

        if len(differences) == 0:
            return

        rows.append(
            {
                "metric": metric,
                "n": int(len(differences)),
                "baseline_mean": float(
                    np.mean(baseline_values[finite])
                ),
                "comparison_mean": float(
                    np.mean(comparison_values[finite])
                ),
                "mean_difference": float(
                    np.mean(differences)
                ),
            }
        )

    for metric in DECISION_AUDIT_METRICS:
        if (
            f"baseline_{metric}" not in merged.columns
            or f"comparison_{metric}" not in merged.columns
        ):
            continue
        add_numeric_difference(metric)

    candidate_set_same = (
        merged["baseline_candidate_node_ids"]
        == merged["comparison_candidate_node_ids"]
    )

    selected_node_same = (
        merged["baseline_selected_node_id"]
        == merged["comparison_selected_node_id"]
    )

    rows.extend(
        [
            {
                "metric": "candidate_set_identity_rate",
                "n": int(len(merged)),
                "baseline_mean": 1.0,
                "comparison_mean": float(
                    candidate_set_same.mean()
                ),
                "mean_difference": float(
                    candidate_set_same.mean()
                ),
            },
            {
                "metric": "selected_node_identity_rate",
                "n": int(len(merged)),
                "baseline_mean": 1.0,
                "comparison_mean": float(
                    selected_node_same.mean()
                ),
                "mean_difference": float(
                    selected_node_same.mean()
                ),
            },
        ]
    )

    for priority in (1, 2, 3):
        priority_frame = merged.loc[
            merged["baseline_priority"] == priority
        ]

        if priority_frame.empty:
            continue

        priority_same = (
            priority_frame["baseline_selected_node_id"]
            == priority_frame["comparison_selected_node_id"]
        )

        rows.append(
            {
                "metric": (
                    f"priority_{priority}_selected_node_identity_rate"
                ),
                "n": int(len(priority_frame)),
                "baseline_mean": 1.0,
                "comparison_mean": float(
                    priority_same.mean()
                ),
                "mean_difference": float(
                    priority_same.mean()
                ),
            }
        )

    return pd.DataFrame(rows)