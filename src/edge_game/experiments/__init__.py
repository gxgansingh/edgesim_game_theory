"""Repeated experiment execution and statistical analysis."""

from .runner import (
    RepeatedExperimentResult,
    build_mean_field_policy,
    run_repeated_experiments,
)
from .statistics import (
    aggregate_policy_metrics,
    paired_policy_comparison,
)

__all__ = [
    "RepeatedExperimentResult",
    "build_mean_field_policy",
    "run_repeated_experiments",
    "aggregate_policy_metrics",
    "paired_policy_comparison",
]