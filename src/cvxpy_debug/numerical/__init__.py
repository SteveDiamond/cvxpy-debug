"""Numerical diagnostics module for CVXPY problems."""

from cvxpy_debug.numerical.dataclasses import (
    ConditioningAnalysis,
    ConstraintViolation,
    NumericalAnalysis,
    ScalingAnalysis,
    SolverRecommendation,
    SolverStatsAnalysis,
    ViolationAnalysis,
)
from cvxpy_debug.numerical.diagnose import debug_numerical_issues

__all__ = [
    "debug_numerical_issues",
    "NumericalAnalysis",
    "ScalingAnalysis",
    "ViolationAnalysis",
    "ConstraintViolation",
    "ConditioningAnalysis",
    "SolverStatsAnalysis",
    "SolverRecommendation",
]
