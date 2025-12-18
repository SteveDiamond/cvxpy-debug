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
from cvxpy_debug.numerical.diagnose import diagnose_numerical_issues

__all__ = [
    "diagnose_numerical_issues",
    "NumericalAnalysis",
    "ScalingAnalysis",
    "ViolationAnalysis",
    "ConstraintViolation",
    "ConditioningAnalysis",
    "SolverStatsAnalysis",
    "SolverRecommendation",
]
