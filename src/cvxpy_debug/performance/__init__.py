"""Performance analysis module for CVXPY problems."""

from cvxpy_debug.performance.dataclasses import (
    AntiPattern,
    AntiPatternType,
    MatrixStructure,
    PerformanceAnalysis,
    ProblemMetrics,
)
from cvxpy_debug.performance.diagnose import debug_performance

__all__ = [
    "debug_performance",
    "PerformanceAnalysis",
    "ProblemMetrics",
    "MatrixStructure",
    "AntiPattern",
    "AntiPatternType",
]
