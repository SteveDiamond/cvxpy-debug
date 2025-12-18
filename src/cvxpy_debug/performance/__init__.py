"""Performance analysis module for CVXPY problems."""

from cvxpy_debug.performance.dataclasses import (
    AntiPattern,
    AntiPatternType,
    MatrixStructure,
    PerformanceAnalysis,
    ProblemMetrics,
)
from cvxpy_debug.performance.diagnose import diagnose_performance

__all__ = [
    "diagnose_performance",
    "PerformanceAnalysis",
    "ProblemMetrics",
    "MatrixStructure",
    "AntiPattern",
    "AntiPatternType",
]
