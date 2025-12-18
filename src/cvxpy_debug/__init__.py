"""CVXPY Debug - Diagnostic tools for CVXPY optimization problems."""

from cvxpy_debug.debug import debug
from cvxpy_debug.infeasibility import debug_infeasibility
from cvxpy_debug.numerical import debug_numerical_issues
from cvxpy_debug.performance import debug_performance
from cvxpy_debug.report.report import DebugReport
from cvxpy_debug.unbounded import debug_unboundedness

__version__ = "0.1.0"
__all__ = [
    "debug",
    "debug_infeasibility",
    "debug_unboundedness",
    "debug_numerical_issues",
    "debug_performance",
    "DebugReport",
]
