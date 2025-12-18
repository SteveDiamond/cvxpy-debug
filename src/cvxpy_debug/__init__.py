"""CVXPY Debug - Diagnostic tools for CVXPY optimization problems."""

from cvxpy_debug.debug import debug
from cvxpy_debug.numerical import diagnose_numerical_issues
from cvxpy_debug.report.report import DebugReport
from cvxpy_debug.unbounded import diagnose_unboundedness

__version__ = "0.1.0"
__all__ = ["debug", "DebugReport", "diagnose_unboundedness", "diagnose_numerical_issues"]
