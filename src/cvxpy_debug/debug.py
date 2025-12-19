"""Main entry point for cvxpy-debug."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from cvxpy_debug.infeasibility import debug_infeasibility
from cvxpy_debug.numerical import debug_numerical_issues
from cvxpy_debug.performance import debug_performance
from cvxpy_debug.report.report import DebugReport
from cvxpy_debug.unbounded import debug_unboundedness


def debug(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    verbose: bool = True,
    find_minimal_iis: bool = True,
    include_conditioning: bool = True,
    include_performance: bool = True,
) -> DebugReport:
    """
    Debug a CVXPY optimization problem.

    Analyzes the problem to identify issues such as infeasibility,
    unboundedness, or numerical problems.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to debug.
    solver : optional
        Solver to use for diagnostic solves. If None, uses default.
    verbose : bool, default True
        If True, print the diagnostic report.
    find_minimal_iis : bool, default True
        If True, refine infeasibility diagnosis to find minimal
        irreducible infeasible subsystem. Slower but more precise.
    include_conditioning : bool, default True
        If True, estimate condition number for numerical analysis.
        Can be slow for large problems.
    include_performance : bool, default True
        If True, analyze problem structure for performance anti-patterns
        like loop-generated constraints.

    Returns
    -------
    DebugReport
        Diagnostic report with findings and suggestions.

    Examples
    --------
    >>> import cvxpy as cp
    >>> import cvxpy_debug
    >>> x = cp.Variable()
    >>> prob = cp.Problem(cp.Minimize(x), [x >= 5, x <= 3])
    >>> prob.solve()
    >>> report = cvxpy_debug.debug(prob)
    """
    # First, check if problem has been solved
    if problem.status is None:
        # Problem hasn't been solved yet, solve it first
        try:
            if solver is not None:
                problem.solve(solver=solver)
            else:
                problem.solve()
        except Exception:
            pass  # We'll diagnose based on whatever state we have

    # Create base report
    report = DebugReport(problem=problem)

    # Diagnose based on problem status
    if problem.status == cp.INFEASIBLE:
        debug_infeasibility(
            problem,
            report,
            solver=solver,
            find_minimal_iis=find_minimal_iis,
        )
    elif problem.status == cp.UNBOUNDED:
        debug_unboundedness(problem, report, solver=solver)
    elif problem.status == cp.OPTIMAL:
        report.add_finding("Problem solved successfully.")
    elif problem.status in (
        cp.OPTIMAL_INACCURATE,
        cp.INFEASIBLE_INACCURATE,
        cp.UNBOUNDED_INACCURATE,
    ):
        # Run numerical diagnostics for inaccurate statuses
        debug_numerical_issues(
            problem,
            report,
            solver=solver,
            include_conditioning=include_conditioning,
        )
        # Also run base diagnosis for _INACCURATE variants
        if problem.status == cp.INFEASIBLE_INACCURATE:
            debug_infeasibility(
                problem,
                report,
                solver=solver,
                find_minimal_iis=find_minimal_iis,
            )
        elif problem.status == cp.UNBOUNDED_INACCURATE:
            debug_unboundedness(problem, report, solver=solver)
    else:
        report.add_finding(f"Problem status: {problem.status}")

    # Always run performance analysis if requested
    if include_performance:
        debug_performance(problem, report)

    if verbose:
        print(report)

    return report
