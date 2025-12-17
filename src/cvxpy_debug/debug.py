"""Main entry point for cvxpy-debug."""

from typing import Any

import cvxpy as cp

from cvxpy_debug.infeasibility import diagnose_infeasibility
from cvxpy_debug.report.report import DebugReport
from cvxpy_debug.unbounded import diagnose_unboundedness


def debug(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    verbose: bool = True,
    find_minimal_iis: bool = True,
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
    if problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE):
        diagnose_infeasibility(
            problem,
            report,
            solver=solver,
            find_minimal_iis=find_minimal_iis,
        )
    elif problem.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE):
        diagnose_unboundedness(problem, report, solver=solver)
    elif problem.status == cp.OPTIMAL:
        report.add_finding("Problem solved successfully.")
    else:
        report.add_finding(f"Problem status: {problem.status}")

    if verbose:
        print(report)

    return report
