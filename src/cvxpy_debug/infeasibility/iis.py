"""IIS (Irreducible Infeasible Subsystem) refinement algorithms."""

from __future__ import annotations

from typing import Any

import cvxpy as cp


def find_minimal_iis(
    problem: cp.Problem,
    candidates: list,
    *,
    solver: Any | None = None,
) -> list:
    """
    Find minimal IIS from candidate infeasible constraints.

    Uses the deletion filter algorithm: iteratively remove each constraint
    and check if the remaining set is still infeasible.

    Parameters
    ----------
    problem : cp.Problem
        The original infeasible problem.
    candidates : list
        Candidate constraints (from elastic relaxation).
    solver : optional
        Solver to use for feasibility checks.

    Returns
    -------
    iis : list
        Minimal irreducible infeasible subsystem.
    """
    if len(candidates) <= 1:
        return candidates

    # Start with all candidates
    iis = list(candidates)

    # For each candidate, check if it's essential for infeasibility
    for constraint in candidates:
        if constraint not in iis:
            continue

        # Try without this constraint
        test_set = [c for c in iis if c is not constraint]

        if not test_set:
            # Can't remove the last constraint
            break

        if _is_infeasible(problem, test_set, solver=solver):
            # Still infeasible without this constraint - it's not essential
            iis = test_set

    return iis


def _is_infeasible(
    problem: cp.Problem,
    constraints: list,
    *,
    solver: Any | None = None,
) -> bool:
    """
    Check if a set of constraints is infeasible.

    Parameters
    ----------
    problem : cp.Problem
        Original problem (used for objective/variables context).
    constraints : list
        Constraints to check.
    solver : optional
        Solver to use.

    Returns
    -------
    bool
        True if infeasible, False otherwise.
    """
    # Create a feasibility problem with the given constraints
    # Use a dummy objective
    test_problem = cp.Problem(cp.Minimize(0), constraints)

    try:
        if solver is not None:
            test_problem.solve(solver=solver)
        else:
            test_problem.solve()
    except Exception:
        # Solver error - assume infeasible
        return True

    return test_problem.status in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)


def find_all_iis(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    max_iis: int = 10,
) -> list[list]:
    """
    Find multiple IIS in the problem.

    Some problems have multiple independent sources of infeasibility.
    This function attempts to find them all.

    Parameters
    ----------
    problem : cp.Problem
        The infeasible problem.
    solver : optional
        Solver to use.
    max_iis : int
        Maximum number of IIS to find.

    Returns
    -------
    all_iis : list of lists
        List of all IIS found.

    Notes
    -----
    This is more expensive than finding a single IIS.
    """
    from cvxpy_debug.infeasibility.elastic import find_infeasibility_contributors

    all_iis = []
    remaining_constraints = list(problem.constraints)

    for _ in range(max_iis):
        if not remaining_constraints:
            break

        # Check if remaining constraints are still infeasible
        if not _is_infeasible(problem, remaining_constraints, solver=solver):
            break

        # Find IIS in remaining constraints
        temp_problem = cp.Problem(cp.Minimize(0), remaining_constraints)
        contributors, _ = find_infeasibility_contributors(temp_problem, solver=solver)

        if not contributors:
            break

        iis = find_minimal_iis(temp_problem, contributors, solver=solver)

        if not iis:
            break

        all_iis.append(iis)

        # Remove one constraint from this IIS to find other IIS
        # We remove the first one; could be smarter about this
        constraint_to_remove = iis[0]
        remaining_constraints = [c for c in remaining_constraints if c is not constraint_to_remove]

    return all_iis
