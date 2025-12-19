"""Elastic relaxation method for finding infeasibility contributors."""

from __future__ import annotations

from typing import Any

import cvxpy as cp
import numpy as np

from cvxpy_debug.infeasibility.relaxation import relax_constraint


def find_infeasibility_contributors(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    tol: float = 1e-6,
) -> tuple[list, dict]:
    """
    Find constraints contributing to infeasibility using elastic relaxation.

    Adds a slack variable to each constraint and minimizes total slack.
    Constraints with non-zero slack are infeasibility contributors.

    Parameters
    ----------
    problem : cp.Problem
        The infeasible problem.
    solver : optional
        Solver to use. If None, uses default.
    tol : float
        Tolerance for considering slack non-zero.

    Returns
    -------
    contributors : list
        List of constraints that contribute to infeasibility.
    slack_values : dict
        Mapping from constraint id to slack value.
    """
    if not problem.constraints:
        return [], {}

    # Build relaxed problem
    slacks = []
    relaxed_constraints = []
    constraint_to_slack = {}  # Map original constraint to its slack variable

    for constraint in problem.constraints:
        slack, relaxed = relax_constraint(constraint)
        if slack is not None:
            slacks.append(slack)
            constraint_to_slack[id(constraint)] = slack
        relaxed_constraints.extend(relaxed)

    # Also handle variable bounds as implicit constraints
    bound_slacks, bound_constraints, bound_mapping = _create_bound_relaxations(problem)
    slacks.extend(bound_slacks)
    relaxed_constraints.extend(bound_constraints)

    if not slacks:
        return [], {}

    # Minimize total slack (L1 relaxation)
    # All slacks are now scalar, so Python sum works
    total_slack = sum(slacks)
    objective = cp.Minimize(total_slack)
    relaxed_problem = cp.Problem(objective, relaxed_constraints)

    # Solve the relaxed problem
    try:
        if solver is not None:
            relaxed_problem.solve(solver=solver)
        else:
            relaxed_problem.solve()
    except Exception as e:
        # If even the relaxed problem fails, we can't diagnose
        raise RuntimeError(f"Could not solve relaxed problem: {e}") from e

    if relaxed_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # Relaxed problem should always be feasible
        raise RuntimeError(f"Relaxed problem has unexpected status: {relaxed_problem.status}")

    # Find constraints with non-zero slack
    contributors = []
    slack_values = {}

    for constraint in problem.constraints:
        if id(constraint) in constraint_to_slack:
            slack_var = constraint_to_slack[id(constraint)]
            slack_val = _get_slack_value(slack_var)
            slack_values[id(constraint)] = slack_val
            if slack_val > tol:
                contributors.append(constraint)

    # Check bound violations
    for var_id, (slack_var, description) in bound_mapping.items():
        slack_val = _get_slack_value(slack_var)
        if slack_val > tol:
            # Create a synthetic constraint object for reporting
            # This is a simplified approach - in practice we'd want better handling
            slack_values[f"bound_{var_id}"] = slack_val

    return contributors, slack_values


def _get_slack_value(slack_var: cp.Variable) -> float:
    """Get the value of a slack variable, handling edge cases."""
    if slack_var.value is None:
        return 0.0
    val = float(np.sum(slack_var.value))  # Sum in case of vector slack
    return max(0.0, val)  # Ensure non-negative


def _create_bound_relaxations(
    problem: cp.Problem,
) -> tuple[list, list, dict]:
    """
    Create slack variables for variable bounds.

    Returns
    -------
    slacks : list
        Slack variables for bounds.
    constraints : list
        Relaxed bound constraints.
    mapping : dict
        Mapping from variable id to (slack, description).
    """
    slacks = []
    constraints = []
    mapping = {}

    for var in problem.variables():
        var_id = id(var)

        # Handle nonneg constraint
        if var.is_nonneg():
            # Use scalar slack - relax as: var >= -slack (element-wise)
            slack = cp.Variable(nonneg=True)
            slacks.append(slack)
            constraints.append(var >= -slack)
            mapping[var_id] = (slack, f"{var.name()} >= 0")

        # Handle nonpos constraint
        elif var.is_nonpos():
            slack = cp.Variable(nonneg=True)
            slacks.append(slack)
            constraints.append(var <= slack)
            mapping[var_id] = (slack, f"{var.name()} <= 0")

        # Handle PSD constraint
        elif hasattr(var, "is_psd") and var.is_psd():
            # For PSD, we add slack to the diagonal
            n = var.shape[0]
            slack = cp.Variable(nonneg=True)
            slacks.append(slack)
            constraints.append(var + slack * np.eye(n) >> 0)
            mapping[var_id] = (slack, f"{var.name()} >> 0")

        # Handle NSD constraint
        elif hasattr(var, "is_nsd") and var.is_nsd():
            n = var.shape[0]
            slack = cp.Variable(nonneg=True)
            slacks.append(slack)
            constraints.append(var - slack * np.eye(n) << 0)
            mapping[var_id] = (slack, f"{var.name()} << 0")

    return slacks, constraints, mapping
