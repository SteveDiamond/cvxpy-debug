"""Find unbounded ray/direction in optimization problems."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass
class UnboundedRay:
    """
    Information about an unbounded direction.

    Attributes
    ----------
    variables : dict
        Mapping from variable to direction coefficient.
        Positive means variable goes to +inf, negative to -inf.
    objective_direction : str
        'decrease' for minimization going to -inf,
        'increase' for maximization going to +inf.
    active_bounds : list
        List of artificial bounds that were active at M or -M.
    """

    variables: dict = field(default_factory=dict)
    objective_direction: str = ""
    active_bounds: list = field(default_factory=list)


def find_unbounded_ray(
    problem: cp.Problem,
    *,
    solver: Any | None = None,
    M: float = 1e6,
) -> UnboundedRay | None:
    """
    Find the direction of unboundedness using bounded relaxation.

    Algorithm:
    1. Add artificial bounds [-M, M] to all variables
    2. Solve the bounded problem
    3. Check which artificial bounds are active (at M or -M)
    4. Those variables indicate unbounded directions

    Parameters
    ----------
    problem : cp.Problem
        The unbounded problem.
    solver : optional
        Solver to use.
    M : float
        Large bound to use for artificial constraints.

    Returns
    -------
    UnboundedRay | None
        Information about unbounded direction, or None if analysis fails.
    """
    # Determine if minimizing or maximizing
    is_minimize = isinstance(problem.objective, cp.Minimize)

    # Get all variables and their current bound info
    variables = problem.variables()

    # Create bounded version of problem
    bounded_constraints = list(problem.constraints)
    artificial_lower: dict[int, cp.Constraint] = {}
    artificial_upper: dict[int, cp.Constraint] = {}

    for var in variables:
        var_id = id(var)

        # Check if variable already has bounds from attributes
        has_lower = _has_lower_bound(var)
        has_upper = _has_upper_bound(var)

        # Add artificial bounds where needed
        if not has_lower:
            lower_constraint = var >= -M
            bounded_constraints.append(lower_constraint)
            artificial_lower[var_id] = lower_constraint

        if not has_upper:
            upper_constraint = var <= M
            bounded_constraints.append(upper_constraint)
            artificial_upper[var_id] = upper_constraint

    # Create and solve bounded problem
    bounded_problem = cp.Problem(problem.objective, bounded_constraints)

    try:
        if solver is not None:
            bounded_problem.solve(solver=solver)
        else:
            bounded_problem.solve()
    except Exception:
        return None

    if bounded_problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        # If bounded problem is still infeasible, original might be infeasible
        # rather than unbounded
        return None

    # Check which artificial bounds are active
    ray = UnboundedRay()
    ray.objective_direction = "decrease" if is_minimize else "increase"

    tol = M * 0.01  # Tolerance for detecting active bound

    for var in variables:
        var_id = id(var)
        var_value = var.value

        if var_value is None:
            continue

        # Handle scalar and array variables
        # Check if variable is scalar by its shape, not by the value type
        # (CVXPY returns numpy arrays even for scalars)
        if var.is_scalar():
            # Scalar variable - get the scalar value
            scalar_val = float(np.atleast_1d(var_value).flat[0])
            _check_bound_active(var, scalar_val, M, tol, ray, artificial_lower, artificial_upper)
        else:
            # For array variables, check each element
            var_array = np.atleast_1d(var_value)
            for idx in np.ndindex(var_array.shape):
                val = var_array[idx]
                _check_bound_active_element(
                    var, idx, val, M, tol, ray, artificial_lower, artificial_upper
                )

    if not ray.variables:
        return None

    return ray


def _has_lower_bound(var: cp.Variable) -> bool:
    """Check if variable has a lower bound from attributes."""
    if hasattr(var, "attributes"):
        attrs = var.attributes
        if attrs.get("nonneg", False):
            return True
        if attrs.get("pos", False):
            return True
    return False


def _has_upper_bound(var: cp.Variable) -> bool:
    """Check if variable has an upper bound from attributes."""
    if hasattr(var, "attributes"):
        attrs = var.attributes
        if attrs.get("nonpos", False):
            return True
        if attrs.get("neg", False):
            return True
    return False


def _check_bound_active(
    var: cp.Variable,
    value: float,
    M: float,
    tol: float,
    ray: UnboundedRay,
    artificial_lower: dict,
    artificial_upper: dict,
) -> None:
    """Check if artificial bound is active for scalar variable."""
    var_id = id(var)

    if var_id in artificial_lower and value <= -M + tol:
        # Lower artificial bound is active - variable wants to go to -inf
        ray.variables[var] = -1.0
        ray.active_bounds.append(("lower", var))
    elif var_id in artificial_upper and value >= M - tol:
        # Upper artificial bound is active - variable wants to go to +inf
        ray.variables[var] = 1.0
        ray.active_bounds.append(("upper", var))


def _check_bound_active_element(
    var: cp.Variable,
    idx: tuple,
    value: float,
    M: float,
    tol: float,
    ray: UnboundedRay,
    artificial_lower: dict,
    artificial_upper: dict,
) -> None:
    """Check if artificial bound is active for array variable element."""
    var_id = id(var)

    # For array variables, we record the index
    key = (var, idx)

    if var_id in artificial_lower and value <= -M + tol:
        ray.variables[key] = -1.0
        ray.active_bounds.append(("lower", var, idx))
    elif var_id in artificial_upper and value >= M - tol:
        ray.variables[key] = 1.0
        ray.active_bounds.append(("upper", var, idx))


def analyze_objective_sensitivity(
    problem: cp.Problem,
    ray: UnboundedRay,
) -> dict[str, Any]:
    """
    Analyze how the objective changes along the unbounded direction.

    Parameters
    ----------
    problem : cp.Problem
        The unbounded problem.
    ray : UnboundedRay
        The unbounded direction.

    Returns
    -------
    dict
        Analysis including which variables most affect objective.
    """
    analysis = {
        "primary_variables": [],
        "explanation": "",
    }

    is_minimize = isinstance(problem.objective, cp.Minimize)
    obj_expr = problem.objective.expr

    # Find which variables in the ray appear in the objective
    for var_key, direction in ray.variables.items():
        # Handle both scalar (var) and indexed ((var, idx)) keys
        if isinstance(var_key, tuple):
            var, idx = var_key
        else:
            var = var_key
            idx = None

        # Check if this variable is in the objective
        if var in obj_expr.variables():
            analysis["primary_variables"].append((var, idx, direction))

    # Generate explanation
    if analysis["primary_variables"]:
        var, idx, direction = analysis["primary_variables"][0]
        var_name = _get_variable_name(var, idx)
        dir_word = "increasing" if direction > 0 else "decreasing"

        if is_minimize:
            if direction > 0:
                analysis["explanation"] = (
                    f"The objective can decrease without bound by {dir_word} {var_name}."
                )
            else:
                analysis["explanation"] = (
                    f"The objective can decrease without bound by {dir_word} {var_name}."
                )
        else:
            analysis["explanation"] = (
                f"The objective can increase without bound by {dir_word} {var_name}."
            )

    return analysis


def _get_variable_name(var: cp.Variable, idx: tuple | None = None) -> str:
    """Get a human-readable name for a variable or indexed element."""
    if var.name():
        name = var.name()
    else:
        name = "x"

    if idx is not None:
        idx_str = ",".join(str(i) for i in idx)
        name = f"{name}[{idx_str}]"

    return name
