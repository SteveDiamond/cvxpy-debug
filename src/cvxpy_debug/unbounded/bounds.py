"""Analyze variable bounds from constraints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass
class BoundInfo:
    """Information about bounds on a variable.

    Attributes
    ----------
    lower : float | None
        Lower bound on the variable (None if unbounded below).
    upper : float | None
        Upper bound on the variable (None if unbounded above).
    from_constraints : list
        Constraints that contribute to these bounds.
    """

    lower: float | None
    upper: float | None
    from_constraints: list

    @property
    def is_bounded_below(self) -> bool:
        """Check if variable is bounded below."""
        return self.lower is not None

    @property
    def is_bounded_above(self) -> bool:
        """Check if variable is bounded above."""
        return self.upper is not None

    @property
    def is_fully_bounded(self) -> bool:
        """Check if variable is bounded in both directions."""
        return self.is_bounded_below and self.is_bounded_above


def get_variable_attributes(var: cp.Variable) -> dict[str, Any]:
    """
    Extract bound-related attributes from a CVXPY variable.

    Parameters
    ----------
    var : cp.Variable
        The variable to analyze.

    Returns
    -------
    dict
        Dictionary with 'lower', 'upper' bounds from variable attributes.
    """
    lower = None
    upper = None

    # Check for nonneg attribute
    if hasattr(var, "attributes") and var.attributes.get("nonneg", False):
        lower = 0.0

    # Check for nonpos attribute
    if hasattr(var, "attributes") and var.attributes.get("nonpos", False):
        upper = 0.0

    # Check for explicit bounds
    if hasattr(var, "attributes"):
        if var.attributes.get("pos", False):
            lower = 0.0  # Strictly positive, but for bounds we use 0
        if var.attributes.get("neg", False):
            upper = 0.0  # Strictly negative

    return {"lower": lower, "upper": upper}


def analyze_variable_bounds(problem: cp.Problem) -> dict[int, BoundInfo]:
    """
    Analyze bounds on each variable from constraints and attributes.

    This function examines the problem's constraints and variable
    attributes to determine effective bounds on each variable.

    Parameters
    ----------
    problem : cp.Problem
        The problem to analyze.

    Returns
    -------
    dict[int, BoundInfo]
        Mapping from variable id to BoundInfo.
    """
    bounds: dict[int, BoundInfo] = {}

    # Get all variables
    variables = problem.variables()

    for var in variables:
        var_id = id(var)

        # Start with attribute-based bounds
        attr_bounds = get_variable_attributes(var)
        lower = attr_bounds["lower"]
        upper = attr_bounds["upper"]
        contributing_constraints: list = []

        # Check constraints for simple bounds
        for constraint in problem.constraints:
            bound_info = _extract_bound_from_constraint(constraint, var)
            if bound_info:
                if bound_info["type"] == "lower":
                    if lower is None or bound_info["value"] > lower:
                        lower = bound_info["value"]
                        contributing_constraints.append(constraint)
                elif bound_info["type"] == "upper":
                    if upper is None or bound_info["value"] < upper:
                        upper = bound_info["value"]
                        contributing_constraints.append(constraint)

        bounds[var_id] = BoundInfo(
            lower=lower,
            upper=upper,
            from_constraints=contributing_constraints,
        )

    return bounds


def _extract_bound_from_constraint(constraint: cp.Constraint, var: cp.Variable) -> dict | None:
    """
    Try to extract a bound on var from a constraint.

    Handles simple constraints of the form:
    - var >= constant
    - var <= constant
    - var == constant

    Parameters
    ----------
    constraint : cp.Constraint
        The constraint to analyze.
    var : cp.Variable
        The variable to find bounds for.

    Returns
    -------
    dict | None
        Dictionary with 'type' ('lower', 'upper', or 'both') and 'value',
        or None if no bound can be extracted.
    """
    # Handle different constraint types by name (works across CVXPY versions)
    constraint_type = type(constraint).__name__

    if constraint_type == "Equality" or constraint_type == "Zero":
        return _check_equality_bound(constraint, var)
    elif constraint_type == "Inequality":
        return _check_inequality_bound(constraint, var)
    elif constraint_type == "NonPos":
        return _check_nonpos_bound(constraint, var)
    elif constraint_type == "NonNeg":
        return _check_nonneg_bound(constraint, var)

    return None


def _check_equality_bound(constraint: cp.Constraint, var: cp.Variable) -> dict | None:
    """Check if equality constraint gives bound on var."""
    expr = constraint.args[0]

    # Check if expr is just var - constant or constant - var
    if expr.is_affine() and _is_single_var_expr(expr, var):
        value = _extract_constant_bound(expr, var)
        if value is not None:
            return {"type": "both", "value": value}

    return None


def _check_inequality_bound(constraint: cp.Constraint, var: cp.Variable) -> dict | None:
    """Check if inequality constraint gives bound on var (expr <= 0 form)."""
    expr = constraint.args[0]

    if expr.is_affine() and _is_single_var_expr(expr, var):
        # expr <= 0 means var <= -constant or -var <= -constant
        value = _extract_constant_bound(expr, var)
        if value is not None:
            return {"type": "upper", "value": value}

    return None


def _check_nonpos_bound(constraint: cp.Constraint, var: cp.Variable) -> dict | None:
    """Check NonPos constraint (expr <= 0)."""
    expr = constraint.args[0]

    if expr.is_affine() and _is_single_var_expr(expr, var):
        value = _extract_constant_bound(expr, var)
        if value is not None:
            return {"type": "upper", "value": value}

    return None


def _check_nonneg_bound(constraint: cp.Constraint, var: cp.Variable) -> dict | None:
    """Check NonNeg constraint (expr >= 0)."""
    expr = constraint.args[0]

    if expr.is_affine() and _is_single_var_expr(expr, var):
        value = _extract_constant_bound(expr, var)
        if value is not None:
            return {"type": "lower", "value": value}

    return None


def _is_single_var_expr(expr: cp.Expression, var: cp.Variable) -> bool:
    """Check if expression involves only this single variable."""
    expr_vars = expr.variables()
    if len(expr_vars) != 1:
        return False
    return id(expr_vars[0]) == id(var)


def _extract_constant_bound(expr: cp.Expression, var: cp.Variable) -> float | None:
    """
    Extract constant bound from affine expression.

    For expr = a*var + b, bound on var is -b/a.
    """
    try:
        # Get coefficients - this is a simplified approach
        # For scalar var, check if expression is simply var - c or c - var
        if var.is_scalar():
            # This is a heuristic - works for simple cases
            if hasattr(expr, "args") and len(expr.args) == 2:
                # Binary operation like add/sub
                for arg in expr.args:
                    if arg.is_constant():
                        const_val = arg.value
                        if const_val is not None:
                            # Determine sign of var coefficient
                            # If expr = var - c, bound is c
                            # If expr = -var + c, bound is c
                            return float(-const_val) if np.isscalar(const_val) else None
    except Exception:
        pass

    return None


def get_unbounded_variables(problem: cp.Problem) -> list[tuple[cp.Variable, str]]:
    """
    Find variables that lack bounds in at least one direction.

    Parameters
    ----------
    problem : cp.Problem
        The problem to analyze.

    Returns
    -------
    list[tuple[cp.Variable, str]]
        List of (variable, direction) tuples where direction is
        'below', 'above', or 'both'.
    """
    bounds = analyze_variable_bounds(problem)
    unbounded = []

    for var in problem.variables():
        var_id = id(var)
        info = bounds.get(var_id)

        if info is None:
            unbounded.append((var, "both"))
        elif not info.is_bounded_below and not info.is_bounded_above:
            unbounded.append((var, "both"))
        elif not info.is_bounded_below:
            unbounded.append((var, "below"))
        elif not info.is_bounded_above:
            unbounded.append((var, "above"))

    return unbounded
