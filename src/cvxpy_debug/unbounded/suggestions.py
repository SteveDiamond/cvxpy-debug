"""Generate suggestions for fixing unbounded problems."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from cvxpy_debug.unbounded.ray import UnboundedRay


def generate_suggestions(
    problem: cp.Problem,
    ray: UnboundedRay | None,
    unbounded_vars: list[tuple[cp.Variable, str]],
) -> list[str]:
    """
    Generate suggestions for fixing an unbounded problem.

    Parameters
    ----------
    problem : cp.Problem
        The unbounded problem.
    ray : UnboundedRay | None
        Information about unbounded direction.
    unbounded_vars : list[tuple[Variable, str]]
        Variables lacking bounds and their direction ('above', 'below', 'both').

    Returns
    -------
    list[str]
        List of suggestion strings.
    """
    suggestions = []

    # Suggest bounds based on ray analysis
    if ray and ray.variables:
        for var_key, direction in ray.variables.items():
            # Handle both scalar and indexed variables
            if isinstance(var_key, tuple):
                var, idx = var_key
                var_name = _get_var_name(var, idx)
            else:
                var = var_key
                var_name = _get_var_name(var)

            if direction > 0:
                # Variable going to +inf
                suggestions.append(f"Add upper bound to '{var_name}' (e.g., {var_name} <= <value>)")
            else:
                # Variable going to -inf
                suggestions.append(f"Add lower bound to '{var_name}' (e.g., {var_name} >= <value>)")

    # If no ray but we have unbounded variables, suggest based on that
    elif unbounded_vars:
        for var, direction in unbounded_vars:
            var_name = _get_var_name(var)

            if direction == "both":
                suggestions.append(
                    f"'{var_name}' has no bounds. Consider adding "
                    f"{var_name} >= 0 or other appropriate bounds."
                )
            elif direction == "below":
                suggestions.append(
                    f"'{var_name}' has no lower bound. Consider adding " f"{var_name} >= <value>."
                )
            elif direction == "above":
                suggestions.append(
                    f"'{var_name}' has no upper bound. Consider adding " f"{var_name} <= <value>."
                )

    # Generic suggestions
    if not suggestions:
        suggestions.append(
            "Add bounds to your variables or additional constraints to "
            "limit the feasible region."
        )

    # Add suggestion about nonnegativity if applicable
    if _should_suggest_nonneg(problem, ray, unbounded_vars):
        suggestions.append("If variables should be nonnegative, use cp.Variable(nonneg=True).")

    return suggestions


def _get_var_name(var: cp.Variable, idx: tuple | None = None) -> str:
    """Get human-readable variable name."""
    name = var.name() if var.name() else "x"

    if idx is not None:
        idx_str = ",".join(str(i) for i in idx)
        name = f"{name}[{idx_str}]"

    return name


def _should_suggest_nonneg(
    problem: cp.Problem,
    ray: UnboundedRay | None,
    unbounded_vars: list,
) -> bool:
    """Check if we should suggest nonneg=True."""
    # Suggest if any variable is unbounded below
    if ray:
        for var_key, direction in ray.variables.items():
            if direction < 0:  # Going to -inf
                return True

    for var, direction in unbounded_vars:
        if direction in ("below", "both"):
            # Check if var already has nonneg
            if not _is_nonneg(var):
                return True

    return False


def _is_nonneg(var: cp.Variable) -> bool:
    """Check if variable has nonneg attribute."""
    return hasattr(var, "attributes") and var.attributes.get("nonneg", False)


def format_unbounded_variable_info(
    var: cp.Variable,
    direction: str,
    idx: tuple | None = None,
) -> dict[str, Any]:
    """
    Format information about an unbounded variable.

    Parameters
    ----------
    var : cp.Variable
        The variable.
    direction : str
        Direction of unboundedness ('above', 'below', 'both').
    idx : tuple | None
        Index for array variables.

    Returns
    -------
    dict
        Formatted information.
    """
    var_name = _get_var_name(var, idx)

    direction_symbols = {
        "above": "+inf",
        "below": "-inf",
        "both": "+/-inf",
    }

    return {
        "name": var_name,
        "direction": direction,
        "direction_symbol": direction_symbols.get(direction, "?"),
        "variable": var,
        "index": idx,
    }
