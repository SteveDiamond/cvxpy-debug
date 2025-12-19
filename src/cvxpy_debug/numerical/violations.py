"""Constraint violation analysis for CVXPY problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np

from cvxpy_debug.numerical.dataclasses import ConstraintViolation, ViolationAnalysis

# Default tolerances for common solvers
DEFAULT_TOLERANCES: dict[str, dict[str, float]] = {
    "ECOS": {"eps_abs": 1e-8, "eps_rel": 1e-8},
    "SCS": {"eps_abs": 1e-4, "eps_rel": 1e-4},
    "OSQP": {"eps_abs": 1e-3, "eps_rel": 1e-3},
    "MOSEK": {"eps": 1e-8},
    "CLARABEL": {"tol_gap_abs": 1e-8, "tol_gap_rel": 1e-8},
    "CVXOPT": {"abstol": 1e-7, "reltol": 1e-6},
    "GLPK": {"tol": 1e-7},
    "HIGHS": {"primal_feasibility_tolerance": 1e-7},
    "GUROBI": {"FeasibilityTol": 1e-6},
    "CPLEX": {"feasibility_tolerance": 1e-6},
}

# Default tolerance when solver is unknown
DEFAULT_TOLERANCE = 1e-6


def analyze_violations(
    problem: cp.Problem,
    solver_name: str | None = None,
) -> ViolationAnalysis:
    """
    Measure actual constraint violations for a solved problem.

    Parameters
    ----------
    problem : cp.Problem
        The solved CVXPY problem to analyze.
    solver_name : str, optional
        Name of the solver used, for tolerance lookup.

    Returns
    -------
    ViolationAnalysis
        Analysis results including per-constraint violations.
    """
    # Get solver tolerances
    if solver_name and solver_name.upper() in DEFAULT_TOLERANCES:
        solver_tolerances = DEFAULT_TOLERANCES[solver_name.upper()]
    else:
        solver_tolerances = {"default": DEFAULT_TOLERANCE}

    # Get the tolerance to use for comparison
    tolerance = _get_tolerance(solver_tolerances)

    violations: list[ConstraintViolation] = []
    violation_amounts: list[float] = []

    for i, constraint in enumerate(problem.constraints):
        try:
            violation_amount = constraint.violation()
            if violation_amount is None:
                continue

            # Convert to scalar if needed
            if hasattr(violation_amount, "__len__"):
                violation_amount = float(np.max(np.abs(violation_amount)))
            else:
                violation_amount = float(violation_amount)

            # Compute relative violation
            relative_violation = _compute_relative_violation(constraint, violation_amount)

            exceeds_tolerance = violation_amount > tolerance

            violations.append(
                ConstraintViolation(
                    constraint=constraint,
                    constraint_id=i,
                    label=_get_constraint_label(constraint, i),
                    violation_amount=violation_amount,
                    relative_violation=relative_violation,
                    tolerance=tolerance,
                    exceeds_tolerance=exceeds_tolerance,
                )
            )

            if violation_amount > 0:
                violation_amounts.append(violation_amount)

        except (ValueError, AttributeError):
            # Skip constraints that can't be evaluated
            continue

    # Compute aggregate statistics
    total_violations = sum(1 for v in violations if v.exceeds_tolerance)
    max_violation = max(violation_amounts) if violation_amounts else 0.0
    mean_violation = float(np.mean(violation_amounts)) if violation_amounts else 0.0

    return ViolationAnalysis(
        total_violations=total_violations,
        max_violation=max_violation,
        mean_violation=mean_violation,
        violations=violations,
        solver_tolerances=solver_tolerances,
    )


def _get_tolerance(tolerances: dict[str, float]) -> float:
    """Get the primary tolerance value from a tolerances dict."""
    # Prefer absolute tolerance keys
    for key in [
        "eps_abs",
        "abstol",
        "tol_gap_abs",
        "tol",
        "default",
        "primal_feasibility_tolerance",
        "FeasibilityTol",
        "feasibility_tolerance",
        "eps",
    ]:
        if key in tolerances:
            return tolerances[key]
    # Fall back to first value
    if tolerances:
        return next(iter(tolerances.values()))
    return DEFAULT_TOLERANCE


def _compute_relative_violation(constraint: cp.Constraint, violation_amount: float) -> float:
    """
    Compute relative violation as violation / scale.

    The scale is the maximum absolute value in the constraint expression.
    """
    try:
        # Get the expression value for scaling
        if hasattr(constraint, "args") and constraint.args:
            expr = constraint.args[0]
            if expr.value is not None:
                scale = float(np.max(np.abs(expr.value)))
                if scale > 0:
                    return violation_amount / scale
    except Exception:
        pass

    # Fall back to absolute violation
    return violation_amount


def _get_constraint_label(constraint: cp.Constraint, index: int) -> str:
    """Get a human-readable label for a constraint."""
    # Check for user-provided name
    if hasattr(constraint, "name") and constraint.name:
        return constraint.name

    # Generate label from constraint type and index
    constraint_type = type(constraint).__name__
    return f"{constraint_type}[{index}]"
