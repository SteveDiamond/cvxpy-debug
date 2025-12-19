"""Coefficient scaling analysis for CVXPY problems."""

from typing import Any

import cvxpy as cp
import numpy as np

from cvxpy_debug.numerical.dataclasses import ScalingAnalysis

# Thresholds for scaling detection
BADLY_SCALED_THRESHOLD = 1e6  # ratio above this is problematic
VERY_SMALL_THRESHOLD = 1e-8  # coefficients below this
VERY_LARGE_THRESHOLD = 1e8  # coefficients above this


def analyze_scaling(problem: cp.Problem) -> ScalingAnalysis:
    """
    Analyze coefficient magnitudes in objective and constraints.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to analyze.

    Returns
    -------
    ScalingAnalysis
        Analysis results including coefficient ranges and scaling issues.
    """
    all_coefficients: list[float] = []
    very_small: list[dict] = []
    very_large: list[dict] = []

    # Analyze objective
    obj_coeffs = _extract_coefficients(problem.objective.expr)
    obj_range = _compute_range(obj_coeffs)
    all_coefficients.extend(obj_coeffs)

    # Check for extreme coefficients in objective
    for coeff in obj_coeffs:
        abs_coeff = abs(coeff)
        if abs_coeff > 0 and abs_coeff < VERY_SMALL_THRESHOLD:
            very_small.append({"location": "objective", "value": coeff})
        elif abs_coeff > VERY_LARGE_THRESHOLD:
            very_large.append({"location": "objective", "value": coeff})

    # Analyze constraints
    constraint_ranges: dict[int, tuple[float, float]] = {}
    for i, constraint in enumerate(problem.constraints):
        coeffs = _extract_constraint_coefficients(constraint)
        if coeffs:
            constraint_ranges[i] = _compute_range(coeffs)
            all_coefficients.extend(coeffs)

            # Check for extreme coefficients
            for coeff in coeffs:
                abs_coeff = abs(coeff)
                if abs_coeff > 0 and abs_coeff < VERY_SMALL_THRESHOLD:
                    very_small.append({"location": f"constraint_{i}", "value": coeff})
                elif abs_coeff > VERY_LARGE_THRESHOLD:
                    very_large.append({"location": f"constraint_{i}", "value": coeff})

    # Compute overall range ratio
    overall_range = _compute_range(all_coefficients)
    if overall_range[0] > 0:
        overall_ratio = overall_range[1] / overall_range[0]
    else:
        overall_ratio = float("inf") if overall_range[1] > 0 else 1.0

    badly_scaled = overall_ratio > BADLY_SCALED_THRESHOLD

    return ScalingAnalysis(
        objective_range=obj_range,
        constraint_ranges=constraint_ranges,
        overall_range_ratio=overall_ratio,
        badly_scaled=badly_scaled,
        very_small_coefficients=very_small,
        very_large_coefficients=very_large,
    )


def _extract_coefficients(expr: Any) -> list[float]:
    """
    Extract numerical coefficients from a CVXPY expression.

    Parameters
    ----------
    expr : Expression
        A CVXPY expression.

    Returns
    -------
    list[float]
        List of numerical coefficient values.
    """
    coefficients: list[float] = []

    try:
        # Get constants from the expression
        for const in expr.constants():
            values = np.asarray(const.value).flatten()
            for v in values:
                if np.isfinite(v) and v != 0:
                    coefficients.append(float(abs(v)))

        # Get parameters (their current values)
        for param in expr.parameters():
            if param.value is not None:
                values = np.asarray(param.value).flatten()
                for v in values:
                    if np.isfinite(v) and v != 0:
                        coefficients.append(float(abs(v)))

    except Exception:
        # If extraction fails, return empty list
        pass

    return coefficients


def _extract_constraint_coefficients(constraint: cp.Constraint) -> list[float]:
    """
    Extract coefficients from a constraint.

    Parameters
    ----------
    constraint : cp.Constraint
        A CVXPY constraint.

    Returns
    -------
    list[float]
        List of numerical coefficient values.
    """
    coefficients: list[float] = []

    try:
        # Handle different constraint types
        if hasattr(constraint, "args"):
            for arg in constraint.args:
                coefficients.extend(_extract_coefficients(arg))
    except Exception:
        pass

    return coefficients


def _compute_range(coefficients: list[float]) -> tuple[float, float]:
    """
    Compute the min and max absolute values of coefficients.

    Parameters
    ----------
    coefficients : list[float]
        List of coefficient values.

    Returns
    -------
    tuple[float, float]
        (min_abs, max_abs) of non-zero coefficients.
    """
    if not coefficients:
        return (0.0, 0.0)

    abs_coeffs = [abs(c) for c in coefficients if c != 0]
    if not abs_coeffs:
        return (0.0, 0.0)

    return (min(abs_coeffs), max(abs_coeffs))
