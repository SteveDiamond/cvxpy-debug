"""Constraint relaxation utilities."""

from __future__ import annotations

import cvxpy as cp
import numpy as np


def relax_constraint(constraint: cp.Constraint) -> tuple[cp.Variable | None, list]:
    """
    Create a relaxed version of a constraint with slack variable.

    Parameters
    ----------
    constraint : cp.Constraint
        The constraint to relax.

    Returns
    -------
    slack : cp.Variable or None
        The slack variable (None if constraint can't be relaxed).
    relaxed : list
        List of relaxed constraints.
    """
    # Check for Equality first (x == y becomes Zero constraint internally)
    if isinstance(constraint, cp.constraints.zero.Equality):
        return _relax_equality(constraint)

    # Check for Inequality (x >= y or x <= y both become Inequality)
    if isinstance(constraint, cp.constraints.nonpos.Inequality):
        return _relax_inequality(constraint)

    # Check for SOC constraint
    if isinstance(constraint, cp.constraints.second_order.SOC):
        return _relax_soc(constraint)

    # Check for PSD constraint
    if isinstance(constraint, cp.constraints.psd.PSD):
        return _relax_psd(constraint)

    # Check for exponential cone
    if isinstance(constraint, cp.constraints.exponential.ExpCone):
        return _relax_exp_cone(constraint)

    # Unknown constraint type - can't relax, return unchanged
    return None, [constraint]


def _relax_equality(constraint):
    """
    Relax equality constraint: expr == 0 -> |expr| <= slack.
    """
    # Equality constraint has args[0] - args[1] == 0
    lhs = constraint.args[0]
    rhs = constraint.args[1]
    expr = lhs - rhs

    # Use scalar slack for simplicity, sum of absolute deviations
    slack = cp.Variable(nonneg=True)

    if expr.is_scalar():
        relaxed = [expr <= slack, expr >= -slack]
    else:
        # For vector/matrix, use sum of element-wise absolute values
        relaxed = [cp.sum(cp.abs(expr)) <= slack]

    return slack, relaxed


def _relax_inequality(constraint):
    """
    Relax inequality constraint: lhs <= rhs -> lhs <= rhs + slack.
    """
    lhs = constraint.args[0]
    rhs = constraint.args[1]

    # Use scalar slack
    slack = cp.Variable(nonneg=True)

    # Relax: lhs <= rhs + slack
    if lhs.is_scalar() and rhs.is_scalar():
        relaxed = [lhs <= rhs + slack]
    else:
        # For vectors, add slack to each element
        relaxed = [lhs <= rhs + slack]

    return slack, relaxed


def _relax_soc(constraint: cp.constraints.second_order.SOC):
    """
    Relax SOC constraint: ||x||_2 <= t -> ||x||_2 <= t + slack.

    SOC constraint is: norm2(x) <= t, stored as SOC(t, x).
    """
    # SOC stores (t, X) where the constraint is ||X||_2 <= t
    t = constraint.args[0]  # The bound
    X = constraint.args[1]  # The vector being bounded

    slack = cp.Variable(nonneg=True)

    # Relaxed: ||X||_2 <= t + slack
    relaxed = [cp.SOC(t + slack, X)]

    return slack, relaxed


def _relax_psd(constraint: cp.constraints.psd.PSD):
    """
    Relax PSD constraint: X >> 0 -> X + slack*I >> 0.
    """
    X = constraint.args[0]
    n = X.shape[0]

    slack = cp.Variable(nonneg=True)

    # Add slack to diagonal
    relaxed = [X + slack * np.eye(n) >> 0]

    return slack, relaxed


def _relax_exp_cone(constraint: cp.constraints.exponential.ExpCone):
    """
    Relax exponential cone constraint.

    ExpCone(x, y, z) means: y * exp(x/y) <= z, y > 0
    We relax by adding slack to z.
    """
    x = constraint.args[0]
    y = constraint.args[1]
    z = constraint.args[2]

    slack = cp.Variable(nonneg=True)

    # Relaxed: y * exp(x/y) <= z + slack
    relaxed = [cp.constraints.exponential.ExpCone(x, y, z + slack)]

    return slack, relaxed
