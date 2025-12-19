"""Condition number estimation for CVXPY problems."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy import sparse

from cvxpy_debug.numerical.dataclasses import ConditioningAnalysis

# Threshold for considering a problem ill-conditioned
ILL_CONDITIONED_THRESHOLD = 1e12

# Maximum size for exact condition number computation
MAX_SIZE_EXACT = 1000


def analyze_conditioning(problem: cp.Problem) -> ConditioningAnalysis:
    """
    Estimate problem conditioning where feasible.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to analyze.

    Returns
    -------
    ConditioningAnalysis
        Analysis results including condition number estimate.
    """
    matrix_info: dict = {}

    try:
        # Try to get problem data for a conic solver
        # This gives us the constraint matrices
        data, _, _ = problem.get_problem_data(cp.SCS)

        # Extract the A matrix (equality constraints) or G matrix (inequality)
        A = data.get("A")  # Equality constraint matrix
        G = data.get("G")  # Inequality constraint matrix

        # Use whichever matrix is available and larger
        matrix = None
        if A is not None and hasattr(A, "shape"):
            matrix = A
            matrix_info["type"] = "equality_constraints"
            matrix_info["shape"] = A.shape
        if G is not None and hasattr(G, "shape"):
            if matrix is None or (G.shape[0] * G.shape[1] > matrix.shape[0] * matrix.shape[1]):
                matrix = G
                matrix_info["type"] = "inequality_constraints"
                matrix_info["shape"] = G.shape

        if matrix is None:
            return ConditioningAnalysis(
                estimated=False,
                condition_number=None,
                ill_conditioned=False,
                matrix_info={"error": "No constraint matrix found"},
            )

        # Convert sparse matrix to dense if needed and small enough
        if sparse.issparse(matrix):
            matrix_info["sparse"] = True
            matrix_info["nnz"] = matrix.nnz

            # For large sparse matrices, use randomized SVD
            if matrix.shape[0] * matrix.shape[1] > MAX_SIZE_EXACT * MAX_SIZE_EXACT:
                condition_number = _estimate_condition_sparse(matrix)
                matrix_info["estimation_method"] = "randomized_svd"
            else:
                matrix = matrix.toarray()
                condition_number = _compute_condition_number(matrix)
                matrix_info["estimation_method"] = "exact"
        else:
            matrix_info["sparse"] = False
            condition_number = _compute_condition_number(matrix)
            matrix_info["estimation_method"] = "exact"

        ill_conditioned = (
            condition_number is not None and condition_number > ILL_CONDITIONED_THRESHOLD
        )

        return ConditioningAnalysis(
            estimated=True,
            condition_number=condition_number,
            ill_conditioned=ill_conditioned,
            matrix_info=matrix_info,
        )

    except Exception as e:
        return ConditioningAnalysis(
            estimated=False,
            condition_number=None,
            ill_conditioned=False,
            matrix_info={"error": str(e)},
        )


def _compute_condition_number(matrix: np.ndarray) -> float | None:
    """
    Compute the condition number of a matrix.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix to analyze.

    Returns
    -------
    float | None
        The condition number, or None if computation fails.
    """
    try:
        # Use 2-norm condition number
        cond = np.linalg.cond(matrix)
        if np.isfinite(cond):
            return float(cond)
        return None
    except Exception:
        return None


def _estimate_condition_sparse(matrix: sparse.spmatrix) -> float | None:
    """
    Estimate condition number for large sparse matrices using randomized SVD.

    Parameters
    ----------
    matrix : sparse matrix
        The sparse matrix to analyze.

    Returns
    -------
    float | None
        Estimated condition number, or None if estimation fails.
    """
    try:
        from scipy.sparse.linalg import svds

        # Get largest and smallest singular values
        k = min(6, min(matrix.shape) - 1)
        if k < 1:
            return None

        # Largest singular values
        _, s_large, _ = svds(matrix.astype(float), k=k, which="LM")
        sigma_max = np.max(s_large)

        # Smallest singular values (may fail for rank-deficient matrices)
        try:
            _, s_small, _ = svds(matrix.astype(float), k=k, which="SM")
            sigma_min = np.min(s_small[s_small > 1e-15])
        except Exception:
            # If smallest SVD fails, use smallest from large SVD
            sigma_min = np.min(s_large[s_large > 1e-15])

        if sigma_min > 0:
            return float(sigma_max / sigma_min)
        return float("inf")

    except Exception:
        return None
