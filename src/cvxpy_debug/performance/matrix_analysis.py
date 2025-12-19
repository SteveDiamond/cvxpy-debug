"""Constraint matrix structure analysis."""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy import sparse

from cvxpy_debug.performance.dataclasses import MatrixStructure


def analyze_matrix_structure(problem: cp.Problem) -> MatrixStructure | None:
    """
    Analyze the structure of the constraint matrix.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to analyze.

    Returns
    -------
    MatrixStructure or None
        Matrix structure analysis, or None if analysis fails.
    """
    try:
        # Get problem data using SCS format
        data, _, _ = problem.get_problem_data(solver=cp.SCS)

        # Extract constraint matrix (A for equality, may also have G for inequality)
        A = data.get("A")
        if A is None:
            return None

        # Convert to sparse CSR if needed
        if not sparse.issparse(A):
            A = sparse.csr_matrix(A)
        else:
            A = A.tocsr()

        num_rows, num_cols = A.shape
        num_nonzeros = A.nnz

        # Compute sparsity
        total_elements = num_rows * num_cols
        sparsity = num_nonzeros / total_elements if total_elements > 0 else 0

        # Analyze row patterns
        row_pattern_counts = _analyze_row_patterns(A)

        # Find repeated rows
        has_repeated, repeated_groups = _find_repeated_rows(A)

        return MatrixStructure(
            sparsity=sparsity,
            num_rows=num_rows,
            num_cols=num_cols,
            num_nonzeros=num_nonzeros,
            has_repeated_rows=has_repeated,
            repeated_row_groups=repeated_groups,
            row_pattern_counts=row_pattern_counts,
        )

    except Exception:
        return None


def _analyze_row_patterns(A: sparse.csr_matrix) -> dict[str, int]:
    """
    Categorize rows by their non-zero pattern.

    Parameters
    ----------
    A : sparse.csr_matrix
        The constraint matrix in CSR format.

    Returns
    -------
    dict[str, int]
        Pattern type to count mapping.
    """
    patterns: dict[str, int] = {}

    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        nnz_in_row = row_end - row_start

        if nnz_in_row == 0:
            pattern = "empty"
        elif nnz_in_row == 1:
            pattern = "singleton"
        elif nnz_in_row == 2:
            pattern = "pair"
        elif nnz_in_row <= 5:
            pattern = "sparse"
        elif nnz_in_row <= A.shape[1] * 0.1:
            pattern = "moderate"
        else:
            pattern = "dense"

        patterns[pattern] = patterns.get(pattern, 0) + 1

    return patterns


def _find_repeated_rows(A: sparse.csr_matrix) -> tuple[bool, list[list[int]]]:
    """
    Find rows with identical non-zero patterns and values.

    Parameters
    ----------
    A : sparse.csr_matrix
        The constraint matrix in CSR format.

    Returns
    -------
    tuple[bool, list[list[int]]]
        (has_repeated, repeated_groups) where repeated_groups contains
        lists of row indices that are identical.
    """
    # Hash each row for quick comparison
    row_hashes: dict[tuple, list[int]] = {}

    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i + 1]
        indices = tuple(A.indices[row_start:row_end])
        # Round values to avoid floating point comparison issues
        values = tuple(np.round(A.data[row_start:row_end], 10))
        row_key = (indices, values)

        if row_key not in row_hashes:
            row_hashes[row_key] = []
        row_hashes[row_key].append(i)

    # Find groups with more than one row
    repeated_groups = [indices for indices in row_hashes.values() if len(indices) > 1]
    has_repeated = len(repeated_groups) > 0

    return has_repeated, repeated_groups
