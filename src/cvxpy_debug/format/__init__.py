"""CVX format (.cvx) - JSON serialization for CVXPY problems.

This module provides functions to save and load CVXPY problems in a
human-readable JSON format that preserves the full problem structure.

Example:
    >>> import cvxpy as cp
    >>> from cvxpy_debug.format import save_cvx, load_cvx
    >>>
    >>> # Create a problem
    >>> x = cp.Variable(3, name="x", nonneg=True)
    >>> problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
    >>>
    >>> # Save to file
    >>> save_cvx(problem, "my_problem.cvx")
    >>>
    >>> # Load from file
    >>> loaded = load_cvx("my_problem.cvx")
    >>> loaded.solve()
"""

from .reader import from_cvx_dict, load_cvx
from .writer import save_cvx, to_cvx_dict

__all__ = [
    "save_cvx",
    "load_cvx",
    "to_cvx_dict",
    "from_cvx_dict",
]
