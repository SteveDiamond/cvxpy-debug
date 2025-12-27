"""CVXPY Problem Format (CPF) - JSON serialization for CVXPY problems.

This module provides functions to save and load CVXPY problems in a
human-readable JSON format that preserves the full problem structure.

Example:
    >>> import cvxpy as cp
    >>> from cvxpy_debug.format import save_cpf, load_cpf
    >>>
    >>> # Create a problem
    >>> x = cp.Variable(3, name="x", nonneg=True)
    >>> problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1])
    >>>
    >>> # Save to file
    >>> save_cpf(problem, "my_problem.cpf")
    >>>
    >>> # Load from file
    >>> loaded = load_cpf("my_problem.cpf")
    >>> loaded.solve()
"""

from .reader import from_cpf_dict, load_cpf
from .writer import save_cpf, to_cpf_dict

__all__ = [
    "save_cpf",
    "load_cpf",
    "to_cpf_dict",
    "from_cpf_dict",
]
