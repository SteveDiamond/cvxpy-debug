"""Serialize CVXPY problems to CVX format (.cvx) JSON.

Uses CVXPY's get_data() pattern for uniform atom serialization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cvxpy as cp
import numpy as np

from .values import DataManager, serialize_value

CVX_FORMAT_VERSION = "1.0.0"


class CVXWriter:
    """Serialize a CVXPY Problem to CVX format."""

    def __init__(
        self,
        externalize_threshold: int = 1000,
        include_values: bool = True,
    ):
        """Initialize the writer.

        Args:
            externalize_threshold: Array size threshold for external files
            include_values: Whether to include variable/parameter values
        """
        self.externalize_threshold = externalize_threshold
        self.include_values = include_values
        self._var_registry: dict[int, str] = {}
        self._param_registry: dict[int, str] = {}
        self._const_registry: dict[str, Any] = {}
        self._const_counter = 0
        self._data_manager: DataManager | None = None

    def write(
        self,
        problem: cp.Problem,
        path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Write problem to CVX file.

        Args:
            problem: CVXPY problem to serialize
            path: Output file path (.cvx extension recommended)
            metadata: Optional metadata to include
        """
        path = Path(path)
        self._data_manager = DataManager(path, self.externalize_threshold)

        cvx_data = self._serialize_problem(problem, metadata)

        with open(path, "w") as f:
            json.dump(cvx_data, f, indent=2)

    def to_dict(
        self,
        problem: cp.Problem,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Convert problem to CVX dictionary (no external files).

        Args:
            problem: CVXPY problem to serialize
            metadata: Optional metadata to include

        Returns:
            CVX-format dictionary
        """
        self._data_manager = None
        return self._serialize_problem(problem, metadata)

    def _serialize_problem(
        self,
        problem: cp.Problem,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """Serialize a problem to CVX dictionary."""
        # Reset registries
        self._var_registry = {}
        self._param_registry = {}
        self._const_registry = {}
        self._const_counter = 0

        # First pass: register all variables and parameters
        for var in problem.variables():
            self._register_variable(var)
        for param in problem.parameters():
            self._register_parameter(param)

        # Serialize components
        cvx = {
            "cvx_version": CVX_FORMAT_VERSION,
            "variables": self._serialize_variables(problem.variables()),
            "parameters": self._serialize_parameters(problem.parameters()),
            "constants": {},  # Will be populated during expression serialization
            "objective": self._serialize_objective(problem.objective),
            "constraints": [self._serialize_constraint(c) for c in problem.constraints],
            "metadata": metadata or {},
        }

        # Add constants that were referenced
        cvx["constants"] = self._serialize_constants()

        return cvx

    def _register_variable(self, var: cp.Variable) -> str:
        """Register a variable and return its ID."""
        var_id = id(var)
        if var_id not in self._var_registry:
            name = var.name() if var.name() else f"var_{len(self._var_registry)}"
            self._var_registry[var_id] = name
        return self._var_registry[var_id]

    def _register_parameter(self, param: cp.Parameter) -> str:
        """Register a parameter and return its ID."""
        param_id = id(param)
        if param_id not in self._param_registry:
            name = param.name() if param.name() else f"param_{len(self._param_registry)}"
            self._param_registry[param_id] = name
        return self._param_registry[param_id]

    def _register_constant(self, value: Any) -> str:
        """Register a constant value and return its ID."""
        # Use a simple counter for constants
        const_id = f"const_{self._const_counter}"
        self._const_counter += 1
        self._const_registry[const_id] = value
        return const_id

    def _serialize_variables(self, variables: list) -> dict[str, Any]:
        """Serialize all variables."""
        result = {}
        for var in variables:
            var_key = self._var_registry[id(var)]
            result[var_key] = self._serialize_variable(var)
        return result

    def _serialize_variable(self, var: cp.Variable) -> dict[str, Any]:
        """Serialize a single variable."""
        # Use CVXPY's attributes dictionary directly
        attrs = {}
        for attr in [
            "nonneg",
            "nonpos",
            "pos",
            "neg",
            "symmetric",
            "diag",
            "PSD",
            "NSD",
            "hermitian",
            "boolean",
            "integer",
            "complex",
        ]:
            if var.attributes.get(attr):
                attrs[attr] = True

        data = {
            "shape": list(var.shape),
            "attributes": attrs,
        }

        if self.include_values and var.value is not None:
            data["value"] = serialize_value(var.value, self._data_manager)

        return data

    def _serialize_parameters(self, parameters: list) -> dict[str, Any]:
        """Serialize all parameters."""
        result = {}
        for param in parameters:
            param_key = self._param_registry[id(param)]
            result[param_key] = self._serialize_parameter(param)
        return result

    def _serialize_parameter(self, param: cp.Parameter) -> dict[str, Any]:
        """Serialize a single parameter."""
        # Use CVXPY's attributes dictionary directly
        attrs = {}
        for attr in [
            "nonneg",
            "nonpos",
            "pos",
            "neg",
            "symmetric",
            "diag",
            "PSD",
            "NSD",
            "hermitian",
            "complex",
        ]:
            if param.attributes.get(attr):
                attrs[attr] = True

        data = {
            "shape": list(param.shape),
            "attributes": attrs,
        }

        if param.value is not None:
            data["value"] = serialize_value(param.value, self._data_manager)

        return data

    def _serialize_constants(self) -> dict[str, Any]:
        """Serialize all registered constants."""
        result = {}
        for const_id, value in self._const_registry.items():
            result[const_id] = {
                "value": serialize_value(value, self._data_manager),
            }
        return result

    def _serialize_objective(self, objective: cp.Minimize | cp.Maximize) -> dict[str, Any]:
        """Serialize the objective function."""
        return {
            "sense": "minimize" if isinstance(objective, cp.Minimize) else "maximize",
            "expr": self._serialize_expression(objective.expr),
        }

    def _serialize_constraint(self, constraint) -> dict[str, Any]:
        """Serialize a constraint."""
        constraint_type = type(constraint).__name__
        module = type(constraint).__module__

        result = {
            "type": constraint_type,
            "module": module,
            "args": [self._serialize_expression(arg) for arg in constraint.args],
        }

        # Add get_data() for constraints that have it
        data = constraint.get_data() if hasattr(constraint, "get_data") else None
        if data is not None:
            result["data"] = [serialize_value(d, self._data_manager) for d in data]

        return result

    def _serialize_expression(self, expr) -> dict[str, Any]:
        """Serialize an expression using the get_data() pattern."""
        # Handle leaf nodes
        if isinstance(expr, cp.Variable):
            return {"$var": self._var_registry[id(expr)]}

        if isinstance(expr, cp.Parameter):
            return {"$param": self._param_registry[id(expr)]}

        if isinstance(expr, cp.Constant):
            # For small constants, inline. For large, register and reference.
            value = expr.value
            if isinstance(value, np.ndarray) and value.size > 100:
                const_id = self._register_constant(value)
                return {"$const": const_id}
            return {"$const_inline": serialize_value(value, self._data_manager)}

        # Handle atoms (operations)
        expr_type = type(expr).__name__
        module = type(expr).__module__

        result = {
            "type": expr_type,
            "module": module,
            "args": [self._serialize_expression(arg) for arg in expr.args],
        }

        # Get additional data using get_data()
        data = expr.get_data() if hasattr(expr, "get_data") else None
        if data is not None:
            result["data"] = [serialize_value(d, self._data_manager) for d in data]

        return result


def save_cvx(
    problem: cp.Problem,
    path: str | Path,
    *,
    externalize_threshold: int = 1000,
    include_values: bool = True,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a CVXPY problem to CVX format.

    Args:
        problem: The CVXPY problem to save
        path: Output file path (.cvx extension recommended)
        externalize_threshold: Array size threshold for external files
        include_values: Whether to include current variable/parameter values
        metadata: Optional metadata dictionary to include

    Example:
        >>> import cvxpy as cp
        >>> from cvxpy_debug.format import save_cvx
        >>> x = cp.Variable(3, name="x")
        >>> problem = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        >>> save_cvx(problem, "my_problem.cvx")
    """
    writer = CVXWriter(
        externalize_threshold=externalize_threshold,
        include_values=include_values,
    )
    writer.write(problem, path, metadata=metadata)


def to_cvx_dict(
    problem: cp.Problem,
    *,
    include_values: bool = True,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert a CVXPY problem to CVX dictionary (no external files).

    Args:
        problem: The CVXPY problem to convert
        include_values: Whether to include current variable/parameter values
        metadata: Optional metadata dictionary to include

    Returns:
        CVX-format dictionary
    """
    writer = CVXWriter(include_values=include_values)
    return writer.to_dict(problem, metadata=metadata)
