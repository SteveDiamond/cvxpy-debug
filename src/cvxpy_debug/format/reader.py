"""Deserialize CPF (CVXPY Problem Format) JSON to CVXPY problems.

Uses CVXPY's reconstruction pattern: type(self)(*(args + data))
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

import cvxpy as cp

from .values import DataManager, deserialize_value

CPF_VERSION = "1.0.0"


class CPFReader:
    """Deserialize a CPF file to a CVXPY Problem."""

    def __init__(self):
        """Initialize the reader."""
        self._var_map: dict[str, cp.Variable] = {}
        self._param_map: dict[str, cp.Parameter] = {}
        self._const_map: dict[str, Any] = {}
        self._data_manager: DataManager | None = None

    def read(self, path: str | Path) -> cp.Problem:
        """Read a CPF file and return a CVXPY Problem.

        Args:
            path: Path to the CPF file

        Returns:
            Reconstructed CVXPY Problem
        """
        path = Path(path)
        self._data_manager = DataManager(path, threshold=0)  # threshold unused for reading

        with open(path) as f:
            cpf_data = json.load(f)

        return self._deserialize_problem(cpf_data)

    def from_dict(self, cpf_data: dict[str, Any]) -> cp.Problem:
        """Reconstruct a Problem from a CPF dictionary.

        Args:
            cpf_data: CPF-format dictionary

        Returns:
            Reconstructed CVXPY Problem
        """
        self._data_manager = None
        return self._deserialize_problem(cpf_data)

    def _deserialize_problem(self, cpf_data: dict[str, Any]) -> cp.Problem:
        """Deserialize the problem from CPF data."""
        # Check version compatibility
        version = cpf_data.get("cpf_version", "0.0.0")
        major = int(version.split(".")[0])
        current_major = int(CPF_VERSION.split(".")[0])
        if major != current_major:
            raise ValueError(
                f"CPF version {version} is not compatible with reader version {CPF_VERSION}"
            )

        # Reset maps
        self._var_map = {}
        self._param_map = {}
        self._const_map = {}

        # First, reconstruct all variables, parameters, and constants
        self._reconstruct_variables(cpf_data.get("variables", {}))
        self._reconstruct_parameters(cpf_data.get("parameters", {}))
        self._reconstruct_constants(cpf_data.get("constants", {}))

        # Reconstruct objective
        objective = self._deserialize_objective(cpf_data["objective"])

        # Reconstruct constraints
        constraints = [self._deserialize_constraint(c) for c in cpf_data.get("constraints", [])]

        return cp.Problem(objective, constraints)

    def _reconstruct_variables(self, variables: dict[str, Any]) -> None:
        """Reconstruct all variables."""
        for var_key, var_data in variables.items():
            self._var_map[var_key] = self._create_variable(var_key, var_data)

    def _create_variable(self, name: str, data: dict[str, Any]) -> cp.Variable:
        """Create a CVXPY Variable from serialized data."""
        shape = tuple(data["shape"])
        attrs = data.get("attributes", {})

        # Build kwargs for Variable constructor
        kwargs: dict[str, Any] = {"name": name}

        # Map attribute names to constructor kwargs
        attr_mapping = {
            "nonneg": "nonneg",
            "nonpos": "nonpos",
            "pos": "pos",
            "neg": "neg",
            "symmetric": "symmetric",
            "diag": "diag",
            "PSD": "PSD",
            "NSD": "NSD",
            "hermitian": "hermitian",
            "boolean": "boolean",
            "integer": "integer",
            "complex": "complex",
        }

        for attr, kwarg in attr_mapping.items():
            if attrs.get(attr):
                kwargs[kwarg] = True

        var = cp.Variable(shape, **kwargs)

        # Set value if present
        if "value" in data:
            value = deserialize_value(data["value"], self._data_manager)
            if value is not None:
                var.value = value

        return var

    def _reconstruct_parameters(self, parameters: dict[str, Any]) -> None:
        """Reconstruct all parameters."""
        for param_key, param_data in parameters.items():
            self._param_map[param_key] = self._create_parameter(param_key, param_data)

    def _create_parameter(self, name: str, data: dict[str, Any]) -> cp.Parameter:
        """Create a CVXPY Parameter from serialized data."""
        shape = tuple(data["shape"])
        attrs = data.get("attributes", {})

        # Build kwargs for Parameter constructor
        kwargs: dict[str, Any] = {"name": name}

        # Map attribute names to constructor kwargs
        attr_mapping = {
            "nonneg": "nonneg",
            "nonpos": "nonpos",
            "pos": "pos",
            "neg": "neg",
            "symmetric": "symmetric",
            "diag": "diag",
            "PSD": "PSD",
            "NSD": "NSD",
            "hermitian": "hermitian",
            "complex": "complex",
        }

        for attr, kwarg in attr_mapping.items():
            if attrs.get(attr):
                kwargs[kwarg] = True

        param = cp.Parameter(shape, **kwargs)

        # Set value if present
        if "value" in data:
            value = deserialize_value(data["value"], self._data_manager)
            if value is not None:
                param.value = value

        return param

    def _reconstruct_constants(self, constants: dict[str, Any]) -> None:
        """Reconstruct all constants."""
        for const_key, const_data in constants.items():
            value = deserialize_value(const_data["value"], self._data_manager)
            self._const_map[const_key] = value

    def _deserialize_objective(self, obj_data: dict[str, Any]) -> cp.Minimize | cp.Maximize:
        """Deserialize the objective function."""
        sense = obj_data["sense"]
        expr = self._deserialize_expression(obj_data["expr"])

        if sense == "minimize":
            return cp.Minimize(expr)
        else:
            return cp.Maximize(expr)

    def _deserialize_constraint(self, const_data: dict[str, Any]) -> Any:
        """Deserialize a constraint."""
        const_type = const_data["type"]
        module = const_data["module"]

        # Get the constraint class
        cls = self._get_class(module, const_type)

        # Deserialize args
        args = [self._deserialize_expression(arg) for arg in const_data.get("args", [])]

        # Deserialize data
        data = const_data.get("data")
        if data is not None:
            data = [deserialize_value(d, self._data_manager) for d in data]
            # Filter out constraint ID if present (last element is often the ID)
            # We want new IDs for the reconstructed constraints
            if data and isinstance(data[-1], int):
                data = data[:-1]
            return cls(*args, *data) if data else cls(*args)

        return cls(*args)

    def _deserialize_expression(self, expr_data: dict[str, Any]) -> Any:
        """Deserialize an expression using the reconstruction pattern."""
        # Handle leaf references
        if "$var" in expr_data:
            var_key = expr_data["$var"]
            if var_key not in self._var_map:
                raise ValueError(f"Unknown variable reference: {var_key}")
            return self._var_map[var_key]

        if "$param" in expr_data:
            param_key = expr_data["$param"]
            if param_key not in self._param_map:
                raise ValueError(f"Unknown parameter reference: {param_key}")
            return self._param_map[param_key]

        if "$const" in expr_data:
            const_key = expr_data["$const"]
            if const_key not in self._const_map:
                raise ValueError(f"Unknown constant reference: {const_key}")
            value = self._const_map[const_key]
            return cp.Constant(value)

        if "$const_inline" in expr_data:
            value = deserialize_value(expr_data["$const_inline"], self._data_manager)
            return cp.Constant(value)

        # Handle atoms (operations)
        expr_type = expr_data["type"]
        module = expr_data["module"]

        # Deserialize args
        args = [self._deserialize_expression(arg) for arg in expr_data.get("args", [])]

        # Deserialize data
        data = expr_data.get("data")
        if data is not None:
            data = [deserialize_value(d, self._data_manager) for d in data]

        # Special handling for atoms that use Python operators instead of constructors
        # AddExpression: use + operator
        if expr_type == "AddExpression":
            if len(args) == 0:
                return cp.Constant(0)
            result = args[0]
            for arg in args[1:]:
                result = result + arg
            return result

        # NegExpression: use unary - operator
        if expr_type == "NegExpression":
            return -args[0]

        # MulExpression: use * operator (for scalar multiplication)
        if expr_type == "MulExpression":
            return args[0] * args[1] if len(args) == 2 else args[0]

        # multiply (elementwise): use cp.multiply
        if expr_type == "multiply":
            return cp.multiply(args[0], args[1])

        # DivExpression: use / operator
        if expr_type == "DivExpression":
            return args[0] / args[1]

        # Get the atom class for standard atoms
        cls = self._get_class(module, expr_type)

        # Standard reconstruction: cls(*args, *data)
        if data is not None:
            return cls(*args, *data)
        return cls(*args)

    def _get_class(self, module: str, class_name: str) -> type:
        """Get a class from module and class name."""
        try:
            mod = importlib.import_module(module)
            return getattr(mod, class_name)
        except (ImportError, AttributeError) as e:
            # Try common CVXPY locations
            common_modules = [
                "cvxpy",
                "cvxpy.atoms",
                "cvxpy.atoms.affine",
                "cvxpy.atoms.elementwise",
                "cvxpy.constraints",
                "cvxpy.constraints.zero",
                "cvxpy.constraints.nonpos",
                "cvxpy.constraints.second_order",
                "cvxpy.constraints.psd",
                "cvxpy.constraints.exponential",
                "cvxpy.constraints.power",
            ]
            for mod_name in common_modules:
                try:
                    mod = importlib.import_module(mod_name)
                    if hasattr(mod, class_name):
                        return getattr(mod, class_name)
                except ImportError:
                    continue

            raise ValueError(f"Cannot find class {class_name} in module {module}") from e


def load_cpf(path: str | Path) -> cp.Problem:
    """Load a CVXPY problem from a CPF file.

    Args:
        path: Path to the CPF file

    Returns:
        Reconstructed CVXPY Problem

    Example:
        >>> from cvxpy_debug.format import load_cpf
        >>> problem = load_cpf("my_problem.cpf")
        >>> problem.solve()
    """
    reader = CPFReader()
    return reader.read(path)


def from_cpf_dict(cpf_data: dict[str, Any]) -> cp.Problem:
    """Reconstruct a CVXPY problem from a CPF dictionary.

    Args:
        cpf_data: CPF-format dictionary

    Returns:
        Reconstructed CVXPY Problem
    """
    reader = CPFReader()
    return reader.from_dict(cpf_data)
