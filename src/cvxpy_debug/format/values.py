"""Serialize and deserialize get_data() values for the CVX format.

This module handles the special value types that can appear in CVXPY's
get_data() return values, including:
- Primitives (int, float, bool, str, None)
- Fractions
- Slices
- Tuples/lists
- Numpy arrays (inline for small, external ref for large)
- Scipy sparse matrices
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp


def serialize_value(value: Any, data_manager: DataManager | None = None) -> Any:
    """Serialize a value from get_data() to JSON-compatible format.

    Args:
        value: Any value that might appear in get_data() output
        data_manager: Optional manager for externalizing large arrays

    Returns:
        JSON-serializable representation of the value
    """
    if value is None:
        return None

    if isinstance(value, (bool, int, str)):
        return value

    if isinstance(value, float):
        if np.isnan(value):
            return {"$float": "nan"}
        if np.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return value

    if isinstance(value, Fraction):
        return {"$fraction": [value.numerator, value.denominator]}

    if isinstance(value, slice):
        return {"$slice": [value.start, value.stop, value.step]}

    if isinstance(value, tuple):
        return {"$tuple": [serialize_value(v, data_manager) for v in value]}

    if isinstance(value, list):
        return [serialize_value(v, data_manager) for v in value]

    if isinstance(value, np.ndarray):
        return _serialize_ndarray(value, data_manager)

    if sp.issparse(value):
        return _serialize_sparse(value, data_manager)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        if np.isnan(value):
            return {"$float": "nan"}
        if np.isinf(value):
            return {"$float": "inf" if value > 0 else "-inf"}
        return float(value)

    if isinstance(value, np.bool_):
        return bool(value)

    # For CVXPY expressions that might appear in get_data() (rare)
    # We'll handle these separately in the expression serializer
    if hasattr(value, "args") and hasattr(value, "get_data"):
        return {"$expr": True}  # Placeholder - handled by expression serializer

    # Fallback: try to convert to string
    return {"$str": str(value)}


def _serialize_ndarray(arr: np.ndarray, data_manager: DataManager | None) -> Any:
    """Serialize a numpy array."""
    # For small arrays, inline as list
    if arr.size <= 1000 and data_manager is None:
        return _ndarray_to_list(arr)

    # For large arrays or when data_manager is provided, externalize
    if data_manager is not None and arr.size > 1000:
        ref = data_manager.save_array(arr)
        return {"$ref": ref, "shape": list(arr.shape), "dtype": str(arr.dtype)}

    return _ndarray_to_list(arr)


def _ndarray_to_list(arr: np.ndarray) -> dict[str, Any]:
    """Convert numpy array to inline JSON representation."""
    # Handle complex numbers
    if np.iscomplexobj(arr):
        return {
            "$ndarray": {
                "real": arr.real.tolist(),
                "imag": arr.imag.tolist(),
            },
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }
    return {
        "$ndarray": arr.tolist(),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def _serialize_sparse(matrix: sp.spmatrix, data_manager: DataManager | None) -> Any:
    """Serialize a scipy sparse matrix."""
    if data_manager is not None:
        ref = data_manager.save_sparse(matrix)
        return {
            "$sparse_ref": ref,
            "shape": list(matrix.shape),
            "format": matrix.format,
            "nnz": matrix.nnz,
        }

    # Inline sparse as COO format
    coo = matrix.tocoo()
    return {
        "$sparse": {
            "data": coo.data.tolist(),
            "row": coo.row.tolist(),
            "col": coo.col.tolist(),
        },
        "shape": list(matrix.shape),
        "dtype": str(matrix.dtype),
    }


def deserialize_value(obj: Any, data_manager: DataManager | None = None) -> Any:
    """Deserialize a JSON value back to Python object.

    Args:
        obj: JSON-deserialized object
        data_manager: Optional manager for loading external arrays

    Returns:
        Reconstructed Python value
    """
    if obj is None:
        return None

    if isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, list):
        return [deserialize_value(v, data_manager) for v in obj]

    if isinstance(obj, dict):
        # Check for special markers
        if "$fraction" in obj:
            num, denom = obj["$fraction"]
            return Fraction(num, denom)

        if "$slice" in obj:
            start, stop, step = obj["$slice"]
            return slice(start, stop, step)

        if "$tuple" in obj:
            return tuple(deserialize_value(v, data_manager) for v in obj["$tuple"])

        if "$float" in obj:
            val = obj["$float"]
            if val == "nan":
                return float("nan")
            if val == "inf":
                return float("inf")
            if val == "-inf":
                return float("-inf")

        if "$ndarray" in obj:
            return _deserialize_ndarray(obj)

        if "$ref" in obj:
            if data_manager is None:
                raise ValueError(f"External reference {obj['$ref']} but no data manager")
            return data_manager.load_array(obj["$ref"])

        if "$sparse" in obj:
            return _deserialize_sparse_inline(obj)

        if "$sparse_ref" in obj:
            if data_manager is None:
                raise ValueError(f"Sparse reference {obj['$sparse_ref']} but no data manager")
            return data_manager.load_sparse(obj["$sparse_ref"])

        if "$str" in obj:
            return obj["$str"]

        if "$expr" in obj:
            # This should be handled by the expression deserializer
            raise ValueError("Expression marker found in value deserializer")

        # Regular dict (shouldn't happen in get_data but handle gracefully)
        return {k: deserialize_value(v, data_manager) for k, v in obj.items()}

    return obj


def _deserialize_ndarray(obj: dict) -> np.ndarray:
    """Deserialize inline numpy array."""
    data = obj["$ndarray"]
    shape = tuple(obj["shape"])
    dtype = np.dtype(obj["dtype"])

    # Handle complex
    if isinstance(data, dict) and "real" in data:
        real = np.array(data["real"], dtype=np.float64)
        imag = np.array(data["imag"], dtype=np.float64)
        arr = real + 1j * imag
        return arr.reshape(shape).astype(dtype)

    return np.array(data, dtype=dtype).reshape(shape)


def _deserialize_sparse_inline(obj: dict) -> sp.coo_matrix:
    """Deserialize inline sparse matrix."""
    data = obj["$sparse"]
    shape = tuple(obj["shape"])
    dtype = np.dtype(obj["dtype"])

    return sp.coo_matrix(
        (data["data"], (data["row"], data["col"])),
        shape=shape,
        dtype=dtype,
    )


class DataManager:
    """Manages external data files for large arrays."""

    def __init__(self, base_path: Path, threshold: int = 1000):
        """Initialize the data manager.

        Args:
            base_path: Base path for the CVX file (without extension)
            threshold: Size threshold for externalizing arrays
        """
        self.base_path = Path(base_path)
        self.data_dir = self.base_path.parent / f"{self.base_path.name}.data"
        self.threshold = threshold
        self._counter = 0

    def _ensure_data_dir(self) -> None:
        """Create data directory if it doesn't exist."""
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def save_array(self, arr: np.ndarray) -> str:
        """Save array to external file and return reference path."""
        self._ensure_data_dir()
        filename = f"array_{self._counter}.npy"
        self._counter += 1
        filepath = self.data_dir / filename
        np.save(filepath, arr)
        return f"{self.data_dir.name}/{filename}"

    def save_sparse(self, matrix: sp.spmatrix) -> str:
        """Save sparse matrix to external file and return reference path."""
        self._ensure_data_dir()
        filename = f"sparse_{self._counter}.npz"
        self._counter += 1
        filepath = self.data_dir / filename
        sp.save_npz(filepath, matrix.tocsr())
        return f"{self.data_dir.name}/{filename}"

    def load_array(self, ref: str) -> np.ndarray:
        """Load array from external reference."""
        filepath = self.base_path.parent / ref
        return np.load(filepath)

    def load_sparse(self, ref: str) -> sp.spmatrix:
        """Load sparse matrix from external reference."""
        filepath = self.base_path.parent / ref
        return sp.load_npz(filepath)
