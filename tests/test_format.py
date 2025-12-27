"""Tests for the CPF (CVXPY Problem Format) serialization module."""

from __future__ import annotations

import tempfile
from fractions import Fraction
from pathlib import Path

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from cvxpy_debug.format import from_cpf_dict, load_cpf, save_cpf, to_cpf_dict
from cvxpy_debug.format.values import deserialize_value, serialize_value


class TestValueSerialization:
    """Test serialization of get_data() values."""

    def test_primitives(self):
        """Test primitive types."""
        assert serialize_value(None) is None
        assert serialize_value(True) is True
        assert serialize_value(42) == 42
        assert serialize_value(3.14) == 3.14
        assert serialize_value("hello") == "hello"

    def test_special_floats(self):
        """Test NaN and Inf."""
        nan_ser = serialize_value(float("nan"))
        assert nan_ser == {"$float": "nan"}
        assert np.isnan(deserialize_value(nan_ser))

        inf_ser = serialize_value(float("inf"))
        assert inf_ser == {"$float": "inf"}
        assert deserialize_value(inf_ser) == float("inf")

        neg_inf_ser = serialize_value(float("-inf"))
        assert neg_inf_ser == {"$float": "-inf"}
        assert deserialize_value(neg_inf_ser) == float("-inf")

    def test_fraction(self):
        """Test Fraction serialization."""
        frac = Fraction(5, 2)
        serialized = serialize_value(frac)
        assert serialized == {"$fraction": [5, 2]}
        assert deserialize_value(serialized) == frac

    def test_slice(self):
        """Test slice serialization."""
        s = slice(0, 10, 2)
        serialized = serialize_value(s)
        assert serialized == {"$slice": [0, 10, 2]}
        assert deserialize_value(serialized) == s

    def test_tuple(self):
        """Test tuple serialization."""
        t = (1, 2, 3)
        serialized = serialize_value(t)
        assert serialized == {"$tuple": [1, 2, 3]}
        assert deserialize_value(serialized) == t

    def test_list(self):
        """Test list serialization."""
        lst = [1, 2, 3]
        serialized = serialize_value(lst)
        assert serialized == [1, 2, 3]
        assert deserialize_value(serialized) == lst

    def test_numpy_array_small(self):
        """Test small numpy array is inlined."""
        arr = np.array([[1, 2], [3, 4]])
        serialized = serialize_value(arr)
        assert "$ndarray" in serialized
        result = deserialize_value(serialized)
        np.testing.assert_array_equal(result, arr)

    def test_numpy_array_complex(self):
        """Test complex numpy array."""
        arr = np.array([1 + 2j, 3 + 4j])
        serialized = serialize_value(arr)
        assert "$ndarray" in serialized
        assert "real" in serialized["$ndarray"]
        result = deserialize_value(serialized)
        np.testing.assert_array_equal(result, arr)

    def test_nested_tuple(self):
        """Test nested structures."""
        t = (Fraction(1, 2), [1, 2], slice(0, 5, 1))
        serialized = serialize_value(t)
        result = deserialize_value(serialized)
        assert result[0] == Fraction(1, 2)
        assert result[1] == [1, 2]
        assert result[2] == slice(0, 5, 1)


class TestRoundTrip:
    """Test round-trip serialization of CVXPY problems."""

    def test_simple_lp(self):
        """Test simple linear program."""
        x = cp.Variable(3, name="x", nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 1, x <= 10])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        # Check structure
        assert len(loaded.variables()) == 1
        assert len(loaded.constraints) == 2
        assert isinstance(loaded.objective, cp.Minimize)

        # Solve both and compare
        prob.solve()
        loaded.solve()
        assert np.isclose(prob.value, loaded.value)

    def test_qp(self):
        """Test quadratic program."""
        x = cp.Variable(2, name="x")
        Q = np.eye(2)
        prob = cp.Problem(
            cp.Minimize(cp.quad_form(x, Q)),
            [x >= -1, x <= 1],
        )

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)

    def test_socp(self):
        """Test second-order cone program."""
        x = cp.Variable(3, name="x")
        t = cp.Variable(name="t")
        prob = cp.Problem(
            cp.Minimize(t),
            [cp.norm(x) <= t, x >= -1, cp.sum(x) >= 0],
        )

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)

    def test_sdp(self):
        """Test semidefinite program."""
        X = cp.Variable((2, 2), PSD=True, name="X")
        prob = cp.Problem(
            cp.Minimize(cp.trace(X)),
            [X[0, 0] >= 1, X[1, 1] >= 1],
        )

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)

    def test_parameters(self):
        """Test problem with parameters."""
        x = cp.Variable(2, name="x")
        p = cp.Parameter(2, name="p", value=np.array([1.0, 2.0]))

        prob = cp.Problem(cp.Minimize(p @ x), [x >= 0, cp.sum(x) == 1])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)

    def test_maximize(self):
        """Test maximization problem."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Maximize(x), [x >= 0, x <= 10])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        assert np.isclose(prob.value, loaded.value)

    def test_norm_with_p(self):
        """Test p-norm with different p values."""
        x = cp.Variable(3, name="x")

        for p in [1, 2, np.inf]:
            prob = cp.Problem(cp.Minimize(cp.norm(x, p)), [x >= 1])
            cpf = to_cpf_dict(prob)
            loaded = from_cpf_dict(cpf)

            prob.solve()
            loaded.solve()
            np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-4)

    def test_equality_constraints(self):
        """Test equality constraints."""
        x = cp.Variable(3, name="x")
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.sum(x) == 5, x >= 0],
        )

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)


class TestFileIO:
    """Test file I/O operations."""

    def test_save_and_load(self):
        """Test saving to file and loading back."""
        x = cp.Variable(2, name="x")
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.cpf"
            save_cpf(prob, path)

            # Check file exists
            assert path.exists()

            # Load and verify
            loaded = load_cpf(path)
            prob.solve()
            loaded.solve()
            assert np.isclose(prob.value, loaded.value)

    def test_metadata(self):
        """Test metadata is preserved."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [x >= 0])

        metadata = {"name": "test_problem", "version": "1.0"}
        cpf = to_cpf_dict(prob, metadata=metadata)

        assert cpf["metadata"] == metadata


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_empty_constraints(self):
        """Test problem with no constraints."""
        x = cp.Variable(name="x", nonneg=True)
        prob = cp.Problem(cp.Minimize(x), [])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        assert len(loaded.constraints) == 0

    def test_scalar_variable(self):
        """Test scalar variable."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [x >= 0, x <= 1])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        assert np.isclose(prob.value, loaded.value)

    def test_large_matrix(self):
        """Test with matrix constants."""
        n = 10
        x = cp.Variable(n, name="x")
        A = np.random.randn(5, n)
        b = np.random.randn(5)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])

        cpf = to_cpf_dict(prob)
        loaded = from_cpf_dict(cpf)

        prob.solve()
        loaded.solve()
        np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-4)

    def test_variable_attributes(self):
        """Test various variable attributes are preserved."""
        x1 = cp.Variable(name="x1", nonneg=True)
        x2 = cp.Variable(name="x2", nonpos=True)
        x3 = cp.Variable(name="x3", integer=True)
        X = cp.Variable((2, 2), name="X", PSD=True)

        prob = cp.Problem(
            cp.Minimize(x1 + x2 + x3 + cp.trace(X)),
            [x1 <= 10, x2 >= -10, x3 <= 5, x3 >= 0, X[0, 0] >= 1],
        )

        cpf = to_cpf_dict(prob)

        # Check attributes in serialized form
        assert cpf["variables"]["x1"]["attributes"].get("nonneg") is True
        assert cpf["variables"]["x2"]["attributes"].get("nonpos") is True
        assert cpf["variables"]["x3"]["attributes"].get("integer") is True
        assert cpf["variables"]["X"]["attributes"].get("PSD") is True


class TestExternalData:
    """Test external data file handling for large matrices."""

    def test_large_matrix_externalized(self):
        """Test that large matrices are externalized to .npy files."""
        n = 50  # 50x50 = 2500 elements > 1000 threshold
        x = cp.Variable(n, name="x")
        A = np.random.randn(n, n)
        b = np.random.randn(n)

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x <= b])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large_problem.cpf"
            save_cpf(prob, path, externalize_threshold=1000)

            # Check that data directory was created
            data_dir = Path(tmpdir) / "large_problem.cpf.data"
            assert data_dir.exists(), "Data directory should be created"

            # Check that .npy files exist
            npy_files = list(data_dir.glob("*.npy"))
            assert len(npy_files) > 0, "Should have externalized arrays"

            # Load and verify
            loaded = load_cpf(path)
            prob.solve()
            loaded.solve()
            np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-4)

    def test_small_matrix_inlined(self):
        """Test that small matrices are kept inline."""
        x = cp.Variable(3, name="x")
        A = np.array([[1, 2, 3], [4, 5, 6]])  # 6 elements < 1000
        b = np.array([1, 2])

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "small_problem.cpf"
            save_cpf(prob, path, externalize_threshold=1000)

            # Check that no data directory was created (small matrices inlined)
            data_dir = Path(tmpdir) / "small_problem.cpf.data"
            # Data dir might exist but should be empty or not exist
            if data_dir.exists():
                npy_files = list(data_dir.glob("*.npy"))
                assert len(npy_files) == 0, "Small matrices should be inlined"

            # Load and verify
            loaded = load_cpf(path)
            prob.solve()
            loaded.solve()
            np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-5)

    def test_sparse_matrix_externalized(self):
        """Test sparse matrix externalization."""
        n = 100
        x = cp.Variable(n, name="x")

        # Create sparse constraint matrix
        A = sp.random(50, n, density=0.1, format="csr")
        b = np.random.randn(50)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sparse_problem.cpf"
            save_cpf(prob, path, externalize_threshold=100)

            # Load and verify
            loaded = load_cpf(path)
            prob.solve()
            loaded.solve()
            np.testing.assert_allclose(prob.value, loaded.value, rtol=1e-4)


class TestWithFixtures:
    """Test using existing conftest fixtures."""

    def test_simple_infeasible(self, simple_infeasible):
        """Round-trip simple infeasible problem."""
        cpf = to_cpf_dict(simple_infeasible)
        loaded = from_cpf_dict(cpf)

        # Both should be infeasible
        simple_infeasible.solve()
        loaded.solve()
        assert simple_infeasible.status == loaded.status

    def test_budget_infeasible(self, budget_infeasible):
        """Round-trip budget allocation problem."""
        cpf = to_cpf_dict(budget_infeasible)
        loaded = from_cpf_dict(cpf)

        budget_infeasible.solve()
        loaded.solve()
        assert budget_infeasible.status == loaded.status

    def test_soc_infeasible(self, soc_infeasible):
        """Round-trip SOC problem."""
        cpf = to_cpf_dict(soc_infeasible)
        loaded = from_cpf_dict(cpf)

        soc_infeasible.solve()
        loaded.solve()
        assert soc_infeasible.status == loaded.status

    def test_feasible_problem(self, feasible_problem):
        """Round-trip feasible problem."""
        cpf = to_cpf_dict(feasible_problem)
        loaded = from_cpf_dict(cpf)

        feasible_problem.solve()
        loaded.solve()
        np.testing.assert_allclose(feasible_problem.value, loaded.value, rtol=1e-5)

    def test_well_scaled_problem(self, well_scaled_problem):
        """Round-trip well-scaled problem."""
        cpf = to_cpf_dict(well_scaled_problem)
        loaded = from_cpf_dict(cpf)

        well_scaled_problem.solve()
        loaded.solve()
        np.testing.assert_allclose(well_scaled_problem.value, loaded.value, rtol=1e-4)
