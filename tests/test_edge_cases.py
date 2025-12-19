"""Tests for edge cases and error handling."""

import cvxpy as cp
import numpy as np

from cvxpy_debug import debug


class TestEmptyProblems:
    """Tests for empty or trivial problems."""

    def test_no_constraints(self):
        """Test problem with no constraints."""
        x = cp.Variable(name="x")

        # Unbounded - no constraints
        prob = cp.Problem(cp.Minimize(x), [])

        report = debug(prob)

        assert report is not None
        # Should detect unboundedness
        assert "unbounded" in str(report).lower() or len(report.findings) > 0

    def test_single_variable_bounded(self):
        """Test trivial single-variable bounded problem."""
        x = cp.Variable(nonneg=True, name="x")

        prob = cp.Problem(cp.Minimize(x), [x <= 10])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_zero_dimensional_problem(self):
        """Test problem that effectively has no degrees of freedom."""
        x = cp.Variable(name="x")

        # Fully determined
        prob = cp.Problem(cp.Minimize(x), [x == 5])
        prob.solve()

        report = debug(prob)

        assert report is not None


class TestExtremeCoefficients:
    """Tests for extreme numerical values."""

    def test_coefficient_near_zero(self):
        """Test handling of very small coefficients."""
        x = cp.Variable(3, name="x")

        # Coefficients near machine epsilon
        A = np.array([[1e-15, 1, 1], [1, 1e-15, 1], [1, 1, 1e-15]])
        b = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_coefficient_very_large(self):
        """Test handling of very large coefficients."""
        x = cp.Variable(3, name="x")

        # Large coefficients (but not so large as to cause solver failure)
        A = np.array([[1e6, 1, 1], [1, 1e6, 1], [1, 1, 1e6]])
        b = np.ones(3) * 1e6

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

        try:
            prob.solve()
            report = debug(prob)
            assert report is not None
        except cp.SolverError:
            # Solver may fail on extreme problems - that's acceptable
            pass

    def test_mixed_extreme_coefficients(self):
        """Test handling of mixed extreme coefficients."""
        x = cp.Variable(3, name="x")

        # Mix of very small and very large
        A = np.array([[1e-10, 1e10, 1], [1e10, 1e-10, 1], [1, 1, 1]])
        b = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

        # Should not crash
        try:
            prob.solve()
            report = debug(prob)
            assert report is not None
        except cp.SolverError:
            # Solver may fail on extreme problems
            pass


class TestSpecialValues:
    """Tests for special numerical values."""

    def test_problem_with_zeros(self):
        """Test problem with many zero coefficients."""
        x = cp.Variable(5, name="x")

        # Sparse constraint matrix
        A = np.zeros((3, 5))
        A[0, 0] = 1
        A[1, 2] = 1
        A[2, 4] = 1
        b = np.ones(3)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_integer_coefficients(self):
        """Test with integer coefficients."""
        x = cp.Variable(3, name="x")

        A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float)
        b = np.array([10, 20, 30], dtype=float)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])
        prob.solve()

        report = debug(prob)

        assert report is not None


class TestConstraintVariety:
    """Tests for various constraint configurations."""

    def test_only_equality_constraints(self):
        """Test problem with only equality constraints."""
        x = cp.Variable(3, name="x")

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [x[0] + x[1] == 1, x[1] + x[2] == 2])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_only_inequality_constraints(self):
        """Test problem with only inequality constraints."""
        x = cp.Variable(3, name="x")

        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 10, cp.sum(x) >= 5])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_many_constraints(self):
        """Test problem with many constraints."""
        n = 50
        x = cp.Variable(n, name="x")

        constraints = [x >= 0, x <= 1]
        for i in range(n - 1):
            constraints.append(x[i] + x[i + 1] <= 1.5)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)
        prob.solve()

        report = debug(prob)

        assert report is not None


class TestVariableTypes:
    """Tests for different variable configurations."""

    def test_vector_variable(self):
        """Test with vector variables."""
        x = cp.Variable(10, name="x")

        prob = cp.Problem(cp.Minimize(cp.norm(x)), [cp.sum(x) == 1, x >= 0])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_matrix_variable(self):
        """Test with matrix variables."""
        X = cp.Variable((3, 3), name="X")

        prob = cp.Problem(cp.Minimize(cp.norm(X, "fro")), [cp.sum(X) == 1])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_symmetric_variable(self):
        """Test with symmetric matrix variable."""
        X = cp.Variable((3, 3), symmetric=True, name="X")

        prob = cp.Problem(cp.Minimize(cp.trace(X)), [X >> 0, cp.trace(X) >= 1])
        prob.solve()

        report = debug(prob)

        assert report is not None

    def test_multiple_variables(self):
        """Test with multiple variables of different types."""
        x = cp.Variable(3, name="x")
        y = cp.Variable(name="y")
        Z = cp.Variable((2, 2), symmetric=True, name="Z")

        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + y + cp.trace(Z)), [x >= 0, y >= 0, Z >> 0, cp.sum(x) + y == 5]
        )
        prob.solve()

        report = debug(prob)

        assert report is not None


class TestBoundaryConditions:
    """Tests for boundary and limiting conditions."""

    def test_tight_constraints(self):
        """Test with very tight constraint margins."""
        x = cp.Variable(3, nonneg=True, name="x")

        # Barely feasible
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [
                cp.sum(x) <= 100,
                x[0] >= 33.33,
                x[1] >= 33.33,
                x[2] >= 33.33,
            ],
        )

        report = debug(prob)

        assert report is not None

    def test_redundant_constraints(self):
        """Test with redundant constraints."""
        x = cp.Variable(name="x")

        # Many redundant constraints
        prob = cp.Problem(
            cp.Minimize(x), [x >= 0, x >= -1, x >= -2, x >= -10, x <= 100, x <= 50, x <= 10]
        )
        prob.solve()

        report = debug(prob)

        assert report is not None


class TestInfeasibilityVariants:
    """Tests for different types of infeasibility."""

    def test_bound_conflict(self):
        """Test simple bound conflict."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

        report = debug(prob)

        assert report is not None
        assert len(report.findings) > 0

    def test_equality_conflict(self):
        """Test conflicting equality constraints."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x == 5, x == 10])

        report = debug(prob)

        assert report is not None
        assert len(report.findings) > 0

    def test_sum_conflict(self):
        """Test sum-based infeasibility."""
        x = cp.Variable(3, nonneg=True, name="x")

        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [cp.sum(x) <= 10, x >= 5],  # Need 15 but max is 10
        )

        report = debug(prob)

        assert report is not None
        assert len(report.findings) > 0


class TestUnboundednessVariants:
    """Tests for different types of unboundedness."""

    def test_simple_unbounded(self):
        """Test simple unbounded minimization."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [])

        report = debug(prob)

        assert report is not None

    def test_unbounded_with_partial_bounds(self):
        """Test unbounded with some variables bounded."""
        x = cp.Variable(2, name="x")

        # x[0] is bounded, x[1] is not
        prob = cp.Problem(cp.Minimize(x[0] + x[1]), [x[0] >= 0])

        report = debug(prob)

        assert report is not None

    def test_unbounded_maximization(self):
        """Test unbounded maximization."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Maximize(x), [x >= 0])

        report = debug(prob)

        assert report is not None
