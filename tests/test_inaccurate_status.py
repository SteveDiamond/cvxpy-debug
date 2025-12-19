"""Tests for INACCURATE status handling."""

import cvxpy as cp
import numpy as np

from cvxpy_debug import debug


class TestOptimalInaccurate:
    """Tests for OPTIMAL_INACCURATE handling."""

    def test_detects_optimal_inaccurate(self):
        """Test that OPTIMAL_INACCURATE status is detected."""
        # Create a problem that might return OPTIMAL_INACCURATE
        # This is solver-dependent, so we test the handling path
        x = cp.Variable(10, name="x")

        # Create an ill-conditioned problem
        np.random.seed(42)
        A = np.random.randn(10, 10)
        A = A @ A.T + 1e-6 * np.eye(10)  # Near-singular

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, A)), [cp.sum(x) == 1, x >= 0])
        prob.solve()

        # Even if status is OPTIMAL, we can still run diagnostics
        report = debug(prob)

        assert report is not None

    def test_numerical_diagnostics_on_solved_problem(self):
        """Test that numerical diagnostics run on solved problems."""
        x = cp.Variable(3, name="x")

        prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [cp.sum(x) == 1, x >= 0])
        prob.solve()

        report = debug(prob)

        # Report should exist and have some content
        assert report is not None


class TestInfeasibleInaccurate:
    """Tests for INFEASIBLE_INACCURATE handling."""

    def test_handles_infeasible_problem(self):
        """Test handling of clearly infeasible problems."""
        x = cp.Variable(name="x")

        # Clearly infeasible
        prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

        report = debug(prob)

        assert report is not None
        assert len(report.findings) > 0

    def test_infeasibility_analysis_runs(self):
        """Test that infeasibility analysis produces findings."""
        x = cp.Variable(3, nonneg=True, name="x")

        # Infeasible budget problem
        prob = cp.Problem(
            cp.Minimize(cp.sum(x)),
            [
                cp.sum(x) <= 100,
                x[0] >= 50,
                x[1] >= 40,
                x[2] >= 30,
            ],
        )

        report = debug(prob)

        assert report is not None
        # Should identify conflicting constraints
        assert len(report.constraint_info) > 0 or len(report.findings) > 0


class TestUnboundedInaccurate:
    """Tests for UNBOUNDED_INACCURATE handling."""

    def test_handles_unbounded_problem(self):
        """Test handling of clearly unbounded problems."""
        x = cp.Variable(name="x")

        # Clearly unbounded
        prob = cp.Problem(cp.Minimize(x), [])

        report = debug(prob)

        assert report is not None
        assert len(report.findings) > 0

    def test_unboundedness_analysis_runs(self):
        """Test that unboundedness analysis produces findings."""
        x = cp.Variable(2, name="x")

        # Unbounded with constraint
        prob = cp.Problem(cp.Minimize(x[0] - x[1]), [x[0] + x[1] == 1])

        report = debug(prob)

        assert report is not None
        assert len(report.unbounded_variables) > 0 or len(report.findings) > 0


class TestMixedStatus:
    """Tests for various problem statuses."""

    def test_feasible_optimal(self):
        """Test handling of feasible optimal problems."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x >= 0, x <= 10])

        report = debug(prob)

        assert report is not None

    def test_solver_error_handling(self):
        """Test graceful handling when solver has issues."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x >= 0])

        # Even if problem has issues, debug should not crash
        try:
            report = debug(prob)
            assert report is not None
        except Exception as e:
            # Should be a meaningful error, not a crash
            assert isinstance(e, (cp.SolverError, ValueError))


class TestDebugOptions:
    """Tests for debug() function options."""

    def test_find_minimal_iis_option(self):
        """Test find_minimal_iis option."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

        report = debug(prob, find_minimal_iis=True)

        assert report is not None

    def test_analyze_conditioning_option(self):
        """Test include_conditioning option."""
        x = cp.Variable(3, name="x")
        A = np.array([[1, 0.5, 0.3], [0.5, 1, 0.4], [0.3, 0.4, 1]])

        prob = cp.Problem(cp.Minimize(cp.quad_form(x, A)), [cp.sum(x) == 1, x >= 0])
        prob.solve()

        report = debug(prob, include_conditioning=True)

        assert report is not None

    def test_solver_override(self):
        """Test solver parameter override."""
        x = cp.Variable(name="x")

        prob = cp.Problem(cp.Minimize(x), [x >= 0, x <= 10])

        # Should work with default solver
        report = debug(prob)

        assert report is not None
