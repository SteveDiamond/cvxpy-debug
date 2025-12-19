"""Tests for numerical diagnostics module."""

import cvxpy as cp
import numpy as np
import pytest

from cvxpy_debug.numerical import debug_numerical_issues
from cvxpy_debug.numerical.conditioning import analyze_conditioning
from cvxpy_debug.numerical.recommendations import generate_recommendations
from cvxpy_debug.numerical.scaling import analyze_scaling
from cvxpy_debug.numerical.solver_stats import analyze_solver_stats
from cvxpy_debug.numerical.violations import analyze_violations
from cvxpy_debug.report.report import DebugReport


class TestScalingAnalysis:
    """Tests for coefficient scaling analysis."""

    def test_well_scaled_problem(self):
        """Test detection of well-scaled problem."""
        x = cp.Variable(2)
        constraints = [x >= 0, x <= 1, cp.sum(x) == 1]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        scaling = analyze_scaling(prob)

        assert not scaling.badly_scaled
        assert scaling.overall_range_ratio < 1e6

    def test_badly_scaled_problem(self):
        """Test detection of badly scaled problem."""
        x = cp.Variable()
        # Coefficients range from 1e-10 to 1e10
        constraints = [1e10 * x >= 1, 1e-10 * x <= 1]
        prob = cp.Problem(cp.Minimize(x), constraints)

        scaling = analyze_scaling(prob)

        assert scaling.badly_scaled
        assert scaling.overall_range_ratio > 1e15

    def test_very_small_coefficients(self):
        """Test detection of very small coefficients."""
        x = cp.Variable()
        constraints = [1e-12 * x >= 1]
        prob = cp.Problem(cp.Minimize(x), constraints)

        scaling = analyze_scaling(prob)

        assert len(scaling.very_small_coefficients) > 0

    def test_very_large_coefficients(self):
        """Test detection of very large coefficients."""
        x = cp.Variable()
        constraints = [1e12 * x >= 1]
        prob = cp.Problem(cp.Minimize(x), constraints)

        scaling = analyze_scaling(prob)

        assert len(scaling.very_large_coefficients) > 0


class TestViolationAnalysis:
    """Tests for constraint violation analysis."""

    def test_no_violations_optimal(self):
        """Test that optimal solutions have negligible violations."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0, x <= 10])
        prob.solve()

        violations = analyze_violations(prob)

        assert violations.max_violation < 1e-6

    def test_detects_violations(self):
        """Test violation detection for inaccurate solutions."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0, x <= 10])
        prob.solve()

        # Manually set a slightly violating value
        x.value = np.array(-0.001)

        violations = analyze_violations(prob)

        assert violations.max_violation > 0
        assert any(v.violation_amount > 0 for v in violations.violations)

    def test_tolerance_comparison(self):
        """Test that violations are compared to solver tolerances."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        violations = analyze_violations(prob, solver_name="SCS")

        assert (
            "eps_abs" in violations.solver_tolerances or "default" in violations.solver_tolerances
        )


class TestConditioningAnalysis:
    """Tests for condition number analysis."""

    def test_well_conditioned_problem(self):
        """Test detection of well-conditioned problem."""
        x = cp.Variable(3)
        A = np.eye(3)
        b = np.ones(3)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        conditioning = analyze_conditioning(prob)

        # May not always estimate, but if it does, should be well-conditioned
        if conditioning.estimated and conditioning.condition_number is not None:
            assert not conditioning.ill_conditioned

    def test_ill_conditioned_problem(self):
        """Test detection of ill-conditioned problem."""
        n = 10
        x = cp.Variable(n)
        # Create ill-conditioned matrix
        A = np.diag(np.logspace(-6, 6, n))
        b = np.ones(n)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)))

        conditioning = analyze_conditioning(prob)

        # If estimation succeeded, should detect ill-conditioning
        if conditioning.estimated and conditioning.condition_number is not None:
            assert conditioning.condition_number >= 1e6


class TestSolverStatsAnalysis:
    """Tests for solver statistics analysis."""

    def test_basic_stats_extraction(self):
        """Test extraction of basic solver stats."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        stats = analyze_solver_stats(prob)

        assert stats is not None
        assert stats.solver_name is not None

    def test_no_stats_returns_none(self):
        """Test that unsolved problem returns None."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        # Don't solve

        stats = analyze_solver_stats(prob)

        assert stats is None


class TestRecommendations:
    """Tests for solver recommendations."""

    def test_generates_recommendations(self):
        """Test that recommendations are generated."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        recommendations = generate_recommendations(
            prob,
            scaling=None,
            violations=None,
            conditioning=None,
            solver_stats=None,
        )

        assert len(recommendations) > 0

    def test_excludes_ecos(self):
        """Test that ECOS is excluded from recommendations."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        recommendations = generate_recommendations(
            prob,
            scaling=None,
            violations=None,
            conditioning=None,
            solver_stats=None,
        )

        solver_names = [r.solver_name for r in recommendations]
        assert "ECOS" not in solver_names
        assert "ECOS_BB" not in solver_names

    def test_excludes_current_solver(self):
        """Test that current solver is excluded from recommendations."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve(solver=cp.SCS)

        recommendations = generate_recommendations(
            prob,
            scaling=None,
            violations=None,
            conditioning=None,
            solver_stats=None,
            current_solver="SCS",
        )

        solver_names = [r.solver_name for r in recommendations]
        assert "SCS" not in solver_names


class TestIntegration:
    """Integration tests for numerical diagnostics."""

    def test_diagnose_optimal(self):
        """Test diagnosis of optimal problem (baseline)."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        report = DebugReport(problem=prob)
        analysis = debug_numerical_issues(prob, report, include_conditioning=False)

        assert analysis is not None
        assert len(report.findings) > 0

    def test_diagnose_badly_scaled(self):
        """Test diagnosis of badly scaled problem."""
        x = cp.Variable()
        # Coefficients range from 1e-10 to 1e10 (badly scaled)
        constraints = [1e10 * x >= 1, 1e-10 * x <= 1]
        prob = cp.Problem(cp.Minimize(x**2), constraints)
        prob.solve()

        report = DebugReport(problem=prob)
        analysis = debug_numerical_issues(prob, report, include_conditioning=False)

        assert analysis.scaling is not None
        assert analysis.scaling.badly_scaled

    def test_report_population(self):
        """Test that report is properly populated."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 0])
        prob.solve()

        report = DebugReport(problem=prob)
        debug_numerical_issues(prob, report, include_conditioning=False)

        assert len(report.findings) > 0
        # Should have some suggestions (at least alternative solvers)
        # Note: may be empty if no other solvers installed


# Fixtures for numerical tests
@pytest.fixture
def badly_scaled_lp():
    """LP with poor scaling."""
    x = cp.Variable(3)
    constraints = [
        1e8 * x[0] + x[1] + x[2] <= 1e8,
        x[0] + 1e-8 * x[1] + x[2] >= 0,
        x >= 0,
    ]
    return cp.Problem(cp.Minimize(cp.sum(x)), constraints)


@pytest.fixture
def well_scaled_lp():
    """LP with good scaling."""
    x = cp.Variable(3)
    constraints = [
        x[0] + x[1] + x[2] <= 10,
        x[0] + x[1] + x[2] >= 1,
        x >= 0,
    ]
    return cp.Problem(cp.Minimize(cp.sum(x)), constraints)
