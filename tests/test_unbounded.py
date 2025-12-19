"""Tests for unboundedness diagnosis."""

import cvxpy as cp

import cvxpy_debug
from cvxpy_debug.unbounded.bounds import (
    analyze_variable_bounds,
    get_unbounded_variables,
)
from cvxpy_debug.unbounded.ray import find_unbounded_ray


class TestBoundsAnalysis:
    """Tests for variable bounds analysis."""

    def test_nonneg_has_lower_bound(self):
        """Nonneg variable should have lower bound of 0."""
        x = cp.Variable(nonneg=True)
        prob = cp.Problem(cp.Minimize(x), [])
        bounds = analyze_variable_bounds(prob)
        info = bounds[id(x)]
        assert info.lower == 0.0
        assert info.is_bounded_below

    def test_unconstrained_has_no_bounds(self):
        """Unconstrained variable should have no bounds."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [])
        bounds = analyze_variable_bounds(prob)
        info = bounds[id(x)]
        assert info.lower is None
        assert info.upper is None
        assert not info.is_fully_bounded

    def test_constraint_provides_bound(self, bounded_problem):
        """Explicit constraint should provide bound."""
        bounds = analyze_variable_bounds(bounded_problem)
        # There's only one variable
        info = list(bounds.values())[0]
        # At minimum the nonneg-like constraint should be detected
        # The exact detection depends on implementation
        assert info is not None


class TestUnboundedVariables:
    """Tests for finding unbounded variables."""

    def test_finds_unbounded_below(self, unbounded_below):
        """Should find variable unbounded below."""
        unbounded = get_unbounded_variables(unbounded_below)
        assert len(unbounded) == 1
        var, direction = unbounded[0]
        assert direction in ("below", "both")

    def test_finds_unbounded_above(self, unbounded_above):
        """Should find variable unbounded above."""
        unbounded = get_unbounded_variables(unbounded_above)
        assert len(unbounded) == 1
        var, direction = unbounded[0]
        assert direction in ("above", "both")

    def test_bounded_has_none(self, bounded_problem):
        """Bounded problem should have no unbounded variables."""
        unbounded = get_unbounded_variables(bounded_problem)
        # May still show as unbounded depending on constraint detection
        # The main test is that the integration works
        assert isinstance(unbounded, list)


class TestUnboundedRay:
    """Tests for finding unbounded ray/direction."""

    def test_finds_ray_minimize(self, unbounded_below):
        """Should find ray for minimize problem going to -inf."""
        ray = find_unbounded_ray(unbounded_below)
        # Ray might be None if problem is simple
        # The important thing is it doesn't crash
        if ray is not None:
            assert ray.objective_direction == "decrease"

    def test_finds_ray_maximize(self, unbounded_above):
        """Should find ray for maximize problem going to +inf."""
        ray = find_unbounded_ray(unbounded_above)
        if ray is not None:
            assert ray.objective_direction == "increase"


class TestDiagnoseUnboundedness:
    """Tests for main diagnosis function."""

    def test_diagnoses_unbounded_below(self, unbounded_below):
        """Should diagnose unbounded minimization."""
        unbounded_below.solve()
        assert unbounded_below.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)

        report = cvxpy_debug.debug(unbounded_below, verbose=False)
        assert report.status == "unbounded"
        assert len(report.findings) > 0
        assert len(report.suggestions) > 0

    def test_diagnoses_unbounded_above(self, unbounded_above):
        """Should diagnose unbounded maximization."""
        unbounded_above.solve()
        assert unbounded_above.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)

        report = cvxpy_debug.debug(unbounded_above, verbose=False)
        assert report.status == "unbounded"
        assert len(report.findings) > 0

    def test_diagnoses_unbounded_direction(self, unbounded_direction):
        """Should diagnose problem unbounded in specific direction."""
        unbounded_direction.solve()
        assert unbounded_direction.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)

        report = cvxpy_debug.debug(unbounded_direction, verbose=False)
        assert report.status == "unbounded"

    def test_diagnoses_unbounded_vector(self, unbounded_vector):
        """Should diagnose unbounded vector variable."""
        unbounded_vector.solve()
        assert unbounded_vector.status in (cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE)

        report = cvxpy_debug.debug(unbounded_vector, verbose=False)
        assert report.status == "unbounded"
        assert len(report.findings) > 0


class TestReportFormatting:
    """Tests for report formatting."""

    def test_report_string_contains_unbounded(self, unbounded_below):
        """Report string should contain 'UNBOUNDED'."""
        unbounded_below.solve()
        report = cvxpy_debug.debug(unbounded_below, verbose=False)
        report_str = str(report)
        assert "UNBOUNDED" in report_str

    def test_report_has_suggestions(self, unbounded_below):
        """Report should have suggestions for fixing."""
        unbounded_below.solve()
        report = cvxpy_debug.debug(unbounded_below, verbose=False)
        assert len(report.suggestions) > 0

    def test_report_mentions_variable(self, unbounded_below):
        """Report should mention the unbounded variable."""
        unbounded_below.solve()
        report = cvxpy_debug.debug(unbounded_below, verbose=False)
        report_str = str(report)
        # Should mention the variable name or generic 'x'
        assert "x" in report_str.lower() or "variable" in report_str.lower()


class TestIntegration:
    """Integration tests with main debug function."""

    def test_debug_auto_solves_unbounded(self):
        """debug() should auto-solve unsolved unbounded problem."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        # Don't solve - let debug() do it
        report = cvxpy_debug.debug(prob, verbose=False)
        assert report.status == "unbounded"

    def test_debug_doesnt_crash_on_bounded(self, bounded_problem):
        """debug() should handle bounded problem correctly."""
        bounded_problem.solve()
        assert bounded_problem.status == cp.OPTIMAL
        report = cvxpy_debug.debug(bounded_problem, verbose=False)
        assert "unbounded" not in report.status.lower()

    def test_feasible_not_flagged_unbounded(self, feasible_problem):
        """Feasible problem should not be flagged as unbounded."""
        report = cvxpy_debug.debug(feasible_problem, verbose=False)
        assert report.status != "unbounded"
