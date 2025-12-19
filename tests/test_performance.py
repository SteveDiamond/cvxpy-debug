"""Tests for performance diagnostics module."""

import cvxpy as cp

from cvxpy_debug.performance import (
    AntiPatternType,
    debug_performance,
)
from cvxpy_debug.performance.matrix_analysis import analyze_matrix_structure
from cvxpy_debug.report.report import DebugReport


class TestMetrics:
    """Tests for problem metrics computation."""

    def test_basic_metrics(self):
        """Test basic metric computation."""
        x = cp.Variable(5)
        constraints = [x >= 0, x <= 1]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report, include_matrix_analysis=False)

        assert analysis.metrics.num_variables == 1
        assert analysis.metrics.num_scalar_variables == 5
        assert analysis.metrics.num_constraints == 2

    def test_multiple_variables(self):
        """Test metrics with multiple variables."""
        x = cp.Variable(3)
        y = cp.Variable(2)
        z = cp.Variable()  # scalar
        prob = cp.Problem(
            cp.Minimize(cp.sum(x) + cp.sum(y) + z),
            [x >= 0, y >= 0, z >= 0],
        )

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report, include_matrix_analysis=False)

        assert analysis.metrics.num_variables == 3
        assert analysis.metrics.num_scalar_variables == 6  # 3 + 2 + 1


class TestLoopDetection:
    """Tests for loop-generated constraint detection."""

    def test_detects_loop_constraints(self):
        """Test detection of loop-generated constraints."""
        n = 50
        x = cp.Variable(n)

        # Anti-pattern: loop-generated constraints
        constraints = []
        for i in range(n):
            constraints.append(x[i] >= 0)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        # Should detect the anti-pattern
        pattern_types = [p.pattern_type for p in analysis.anti_patterns]
        assert any(
            p
            in (
                AntiPatternType.LOOP_GENERATED_CONSTRAINTS,
                AntiPatternType.SCALAR_ON_VECTOR,
                AntiPatternType.HIGH_CONSTRAINT_RATIO,
            )
            for p in pattern_types
        )

    def test_no_false_positive_vectorized(self):
        """Test that vectorized constraints don't trigger detection."""
        n = 100
        x = cp.Variable(n)

        # Correct pattern: vectorized constraint
        constraints = [x >= 0]

        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        # Should not detect loop anti-pattern
        high_severity = [p for p in analysis.anti_patterns if p.severity == "high"]
        assert len(high_severity) == 0


class TestMatrixAnalysis:
    """Tests for constraint matrix analysis."""

    def test_sparse_detection(self):
        """Test sparsity detection."""
        x = cp.Variable(10)
        # Sparse constraints
        constraints = [x >= 0, cp.sum(x) <= 10]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        structure = analyze_matrix_structure(prob)

        assert structure is not None
        assert structure.num_rows > 0
        assert structure.num_cols > 0
        assert structure.sparsity >= 0

    def test_matrix_dimensions(self):
        """Test matrix dimension calculation."""
        x = cp.Variable(5)
        constraints = [x >= 0]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        structure = analyze_matrix_structure(prob)

        assert structure is not None
        # Should have appropriate dimensions
        assert structure.num_cols >= 5  # at least 5 variables


class TestHighConstraintRatio:
    """Tests for high constraint ratio detection."""

    def test_detects_high_ratio(self):
        """Test detection of high constraint ratio."""
        x = cp.Variable(5)

        # Create many constraints for few variables
        constraints = []
        for i in range(5):
            for j in range(25):
                constraints.append(x[i] >= -j)

        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        # Should have high constraint ratio
        assert analysis.metrics.constraint_variable_ratio > 10


class TestSuggestions:
    """Tests for suggestion generation."""

    def test_generates_suggestions_for_antipatterns(self):
        """Test that suggestions are generated for anti-patterns."""
        n = 30
        x = cp.Variable(n)
        constraints = [x[i] >= 0 for i in range(n)]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        # Should have either anti-patterns with suggestions or general suggestions
        assert len(analysis.suggestions) > 0 or len(analysis.anti_patterns) > 0


class TestIntegration:
    """Integration tests."""

    def test_report_population(self):
        """Test that report is properly populated."""
        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])

        report = DebugReport(problem=prob)
        debug_performance(prob, report)

        assert len(report.findings) > 0
        assert report.performance_analysis is not None

    def test_full_debug_includes_performance(self):
        """Test that full debug() call includes performance analysis."""
        import cvxpy_debug

        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        prob.solve()

        report = cvxpy_debug.debug(prob, verbose=False)

        assert report.performance_analysis is not None
        assert report.performance_analysis.metrics is not None

    def test_debug_can_disable_performance(self):
        """Test that performance analysis can be disabled."""
        import cvxpy_debug

        x = cp.Variable(10)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0])
        prob.solve()

        report = cvxpy_debug.debug(prob, verbose=False, include_performance=False)

        assert report.performance_analysis is None


class TestSummary:
    """Tests for summary generation."""

    def test_summary_no_issues(self):
        """Test summary when no issues detected."""
        x = cp.Variable(5)
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [x >= 0, x <= 10])

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        assert "No major performance issues" in analysis.summary or analysis.summary == ""

    def test_summary_with_issues(self):
        """Test summary when issues are detected."""
        n = 50
        x = cp.Variable(n)
        constraints = [x[i] >= 0 for i in range(n)]
        prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

        report = DebugReport(problem=prob)
        analysis = debug_performance(prob, report)

        # Should have some summary content
        assert len(analysis.summary) > 0
