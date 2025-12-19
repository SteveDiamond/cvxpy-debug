"""Tests for report module."""

import cvxpy as cp

from cvxpy_debug.report.report import (
    DebugReport,
    _format_unbounded_table,
    _get_report_title,
    format_report,
)


class TestDebugReportDataclass:
    """Tests for DebugReport class."""

    def test_default_values(self):
        """Test default initialization."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        report = DebugReport(problem=prob)

        assert report.problem is prob
        assert report.status == ""
        assert report.iis == []
        assert report.slack_values == {}
        assert report.constraint_info == []
        assert report.findings == []
        assert report.suggestions == []
        assert report.unbounded_variables == []
        assert report.unbounded_ray is None
        assert report.numerical_analysis is None
        assert report.performance_analysis is None

    def test_add_finding(self):
        """Test add_finding method."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob)

        report.add_finding("Test finding 1")
        report.add_finding("Test finding 2")

        assert len(report.findings) == 2
        assert "Test finding 1" in report.findings
        assert "Test finding 2" in report.findings

    def test_add_suggestion(self):
        """Test add_suggestion method."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob)

        report.add_suggestion("Try this")
        report.add_suggestion("Or try that")

        assert len(report.suggestions) == 2
        assert "Try this" in report.suggestions

    def test_str_method(self):
        """Test __str__ method returns formatted report."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(
            problem=prob,
            status="infeasible",
            findings=["Problem is infeasible"],
        )

        result = str(report)

        assert isinstance(result, str)
        assert len(result) > 0

    def test_with_all_fields(self):
        """Test report with all fields populated."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

        report = DebugReport(
            problem=prob,
            status="infeasible",
            iis=[x >= 10, x <= 5],
            slack_values={0: 5.0},
            constraint_info=[{"label": "x >= 10", "slack": 5.0}],
            findings=["Conflicting constraints found"],
            suggestions=["Relax one of the constraints"],
        )

        assert report.status == "infeasible"
        assert len(report.iis) == 2
        assert len(report.findings) == 1


class TestFormatReport:
    """Tests for format_report()."""

    def test_infeasibility_title(self):
        """Test infeasibility report has correct title."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="infeasible")

        result = format_report(report)

        assert "INFEASIBILITY" in result

    def test_unboundedness_title(self):
        """Test unboundedness report has correct title."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="unbounded")

        result = format_report(report)

        assert "UNBOUNDEDNESS" in result

    def test_numerical_accuracy_title(self):
        """Test numerical accuracy report has correct title."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="optimal_inaccurate")

        result = format_report(report)

        assert "NUMERICAL" in result or "ACCURACY" in result

    def test_constraint_table_included(self):
        """Test that constraint table is included when there's constraint info."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(
            problem=prob,
            status="infeasible",
            constraint_info=[{"label": "x >= 10", "slack": 5.0}],
        )

        result = format_report(report)

        assert "CONFLICTING" in result
        assert "x >= 10" in result

    def test_unbounded_table_included(self):
        """Test that unbounded table is included when there are unbounded vars."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(
            problem=prob,
            status="unbounded",
            unbounded_variables=[{"name": "x", "direction": "below", "direction_symbol": "-inf"}],
        )

        result = format_report(report)

        assert "UNBOUNDED" in result

    def test_suggestions_section(self):
        """Test that suggestions section is included."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(
            problem=prob,
            status="infeasible",
            suggestions=["Add nonneg=True", "Relax constraint"],
        )

        result = format_report(report)

        assert "SUGGESTED" in result or "FIX" in result
        assert "nonneg" in result

    def test_findings_included(self):
        """Test that findings are included in report."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(
            problem=prob,
            status="infeasible",
            findings=["This is a test finding"],
        )

        result = format_report(report)

        assert "test finding" in result

    def test_empty_report(self):
        """Test formatting of empty report."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob)

        result = format_report(report)

        # Should not crash, should return some string
        assert isinstance(result, str)


class TestGetReportTitle:
    """Tests for _get_report_title()."""

    def test_infeasible_status(self):
        """Test title for infeasible status."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="infeasible")

        result = _get_report_title(report)

        assert "INFEASIBILITY" in result

    def test_unbounded_status(self):
        """Test title for unbounded status."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="unbounded")

        result = _get_report_title(report)

        assert "UNBOUNDEDNESS" in result

    def test_optimal_inaccurate_status(self):
        """Test title for optimal_inaccurate status."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="optimal_inaccurate")

        result = _get_report_title(report)

        assert "NUMERICAL" in result or "ACCURACY" in result

    def test_infeasible_inaccurate_status(self):
        """Test title for infeasible_inaccurate status."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="infeasible_inaccurate")

        result = _get_report_title(report)

        assert "NUMERICAL" in result or "ACCURACY" in result

    def test_unknown_status(self):
        """Test title for unknown status."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])
        report = DebugReport(problem=prob, status="something_else")

        result = _get_report_title(report)

        assert "DEBUG" in result


class TestFormatUnboundedTable:
    """Tests for _format_unbounded_table()."""

    def test_empty_list(self):
        """Test formatting empty list."""
        result = _format_unbounded_table([])

        assert "none" in result.lower()

    def test_single_variable(self):
        """Test formatting single variable."""
        unbounded = [{"name": "x", "direction": "below", "direction_symbol": "-inf"}]

        result = _format_unbounded_table(unbounded)

        assert "x" in result
        assert "-inf" in result

    def test_multiple_variables(self):
        """Test formatting multiple variables."""
        unbounded = [
            {"name": "x", "direction": "below", "direction_symbol": "-inf"},
            {"name": "y", "direction": "above", "direction_symbol": "+inf"},
            {"name": "z", "direction": "both", "direction_symbol": "+/-inf"},
        ]

        result = _format_unbounded_table(unbounded)

        assert "x" in result
        assert "y" in result
        assert "z" in result
        assert "-inf" in result
        assert "+inf" in result

    def test_column_headers(self):
        """Test that column headers are present."""
        unbounded = [{"name": "x", "direction": "below", "direction_symbol": "-inf"}]

        result = _format_unbounded_table(unbounded)

        assert "Variable" in result or "variable" in result.lower()
        assert "Direction" in result or "direction" in result.lower()


class TestReportIntegration:
    """Integration tests for report formatting."""

    def test_full_infeasibility_report(self):
        """Test complete infeasibility report."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

        report = DebugReport(
            problem=prob,
            status="infeasible",
            iis=[prob.constraints[0], prob.constraints[1]],
            constraint_info=[
                {"label": "x >= 10", "slack": 5.0},
                {"label": "x <= 5", "slack": 0.0},
            ],
            findings=[
                "Problem is infeasible due to conflicting constraints",
                "Found 2 conflicting constraints",
            ],
            suggestions=[
                "Increase upper bound from 5 to at least 10",
                "Or decrease lower bound from 10",
            ],
        )

        result = str(report)

        assert "INFEASIBILITY" in result
        assert "x >= 10" in result
        assert "conflicting" in result.lower()

    def test_full_unboundedness_report(self):
        """Test complete unboundedness report."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        report = DebugReport(
            problem=prob,
            status="unbounded",
            unbounded_variables=[{"name": "x", "direction": "below", "direction_symbol": "-inf"}],
            findings=["Problem is unbounded below"],
            suggestions=["Add lower bound to x"],
        )

        result = str(report)

        assert "UNBOUNDEDNESS" in result
        assert "lower bound" in result.lower() or "nonneg" in result.lower()
