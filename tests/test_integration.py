"""Integration tests for the main debug function."""

import cvxpy as cp

import cvxpy_debug


class TestDebugFunction:
    """Tests for the main debug() function."""

    def test_infeasible_problem(self, simple_infeasible):
        """Test debugging an infeasible problem."""
        simple_infeasible.solve()

        report = cvxpy_debug.debug(simple_infeasible, verbose=False)

        assert report.status == "infeasible"
        assert len(report.iis) > 0
        assert len(report.findings) > 0
        assert len(report.suggestions) > 0

    def test_feasible_problem(self, feasible_problem):
        """Test debugging a feasible problem."""
        feasible_problem.solve()

        report = cvxpy_debug.debug(feasible_problem, verbose=False)

        assert "successfully" in report.findings[0].lower()

    def test_unsolved_problem(self):
        """Test debugging a problem that hasn't been solved."""
        x = cp.Variable()
        prob = cp.Problem(cp.Minimize(x), [x >= 5, x <= 3])

        # Don't solve it first
        report = cvxpy_debug.debug(prob, verbose=False)

        # Should still diagnose infeasibility
        assert report.status == "infeasible"

    def test_report_string_format(self, simple_infeasible):
        """Test that report formats correctly as string."""
        simple_infeasible.solve()

        report = cvxpy_debug.debug(simple_infeasible, verbose=False)
        report_str = str(report)

        assert "INFEASIBILITY REPORT" in report_str
        assert "CONFLICTING CONSTRAINTS" in report_str
        assert "SUGGESTED FIXES" in report_str

    def test_budget_problem_output(self, budget_infeasible):
        """Test full output for budget problem."""
        budget_infeasible.solve()

        report = cvxpy_debug.debug(budget_infeasible, verbose=False)

        assert report.status == "infeasible"
        assert len(report.iis) >= 1

        # Check slack values make sense
        total_slack = sum(report.slack_values.values())
        assert total_slack > 0


class TestDebugOptions:
    """Tests for debug function options."""

    def test_verbose_false(self, simple_infeasible, capsys):
        """Test that verbose=False suppresses output."""
        simple_infeasible.solve()

        cvxpy_debug.debug(simple_infeasible, verbose=False)

        captured = capsys.readouterr()
        assert captured.out == ""

    def test_verbose_true(self, simple_infeasible, capsys):
        """Test that verbose=True prints output."""
        simple_infeasible.solve()

        cvxpy_debug.debug(simple_infeasible, verbose=True)

        captured = capsys.readouterr()
        assert "INFEASIBILITY REPORT" in captured.out

    def test_find_minimal_iis_option(self):
        """Test find_minimal_iis option."""
        x = cp.Variable()
        constraints = [x >= 5, x <= 3, x >= 0]  # Third is redundant
        prob = cp.Problem(cp.Minimize(x), constraints)
        prob.solve()

        # With minimal IIS finding
        report = cvxpy_debug.debug(prob, verbose=False, find_minimal_iis=True)
        assert len(report.iis) == 2  # Only essential constraints

        # Without minimal IIS finding (just elastic relaxation)
        report_no_min = cvxpy_debug.debug(prob, verbose=False, find_minimal_iis=False)
        # May have more constraints in non-minimal version
        assert len(report_no_min.iis) >= 2
