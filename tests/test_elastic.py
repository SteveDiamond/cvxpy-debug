"""Tests for elastic relaxation."""

import cvxpy as cp

from cvxpy_debug.infeasibility.elastic import find_infeasibility_contributors


class TestElasticRelaxation:
    """Tests for the elastic relaxation method."""

    def test_simple_infeasible(self, simple_infeasible):
        """Test simple bound conflict."""
        simple_infeasible.solve()
        assert simple_infeasible.status == cp.INFEASIBLE

        contributors, slack_values = find_infeasibility_contributors(simple_infeasible)

        assert len(contributors) >= 1
        assert len(slack_values) >= 1

    def test_budget_infeasible(self, budget_infeasible):
        """Test budget allocation conflict."""
        budget_infeasible.solve()
        assert budget_infeasible.status == cp.INFEASIBLE

        contributors, slack_values = find_infeasibility_contributors(budget_infeasible)

        # Should identify constraints that conflict
        assert len(contributors) >= 1

        # Total slack should be about 20 (120 - 100)
        total_slack = sum(slack_values.values())
        assert total_slack > 0

    def test_equality_infeasible(self, equality_infeasible):
        """Test conflicting equality constraints."""
        equality_infeasible.solve()
        assert equality_infeasible.status == cp.INFEASIBLE

        contributors, slack_values = find_infeasibility_contributors(equality_infeasible)

        assert len(contributors) >= 1

    def test_feasible_returns_empty(self, feasible_problem):
        """Test that feasible problem returns no contributors."""
        feasible_problem.solve()
        assert feasible_problem.status == cp.OPTIMAL

        # For feasible problems, elastic relaxation should give zero slack
        contributors, slack_values = find_infeasibility_contributors(feasible_problem)

        # No constraints should have significant slack
        assert len(contributors) == 0


class TestSOCRelaxation:
    """Tests for SOC constraint relaxation."""

    def test_soc_infeasible(self, soc_infeasible):
        """Test SOC constraint conflict."""
        soc_infeasible.solve()
        assert soc_infeasible.status == cp.INFEASIBLE

        contributors, slack_values = find_infeasibility_contributors(soc_infeasible)

        assert len(contributors) >= 1


class TestPSDRelaxation:
    """Tests for PSD constraint relaxation."""

    def test_psd_infeasible(self, psd_infeasible):
        """Test PSD constraint conflict."""
        psd_infeasible.solve()
        assert psd_infeasible.status == cp.INFEASIBLE

        contributors, slack_values = find_infeasibility_contributors(psd_infeasible)

        # Should identify the conflicting constraint
        assert len(contributors) >= 1
