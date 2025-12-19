"""Tests for IIS (Irreducible Infeasible Subsystem) finding."""

import cvxpy as cp

from cvxpy_debug.infeasibility.iis import find_all_iis, find_minimal_iis


class TestMinimalIIS:
    """Tests for minimal IIS finding."""

    def test_simple_iis(self, simple_infeasible):
        """Test IIS for simple bound conflict."""
        simple_infeasible.solve()

        # Both constraints are needed for infeasibility
        iis = find_minimal_iis(
            simple_infeasible,
            simple_infeasible.constraints,
        )

        # Should find both constraints (minimal IIS is size 2)
        assert len(iis) == 2

    def test_redundant_constraint_removed(self):
        """Test that non-essential constraints are removed from IIS."""
        x = cp.Variable(name="x")
        constraints = [
            x >= 5,  # Essential for infeasibility
            x <= 3,  # Essential for infeasibility
            x >= 0,  # NOT essential (redundant with x >= 5)
        ]
        prob = cp.Problem(cp.Minimize(x), constraints)
        prob.solve()

        assert prob.status == cp.INFEASIBLE

        iis = find_minimal_iis(prob, constraints)

        # Should only include the two conflicting constraints
        assert len(iis) == 2
        # The x >= 0 constraint should not be in the IIS
        assert constraints[2] not in iis

    def test_budget_iis(self, budget_infeasible):
        """Test IIS for budget allocation problem."""
        budget_infeasible.solve()

        iis = find_minimal_iis(
            budget_infeasible,
            budget_infeasible.constraints,
        )

        # All 4 constraints are needed (budget + 3 minimums)
        assert len(iis) == 4


class TestMultipleIIS:
    """Tests for finding multiple IIS."""

    def test_single_iis(self, simple_infeasible):
        """Problem with only one IIS."""
        simple_infeasible.solve()

        all_iis = find_all_iis(simple_infeasible)

        assert len(all_iis) == 1
        assert len(all_iis[0]) == 2

    def test_multiple_independent_iis(self):
        """Problem with multiple independent IIS."""
        x = cp.Variable(name="x")
        y = cp.Variable(name="y")

        constraints = [
            # First IIS: x >= 5 and x <= 3
            x >= 5,
            x <= 3,
            # Second IIS: y >= 10 and y <= 5
            y >= 10,
            y <= 5,
        ]
        prob = cp.Problem(cp.Minimize(x + y), constraints)
        prob.solve()

        assert prob.status == cp.INFEASIBLE

        all_iis = find_all_iis(prob, max_iis=5)

        # Should find at least 2 IIS
        assert len(all_iis) >= 2
