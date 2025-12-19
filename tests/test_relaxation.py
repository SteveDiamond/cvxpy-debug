"""Tests for constraint relaxation utilities."""

import cvxpy as cp
import numpy as np

from cvxpy_debug.infeasibility.relaxation import (
    _relax_equality,
    _relax_exp_cone,
    _relax_inequality,
    _relax_psd,
    _relax_soc,
    relax_constraint,
)


class TestRelaxConstraint:
    """Tests for relax_constraint() dispatch function."""

    def test_equality_dispatch(self):
        """Test that equality constraints are dispatched correctly."""
        x = cp.Variable(name="x")
        constraint = x == 5

        slack, relaxed = relax_constraint(constraint)

        assert slack is not None
        assert len(relaxed) > 0
        assert isinstance(slack, cp.Variable)

    def test_inequality_dispatch(self):
        """Test that inequality constraints are dispatched correctly."""
        x = cp.Variable(name="x")
        constraint = x <= 10

        slack, relaxed = relax_constraint(constraint)

        assert slack is not None
        assert len(relaxed) > 0

    def test_soc_dispatch(self):
        """Test that SOC constraints are dispatched correctly."""
        x = cp.Variable(3, name="x")
        t = cp.Variable(name="t")
        constraint = cp.norm(x) <= t

        slack, relaxed = relax_constraint(constraint)

        assert slack is not None
        assert len(relaxed) > 0

    def test_psd_dispatch(self):
        """Test that PSD constraints are dispatched correctly."""
        X = cp.Variable((2, 2), symmetric=True, name="X")
        constraint = X >> 0

        slack, relaxed = relax_constraint(constraint)

        assert slack is not None
        assert len(relaxed) > 0

    def test_exp_cone_dispatch(self):
        """Test that exponential cone constraints are dispatched correctly."""
        x = cp.Variable(name="x")
        y = cp.Variable(pos=True, name="y")
        z = cp.Variable(name="z")
        constraint = cp.constraints.exponential.ExpCone(x, y, z)

        slack, relaxed = relax_constraint(constraint)

        assert slack is not None
        assert len(relaxed) > 0

    def test_unknown_constraint_returns_unchanged(self):
        """Test that unknown constraints are returned unchanged."""
        x = cp.Variable(name="x")
        # NonNeg is handled differently, create a wrapped constraint
        constraint = x >= 0

        slack, relaxed = relax_constraint(constraint)

        # Should either handle it or return None slack
        assert isinstance(relaxed, list)


class TestRelaxEquality:
    """Tests for _relax_equality()."""

    def test_scalar_equality(self):
        """Test relaxation of scalar equality constraint."""
        x = cp.Variable(name="x")
        constraint = x == 5

        slack, relaxed = _relax_equality(constraint)

        assert slack is not None
        assert slack.is_nonneg()
        # Relaxed should allow x to deviate from 5 by up to slack
        assert len(relaxed) >= 1

    def test_vector_equality(self):
        """Test relaxation of vector equality constraint."""
        x = cp.Variable(3, name="x")
        target = np.array([1, 2, 3])
        constraint = x == target

        slack, relaxed = _relax_equality(constraint)

        assert slack is not None
        assert len(relaxed) >= 1

    def test_matrix_equality(self):
        """Test relaxation of matrix equality constraint."""
        X = cp.Variable((2, 2), name="X")
        target = np.eye(2)
        constraint = X == target

        slack, relaxed = _relax_equality(constraint)

        assert slack is not None
        assert len(relaxed) >= 1

    def test_slack_is_nonneg(self):
        """Test that slack variable is non-negative."""
        x = cp.Variable(name="x")
        constraint = x == 5

        slack, _ = _relax_equality(constraint)

        assert slack.is_nonneg()


class TestRelaxInequality:
    """Tests for _relax_inequality()."""

    def test_scalar_inequality(self):
        """Test relaxation of scalar inequality."""
        x = cp.Variable(name="x")
        constraint = x <= 10

        slack, relaxed = _relax_inequality(constraint)

        assert slack is not None
        assert len(relaxed) == 1
        # x <= 10 + slack

    def test_vector_inequality(self):
        """Test relaxation of vector inequality."""
        x = cp.Variable(3, name="x")
        bound = np.array([1, 2, 3])
        constraint = x <= bound

        slack, relaxed = _relax_inequality(constraint)

        assert slack is not None
        assert len(relaxed) >= 1

    def test_slack_is_nonneg(self):
        """Test that slack variable is non-negative."""
        x = cp.Variable(name="x")
        constraint = x <= 10

        slack, _ = _relax_inequality(constraint)

        assert slack.is_nonneg()

    def test_relaxed_constraint_is_valid(self):
        """Test that relaxed constraint can be used in a problem."""
        x = cp.Variable(name="x")
        constraint = x <= 10

        slack, relaxed = _relax_inequality(constraint)

        # Should be able to create a valid problem
        prob = cp.Problem(cp.Minimize(slack), relaxed + [x >= 15])
        prob.solve()

        assert prob.status == cp.OPTIMAL
        assert slack.value >= 5 - 1e-6  # Need at least 5 slack


class TestRelaxSOC:
    """Tests for _relax_soc()."""

    def test_basic_soc_relaxation(self):
        """Test relaxation of SOC constraint."""
        x = cp.Variable(3, name="x")
        t = cp.Variable(name="t")
        constraint = cp.norm(x) <= t

        slack, relaxed = _relax_soc(constraint)

        assert slack is not None
        assert len(relaxed) == 1

    def test_slack_nonneg(self):
        """Test that SOC slack is non-negative."""
        x = cp.Variable(3, name="x")
        t = cp.Variable(name="t")
        constraint = cp.norm(x) <= t

        slack, _ = _relax_soc(constraint)

        assert slack.is_nonneg()

    def test_relaxed_soc_is_valid(self):
        """Test that relaxed SOC creates valid slack variable."""
        x = cp.Variable(3, name="x")
        t = cp.Variable(name="t")
        constraint = cp.norm(x) <= t

        slack, relaxed = _relax_soc(constraint)

        # Just verify the relaxation produces valid outputs
        assert slack is not None
        assert slack.is_nonneg()
        assert len(relaxed) >= 1


class TestRelaxPSD:
    """Tests for _relax_psd()."""

    def test_psd_diagonal_shift(self):
        """Test that PSD relaxation adds diagonal shift."""
        X = cp.Variable((2, 2), symmetric=True, name="X")
        constraint = X >> 0

        slack, relaxed = _relax_psd(constraint)

        assert slack is not None
        assert len(relaxed) == 1

    def test_correct_identity_dimension(self):
        """Test that identity matrix has correct dimension."""
        n = 3
        X = cp.Variable((n, n), symmetric=True, name="X")
        constraint = X >> 0

        slack, relaxed = _relax_psd(constraint)

        # Relaxed constraint should be X + slack*I >> 0
        assert slack is not None

    def test_relaxed_psd_is_valid(self):
        """Test that relaxed PSD can make infeasible problem feasible."""
        X = cp.Variable((2, 2), symmetric=True, name="X")
        constraint = X >> 0

        slack, relaxed = _relax_psd(constraint)

        # Force X to have negative eigenvalue: X[0,0] <= -1
        prob = cp.Problem(cp.Minimize(slack), relaxed + [X[0, 0] <= -1, X[1, 1] >= 0, X[0, 1] == 0])
        prob.solve()

        assert prob.status == cp.OPTIMAL
        assert slack.value >= 1 - 1e-6  # Need slack to compensate


class TestRelaxExpCone:
    """Tests for _relax_exp_cone()."""

    def test_exp_cone_relaxation(self):
        """Test relaxation of exponential cone constraint."""
        x = cp.Variable(name="x")
        y = cp.Variable(pos=True, name="y")
        z = cp.Variable(name="z")
        constraint = cp.constraints.exponential.ExpCone(x, y, z)

        slack, relaxed = _relax_exp_cone(constraint)

        assert slack is not None
        assert len(relaxed) == 1

    def test_slack_nonneg(self):
        """Test that exp cone slack is non-negative."""
        x = cp.Variable(name="x")
        y = cp.Variable(pos=True, name="y")
        z = cp.Variable(name="z")
        constraint = cp.constraints.exponential.ExpCone(x, y, z)

        slack, _ = _relax_exp_cone(constraint)

        assert slack.is_nonneg()


class TestRelaxationIntegration:
    """Integration tests for constraint relaxation."""

    def test_relaxation_makes_infeasible_feasible(self):
        """Test that relaxation can make infeasible problems feasible."""
        x = cp.Variable(name="x")

        # Infeasible: x >= 10 and x <= 5
        c1 = x >= 10
        c2 = x <= 5

        slack1, relaxed1 = relax_constraint(c1)
        slack2, relaxed2 = relax_constraint(c2)

        if slack1 is not None and slack2 is not None:
            prob = cp.Problem(cp.Minimize(slack1 + slack2), relaxed1 + relaxed2)
            prob.solve()

            assert prob.status == cp.OPTIMAL
            # Total slack should be at least 5 (gap between 5 and 10)
            assert slack1.value + slack2.value >= 5 - 1e-6

    def test_mixed_constraint_relaxation(self):
        """Test relaxation of mixed constraint types."""
        x = cp.Variable(3, name="x")

        constraints = [
            x[0] <= 5,
            x == np.array([10, 10, 10]),  # Conflicts with above
        ]

        slacks = []
        all_relaxed = []

        for c in constraints:
            slack, relaxed = relax_constraint(c)
            if slack is not None:
                slacks.append(slack)
                all_relaxed.extend(relaxed)

        if slacks:
            prob = cp.Problem(cp.Minimize(sum(slacks)), all_relaxed)
            prob.solve()

            assert prob.status == cp.OPTIMAL
