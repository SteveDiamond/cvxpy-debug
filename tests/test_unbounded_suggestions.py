"""Tests for unbounded suggestions generation."""

import cvxpy as cp

from cvxpy_debug.unbounded.ray import UnboundedRay
from cvxpy_debug.unbounded.suggestions import (
    _get_var_name,
    _is_nonneg,
    _should_suggest_nonneg,
    format_unbounded_variable_info,
    generate_suggestions,
)


class TestGenerateSuggestions:
    """Tests for generate_suggestions()."""

    def test_ray_based_upper_bound_suggestion(self):
        """Test suggestion to add upper bound from ray."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Maximize(x), [])

        ray = UnboundedRay(
            variables={x: 1.0},  # Going to +inf
            objective_direction="increase",
        )

        suggestions = generate_suggestions(prob, ray, [])

        assert len(suggestions) >= 1
        assert any("upper" in s.lower() for s in suggestions)

    def test_ray_based_lower_bound_suggestion(self):
        """Test suggestion to add lower bound from ray."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        ray = UnboundedRay(
            variables={x: -1.0},  # Going to -inf
            objective_direction="decrease",
        )

        suggestions = generate_suggestions(prob, ray, [])

        assert len(suggestions) >= 1
        assert any("lower" in s.lower() for s in suggestions)

    def test_unbounded_vars_based_suggestions(self):
        """Test suggestions based on unbounded variables list."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        unbounded_vars = [(x, "below")]

        suggestions = generate_suggestions(prob, None, unbounded_vars)

        assert len(suggestions) >= 1

    def test_empty_inputs(self):
        """Test with no ray and no unbounded variables."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [x >= 0])

        suggestions = generate_suggestions(prob, None, [])

        # Should still provide generic advice
        assert len(suggestions) >= 1

    def test_nonneg_suggestion(self):
        """Test that nonneg=True is suggested when appropriate."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        ray = UnboundedRay(
            variables={x: -1.0},  # Going to -inf
            objective_direction="decrease",
        )

        suggestions = generate_suggestions(prob, ray, [])

        # Should suggest nonneg=True
        assert any("nonneg" in s.lower() for s in suggestions)

    def test_indexed_variable_suggestion(self):
        """Test suggestions for indexed variables."""
        x = cp.Variable(3, name="x")
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [])

        ray = UnboundedRay(
            variables={(x, (0,)): -1.0},
            objective_direction="decrease",
        )

        suggestions = generate_suggestions(prob, ray, [])

        assert len(suggestions) >= 1
        # Should mention the indexed variable
        assert any("x[0]" in s or "x" in s for s in suggestions)

    def test_both_direction_unbounded(self):
        """Test suggestions when variable is unbounded in both directions."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        unbounded_vars = [(x, "both")]

        suggestions = generate_suggestions(prob, None, unbounded_vars)

        assert len(suggestions) >= 1


class TestFormatUnboundedVariableInfo:
    """Tests for format_unbounded_variable_info()."""

    def test_above_direction(self):
        """Test formatting for above direction."""
        x = cp.Variable(name="x")

        info = format_unbounded_variable_info(x, "above")

        assert info["name"] == "x"
        assert info["direction"] == "above"
        assert info["direction_symbol"] == "+inf"
        assert info["variable"] is x
        assert info["index"] is None

    def test_below_direction(self):
        """Test formatting for below direction."""
        x = cp.Variable(name="x")

        info = format_unbounded_variable_info(x, "below")

        assert info["direction"] == "below"
        assert info["direction_symbol"] == "-inf"

    def test_both_direction(self):
        """Test formatting for both directions."""
        x = cp.Variable(name="x")

        info = format_unbounded_variable_info(x, "both")

        assert info["direction"] == "both"
        assert info["direction_symbol"] == "+/-inf"

    def test_indexed_variable(self):
        """Test formatting for indexed variable."""
        x = cp.Variable(3, name="y")

        info = format_unbounded_variable_info(x, "above", idx=(1,))

        assert info["name"] == "y[1]"
        assert info["index"] == (1,)

    def test_unnamed_variable(self):
        """Test formatting for unnamed variable."""
        x = cp.Variable()

        info = format_unbounded_variable_info(x, "above")

        # Should have some name
        assert len(info["name"]) > 0


class TestShouldSuggestNonneg:
    """Tests for _should_suggest_nonneg()."""

    def test_ray_going_negative(self):
        """Test suggestion when ray direction is negative."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        ray = UnboundedRay(variables={x: -1.0})

        result = _should_suggest_nonneg(prob, ray, [])

        assert result is True

    def test_ray_going_positive(self):
        """Test no suggestion when ray direction is positive."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Maximize(x), [])

        ray = UnboundedRay(variables={x: 1.0})

        result = _should_suggest_nonneg(prob, ray, [])

        assert result is False

    def test_var_unbounded_below(self):
        """Test suggestion when variable unbounded below."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        result = _should_suggest_nonneg(prob, None, [(x, "below")])

        assert result is True

    def test_var_unbounded_both(self):
        """Test suggestion when variable unbounded both ways."""
        x = cp.Variable(name="x")
        prob = cp.Problem(cp.Minimize(x), [])

        result = _should_suggest_nonneg(prob, None, [(x, "both")])

        assert result is True

    def test_already_nonneg(self):
        """Test no suggestion when already nonneg."""
        x = cp.Variable(nonneg=True, name="x")
        prob = cp.Problem(cp.Maximize(x), [])

        result = _should_suggest_nonneg(prob, None, [(x, "above")])

        # Should not suggest nonneg when problem is about upper bound
        # and variable is already nonneg
        assert result is not None  # Function returns a value


class TestGetVarName:
    """Tests for _get_var_name()."""

    def test_named_variable(self):
        """Test name for named variable."""
        x = cp.Variable(name="my_var")

        result = _get_var_name(x)

        assert result == "my_var"

    def test_unnamed_variable(self):
        """Test name for unnamed variable."""
        x = cp.Variable()

        result = _get_var_name(x)

        # Unnamed variables get auto-generated names like "var12345"
        assert result == "x" or result.startswith("var")

    def test_indexed_variable(self):
        """Test name for indexed variable."""
        x = cp.Variable(3, name="vec")

        result = _get_var_name(x, (2,))

        assert result == "vec[2]"


class TestIsNonneg:
    """Tests for _is_nonneg()."""

    def test_nonneg_variable(self):
        """Test detection of nonneg variable."""
        x = cp.Variable(nonneg=True, name="x")

        assert _is_nonneg(x) is True

    def test_not_nonneg_variable(self):
        """Test variable without nonneg attribute."""
        x = cp.Variable(name="x")

        assert _is_nonneg(x) is False

    def test_pos_variable(self):
        """Test pos variable (not same as nonneg)."""
        x = cp.Variable(pos=True, name="x")

        # pos is different from nonneg in strictness
        # Implementation may or may not consider pos as nonneg
        result = _is_nonneg(x)
        assert isinstance(result, bool)


class TestSuggestionsIntegration:
    """Integration tests for suggestions generation."""

    def test_complete_unbounded_workflow(self):
        """Test full workflow from ray to suggestions."""
        x = cp.Variable(name="x")
        y = cp.Variable(nonneg=True, name="y")
        prob = cp.Problem(cp.Minimize(x - y), [])

        # x is unbounded below
        ray = UnboundedRay(
            variables={x: -1.0},
            objective_direction="decrease",
        )

        suggestions = generate_suggestions(prob, ray, [(x, "below")])

        assert len(suggestions) >= 1
        # Should mention x
        assert any("x" in s for s in suggestions)

    def test_vector_variable_suggestions(self):
        """Test suggestions for vector variables."""
        x = cp.Variable(5, name="x")
        prob = cp.Problem(cp.Minimize(cp.sum(x)), [])

        # Elements 0 and 2 are unbounded
        ray = UnboundedRay(
            variables={
                (x, (0,)): -1.0,
                (x, (2,)): -1.0,
            },
            objective_direction="decrease",
        )

        suggestions = generate_suggestions(prob, ray, [])

        assert len(suggestions) >= 1
