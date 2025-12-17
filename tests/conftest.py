"""Pytest configuration and fixtures."""

import pytest
import cvxpy as cp


@pytest.fixture
def simple_infeasible():
    """Simple infeasible problem: x >= 5 and x <= 3."""
    x = cp.Variable(name="x")
    constraints = [x >= 5, x <= 3]
    prob = cp.Problem(cp.Minimize(x), constraints)
    return prob


@pytest.fixture
def budget_infeasible():
    """Budget allocation problem that's infeasible."""
    alloc = cp.Variable(3, nonneg=True, name="alloc")
    constraints = [
        cp.sum(alloc) <= 100,
        alloc[0] >= 50,
        alloc[1] >= 40,
        alloc[2] >= 30,  # Sum = 120 > 100
    ]
    prob = cp.Problem(cp.Minimize(cp.sum(alloc)), constraints)
    return prob


@pytest.fixture
def soc_infeasible():
    """SOC constraint that conflicts with bounds."""
    x = cp.Variable(3, name="x")
    t = cp.Variable(name="t")
    constraints = [
        cp.norm(x) <= t,
        t <= 1,
        x[0] >= 2,  # norm >= 2 but t <= 1
    ]
    prob = cp.Problem(cp.Minimize(t), constraints)
    return prob


@pytest.fixture
def psd_infeasible():
    """PSD constraint that conflicts with element bound."""
    X = cp.Variable((2, 2), PSD=True, name="X")
    constraints = [
        X[0, 0] <= -1,  # Diagonal must be >= 0 for PSD
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)
    return prob


@pytest.fixture
def feasible_problem():
    """A simple feasible problem."""
    x = cp.Variable(name="x")
    constraints = [x >= 0, x <= 10]
    prob = cp.Problem(cp.Minimize(x), constraints)
    return prob


@pytest.fixture
def equality_infeasible():
    """Infeasible equality constraints."""
    x = cp.Variable(name="x")
    constraints = [x == 5, x == 3]
    prob = cp.Problem(cp.Minimize(x), constraints)
    return prob


# Unboundedness fixtures


@pytest.fixture
def unbounded_below():
    """Minimize x with no lower bound - unbounded below."""
    x = cp.Variable(name="x")
    prob = cp.Problem(cp.Minimize(x), [])
    return prob


@pytest.fixture
def unbounded_above():
    """Maximize x with only lower bound - unbounded above."""
    x = cp.Variable(name="x")
    constraints = [x >= 0]
    prob = cp.Problem(cp.Maximize(x), constraints)
    return prob


@pytest.fixture
def unbounded_direction():
    """Unbounded in a specific direction with multiple variables."""
    x = cp.Variable(2, name="x")
    # Minimize x[0] - x[1] with constraint x[0] + x[1] == 1
    # x[1] can go to +inf while x[0] goes to -inf
    constraints = [x[0] + x[1] == 1]
    prob = cp.Problem(cp.Minimize(x[0] - x[1]), constraints)
    return prob


@pytest.fixture
def bounded_problem():
    """A bounded feasible problem."""
    x = cp.Variable(name="x")
    constraints = [x >= 0, x <= 10]
    prob = cp.Problem(cp.Minimize(x), constraints)
    return prob


@pytest.fixture
def unbounded_nonneg():
    """Maximize with nonneg variable - still unbounded above."""
    x = cp.Variable(nonneg=True, name="x")
    prob = cp.Problem(cp.Maximize(x), [])
    return prob


@pytest.fixture
def unbounded_vector():
    """Unbounded minimization with vector variable."""
    x = cp.Variable(3, name="x")
    # Minimize sum with no bounds - unbounded below
    prob = cp.Problem(cp.Minimize(cp.sum(x)), [])
    return prob
