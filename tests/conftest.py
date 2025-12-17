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
