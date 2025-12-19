"""
Cone Constraints Diagnosis Example
==================================

This example demonstrates diagnosing infeasibility and issues
with non-linear cone constraints: SOC, PSD, and exponential cones.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_soc_infeasibility():
    """Second-order cone constraint conflicts."""
    print("=" * 60)
    print("Example 1: Second-Order Cone (SOC) Infeasibility")
    print("=" * 60)

    # SOC constraint: ||x||_2 <= t
    x = cp.Variable(3, name="x")
    t = cp.Variable(name="t")

    # Conflict: norm(x) <= t, but t <= 1 and x[0] >= 2
    # norm(x) >= |x[0]| >= 2, but t <= 1, so norm(x) <= 1 < 2
    constraints = [
        cp.norm(x) <= t,  # SOC constraint
        t <= 1,  # Upper bound on t
        x[0] >= 2,  # Forces norm(x) >= 2
    ]

    prob = cp.Problem(cp.Minimize(t), constraints)

    print("\nSOC constraint conflict:")
    print("  - norm(x) <= t")
    print("  - t <= 1")
    print("  - x[0] >= 2")
    print("This is infeasible: norm(x) >= |x[0]| >= 2 > 1 >= t\n")

    cvxpy_debug.debug(prob)


def example_soc_feasibility_boundary():
    """SOC constraint at the boundary."""
    print("\n" + "=" * 60)
    print("Example 2: SOC at Feasibility Boundary")
    print("=" * 60)

    x = cp.Variable(3, name="x")
    t = cp.Variable(name="t")

    # Just barely feasible
    constraints = [
        cp.norm(x) <= t,
        t <= 2,
        x[0] >= 1.9,  # norm(x) >= 1.9, t >= 1.9, t <= 2
    ]

    prob = cp.Problem(cp.Minimize(t), constraints)

    print("\nSOC constraint near boundary:")
    print("  - norm(x) <= t, t <= 2, x[0] >= 1.9")
    print("  - Barely feasible (optimal t near 1.9)\n")

    prob.solve()
    print(f"Status: {prob.status}")
    print(f"Optimal t = {t.value:.4f}")
    print(f"norm(x) = {np.linalg.norm(x.value):.4f}\n")

    cvxpy_debug.debug(prob)


def example_psd_infeasibility():
    """Positive semidefinite constraint conflicts."""
    print("\n" + "=" * 60)
    print("Example 3: PSD (Positive Semidefinite) Infeasibility")
    print("=" * 60)

    # PSD matrix with conflicting element constraint
    X = cp.Variable((2, 2), PSD=True, name="X")

    # PSD requires diagonal elements >= 0, but we force X[0,0] <= -1
    constraints = [
        X[0, 0] <= -1,  # Conflicts with PSD requirement
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)

    print("\nPSD constraint conflict:")
    print("  - X is PSD (implies X[0,0] >= 0)")
    print("  - X[0,0] <= -1")
    print("This is infeasible: PSD diagonal must be non-negative.\n")

    cvxpy_debug.debug(prob)


def example_psd_eigenvalue_constraint():
    """PSD with eigenvalue constraints."""
    print("\n" + "=" * 60)
    print("Example 4: PSD with Eigenvalue Bounds")
    print("=" * 60)

    n = 3
    X = cp.Variable((n, n), symmetric=True, name="X")

    # Want minimum eigenvalue >= 1, but trace <= 1
    # For n=3, if all eigenvalues >= 1, trace >= 3
    constraints = [
        X >> np.eye(n),  # X - I is PSD, so min eigenvalue >= 1
        cp.trace(X) <= 1,  # But trace = sum of eigenvalues <= 1
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(X)), constraints)

    print("\nPSD eigenvalue conflict:")
    print("  - X >> I (minimum eigenvalue >= 1)")
    print("  - trace(X) <= 1")
    print(f"Infeasible: if min eigenvalue >= 1, trace >= {n}\n")

    cvxpy_debug.debug(prob)


def example_multiple_psd_constraints():
    """Multiple PSD constraints interacting."""
    print("\n" + "=" * 60)
    print("Example 5: Multiple PSD Constraints")
    print("=" * 60)

    n = 2
    X = cp.Variable((n, n), symmetric=True, name="X")
    Y = cp.Variable((n, n), symmetric=True, name="Y")

    # Constraints that interact
    A = np.array([[1, 0.5], [0.5, 1]])

    constraints = [
        X >> 0,
        Y >> 0,
        X + Y == A,  # A is positive definite
        X[0, 0] >= 2,  # Forces X to be "large"
        Y[0, 0] >= 2,  # Forces Y to be "large"
        # But (X+Y)[0,0] = 1, so X[0,0] + Y[0,0] = 1 < 4
    ]

    prob = cp.Problem(cp.Minimize(cp.trace(X) + cp.trace(Y)), constraints)

    print("\nMultiple PSD constraint conflict:")
    print("  - X >> 0, Y >> 0")
    print("  - X + Y = A (where A[0,0] = 1)")
    print("  - X[0,0] >= 2, Y[0,0] >= 2")
    print("Infeasible: X[0,0] + Y[0,0] >= 4 but (X+Y)[0,0] = 1\n")

    cvxpy_debug.debug(prob)


def example_exp_cone():
    """Exponential cone constraints."""
    print("\n" + "=" * 60)
    print("Example 6: Exponential Cone")
    print("=" * 60)

    # Exponential cone: y * exp(x/y) <= z, y > 0
    # Using log-sum-exp: log(sum(exp(x))) <= t
    x = cp.Variable(3, name="x")
    t = cp.Variable(name="t")

    # log(sum(exp(x))) <= t, but also x >= 10
    # exp(10) ≈ 22026, so log(3 * 22026) ≈ 11.1
    constraints = [
        cp.log_sum_exp(x) <= t,
        x >= 10,
        t <= 5,  # Too restrictive
    ]

    prob = cp.Problem(cp.Minimize(t), constraints)

    print("\nExponential cone conflict:")
    print("  - log_sum_exp(x) <= t")
    print("  - x >= 10 (so log_sum_exp(x) >= log(3*exp(10)) ≈ 11.1)")
    print("  - t <= 5")
    print("Infeasible: log_sum_exp(x) > 11 but t <= 5\n")

    cvxpy_debug.debug(prob)


def example_mixed_cones():
    """Problem with multiple cone types."""
    print("\n" + "=" * 60)
    print("Example 7: Mixed Cone Constraints")
    print("=" * 60)

    x = cp.Variable(3, name="x")
    X = cp.Variable((2, 2), PSD=True, name="X")
    t = cp.Variable(name="t")

    constraints = [
        # SOC constraint
        cp.norm(x) <= t,
        # PSD constraint on X
        X >> 0,
        # Linking constraint
        cp.sum(x) == cp.trace(X),
        # Bounds that cause conflict
        t <= 1,
        cp.trace(X) >= 10,  # So sum(x) >= 10, but norm(x) >= sum(x)/sqrt(3)
    ]

    prob = cp.Problem(cp.Minimize(t + cp.trace(X)), constraints)

    print("\nMixed cone constraints:")
    print("  - norm(x) <= t, t <= 1 (SOC)")
    print("  - X >> 0, trace(X) >= 10 (PSD)")
    print("  - sum(x) = trace(X)")
    print("Conflict: sum(x) >= 10 implies norm(x) >= 10/sqrt(3) > 1 >= t\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all cone constraint examples."""
    example_soc_infeasibility()
    example_soc_feasibility_boundary()
    example_psd_infeasibility()
    example_psd_eigenvalue_constraint()
    example_multiple_psd_constraints()
    example_exp_cone()
    example_mixed_cones()

    print("\n" + "=" * 60)
    print("Summary: Cone Constraints")
    print("=" * 60)
    print("""
Cone constraints in CVXPY:

1. SOC (Second-Order Cone): ||x||_2 <= t
   - Used for: norm bounds, robust optimization
   - Common issue: norm lower bound conflicts with t upper bound

2. PSD (Positive Semidefinite): X >> 0
   - Used for: covariance matrices, SDP relaxations
   - Common issues:
     * Diagonal elements must be non-negative
     * Trace (sum of eigenvalues) constraints
     * Off-diagonal magnitude limits

3. Exponential Cone: y * exp(x/y) <= z
   - Used for: log-sum-exp, entropy, KL divergence
   - Common issue: exp grows quickly, bounds conflict

Debugging tips:
- Check if bounds on individual elements conflict with cone structure
- Trace constraints interact with eigenvalue requirements
- Norms and traces provide implicit lower/upper bounds
""")


if __name__ == "__main__":
    main()
