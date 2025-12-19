"""
Comprehensive Numerical Issues Diagnosis Example
=================================================

This example demonstrates diagnosing numerical stability issues
including scaling, conditioning, constraint violations, and
solver-specific recommendations.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_extreme_scaling():
    """Problem with extreme coefficient scaling."""
    print("=" * 60)
    print("Example 1: Extreme Coefficient Scaling")
    print("=" * 60)

    n = 10
    x = cp.Variable(n, name="x")

    # Create coefficients spanning many orders of magnitude
    # This is a common source of numerical issues
    np.random.seed(42)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, :] = np.random.randn(n) * (10 ** (i - 5))  # 1e-5 to 1e4

    b = np.ones(n)

    prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

    print("\nProblem with coefficient matrix spanning 1e-5 to 1e4")
    print("This 9 orders of magnitude range causes numerical issues.\n")

    prob.solve()
    cvxpy_debug.debug(prob)


def example_ill_conditioned():
    """Problem with ill-conditioned constraint matrix."""
    print("\n" + "=" * 60)
    print("Example 2: Ill-Conditioned Constraint Matrix")
    print("=" * 60)

    n = 5
    x = cp.Variable(n, name="x")

    # Create a nearly singular matrix (high condition number)
    np.random.seed(42)
    U = np.random.randn(n, n)
    # Create singular values with large ratio
    s = np.array([1.0, 0.1, 0.01, 0.001, 0.0001])
    A = U @ np.diag(s) @ U.T  # Condition number ~ 1e4

    b = np.ones(n)

    prob = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x == b, x >= -10, x <= 10])

    print("\nProblem with ill-conditioned matrix (condition number ~ 1e4)")
    print("High condition numbers amplify floating-point errors.\n")

    prob.solve()
    cvxpy_debug.debug(prob, include_conditioning=True)


def example_constraint_violations():
    """Detecting constraint violations in the solution."""
    print("\n" + "=" * 60)
    print("Example 3: Constraint Violations")
    print("=" * 60)

    n = 20
    x = cp.Variable(n, name="x")

    # Create a problem that may have small violations
    np.random.seed(42)
    A = np.random.randn(n, n) * 100
    b = np.random.randn(n) * 100

    prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), [A @ x == b])

    print("\nLeast squares with equality constraints")
    print("Checking if the solution satisfies constraints within tolerance.\n")

    prob.solve()
    cvxpy_debug.debug(prob)

    print("\nThe violation analysis shows:")
    print("  - Maximum constraint violation")
    print("  - Which constraints are violated")
    print("  - Whether violations exceed solver tolerances")


def example_solver_comparison():
    """Comparing behavior across different solvers."""
    print("\n" + "=" * 60)
    print("Example 4: Solver-Specific Behavior")
    print("=" * 60)

    n = 10
    x = cp.Variable(n, name="x")

    # Problem that may behave differently across solvers
    np.random.seed(42)
    A = np.random.randn(n, n)
    A = A @ A.T + 0.1 * np.eye(n)  # Make positive definite
    c = np.random.randn(n)

    prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, A) + c @ x), [cp.sum(x) == 1, x >= 0])

    print("\nQuadratic program solved with default solver")
    print("The diagnostic extracts solver-specific statistics.\n")

    prob.solve()
    cvxpy_debug.debug(prob)


def example_near_infeasibility():
    """Problem that is barely feasible (near the boundary)."""
    print("\n" + "=" * 60)
    print("Example 5: Near-Infeasibility")
    print("=" * 60)

    x = cp.Variable(3, nonneg=True, name="x")

    # Constraints that are barely satisfiable
    constraints = [
        cp.sum(x) <= 100,
        x[0] >= 33.333,
        x[1] >= 33.333,
        x[2] >= 33.333,
        # Sum of minimums = 99.999, just under 100
    ]

    prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

    print("\nProblem is barely feasible (minimums sum to 99.999, limit is 100)")
    print("Near-infeasible problems can have numerical sensitivity.\n")

    prob.solve()
    cvxpy_debug.debug(prob)


def example_recommended_rescaling():
    """Demonstrating how to rescale a problem."""
    print("\n" + "=" * 60)
    print("Example 6: Rescaling a Problem")
    print("=" * 60)

    # Original badly-scaled problem
    x = cp.Variable(2, name="x")
    A = np.array([[1e6, 1e-6], [1e-6, 1e6]])
    b = np.array([1e6, 1e6])

    prob_bad = cp.Problem(cp.Minimize(cp.sum(x)), [A @ x <= b, x >= 0])

    print("\nOriginal problem with coefficients from 1e-6 to 1e6:")
    prob_bad.solve()
    cvxpy_debug.debug(prob_bad)

    print("\n" + "-" * 40)
    print("Rescaled Problem:")
    print("-" * 40)

    # Rescaled version
    # Scale rows of A and b by 1/max(|row|)
    row_scales = np.max(np.abs(A), axis=1)
    A_scaled = A / row_scales[:, np.newaxis]
    b_scaled = b / row_scales

    y = cp.Variable(2, name="y")
    prob_good = cp.Problem(cp.Minimize(cp.sum(y)), [A_scaled @ y <= b_scaled, y >= 0])

    print("\nRescaled so each row has max coefficient = 1:")
    prob_good.solve()
    cvxpy_debug.debug(prob_good)

    print("\nRescaling improved the coefficient range from 1e12 to ~1.")


def main():
    """Run all numerical issues diagnosis examples."""
    example_extreme_scaling()
    example_ill_conditioned()
    example_constraint_violations()
    example_solver_comparison()
    example_near_infeasibility()
    example_recommended_rescaling()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways:
1. Coefficient scaling: keep all coefficients within 2-3 orders of magnitude
2. Condition number: high values (>1e6) indicate numerical sensitivity
3. Constraint violations: check if solution satisfies constraints
4. Solver choice: some problems work better with specific solvers

Common fixes:
- Rescale variables: x_scaled = x / scale_factor
- Normalize constraints: divide each row by its largest coefficient
- Use higher-precision solvers or tighter tolerances
- Reformulate to avoid extreme coefficients
""")


if __name__ == "__main__":
    main()
