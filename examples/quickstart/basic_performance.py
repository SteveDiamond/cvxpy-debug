"""
Basic Performance Analysis Example
==================================

This example demonstrates how to detect performance anti-patterns
in CVXPY problems that can slow down solving.

The classic example: creating constraints in a loop instead of using
vectorized operations.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def main():
    print("=" * 60)
    print("Example 1: Loop-Generated Constraints (Anti-Pattern)")
    print("=" * 60)

    # BAD: Creating constraints in a loop
    n = 100
    x = cp.Variable(n, name="x")
    a = np.random.randn(n)
    b = np.random.randn(n)

    # This creates n separate constraint objects - slow!
    constraints_loop = []
    for i in range(n):
        constraints_loop.append(a[i] * x[i] <= b[i])

    constraints_loop.append(cp.sum(x) == 1)
    constraints_loop.append(x >= 0)

    problem_slow = cp.Problem(cp.Minimize(cp.sum(x)), constraints_loop)

    print("\nProblem with loop-generated constraints (n=100)")
    print("Each constraint created separately in a Python loop.\n")

    problem_slow.solve()
    cvxpy_debug.debug(problem_slow)

    print("\n" + "=" * 60)
    print("Example 2: Vectorized Constraints (Best Practice)")
    print("=" * 60)

    # GOOD: Using vectorized operations
    y = cp.Variable(n, name="y")

    # This creates a single vectorized constraint - fast!
    constraints_vec = [
        cp.multiply(a, y) <= b,  # Single vectorized constraint
        cp.sum(y) == 1,
        y >= 0,
    ]

    problem_fast = cp.Problem(cp.Minimize(cp.sum(y)), constraints_vec)

    print("\nProblem with vectorized constraints")
    print("Single constraint using element-wise operations.\n")

    problem_fast.solve()
    cvxpy_debug.debug(problem_fast)

    print("\n" + "=" * 60)
    print("Example 3: High Constraint-to-Variable Ratio")
    print("=" * 60)

    # Problem with many more constraints than variables
    z = cp.Variable(5, name="z")
    many_constraints = [z >= 0, z <= 1]

    # Add many redundant constraints
    for i in range(50):
        many_constraints.append(cp.sum(z) <= 10 + i * 0.1)

    problem_ratio = cp.Problem(cp.Minimize(cp.sum(z)), many_constraints)

    print("\nProblem with 5 variables but 52+ constraints")
    print("High constraint-to-variable ratio may indicate redundancy.\n")

    problem_ratio.solve()
    cvxpy_debug.debug(problem_ratio)

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
The performance analyzer detects:
- Loop-generated constraints (use vectorized operations instead)
- High constraint-to-variable ratios (may indicate redundancy)
- Scalar operations on vectors (use broadcasting)

Best practices for CVXPY performance:
1. Use vectorized operations: cp.multiply(a, x) instead of loops
2. Use matrix operations: A @ x instead of row-by-row constraints
3. Remove redundant constraints
4. Prefer variable attributes (nonneg=True) over explicit bounds
""")


if __name__ == "__main__":
    main()
