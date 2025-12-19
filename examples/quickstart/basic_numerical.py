"""
Basic Numerical Issues Diagnosis Example
=========================================

This example demonstrates how to diagnose numerical/scaling issues
that can cause solvers to return inaccurate solutions.

The classic example: coefficients that vary by many orders of magnitude.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def main():
    print("=" * 60)
    print("Example 1: Badly Scaled Problem")
    print("=" * 60)

    # Create a problem with very different magnitude coefficients
    # This can cause numerical precision issues
    x = cp.Variable(3, name="x")

    # Coefficients range from 1e-8 to 1e8 - very bad scaling!
    A = np.array(
        [
            [1e8, 1e-8, 1.0],
            [1e-8, 1e8, 1.0],
            [1.0, 1.0, 1e8],
        ]
    )
    b = np.array([1e8, 1e8, 1e8])

    constraints = [A @ x <= b, x >= 0]
    objective = cp.Minimize(cp.sum(x))
    problem = cp.Problem(objective, constraints)

    print("\nProblem with coefficient range from 1e-8 to 1e8")
    print("This extreme scaling can cause numerical issues.\n")

    # Solve and diagnose
    problem.solve()
    cvxpy_debug.debug(problem)

    print("\n" + "=" * 60)
    print("Example 2: Well-Scaled Problem (for comparison)")
    print("=" * 60)

    # Create a well-scaled version
    y = cp.Variable(3, name="y")

    # Coefficients all around 1.0 - good scaling
    A_good = np.array(
        [
            [1.0, 0.5, 0.3],
            [0.5, 1.0, 0.4],
            [0.3, 0.4, 1.0],
        ]
    )
    b_good = np.array([1.0, 1.0, 1.0])

    constraints_good = [A_good @ y <= b_good, y >= 0]
    objective_good = cp.Minimize(cp.sum(y))
    problem_good = cp.Problem(objective_good, constraints_good)

    print("\nProblem with coefficients all near 1.0")
    print("Well-scaled problems are more numerically stable.\n")

    problem_good.solve()
    cvxpy_debug.debug(problem_good)

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
The diagnostic analyzes:
- Coefficient magnitude ranges (scaling)
- Condition number estimates
- Constraint violations in the solution
- Solver-specific statistics

Common fixes for numerical issues:
1. Rescale variables: x_scaled = x / scale_factor
2. Normalize constraints to have similar magnitudes
3. Use higher precision solvers (e.g., SCS with higher accuracy)
4. Reformulate the problem to avoid extreme coefficients
""")


if __name__ == "__main__":
    main()
