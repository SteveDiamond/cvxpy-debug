"""
Basic Unboundedness Diagnosis Example
=====================================

This example demonstrates how to diagnose an unbounded optimization problem
where the objective can improve without limit.

The classic example: minimize a variable with no lower bound.
"""

import cvxpy as cp

import cvxpy_debug


def main():
    print("=" * 60)
    print("Example 1: Simple Unbounded Minimization")
    print("=" * 60)

    # Create a simple unbounded problem
    # Minimize x with no constraints - x can go to -infinity
    x = cp.Variable(name="x")
    problem = cp.Problem(cp.Minimize(x), [])

    print("\nProblem: minimize x (no constraints)")
    print("This is unbounded because x can decrease to -infinity.\n")

    # Diagnose the unboundedness
    cvxpy_debug.debug(problem)

    print("\n" + "=" * 60)
    print("Example 2: Unbounded Maximization")
    print("=" * 60)

    # Maximize x with only a lower bound
    # x can go to +infinity
    y = cp.Variable(name="y")
    problem2 = cp.Problem(cp.Maximize(y), [y >= 0])

    print("\nProblem: maximize y subject to y >= 0")
    print("This is unbounded because y can increase to +infinity.\n")

    cvxpy_debug.debug(problem2)

    print("\n" + "=" * 60)
    print("Example 3: Multi-Variable Unboundedness")
    print("=" * 60)

    # Minimize x[0] - x[1] with constraint x[0] + x[1] == 1
    # x[1] can go to +inf while x[0] goes to -inf (objective decreases)
    z = cp.Variable(2, name="z")
    problem3 = cp.Problem(cp.Minimize(z[0] - z[1]), [z[0] + z[1] == 1])

    print("\nProblem: minimize z[0] - z[1] subject to z[0] + z[1] == 1")
    print("Unbounded: z[1] -> +inf, z[0] -> -inf keeps constraint satisfied")
    print("while objective z[0] - z[1] -> -infinity.\n")

    cvxpy_debug.debug(problem3)

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
The diagnostic shows:
- Which variables are unbounded
- The direction of unboundedness (increasing/decreasing)
- How to fix it (add bounds like nonneg=True)

Common fixes:
1. Add nonneg=True to variables that should be non-negative
2. Add explicit bounds: x >= lower, x <= upper
3. Add constraints that implicitly bound the variable
""")


if __name__ == "__main__":
    main()
