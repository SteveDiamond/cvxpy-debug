"""
Comprehensive Unboundedness Diagnosis Example
==============================================

This example demonstrates the full range of unboundedness diagnosis
including ray analysis, multi-variable cases, and objective sensitivity.
"""

import cvxpy as cp

import cvxpy_debug


def example_simple_unbounded():
    """Basic unbounded minimization and maximization."""
    print("=" * 60)
    print("Example 1: Simple Unbounded Problems")
    print("=" * 60)

    # Minimize without lower bound
    x = cp.Variable(name="x")
    prob_min = cp.Problem(cp.Minimize(x), [])

    print("\nMinimize x (no constraints)")
    print("x can decrease to -infinity.\n")
    cvxpy_debug.debug(prob_min)

    print("\n" + "-" * 40)

    # Maximize without upper bound
    y = cp.Variable(name="y")
    prob_max = cp.Problem(cp.Maximize(y), [y >= -10])

    print("\nMaximize y subject to y >= -10")
    print("y can increase to +infinity.\n")
    cvxpy_debug.debug(prob_max)


def example_multi_variable_unboundedness():
    """Unboundedness involving multiple variables."""
    print("\n" + "=" * 60)
    print("Example 2: Multi-Variable Unboundedness")
    print("=" * 60)

    # Two variables with a sum constraint
    x = cp.Variable(2, name="x")

    # Minimize x[0] - x[1] subject to x[0] + x[1] == 10
    # Both can go to infinity in opposite directions
    prob = cp.Problem(cp.Minimize(x[0] - x[1]), [x[0] + x[1] == 10])

    print("\nMinimize x[0] - x[1] subject to x[0] + x[1] == 10")
    print("x[0] -> -inf, x[1] -> +inf satisfies constraint")
    print("while driving objective to -infinity.\n")

    cvxpy_debug.debug(prob)

    print("\nThe 'ray' shows the unbounded direction:")
    print("  x[0] decreases (coefficient in objective is +1)")
    print("  x[1] increases (coefficient in objective is -1)")


def example_partial_bounds():
    """Some variables bounded, others not."""
    print("\n" + "=" * 60)
    print("Example 3: Partially Bounded Variables")
    print("=" * 60)

    # x is nonneg (bounded below), y is unbounded
    x = cp.Variable(nonneg=True, name="x")
    y = cp.Variable(name="y")

    # Minimize x + y - only y is unbounded
    prob = cp.Problem(cp.Minimize(x + y), [x + y >= 5])

    print("\nMinimize x + y where x >= 0 (nonneg) but y is free")
    print("x is bounded, but y can go to -infinity.\n")

    cvxpy_debug.debug(prob)

    print("\nThe diagnostic identifies which variables cause unboundedness:")
    print("  x: has lower bound (nonneg=True)")
    print("  y: unbounded below")


def example_objective_sensitivity():
    """Understanding how the objective relates to unboundedness."""
    print("\n" + "=" * 60)
    print("Example 4: Objective Sensitivity Analysis")
    print("=" * 60)

    x = cp.Variable(3, name="x")

    # Objective: 2*x[0] - 3*x[1] + x[2]
    # Only x[1] appears with negative coefficient (minimize wants it positive)
    prob = cp.Problem(
        cp.Minimize(2 * x[0] - 3 * x[1] + x[2]),
        [x[0] >= 0, x[2] >= 0],  # x[1] is unbounded
    )

    print("\nMinimize 2*x[0] - 3*x[1] + x[2]")
    print("x[0] >= 0, x[2] >= 0, but x[1] is free")
    print("Increasing x[1] decreases objective (coefficient -3).\n")

    cvxpy_debug.debug(prob)

    print("\nObjective sensitivity:")
    print("  x[0]: coefficient +2, bounded below, can't decrease objective")
    print("  x[1]: coefficient -3, unbounded, can increase to decrease obj")
    print("  x[2]: coefficient +1, bounded below, can't decrease objective")


def example_constraint_caused_unboundedness():
    """Unboundedness despite having constraints."""
    print("\n" + "=" * 60)
    print("Example 5: Unboundedness with Constraints")
    print("=" * 60)

    x = cp.Variable(2, name="x")

    # Constraints define a half-plane, but objective can still be unbounded
    prob = cp.Problem(
        cp.Minimize(-x[0] - x[1]),  # Want x[0], x[1] to go to +infinity
        [
            x[0] - x[1] <= 10,  # Difference bounded
            x[0] + x[1] >= 0,  # Sum bounded below
        ],
    )

    print("\nMinimize -x[0] - x[1] (maximize sum)")
    print("Constraints: x[0] - x[1] <= 10, x[0] + x[1] >= 0")
    print("The feasible region is unbounded in the (1, 1) direction.\n")

    cvxpy_debug.debug(prob)


def example_vector_unboundedness():
    """Unboundedness with vector variables."""
    print("\n" + "=" * 60)
    print("Example 6: Vector Variable Unboundedness")
    print("=" * 60)

    n = 5
    x = cp.Variable(n, name="x")

    # Minimize sum of x with only some elements bounded
    prob = cp.Problem(
        cp.Minimize(cp.sum(x)),
        [
            x[0] >= 0,
            x[1] >= 0,
            # x[2], x[3], x[4] are unbounded
        ],
    )

    print(f"\nMinimize sum(x) for x with {n} elements")
    print("Only x[0] >= 0 and x[1] >= 0 are constrained.")
    print("x[2], x[3], x[4] can go to -infinity.\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all unboundedness diagnosis examples."""
    example_simple_unbounded()
    example_multi_variable_unboundedness()
    example_partial_bounds()
    example_objective_sensitivity()
    example_constraint_caused_unboundedness()
    example_vector_unboundedness()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways:
1. Unboundedness means the objective can improve without limit
2. Ray analysis shows the direction of unboundedness
3. Variables can be individually bounded but problem still unbounded
4. The objective coefficients determine which direction is 'improving'
5. Common fix: add bounds with nonneg=True or explicit constraints

Fixes by problem type:
- Minimization unbounded below: add lower bounds (nonneg=True)
- Maximization unbounded above: add upper bounds (x <= M)
- Multi-variable: identify which variable(s) need bounding
""")


if __name__ == "__main__":
    main()
