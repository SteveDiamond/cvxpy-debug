"""
Comprehensive Infeasibility Diagnosis Example
==============================================

This example demonstrates the full range of infeasibility diagnosis
capabilities including IIS finding, elastic relaxation, and various
constraint types.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_multiple_iis():
    """Problem with multiple independent infeasible subsystems."""
    print("=" * 60)
    print("Example 1: Multiple Independent IIS")
    print("=" * 60)

    # Create a problem with TWO separate infeasibility issues
    x = cp.Variable(name="x")
    y = cp.Variable(name="y")

    constraints = [
        # First IIS: x >= 10 and x <= 5 conflict
        x >= 10,
        x <= 5,
        # Second IIS: y >= 20 and y <= 15 conflict
        y >= 20,
        y <= 15,
    ]

    problem = cp.Problem(cp.Minimize(x + y), constraints)

    print("\nProblem has two independent conflicts:")
    print("  - x >= 10 and x <= 5")
    print("  - y >= 20 and y <= 15\n")

    # Diagnose with minimal IIS to find one conflict at a time
    cvxpy_debug.debug(problem, find_minimal_iis=True)

    print("\nNote: With find_minimal_iis=True, only one IIS is reported.")
    print("In practice, fix one IIS, re-solve, and diagnose again.")


def example_equality_conflicts():
    """Infeasibility from conflicting equality constraints."""
    print("\n" + "=" * 60)
    print("Example 2: Equality Constraint Conflicts")
    print("=" * 60)

    x = cp.Variable(3, name="x")

    # These equalities are inconsistent
    constraints = [
        x[0] + x[1] == 10,
        x[1] + x[2] == 15,
        x[0] + x[2] == 8,
        # Adding these up: 2*(x[0] + x[1] + x[2]) == 33
        # But also need: x[0] + x[1] + x[2] == sum of any two / 2
        # This system is over-determined and inconsistent
        x[0] + x[1] + x[2] == 20,  # But 33/2 = 16.5 != 20
    ]

    problem = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

    print("\nSystem of equalities that cannot all be satisfied:")
    print("  x[0] + x[1] = 10")
    print("  x[1] + x[2] = 15")
    print("  x[0] + x[2] = 8")
    print("  x[0] + x[1] + x[2] = 20  (inconsistent with above)\n")

    cvxpy_debug.debug(problem)


def example_bound_vs_constraint():
    """Infeasibility from variable bounds conflicting with constraints."""
    print("\n" + "=" * 60)
    print("Example 3: Variable Bounds vs Constraints")
    print("=" * 60)

    # Variable with nonneg attribute
    x = cp.Variable(nonneg=True, name="x")

    # Constraint that conflicts with nonneg
    constraints = [
        x <= -5,  # Can't be <= -5 if nonneg!
    ]

    problem = cp.Problem(cp.Minimize(x), constraints)

    print("\nVariable x is nonneg=True, but constraint says x <= -5")
    print("The implicit bound x >= 0 conflicts with x <= -5.\n")

    cvxpy_debug.debug(problem)


def example_elastic_relaxation_interpretation():
    """Understanding the slack values from elastic relaxation."""
    print("\n" + "=" * 60)
    print("Example 4: Interpreting Slack Values")
    print("=" * 60)

    # Budget allocation with specific slack interpretation
    allocation = cp.Variable(3, nonneg=True, name="alloc")

    constraints = [
        cp.sum(allocation) <= 100,
        allocation[0] >= 50,
        allocation[1] >= 40,
        allocation[2] >= 35,  # Total minimum = 125
    ]

    problem = cp.Problem(cp.Minimize(cp.sum(allocation)), constraints)

    print("\nBudget of 100, but minimums sum to 125 (gap of 25)")
    print("The slack values show how much each constraint contributes.\n")

    cvxpy_debug.debug(problem)

    print("\n" + "-" * 40)
    print("Interpreting Slack Values:")
    print("-" * 40)
    print("""
The 'slack needed' shows how much each constraint must be relaxed:
- If budget constraint has slack 25, budget needs to increase by 25
- If a minimum constraint has slack 10, that minimum should decrease by 10
- The total slack equals the infeasibility gap (25)

You can choose which constraint to relax based on what's practical.
""")


def example_complex_structure():
    """Infeasibility in a more complex optimization structure."""
    print("\n" + "=" * 60)
    print("Example 5: Complex Multi-Constraint Infeasibility")
    print("=" * 60)

    # Production planning with multiple resources
    n_products = 5
    n_resources = 3

    production = cp.Variable(n_products, nonneg=True, name="production")

    # Resource usage matrix (each product uses resources)
    np.random.seed(42)
    resource_usage = np.random.rand(n_resources, n_products) * 10

    # Available resources (set too low to meet demand)
    resource_limits = np.array([15, 20, 18])

    # Minimum production requirements (set too high)
    min_production = np.array([5, 5, 5, 5, 5])  # 25 total minimum

    constraints = [
        resource_usage @ production <= resource_limits,
        production >= min_production,
    ]

    # Profit coefficients
    profits = np.array([10, 15, 12, 8, 20])
    problem = cp.Problem(cp.Maximize(profits @ production), constraints)

    print("\nProduction planning:")
    print(f"  - {n_products} products, {n_resources} resources")
    print("  - Minimum production requirements conflict with resource limits\n")

    cvxpy_debug.debug(problem, find_minimal_iis=True)


def main():
    """Run all infeasibility diagnosis examples."""
    example_multiple_iis()
    example_equality_conflicts()
    example_bound_vs_constraint()
    example_elastic_relaxation_interpretation()
    example_complex_structure()

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print("""
Key takeaways:
1. IIS (Irreducible Infeasible Subsystem) identifies minimal conflicts
2. Slack values quantify how much constraints need relaxation
3. Variable bounds (nonneg, nonpos) are treated as implicit constraints
4. Use find_minimal_iis=True to get a single, minimal conflicting set
5. Complex problems may have multiple independent IIS
""")


if __name__ == "__main__":
    main()
