"""
Basic Infeasibility Diagnosis Example
=====================================

This example demonstrates how to diagnose an infeasible optimization problem
where constraints conflict with each other.

The classic example: allocate resources with minimum requirements that exceed
the available budget.
"""

import cvxpy as cp

import cvxpy_debug


def main():
    # Create a budget allocation problem
    # We have 3 departments that need funding
    allocation = cp.Variable(3, nonneg=True, name="allocation")

    # Constraints:
    # - Total budget is 100
    # - Department 0 needs at least 50
    # - Department 1 needs at least 40
    # - Department 2 needs at least 30
    # Problem: 50 + 40 + 30 = 120 > 100 (infeasible!)
    constraints = [
        cp.sum(allocation) <= 100,  # Total budget constraint
        allocation[0] >= 50,  # Dept 0 minimum
        allocation[1] >= 40,  # Dept 1 minimum
        allocation[2] >= 30,  # Dept 2 minimum
    ]

    # Objective: minimize total spending
    objective = cp.Minimize(cp.sum(allocation))
    problem = cp.Problem(objective, constraints)

    print("=" * 60)
    print("Budget Allocation Problem")
    print("=" * 60)
    print("\nThis problem is infeasible because the minimum requirements")
    print("(50 + 40 + 30 = 120) exceed the budget (100).\n")

    # Diagnose the infeasibility
    cvxpy_debug.debug(problem)

    # The report shows:
    # 1. Which constraints are conflicting
    # 2. How much each constraint needs to be relaxed (slack)
    # 3. Suggestions for fixing the problem

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
The diagnostic shows the Irreducible Infeasible Subsystem (IIS):
- These are the minimal set of constraints that conflict
- The 'slack needed' shows how much each constraint must be relaxed

To fix this problem, you could:
1. Increase the budget to at least 120
2. Reduce one or more department minimums
3. Remove one of the conflicting constraints
""")


if __name__ == "__main__":
    main()
