"""
Resource Allocation Example
===========================

This example demonstrates debugging common issues in resource
allocation and budget optimization problems.

Common issues:
- Budget constraints conflicting with minimum requirements
- Capacity limits causing infeasibility
- Multi-period planning conflicts
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_budget_allocation():
    """Classic budget allocation with conflicting minimums."""
    print("=" * 60)
    print("Example 1: Department Budget Allocation")
    print("=" * 60)

    # Department budget allocation
    n_depts = 5
    dept_names = ["Engineering", "Marketing", "Sales", "Support", "R&D"]

    allocation = cp.Variable(n_depts, nonneg=True, name="allocation")

    # Budget and requirements
    total_budget = 1_000_000
    min_allocations = np.array([300_000, 200_000, 250_000, 150_000, 200_000])
    # Sum of minimums = 1,100,000 > 1,000,000

    constraints = [
        cp.sum(allocation) <= total_budget,
    ]
    # Add minimum allocation constraints
    for i, (name, min_val) in enumerate(zip(dept_names, min_allocations)):
        constraints.append(allocation[i] >= min_val)

    # Maximize some utility (e.g., weighted sum)
    utility_weights = np.array([1.2, 1.0, 1.1, 0.9, 1.3])  # R&D is most valuable
    prob = cp.Problem(cp.Maximize(utility_weights @ allocation), constraints)

    print(f"\nBudget: ${total_budget:,}")
    print(f"Minimum requirements: ${sum(min_allocations):,}")
    print(f"Shortfall: ${sum(min_allocations) - total_budget:,}\n")

    cvxpy_debug.debug(prob)

    print("\nThe diagnostic shows which minimums conflict with budget.")
    print("Options: increase budget, reduce minimums, or eliminate a department.")


def example_capacity_planning():
    """Manufacturing capacity with demand constraints."""
    print("\n" + "=" * 60)
    print("Example 2: Manufacturing Capacity Planning")
    print("=" * 60)

    n_products = 4
    n_machines = 3

    production = cp.Variable(n_products, nonneg=True, name="production")

    # Machine hours required per product
    machine_hours = np.array(
        [
            [2, 3, 1, 4],  # Machine 1
            [1, 2, 3, 2],  # Machine 2
            [3, 1, 2, 1],  # Machine 3
        ]
    )

    # Available hours per machine
    available_hours = np.array([100, 80, 90])

    # Minimum demand that must be met
    min_demand = np.array([20, 15, 25, 10])  # Units

    constraints = [
        machine_hours @ production <= available_hours,  # Capacity limits
        production >= min_demand,  # Must meet demand
    ]

    # Maximize profit
    profit_per_unit = np.array([10, 15, 12, 8])
    prob = cp.Problem(cp.Maximize(profit_per_unit @ production), constraints)

    print("\nManufacturing problem:")
    print(f"  - {n_products} products, {n_machines} machines")
    print(f"  - Minimum demand: {min_demand}")
    print(f"  - Available hours: {available_hours}")

    # Check if minimum demand is achievable
    min_hours_needed = machine_hours @ min_demand
    print(f"\nHours needed for min demand: {min_hours_needed}")
    print(f"Hours available: {available_hours}")

    if any(min_hours_needed > available_hours):
        print("\nInfeasible: not enough capacity for minimum demand.\n")

    cvxpy_debug.debug(prob)


def example_workforce_scheduling():
    """Workforce scheduling with skill requirements."""
    print("\n" + "=" * 60)
    print("Example 3: Workforce Scheduling")
    print("=" * 60)

    n_employees = 10
    n_shifts = 5
    n_skills = 3

    # Assignment matrix: employee i to shift j
    assignment = cp.Variable((n_employees, n_shifts), boolean=False, nonneg=True)
    # Note: Using continuous relaxation for this example

    # Employee skill levels (rows: employees, cols: skills)
    np.random.seed(42)
    skills = np.random.randint(0, 5, (n_employees, n_skills))

    # Minimum skill requirement per shift
    min_skill_per_shift = np.array(
        [
            [8, 6, 10],  # Shift 1 needs high skill 3
            [5, 8, 5],  # Shift 2 needs high skill 2
            [10, 5, 5],  # Shift 3 needs high skill 1
            [6, 6, 6],  # Shift 4 balanced
            [12, 8, 10],  # Shift 5 needs lots of everything
        ]
    )

    constraints = [
        # Each employee works at most one shift
        cp.sum(assignment, axis=1) <= 1,
        # Each shift needs at least 2 people
        cp.sum(assignment, axis=0) >= 2,
    ]

    # Skill requirements per shift
    for j in range(n_shifts):
        for k in range(n_skills):
            # Total skill k in shift j must meet requirement
            constraints.append(skills[:, k] @ assignment[:, j] >= min_skill_per_shift[j, k])

    # Minimize total overtime (assume assignment > 0.5 means overtime)
    prob = cp.Problem(cp.Minimize(cp.sum(assignment)), constraints)

    print(f"\nScheduling {n_employees} employees across {n_shifts} shifts")
    print(f"Each shift needs minimum skills: {min_skill_per_shift[4]} (shift 5)")
    print(f"Total skill available: {skills.sum(axis=0)}")
    print("\nShift 5 may be infeasible if skill requirements are too high.\n")

    cvxpy_debug.debug(prob)


def example_multi_period_planning():
    """Multi-period resource planning with carryover."""
    print("\n" + "=" * 60)
    print("Example 4: Multi-Period Resource Planning")
    print("=" * 60)

    n_periods = 4
    n_resources = 3

    # Allocation per period
    allocation = cp.Variable((n_periods, n_resources), nonneg=True, name="alloc")

    # Budget per period (decreasing)
    period_budgets = np.array([100, 80, 60, 40])

    # Minimum requirements per period (increasing)
    min_requirements = np.array(
        [
            [20, 20, 20],  # Period 1
            [30, 25, 25],  # Period 2
            [40, 35, 30],  # Period 3
            [50, 40, 35],  # Period 4 - high requirements, low budget
        ]
    )

    constraints = []
    for t in range(n_periods):
        constraints.append(cp.sum(allocation[t, :]) <= period_budgets[t])
        constraints.append(allocation[t, :] >= min_requirements[t, :])

    # Maximize total value
    value_weights = np.ones((n_periods, n_resources))
    prob = cp.Problem(cp.Maximize(cp.sum(cp.multiply(value_weights, allocation))), constraints)

    print("\nMulti-period planning:")
    for t in range(n_periods):
        req_sum = sum(min_requirements[t, :])
        print(f"  Period {t+1}: budget={period_budgets[t]}, min requirements sum={req_sum}")

    print("\nPeriod 4 is infeasible: requirements (125) exceed budget (40).\n")

    cvxpy_debug.debug(prob)


def example_fair_allocation():
    """Fair allocation with min-max objectives."""
    print("\n" + "=" * 60)
    print("Example 5: Fair Resource Allocation")
    print("=" * 60)

    n_groups = 4
    allocation = cp.Variable(n_groups, nonneg=True, name="allocation")

    total_resources = 100
    min_per_group = 30  # Each group wants at least 30

    # Fairness: maximize the minimum allocation (max-min fairness)
    min_allocation = cp.Variable(name="min_alloc")

    constraints = [
        cp.sum(allocation) <= total_resources,
        allocation >= min_per_group,  # Minimum guarantee
        allocation >= min_allocation,  # For max-min objective
    ]

    prob = cp.Problem(cp.Maximize(min_allocation), constraints)

    print(f"\nFair allocation of {total_resources} resources to {n_groups} groups")
    print(f"Each group requires minimum {min_per_group}")
    print(f"Total minimum: {n_groups * min_per_group}")

    if n_groups * min_per_group > total_resources:
        print(f"\nInfeasible: {n_groups * min_per_group} > {total_resources}\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all resource allocation examples."""
    example_budget_allocation()
    example_capacity_planning()
    example_workforce_scheduling()
    example_multi_period_planning()
    example_fair_allocation()

    print("\n" + "=" * 60)
    print("Summary: Resource Allocation Issues")
    print("=" * 60)
    print("""
Common issues and fixes:

1. BUDGET vs MINIMUM REQUIREMENTS
   - Sum of minimums exceeds total budget
   - Fix: Increase budget, reduce minimums, or prioritize

2. CAPACITY LIMITS
   - Production/service capacity can't meet demand
   - Fix: Add capacity, reduce demand, or allow backlog

3. SKILL/CAPABILITY MISMATCH
   - Required skills exceed available workforce
   - Fix: Hire specialists, reduce requirements, or cross-train

4. TEMPORAL CONFLICTS
   - Requirements grow while resources shrink
   - Fix: Allow carryover, borrowing, or re-prioritization

5. FAIRNESS CONSTRAINTS
   - Equal treatment requirements may be infeasible
   - Fix: Use proportional allocation or priority-based

Use cvxpy_debug to identify exactly which constraints conflict.
""")


if __name__ == "__main__":
    main()
