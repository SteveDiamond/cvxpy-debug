"""
Scheduling Example
==================

This example demonstrates debugging common issues in task and
job scheduling optimization problems.

Common issues:
- Precedence constraints causing cycles or infeasibility
- Resource conflicts across parallel tasks
- Deadline constraints that can't be met
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_task_scheduling():
    """Basic task scheduling with precedence constraints."""
    print("=" * 60)
    print("Example 1: Task Scheduling with Precedence")
    print("=" * 60)

    n_tasks = 5
    task_names = ["Design", "Develop", "Test", "Document", "Deploy"]
    task_durations = np.array([5, 10, 4, 3, 2])  # Days

    # Start times for each task
    start_times = cp.Variable(n_tasks, nonneg=True, name="start")

    # Precedence constraints (task i must finish before task j starts)
    # Design -> Develop -> Test -> Deploy
    # Design -> Document (can happen in parallel with Develop)
    precedence = [
        (0, 1),  # Design before Develop
        (1, 2),  # Develop before Test
        (2, 4),  # Test before Deploy
        (0, 3),  # Design before Document
        (3, 4),  # Document before Deploy
    ]

    constraints = []
    for i, j in precedence:
        # Task j starts after task i finishes
        constraints.append(start_times[j] >= start_times[i] + task_durations[i])

    # Deadline constraint (too tight)
    deadline = 15  # Days - but critical path is Design(5) + Develop(10) + Test(4) + Deploy(2) = 21
    constraints.append(start_times[4] + task_durations[4] <= deadline)

    # Minimize makespan (completion time of last task)
    prob = cp.Problem(cp.Minimize(start_times[4] + task_durations[4]), constraints)

    print("\nTask scheduling with precedence constraints:")
    for i, j in precedence:
        print(f"  {task_names[i]} ({task_durations[i]}d) -> {task_names[j]}")
    print(f"\nDeadline: {deadline} days")
    print("Critical path: 5 + 10 + 4 + 2 = 21 days (infeasible!)\n")

    cvxpy_debug.debug(prob)


def example_resource_constrained():
    """Scheduling with limited resources."""
    print("\n" + "=" * 60)
    print("Example 2: Resource-Constrained Scheduling")
    print("=" * 60)

    n_tasks = 4
    n_resources = 2
    n_periods = 10

    # Binary assignment: task i runs in period t
    # Using continuous relaxation for demonstration
    schedule = cp.Variable((n_tasks, n_periods), nonneg=True, name="schedule")

    task_durations = np.array([3, 4, 2, 5])
    resource_usage = np.array(
        [
            [2, 1],  # Task 0 uses 2 of resource 1, 1 of resource 2
            [1, 2],
            [2, 2],
            [1, 1],
        ]
    )
    resource_capacity = np.array([3, 3])  # Per period

    constraints = []

    # Each task runs for its duration (simplified: total time assigned)
    for i in range(n_tasks):
        constraints.append(cp.sum(schedule[i, :]) >= task_durations[i])

    # Resource capacity per period
    for t in range(n_periods):
        for r in range(n_resources):
            constraints.append(resource_usage[:, r] @ schedule[:, t] <= resource_capacity[r])

    # All tasks must complete
    constraints.append(schedule <= 1)  # At most 1 unit per period

    # Minimize makespan (last period with activity)
    prob = cp.Problem(cp.Minimize(cp.sum(schedule)), constraints)

    total_work = sum(task_durations)

    print(f"\n{n_tasks} tasks over {n_periods} periods")
    print(f"Total work needed: {total_work} task-periods")
    print(f"Resource capacity per period: {resource_capacity}")

    cvxpy_debug.debug(prob)


def example_machine_scheduling():
    """Job shop scheduling with machine conflicts."""
    print("\n" + "=" * 60)
    print("Example 3: Machine Scheduling Conflicts")
    print("=" * 60)

    n_jobs = 3

    # Each job has multiple operations, each on a specific machine
    # Job 0: Machine 0 (3h) -> Machine 1 (2h)
    # Job 1: Machine 1 (4h) -> Machine 0 (3h)
    # Job 2: Machine 0 (2h) -> Machine 1 (5h)

    # Start times for each operation
    # op[job][operation]
    op_start = [
        [cp.Variable(nonneg=True, name="j0_op0"), cp.Variable(nonneg=True, name="j0_op1")],
        [cp.Variable(nonneg=True, name="j1_op0"), cp.Variable(nonneg=True, name="j1_op1")],
        [cp.Variable(nonneg=True, name="j2_op0"), cp.Variable(nonneg=True, name="j2_op1")],
    ]

    # Duration of each operation
    durations = [[3, 2], [4, 3], [2, 5]]

    constraints = []

    # Precedence within jobs
    for j in range(n_jobs):
        constraints.append(op_start[j][1] >= op_start[j][0] + durations[j][0])

    # Machine conflicts - operations on same machine can't overlap
    # This is a disjunctive constraint, simplified here
    # For Machine 0: j0_op0, j1_op1, j2_op0
    # For Machine 1: j0_op1, j1_op0, j2_op1

    # Simplified: require sequential processing on each machine
    # Machine 0: j0_op0 -> j1_op1 -> j2_op0 (one possible order)
    constraints.append(op_start[1][1] >= op_start[0][0] + durations[0][0])
    constraints.append(op_start[2][0] >= op_start[1][1] + durations[1][1])

    # Machine 1: j1_op0 -> j0_op1 -> j2_op1 (one possible order)
    constraints.append(op_start[0][1] >= op_start[1][0] + durations[1][0])
    constraints.append(op_start[2][1] >= op_start[0][1] + durations[0][1])

    # Very tight deadline
    deadline = 8
    for j in range(n_jobs):
        constraints.append(op_start[j][1] + durations[j][1] <= deadline)

    # Minimize makespan
    makespan = cp.Variable(name="makespan")
    for j in range(n_jobs):
        constraints.append(makespan >= op_start[j][1] + durations[j][1])

    prob = cp.Problem(cp.Minimize(makespan), constraints)

    print("\nJob shop with 3 jobs, 2 machines:")
    print("  Job 0: M0(3h) -> M1(2h)")
    print("  Job 1: M1(4h) -> M0(3h)")
    print("  Job 2: M0(2h) -> M1(5h)")
    print(f"\nDeadline: {deadline} hours")
    print("This may be infeasible due to machine conflicts.\n")

    cvxpy_debug.debug(prob)


def example_shift_scheduling():
    """Employee shift scheduling with coverage requirements."""
    print("\n" + "=" * 60)
    print("Example 4: Shift Scheduling")
    print("=" * 60)

    n_employees = 5
    n_shifts = 7  # One week

    # Assignment: employee i works shift j
    works = cp.Variable((n_employees, n_shifts), nonneg=True, name="works")

    # Coverage requirements per shift
    min_coverage = np.array([3, 4, 4, 3, 3, 5, 5])  # Weekend needs more

    # Employee constraints
    max_shifts_per_employee = 4  # Max 4 shifts per week
    min_shifts_per_employee = 2  # Min 2 shifts per week (part-time)

    constraints = [
        works <= 1,  # Binary (relaxed)
        # Coverage
        cp.sum(works, axis=0) >= min_coverage,
        # Employee limits
        cp.sum(works, axis=1) <= max_shifts_per_employee,
        cp.sum(works, axis=1) >= min_shifts_per_employee,
    ]

    # Minimize total shifts (cost)
    prob = cp.Problem(cp.Minimize(cp.sum(works)), constraints)

    total_min_coverage = sum(min_coverage)
    total_max_capacity = n_employees * max_shifts_per_employee
    total_min_shifts = n_employees * min_shifts_per_employee

    print(f"\n{n_employees} employees, {n_shifts} shifts")
    print(f"Coverage needed: {min_coverage} (total: {total_min_coverage})")
    print(f"Employee min/max: {min_shifts_per_employee}/{max_shifts_per_employee} shifts")
    print(f"\nCapacity: {total_max_capacity}, Minimum work: {total_min_shifts}")

    if total_min_coverage > total_max_capacity:
        print(f"Infeasible: need {total_min_coverage} but max is {total_max_capacity}\n")

    cvxpy_debug.debug(prob)


def example_cyclic_precedence():
    """Detect cyclic precedence constraints."""
    print("\n" + "=" * 60)
    print("Example 5: Cyclic Precedence (Always Infeasible)")
    print("=" * 60)

    n_tasks = 3

    start_times = cp.Variable(n_tasks, nonneg=True, name="start")
    durations = np.array([2, 3, 4])

    # Cyclic precedence: A -> B -> C -> A (impossible!)
    constraints = [
        start_times[1] >= start_times[0] + durations[0],  # A before B
        start_times[2] >= start_times[1] + durations[1],  # B before C
        start_times[0] >= start_times[2] + durations[2],  # C before A (cycle!)
    ]

    prob = cp.Problem(cp.Minimize(cp.max(start_times + durations)), constraints)

    print("\nCyclic precedence constraints:")
    print("  Task A (2) -> Task B (3) -> Task C (4) -> Task A")
    print("\nThis creates a cycle and is always infeasible.\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all scheduling examples."""
    example_task_scheduling()
    example_resource_constrained()
    example_machine_scheduling()
    example_shift_scheduling()
    example_cyclic_precedence()

    print("\n" + "=" * 60)
    print("Summary: Scheduling Issues")
    print("=" * 60)
    print("""
Common issues and fixes:

1. TIGHT DEADLINES
   - Critical path exceeds deadline
   - Fix: Extend deadline, add resources, or parallelize

2. RESOURCE CONFLICTS
   - Multiple tasks need same resource simultaneously
   - Fix: Add resources, sequence tasks, or allow preemption

3. COVERAGE REQUIREMENTS
   - Not enough capacity to meet all coverage needs
   - Fix: Hire more, reduce requirements, or use overtime

4. CYCLIC PRECEDENCE
   - Logical error in task dependencies
   - Fix: Review and remove cycles from dependency graph

5. MACHINE CONFLICTS
   - Jobs can't be sequenced to meet all constraints
   - Fix: Add machines, extend horizon, or relax deadlines

Use cvxpy_debug to identify which constraints form the conflict.
""")


if __name__ == "__main__":
    main()
