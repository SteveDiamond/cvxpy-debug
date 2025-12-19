"""
Comprehensive Performance Anti-Patterns Example
================================================

This example demonstrates detecting and fixing common performance
anti-patterns in CVXPY problems.
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_loop_constraints():
    """Loop-generated constraints vs vectorized operations."""
    print("=" * 60)
    print("Example 1: Loop-Generated Constraints")
    print("=" * 60)

    n = 50
    np.random.seed(42)
    a = np.random.randn(n)
    b = np.random.randn(n)

    # BAD: Loop-generated constraints
    x_loop = cp.Variable(n, name="x_loop")
    constraints_loop = [x_loop >= 0]
    for i in range(n):
        constraints_loop.append(a[i] * x_loop[i] <= b[i])

    prob_loop = cp.Problem(cp.Minimize(cp.sum(x_loop)), constraints_loop)

    print("\nBAD: Creating constraints in a Python loop")
    print(f"     {n} separate constraint objects created\n")

    prob_loop.solve()
    cvxpy_debug.debug(prob_loop)

    print("\n" + "-" * 40)
    print("GOOD: Vectorized Alternative")
    print("-" * 40)

    # GOOD: Vectorized constraint
    x_vec = cp.Variable(n, name="x_vec")
    constraints_vec = [
        x_vec >= 0,
        cp.multiply(a, x_vec) <= b,  # Single vectorized constraint
    ]

    prob_vec = cp.Problem(cp.Minimize(cp.sum(x_vec)), constraints_vec)

    print("\nGOOD: Single vectorized constraint")
    print("      cp.multiply(a, x) <= b\n")

    prob_vec.solve()
    cvxpy_debug.debug(prob_vec)


def example_scalar_on_vector():
    """Scalar operations on vector elements."""
    print("\n" + "=" * 60)
    print("Example 2: Scalar Operations on Vector Elements")
    print("=" * 60)

    n = 30
    np.random.seed(42)
    lower = np.random.randn(n)
    upper = lower + np.abs(np.random.randn(n))

    # BAD: Individual bounds on each element
    x_scalar = cp.Variable(n, name="x_scalar")
    constraints_scalar = []
    for i in range(n):
        constraints_scalar.append(x_scalar[i] >= lower[i])
        constraints_scalar.append(x_scalar[i] <= upper[i])

    prob_scalar = cp.Problem(cp.Minimize(cp.sum(x_scalar)), constraints_scalar)

    print("\nBAD: Individual bounds on each element")
    print(f"     {2*n} constraints for {n}-element vector\n")

    prob_scalar.solve()
    cvxpy_debug.debug(prob_scalar)

    print("\n" + "-" * 40)
    print("GOOD: Vectorized Bounds")
    print("-" * 40)

    # GOOD: Vectorized bounds
    x_vector = cp.Variable(n, name="x_vector")
    constraints_vector = [
        x_vector >= lower,  # Single constraint
        x_vector <= upper,  # Single constraint
    ]

    prob_vector = cp.Problem(cp.Minimize(cp.sum(x_vector)), constraints_vector)

    print("\nGOOD: Vectorized bounds")
    print("      x >= lower, x <= upper (2 constraints total)\n")

    prob_vector.solve()
    cvxpy_debug.debug(prob_vector)


def example_high_constraint_ratio():
    """High constraint-to-variable ratio indicating redundancy."""
    print("\n" + "=" * 60)
    print("Example 3: High Constraint-to-Variable Ratio")
    print("=" * 60)

    # Problem with many redundant constraints
    x = cp.Variable(5, name="x")

    constraints = [x >= 0, x <= 10]

    # Add many similar constraints that are mostly redundant
    for i in range(20):
        constraints.append(cp.sum(x) <= 50 + i)  # Only tightest matters

    prob_redundant = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

    print("\nProblem with 5 variables but 22+ constraints")
    print("Most constraints cp.sum(x) <= 50+i are redundant.\n")

    prob_redundant.solve()
    cvxpy_debug.debug(prob_redundant)

    print("\n" + "-" * 40)
    print("BETTER: Remove Redundant Constraints")
    print("-" * 40)

    constraints_clean = [
        x >= 0,
        x <= 10,
        cp.sum(x) <= 50,  # Only the tightest constraint
    ]

    prob_clean = cp.Problem(cp.Minimize(cp.sum(x)), constraints_clean)

    print("\nKeep only the binding constraint: cp.sum(x) <= 50\n")

    prob_clean.solve()
    cvxpy_debug.debug(prob_clean)


def example_matrix_constraints():
    """Matrix operations vs element-by-element."""
    print("\n" + "=" * 60)
    print("Example 4: Matrix Constraints")
    print("=" * 60)

    m, n = 10, 20
    np.random.seed(42)
    A = np.random.randn(m, n)
    b = np.random.randn(m)

    # BAD: Row-by-row constraints
    x_rows = cp.Variable(n, name="x_rows")
    constraints_rows = [x_rows >= 0]
    for i in range(m):
        constraints_rows.append(A[i, :] @ x_rows <= b[i])

    prob_rows = cp.Problem(cp.Minimize(cp.sum(x_rows)), constraints_rows)

    print("\nBAD: Creating constraints row-by-row")
    print(f"     {m} separate constraints for A @ x <= b\n")

    prob_rows.solve()
    cvxpy_debug.debug(prob_rows)

    print("\n" + "-" * 40)
    print("GOOD: Single Matrix Constraint")
    print("-" * 40)

    # GOOD: Single matrix constraint
    x_matrix = cp.Variable(n, name="x_matrix")
    constraints_matrix = [
        x_matrix >= 0,
        A @ x_matrix <= b,  # Single constraint
    ]

    prob_matrix = cp.Problem(cp.Minimize(cp.sum(x_matrix)), constraints_matrix)

    print("\nGOOD: Single matrix inequality A @ x <= b\n")

    prob_matrix.solve()
    cvxpy_debug.debug(prob_matrix)


def example_variable_attributes():
    """Using variable attributes vs explicit constraints."""
    print("\n" + "=" * 60)
    print("Example 5: Variable Attributes vs Constraints")
    print("=" * 60)

    n = 20

    # BAD: Explicit non-negativity constraints
    x_explicit = cp.Variable(n, name="x_explicit")
    constraints_explicit = [
        x_explicit >= 0,  # Explicit constraint
        cp.sum(x_explicit) == 1,
    ]

    prob_explicit = cp.Problem(cp.Minimize(cp.sum_squares(x_explicit)), constraints_explicit)

    print("\nUsing explicit constraint x >= 0")
    prob_explicit.solve()
    cvxpy_debug.debug(prob_explicit)

    print("\n" + "-" * 40)
    print("BETTER: Using nonneg=True Attribute")
    print("-" * 40)

    # BETTER: Using variable attribute
    x_attr = cp.Variable(n, nonneg=True, name="x_attr")
    constraints_attr = [cp.sum(x_attr) == 1]

    prob_attr = cp.Problem(cp.Minimize(cp.sum_squares(x_attr)), constraints_attr)

    print("\nUsing nonneg=True variable attribute")
    print("More efficient - solver knows about this structure.\n")

    prob_attr.solve()
    cvxpy_debug.debug(prob_attr)


def main():
    """Run all performance anti-pattern examples."""
    example_loop_constraints()
    example_scalar_on_vector()
    example_high_constraint_ratio()
    example_matrix_constraints()
    example_variable_attributes()

    print("\n" + "=" * 60)
    print("Summary: Performance Best Practices")
    print("=" * 60)
    print("""
1. VECTORIZE: Replace loops with vectorized operations
   BAD:  for i in range(n): constraints.append(a[i]*x[i] <= b[i])
   GOOD: constraints.append(cp.multiply(a, x) <= b)

2. USE MATRIX OPERATIONS: Single matrix constraint vs row-by-row
   BAD:  for i in range(m): constraints.append(A[i,:] @ x <= b[i])
   GOOD: constraints.append(A @ x <= b)

3. REMOVE REDUNDANT CONSTRAINTS: Keep only binding/necessary ones
   BAD:  [sum(x) <= 50, sum(x) <= 51, sum(x) <= 52, ...]
   GOOD: [sum(x) <= 50]

4. USE VARIABLE ATTRIBUTES: nonneg, nonpos, symmetric, PSD
   BAD:  x = cp.Variable(n); constraints = [x >= 0]
   GOOD: x = cp.Variable(n, nonneg=True)

5. PREFER BROADCASTING: Element-wise operations over loops
   BAD:  [x[i] >= lower[i] for i in range(n)]
   GOOD: [x >= lower]
""")


if __name__ == "__main__":
    main()
