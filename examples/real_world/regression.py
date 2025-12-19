"""
Regression Example
==================

This example demonstrates debugging common issues in regression
and statistical estimation problems.

Common issues:
- Numerical scaling from diverse feature magnitudes
- Constraint conflicts in constrained regression
- Regularization causing infeasibility with constraints
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_badly_scaled_features():
    """Regression with features of very different scales."""
    print("=" * 60)
    print("Example 1: Badly Scaled Features")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100
    n_features = 5

    # Features with very different scales
    X = np.column_stack(
        [
            np.random.randn(n_samples) * 1e-6,  # Very small
            np.random.randn(n_samples) * 1,  # Normal
            np.random.randn(n_samples) * 1e3,  # Large
            np.random.randn(n_samples) * 1e6,  # Very large
            np.random.randn(n_samples) * 1e-3,  # Small
        ]
    )

    # True coefficients (all around 1)
    true_beta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    y = X @ true_beta + np.random.randn(n_samples) * 0.1

    # Regression
    beta = cp.Variable(n_features, name="beta")
    residuals = y - X @ beta

    prob = cp.Problem(cp.Minimize(cp.sum_squares(residuals)))

    print("\nRegression with features scaled from 1e-6 to 1e6")
    print(f"Feature scales: {X.std(axis=0)}")
    print("\nThis can cause numerical issues in the solver.\n")

    prob.solve()
    cvxpy_debug.debug(prob, include_conditioning=True)

    print("\n" + "-" * 40)
    print("Better: Standardized Features")
    print("-" * 40)

    # Standardize features
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_standardized = (X - X_mean) / X_std

    beta_std = cp.Variable(n_features, name="beta_std")
    residuals_std = y - X_standardized @ beta_std

    prob_std = cp.Problem(cp.Minimize(cp.sum_squares(residuals_std)))

    print("\nStandardized features (mean=0, std=1)")
    prob_std.solve()
    cvxpy_debug.debug(prob_std)


def example_constrained_regression():
    """Regression with sign or bound constraints."""
    print("\n" + "=" * 60)
    print("Example 2: Constrained Regression")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 50
    n_features = 4

    # Design matrix
    X = np.random.randn(n_samples, n_features)

    # True coefficients (some negative)
    true_beta = np.array([2.0, -1.5, 0.5, -0.8])
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    # Non-negative regression (but true coefficients are negative!)
    beta = cp.Variable(n_features, nonneg=True, name="beta")
    residuals = y - X @ beta

    # Additional constraint that conflicts
    constraints = [
        cp.sum(beta) >= 5,  # Sum must be at least 5
        beta <= 1,  # Each coefficient at most 1
    ]

    prob = cp.Problem(cp.Minimize(cp.sum_squares(residuals)), constraints)

    print("\nNon-negative regression with constraints:")
    print("  - beta >= 0 (nonneg)")
    print("  - sum(beta) >= 5")
    print("  - beta <= 1 (each)")
    print(f"\nWith {n_features} coefficients, max sum is {n_features}")
    print("But sum(beta) >= 5 requires 5. Infeasible if n_features < 5.\n")

    cvxpy_debug.debug(prob)


def example_ridge_with_equality():
    """Ridge regression with conflicting equality constraints."""
    print("\n" + "=" * 60)
    print("Example 3: Ridge Regression with Equality Constraints")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 30
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    beta = cp.Variable(n_features, name="beta")

    # Ridge regression with regularization
    lambda_reg = 1.0
    loss = cp.sum_squares(y - X @ beta) + lambda_reg * cp.sum_squares(beta)

    # Equality constraints that may conflict
    A_eq = np.random.randn(5, n_features)  # 5 equality constraints
    b_eq = np.random.randn(5) * 10  # May not be achievable with regularization

    constraints = [A_eq @ beta == b_eq]

    prob = cp.Problem(cp.Minimize(loss), constraints)

    print("\nRidge regression with 5 equality constraints")
    print("The regularization pulls beta toward 0")
    print("But equality constraints may require large beta values.\n")

    cvxpy_debug.debug(prob)


def example_lasso_infeasibility():
    """LASSO with constraints that conflict with sparsity."""
    print("\n" + "=" * 60)
    print("Example 4: LASSO with Conflicting Constraints")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 50
    n_features = 20

    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    beta = cp.Variable(n_features, name="beta")

    # LASSO: minimize ||y - X*beta||^2 + lambda * ||beta||_1
    lambda_reg = 10.0  # Strong regularization pushes toward sparsity

    # Constraint: all coefficients must be at least 0.5
    # This conflicts with LASSO's tendency to make coefficients exactly 0
    constraints = [
        beta >= 0.5,
        beta <= 2.0,
    ]

    loss = cp.sum_squares(y - X @ beta) + lambda_reg * cp.norm(beta, 1)
    prob = cp.Problem(cp.Minimize(loss), constraints)

    print("\nLASSO with minimum coefficient constraint:")
    print("  - L1 regularization (lambda=10) promotes sparsity")
    print("  - But beta >= 0.5 requires all coefficients to be nonzero")
    print("\nThese goals conflict - solution will be at constraint boundary.\n")

    prob.solve()
    cvxpy_debug.debug(prob)

    if beta.value is not None:
        print(f"\nSolution: {np.sum(np.abs(beta.value) < 0.6)} coefficients near lower bound")


def example_quantile_regression():
    """Quantile regression with crossing quantiles."""
    print("\n" + "=" * 60)
    print("Example 5: Quantile Regression")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 100

    # Simple linear relationship
    x = np.linspace(0, 10, n_samples)
    X = np.column_stack([np.ones(n_samples), x])
    y = 2 + 0.5 * x + np.random.randn(n_samples)

    # Fit multiple quantiles
    beta_25 = cp.Variable(2, name="beta_25")
    beta_75 = cp.Variable(2, name="beta_75")

    # Pinball loss for quantile regression
    tau_25, tau_75 = 0.25, 0.75

    residuals_25 = y - X @ beta_25
    residuals_75 = y - X @ beta_75

    loss_25 = cp.sum(cp.maximum(tau_25 * residuals_25, (tau_25 - 1) * residuals_25))
    loss_75 = cp.sum(cp.maximum(tau_75 * residuals_75, (tau_75 - 1) * residuals_75))

    # Non-crossing constraint (25th percentile <= 75th percentile for all x)
    # This means: beta_25[0] + beta_25[1]*x <= beta_75[0] + beta_75[1]*x for all x
    # Simplified: require beta_75[0] >= beta_25[0] and beta_75[1] >= beta_25[1]

    # But also add conflicting constraints
    constraints = [
        beta_25[1] >= 1.0,  # 25th quantile has steep slope
        beta_75[1] <= 0.3,  # 75th quantile has shallow slope
        # This causes crossing!
    ]

    prob = cp.Problem(cp.Minimize(loss_25 + loss_75), constraints)

    print("\nQuantile regression with crossing constraints:")
    print("  - 25th percentile slope >= 1.0")
    print("  - 75th percentile slope <= 0.3")
    print("\nThese constraints cause quantiles to cross (infeasible).\n")

    cvxpy_debug.debug(prob)


def example_robust_regression():
    """Huber regression with outlier constraints."""
    print("\n" + "=" * 60)
    print("Example 6: Robust Regression with Outlier Bounds")
    print("=" * 60)

    np.random.seed(42)
    n_samples = 50
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    true_beta = np.array([1.0, -0.5, 0.8])
    y = X @ true_beta + np.random.randn(n_samples) * 0.5

    # Add some outliers
    y[:5] += 10  # Large positive outliers

    beta = cp.Variable(n_features, name="beta")
    residuals = y - X @ beta

    # Huber loss (robust to outliers)
    M = 1.0  # Huber threshold
    loss = cp.sum(cp.huber(residuals, M))

    # Constraint: all residuals must be small (conflicts with outliers!)
    constraints = [
        cp.abs(residuals) <= 2.0,  # Max residual of 2
    ]

    prob = cp.Problem(cp.Minimize(loss), constraints)

    print("\nHuber regression with max residual constraint:")
    print("  - Data has outliers (y[:5] shifted by +10)")
    print("  - Constraint: |residual| <= 2 for all points")
    print("\nCannot fit outliers within residual bound - infeasible.\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all regression examples."""
    example_badly_scaled_features()
    example_constrained_regression()
    example_ridge_with_equality()
    example_lasso_infeasibility()
    example_quantile_regression()
    example_robust_regression()

    print("\n" + "=" * 60)
    print("Summary: Regression Issues")
    print("=" * 60)
    print("""
Common issues and fixes:

1. BADLY SCALED FEATURES
   - Features with different magnitudes cause numerical issues
   - Fix: Standardize features (mean=0, std=1)

2. SIGN/BOUND CONSTRAINTS
   - Non-negative constraints may be infeasible for the data
   - Fix: Check if constraints match the underlying relationship

3. REGULARIZATION + CONSTRAINTS
   - Regularization pulls toward 0, constraints may require large values
   - Fix: Reduce regularization or relax constraints

4. SPARSITY + MINIMUM VALUES
   - L1 regularization conflicts with minimum coefficient requirements
   - Fix: Use elastic net or relax minimum requirements

5. QUANTILE CROSSING
   - Constraints that cause quantile functions to cross
   - Fix: Add explicit non-crossing constraints

6. OUTLIERS + RESIDUAL BOUNDS
   - Cannot fit all points within residual constraints
   - Fix: Remove outliers or use soft constraints

Use cvxpy_debug to identify which data points or constraints cause issues.
""")


if __name__ == "__main__":
    main()
