"""
Portfolio Optimization Example
==============================

This example demonstrates debugging common issues in Markowitz
mean-variance portfolio optimization problems.

Common issues:
- Infeasible target returns
- Unbounded positions (no shorting constraints)
- Numerical issues with correlation matrices
"""

import cvxpy as cp
import numpy as np

import cvxpy_debug


def example_infeasible_target_return():
    """Target return that's impossible to achieve."""
    print("=" * 60)
    print("Example 1: Infeasible Target Return")
    print("=" * 60)

    # Asset data
    np.random.seed(42)
    n_assets = 5
    expected_returns = np.array([0.05, 0.08, 0.12, 0.06, 0.10])  # 5-12% returns
    cov_matrix = np.array(
        [
            [0.04, 0.01, 0.02, 0.01, 0.01],
            [0.01, 0.09, 0.03, 0.02, 0.02],
            [0.02, 0.03, 0.16, 0.04, 0.03],
            [0.01, 0.02, 0.04, 0.05, 0.02],
            [0.01, 0.02, 0.03, 0.02, 0.08],
        ]
    )

    # Portfolio weights
    weights = cp.Variable(n_assets, name="weights")

    # Constraints
    target_return = 0.20  # 20% - impossible with max single asset return of 12%

    constraints = [
        cp.sum(weights) == 1,  # Fully invested
        weights >= 0,  # No short selling
        expected_returns @ weights >= target_return,  # Target return
    ]

    # Minimize variance
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)

    print("\nMarkowitz portfolio optimization:")
    print(f"  - {n_assets} assets with returns {expected_returns}")
    print(f"  - Target return: {target_return:.0%}")
    print(f"  - Max achievable: {max(expected_returns):.0%}")
    print("\nThis is infeasible: target exceeds best single asset return.\n")

    cvxpy_debug.debug(prob)

    print("\nTo fix: Lower target return to at most", f"{max(expected_returns):.0%}")


def example_unbounded_short_selling():
    """Unbounded positions when short selling is allowed."""
    print("\n" + "=" * 60)
    print("Example 2: Unbounded Short Selling")
    print("=" * 60)

    n_assets = 3
    expected_returns = np.array([0.05, 0.10, 0.08])

    # No short-selling constraints - weights can go to -infinity
    weights = cp.Variable(n_assets, name="weights")

    constraints = [
        cp.sum(weights) == 1,  # Fully invested
        # Missing: weights >= 0 or weights >= -limit
    ]

    # Maximize expected return (unbounded!)
    prob = cp.Problem(cp.Maximize(expected_returns @ weights), constraints)

    print("\nPortfolio maximizing return without position limits:")
    print("  - Short asset 0 (5% return) infinitely")
    print("  - Long asset 1 (10% return) infinitely")
    print("  - Constraint sum(w) == 1 is satisfied with w0 -> -inf, w1 -> +inf\n")

    cvxpy_debug.debug(prob)

    print("\nTo fix: Add position limits like weights >= -1 or weights >= 0")


def example_ill_conditioned_covariance():
    """Numerical issues with nearly singular covariance matrix."""
    print("\n" + "=" * 60)
    print("Example 3: Ill-Conditioned Covariance Matrix")
    print("=" * 60)

    n_assets = 10
    np.random.seed(42)

    # Create ill-conditioned covariance matrix
    # Highly correlated assets lead to near-singular matrix
    base = np.random.randn(n_assets, 3)  # Low rank factor
    cov_matrix = base @ base.T + 1e-6 * np.eye(n_assets)

    expected_returns = np.linspace(0.05, 0.15, n_assets)

    weights = cp.Variable(n_assets, name="weights")

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        expected_returns @ weights >= 0.10,
    ]

    portfolio_variance = cp.quad_form(weights, cov_matrix)
    prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)

    print("\nPortfolio with nearly-singular covariance matrix:")
    print("  - 10 assets with high correlation (low-rank factor model)")
    print(f"  - Condition number: ~{np.linalg.cond(cov_matrix):.0e}")
    print("\nThis may cause numerical issues in optimization.\n")

    prob.solve()
    cvxpy_debug.debug(prob, include_conditioning=True)


def example_efficient_frontier():
    """Building efficient frontier with debugging."""
    print("\n" + "=" * 60)
    print("Example 4: Efficient Frontier Construction")
    print("=" * 60)

    n_assets = 4
    expected_returns = np.array([0.06, 0.10, 0.14, 0.08])
    cov_matrix = np.array(
        [
            [0.04, 0.01, 0.02, 0.01],
            [0.01, 0.09, 0.04, 0.02],
            [0.02, 0.04, 0.20, 0.05],
            [0.01, 0.02, 0.05, 0.06],
        ]
    )

    weights = cp.Variable(n_assets, name="weights")
    target_return = cp.Parameter(name="target_return")

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        expected_returns @ weights >= target_return,
    ]

    portfolio_variance = cp.quad_form(weights, cov_matrix)
    prob = cp.Problem(cp.Minimize(portfolio_variance), constraints)

    print("\nTracing efficient frontier from 6% to 14% target return:")
    print("-" * 50)

    returns_range = np.linspace(0.06, 0.16, 6)  # Includes infeasible 15%, 16%

    for r in returns_range:
        target_return.value = r
        prob.solve()
        if prob.status == cp.OPTIMAL:
            print(f"  Target {r:.0%}: variance = {prob.value:.6f}")
        else:
            print(f"  Target {r:.0%}: {prob.status}")
            # Debug the infeasible case
            cvxpy_debug.debug(prob)
            break

    print("\nThe frontier ends when target return becomes infeasible.")


def example_robust_portfolio():
    """Portfolio with worst-case return constraint."""
    print("\n" + "=" * 60)
    print("Example 5: Robust Portfolio Optimization")
    print("=" * 60)

    n_assets = 4
    expected_returns = np.array([0.08, 0.10, 0.12, 0.09])
    return_uncertainty = np.array([0.02, 0.03, 0.05, 0.02])  # Uncertainty in returns

    weights = cp.Variable(n_assets, name="weights")

    # Robust constraint: worst-case return >= target
    # expected_return - uncertainty * |weight| >= target
    # For long-only: expected_return - uncertainty * weight >= target
    target_return = 0.11

    constraints = [
        cp.sum(weights) == 1,
        weights >= 0,
        # Worst-case return (subtracting uncertainty)
        (expected_returns - return_uncertainty) @ weights >= target_return,
    ]

    # Minimize variance
    cov_matrix = np.diag([0.04, 0.06, 0.10, 0.05])
    prob = cp.Problem(cp.Minimize(cp.quad_form(weights, cov_matrix)), constraints)

    print("\nRobust portfolio with uncertainty:")
    print(f"  - Expected returns: {expected_returns}")
    print(f"  - Uncertainty: +/- {return_uncertainty}")
    print(f"  - Worst-case returns: {expected_returns - return_uncertainty}")
    print(f"  - Target: {target_return:.0%}")
    print(f"\nMax worst-case return: {max(expected_returns - return_uncertainty):.0%}")

    if target_return > max(expected_returns - return_uncertainty):
        print("This is infeasible: target exceeds best worst-case return.\n")

    cvxpy_debug.debug(prob)


def main():
    """Run all portfolio optimization examples."""
    example_infeasible_target_return()
    example_unbounded_short_selling()
    example_ill_conditioned_covariance()
    example_efficient_frontier()
    example_robust_portfolio()

    print("\n" + "=" * 60)
    print("Summary: Portfolio Optimization Issues")
    print("=" * 60)
    print("""
Common issues and fixes:

1. INFEASIBLE TARGET RETURN
   - Target return exceeds best achievable
   - Fix: Lower target or allow leverage/short-selling

2. UNBOUNDED POSITIONS
   - No position limits with return maximization
   - Fix: Add weights >= 0 or position limits

3. ILL-CONDITIONED COVARIANCE
   - Highly correlated assets cause numerical issues
   - Fix: Regularize covariance, use factor models, or shrinkage

4. ROBUST CONSTRAINTS
   - Worst-case constraints may be too conservative
   - Fix: Reduce uncertainty bounds or lower targets

5. CONSTRAINT CONFLICTS
   - Sector/weight limits may conflict with return target
   - Fix: Use cvxpy_debug to identify conflicting constraints
""")


if __name__ == "__main__":
    main()
