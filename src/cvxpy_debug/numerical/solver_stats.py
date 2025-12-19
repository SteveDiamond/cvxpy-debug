"""Solver statistics analysis for CVXPY problems."""

from __future__ import annotations

import cvxpy as cp

from cvxpy_debug.numerical.dataclasses import SolverStatsAnalysis

# Default maximum iterations for common solvers
SOLVER_DEFAULTS: dict[str, dict[str, int]] = {
    "ECOS": {"max_iters": 100},
    "SCS": {"max_iters": 5000},
    "OSQP": {"max_iter": 4000},
    "MOSEK": {"max_iters": 10000},
    "CLARABEL": {"max_iter": 200},
    "CVXOPT": {"maxiters": 100},
    "GUROBI": {"IterationLimit": 1000000},
    "CPLEX": {"iterations_limit": 1000000},
    "HIGHS": {"iteration_limit": 1000000},
    "GLPK": {"it_lim": 10000},
}


def analyze_solver_stats(problem: cp.Problem) -> SolverStatsAnalysis | None:
    """
    Parse and analyze solver statistics for convergence issues.

    Parameters
    ----------
    problem : cp.Problem
        The solved CVXPY problem.

    Returns
    -------
    SolverStatsAnalysis | None
        Analysis results, or None if no stats available.
    """
    stats = problem.solver_stats
    if stats is None:
        return None

    solver_name = stats.solver_name if stats.solver_name else "UNKNOWN"
    convergence_issues: list[str] = []
    raw_stats: dict = {}

    # Extract basic stats
    iterations = None
    solve_time = None

    if hasattr(stats, "num_iters") and stats.num_iters is not None:
        iterations = int(stats.num_iters)
        raw_stats["num_iters"] = iterations

    if hasattr(stats, "solve_time") and stats.solve_time is not None:
        solve_time = float(stats.solve_time)
        raw_stats["solve_time"] = solve_time

    if hasattr(stats, "setup_time") and stats.setup_time is not None:
        raw_stats["setup_time"] = float(stats.setup_time)

    # Get extra stats if available
    if hasattr(stats, "extra_stats") and stats.extra_stats:
        raw_stats["extra_stats"] = stats.extra_stats

    # Get default max iterations for this solver
    solver_key = solver_name.upper()
    max_iterations = None
    if solver_key in SOLVER_DEFAULTS:
        defaults = SOLVER_DEFAULTS[solver_key]
        for key in [
            "max_iters",
            "max_iter",
            "maxiters",
            "IterationLimit",
            "iterations_limit",
            "iteration_limit",
            "it_lim",
        ]:
            if key in defaults:
                max_iterations = defaults[key]
                break

    # Compute iteration ratio
    iteration_ratio = None
    hit_iteration_limit = False
    if iterations is not None and max_iterations is not None:
        iteration_ratio = iterations / max_iterations
        # Consider hitting limit if >95% of max iterations
        hit_iteration_limit = iteration_ratio > 0.95
        if hit_iteration_limit:
            convergence_issues.append(
                f"Solver used {iterations}/{max_iterations} iterations "
                f"({iteration_ratio:.1%}) - may have hit iteration limit"
            )

    # Check for solver-specific convergence indicators
    _check_solver_specific_issues(solver_name, stats, convergence_issues)

    return SolverStatsAnalysis(
        solver_name=solver_name,
        iterations=iterations,
        max_iterations=max_iterations,
        iteration_ratio=iteration_ratio,
        hit_iteration_limit=hit_iteration_limit,
        solve_time=solve_time,
        convergence_issues=convergence_issues,
        raw_stats=raw_stats,
    )


def _check_solver_specific_issues(
    solver_name: str,
    stats: cp.problems.problem.SolverStats,
    issues: list[str],
) -> None:
    """
    Check for solver-specific convergence problems.

    Parameters
    ----------
    solver_name : str
        Name of the solver.
    stats : SolverStats
        The solver statistics object.
    issues : list[str]
        List to append issues to.
    """
    extra = getattr(stats, "extra_stats", None)
    if extra is None:
        return

    solver_key = solver_name.upper()

    # SCS-specific checks
    if solver_key == "SCS":
        if isinstance(extra, dict):
            if extra.get("res_pri", 0) > 1e-3:
                issues.append(f"SCS primal residual ({extra.get('res_pri'):.2e}) is high")
            if extra.get("res_dual", 0) > 1e-3:
                issues.append(f"SCS dual residual ({extra.get('res_dual'):.2e}) is high")

    # OSQP-specific checks
    elif solver_key == "OSQP":
        if hasattr(extra, "info"):
            info = extra.info
            if hasattr(info, "status") and "inaccurate" in str(info.status).lower():
                issues.append(f"OSQP reports inaccurate status: {info.status}")

    # ECOS-specific checks
    elif solver_key == "ECOS":
        if isinstance(extra, dict):
            exit_flag = extra.get("exitFlag", 0)
            if exit_flag == 10:  # ECOS_INACC_OFFSET
                issues.append("ECOS returned inaccurate result (exit flag 10)")
            elif exit_flag == -2:  # ECOS_NUMERICS
                issues.append("ECOS encountered numerical issues (exit flag -2)")
            elif exit_flag == -7:  # ECOS hit iteration limit
                issues.append("ECOS hit iteration limit (exit flag -7)")
