"""Solver recommendations for CVXPY problems."""

from __future__ import annotations

from typing import Any

import cvxpy as cp
from cvxpy.reductions.solvers import defines as slv_def

from cvxpy_debug.numerical.dataclasses import (
    ConditioningAnalysis,
    ScalingAnalysis,
    SolverRecommendation,
    SolverStatsAnalysis,
    ViolationAnalysis,
)

# Solvers to exclude from recommendations
EXCLUDED_SOLVERS = {"ECOS", "ECOS_BB"}


def generate_recommendations(
    problem: cp.Problem,
    scaling: ScalingAnalysis | None,
    violations: ViolationAnalysis | None,
    conditioning: ConditioningAnalysis | None,
    solver_stats: SolverStatsAnalysis | None,
    current_solver: str | None = None,
) -> list[SolverRecommendation]:
    """
    Generate prioritized solver recommendations based on analysis results.

    Uses CVXPY's own solver selection logic to find compatible solvers,
    then orders them according to CVXPY's priority ordering.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem.
    scaling : ScalingAnalysis | None
        Scaling analysis results.
    violations : ViolationAnalysis | None
        Violation analysis results.
    conditioning : ConditioningAnalysis | None
        Conditioning analysis results.
    solver_stats : SolverStatsAnalysis | None
        Solver statistics analysis.
    current_solver : str | None
        The solver that was used (to exclude from recommendations).

    Returns
    -------
    list[SolverRecommendation]
        List of solver recommendations in priority order.
    """
    recommendations: list[SolverRecommendation] = []

    # Get compatible solvers using CVXPY's own logic
    compatible_solvers = _get_compatible_solvers(problem, current_solver)

    # Add solver recommendations
    for priority, solver_name in enumerate(compatible_solvers):
        is_installed = solver_name in slv_def.INSTALLED_SOLVERS
        reason = _get_solver_reason(solver_name, problem)
        param_adjustments = _get_parameter_adjustments(solver_name, solver_stats, violations)

        recommendations.append(
            SolverRecommendation(
                solver_name=solver_name,
                reason=reason,
                is_installed=is_installed,
                parameter_adjustments=param_adjustments,
                priority=priority,
            )
        )

    # Add scaling recommendation if problem is badly scaled
    if scaling and scaling.badly_scaled:
        recommendations.insert(
            0,
            SolverRecommendation(
                solver_name="RESCALE",
                reason=f"Problem is badly scaled (ratio: {scaling.overall_range_ratio:.2e}). "
                "Consider rescaling variables or constraints.",
                is_installed=True,
                priority=-1,
            ),
        )

    return recommendations


def _get_compatible_solvers(
    problem: cp.Problem,
    current_solver: str | None = None,
) -> list[str]:
    """
    Get compatible solvers for the problem using CVXPY's logic.

    Returns solvers in CVXPY's priority order, excluding ECOS and the current solver.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem.
    current_solver : str | None
        The solver that was used (to exclude).

    Returns
    -------
    list[str]
        List of compatible solver names in priority order.
    """
    try:
        # Use CVXPY's internal method to find candidates
        candidates = problem._find_candidate_solvers(solver=None, gp=False)
    except Exception:
        # Fall back to all installed solvers
        candidates = {
            "qp_solvers": [s for s in slv_def.INSTALLED_SOLVERS if s in slv_def.QP_SOLVERS],
            "conic_solvers": [s for s in slv_def.INSTALLED_SOLVERS if s in slv_def.CONIC_SOLVERS],
        }

    # Sort according to CVXPY's priority ordering
    qp_solvers = sorted(
        candidates.get("qp_solvers", []),
        key=lambda s: slv_def.QP_SOLVERS.index(s) if s in slv_def.QP_SOLVERS else 999,
    )
    conic_solvers = sorted(
        candidates.get("conic_solvers", []),
        key=lambda s: (slv_def.CONIC_SOLVERS.index(s) if s in slv_def.CONIC_SOLVERS else 999),
    )

    # Determine which list to prefer based on problem type
    if _is_qp(problem):
        all_solvers = qp_solvers + [s for s in conic_solvers if s not in qp_solvers]
    else:
        all_solvers = conic_solvers + [s for s in qp_solvers if s not in conic_solvers]

    # Filter out excluded solvers and current solver
    filtered = []
    current_upper = current_solver.upper() if current_solver else None
    for solver in all_solvers:
        if solver in EXCLUDED_SOLVERS:
            continue
        if current_upper and solver.upper() == current_upper:
            continue
        if solver not in filtered:
            filtered.append(solver)

    return filtered


def _is_qp(problem: cp.Problem) -> bool:
    """Check if the problem is a QP (quadratic program)."""
    try:
        return problem.is_qp()
    except Exception:
        return False


def _get_solver_reason(solver_name: str, problem: cp.Problem) -> str:
    """Get a reason string for why a solver might help."""
    solver_upper = solver_name.upper()

    # Solver-specific reasons
    reasons = {
        "MOSEK": "Commercial solver with high numerical accuracy",
        "CLARABEL": "Modern interior-point solver, good for SDP/SOCP",
        "SCS": "First-order method, handles large-scale problems",
        "OSQP": "Specialized for QP, efficient for convex quadratic problems",
        "GUROBI": "Commercial solver, excellent for LP/QP/MIP",
        "CPLEX": "Commercial solver, robust numerical handling",
        "HIGHS": "Open-source LP/MIP solver with good performance",
        "CVXOPT": "Interior-point solver, good for small-medium problems",
        "GLPK": "Open-source simplex solver for LP",
        "GLPK_MI": "Open-source MIP solver",
        "SCIP": "Open-source solver for nonlinear MIP",
        "PROXQP": "Proximal QP solver, good numerical stability",
        "PIQP": "Proximal interior point QP solver",
        "DAQP": "Dual active-set QP solver",
    }

    return reasons.get(solver_upper, "Alternative solver option")


def _get_parameter_adjustments(
    solver_name: str,
    solver_stats: SolverStatsAnalysis | None,
    violations: ViolationAnalysis | None,
) -> dict[str, Any]:
    """
    Get recommended parameter adjustments for a solver.

    Parameters
    ----------
    solver_name : str
        The solver name.
    solver_stats : SolverStatsAnalysis | None
        Analysis of previous solver run.
    violations : ViolationAnalysis | None
        Violation analysis results.

    Returns
    -------
    dict[str, Any]
        Recommended parameter adjustments.
    """
    params: dict[str, Any] = {}
    solver_upper = solver_name.upper()

    # If previous solver hit iteration limit, suggest more iterations
    if solver_stats and solver_stats.hit_iteration_limit:
        iter_params = {
            "SCS": {"max_iters": 100000},
            "OSQP": {"max_iter": 100000},
            "CLARABEL": {"max_iter": 1000},
            "CVXOPT": {"maxiters": 500},
        }
        if solver_upper in iter_params:
            params.update(iter_params[solver_upper])

    # If there are significant violations, suggest tighter tolerances
    if violations and violations.max_violation > 1e-4:
        tol_params = {
            "SCS": {"eps_abs": 1e-8, "eps_rel": 1e-8},
            "OSQP": {"eps_abs": 1e-8, "eps_rel": 1e-8},
            "CLARABEL": {"tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9},
        }
        if solver_upper in tol_params:
            params.update(tol_params[solver_upper])

    return params
