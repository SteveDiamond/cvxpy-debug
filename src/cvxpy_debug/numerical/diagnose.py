"""Main orchestration for numerical diagnostics."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from cvxpy_debug.numerical.conditioning import analyze_conditioning
from cvxpy_debug.numerical.dataclasses import NumericalAnalysis
from cvxpy_debug.numerical.recommendations import generate_recommendations
from cvxpy_debug.numerical.scaling import analyze_scaling
from cvxpy_debug.numerical.solver_stats import analyze_solver_stats
from cvxpy_debug.numerical.violations import analyze_violations
from cvxpy_debug.report.report import DebugReport


def debug_numerical_issues(
    problem: cp.Problem,
    report: DebugReport,
    *,
    solver: Any | None = None,
    include_conditioning: bool = True,
) -> NumericalAnalysis:
    """
    Diagnose numerical/accuracy issues in a problem.

    Called for statuses: OPTIMAL_INACCURATE, INFEASIBLE_INACCURATE,
                         UNBOUNDED_INACCURATE

    Parameters
    ----------
    problem : cp.Problem
        The problem with inaccurate status.
    report : DebugReport
        Report to add findings to.
    solver : optional
        Solver used (for tolerance lookup).
    include_conditioning : bool
        If True, estimate condition number (can be slow for large problems).

    Returns
    -------
    NumericalAnalysis
        Complete numerical analysis results.
    """
    # Get solver name from stats if available
    solver_name = None
    if problem.solver_stats and problem.solver_stats.solver_name:
        solver_name = problem.solver_stats.solver_name
    elif solver:
        solver_name = str(solver)

    # Run scaling analysis
    scaling = analyze_scaling(problem)

    # Run violation analysis if problem has a solution
    violations = None
    if _has_solution(problem):
        violations = analyze_violations(problem, solver_name)

    # Run conditioning analysis if requested
    conditioning = None
    if include_conditioning:
        conditioning = analyze_conditioning(problem)

    # Run solver stats analysis
    solver_stats = None
    if problem.solver_stats:
        solver_stats = analyze_solver_stats(problem)

    # Generate recommendations
    recommendations = generate_recommendations(
        problem, scaling, violations, conditioning, solver_stats, solver_name
    )

    # Build summary
    summary = _build_summary(scaling, violations, conditioning, solver_stats)

    # Create analysis result
    analysis = NumericalAnalysis(
        status=problem.status,
        scaling=scaling,
        violations=violations,
        conditioning=conditioning,
        solver_stats=solver_stats,
        recommendations=recommendations,
        summary=summary,
    )

    # Populate report
    _populate_report(report, analysis)

    # Store analysis in report
    report.numerical_analysis = analysis

    return analysis


def _has_solution(problem: cp.Problem) -> bool:
    """Check if problem has variable values set."""
    for var in problem.variables():
        if var.value is not None:
            return True
    return False


def _build_summary(
    scaling,
    violations,
    conditioning,
    solver_stats,
) -> str:
    """Build a human-readable summary of issues found."""
    issues = []

    if scaling and scaling.badly_scaled:
        issues.append(f"Badly scaled (ratio: {scaling.overall_range_ratio:.2e})")

    if violations and violations.max_violation > 1e-4:
        issues.append(f"Constraint violations (max: {violations.max_violation:.2e})")

    if conditioning and conditioning.ill_conditioned:
        issues.append(f"Ill-conditioned (cond: {conditioning.condition_number:.2e})")

    if solver_stats and solver_stats.hit_iteration_limit:
        issues.append("Hit iteration limit")

    if not issues:
        return "No major numerical issues detected."

    return "; ".join(issues)


def _populate_report(report: DebugReport, analysis: NumericalAnalysis) -> None:
    """Populate DebugReport with analysis findings and suggestions."""
    # Set status
    status_map = {
        cp.OPTIMAL_INACCURATE: "optimal_inaccurate",
        cp.INFEASIBLE_INACCURATE: "infeasible_inaccurate",
        cp.UNBOUNDED_INACCURATE: "unbounded_inaccurate",
    }
    report.status = status_map.get(analysis.status, str(analysis.status))

    # Add summary finding
    report.add_finding(f"Status: {report.status}")
    if analysis.summary:
        report.add_finding(analysis.summary)

    # Add scaling findings
    if analysis.scaling:
        if analysis.scaling.badly_scaled:
            report.add_finding(
                f"Problem scaling ratio: {analysis.scaling.overall_range_ratio:.2e} "
                "(>1e6 indicates poor scaling)"
            )
        if analysis.scaling.very_small_coefficients:
            report.add_finding(
                f"Found {len(analysis.scaling.very_small_coefficients)} very small "
                "coefficients (<1e-8)"
            )
        if analysis.scaling.very_large_coefficients:
            report.add_finding(
                f"Found {len(analysis.scaling.very_large_coefficients)} very large "
                "coefficients (>1e8)"
            )

    # Add violation findings
    if analysis.violations and analysis.violations.violations:
        top_violations = sorted(
            [v for v in analysis.violations.violations if v.exceeds_tolerance],
            key=lambda v: v.violation_amount,
            reverse=True,
        )[:5]
        for v in top_violations:
            report.add_finding(f"Constraint '{v.label}' violation: {v.violation_amount:.4e}")

    # Add conditioning findings
    if analysis.conditioning and analysis.conditioning.estimated:
        if analysis.conditioning.ill_conditioned:
            report.add_finding(
                f"Problem is ill-conditioned (condition number: "
                f"{analysis.conditioning.condition_number:.2e})"
            )

    # Add solver stats findings
    if analysis.solver_stats:
        if analysis.solver_stats.hit_iteration_limit:
            report.add_finding(
                f"Solver {analysis.solver_stats.solver_name} may have hit "
                f"iteration limit ({analysis.solver_stats.iterations} iterations)"
            )
        for issue in analysis.solver_stats.convergence_issues:
            report.add_finding(issue)

    # Add suggestions from recommendations
    # First add installed solvers
    installed_recs = [r for r in analysis.recommendations if r.is_installed]
    for rec in installed_recs[:3]:
        if rec.solver_name == "RESCALE":
            report.add_suggestion(rec.reason)
        elif rec.parameter_adjustments:
            params = ", ".join(f"{k}={v}" for k, v in rec.parameter_adjustments.items())
            report.add_suggestion(f"Try solver={rec.solver_name} with {params}")
        else:
            report.add_suggestion(f"Try solver={rec.solver_name}: {rec.reason}")

    # Mention uninstalled solvers that could help
    uninstalled_recs = [
        r for r in analysis.recommendations if not r.is_installed and r.solver_name != "RESCALE"
    ][:2]
    if uninstalled_recs:
        names = ", ".join(r.solver_name for r in uninstalled_recs)
        report.add_suggestion(f"Consider installing: {names} (may provide better accuracy)")
