"""Main orchestration for performance diagnostics."""

import cvxpy as cp

from cvxpy_debug.performance.dataclasses import (
    PerformanceAnalysis,
    ProblemMetrics,
)
from cvxpy_debug.performance.matrix_analysis import analyze_matrix_structure
from cvxpy_debug.performance.patterns import detect_anti_patterns
from cvxpy_debug.performance.suggestions import generate_suggestions, generate_summary
from cvxpy_debug.report.report import DebugReport


def debug_performance(
    problem: cp.Problem,
    report: DebugReport,
    *,
    include_matrix_analysis: bool = True,
) -> PerformanceAnalysis:
    """
    Diagnose performance issues in a CVXPY problem.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to analyze.
    report : DebugReport
        Report to add findings to.
    include_matrix_analysis : bool
        If True, analyze constraint matrix structure.
        Can be slow for large problems.

    Returns
    -------
    PerformanceAnalysis
        Complete performance analysis results.
    """
    # Step 1: Compute basic metrics
    metrics = _compute_metrics(problem)

    # Step 2: Analyze matrix structure if requested
    matrix_structure = None
    if include_matrix_analysis:
        matrix_structure = analyze_matrix_structure(problem)

    # Step 3: Detect anti-patterns
    anti_patterns = detect_anti_patterns(problem, metrics, matrix_structure)

    # Step 4: Generate suggestions
    suggestions = generate_suggestions(metrics, matrix_structure, anti_patterns)

    # Step 5: Build summary
    summary = generate_summary(metrics, anti_patterns)

    # Create analysis result
    analysis = PerformanceAnalysis(
        metrics=metrics,
        matrix_structure=matrix_structure,
        anti_patterns=anti_patterns,
        suggestions=suggestions,
        summary=summary,
    )

    # Populate report
    _populate_report(report, analysis)

    # Store analysis in report
    report.performance_analysis = analysis

    return analysis


def _compute_metrics(problem: cp.Problem) -> ProblemMetrics:
    """Compute basic problem metrics."""
    variables = problem.variables()
    num_variables = len(variables)
    num_scalar_variables = sum(v.size for v in variables)

    constraints = problem.constraints
    num_constraints = len(constraints)
    num_scalar_constraints = sum(_constraint_size(c) for c in constraints)

    ratio = num_scalar_constraints / max(num_scalar_variables, 1)

    return ProblemMetrics(
        num_variables=num_variables,
        num_scalar_variables=num_scalar_variables,
        num_constraints=num_constraints,
        num_scalar_constraints=num_scalar_constraints,
        constraint_variable_ratio=ratio,
    )


def _constraint_size(constraint: cp.Constraint) -> int:
    """Get the number of scalar constraints in a constraint."""
    if hasattr(constraint, "size"):
        return constraint.size
    return 1


def _populate_report(report: DebugReport, analysis: PerformanceAnalysis) -> None:
    """Populate DebugReport with performance findings."""
    # Add metrics finding
    report.add_finding(
        f"Problem structure: {analysis.metrics.num_variables} variable(s) "
        f"({analysis.metrics.num_scalar_variables} scalar), "
        f"{analysis.metrics.num_constraints} constraint(s) "
        f"({analysis.metrics.num_scalar_constraints} scalar)"
    )

    if analysis.metrics.constraint_variable_ratio > 10:
        report.add_finding(
            f"High constraint/variable ratio: {analysis.metrics.constraint_variable_ratio:.1f}x "
            "(may indicate loop-generated constraints)"
        )

    # Add anti-pattern findings
    for pattern in analysis.anti_patterns:
        if pattern.severity in ("high", "medium"):
            report.add_finding(
                f"[{pattern.severity.upper()}] {pattern.pattern_type.value}: "
                f"{pattern.description}"
            )

    # Add suggestions (limit to top 5 to avoid clutter)
    for suggestion in analysis.suggestions[:5]:
        report.add_suggestion(suggestion)
