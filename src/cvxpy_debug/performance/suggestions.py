"""Generate fix suggestions for performance anti-patterns."""

from __future__ import annotations

from cvxpy_debug.performance.dataclasses import (
    AntiPattern,
    MatrixStructure,
    ProblemMetrics,
)


def generate_suggestions(
    metrics: ProblemMetrics,
    matrix_structure: MatrixStructure | None,
    anti_patterns: list[AntiPattern],
) -> list[str]:
    """
    Generate actionable suggestions based on analysis.

    Parameters
    ----------
    metrics : ProblemMetrics
        Problem metrics.
    matrix_structure : MatrixStructure, optional
        Matrix structure analysis.
    anti_patterns : list[AntiPattern]
        Detected anti-patterns.

    Returns
    -------
    list[str]
        List of suggestion strings.
    """
    suggestions = []

    # Collect suggestions from anti-patterns
    for pattern in anti_patterns:
        if pattern.suggestion:
            suggestions.append(pattern.suggestion)

    # Add general suggestions based on metrics
    if metrics.num_constraints > 1000:
        suggestions.append(
            f"Large problem ({metrics.num_constraints} constraints). "
            "Consider using DPP (Disciplined Parameterized Programming) "
            "for repeated solves with different parameter values."
        )

    if metrics.num_scalar_constraints > 10000:
        suggestions.append(
            "For very large problems, consider commercial solvers "
            "(GUROBI, MOSEK) or specialized open-source solvers (HIGHS for LP/MIP)."
        )

    # Deduplicate while preserving order
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s not in seen:
            seen.add(s)
            unique_suggestions.append(s)

    return unique_suggestions


def generate_summary(
    metrics: ProblemMetrics,
    anti_patterns: list[AntiPattern],
) -> str:
    """
    Generate a one-line summary of performance issues.

    Parameters
    ----------
    metrics : ProblemMetrics
        Problem metrics.
    anti_patterns : list[AntiPattern]
        Detected anti-patterns.

    Returns
    -------
    str
        Summary string.
    """
    issues = []

    # Check for high-severity patterns
    high_severity = [p for p in anti_patterns if p.severity == "high"]
    medium_severity = [p for p in anti_patterns if p.severity == "medium"]

    if high_severity:
        issues.append(f"{len(high_severity)} high-severity anti-pattern(s)")

    if medium_severity:
        issues.append(f"{len(medium_severity)} medium-severity anti-pattern(s)")

    if metrics.constraint_variable_ratio > 10:
        issues.append(f"high constraint ratio ({metrics.constraint_variable_ratio:.1f}x)")

    if not issues:
        return "No major performance issues detected."

    return "Performance issues: " + "; ".join(issues)
