"""Debug report class and formatting."""

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp

from cvxpy_debug.infeasibility.mapping import format_constraint_table


@dataclass
class DebugReport:
    """
    Diagnostic report for a CVXPY problem.

    Attributes
    ----------
    problem : cp.Problem
        The problem that was debugged.
    status : str
        Problem status (infeasible, unbounded, optimal, etc.).
    iis : list
        Irreducible infeasible subsystem (if infeasible).
    slack_values : dict
        Mapping from constraint id to slack value.
    constraint_info : list
        Detailed constraint information.
    findings : list
        List of diagnostic findings.
    suggestions : list
        List of fix suggestions.
    unbounded_variables : list
        Variables that are unbounded (if unbounded).
    unbounded_ray : Any
        Direction of unboundedness (if unbounded).
    numerical_analysis : Any
        Numerical analysis results (if inaccurate status).
    performance_analysis : Any
        Performance analysis results.
    """

    problem: cp.Problem
    status: str = ""
    iis: list = field(default_factory=list)
    slack_values: dict = field(default_factory=dict)
    constraint_info: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)
    unbounded_variables: list = field(default_factory=list)
    unbounded_ray: Any = None
    numerical_analysis: Any = None
    performance_analysis: Any = None

    def add_finding(self, finding: str) -> None:
        """Add a diagnostic finding."""
        self.findings.append(finding)

    def add_suggestion(self, suggestion: str) -> None:
        """Add a fix suggestion."""
        self.suggestions.append(suggestion)

    def __str__(self) -> str:
        """Format report as string for terminal output."""
        return format_report(self)


def format_report(report: DebugReport) -> str:
    """
    Format a debug report for terminal display.

    Parameters
    ----------
    report : DebugReport
        The report to format.

    Returns
    -------
    str
        Formatted report string.
    """
    lines = []

    # Header
    width = 64
    lines.append("═" * width)
    title = _get_report_title(report)
    lines.append(title.center(width))
    lines.append("═" * width)
    lines.append("")

    # Findings
    if report.findings:
        for finding in report.findings:
            lines.append(finding)
        lines.append("")

    # Conflicting constraints (for infeasibility)
    if report.constraint_info:
        lines.append("CONFLICTING CONSTRAINTS")
        lines.append("─" * 23)
        lines.append(format_constraint_table(report.constraint_info))
        lines.append("")

    # Unbounded variables (for unboundedness)
    if report.unbounded_variables:
        lines.append("UNBOUNDED VARIABLES")
        lines.append("─" * 19)
        lines.append(_format_unbounded_table(report.unbounded_variables))
        lines.append("")

    # Performance analysis
    if report.performance_analysis and report.performance_analysis.anti_patterns:
        lines.append("PERFORMANCE ANALYSIS")
        lines.append("─" * 20)
        for pattern in report.performance_analysis.anti_patterns:
            severity_marker = {"high": "!!", "medium": "!", "low": ""}
            marker = severity_marker.get(pattern.severity, "")
            lines.append(f"  {marker}[{pattern.severity.upper()}] {pattern.description}")
        if report.performance_analysis.summary:
            lines.append(f"  Summary: {report.performance_analysis.summary}")
        lines.append("")

    # Suggestions
    if report.suggestions:
        lines.append("SUGGESTED FIXES")
        lines.append("─" * 15)
        for suggestion in report.suggestions:
            lines.append(f"• {suggestion}")
        lines.append("")

    return "\n".join(lines)


def _get_report_title(report: DebugReport) -> str:
    """Get appropriate title based on report status."""
    if report.status == "infeasible":
        return "INFEASIBILITY REPORT"
    elif report.status == "unbounded":
        return "UNBOUNDEDNESS REPORT"
    elif report.status in ("optimal_inaccurate", "infeasible_inaccurate", "unbounded_inaccurate"):
        return "NUMERICAL ACCURACY REPORT"
    else:
        return "DEBUG REPORT"


def _format_unbounded_table(unbounded_variables: list) -> str:
    """Format table of unbounded variables."""
    if not unbounded_variables:
        return "  (none)"

    lines = []
    lines.append("  Variable            Direction")
    lines.append("  ─────────────────   ─────────")

    for info in unbounded_variables:
        name = info.get("name", "?")
        direction_sym = info.get("direction_symbol", "?")
        lines.append(f"  {name:<18}  {direction_sym}")

    return "\n".join(lines)
