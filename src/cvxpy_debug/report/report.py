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
    """

    problem: cp.Problem
    status: str = ""
    iis: list = field(default_factory=list)
    slack_values: dict = field(default_factory=dict)
    constraint_info: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    suggestions: list = field(default_factory=list)

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
    title = "INFEASIBILITY REPORT" if report.status == "infeasible" else "DEBUG REPORT"
    lines.append(title.center(width))
    lines.append("═" * width)
    lines.append("")

    # Findings
    if report.findings:
        for finding in report.findings:
            lines.append(finding)
        lines.append("")

    # Conflicting constraints
    if report.constraint_info:
        lines.append("CONFLICTING CONSTRAINTS")
        lines.append("─" * 23)
        lines.append(format_constraint_table(report.constraint_info))
        lines.append("")

    # Suggestions
    if report.suggestions:
        lines.append("SUGGESTED FIXES")
        lines.append("─" * 15)
        for suggestion in report.suggestions:
            lines.append(f"• {suggestion}")
        lines.append("")

    return "\n".join(lines)
