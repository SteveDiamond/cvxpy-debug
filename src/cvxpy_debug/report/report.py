"""Debug report class and formatting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
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

    def to_dict(self) -> dict[str, Any]:
        """
        Convert report to a JSON-serializable dictionary.

        Returns
        -------
        dict
            Dictionary representation of the report suitable for JSON serialization.
            The 'problem' field is omitted as cp.Problem is not serializable.
        """
        return {
            "status": self.status,
            "iis": [_serialize_constraint(c) for c in self.iis],
            "slack_values": {_serialize_constraint(k): v for k, v in self.slack_values.items()}
            if self.slack_values
            else {},
            "constraint_info": self.constraint_info,  # Already dicts
            "findings": self.findings,
            "suggestions": self.suggestions,
            "unbounded_variables": self.unbounded_variables,  # Already dicts
            "unbounded_ray": _serialize_value(self.unbounded_ray),
            "numerical_analysis": _serialize_dataclass(self.numerical_analysis),
            "performance_analysis": _serialize_dataclass(self.performance_analysis),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """
        Convert report to a JSON string.

        Parameters
        ----------
        indent : int | None, optional
            Indentation level for pretty-printing. Default is 2.
            Use None for compact output.

        Returns
        -------
        str
            JSON string representation of the report.
        """
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def to_html(self, file: str | None = None) -> str:
        """
        Convert report to an HTML string.

        Parameters
        ----------
        file : str | None, optional
            If provided, save the HTML to this file path.

        Returns
        -------
        str
            HTML string representation of the report.
        """
        from cvxpy_debug.report.html import render_html, save_html

        html_content = render_html(self)
        if file is not None:
            save_html(self, file)
        return html_content

    def _repr_html_(self) -> str:
        """
        HTML representation for Jupyter notebooks.

        This method is automatically called by Jupyter when displaying
        a DebugReport object in a cell.

        Returns
        -------
        str
            HTML string for Jupyter display.
        """
        return self.to_html()


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


# --- Serialization helpers ---


def _serialize_constraint(constraint: Any) -> str:
    """Convert a CVXPY constraint to a string representation."""
    if constraint is None:
        return ""
    if isinstance(constraint, str):
        return constraint
    # For CVXPY constraints, use their string representation
    try:
        return str(constraint)
    except Exception:
        return repr(constraint)


def _serialize_value(value: Any) -> Any:
    """Convert a value to a JSON-serializable form."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    # Handle numpy arrays
    if hasattr(value, "tolist"):
        return value.tolist()
    # For CVXPY objects or other non-serializable types
    return str(value)


def _serialize_dataclass(obj: Any) -> dict[str, Any] | None:
    """
    Recursively convert a dataclass to a JSON-serializable dictionary.

    Handles nested dataclasses, enums, numpy arrays, and CVXPY objects.
    """
    if obj is None:
        return None

    if not is_dataclass(obj) or isinstance(obj, type):
        return _serialize_value(obj)

    result = {}
    for field_name, field_value in asdict(obj).items():
        # Skip constraint objects in nested dataclasses (not serializable)
        if field_name == "constraint" and hasattr(field_value, "expr"):
            result[field_name] = str(field_value)
            continue

        if is_dataclass(field_value) and not isinstance(field_value, type):
            result[field_name] = _serialize_dataclass(field_value)
        elif isinstance(field_value, list):
            result[field_name] = [
                _serialize_dataclass(item) if is_dataclass(item) else _serialize_value(item)
                for item in field_value
            ]
        elif isinstance(field_value, dict):
            result[field_name] = {
                str(k): _serialize_dataclass(v) if is_dataclass(v) else _serialize_value(v)
                for k, v in field_value.items()
            }
        elif isinstance(field_value, Enum):
            result[field_name] = field_value.value
        else:
            result[field_name] = _serialize_value(field_value)

    return result
