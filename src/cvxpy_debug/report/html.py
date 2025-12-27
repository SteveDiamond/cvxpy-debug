"""HTML report generation for cvxpy-debug."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, select_autoescape

if TYPE_CHECKING:
    from cvxpy_debug.report.report import DebugReport

# Template directory
TEMPLATE_DIR = Path(__file__).parent / "templates"


def _get_jinja_env() -> Environment:
    """Get configured Jinja2 environment."""
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_html(report: DebugReport) -> str:
    """
    Render a debug report as HTML.

    Parameters
    ----------
    report : DebugReport
        The debug report to render.

    Returns
    -------
    str
        HTML string representation of the report.
    """
    env = _get_jinja_env()
    template = env.get_template("report.html.jinja2")

    # Convert to dict for template rendering
    report_dict = report.to_dict()

    return template.render(report=report_dict)


def save_html(report: DebugReport, path: str | Path) -> None:
    """
    Save a debug report as an HTML file.

    Parameters
    ----------
    report : DebugReport
        The debug report to save.
    path : str | Path
        Path to the output HTML file.
    """
    html_content = render_html(report)
    Path(path).write_text(html_content, encoding="utf-8")
