"""Main infeasibility diagnosis orchestration."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from cvxpy_debug.infeasibility.elastic import find_infeasibility_contributors
from cvxpy_debug.infeasibility.iis import find_minimal_iis
from cvxpy_debug.infeasibility.mapping import get_constraint_info
from cvxpy_debug.report.report import DebugReport


def debug_infeasibility(
    problem: cp.Problem,
    report: DebugReport,
    *,
    solver: Any | None = None,
    find_minimal_iis: bool = True,
) -> None:
    """
    Diagnose why a problem is infeasible.

    Modifies the report in-place with findings.

    Parameters
    ----------
    problem : cp.Problem
        The infeasible problem.
    report : DebugReport
        Report to add findings to.
    solver : optional
        Solver to use for diagnostic solves.
    find_minimal_iis : bool
        If True, refine to minimal IIS.
    """
    report.status = "infeasible"

    # Step 1: Find infeasibility contributors via elastic relaxation
    contributors, slack_values = find_infeasibility_contributors(problem, solver=solver)

    if not contributors:
        report.add_finding(
            "Could not identify specific infeasibility contributors. "
            "The problem may have numerical issues."
        )
        return

    # Step 2: Optionally refine to minimal IIS
    if find_minimal_iis and len(contributors) > 1:
        iis = find_minimal_iis_set(problem, contributors, solver=solver)
    else:
        iis = contributors

    # Step 3: Build report
    report.iis = iis
    report.slack_values = slack_values

    # Add constraint info
    constraint_info = []
    for constraint in iis:
        info = get_constraint_info(constraint)
        slack = slack_values.get(id(constraint), 0.0)
        constraint_info.append(
            {
                "constraint": constraint,
                "label": info["label"],
                "expression": info["expression"],
                "slack": slack,
            }
        )

    report.constraint_info = constraint_info

    # Generate findings
    n_total = len(problem.constraints)
    n_conflict = len(iis)
    report.add_finding(f"Problem has {n_total} constraints. Found {n_conflict} that conflict.")

    # Generate suggestions
    _generate_suggestions(report, constraint_info, slack_values)


def find_minimal_iis_set(
    problem: cp.Problem,
    candidates: list,
    *,
    solver: Any | None = None,
) -> list:
    """Wrapper to call IIS finder."""
    return find_minimal_iis(problem, candidates, solver=solver)


def _generate_suggestions(
    report: DebugReport,
    constraint_info: list,
    slack_values: dict,
) -> None:
    """Generate fix suggestions based on constraint analysis."""
    # Find constraints with non-zero slack (the ones that need relaxing)
    relaxation_needed = [info for info in constraint_info if info["slack"] > 1e-6]

    if relaxation_needed:
        for info in relaxation_needed:
            report.add_suggestion(f"Relax '{info['label']}' by {info['slack']:.4g}")
    else:
        # All slacks are zero - this can happen with bound conflicts
        report.add_suggestion("Review the constraints in the IIS - they cannot all be satisfied.")
