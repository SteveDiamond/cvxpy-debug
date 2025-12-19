"""Main unboundedness diagnosis orchestration."""

from __future__ import annotations

from typing import Any

import cvxpy as cp

from cvxpy_debug.report.report import DebugReport
from cvxpy_debug.unbounded.bounds import get_unbounded_variables
from cvxpy_debug.unbounded.ray import (
    UnboundedRay,
    analyze_objective_sensitivity,
    find_unbounded_ray,
)
from cvxpy_debug.unbounded.suggestions import (
    format_unbounded_variable_info,
    generate_suggestions,
)


def debug_unboundedness(
    problem: cp.Problem,
    report: DebugReport,
    *,
    solver: Any | None = None,
) -> None:
    """
    Diagnose why a problem is unbounded.

    Modifies the report in-place with findings about:
    - Which variables are unbounded
    - Direction of unboundedness
    - Suggestions for fixing

    Parameters
    ----------
    problem : cp.Problem
        The unbounded problem.
    report : DebugReport
        Report to add findings to.
    solver : optional
        Solver to use for diagnostic solves.
    """
    report.status = "unbounded"

    # Step 1: Find unbounded ray using bounded relaxation
    ray = find_unbounded_ray(problem, solver=solver)

    # Step 2: Analyze variable bounds
    unbounded_vars = get_unbounded_variables(problem)

    # Step 3: Store results in report
    report.unbounded_ray = ray

    # Build unbounded_variables list for report
    if ray and ray.variables:
        # Use ray analysis - more precise
        unbounded_info = []
        for var_key, direction in ray.variables.items():
            if isinstance(var_key, tuple):
                var, idx = var_key
                dir_str = "above" if direction > 0 else "below"
            else:
                var = var_key
                idx = None
                dir_str = "above" if direction > 0 else "below"

            unbounded_info.append(format_unbounded_variable_info(var, dir_str, idx))
        report.unbounded_variables = unbounded_info
    elif unbounded_vars:
        # Fall back to static bound analysis
        report.unbounded_variables = [
            format_unbounded_variable_info(var, direction) for var, direction in unbounded_vars
        ]
    else:
        report.unbounded_variables = []

    # Step 4: Generate findings
    _generate_findings(problem, report, ray, unbounded_vars)

    # Step 5: Generate suggestions
    suggestions = generate_suggestions(problem, ray, unbounded_vars)
    for suggestion in suggestions:
        report.add_suggestion(suggestion)


def _generate_findings(
    problem: cp.Problem,
    report: DebugReport,
    ray: UnboundedRay | None,
    unbounded_vars: list,
) -> None:
    """Generate diagnostic findings."""
    is_minimize = isinstance(problem.objective, cp.Minimize)

    # Main finding
    if ray and ray.variables:
        n_unbounded = len(ray.variables)
    else:
        n_unbounded = len(unbounded_vars)

    report.add_finding(
        f"Problem is UNBOUNDED. Found {n_unbounded} variable(s) without "
        f"effective bounds in the direction of optimization."
    )

    # Objective direction
    if ray:
        analysis = analyze_objective_sensitivity(problem, ray)
        if analysis["explanation"]:
            report.add_finding(analysis["explanation"])
    else:
        if is_minimize:
            report.add_finding("The objective can decrease without bound.")
        else:
            report.add_finding("The objective can increase without bound.")

    # Specific variable findings
    if ray and ray.variables:
        for var_key, direction in ray.variables.items():
            if isinstance(var_key, tuple):
                var, idx = var_key
                var_name = _get_var_name(var, idx)
            else:
                var = var_key
                var_name = _get_var_name(var)

            if direction > 0:
                report.add_finding(f"No constraint limits '{var_name}' from above.")
            else:
                report.add_finding(f"No constraint limits '{var_name}' from below.")


def _get_var_name(var: cp.Variable, idx: tuple | None = None) -> str:
    """Get human-readable variable name."""
    name = var.name() if var.name() else "x"

    if idx is not None:
        idx_str = ",".join(str(i) for i in idx)
        name = f"{name}[{idx_str}]"

    return name
