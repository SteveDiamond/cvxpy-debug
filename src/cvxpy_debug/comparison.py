"""Multi-solver comparison for CVXPY problems."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp
import numpy as np


@dataclass
class SolverResult:
    """Result from solving a problem with a specific solver."""

    solver_name: str
    status: str
    objective_value: float | None
    solve_time: float
    error: str | None = None
    solution: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "solver_name": self.solver_name,
            "status": self.status,
            "objective_value": self.objective_value,
            "solve_time": self.solve_time,
            "error": self.error,
            "solution": {
                k: v.tolist() if hasattr(v, "tolist") else v for k, v in self.solution.items()
            },
        }


@dataclass
class SolverComparison:
    """Comparison of results across multiple solvers."""

    problem_info: dict[str, Any]
    results: list[SolverResult] = field(default_factory=list)
    reference_solver: str | None = None
    solution_differences: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "problem_info": self.problem_info,
            "results": [r.to_dict() for r in self.results],
            "reference_solver": self.reference_solver,
            "solution_differences": self.solution_differences,
        }

    def __str__(self) -> str:
        """Format comparison as a readable table."""
        lines = []
        width = 70

        lines.append("=" * width)
        lines.append("MULTI-SOLVER COMPARISON".center(width))
        lines.append("=" * width)
        lines.append("")

        # Problem info
        lines.append(f"Variables: {self.problem_info.get('num_variables', '?')}")
        lines.append(f"Constraints: {self.problem_info.get('num_constraints', '?')}")
        lines.append("")

        # Results table
        lines.append("RESULTS")
        lines.append("-" * 7)
        lines.append(f"  {'Solver':<15} {'Status':<15} {'Objective':>15} {'Time (s)':>12}")
        lines.append(f"  {'-' * 15} {'-' * 15} {'-' * 15} {'-' * 12}")

        for r in self.results:
            obj_str = f"{r.objective_value:.6g}" if r.objective_value is not None else "—"
            time_str = f"{r.solve_time:.4f}"
            status = r.status if not r.error else "ERROR"
            lines.append(f"  {r.solver_name:<15} {status:<15} {obj_str:>15} {time_str:>12}")

        lines.append("")

        # Solution differences (if we have a reference)
        if self.solution_differences:
            lines.append("SOLUTION DIFFERENCES (from reference)")
            lines.append("-" * 37)
            for solver, diff in self.solution_differences.items():
                lines.append(f"  {solver:<20} max diff: {diff:.2e}")
            lines.append("")

        # Summary
        optimal_count = sum(1 for r in self.results if r.status == "optimal")
        failed_count = sum(1 for r in self.results if r.error is not None)
        total = len(self.results)
        lines.append(f"Summary: {optimal_count}/{total} optimal, {failed_count} errored")

        return "\n".join(lines)


def _get_available_solvers(problem: cp.Problem) -> list[str]:
    """Get list of solvers that can solve this problem and are installed."""
    from cvxpy.reductions.solvers.defines import INSTALLED_SOLVERS

    # Get solvers that support this problem type
    candidates = []
    for solver_name in INSTALLED_SOLVERS:
        try:
            solver = cp.settings.SOLVER_MAP_CONIC.get(solver_name) or cp.settings.SOLVER_MAP_QP.get(
                solver_name
            )
            if solver is not None:
                # Check if solver supports the problem
                if hasattr(solver, "accepts"):
                    if solver.accepts(problem):
                        candidates.append(solver_name)
                else:
                    candidates.append(solver_name)
        except Exception:
            continue

    return candidates


def _extract_solution(problem: cp.Problem) -> dict[str, Any]:
    """Extract variable values from a solved problem."""
    solution = {}
    for var in problem.variables():
        if var.value is not None:
            name = var.name() if var.name() else f"var_{var.id}"
            solution[name] = np.array(var.value)
    return solution


def _compute_solution_diff(sol1: dict[str, Any], sol2: dict[str, Any]) -> float:
    """Compute maximum absolute difference between two solutions."""
    max_diff = 0.0
    for name in sol1:
        if name in sol2:
            v1 = np.array(sol1[name])
            v2 = np.array(sol2[name])
            if v1.shape == v2.shape:
                diff = np.max(np.abs(v1 - v2))
                max_diff = max(max_diff, diff)
    return max_diff


def compare_solvers(
    problem: cp.Problem,
    solvers: list[str] | None = None,
    verbose: bool = False,
) -> SolverComparison:
    """
    Compare multiple solvers on the same problem.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to solve.
    solvers : list[str] | None, optional
        List of solvers to try. If None, uses all available compatible solvers.
    verbose : bool, optional
        If True, print progress. Default is False.

    Returns
    -------
    SolverComparison
        Comparison results including status, objective, timing, and solution differences.
    """
    # Get available solvers
    if solvers is None:
        solvers = _get_available_solvers(problem)
        if not solvers:
            # Fallback to commonly available solvers
            solvers = ["ECOS", "SCS", "OSQP"]

    # Problem info
    problem_info = {
        "num_variables": len(problem.variables()),
        "num_constraints": len(problem.constraints),
        "is_dcp": problem.is_dcp(),
    }

    comparison = SolverComparison(problem_info=problem_info)
    results = []
    reference_solution = None

    for solver_name in solvers:
        if verbose:
            print(f"Trying {solver_name}...", end=" ", flush=True)

        result = SolverResult(
            solver_name=solver_name,
            status="unknown",
            objective_value=None,
            solve_time=0.0,
        )

        try:
            # Make a copy of the problem to avoid interference
            # (Can't easily copy, so we just solve in place and restore)
            start_time = time.time()
            problem.solve(solver=solver_name, verbose=False)
            result.solve_time = time.time() - start_time

            result.status = problem.status or "unknown"
            result.objective_value = float(problem.value) if problem.value is not None else None

            if problem.status == cp.OPTIMAL:
                result.solution = _extract_solution(problem)
                if reference_solution is None:
                    reference_solution = result.solution
                    comparison.reference_solver = solver_name

            if verbose:
                print(f"{result.status} ({result.solve_time:.3f}s)")

        except Exception as e:
            result.error = str(e)
            result.status = "error"
            if verbose:
                print(f"ERROR: {e}")

        results.append(result)

    comparison.results = results

    # Compute solution differences from reference
    if reference_solution:
        for r in results:
            if r.solver_name != comparison.reference_solver and r.solution:
                diff = _compute_solution_diff(reference_solution, r.solution)
                comparison.solution_differences[r.solver_name] = diff

    return comparison
