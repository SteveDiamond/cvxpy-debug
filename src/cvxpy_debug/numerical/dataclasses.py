"""Data structures for numerical analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cvxpy as cp


@dataclass
class ScalingAnalysis:
    """Results from coefficient scaling analysis."""

    objective_range: tuple[float, float]  # (min_abs, max_abs) coefficients
    constraint_ranges: dict[int, tuple[float, float]]  # constraint_id -> range
    overall_range_ratio: float  # max/min across entire problem
    badly_scaled: bool  # True if ratio > threshold
    very_small_coefficients: list[dict] = field(default_factory=list)
    very_large_coefficients: list[dict] = field(default_factory=list)


@dataclass
class ConstraintViolation:
    """Information about a single constraint violation."""

    constraint: cp.Constraint
    constraint_id: int
    label: str
    violation_amount: float
    relative_violation: float  # violation / constraint_scale
    tolerance: float  # solver tolerance for comparison
    exceeds_tolerance: bool


@dataclass
class ViolationAnalysis:
    """Results from constraint violation analysis."""

    total_violations: int
    max_violation: float
    mean_violation: float
    violations: list[ConstraintViolation] = field(default_factory=list)
    solver_tolerances: dict[str, float] = field(default_factory=dict)


@dataclass
class ConditioningAnalysis:
    """Results from condition number analysis."""

    estimated: bool  # True if estimation was possible
    condition_number: float | None
    ill_conditioned: bool  # True if condition number > threshold
    problematic_constraints: list[dict] = field(default_factory=list)
    matrix_info: dict = field(default_factory=dict)


@dataclass
class SolverStatsAnalysis:
    """Results from solver statistics analysis."""

    solver_name: str
    iterations: int | None
    max_iterations: int | None
    iteration_ratio: float | None  # iterations / max_iterations
    hit_iteration_limit: bool
    solve_time: float | None
    convergence_issues: list[str] = field(default_factory=list)
    raw_stats: dict = field(default_factory=dict)


@dataclass
class SolverRecommendation:
    """A solver recommendation."""

    solver_name: str
    reason: str
    is_installed: bool
    parameter_adjustments: dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # lower = higher priority


@dataclass
class NumericalAnalysis:
    """Complete numerical analysis results."""

    status: str  # the problem status
    scaling: ScalingAnalysis | None = None
    violations: ViolationAnalysis | None = None
    conditioning: ConditioningAnalysis | None = None
    solver_stats: SolverStatsAnalysis | None = None
    recommendations: list[SolverRecommendation] = field(default_factory=list)
    summary: str = ""
