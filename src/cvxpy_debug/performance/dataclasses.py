"""Data structures for performance analysis results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AntiPatternType(Enum):
    """Types of performance anti-patterns."""

    LOOP_GENERATED_CONSTRAINTS = "loop_generated_constraints"
    SCALAR_ON_VECTOR = "scalar_on_vector"
    REDUNDANT_CONSTRAINTS = "redundant_constraints"
    HIGH_CONSTRAINT_RATIO = "high_constraint_ratio"


@dataclass
class AntiPattern:
    """A detected performance anti-pattern."""

    pattern_type: AntiPatternType
    description: str
    severity: str  # "high", "medium", "low"
    affected_constraints: list[int] = field(default_factory=list)
    suggestion: str = ""
    estimated_improvement: str = ""


@dataclass
class ProblemMetrics:
    """Basic problem size and structure metrics."""

    num_variables: int
    num_scalar_variables: int
    num_constraints: int
    num_scalar_constraints: int
    constraint_variable_ratio: float


@dataclass
class MatrixStructure:
    """Constraint matrix structural analysis."""

    sparsity: float
    num_rows: int
    num_cols: int
    num_nonzeros: int
    has_repeated_rows: bool
    repeated_row_groups: list[list[int]] = field(default_factory=list)
    row_pattern_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class PerformanceAnalysis:
    """Complete performance analysis results."""

    metrics: ProblemMetrics
    matrix_structure: MatrixStructure | None = None
    anti_patterns: list[AntiPattern] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    summary: str = ""
