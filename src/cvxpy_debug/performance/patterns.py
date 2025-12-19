"""Anti-pattern detection for CVXPY problems."""

from __future__ import annotations

import cvxpy as cp

from cvxpy_debug.performance.dataclasses import (
    AntiPattern,
    AntiPatternType,
    MatrixStructure,
    ProblemMetrics,
)


def detect_anti_patterns(
    problem: cp.Problem,
    metrics: ProblemMetrics,
    matrix_structure: MatrixStructure | None,
) -> list[AntiPattern]:
    """
    Detect performance anti-patterns in a problem.

    Parameters
    ----------
    problem : cp.Problem
        The CVXPY problem to analyze.
    metrics : ProblemMetrics
        Pre-computed problem metrics.
    matrix_structure : MatrixStructure, optional
        Pre-computed matrix structure analysis.

    Returns
    -------
    list[AntiPattern]
        List of detected anti-patterns.
    """
    patterns = []

    # Detection 1: Loop-generated constraints
    loop_pattern = _detect_loop_constraints(problem, metrics, matrix_structure)
    if loop_pattern:
        patterns.append(loop_pattern)

    # Detection 2: Scalar constraints on vector variables
    scalar_patterns = _detect_scalar_on_vector(problem)
    patterns.extend(scalar_patterns)

    # Detection 3: Potentially redundant constraints
    redundant = _detect_redundant_constraints(matrix_structure)
    if redundant:
        patterns.append(redundant)

    # Detection 4: High constraint/variable ratio
    ratio_pattern = _detect_high_constraint_ratio(metrics)
    if ratio_pattern:
        patterns.append(ratio_pattern)

    return patterns


def _detect_loop_constraints(
    problem: cp.Problem,
    metrics: ProblemMetrics,
    matrix_structure: MatrixStructure | None,
) -> AntiPattern | None:
    """
    Detect constraints that appear to be generated in a loop.

    Indicators:
    - Many constraints with identical structure but different indices
    - Repeated row patterns in constraint matrix
    - High constraint count relative to variables
    """
    # Heuristic 1: Count constraints by type
    type_counts: dict[str, list[int]] = {}
    for i, constraint in enumerate(problem.constraints):
        ctype = type(constraint).__name__
        if ctype not in type_counts:
            type_counts[ctype] = []
        type_counts[ctype].append(i)

    # Look for types with many instances
    suspicious_types = []
    for ctype, indices in type_counts.items():
        if len(indices) > 10 and len(indices) > 0.5 * len(problem.constraints):
            suspicious_types.append((ctype, indices))

    # Heuristic 2: Check matrix structure for repeated patterns
    has_repeated = matrix_structure and matrix_structure.has_repeated_rows

    # Heuristic 3: High singleton row count AND many constraints suggests element-wise
    # Note: A single vectorized constraint like `x >= 0` on a 100-element vector
    # will create 100 singleton rows in the matrix, but that's CORRECT behavior.
    # We only flag this if there are also many constraint OBJECTS.
    many_singletons = False
    if matrix_structure and matrix_structure.row_pattern_counts:
        singleton_count = matrix_structure.row_pattern_counts.get("singleton", 0)
        # Only flag if we have many singleton rows AND many constraint objects
        if (
            singleton_count > 10
            and singleton_count > 0.5 * matrix_structure.num_rows
            and metrics.num_constraints > 10
        ):
            many_singletons = True

    if suspicious_types or has_repeated or many_singletons:
        affected = []
        description_parts = []

        if suspicious_types:
            ctype, indices = suspicious_types[0]
            affected = indices[:20]  # First 20 as sample
            description_parts.append(f"{len(indices)} {ctype} constraints (may be loop-generated)")

        if many_singletons and matrix_structure:
            singleton_count = matrix_structure.row_pattern_counts.get("singleton", 0)
            description_parts.append(f"{singleton_count} singleton constraints in matrix")

        if has_repeated and matrix_structure:
            total_repeated = sum(len(g) for g in matrix_structure.repeated_row_groups)
            description_parts.append(f"{total_repeated} repeated constraint rows")

        return AntiPattern(
            pattern_type=AntiPatternType.LOOP_GENERATED_CONSTRAINTS,
            description="; ".join(description_parts),
            severity="high",
            affected_constraints=affected,
            suggestion=(
                "Replace loop-generated constraints with vectorized operations. "
                "Instead of `for i in range(n): constraints.append(x[i] >= 0)`, "
                "use `constraints = [x >= 0]`"
            ),
            estimated_improvement="2-10x compilation speedup",
        )

    return None


def _detect_scalar_on_vector(problem: cp.Problem) -> list[AntiPattern]:
    """
    Detect scalar constraints applied element-wise to vector variables.

    Example anti-pattern:
        for i in range(n):
            constraints.append(x[i] >= lower[i])

    Should be:
        constraints.append(x >= lower)
    """
    patterns = []

    # Group constraints by their variables
    variable_constraints: dict[int, list[tuple[int, cp.Constraint]]] = {}

    for i, constraint in enumerate(problem.constraints):
        try:
            vars_in_constraint = constraint.variables()

            # Check if constraint only involves one variable
            if len(vars_in_constraint) == 1:
                var = vars_in_constraint[0]
                # Only consider vector variables
                if var.size > 1:
                    var_id = id(var)
                    if var_id not in variable_constraints:
                        variable_constraints[var_id] = []
                    variable_constraints[var_id].append((i, constraint))
        except Exception:
            continue

    # Check each variable for element-wise constraint patterns
    for var_id, constraint_list in variable_constraints.items():
        # Threshold: if we have many constraints on one vector variable
        if len(constraint_list) > 5:
            # Get the variable
            var = None
            for _, c in constraint_list:
                vars_in_c = c.variables()
                if vars_in_c:
                    var = vars_in_c[0]
                    break

            if var and len(constraint_list) >= var.size * 0.5:
                var_name = var.name() or "x"
                patterns.append(
                    AntiPattern(
                        pattern_type=AntiPatternType.SCALAR_ON_VECTOR,
                        description=(
                            f"Variable '{var_name}' (size {var.size}) has "
                            f"{len(constraint_list)} individual constraints "
                            "that may be vectorizable"
                        ),
                        severity="high",
                        affected_constraints=[idx for idx, _ in constraint_list[:20]],
                        suggestion=(
                            f"Replace element-wise constraints on '{var_name}' "
                            "with a single vectorized constraint. "
                            f"Instead of looping over {var_name}[i], "
                            f"constrain {var_name} directly."
                        ),
                        estimated_improvement=f"{len(constraint_list)}x constraint reduction",
                    )
                )

    return patterns


def _detect_redundant_constraints(
    matrix_structure: MatrixStructure | None,
) -> AntiPattern | None:
    """
    Detect potentially redundant constraints from identical matrix rows.
    """
    if not matrix_structure or not matrix_structure.has_repeated_rows:
        return None

    repeated_groups = matrix_structure.repeated_row_groups
    if not repeated_groups:
        return None

    total_redundant = sum(len(g) - 1 for g in repeated_groups)
    num_groups = len(repeated_groups)

    return AntiPattern(
        pattern_type=AntiPatternType.REDUNDANT_CONSTRAINTS,
        description=(
            f"Found {total_redundant} potentially redundant constraints "
            f"({num_groups} groups of identical constraint rows)"
        ),
        severity="medium",
        affected_constraints=[idx for group in repeated_groups for idx in group[1:]],
        suggestion="Remove duplicate constraints to reduce problem size.",
        estimated_improvement=f"Remove {total_redundant} constraints",
    )


def _detect_high_constraint_ratio(metrics: ProblemMetrics) -> AntiPattern | None:
    """
    Detect unusually high constraint-to-variable ratio.
    """
    ratio = metrics.constraint_variable_ratio

    if ratio > 20:
        return AntiPattern(
            pattern_type=AntiPatternType.HIGH_CONSTRAINT_RATIO,
            description=(
                f"Very high constraint ratio: {ratio:.1f} constraints per variable "
                f"({metrics.num_constraints} constraints, "
                f"{metrics.num_scalar_variables} scalar variables)"
            ),
            severity="medium" if ratio < 50 else "high",
            suggestion=(
                "Review constraints for redundancy or vectorization opportunities. "
                "A high ratio often indicates loop-generated constraints."
            ),
        )

    return None
