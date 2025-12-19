"""Constraint mapping utilities - map constraints to human-readable info."""

import cvxpy as cp


def get_constraint_info(constraint: cp.Constraint) -> dict:
    """
    Extract human-readable information about a constraint.

    Parameters
    ----------
    constraint : cp.Constraint
        The constraint to analyze.

    Returns
    -------
    info : dict
        Dictionary with keys:
        - label: User-provided label or auto-generated name
        - expression: String representation of the constraint
        - constraint_type: Type of constraint (equality, inequality, etc.)
    """
    # Get label (cvxpy >= 1.4 supports .id attribute, check for label method)
    label = _get_label(constraint)

    # Get string representation
    expression = str(constraint)

    # Determine constraint type
    constraint_type = _get_constraint_type(constraint)

    return {
        "label": label,
        "expression": expression,
        "constraint_type": constraint_type,
    }


def _get_label(constraint: cp.Constraint) -> str:
    """Get the label for a constraint."""
    # First, try to get a meaningful string representation
    expr_str = _constraint_to_string(constraint)

    if expr_str:
        if len(expr_str) > 50:
            expr_str = expr_str[:47] + "..."
        return expr_str

    # Fallback to id-based label
    if hasattr(constraint, "id"):
        return f"constraint_{constraint.id}"

    return str(constraint)[:40]


def _constraint_to_string(constraint: cp.Constraint) -> str:
    """Convert constraint to human-readable string."""
    try:
        if isinstance(constraint, cp.constraints.zero.Equality):
            lhs = constraint.args[0]
            rhs = constraint.args[1]
            return f"{_expr_to_string(lhs)} == {_expr_to_string(rhs)}"

        if isinstance(constraint, cp.constraints.nonpos.Inequality):
            lhs = constraint.args[0]
            rhs = constraint.args[1]
            return f"{_expr_to_string(lhs)} <= {_expr_to_string(rhs)}"

        if isinstance(constraint, cp.constraints.second_order.SOC):
            t = constraint.args[0]
            X = constraint.args[1]
            return f"norm({_expr_to_string(X)}) <= {_expr_to_string(t)}"

        if isinstance(constraint, cp.constraints.psd.PSD):
            X = constraint.args[0]
            return f"{_expr_to_string(X)} >> 0"

    except Exception:
        pass

    return ""


def _expr_to_string(expr) -> str:
    """Convert expression to concise string (symbolic form, not values)."""
    # Check if it's a Constant (has is_constant method)
    if hasattr(expr, "is_constant") and expr.is_constant():
        if hasattr(expr, "value") and expr.value is not None:
            val = expr.value
            if hasattr(val, "__float__"):
                return f"{float(val):.4g}"
            if hasattr(val, "item"):  # numpy scalar
                return f"{val.item():.4g}"
        return str(expr)

    # Handle specific expression types BEFORE checking name
    expr_type = type(expr).__name__

    if expr_type == "Sum":
        # Get the variable being summed
        if expr.args:
            inner = _expr_to_string(expr.args[0])
            return f"sum({inner})"

    if expr_type == "norm":
        if expr.args:
            inner = _expr_to_string(expr.args[0])
            return f"norm({inner})"

    # Check for Variable (has name attribute and is a Variable type)
    if expr_type == "Variable":
        if hasattr(expr, "name") and expr.name():
            return expr.name()

    # For expressions, show symbolic form
    s = str(expr)
    # Clean up common patterns
    s = s.replace(", None, False", "")
    s = s.replace("Expression", "expr")

    if len(s) > 30:
        s = s[:27] + "..."
    return s


def _get_constraint_type(constraint: cp.Constraint) -> str:
    """Determine the type of constraint."""
    if isinstance(constraint, cp.constraints.zero.Zero):
        return "equality"
    elif isinstance(constraint, cp.constraints.nonpos.NonPos):
        return "inequality_leq"
    elif isinstance(constraint, (cp.constraints.nonpos.NonNeg,)):
        return "inequality_geq"
    elif isinstance(constraint, cp.constraints.second_order.SOC):
        return "second_order_cone"
    elif isinstance(constraint, cp.constraints.psd.PSD):
        return "psd"
    elif isinstance(constraint, cp.constraints.exponential.ExpCone):
        return "exponential_cone"
    else:
        return "unknown"


def format_constraint_table(constraint_info: list) -> str:
    """
    Format constraint information as a table.

    Parameters
    ----------
    constraint_info : list
        List of constraint info dicts.

    Returns
    -------
    table : str
        Formatted table string.
    """
    if not constraint_info:
        return "No constraints."

    # Determine column widths
    label_width = max(len(info["label"]) for info in constraint_info)
    label_width = max(label_width, 10)  # Minimum width

    lines = []
    lines.append(f"  {'Constraint':<{label_width}}  Slack needed")
    lines.append(f"  {'─' * label_width}  ─────────────")

    for info in constraint_info:
        slack = info.get("slack", 0.0)
        if slack > 1e-6:
            slack_str = f"{slack:.4g}"
        else:
            slack_str = "0.0"
        lines.append(f"  {info['label']:<{label_width}}  {slack_str}")

    return "\n".join(lines)
