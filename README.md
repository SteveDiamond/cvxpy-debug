# cvxpy-debug

Diagnostic tools for CVXPY optimization problems. When your problem is infeasible, unbounded, or numerically inaccurate, cvxpy-debug tells you why and how to fix it.

## Installation

```bash
pip install cvxpy-debug
```

## Quick Start

```python
import cvxpy as cp
import cvxpy_debug

# Create an infeasible problem
x = cp.Variable(3, nonneg=True)
constraints = [
    cp.sum(x) <= 100,
    x[0] >= 50,
    x[1] >= 40,
    x[2] >= 30,  # Sum of minimums = 120 > 100
]
prob = cp.Problem(cp.Minimize(cp.sum(x)), constraints)

# Debug it - automatically solves and diagnoses
report = cvxpy_debug.debug(prob)
```

Output:
```
════════════════════════════════════════════════════════════════
                     INFEASIBILITY REPORT
════════════════════════════════════════════════════════════════

Problem has 4 constraints. Found 4 that conflict.

CONFLICTING CONSTRAINTS
───────────────────────
  Constraint              Slack needed
  ────────────────────    ─────────────
  sum(x) <= 100           20.0
  x[0] >= 50              0.0
  x[1] >= 40              0.0
  x[2] >= 30              0.0

SUGGESTED FIXES
───────────────
• Increase budget to at least 120
• Reduce one of the minimum bounds
```

## Features

- **Infeasibility diagnosis**: Find which constraints conflict using IIS (Irreducible Infeasible Subsystem)
- **Unboundedness diagnosis**: Identify which variables are unbounded and in which direction
- **Numerical issues**: Detect scaling problems, ill-conditioning, and constraint violations
- **Performance analysis**: Detect anti-patterns like loop-generated constraints
- **Full cone support**: Linear, SOC, SDP, and exponential cone constraints
- **Human-readable reports**: Clear explanations and actionable fix suggestions

## Examples

See the [`examples/`](examples/) folder for comprehensive usage examples:

### Quick Start Examples
- [`basic_infeasibility.py`](examples/quickstart/basic_infeasibility.py) - Diagnose conflicting constraints
- [`basic_unboundedness.py`](examples/quickstart/basic_unboundedness.py) - Diagnose unbounded objectives
- [`basic_numerical.py`](examples/quickstart/basic_numerical.py) - Diagnose scaling issues
- [`basic_performance.py`](examples/quickstart/basic_performance.py) - Detect performance anti-patterns

### Problem Type Examples
- [`infeasibility_diagnosis.py`](examples/problem_types/infeasibility_diagnosis.py) - Full IIS workflow
- [`unboundedness_diagnosis.py`](examples/problem_types/unboundedness_diagnosis.py) - Ray analysis and bounds
- [`numerical_issues.py`](examples/problem_types/numerical_issues.py) - Conditioning and violations
- [`cone_constraints.py`](examples/problem_types/cone_constraints.py) - SOC, PSD, ExpCone examples

### Real-World Scenarios
- [`portfolio_optimization.py`](examples/real_world/portfolio_optimization.py) - Markowitz mean-variance
- [`resource_allocation.py`](examples/real_world/resource_allocation.py) - Budget allocation
- [`scheduling.py`](examples/real_world/scheduling.py) - Task scheduling
- [`regression.py`](examples/real_world/regression.py) - Constrained regression

## API

### Main Function

```python
cvxpy_debug.debug(
    problem,
    solver=None,              # Override solver for diagnostic solves
    find_minimal_iis=False,   # Find minimal conflicting constraint set
    include_conditioning=False,  # Analyze condition numbers (slower)
    include_performance=True,    # Include performance analysis
)
```

### Focused Diagnostics

```python
# Infeasibility analysis
cvxpy_debug.debug_infeasibility(problem, report)

# Unboundedness analysis
cvxpy_debug.debug_unboundedness(problem, report)

# Numerical issues analysis
cvxpy_debug.debug_numerical_issues(problem, report)

# Performance analysis
cvxpy_debug.debug_performance(problem, report)
```

## License

Apache 2.0
