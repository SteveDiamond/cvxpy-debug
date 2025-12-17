# cvxpy-debug

Diagnostic tools for CVXPY optimization problems.

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

# Solve returns infeasible
prob.solve()  # status: infeasible

# Debug it
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

EXPLANATION
───────────
The minimum values sum to 120, exceeding the budget of 100.

SUGGESTED FIXES
───────────────
• Increase budget to at least 120
• Reduce one of the minimum bounds
```

## Features

- **Infeasibility diagnosis**: Find which constraints conflict
- **Solver-agnostic**: Works with any CVXPY solver
- **Full cone support**: Linear, SOC, SDP constraints
- **Human-readable reports**: Clear explanations and fix suggestions

## License

Apache 2.0
