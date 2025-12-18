# cvxpy-debug Examples

This directory contains examples demonstrating how to use cvxpy-debug to diagnose and fix optimization problems.

## Prerequisites

```bash
pip install cvxpy cvxpy-debug
```

## Directory Structure

### Quickstart Examples (`quickstart/`)
Minimal examples to get started quickly:
- `basic_infeasibility.py` - Diagnose conflicting constraints
- `basic_unboundedness.py` - Diagnose unbounded objectives
- `basic_numerical.py` - Diagnose numerical/scaling issues
- `basic_performance.py` - Detect performance anti-patterns

### Problem Type Examples (`problem_types/`)
Comprehensive examples for each diagnostic type:
- `infeasibility_diagnosis.py` - Full infeasibility workflow with IIS
- `unboundedness_diagnosis.py` - Ray analysis and bound suggestions
- `numerical_issues.py` - Scaling, conditioning, and solver recommendations
- `performance_antipatterns.py` - Detecting and fixing anti-patterns
- `cone_constraints.py` - SOC, PSD, and exponential cone examples

### Real-World Scenarios (`real_world/`)
Practical examples based on common optimization applications:
- `portfolio_optimization.py` - Markowitz mean-variance optimization
- `resource_allocation.py` - Budget and resource allocation
- `scheduling.py` - Task scheduling with constraints
- `regression.py` - Robust regression with regularization

## Running Examples

Each example is self-contained and can be run directly:

```bash
python examples/quickstart/basic_infeasibility.py
```

## Common Workflow

1. Create your CVXPY problem
2. Call `cvxpy_debug.debug(problem)` to get a diagnostic report
3. Review the findings and suggestions
4. Apply the suggested fixes

```python
import cvxpy as cp
import cvxpy_debug

# Create problem
x = cp.Variable(nonneg=True)
prob = cp.Problem(cp.Minimize(x), [x >= 10, x <= 5])

# Debug - automatically solves and diagnoses
report = cvxpy_debug.debug(prob)
# Prints detailed diagnostic report with suggestions
```
