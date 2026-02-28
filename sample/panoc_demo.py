"""
File: panoc_demo.py

This script demonstrates the PANOC solver on the Rosenbrock function
with box constraints.

Problem:
    min  f(u) = (a - u0)^2 + b * (u1 - u0^2)^2
    s.t. -1.5 <= u0 <= 1.5
         -0.5 <= u1 <= 2.5

The unconstrained minimum is at (a, a^2) = (1, 1).
With the box constraints above, (1, 1) is feasible,
so the solver should find it.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from python_optimization.panoc import PANOC_Cache, PANOC_Optimizer

# ============================================================
# 1. Define the problem
# ============================================================
# Rosenbrock parameters
a = 1.0
b = 100.0


def cost(u: np.ndarray) -> float:
    """Rosenbrock cost function."""
    return (a - u[0, 0]) ** 2 + b * (u[1, 0] - u[0, 0] ** 2) ** 2


def gradient(u: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock cost function."""
    g = np.zeros((2, 1))
    g[0, 0] = -2.0 * (a - u[0, 0]) - 4.0 * b * \
        u[0, 0] * (u[1, 0] - u[0, 0] ** 2)
    g[1, 0] = 2.0 * b * (u[1, 0] - u[0, 0] ** 2)
    return g


# Box constraints
u_min = np.array([[-1.5], [-0.5]])
u_max = np.array([[1.5], [2.5]])

# ============================================================
# 2. Create cache and solver
# ============================================================
n = 2  # problem dimension
cache = PANOC_Cache(problem_size=n, tolerance=1e-8, lbfgs_memory=10)

solver = PANOC_Optimizer(
    cost_func=cost,
    gradient_func=gradient,
    cache=cache,
    u_min=u_min,
    u_max=u_max,
    max_iteration=200,
)

# ============================================================
# 3. Solve
# ============================================================
u0 = np.array([[-1.0], [2.0]])  # initial guess
print(f"Initial guess : u = {u0.flatten()}")
print(f"Initial cost  : f(u) = {cost(u0):.6f}")
print()

status = solver.solve(u0)
# u0 is modified in-place and now holds the solution

# ============================================================
# 4. Print results
# ============================================================
print("--- PANOC result ---")
print(f"Exit status   : {status.exit_status.name}")
print(f"Iterations    : {status.number_of_iteration}")
print(f"Solution      : u = {u0.flatten()}")
print(f"Cost at sol.  : f(u) = {status.cost_value:.10f}")
print(f"||gamma*FPR|| : {status.norm_fixed_point_residual:.2e}")
print(f"Converged     : {status.has_converged()}")
print()

# Verify feasibility
assert np.all(u0 >= u_min - 1e-12) and np.all(u0 <= u_max + 1e-12), \
    "Solution violates box constraints!"
print("Feasibility check: OK")
print(
    f"Distance to true optimum (1, 1): {np.linalg.norm(u0 - np.array([[1.0], [1.0]])):.2e}")
