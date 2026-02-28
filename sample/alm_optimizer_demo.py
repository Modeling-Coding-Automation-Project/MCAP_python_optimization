"""
File: alm_optimizer_demo.py

This script demonstrates the ALM/PM optimizer on a simple 2D problem
with an additional output (inequality) constraint handled via ALM.

Problem:
    min  f(u) = (1 - u0)^2 + 100*(u1 - u0^2)^2      (Rosenbrock)
     u
    s.t. -1.5 <= u0 <= 1.5                            (box on u)
         -1.5 <= u1 <= 1.5
         u0 + u1 <= 1.0                               (output constraint via ALM)

The unconstrained Rosenbrock minimum is at (1, 1), but the output
constraint u0 + u1 <= 1 makes that point infeasible.
The constrained optimum lies along the line u0 + u1 = 1.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from python_optimization.panoc import PANOC_Cache
from python_optimization.alm_optimizer import (
    ALM_Factory,
    ALM_Problem,
    ALM_Cache,
    ALM_Optimizer,
    BoxProjectionOperator,
    BallProjectionOperator,
)

# ============================================================
# 1. Define the problem
# ============================================================
# Rosenbrock parameters
a = 1.0
b = 100.0


def cost(u: np.ndarray) -> float:
    """Rosenbrock cost function."""
    return (a - u[0]) ** 2 + b * (u[1] - u[0] ** 2) ** 2


def gradient(u: np.ndarray) -> np.ndarray:
    """Gradient of the Rosenbrock cost function."""
    g = np.zeros(2)
    g[0] = -2.0 * (a - u[0]) - 4.0 * b * u[0] * (u[1] - u[0] ** 2)
    g[1] = 2.0 * b * (u[1] - u[0] ** 2)
    return g


# ---- ALM constraint: F1(u) = u0 + u1,  C = (-inf, 1.0] ----
def F1(u: np.ndarray) -> np.ndarray:
    """Mapping F1: output = u0 + u1 (scalar -> 1D array)."""
    return np.array([u[0] + u[1]])


def JF1_trans(u: np.ndarray, d: np.ndarray) -> np.ndarray:
    """JF1(u)^T * d.  Since JF1 = [1, 1], this is [d[0], d[0]]."""
    return np.array([d[0], d[0]])


def project_C(x: np.ndarray) -> None:
    """In-place projection onto C = (-inf, 1.0]."""
    np.minimum(x, 1.0, out=x)


# Box constraints on u
u_min = np.array([-1.5, -1.5])
u_max = np.array([1.5, 1.5])

# Set Y for Lagrange multipliers: ball of large radius (practically unbounded)
project_Y = BallProjectionOperator(center=None, radius=1e6)

# ============================================================
# 2. Build factory, problem, cache, optimizer
# ============================================================
# Factory constructs psi(u; xi) and nabla_psi(u; xi) from f, df, F1, JF1, C
factory = ALM_Factory(
    f=cost,
    df=gradient,
    mapping_f1=F1,
    jacobian_f1_trans=JF1_trans,
    set_c_project=project_C,
    n1=1,       # dim of F1 output
)

# Problem bundles everything the optimizer needs
problem = ALM_Problem(
    parametric_cost=factory.psi,
    parametric_gradient=factory.d_psi,
    u_min=u_min,
    u_max=u_max,
    mapping_f1=F1,
    set_c_project=project_C,
    set_y_project=project_Y,
    n1=1,
)

# Caches (PANOC inner + ALM outer)
n = 2   # decision variable dimension
panoc_cache = PANOC_Cache(problem_size=n, tolerance=1e-6, lbfgs_memory=5)
alm_cache = ALM_Cache(panoc_cache, n1=1)

# Optimizer
optimizer = ALM_Optimizer(
    alm_cache=alm_cache,
    alm_problem=problem,
    epsilon_tolerance=1e-5,
    delta_tolerance=1e-4,
    initial_penalty=10.0,
    max_outer_iterations=50,
    max_inner_iterations=500,
)

# ============================================================
# 3. Solve
# ============================================================
u0 = np.array([0.0, 0.0])  # initial guess
print(f"Initial guess   : u = {u0}")
print(f"Initial cost    : f(u) = {cost(u0):.6f}")
print(f"Initial F1(u)   : {F1(u0)[0]:.6f}  (constraint: <= 1.0)")
print()

status = optimizer.solve(u0)
# u0 is modified in-place and now holds the solution

# ============================================================
# 4. Print results
# ============================================================
print("--- ALM/PM result ---")
print(f"Exit status     : {status.exit_status.name}")
print(f"Outer iterations: {status.num_outer_iterations}")
print(f"Inner iterations: {status.num_inner_iterations}  (total PANOC steps)")
print(f"Solution        : u = [{u0[0]:.8f}, {u0[1]:.8f}]")
print(f"Cost at sol.    : f(u) = {status.cost:.8f}")
print(f"F1(u)           : {F1(u0)[0]:.8f}  (constraint: <= 1.0)")
print(f"||FPR|| (last)  : {status.last_problem_norm_fpr:.2e}")
print(f"Penalty (final) : c = {status.penalty:.1f}")
print(f"||delta y||     : {status.delta_y_norm:.2e}")
print(f"Lagrange mult.  : y = {status.lagrange_multipliers}")
print(f"Converged       : {status.has_converged()}")
print()

# Verify feasibility
f1_val = F1(u0)[0]
assert np.all(u0 >= u_min - 1e-6) and np.all(u0 <= u_max + 1e-6), \
    "Solution violates box constraints!"
assert f1_val <= 1.0 + 1e-4, \
    f"Solution violates output constraint: F1(u) = {f1_val:.8f} > 1.0"
print("Feasibility check: OK  (box + output constraint satisfied)")
