"""
File: sqp_pendulum_demo.py

A sample code for Nonlinear Model Predictive Control (NMPC) using
Sequential Quadratic Programming (SQP) with Active-Set method,
Preconditioned Conjugate Gradient (PCG), and Projected Line Search (PLS).

Plant: Pendulum-like system with nonlinear actuator dynamics
"""
import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import sympy as sp
from dataclasses import dataclass

from optimization_utility.sqp_matrix_utility import SQP_CostMatrices_NMPC
from python_optimization.sqp_active_set_pcg_pls import SQP_ActiveSet_PCG_PLS


def create_plant_model():
    theta, omega, u0, dt, a, b, c, d = sp.symbols(
        'theta omega u0 dt a b c d', real=True)

    theta_next = theta + dt * omega
    omega_dot = -a * sp.sin(theta) - b * omega + c * \
        sp.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot

    f = sp.Matrix([theta_next, omega_next])
    h = sp.Matrix([[theta]])

    x_syms = sp.Matrix([[theta], [omega]])
    u_syms = sp.Matrix([[u0]])

    return f, h, x_syms, u_syms


# --- Nonlinear NMPC Problem (Pendulum-like with nonlinear actuator) ---
nx = 2   # [theta, omega]
nu = 1   # scalar input
ny = 1   # scalar output (theta)
N = 20   # Prediction Horizon

dt = 0.05

# dynamics params
a = 9.81     # gravity/l over I scaling
b = 0.3      # damping
c = 1.2      # state-dependent control effectiveness: cos(theta)*u
d = 0.10     # actuator nonlinearity: u^2


@dataclass
class Parameters:
    a: float = a
    b: float = b
    c: float = c
    d: float = d
    dt: float = dt


state_space_parameters = Parameters()

# cost weights
Qx = np.diag([2.5, 0.5])
Qy = np.array([[2.5]])
R = np.diag([0.05])
Px = Qx.copy()
Py = Qy.copy()

# reference
reference = np.array([0.0])
reference_trajectory = np.tile(reference, (N + 1, 1))

# Create symbolic plant model
f, h, x_syms, u_syms = create_plant_model()

sqp_cost_matrices = SQP_CostMatrices_NMPC(
    x_syms=x_syms,
    u_syms=u_syms,
    state_equation_vector=f,
    measurement_equation_vector=h,
    Np=N,
    Qx=Qx,
    Qy=Qy,
    R=R
)


# --- Example Execution ---
sqp_cost_matrices.state_space_parameters = state_space_parameters
sqp_cost_matrices.reference_trajectory = reference_trajectory

# input bounds
u_min = np.array([-2.0])
u_max = np.array([2.0])

# initial state
X_initial = np.array([np.pi / 4.0, 0.0])


U_initial = np.zeros((N, nu))
u_min_mat = np.tile(u_min, (N, 1))
u_max_mat = np.tile(u_max, (N, 1))

solver = SQP_ActiveSet_PCG_PLS(
    U_size=(U_initial.shape[0], U_initial.shape[1])
)
solver.set_solver_max_iteration(30)

U_opt, J_opt = solver.solve(
    U_initial=U_initial,
    cost_and_gradient_function=sqp_cost_matrices.compute_cost_and_gradient,
    hvp_function=sqp_cost_matrices.hvp_analytic,
    X_initial=X_initial,
    u_min=u_min_mat,
    u_max=u_max_mat,
)

print("Optimized cost:", J_opt)
print("Optimal input sequence:\n", U_opt)
