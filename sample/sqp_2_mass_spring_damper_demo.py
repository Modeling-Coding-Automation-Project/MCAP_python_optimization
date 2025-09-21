"""
File: sqp_nmpc_demo.py

A sample code for Nonlinear Model Predictive Control (NMPC) using
Sequential Quadratic Programming (SQP) with Active-Set method,
Preconditioned Conjugate Gradient (PCG), and Projected Line Search (PLS).

Plant: 2-Mass Spring-Damper System
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
    """Return sympy expressions for the 2-mass spring-damper discrete dynamics.

    States: x1, v1, x2, v2
    Inputs: u1, u2
    Discrete dynamics use Euler integration with time-step dt.
    Returns (f, h, x_syms, u_syms) matching the sqp conventions.
    """
    x1, v1, x2, v2, u1, u2, dt_s, m1_s, m2_s, k1_s, k2_s, k3_s, b1_s, b2_s, b3_s = \
        sp.symbols('x1 v1 x2 v2 u1 u2 dt m1 m2 k1 k2 k3 b1 b2 b3', real=True)

    # continuous-time accelerations
    v1_dot = (-k1_s * x1 - b1_s * v1 - k2_s *
              (x1 - x2) - b2_s * (v1 - v2) + u1) / m1_s
    v2_dot = (-k3_s * x2 - b3_s * v2 - k2_s *
              (x2 - x1) - b2_s * (v2 - v1) + u2) / m2_s

    # discrete-time update (Euler)
    x1_next = x1 + dt_s * v1
    v1_next = v1 + dt_s * v1_dot
    x2_next = x2 + dt_s * v2
    v2_next = v2 + dt_s * v2_dot

    f = sp.Matrix([x1_next, v1_next, x2_next, v2_next])

    # measurement: for this demo we can measure positions only
    h = sp.Matrix([[x1], [x2]])

    x_syms = sp.Matrix([[x1], [v1], [x2], [v2]])
    u_syms = sp.Matrix([[u1], [u2]])

    return f, h, x_syms, u_syms


# --- NMPC Problem Definition (2-Mass Spring-Damper System) ---

nx = 4   # State dimension
nu = 2   # Input dimension
N = 10   # Prediction Horizon

dt = 0.1


@dataclass
class Parameters:
    m1: float = 1.0
    m2: float = 1.0
    k1: float = 10.0
    k2: float = 15.0
    k3: float = 10.0
    b1: float = 1.0
    b2: float = 2.0
    b3: float = 1.0
    dt: float = dt


state_space_parameters = Parameters()

# cost weights
Qx = np.diag([0.5, 0.1, 0.5, 0.1])
Qy = np.diag([0.5, 0.5])
R = np.diag([0.1, 0.1])
Px = Qx.copy()
Py = Qy.copy()

u_min = np.array([[-1.0], [-1.0]])
u_max = np.array([[1.0], [1.0]])

# reference
reference = np.array([0.0])
reference_trajectory = np.tile(reference, (1, N + 1))

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
    R=R,
    U_min=u_min,
    U_max=u_max,
)


# --- Example Execution ---
sqp_cost_matrices.state_space_parameters = state_space_parameters
sqp_cost_matrices.reference_trajectory = reference_trajectory


X_initial = np.array([[5.0], [0.0], [5.0], [0.0]])
U_initial = np.zeros((nu, N))

solver = SQP_ActiveSet_PCG_PLS(
    U_size=(nu, N)
)
solver.set_solver_max_iteration(20)

U_opt = solver.solve(
    U_initial=U_initial,
    cost_and_gradient_function=sqp_cost_matrices.compute_cost_and_gradient,
    cost_function=sqp_cost_matrices.compute_cost,
    hvp_function=sqp_cost_matrices.hvp_analytic,
    X_initial=X_initial,
    U_min_matrix=sqp_cost_matrices.U_min_matrix,
    U_max_matrix=sqp_cost_matrices.U_max_matrix,
)

print("Optimized cost:", solver.J_opt)
print("Optimal input sequence:\n", U_opt)
