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
N = 20   # horizon

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


def state_equation_jacobians(x, u):
    theta, omega = x
    u0 = u[0]

    # df/dx
    A = np.eye(nx)
    A[0, 1] = dt
    domega_dot_dtheta = -a * np.cos(theta) - c * np.sin(theta) * u0
    domega_dot_domega = -b
    A[1, 0] = dt * domega_dot_dtheta
    A[1, 1] = 1.0 + dt * domega_dot_domega

    # df/du
    B = np.zeros((nx, nu))
    domega_dot_du = c * np.cos(theta) + 2.0 * d * u0
    B[1, 0] = dt * domega_dot_du

    return A, B


def measurement_equation_jacobian(x):
    return np.array([[1.0, 0.0]])


def simulate_trajectory(X_initial, U):
    X = np.zeros((N + 1, nx))
    X[0] = X_initial
    for k in range(N):
        X[k + 1] = sqp_cost_matrices.calculate_state_function(
            X[k], U[k], state_space_parameters)

    return X

# --- Cost and gradient (first-order adjoint) ---


def compute_cost_and_gradient(
        X_initial: np.ndarray,
        U: np.ndarray
):
    X = simulate_trajectory(X_initial, U)
    Y = np.zeros((X.shape[0], Qy.shape[0]))
    for k in range(X.shape[0]):
        Y[k] = sqp_cost_matrices.calculate_measurement_function(
            X[k], state_space_parameters)

    J = 0.0
    for k in range(N):
        e_y_r = Y[k] - reference_trajectory[k]
        J += X[k] @ Qx @ X[k] + e_y_r @ Qy @ e_y_r + U[k] @ R @ U[k]

    eN_y_r = Y[N] - reference_trajectory[N]
    J += X[N] @ Px @ X[N] + eN_y_r @ Py @ eN_y_r

    # terminal adjoint
    lam_next = (2 * Px) @ X[N] + \
        measurement_equation_jacobian(X[N]).T @ (2 * Py @ eN_y_r)

    grad = np.zeros_like(U)
    for k in reversed(range(N)):
        Cx_k = measurement_equation_jacobian(X[k])
        ek_y = Y[k] - reference_trajectory[k]

        Ak, B_k = state_equation_jacobians(X[k], U[k])

        grad[k] = 2 * R @ U[k] + B_k.T @ lam_next

        lam_next = 2 * Qx @ X[k] + 2 * Cx_k.T @ (Qy @ ek_y) + Ak.T @ lam_next

    return J, grad

# --- Analytic HVP using 2nd-order adjoints ---


def hvp_analytic(X_initial, U, V):

    # --- 1) forward states
    X = simulate_trajectory(X_initial, U)
    Y = np.zeros((X.shape[0], Qy.shape[0]))
    for k in range(X.shape[0]):
        Y[k] = sqp_cost_matrices.calculate_measurement_function(
            X[k], state_space_parameters)
    yN = sqp_cost_matrices.calculate_measurement_function(
        X[N], state_space_parameters)

    eN_y = yN - reference_trajectory[N]

    # --- 2) first-order adjoint (costate lambda) with output terms
    lam = np.zeros((N + 1, nx))
    Cx_N = measurement_equation_jacobian(X[N])
    lam[N] = 2 * Px @ X[N]

    for k in range(N - 1, -1, -1):
        A_k, _ = state_equation_jacobians(X[k], U[k])
        Cx_k = measurement_equation_jacobian(X[k])
        ek_y = Y[k] - reference_trajectory[k]
        lam[k] = 2 * Qx @ X[k] + Cx_k.T @ (2 * Qy @ ek_y) + \
            A_k.T @ lam[k + 1]

    # --- 3) forward directional state: delta_x ---
    dx = np.zeros((N + 1, nx))
    for k in range(N):
        A_k, B_k = state_equation_jacobians(X[k], U[k])
        dx[k + 1] = A_k @ dx[k] + B_k @ V[k]

    # --- 4) backward second-order adjoint ---
    d_lambda = np.zeros((N + 1, nx))

    # Match the treatment of the terminal term phi_xx = l_xx(X_N,·) (currently 2P)
    # Additionally, contributions from pure second-order output and second derivatives of output
    d_lambda[N] = sqp_cost_matrices.l_xx(X[N], None) @ dx[N] + \
        Cx_N.T @ (2 * Py @ (Cx_N @ dx[N])) + \
        sqp_cost_matrices.hxx_lambda_contract(
            X[N], state_space_parameters, 2 * Py @ eN_y, dx[N])

    Hu = np.zeros_like(U)
    for k in range(N - 1, -1, -1):
        A_k, B_k = state_equation_jacobians(X[k], U[k])
        Cx_k = measurement_equation_jacobian(X[k])
        ek_y = Y[k] - reference_trajectory[k]

        # dlambda_k
        term_xx = sqp_cost_matrices.fx_xx_lambda_contract(
            X[k], U[k], state_space_parameters, lam[k + 1], dx[k])
        term_xu = sqp_cost_matrices.fx_xu_lambda_contract(
            X[k], U[k], state_space_parameters, lam[k + 1], V[k])

        d_lambda[k] = (
            sqp_cost_matrices.l_xx(X[k], U[k]) @ dx[k] +
            sqp_cost_matrices.l_xu(X[k], U[k]) @ V[k] +
            A_k.T @ d_lambda[k + 1] +
            Cx_k.T @ (2 * Qy @ (Cx_k @ dx[k])) +
            sqp_cost_matrices.hxx_lambda_contract(
                X[k], state_space_parameters, 2 * Qy @ ek_y, dx[k]) +
            term_xx + term_xu
        )

        # (HV)_k:
        #   2R V + B^T dlambda_{k+1} + second-order terms from dynamics
        #   (Cu=0 -> no direct contribution from output terms)
        term_ux = sqp_cost_matrices.fu_xx_lambda_contract(
            X[k], U[k], state_space_parameters, lam[k + 1], dx[k])
        term_uu = sqp_cost_matrices.fu_uu_lambda_contract(
            X[k], U[k], state_space_parameters, lam[k + 1], V[k])

        Hu[k] = (
            sqp_cost_matrices.l_uu(X[k], U[k]) @ V[k] +
            sqp_cost_matrices.l_ux(X[k], U[k]) @ dx[k] +
            B_k.T @ d_lambda[k + 1] +
            term_ux + term_uu
        )
    return Hu


# --- Example Execution ---


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
    cost_and_gradient_function=compute_cost_and_gradient,
    hvp_function=hvp_analytic,
    X_initial=X_initial,
    u_min=u_min_mat,
    u_max=u_max_mat,
)

print("Optimized cost:", J_opt)
print("Optimal input sequence:\n", U_opt)
