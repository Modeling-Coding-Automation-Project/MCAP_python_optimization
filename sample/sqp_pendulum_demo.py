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

from python_optimization.sqp_active_set_pcg_pls import SQP_ActiveSet_PCG_PLS


def create_plant_model():
    theta, omega, u0, dt, a, b, c, d = sp.symbols(
        'theta omega u0 dt a b c d', real=True)

    theta_next = theta + dt * omega
    omega_dot = -a * sp.sin(theta) - b * omega + c * \
        sp.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot

    f = sp.Matrix([theta_next, omega_next])
    h = theta

    return f, h


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

# cost weights
Qx = np.diag([2.5, 0.5])
Qy = np.array([[2.5]])
R = np.diag([0.05])
Px = Qx.copy()
Py = Qy.copy()

# reference
reference = np.array([0.0])
reference_trajectory = np.tile(reference, (N + 1, 1))


def state_equation(x, u):
    theta, omega = x
    u0 = u[0]
    theta_next = theta + dt * omega
    omega_dot = -a * np.sin(theta) - b * omega + c * \
        np.cos(theta) * u0 + d * (u0 ** 2)
    omega_next = omega + dt * omega_dot
    return np.array([theta_next, omega_next])


def measurement_equation(x):
    return x[0]


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

# --- Second-order contraction callbacks (non-zero) ---


def fx_xx_T_contract(x, u, lam_next, dx):  # -> (nx,)
    # sum_i lam_i * ( d^2 f_i / dx^2 @ dx )
    theta, _ = x
    u0 = u[0]
    lam = lam_next
    dx0 = dx[0]

    # Only f2 (omega_next) is nonlinear in x:
    # dd f2 / dtheta^2 = dt * ( a*sin(theta) - c*cos(theta)*u )
    h_theta_theta = dt * (a * np.sin(theta) - c * np.cos(theta) * u0)

    v0 = lam[1] * (h_theta_theta * dx0)  # component along theta
    # no cross terms, no dependence on omega in second derivative
    return np.array([v0, 0.0])


def fx_xu_T_contract(x, u, lam_next, du):  # -> (nx,)
    # sum_i lam_i * ( d^2 f_i / (dx du) @ du )
    theta, _ = x
    lam = lam_next
    du0 = du[0]

    # For f2: d/dtheta (d f2/du) = dt * (-c * sin(theta))
    h_theta_u = dt * (-c * np.sin(theta))
    v0 = lam[1] * (h_theta_u * du0)
    return np.array([v0, 0.0])


def fu_xx_T_contract(x, u, lam_next, dx):  # -> (nu,)
    # sum_i lam_i * ( d^2 f_i / (du dx) @ dx )
    theta, _ = x
    lam = lam_next
    dx0 = dx[0]

    # For f2: d/du (d f2/dtheta) = dt * (-c * sin(theta))
    h_u_theta = dt * (-c * np.sin(theta))
    val = lam[1] * (h_u_theta * dx0)
    return np.array([val])


def fu_uu_T_contract(x, u, lam_next, du):  # -> (nu,)
    # sum_i lam_i * ( d^2 f_i / du^2 @ du )
    lam = lam_next
    du0 = du[0]

    # For f2: dd f2 / du^2 = dt * (2d)
    h_uu = dt * (2.0 * d)
    val = lam[1] * (h_uu * du0)
    return np.array([val])


def hxx_T_contract(x, w, dx):
    return np.zeros_like(dx)

# --- Cost Hessians (constant quadratic stage/terminal cost) ---


def l_xx(x, u):
    return 2 * Qx


def l_uu(x, u):
    return 2 * R


def l_xu(x, u):
    return np.zeros((nx, nu))


def l_ux(x, u):
    return np.zeros((nu, nx))

# --- Rollout ---


def simulate_trajectory(X_initial, U):
    X = np.zeros((N + 1, nx))
    X[0] = X_initial
    for k in range(N):
        X[k + 1] = state_equation(X[k], U[k])
    return X

# --- Cost and gradient (first-order adjoint) ---


def compute_cost_and_gradient(
        X_initial: np.ndarray,
        U: np.ndarray
):
    X = simulate_trajectory(X_initial, U)
    Y = np.zeros((X.shape[0], Qy.shape[0]))
    for k in range(X.shape[0]):
        Y[k] = measurement_equation(X[k]).reshape(-1, 1)

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
        Y[k] = measurement_equation(X[k]).reshape(-1, 1)
    yN = measurement_equation(X[N])

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
    d_lambda[N] = l_xx(X[N], None) @ dx[N] + \
        Cx_N.T @ (2 * Py @ (Cx_N @ dx[N])) + \
        hxx_T_contract(X[N], 2 * Py @ eN_y, dx[N])

    Hu = np.zeros_like(U)
    for k in range(N - 1, -1, -1):
        A_k, B_k = state_equation_jacobians(X[k], U[k])
        Cx_k = measurement_equation_jacobian(X[k])
        ek_y = Y[k] - reference_trajectory[k]

        # dlambda_k
        term_xx = fx_xx_T_contract(X[k], U[k], lam[k + 1], dx[k])
        term_xu = fx_xu_T_contract(X[k], U[k], lam[k + 1], V[k])

        d_lambda[k] = (
            l_xx(X[k], U[k]) @ dx[k] +
            l_xu(X[k], U[k]) @ V[k] +
            A_k.T @ d_lambda[k + 1] +
            Cx_k.T @ (2 * Qy @ (Cx_k @ dx[k])) +
            hxx_T_contract(X[k], 2 * Qy @ ek_y, dx[k]) +
            term_xx + term_xu
        )

        # (HV)_k:
        #   2R V + B^T dlambda_{k+1} + second-order terms from dynamics
        #   (Cu=0 -> no direct contribution from output terms)
        term_ux = fu_xx_T_contract(X[k], U[k], lam[k + 1], dx[k])
        term_uu = fu_uu_T_contract(X[k], U[k], lam[k + 1], V[k])
        Hu[k] = (
            l_uu(X[k], U[k]) @ V[k] +
            l_ux(X[k], U[k]) @ dx[k] +
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
