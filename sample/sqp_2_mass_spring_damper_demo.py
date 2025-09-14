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

u_min = np.array([-1.0, -1.0])
u_max = np.array([1.0, 1.0])
x_min = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
x_max = np.array([np.inf, np.inf, np.inf, np.inf])


def plant_dynamics(x, u):
    x1, v1, x2, v2 = x
    u1, u2 = u
    v1_dot = (-k1 * x1 - b1 * v1 - k2 * (x1 - x2) - b2 * (v1 - v2) + u1) / m1
    v2_dot = (-k3 * x2 - b3 * v2 - k2 * (x2 - x1) - b2 * (v2 - v1) + u2) / m2
    x1_next = x1 + dt * v1
    v1_next = v1 + dt * v1_dot
    x2_next = x2 + dt * v2
    v2_next = v2 + dt * v2_dot
    return np.array([x1_next, v1_next, x2_next, v2_next])


def dynamics_jacobians(x, u):
    a11 = -(k1 + k2) / m1
    a12 = (k2) / m1
    a13 = -(b1 + b2) / m1
    a14 = (b2) / m1
    a21 = (k2) / m2
    a22 = -(k3 + k2) / m2
    a23 = (b2) / m2
    a24 = -(b3 + b2) / m2
    A = np.eye(nx)
    A[0, 1] = dt
    A[2, 3] = dt
    A[1, 0] = dt * a11
    A[1, 2] = dt * a12
    A[1, 1] = 1.0 + dt * a13
    A[1, 3] = dt * a14
    A[3, 0] = dt * a21
    A[3, 2] = dt * a22
    A[3, 1] = dt * a23
    A[3, 3] = 1.0 + dt * a24
    B = np.zeros((nx, nu))
    B[1, 0] = dt * (1.0 / m1)
    B[3, 1] = dt * (1.0 / m2)

    return A, B

# Default callbacks for second-order contraction (implement only what's necessary)


def fx_xx_T_contract(x, u, lam_next, dx):  # -> (nx,)
    # sum_i lam_i * ( ddf_i/dx @ dx )
    return np.zeros(nx)


def fx_xu_T_contract(x, u, lam_next, du):  # -> (nx,)
    # sum_i lam_i * ( ddf_i/dxdu @ du )
    return np.zeros(nx)


def fu_xx_T_contract(x, u, lam_next, dx):  # -> (nu,)
    # sum_i lam_i * ( ddf_i/dudx @ dx )
    return np.zeros(nu)


def fu_uu_T_contract(x, u, lam_next, du):  # -> (nu,)
    # sum_i lam_i * ( ddf_i/ddu @ du )
    return np.zeros(nu)

# Default cost Hessian and cross-term functions for use in hvp_analytic


def l_xx(x, u): return 2 * Q
def l_uu(x, u): return 2 * R
def l_xu(x, u): return np.zeros((nx, nu))
def l_ux(x, u): return np.zeros((nu, nx))


def simulate_trajectory(X_initial, U):
    X = np.zeros((N + 1, nx))
    X[0] = X_initial
    for k in range(N):
        X[k + 1] = plant_dynamics(X[k], U[k])
    return X


def compute_cost_and_gradient(X_initial, U):
    X = simulate_trajectory(X_initial, U)
    J = 0.0
    for k in range(N):
        J += X[k] @ Q @ X[k] + U[k] @ R @ U[k]
    J += X[N] @ P @ X[N]
    grad = np.zeros_like(U)
    lambda_next = 2 * P @ X[N]
    for k in reversed(range(N)):
        A_k, B_k = dynamics_jacobians(X[k], U[k])
        grad[k] = 2 * R @ U[k] + B_k.T @ lambda_next
        lambda_next = 2 * Q @ X[k] + A_k.T @ lambda_next
    return J, grad


def hvp_analytic(X_initial, U, V):
    """
    Analytic HVP: For any direction V (N x nu), return H*V (H = Nabla^2 J).
    Steps:
      1) Forward: Generate X
      2) First-order adjoint: Generate lambda
      3) Forward sensitivity: Forward delta_x with V
      4) Backward second-order adjoint: Backward delta_lambda, delta_Q_u
       -> This is H*V
    """

    # --- 1) forward states
    X = simulate_trajectory(X_initial, U)

    # --- 2) first-order adjoint (costate λ)
    lam = np.zeros((N + 1, nx))
    lam[N] = 2 * P @ X[N]
    for k in range(N - 1, -1, -1):
        A_k, B_k = dynamics_jacobians(X[k], U[k])
        lam[k] = 2 * Q @ X[k] + A_k.T @ lam[k + 1]

    # --- 3) forward directional state: δx ---
    dx = np.zeros((N + 1, nx))
    for k in range(N):
        A_k, B_k = dynamics_jacobians(X[k], U[k])
        dx[k + 1] = A_k @ dx[k] + B_k @ V[k]

    # --- 4) backward second-order adjoint（一般形） ---
    d_lambda = np.zeros((N + 1, nx))
    # 端末項 φ_xx = l_xx(X_N,·) の扱いに合わせる（今は2P）
    d_lambda[N] = l_xx(X[N], None) @ dx[N]

    Hu = np.zeros_like(U)
    for k in range(N - 1, -1, -1):
        A_k, B_k = dynamics_jacobians(X[k], U[k])

        # dλ_k
        term_xx = fx_xx_T_contract(X[k], U[k], lam[k + 1], dx[k])
        term_xu = fx_xu_T_contract(X[k], U[k], lam[k + 1], V[k])
        d_lambda[k] = (
            l_xx(X[k], U[k]) @ dx[k] +
            l_xu(X[k], U[k]) @ V[k] +
            A_k.T @ d_lambda[k + 1] +
            term_xx + term_xu
        )

        # (HV)_k
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
X_initial = np.array([5.0, 0.0, 5.0, 0.0])
U_initial = np.zeros((N, nu))
u_min_mat = np.tile(u_min, (N, 1))
u_max_mat = np.tile(u_max, (N, 1))

solver = SQP_ActiveSet_PCG_PLS(
    U_size=(U_initial.shape[0], U_initial.shape[1])
)
solver.set_solver_max_iteration(20)

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
