"""
File: sqp_active_set_pcg_pls.py

SQP: Sequential Quadratic Programming
PCG: Preconditioned Conjugate Gradient
PLS: Projected Line Search
HVP: Hessian-Vector Product

"""
import numpy as np

RHS_NORM_ZERO_LIMIT_DEFAULT = 1e-12

GRADIENT_NORM_ZERO_LIMIT_DEFAULT = 1e-6

FREE_MASK_U_NEAR_LIMIT_DEFAULT = 1e-12
FREE_MASK_GRADIENT_ZERO_LIMIT_DEFAULT = 1e-12

PCG_TOL_DEFAULT = 1e-4
PCG_MAX_ITERATION_DEFAULT = 30
PCG_PHP_MINUS_LIMIT_DEFAULT = 1e-14

LINE_SEARCH_MAX_ITERATION_DEFAULT = 20

ALPHA_SMALL_LIMIT_DEFAULT = 1e-6
ALPHA_DECAY_RATE_DEFAULT = 0.5

SOLVER_MAX_ITERATION_DEFAULT = 100

LAMBDA_FACTOR_DEFAULT = 1e-6


def apply_M_inv(
        x: np.ndarray,
        M_inv=None):

    if M_inv is None:
        return x
    else:
        return x * M_inv


def vec_mask(
        A: np.ndarray,
        mask: np.ndarray):

    return A[mask]


def vec_unmask(
        v: np.ndarray,
        mask: np.ndarray,
        U_shape: tuple):

    out = np.zeros(U_shape)
    out[mask] = v
    return out


class SQP_ActiveSet_PCG_PLS:
    def __init__(
            self,
            U_size: int,
            gradient_norm_zero_limit=GRADIENT_NORM_ZERO_LIMIT_DEFAULT,
            alpha_small_limit=ALPHA_SMALL_LIMIT_DEFAULT,
            alpha_decay_rate=ALPHA_DECAY_RATE_DEFAULT,
            pcg_php_minus_limit=PCG_PHP_MINUS_LIMIT_DEFAULT,
            solver_max_iteration: int = SOLVER_MAX_ITERATION_DEFAULT,
            pcg_max_iteration: int = PCG_MAX_ITERATION_DEFAULT,
            line_search_max_iteration: int = LINE_SEARCH_MAX_ITERATION_DEFAULT,
            pcg_tol: float = PCG_TOL_DEFAULT,
            lambda_factor: float = LAMBDA_FACTOR_DEFAULT,
    ):

        self._gradient_norm_zero_limit = gradient_norm_zero_limit
        self._alpha_small_limit = alpha_small_limit
        self._alpha_decay_rate = alpha_decay_rate
        self._pcg_php_minus_limit = pcg_php_minus_limit

        self._solver_max_iteration = solver_max_iteration
        self._pcg_max_iteration = pcg_max_iteration
        self._line_search_max_iteration = line_search_max_iteration

        self._pcg_tol = pcg_tol
        self._lambda_factor = lambda_factor

        self._diag_R_full = np.ones((U_size))

        self._mask = None

        self.U: np.ndarray = None
        self.hvp_function = None
        self.X_initial = None

    # setter
    def set_gradient_norm_zero_limit(self, limit: float):
        self._gradient_norm_zero_limit = limit

    def set_alpha_small_limit(self, limit: float):
        self._alpha_small_limit = limit

    def set_alpha_decay_rate(self, rate: float):
        self._alpha_decay_rate = rate

    def set_pcg_php_minus_limit(self, limit: float):
        self._pcg_php_minus_limit = limit

    def set_solver_max_iteration(self, max_iteration: int):
        self._solver_max_iteration = max_iteration

    def set_pcg_max_iteration(self, max_iteration: int):
        self._pcg_max_iteration = max_iteration

    def set_line_search_max_iteration(self, max_iteration: int):
        self._line_search_max_iteration = max_iteration

    def set_pcg_tol(self, tol: float):
        self._pcg_tol = tol

    def set_lambda_factor(self, factor: float):
        self._lambda_factor = factor

    def set_diag_R_full(self, diag_R_full: np.ndarray):
        self._diag_R_full = diag_R_full

    # functions
    def hvp_free(
        self,
        p_free_flat: np.ndarray,
    ):
        P = vec_unmask(p_free_flat, self._mask,
                       self.U.shape).reshape(self.U.shape)
        Hv_full = self.hvp_function(self.X_initial, self.U, P)
        Hv_full += self._lambda_factor * P

        return vec_mask(Hv_full, self._mask).reshape(-1)

    def preconditioned_conjugate_gradient(
            self,
            rhs: np.ndarray,
            M_inv=None):
        """
        Solve the system hvp_function(d) = rhs using PCG without matrix.
        hvp_function: function v -> H v
        M_inv: pre-conditioner (None or vector/function).
        """
        r = rhs.copy()
        d = np.zeros_like(rhs)
        if np.linalg.norm(r) < RHS_NORM_ZERO_LIMIT_DEFAULT:
            return d

        # Preconditioning
        z = apply_M_inv(r, M_inv=M_inv)
        p = z.copy()
        rz = np.vdot(r, z)
        r0 = np.linalg.norm(r)

        for _ in range(self._pcg_max_iteration):
            Hp = self.hvp_free(p)

            denominator = np.vdot(p, Hp)

            # Simple handling of negative curvature and semi-definiteness
            if denominator <= self._pcg_php_minus_limit:
                break

            alpha = rz / denominator
            d += alpha * p
            r -= alpha * Hp
            if np.linalg.norm(r) <= self._pcg_tol * r0:
                break
            z = apply_M_inv(r, M_inv=M_inv)
            rz_new = np.vdot(r, z)
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
        return d

    def free_mask(self,
                  U: np.ndarray,
                  gradient: np.ndarray,
                  umin: np.ndarray,
                  umax: np.ndarray,
                  atol: float = FREE_MASK_U_NEAR_LIMIT_DEFAULT,
                  gtol: float = FREE_MASK_GRADIENT_ZERO_LIMIT_DEFAULT):
        """
        True = Free, False = Fixed.
        At lower bound g>0 (going outside) -> Fixed.
        At upper bound g<0 (going outside) -> Fixed.
        """

        m = np.ones_like(U, dtype=bool)
        at_lower = np.isclose(U, umin, atol=atol)
        at_upper = np.isclose(U, umax, atol=atol)

        m[at_lower & (gradient > gtol)] = False
        m[at_upper & (gradient < -gtol)] = False

        return m

    def solve(
        self,
        U_initial: np.ndarray,
        cost_and_gradient_function,
        hvp_function,
        X_initial: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ):
        """
        General SQP solver
        (Active Set + Preconditioned Conjugate Gradient + Projected Line Search).
        - U_initial: Initial input sequence (N, nu)
        - cost_and_gradient_function(X_initial, U): Function that returns (J, gradient)
        - hvp_function(X_initial, U, V): Function that returns HVP (H*V)
        - u_min, u_max: Input lower and upper bounds (N, nu)
        """
        self.X_initial = X_initial
        U = U_initial.copy()

        for _ in range(self._solver_max_iteration):
            # Calculate cost and gradient
            J, gradient = cost_and_gradient_function(X_initial, U)
            g = gradient.copy()
            if np.linalg.norm(g) < self._gradient_norm_zero_limit:
                break
            self._mask = self.free_mask(U, g, u_min, u_max)

            rhs_free = (-vec_mask(g, self._mask)).reshape(-1)

            M_inv_full = 1.0 / (self._diag_R_full + self._lambda_factor)
            M_inv_free = vec_mask(M_inv_full, self._mask).reshape(-1)

            self.U = U
            self.hvp_function = hvp_function

            d_free = self.preconditioned_conjugate_gradient(
                rhs=rhs_free,
                M_inv=M_inv_free)

            # Back-substitution of the solution (fixed components remain 0)
            d = vec_unmask(d_free.reshape(-1), self._mask, U.shape)

            # line search and projection
            # (No distinction between fixed/free is needed here,
            #  project the whole)
            alpha = 1.0
            U_new = U.copy()

            for _ in range(self._line_search_max_iteration):
                U_candidate = U + alpha * d
                U_candidate = np.minimum(np.maximum(U_candidate, u_min), u_max)
                J_candidate, _ = cost_and_gradient_function(
                    X_initial, U_candidate)
                if J_candidate <= J or alpha < self._alpha_small_limit:
                    U_new = U_candidate
                    J = J_candidate
                    break
                alpha *= self._alpha_decay_rate
            U = U_new

        return U, J
