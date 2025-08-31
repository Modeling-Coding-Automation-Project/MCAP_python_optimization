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

ALPHA_SMALL_LIMIT_DEFAULT = 1e-6
ALPHA_DECAY_RATE_DEFAULT = 0.5

SOLVER_MAX_ITERATION_DEFAULT = 100

LAMBDA_FACTOR_DEFAULT = 1e-6


def apply_M_inv(
        x: np.ndarray,
        M_inv):

    if M_inv is None:
        return x
    elif callable(M_inv):
        return M_inv(x)
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


def hvp_free(
        p_free_flat: np.ndarray,
        mask: np.ndarray,
        U: np.ndarray,
        hvp_function,
        x0: np.ndarray,
        lambda_factor: float):

    P = vec_unmask(p_free_flat, mask, U.shape).reshape(U.shape)
    Hv_full = hvp_function(x0, U, P)
    Hv_full += lambda_factor * P
    return vec_mask(Hv_full, mask).reshape(-1)


class SQP_ActiveSet_PCG_PLS:
    def __init__(
            self,
            U_size: int,
            gradient_norm_zero_limit=GRADIENT_NORM_ZERO_LIMIT_DEFAULT,
            alpha_small_limit=ALPHA_SMALL_LIMIT_DEFAULT,
            alpha_decay_rate=ALPHA_DECAY_RATE_DEFAULT,
            pcg_php_minus_limit=PCG_PHP_MINUS_LIMIT_DEFAULT,
    ):

        self.gradient_norm_zero_limit = gradient_norm_zero_limit
        self.alpha_small_limit = alpha_small_limit
        self.alpha_decay_rate = alpha_decay_rate
        self.pcg_php_minus_limit = pcg_php_minus_limit

        self.mask = None
        self.U = None
        self.hvp_function = None
        self.x0 = None
        self.lambda_factor = None

        self.diag_R_full = np.ones((U_size))

    def hvp_free_for_pcg(self, v):
        return hvp_free(
            p_free_flat=v,
            mask=self.mask,
            U=self.U,
            hvp_function=self.hvp_function,
            x0=self.x0,
            lambda_factor=self.lambda_factor)

    def pcg(self,
            hvp_function,
            rhs: np.ndarray,
            tol: float = PCG_TOL_DEFAULT,
            max_it: int = PCG_MAX_ITERATION_DEFAULT,
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
        z = apply_M_inv(r, M_inv)
        p = z.copy()
        rz = np.vdot(r, z)
        r0 = np.linalg.norm(r)

        for _ in range(max_it):
            Hp = hvp_function(p)
            denominator = np.vdot(p, Hp)

            # Simple handling of negative curvature and semi-definiteness
            if denominator <= self.pcg_php_minus_limit:
                break

            alpha = rz / denominator
            d += alpha * p
            r -= alpha * Hp
            if np.linalg.norm(r) <= tol * r0:
                break
            z = apply_M_inv(r, M_inv)
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
        x0: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
        max_iteration: int = SOLVER_MAX_ITERATION_DEFAULT,
        cg_iteration: int = PCG_MAX_ITERATION_DEFAULT,
        cg_tol: float = PCG_TOL_DEFAULT,
        lambda_factor: float = LAMBDA_FACTOR_DEFAULT,
    ):
        """
        General SQP solver
        (Active Set + Preconditioned Conjugate Gradient + Projected Line Search).
        - U_initial: Initial input sequence (N, nu)
        - cost_and_gradient_function(U): Function that returns (J, gradient)
        - hvp_function(x0, U, V): Function that returns HVP (H*V)
        - u_min, u_max: Input lower and upper bounds (N, nu)
        """
        self.x0 = x0
        U = U_initial.copy()

        for iteration in range(max_iteration):
            # Calculate cost and gradient
            J, gradient = cost_and_gradient_function(U)
            g = gradient.copy()
            if np.linalg.norm(g) < self.gradient_norm_zero_limit:
                break
            mask = self.free_mask(U, g, u_min, u_max)

            rhs_free = (-vec_mask(g, mask)).reshape(-1)

            M_inv_full = 1.0 / (self.diag_R_full + lambda_factor)
            M_inv_free = vec_mask(M_inv_full, mask).reshape(-1)

            self.mask = mask
            self.U = U
            self.hvp_function = hvp_function
            self.lambda_factor = lambda_factor

            d_free = self.pcg(self.hvp_free_for_pcg, rhs_free,
                              tol=cg_tol, max_it=cg_iteration, M_inv=M_inv_free)
            d = vec_unmask(d_free.reshape(-1), mask, U.shape)
            alpha = 1.0
            U_new = U.copy()

            while True:
                U_candidate = U + alpha * d
                U_candidate = np.minimum(np.maximum(U_candidate, u_min), u_max)
                J_candidate, _ = cost_and_gradient_function(U_candidate)
                if J_candidate <= J or alpha < self.alpha_small_limit:
                    U_new = U_candidate
                    J = J_candidate
                    break
                alpha *= self.alpha_decay_rate
            U = U_new

        return U, J
