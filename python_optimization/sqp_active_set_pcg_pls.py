"""
File: sqp_active_set_pcg_pls.py

SQP: Sequential Quadratic Programming
PCG: Preconditioned Conjugate Gradient
PLS: Projected Line Search

"""
import numpy as np

RHS_NORM_ZERO_LIMIT_DEFAULT = 1e-12

GRAD_NORM_ZERO_LIMIT_DEFAULT = 1e-6

FREE_MASK_U_NEAR_LIMIT_DEFAULT = 1e-12
FREE_MASK_GRAD_ZERO_LIMIT_DEFAULT = 1e-12

PCG_TOL_DEFAULT = 1e-4
PCG_MAX_ITERATION_DEFAULT = 30
PCG_PHP_MINUS_LIMIT_DEFAULT = 1e-14

ALPHA_SMALL_LIMIT_DEFAULT = 1e-6
ALPHA_DECAY_RATE_DEFAULT = 0.5

SOLVER_MAX_ITERATION_DEFAULT = 100


def vec_mask(A, mask):
    return A[mask]


def vec_unmask(v, mask, U_shape):
    out = np.zeros(U_shape)
    out[mask] = v
    return out


def hvp_free(p_free_flat, mask, U, hvp_fn, lambda_factor):
    P = vec_unmask(p_free_flat, mask, U.shape).reshape(U.shape)
    Hv_full = hvp_fn(U, P)
    Hv_full += lambda_factor * P
    return vec_mask(Hv_full, mask).reshape(-1)


class SQP_ActiveSet_PCG_PLS:
    def __init__(
            self,
            grad_norm_zero_limit=GRAD_NORM_ZERO_LIMIT_DEFAULT):

        self.grad_norm_zero_limit = grad_norm_zero_limit

        self.mask = None
        self.U = None
        self.hvp_fn = None
        self.lambda_factor = None

    def hvp_free_for_pcg(self, v):
        return hvp_free(v, self.mask, self.U, self.hvp_fn, self.lambda_factor)

    def pcg(self,
            hvp,
            rhs: np.ndarray,
            tol: float = PCG_TOL_DEFAULT,
            max_it: int = PCG_MAX_ITERATION_DEFAULT,
            M_inv=None):
        """
        Solve the system hvp(d) = rhs using PCG without matrix.
        hvp: function v -> H v
        M_inv: pre-conditioner (None or vector/function).
        """
        r = rhs.copy()
        d = np.zeros_like(rhs)
        if np.linalg.norm(r) < RHS_NORM_ZERO_LIMIT_DEFAULT:
            return d

        def apply_Minv(x):
            if M_inv is None:
                return x
            elif callable(M_inv):
                return M_inv(x)
            else:
                return x * M_inv
        z = apply_Minv(r)
        p = z.copy()
        rz = np.vdot(r, z)
        r0 = np.linalg.norm(r)
        for _ in range(max_it):
            Hp = hvp(p)
            denom = np.vdot(p, Hp)
            if denom <= PCG_PHP_MINUS_LIMIT_DEFAULT:
                break
            alpha = rz / denom
            d += alpha * p
            r -= alpha * Hp
            if np.linalg.norm(r) <= tol * r0:
                break
            z = apply_Minv(r)
            rz_new = np.vdot(r, z)
            beta = rz_new / rz
            p = z + beta * p
            rz = rz_new
        return d

    def free_mask(self,
                  U: np.ndarray,
                  grad: np.ndarray,
                  umin: np.ndarray,
                  umax: np.ndarray,
                  atol: float = FREE_MASK_U_NEAR_LIMIT_DEFAULT,
                  gtol: float = FREE_MASK_GRAD_ZERO_LIMIT_DEFAULT):
        """
        True = Free, False = Fixed.
        At lower bound g>0 (going outside) -> Fixed.
        At upper bound g<0 (going outside) -> Fixed.
        """

        m = np.ones_like(U, dtype=bool)
        at_lo = np.isclose(U, umin, atol=atol)
        at_hi = np.isclose(U, umax, atol=atol)
        m[at_lo & (grad > gtol)] = False
        m[at_hi & (grad < -gtol)] = False
        return m

    def solve(
        self,
        U_init: np.ndarray,
        cost_and_grad_fn,
        hvp_fn,
        u_min: np.ndarray,
        u_max: np.ndarray,
        max_iter: int = SOLVER_MAX_ITERATION_DEFAULT,
        cg_it: int = PCG_MAX_ITERATION_DEFAULT,
        cg_tol: float = PCG_TOL_DEFAULT,
        lambda_factor: float = 1e-6,
    ):
        """
        General SQP solver
        (Active Set + Preconditioned Conjugate Gradient + Projected Line Search).
        - U_init: Initial input sequence (N, nu)
        - cost_and_grad_fn(U): Function that returns (J, grad)
        - hvp_fn(U, V): Function that returns HVP (H*V)
        - u_min, u_max: Input lower and upper bounds (N, nu)
        """

        U = U_init.copy()
        for iteration in range(max_iter):
            J, grad = cost_and_grad_fn(U)
            g = grad.copy()
            if np.linalg.norm(g) < self.grad_norm_zero_limit:
                break
            mask = self.free_mask(U, g, u_min, u_max)

            rhs_free = (-vec_mask(g, mask)).reshape(-1)

            # If the user provides appropriate preconditioning, replace it here.
            diagR_full = np.ones_like(U)

            M_inv_full = 1.0 / (diagR_full + lambda_factor)
            M_inv_free = vec_mask(M_inv_full, mask).reshape(-1)

            self.mask = mask
            self.U = U
            self.hvp_fn = hvp_fn
            self.lambda_factor = lambda_factor

            d_free = self.pcg(self.hvp_free_for_pcg, rhs_free,
                              tol=cg_tol, max_it=cg_it, M_inv=M_inv_free)
            d = vec_unmask(d_free.reshape(-1), mask, U.shape)
            alpha = 1.0
            U_new = U.copy()
            while True:
                U_cand = U + alpha * d
                U_cand = np.minimum(np.maximum(U_cand, u_min), u_max)
                J_cand, _ = cost_and_grad_fn(U_cand)
                if J_cand <= J or alpha < ALPHA_SMALL_LIMIT_DEFAULT:
                    U_new = U_cand
                    J = J_cand
                    break
                alpha *= ALPHA_DECAY_RATE_DEFAULT
            U = U_new

        return U, J
