"""
File: sqp_active_set_pcg_pls.py

SQP: Sequential Quadratic Programming
PCG: Preconditioned Conjugate Gradient
PLS: Projected Line Search

"""
import numpy as np


class SQP_ActiveSet_PCG_PLS:
    def __init__(self):
        pass

    def pcg(self,
            hvp,
            rhs: np.ndarray,
            tol: float = 1e-6,
            max_it: int = 50,
            M_inv=None):
        """
        Solve the system hvp(d) = rhs using PCG without matrix.
        hvp: function v -> H v
        M_inv: pre-conditioner (None or vector/function).
        """
        r = rhs.copy()
        d = np.zeros_like(rhs)
        if np.linalg.norm(r) < 1e-16:
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
            if denom <= 1e-14:
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
                  atol: float = 1e-12,
                  gtol: float = 1e-12):
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
            max_iter: int = 50,
            cg_it: int = 30,
            cg_tol: float = 1e-4,
            lambda_factor: float = 1e-6,
            callback=None
    ):
        """
        General SQP solver
        (Active Set + Preconditioned Conjugate Gradient + Projected Line Search).
        - U_init: Initial input sequence (N, nu)
        - cost_and_grad_fn(U): Function that returns (J, grad)
        - hvp_fn(U, V): Function that returns HVP (H*V)
        - u_min, u_max: Input lower and upper bounds (N, nu)
        - callback: Function called at each iteration (iteration, U, J, grad)
        """

        U = U_init.copy()
        for iteration in range(max_iter):
            J, grad = cost_and_grad_fn(U)
            g = grad.copy()
            if np.linalg.norm(g) < 1e-6:
                break
            mask = self.free_mask(U, g, u_min, u_max)

            def vec_mask(A):
                return A[mask]

            def vec_unmask(v):
                out = np.zeros_like(U)
                out[mask] = v
                return out

            def hvp_free(p_free_flat):
                P = vec_unmask(p_free_flat).reshape(U.shape)
                Hv_full = hvp_fn(U, P)
                Hv_full += lambda_factor * P
                return vec_mask(Hv_full).reshape(-1)
            rhs_free = (-vec_mask(g)).reshape(-1)

            # If the user provides appropriate preconditioning, replace it here.
            diagR_full = np.ones_like(U)

            M_inv_full = 1.0 / (diagR_full + lambda_factor)
            M_inv_free = vec_mask(M_inv_full).reshape(-1)
            d_free = self.pcg(lambda v: hvp_free(v), rhs_free,
                              tol=cg_tol, max_it=cg_it, M_inv=M_inv_free)
            d = vec_unmask(d_free.reshape(-1))
            alpha = 1.0
            U_new = U.copy()
            while True:
                U_cand = U + alpha * d
                U_cand = np.minimum(np.maximum(U_cand, u_min), u_max)
                J_cand, _ = cost_and_grad_fn(U_cand)
                if J_cand <= J or alpha < 1e-6:
                    U_new = U_cand
                    J = J_cand
                    break
                alpha *= 0.5
            U = U_new
            if callback is not None:
                callback(iteration, U, J, g)
        return U, J
