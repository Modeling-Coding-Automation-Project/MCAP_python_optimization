"""
File: sqp_active_set_pcg_pls.py

SQP: Sequential Quadratic Programming
PCG: Preconditioned Conjugate Gradient
PLS: Projected Line Search

"""
import numpy as np


def pcg(hvp, rhs, tol=1e-6, max_it=50, M_inv=None):
    """
    (行列なし) PCG で hvp(d) = rhs を解く。rhs, d は任意形状OK。
    hvp: 関数 v -> H v
    M_inv: 前処理（None or ベクトル/関数）。
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


def free_mask(U, grad, umin, umax, atol=1e-12, gtol=1e-12):
    """
    True = 自由, False = 固定。
    下限で g>0（外側へ行く）→固定。上限で g<0（外側へ行く）→固定。
    """
    m = np.ones_like(U, dtype=bool)
    at_lo = np.isclose(U, umin, atol=atol)
    at_hi = np.isclose(U, umax, atol=atol)
    m[at_lo & (grad > gtol)] = False
    m[at_hi & (grad < -gtol)] = False
    return m


def solve_sqp(
        U_init,
        cost_and_grad_fn,
        hvp_fn,
        u_min,
        u_max,
        max_iter=50,
        cg_it=30,
        cg_tol=1e-4,
        lam=1e-6,
        callback=None
):
    """
    汎用SQPソルバー（アクティブセット+前処理付き共役勾配法+投影ラインサーチ）。
    - U_init: 初期入力系列 (N, nu)
    - cost_and_grad_fn(U): (J, grad) を返す関数
    - hvp_fn(U, V): HVP (H*V) を返す関数
    - u_min, u_max: 入力下限・上限 (N, nu)
    - callback: 各イテレーションで呼ばれる関数 (iteration, U, J, grad)
    """
    U = U_init.copy()
    for iteration in range(max_iter):
        J, grad = cost_and_grad_fn(U)
        g = grad.copy()
        if np.linalg.norm(g) < 1e-6:
            break
        mask = free_mask(U, g, u_min, u_max)

        def vec_mask(A):
            return A[mask]

        def vec_unmask(v):
            out = np.zeros_like(U)
            out[mask] = v
            return out

        def hvp_free(p_free_flat):
            P = vec_unmask(p_free_flat).reshape(U.shape)
            Hv_full = hvp_fn(U, P)
            Hv_full += lam * P
            return vec_mask(Hv_full).reshape(-1)
        rhs_free = (-vec_mask(g)).reshape(-1)
        diagR_full = np.ones_like(U)  # ユーザーが適切な前処理を与える場合は差し替え
        M_inv_full = 1.0 / (diagR_full + lam)
        M_inv_free = vec_mask(M_inv_full).reshape(-1)
        d_free = pcg(lambda v: hvp_free(v), rhs_free,
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
