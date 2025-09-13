import numpy as np
import sympy as sp


def extract_parameters_from_state_equations(
        f: sp.Matrix,
        h: sp.Matrix,
        x_syms: sp.Matrix,
        u_syms: sp.Matrix
):
    # Helper to collect free symbols from a sympy Matrix/Array
    def _collect_free_symbols(expr_matrix):
        syms = set()
        # expr_matrix can be a Matrix, Array, or other iterable of exprs
        try:
            iterator = list(expr_matrix)
        except Exception:
            iterator = [expr_matrix]

        for elem in iterator:
            # if it's a matrix/array nested, iterate through its entries
            if hasattr(elem, '__iter__') and not isinstance(elem, sp.Symbol):
                for sub in elem:
                    try:
                        syms.update(sub.free_symbols)
                    except Exception:
                        pass
            else:
                try:
                    syms.update(elem.free_symbols)
                except Exception:
                    pass
        return syms

    f_syms = _collect_free_symbols(f)
    h_syms = _collect_free_symbols(h)

    # Symbols that are considered states/inputs
    x_syms_set = set(x_syms)
    u_syms_set = set(u_syms)

    # Parameters are free symbols in f or h but not in x_syms or u_syms
    params = (f_syms | h_syms) - x_syms_set - u_syms_set

    # Return a sorted list for deterministic order
    try:
        sorted_params = sorted(params, key=lambda s: s.name)
    except Exception:
        sorted_params = list(params)

    return sorted_params


class SQP_CostMatrices_NMPC:
    def __init__(
            self,
            x_syms: sp.Matrix,
            u_syms: sp.Matrix,
            state_equation_vector: sp.Matrix,
            measurement_equation_vector: sp.Matrix,
            Np: int,
            Qx: np.ndarray,
            Qy: np.ndarray,
            R: np.ndarray,
            Px: np.ndarray = None,
            Py: np.ndarray = None
    ):
        self.x_syms = x_syms
        self.u_syms = u_syms
        self.f = state_equation_vector
        self.h = measurement_equation_vector

        self.Parameters = extract_parameters_from_state_equations(
            f=self.f,
            h=self.h,
            x_syms=self.x_syms,
            u_syms=self.u_syms
        )

        self.nx = Qx.shape[0]
        self.nu = R.shape[0]
        self.ny = Qy.shape[0]
        self.Np = Np

        self.Qx = Qx
        self.Qy = Qy
        self.R = R

        if Px is None:
            self.Px = Qx.copy()
        else:
            self.Px = Px

        if Py is None:
            self.Py = Qy.copy()
        else:
            self.Py = Py

        # Precompute Hessians
        self.Hf_xx, self.Hf_xu, self.Hf_ux, self.Hf_uu = \
            self._stack_hessians_for_f()
        self.Hh_xx = self._stack_hessians_for_h()

    def _stack_hessians_for_f(self):
        """
        returns:
        Hf_xx: Array shape (nx, nx, nx)
        Hf_xu: Array shape (nx, nx, nu)
        Hf_ux: Array shape (nx, nu, nx)
        Hf_uu: Array shape (nx, nu, nu)
        """
        nx = len(self.x_syms)
        nu = len(self.u_syms)

        Hf_xx_list = []
        Hf_xu_list = []
        Hf_ux_list = []
        Hf_uu_list = []

        for i in range(nx):
            fi = self.f[i, 0]
            # Hessians
            Hxx = sp.hessian(fi, self.x_syms)            # (nx, nx)
            Huu = sp.hessian(fi, self.u_syms) if nu > 0 else sp.Matrix([])
            # Mixed second derivatives (consistent index order)
            # d^2 fi / (dx du) -> rows: x, cols: u
            if nu > 0:
                Hxu = sp.Matrix([[sp.diff(sp.diff(fi, self.x_syms[j]), self.u_syms[k])
                                for k in range(nu)] for j in range(nx)])   # (nx, nu)
                Hux = sp.Matrix([[sp.diff(sp.diff(fi, self.u_syms[k]), self.x_syms[j])
                                for j in range(nx)] for k in range(nu)])   # (nu, nx)
            else:
                Hxu = sp.Matrix(np.zeros((nx, 0)))
                Hux = sp.Matrix(np.zeros((0, nx)))

            Hf_xx_list.append(Hxx)
            Hf_xu_list.append(Hxu)
            Hf_ux_list.append(Hux)
            Hf_uu_list.append(Huu)

        Hf_xx = sp.Array(Hf_xx_list)  # (nx, nx, nx)  index: [i, j, k]
        Hf_xu = sp.Array(Hf_xu_list)  # (nx, nx, nu)  index: [i, j, k]
        Hf_ux = sp.Array(Hf_ux_list)  # (nx, nu, nx)  index: [i, j, k]
        Hf_uu = sp.Array(Hf_uu_list)  # (nx, nu, nu)  index: [i, j, k]

        return Hf_xx, Hf_xu, Hf_ux, Hf_uu

    def _stack_hessians_for_h(self):
        """
        returns:
        Hh_xx: Array shape (ny, nx, nx)
        """
        ny = self.h.shape[0]
        H_list = []
        for j in range(ny):
            hj = self.h[j, 0]
            Hxx = sp.hessian(hj, self.x_syms)  # (nx, nx)
            H_list.append(Hxx)

        Hh_xx = sp.Array(H_list)  # (ny, nx, nx)

        return Hh_xx
