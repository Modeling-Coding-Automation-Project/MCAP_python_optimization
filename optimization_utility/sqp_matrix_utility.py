import os
import inspect
import numpy as np
import sympy as sp

from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

HESSIAN_HF_XX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hf_xx.py"
HESSIAN_HF_XU_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hf_xu.py"
HESSIAN_HF_UX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hf_ux.py"
HESSIAN_HF_UU_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hf_uu.py"
HESSIAN_HH_XX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hh_xx.py"


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
            Py: np.ndarray = None,
            caller_file_name: str = None
    ):
        if caller_file_name is None:
            # % inspect arguments
            # Get the caller's frame
            frame = inspect.currentframe().f_back
            # Get the caller's local variables
            caller_locals = frame.f_locals
            for _, value in caller_locals.items():
                if value is x_syms:
                    break
            # Get the caller's file name
            caller_file_full_path = frame.f_code.co_filename
            caller_file_name = os.path.basename(caller_file_full_path)
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]
        else:
            caller_file_name_without_ext = os.path.splitext(caller_file_name)[
                0]

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
        self.Hf_xx_matrix, self.Hf_xu_matrix, self.Hf_ux_matrix, self.Hf_uu_matrix = \
            self._stack_hessians_for_f()
        self.Hh_xx_matrix = self._stack_hessians_for_h()

        self.create_hessian_numpy_code(
            file_name_without_ext=caller_file_name_without_ext)

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

        Hf_xx_matrix = sp.zeros(self.nx * self.nx, self.nx)
        Hf_xu_matrix = sp.zeros(self.nx * self.nx, self.nu)
        Hf_ux_matrix = sp.zeros(self.nx * self.nu, self.nx)
        Hf_uu_matrix = sp.zeros(self.nx * self.nu, self.nu)

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

            Hf_xx_matrix[i * nx:(i + 1) * nx, :] = Hxx
            Hf_xu_matrix[i * nx:(i + 1) * nx, :] = Hxu
            Hf_ux_matrix[i * nu:(i + 1) * nu, :] = Hux
            Hf_uu_matrix[i * nu:(i + 1) * nu, :] = Huu

        return Hf_xx_matrix, Hf_xu_matrix, Hf_ux_matrix, Hf_uu_matrix

    def _stack_hessians_for_h(self):
        """
        returns:
        Hh_xx: Array shape (ny, nx, nx)
        """
        ny = self.h.shape[0]
        Hh_xx_matrix = sp.zeros(self.ny * self.nx, self.nx)

        for j in range(ny):
            hj = self.h[j, 0]
            Hxx = sp.hessian(hj, self.x_syms)  # (nx, nx)

            Hh_xx_matrix[j * self.nx:(j + 1) * self.nx, :] = Hxx

        return Hh_xx_matrix

    def create_hessian_numpy_code(
            self, file_name_without_ext: str = None):

        hf_xx_code_file_name = HESSIAN_HF_XX_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_xu_code_file_name = HESSIAN_HF_XU_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_ux_code_file_name = HESSIAN_HF_UX_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_uu_code_file_name = HESSIAN_HF_UU_NUMPY_CODE_FILE_NAME_SUFFIX
        hh_xx_code_file_name = HESSIAN_HH_XX_NUMPY_CODE_FILE_NAME_SUFFIX

        if file_name_without_ext is not None:
            hf_xx_code_file_name = file_name_without_ext + \
                "_" + hf_xx_code_file_name
            hf_xu_code_file_name = file_name_without_ext + \
                "_" + hf_xu_code_file_name
            hf_ux_code_file_name = file_name_without_ext + \
                "_" + hf_ux_code_file_name
            hf_uu_code_file_name = file_name_without_ext + \
                "_" + hf_uu_code_file_name
            hh_xx_code_file_name = file_name_without_ext + \
                "_" + hh_xx_code_file_name

        # write code
        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.Hf_xx_matrix,
            sym_object_name=os.path.splitext(hf_xx_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.Hf_xu_matrix,
            sym_object_name=os.path.splitext(hf_xu_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.Hf_ux_matrix,
            sym_object_name=os.path.splitext(hf_ux_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.Hf_uu_matrix,
            sym_object_name=os.path.splitext(hf_uu_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.Hh_xx_matrix,
            sym_object_name=os.path.splitext(hh_xx_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        return hf_xx_code_file_name, \
            hf_xu_code_file_name, \
            hf_ux_code_file_name, \
            hf_uu_code_file_name, \
            hh_xx_code_file_name


# nx = Hf_xx_numpy.shape[1]          # 状態次元
# out = np.zeros(nx, dtype=float)    # 結果ベクトル (nx,)

# for i in range(nx):                # f_i の添字
#     for j in range(nx):            # 出力成分の添字
#         acc = 0.0
#         for k in range(nx):        # dx を掛ける列の添字
#             acc += Hf_xx_numpy[i, j, k] * dx[k]
#         out[j] += lam_next[i] * acc
