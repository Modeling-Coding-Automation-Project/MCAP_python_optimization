import os
import inspect
import numpy as np
import sympy as sp
import importlib
from dataclasses import dataclass

from external_libraries.MCAP_python_control.python_control.control_deploy import ControlDeploy
from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

STATE_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_state_function.py"
MEASUREMENT_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_measurement_function.py"

STATE_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_state_jacobian_x.py"
STATE_JACOBIAN_U_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_state_jacobian_u.py"
MEASUREMENT_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_measurement_jacobian_x.py"

HESSIAN_F_XX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hessian_f_xx.py"
HESSIAN_F_XU_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hessian_f_xu.py"
HESSIAN_F_UX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hessian_f_ux.py"
HESSIAN_F_UU_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hessian_f_uu.py"
HESSIAN_H_XX_NUMPY_CODE_FILE_NAME_SUFFIX = "sqp_hessian_h_xx.py"


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

        # Precompute Jacobians
        self.A_matrix = self.f.jacobian(self.x_syms)  # df/dx
        self.B_matrix = self.f.jacobian(self.u_syms)  # df/du
        self.C_matrix = self.h.jacobian(self.x_syms)  # dh/dx

        # Precompute Hessians
        self.Hf_xx_matrix, self.Hf_xu_matrix, self.Hf_ux_matrix, self.Hf_uu_matrix = \
            self._stack_hessians_for_f()
        self.Hh_xx_matrix = self._stack_hessians_for_h()

        # create code files
        self.state_function_code_file_name, \
            self.measurement_function_code_file_name = \
            self.create_state_measurement_equation_numpy_code(
                file_name_without_ext=caller_file_name_without_ext)

        self.state_jacobian_x_code_file_name, \
            self.state_jacobian_u_code_file_name, \
            self.measurement_jacobian_x_code_file_name = \
            self.create_jacobians_numpy_code(
                file_name_without_ext=caller_file_name_without_ext)

        self.hf_xx_code_file_name, \
            self.hf_xu_code_file_name, \
            self.hf_ux_code_file_name, \
            self.hf_uu_code_file_name, \
            self.hh_xx_code_file_name = \
            self.create_hessian_numpy_code(
                file_name_without_ext=caller_file_name_without_ext)

        # state function module
        self.state_function_code_file_function = self.get_function_caller_from_python_file(
            self.state_function_code_file_name)

        # measurement function module
        self.measurement_function_code_file_function = self.get_function_caller_from_python_file(
            self.measurement_function_code_file_name)

        # jacobian modules
        self.state_jacobian_x_code_file_function = self.get_function_caller_from_python_file(
            self.state_jacobian_x_code_file_name)
        self.state_jacobian_u_code_file_function = self.get_function_caller_from_python_file(
            self.state_jacobian_u_code_file_name)
        self.measurement_jacobian_x_code_file_function = self.get_function_caller_from_python_file(
            self.measurement_jacobian_x_code_file_name)

        # hessian modules
        self.hf_xx_code_file_function = \
            self.get_function_caller_from_python_file(
                self.hf_xx_code_file_name)
        self.hf_xu_code_file_function = \
            self.get_function_caller_from_python_file(
                self.hf_xu_code_file_name)
        self.hf_ux_code_file_function = \
            self.get_function_caller_from_python_file(
                self.hf_ux_code_file_name)
        self.hf_uu_code_file_function = \
            self.get_function_caller_from_python_file(
                self.hf_uu_code_file_name)
        self.hh_xx_code_file_function = \
            self.get_function_caller_from_python_file(
                self.hh_xx_code_file_name)

        self.state_space_parameters = None
        self.reference_trajectory = None

    def create_state_measurement_equation_numpy_code(
            self, file_name_without_ext: str = None):
        state_function_code_file_name = STATE_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX
        measurement_function_code_file_name = MEASUREMENT_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX

        if file_name_without_ext is not None:
            state_function_code_file_name = file_name_without_ext + \
                "_" + state_function_code_file_name
            measurement_function_code_file_name = file_name_without_ext + \
                "_" + measurement_function_code_file_name

        # write code
        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.f,
            sym_object_name=os.path.splitext(state_function_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.h,
            sym_object_name=os.path.splitext(
                measurement_function_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        return state_function_code_file_name, \
            measurement_function_code_file_name

    def create_jacobians_numpy_code(
            self, file_name_without_ext: str = None):

        state_jacobian_x_code_file_name = STATE_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX
        state_jacobian_u_code_file_name = STATE_JACOBIAN_U_NUMPY_CODE_FILE_NAME_SUFFIX
        measurement_jacobian_x_code_file_name = MEASUREMENT_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX

        if file_name_without_ext is not None:
            state_jacobian_x_code_file_name = file_name_without_ext + \
                "_" + state_jacobian_x_code_file_name
            state_jacobian_u_code_file_name = file_name_without_ext + \
                "_" + state_jacobian_u_code_file_name
            measurement_jacobian_x_code_file_name = file_name_without_ext + \
                "_" + measurement_jacobian_x_code_file_name

        # write code
        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.A_matrix,
            sym_object_name=os.path.splitext(
                state_jacobian_x_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.B_matrix,
            sym_object_name=os.path.splitext(
                state_jacobian_u_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.C_matrix,
            sym_object_name=os.path.splitext(
                measurement_jacobian_x_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        return state_jacobian_x_code_file_name, \
            state_jacobian_u_code_file_name, \
            measurement_jacobian_x_code_file_name

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

        hf_xx_code_file_name = HESSIAN_F_XX_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_xu_code_file_name = HESSIAN_F_XU_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_ux_code_file_name = HESSIAN_F_UX_NUMPY_CODE_FILE_NAME_SUFFIX
        hf_uu_code_file_name = HESSIAN_F_UU_NUMPY_CODE_FILE_NAME_SUFFIX
        hh_xx_code_file_name = HESSIAN_H_XX_NUMPY_CODE_FILE_NAME_SUFFIX

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

    def get_function_caller_from_python_file(self, file_name: str):
        module_name = os.path.splitext(os.path.basename(file_name))[0]
        module = importlib.import_module(module_name)

        return getattr(module, 'function', None)

    def l_xx(self, x, u):
        return 2 * self.Qx

    def l_uu(self, x, u):
        return 2 * self.R

    def l_xu(self, x, u):
        return np.zeros((self.nx, self.nu))

    def l_ux(self, x, u):
        return np.zeros((self.nu, self.nx))

    def calculate_state_function(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        X_next = self.state_function_code_file_function(X, U, Parameters)

        return X_next.reshape((self.nx,))

    def calculate_measurement_function(
            self,
            X: np.ndarray,
            Parameters
    ) -> np.ndarray:

        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))

        Y = self.measurement_function_code_file_function(X, U, Parameters)

        return Y.reshape((self.ny,))

    def calculate_state_jacobian_x(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        A = self.state_jacobian_x_code_file_function(X, U, Parameters)

        return A.reshape((self.nx, self.nx))

    def calculate_state_jacobian_u(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        B = self.state_jacobian_u_code_file_function(X, U, Parameters)

        return B.reshape((self.nx, self.nu))

    def calculate_measurement_jacobian_x(
            self,
            X: np.ndarray,
            Parameters
    ) -> np.ndarray:

        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))

        C = self.measurement_jacobian_x_code_file_function(
            X, U, Parameters
        )

        return C.reshape((self.ny, self.nx))

    def fx_xx_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dX: np.ndarray
    ) -> np.ndarray:
        # sum_i lam_i * ( d^2 f_i / dx^2 @ dx )

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        out = np.zeros(self.nx, dtype=float)

        Hf_xx = self.hf_xx_code_file_function(
            X, U, Parameters)

        for i in range(self.nx):
            for j in range(self.nx):
                acc = 0.0
                for k in range(self.nx):
                    acc += Hf_xx[i * self.nx + j, k] * dX[k]
                out[j] += lam_next[i] * acc

        return out

    def fx_xu_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dU: np.ndarray
    ) -> np.ndarray:
        # sum_i lam_i * ( d^2 f_i / (dx du) @ du )

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        out = np.zeros(self.nx, dtype=float)

        Hf_xu = self.hf_xu_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):
                for j in range(self.nx):
                    acc = 0.0
                    for k in range(self.nu):
                        acc += Hf_xu[i * self.nx + j, k] * dU[k]
                    out[j] += lam_next[i] * acc

        return out

    def fu_xx_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dX: np.ndarray
    ) -> np.ndarray:
        # sum_i lam_i * ( d^2 f_i / (du dx) @ dx )

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        out = np.zeros(self.nu, dtype=float)

        Hf_ux = self.hf_ux_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):
                for k in range(self.nu):
                    acc = 0.0
                    for j in range(self.nx):
                        acc += Hf_ux[i * self.nu + k, j] * dX[j]
                    out[k] += lam_next[i] * acc

        return out

    def fu_uu_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dU: np.ndarray
    ) -> np.ndarray:
        # sum_i lam_i * ( d^2 f_i / du^2 @ du )

        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        out = np.zeros(self.nu, dtype=float)

        Hf_uu = self.hf_uu_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):            # f_i の添字
                for j in range(self.nu):        # 出力（入力方向）の添字
                    acc = 0.0
                    for k in range(self.nu):    # du を掛ける列の添字（入力方向）
                        acc += Hf_uu[i * self.nu + j, k] * dU[k]
                    out[j] += lam_next[i] * acc

        return out

    def hxx_lambda_contract(
            self,
            X: np.ndarray,
            Parameters,
            w: np.ndarray,
            dX: np.ndarray
    ) -> np.ndarray:
        # sum_i w_i * ( d^2 h_i / dx^2 @ dx )

        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))

        out = np.zeros(self.nx, dtype=float)

        Hh_xx = self.hh_xx_code_file_function(
            X, U, Parameters)

        for i in range(self.ny):
            for j in range(self.nx):
                acc = 0.0
                for k in range(self.nx):
                    acc += Hh_xx[i * self.nx + j, k] * dX[k]
                out[j] += w[i] * acc

        return out

    def simulate_trajectory(
        self,
        X_initial: np.ndarray,
        U: np.ndarray,
        Parameters
    ):
        X = np.zeros((self.Np + 1, self.nx))
        X[0] = X_initial
        for k in range(self.Np):
            X[k + 1] = self.calculate_state_function(
                X[k], U[k], Parameters)

        return X

    def compute_cost_and_gradient(
            self,
            X_initial: np.ndarray,
            U: np.ndarray
    ):
        # This function will be called from SQP solver,
        # thus Parameters and Reference trajectory must be changed beforehand.

        X = self.simulate_trajectory(X_initial, U, self.state_space_parameters)
        Y = np.zeros((X.shape[0], self.Qy.shape[0]))
        for k in range(X.shape[0]):
            Y[k] = self.calculate_measurement_function(
                X[k], self.state_space_parameters)

        J = 0.0
        for k in range(self.Np):
            e_y_r = Y[k] - self.reference_trajectory[k]
            J += X[k] @ self.Qx @ X[k] + \
                e_y_r @ self.Qy @ e_y_r + U[k] @ self.R @ U[k]

        eN_y_r = Y[self.Np] - self.reference_trajectory[self.Np]
        J += X[self.Np] @ self.Px @ X[self.Np] + eN_y_r @ self.Py @ eN_y_r

        # terminal adjoint
        C_N = self.calculate_measurement_jacobian_x(
            X[self.Np], self.state_space_parameters)
        lam_next = (2 * self.Px) @ X[self.Np] + C_N.T @ (2 * self.Py @ eN_y_r)

        grad = np.zeros_like(U)
        for k in reversed(range(self.Np)):
            Cx_k = self.calculate_measurement_jacobian_x(
                X[k], self.state_space_parameters)
            ek_y = Y[k] - self.reference_trajectory[k]

            A_k = self.calculate_state_jacobian_x(
                X[k], U[k], self.state_space_parameters)
            B_k = self.calculate_state_jacobian_u(
                X[k], U[k], self.state_space_parameters)

            grad[k] = 2 * self.R @ U[k] + B_k.T @ lam_next

            lam_next = 2 * self.Qx @ X[k] + 2 * \
                Cx_k.T @ (self.Qy @ ek_y) + A_k.T @ lam_next

        return J, grad
