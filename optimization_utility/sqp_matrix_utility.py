"""
File: sqp_matrix_utility.py

A utility module for constructing and handling matrices and functions
used in Sequential Quadratic Programming (SQP) for Nonlinear Model Predictive Control (NMPC).
It includes functionality to extract parameters from state-space equations,
generate code for state and measurement functions,
compute Jacobians and Hessians, and evaluate cost functions and their gradients.
"""
import os
import inspect
import numpy as np
import sympy as sp
import importlib

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

Y_MIN_MAX_RHO_FACTOR_DEFAULT = 1.0e2

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


def extract_parameters_from_state_space_equations(
        f: sp.Matrix,
        h: sp.Matrix,
        x_syms: sp.Matrix,
        u_syms: sp.Matrix
):
    """
    Extracts parameters from the state-space equations f and h,
    given the state symbols x_syms and input symbols u_syms.
    Parameters are defined as free symbols in f or h that are not
    part of x_syms or u_syms.
    Returns a sorted list of parameter symbols.
    """
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
    """
    A class to handle state-space equations, cost matrices,
    and related computations for SQP-based NMPC.

    Attributes:
        x_syms (sp.Matrix): Symbolic state variables.
        u_syms (sp.Matrix): Symbolic input variables.
        f (sp.Matrix): State equation vector.
        h (sp.Matrix): Measurement equation vector.
        Parameters (list): List of symbolic parameters extracted from f and h.
        nx (int): Number of states.
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        Np (int): Prediction horizon length.
        Qx (np.ndarray): State cost weight matrix.
        Qy (np.ndarray): Output cost weight matrix.
        R (np.ndarray): Input cost weight matrix.
        Px (np.ndarray): Terminal state cost weight matrix.
        Py (np.ndarray): Terminal output cost weight matrix.
        U_min_matrix (np.ndarray): Minimum input constraint matrix over horizon.
        U_max_matrix (np.ndarray): Maximum input constraint matrix over horizon.
        Y_min_matrix (np.ndarray): Minimum output constraint matrix over horizon.
        Y_max_matrix (np.ndarray): Maximum output constraint matrix over horizon.
        Y_min_max_rho (float): Penalty factor for output constraints.
        A_matrix (sp.Matrix): Jacobian of f w.r.t. x.
        B_matrix (sp.Matrix): Jacobian of f w.r.t. u.
        C_matrix (sp.Matrix): Jacobian of h w.r.t. x.
        Hf_xx_matrix, Hf_xu_matrix, Hf_ux_matrix, Hf_uu_matrix: Hessians of f.
        Hh_xx_matrix: Hessians of h.
        state_function_code_file_name (str): Generated state function code file name.
        measurement_function_code_file_name (str): Generated measurement function code file name.
        state_jacobian_x_code_file_name (str): Generated state Jacobian w.r.t. x code file name.
        state_jacobian_u_code_file_name (str): Generated state Jacobian w.r.t. u code file name.
        measurement_jacobian_x_code_file_name (str): Generated measurement Jacobian w.r.t. x code file name.
        hf_xx_code_file_name, hf_xu_code_file_name,
          hf_ux_code_file_name, hf_uu_code_file_name: Generated Hessian code file names for f.
        hh_xx_code_file_name: Generated Hessian code file name for h.
        state_function_code_file_function (callable): Callable for
    """

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
            U_min: np.ndarray = None,
            U_max: np.ndarray = None,
            Y_min: np.ndarray = None,
            Y_max: np.ndarray = None,
            Y_min_max_rho: float = Y_MIN_MAX_RHO_FACTOR_DEFAULT,
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

        self.Parameters = extract_parameters_from_state_space_equations(
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

        self.U_min_matrix = np.ones((self.nu, self.Np)) * -np.inf
        if U_min is not None:
            self.U_min_matrix = np.tile(U_min, (1, self.Np))

        self.U_max_matrix = np.ones((self.nu, self.Np)) * np.inf
        if U_max is not None:
            self.U_max_matrix = np.tile(U_max, (1, self.Np))

        self.Y_min_matrix = np.ones((self.ny, self.Np + 1)) * -np.inf
        if Y_min is not None:
            self.Y_min_matrix = np.tile(Y_min, (1, self.Np + 1))

        self.Y_max_matrix = np.ones((self.ny, self.Np + 1)) * np.inf
        if Y_max is not None:
            self.Y_max_matrix = np.tile(Y_max, (1, self.Np + 1))

        self.Y_min_max_rho = Y_min_max_rho

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

        self._Y_offset = None

    def set_U_min(self, U_min: np.ndarray):
        """
        Set the minimum input constraint matrix.
        Args:
            U_min (np.ndarray): Minimum input values of shape (nu, 1).
        """
        if U_min.shape[0] != self.nu or U_min.shape[1] != 1:
            raise ValueError(
                f"U_min must have shape ({self.nu}, 1), got {U_min.shape}")
        self.U_min_matrix = np.tile(U_min, (1, self.Np))

    def set_U_max(self, U_max: np.ndarray):
        """
        Set the maximum input constraint matrix.
        Args:
            U_max (np.ndarray): Maximum input values of shape (nu, 1).
        """
        if U_max.shape[0] != self.nu or U_max.shape[1] != 1:
            raise ValueError(
                f"U_max must have shape ({self.nu}, 1), got {U_max.shape}")
        self.U_max_matrix = np.tile(U_max, (1, self.Np))

    def set_Y_min(self, Y_min: np.ndarray):
        """
        Set the minimum output constraint matrix.
        Args:
            Y_min (np.ndarray): Minimum output values of shape (ny, 1).
        """
        if Y_min.shape[0] != self.ny or Y_min.shape[1] != 1:
            raise ValueError(
                f"Y_min must have shape ({self.ny}, 1), got {Y_min.shape}")
        self.Y_min_matrix = np.tile(Y_min, (1, self.Np + 1))

    def set_Y_max(self, Y_max: np.ndarray):
        """
        Set the maximum output constraint matrix.
        Args:
            Y_max (np.ndarray): Maximum output values of shape (ny, 1).
        """
        if Y_max.shape[0] != self.ny or Y_max.shape[1] != 1:
            raise ValueError(
                f"Y_max must have shape ({self.ny}, 1), got {Y_max.shape}")
        self.Y_max_matrix = np.tile(Y_max, (1, self.Np + 1))

    def set_Y_offset(self, Y_offset: np.ndarray):
        """
        Set the output offset matrix.
        Args:
            Y_offset (np.ndarray): Output offset values of shape (ny, 1).
        """
        if Y_offset.shape[0] != self.ny or Y_offset.shape[1] != 1:
            raise ValueError(
                f"Y_offset must have shape ({self.ny}, 1), got {Y_offset.shape}")

        self._Y_offset = Y_offset

    def create_state_measurement_equation_numpy_code(
            self,
            file_name_without_ext: str = None
    ):
        """
        Create numpy code files for state and measurement equations.
        Args:
            file_name_without_ext (str): The file name without extension to use for generated code files.
        Returns:
            state_function_code_file_name (str): The generated state function code file name.
            measurement_function_code_file_name (str): The generated measurement function code file name.
        """
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
        self,
        file_name_without_ext: str = None
    ):
        """
        Create numpy code files for Jacobians of state and measurement equations.
        Args:
            file_name_without_ext (str): The file name without extension to use for generated code files.
        Returns:
            state_jacobian_x_code_file_name (str): The generated state Jacobian w.r.t. x code file name.
            state_jacobian_u_code_file_name (str): The generated state Jacobian w.r.t. u code file name.
            measurement_jacobian_x_code_file_name (str): The generated measurement Jacobian w.r.t. x code file name.
        """
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
        self,
        file_name_without_ext: str = None
    ):
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

    def get_function_caller_from_python_file(
            self,
            file_name: str
    ):
        """
        Dynamically imports a Python file and retrieves the 'function' callable.
        Args:
            file_name (str): The Python file name to import.
        Returns:
            function (callable): The 'function' callable defined in the imported file.
        """

        module_name = os.path.splitext(os.path.basename(file_name))[0]
        module = importlib.import_module(module_name)

        return getattr(module, 'function', None)

    def l_xx(self, x, u):
        """
        Computes the second derivative (Hessian)
        of the objective function with respect to x.

        Args:
            x: The current value of the variable x.
            u: The current value of the variable u.

        Returns:
            The Hessian matrix of the objective function
            with respect to x, calculated as 2 * self.Qx.
        """
        return 2 * self.Qx

    def l_uu(self, x, u):
        """
        Computes the second derivative (Hessian)
          of the cost function with respect to the control variable `u`.
        Parameters
        ----------
        x : array-like
            The state variable(s).
        u : array-like
            The control variable(s).
        Returns
        -------
        ndarray or scalar
            The Hessian of the cost function with respect to `u`,
              which is 2 times the control weighting matrix `R`.
        """

        return 2 * self.R

    def l_xu(self, x, u):
        """
        Computes the cross partial derivatives
        of the Lagrangian with respect to state variables
          `x` and control variables `u`.
        Parameters
        ----------
        x : np.ndarray
            State variables array.
        u : np.ndarray
            Control variables array.
        Returns
        -------
        np.ndarray
            A zero matrix of shape (self.nx, self.nu)
              representing the cross partial derivatives.
        """

        return np.zeros((self.nx, self.nu))

    def l_ux(self, x, u):
        """
        Computes the cross partial derivatives
          of the Lagrangian with respect to control variables (u)
          and state variables (x).
        Parameters
        ----------
        x : array-like
            State variables.
        u : array-like
            Control variables.
        Returns
        -------
        numpy.ndarray
            A zero matrix of shape (self.nu, self.nx),
              representing the cross partial derivatives.
        """

        return np.zeros((self.nu, self.nx))

    def calculate_state_function(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:
        """
        Calculates the next state vector using the provided state function.

        Args:
            X (np.ndarray): Current state vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control input vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the state function.

        Returns:
            np.ndarray: Next state vector reshaped to (nx, 1).
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        X_next = self.state_function_code_file_function(X, U, Parameters)

        return X_next.reshape((self.nx, 1))

    def calculate_measurement_function(
            self,
            X: np.ndarray,
            Parameters
    ) -> np.ndarray:
        """
        Calculates the measurement function output for given state and parameters.

        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            Parameters: Additional parameters required by the measurement function.

        Returns:
            np.ndarray: Measurement output vector of shape (ny, 1).
        """
        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))

        Y = self.measurement_function_code_file_function(X, U, Parameters)

        return Y.reshape((self.ny, 1))

    def calculate_state_jacobian_x(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:
        """
        Calculates the Jacobian matrix of the state with respect to the state variables.

        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control input vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the Jacobian calculation function.

        Returns:
            np.ndarray: The state Jacobian matrix with respect to the state variables,
                        reshaped to (nx, nx).
        """
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
        """
        Calculates the Jacobian matrix of the system states with respect to the control inputs.

        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control input vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the Jacobian computation.

        Returns:
            np.ndarray: The Jacobian matrix of shape (nx, nu),
              representing the partial derivatives of the states
              with respect to the control inputs.
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))

        B = self.state_jacobian_u_code_file_function(X, U, Parameters)

        return B.reshape((self.nx, self.nu))

    def calculate_measurement_jacobian_x(
            self,
            X: np.ndarray,
            Parameters
    ) -> np.ndarray:
        """
        Calculates the measurement Jacobian matrix with respect to the state vector X.

        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            Parameters: Additional parameters required by the measurement Jacobian function.

        Returns:
            np.ndarray: Measurement Jacobian matrix of shape (ny, nx).
        """
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
        """
        Computes the contraction of the second derivative
          of the function f with respect to x (Hessian),
        weighted by the lambda vector and contracted with the direction vector dX.
        Specifically, for each i in range(nx), it calculates:
            sum_i lam_next[i] * (Hf_xx[i * nx + j, k] * dX[k])
        where Hf_xx is the Hessian matrix of f with respect to x,
          evaluated at (X, U, Parameters).
        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the Hessian function.
            lam_next (np.ndarray): Lambda vector of shape (nx,) or (nx, 1) used for weighting.
            dX (np.ndarray): Direction vector of shape (nx,) or (nx, 1) for contraction.
        Returns:
            np.ndarray: Resulting contracted vector of shape (nx, 1).
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))
        lam_next = lam_next.reshape((self.nx, 1))
        dX = dX.reshape((self.nx, 1))

        out = np.zeros((self.nx, 1), dtype=float)

        Hf_xx = self.hf_xx_code_file_function(
            X, U, Parameters)

        for i in range(self.nx):
            for j in range(self.nx):
                acc = 0.0
                for k in range(self.nx):
                    acc += Hf_xx[i * self.nx + j, k] * dX[k, 0]
                out[j, 0] += lam_next[i] * acc

        return out

    def fx_xu_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dU: np.ndarray
    ) -> np.ndarray:
        """
        Computes the contraction of the Lagrange multipliers
          with the mixed second derivatives of the objective function
        with respect to state and control variables, and the control increment vector.
        Specifically, calculates:
            sum_i lam_next[i] * (sum_j sum_k Hf_xu[i * nx + j, k] * dU[k])
              for each state variable j,
        where Hf_xu is the mixed second derivative matrix (d^2 f / dx du).
        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the objective function.
            lam_next (np.ndarray): Lagrange multipliers vector of shape (nx,) or (nx, 1).
            dU (np.ndarray): Control increment vector of shape (nu,) or (nu, 1).
        Returns:
            np.ndarray: Resulting contracted vector of shape (nx, 1).
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))
        lam_next = lam_next.reshape((self.nx, 1))
        dU = dU.reshape((self.nu, 1))

        out = np.zeros((self.nx, 1), dtype=float)

        Hf_xu = self.hf_xu_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):
                for j in range(self.nx):
                    acc = 0.0
                    for k in range(self.nu):
                        acc += Hf_xu[i * self.nx + j, k] * dU[k, 0]
                    out[j, 0] += lam_next[i] * acc

        return out

    def fu_xx_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dX: np.ndarray
    ) -> np.ndarray:
        """
        Computes the contraction of the second mixed partial derivatives
          of a function with respect to control and state variables,
        weighted by the Lagrange multipliers and a direction vector.

        Specifically, calculates the sum over i of lam_next[i] * (d^2 f_i / (du dx) @ dX), where:
            - d^2 f_i / (du dx) is the mixed Hessian of
              f_i with respect to control (u) and state (x),
            - lam_next is the vector of Lagrange multipliers,
            - dX is the direction vector for state variables.

        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control vector of shape (nu,) or (nu, 1).
            Parameters: Additional parameters required by the Hessian function.
            lam_next (np.ndarray): Lagrange multipliers vector of shape (nx,) or (nx, 1).
            dX (np.ndarray): Direction vector for state variables of shape (nx,) or (nx, 1).

        Returns:
            np.ndarray: Resulting contracted vector of shape (nu, 1).
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))
        lam_next = lam_next.reshape((self.nx, 1))
        dX = dX.reshape((self.nx, 1))

        out = np.zeros((self.nu, 1), dtype=float)

        Hf_ux = self.hf_ux_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):
                for k in range(self.nu):
                    acc = 0.0
                    for j in range(self.nx):
                        acc += Hf_ux[i * self.nu + k, j] * dX[j, 0]
                    out[k, 0] += lam_next[i] * acc

        return out

    def fu_uu_lambda_contract(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters,
            lam_next: np.ndarray,
            dU: np.ndarray
    ) -> np.ndarray:
        """
        Computes the contraction of the second derivative of
          the function f with respect to U (control variables),
        weighted by the lambda multipliers and the direction dU.

        Specifically, calculates the sum over i of lam_next[i] * (d^2 f_i / du^2 @ dU), where:
            - Hf_uu is the Hessian of f with respect to U, for each state variable i.
            - lam_next is the vector of multipliers for each state variable.
            - dU is the direction vector for control variables.

        Args:
            X (np.ndarray): State vector, shape (nx, 1) or (nx,).
            U (np.ndarray): Control vector, shape (nu, 1) or (nu,).
            Parameters: Additional parameters required by hf_uu_code_file_function.
            lam_next (np.ndarray): Multipliers for each state variable, shape (nx, 1) or (nx,).
            dU (np.ndarray): Direction vector for control variables, shape (nu, 1) or (nu,).

        Returns:
            np.ndarray: Resulting contracted vector, shape (nu, 1).
        """
        X = X.reshape((self.nx, 1))
        U = U.reshape((self.nu, 1))
        lam_next = lam_next.reshape((self.nx, 1))
        dU = dU.reshape((self.nu, 1))

        out = np.zeros((self.nu, 1), dtype=float)

        Hf_uu = self.hf_uu_code_file_function(
            X, U, Parameters)

        if self.nu == 0:
            pass
        else:
            for i in range(self.nx):
                for j in range(self.nu):
                    acc = 0.0
                    for k in range(self.nu):
                        acc += Hf_uu[i * self.nu + j, k] * dU[k, 0]
                    out[j, 0] += lam_next[i] * acc

        return out

    def hxx_lambda_contract(
            self,
            X: np.ndarray,
            Parameters,
            w: np.ndarray,
            dX: np.ndarray
    ) -> np.ndarray:
        """
        Computes the weighted contraction of the second derivative
          (Hessian) of the function h with respect to x,
        contracted along the direction dX, and summed over all outputs with weights w.
        Specifically, for each output dimension i, it calculates:
            sum_i w_i * (d^2 h_i / dx^2 @ dX)
        where:
            - X: State vector of shape (nx,)
            - Parameters: Additional parameters required by the Hessian function
            - w: Weight vector of shape (ny,)
            - dX: Direction vector for contraction of shape (nx,)
        Returns:
            np.ndarray: Resulting contracted vector of shape (nx, 1)
        """
        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))
        w = w.reshape((self.ny, 1))
        dX = dX.reshape((self.nx, 1))

        out = np.zeros((self.nx, 1), dtype=float)

        Hh_xx = self.hh_xx_code_file_function(
            X, U, Parameters)

        for i in range(self.ny):
            for j in range(self.nx):
                acc = 0.0
                for k in range(self.nx):
                    acc += Hh_xx[i * self.nx + j, k] * dX[k, 0]
                out[j, 0] += w[i] * acc

        return out

    def simulate_trajectory(
        self,
        X_initial: np.ndarray,
        U: np.ndarray,
        Parameters
    ):
        """
        Simulates the trajectory of the system over a prediction horizon.
        Args:
            X_initial (np.ndarray): Initial state vector of the system (shape: [nx,]).
            U (np.ndarray): Control input sequence over the prediction horizon
              (shape: [nu, Np]).
            Parameters: Additional parameters required by the state function.
        Returns:
            np.ndarray: Simulated state trajectory over the prediction horizon
              (shape: [nx, Np + 1]).
        """
        X = np.zeros((self.nx, self.Np + 1))
        X[:, 0] = X_initial.flatten()
        for k in range(self.Np):
            X[:, k + 1] = self.calculate_state_function(
                X[:, k], U[:, k], Parameters).flatten()

        return X

    def calculate_Y_limit_penalty(self, Y: np.ndarray):
        """
        Calculates the penalty matrix Y_limit_penalty for the given output matrix
          Y based on minimum and maximum constraints.
        For each element in Y, the penalty is computed as follows:
            - If Y[i, j] is less than the corresponding minimum constraint
              (self.Y_min_matrix[i, j]),
              the penalty is Y[i, j] - self.Y_min_matrix[i, j].
            - If Y[i, j] is greater than the corresponding maximum constraint
              (self.Y_max_matrix[i, j]),
              the penalty is Y[i, j] - self.Y_max_matrix[i, j].
            - Otherwise, the penalty is 0.
        Args:
            Y (np.ndarray): Output matrix of shape (self.ny, self.Np + 1)
              to be checked against constraints.
        Returns:
            np.ndarray: Penalty matrix of the same shape as Y,
              containing the calculated penalties.
        """
        Y_limit_penalty = np.zeros((self.ny, self.Np + 1))
        for i in range(self.ny):
            for j in range(self.Np + 1):
                if Y[i, j] < self.Y_min_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_min_matrix[i, j]
                elif Y[i, j] > self.Y_max_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_max_matrix[i, j]

        return Y_limit_penalty

    def calculate_Y_limit_penalty_and_active(self, Y: np.ndarray):
        """
        Calculates the penalty and activation matrices for output variable limits.
        For each output variable and prediction step,
          this method checks if the value in `Y` violates
        the corresponding minimum (`Y_min_matrix`) or
          maximum (`Y_max_matrix`) limits. If a violation occurs,
        a penalty is computed as the difference between `Y` and
          the violated limit, and the corresponding
        entry in the activation matrix is set to 1.0.
        Args:
            Y (np.ndarray): Array of output variable values with shape (ny, Np + 1).
        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - Y_limit_penalty (np.ndarray): Matrix of penalties for limit violations,
                  shape (ny, Np + 1).
                - Y_limit_active (np.ndarray): Matrix indicating active limit violations
                  (1.0 if violated, 0.0 otherwise), shape (ny, Np + 1).
        """
        Y_limit_penalty = np.zeros((self.ny, self.Np + 1))
        Y_limit_active = np.zeros((self.ny, self.Np + 1))

        for i in range(self.ny):
            for j in range(self.Np + 1):
                if Y[i, j] < self.Y_min_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_min_matrix[i, j]
                    Y_limit_active[i, j] = 1.0
                elif Y[i, j] > self.Y_max_matrix[i, j]:
                    Y_limit_penalty[i, j] = Y[i, j] - self.Y_max_matrix[i, j]
                    Y_limit_active[i, j] = 1.0

        return Y_limit_penalty, Y_limit_active

    def compute_cost_and_gradient(
            self,
            X_initial: np.ndarray,
            U: np.ndarray
    ):
        """
        Computes the cost function value and its gradient with respect to the control input sequence
          for a given initial state and control trajectory.
        This function simulates the system trajectory using the provided initial state and control inputs,
          evaluates the cost function over the prediction horizon,
          and calculates the gradient of the cost with respect to the control inputs
          using adjoint (backpropagation) methods.
        Parameters
        ----------
        X_initial : np.ndarray
            The initial state vector of the system.
        U : np.ndarray
            The control input sequence over the prediction horizon, with shape (nu, Np).
        Returns
        -------
        J : float
            The computed cost function value for the given trajectory and control inputs.
        gradient : np.ndarray
            The gradient of the cost function with respect to the control inputs, with shape matching U.
        Notes
        -----
        - The function assumes that system parameters and reference trajectory are set prior to invocation.
        - The cost function includes state, output, and control penalties over the horizon, as well as terminal penalties.
        - The gradient is computed using backward recursion with adjoint variables.
        """
        X = self.simulate_trajectory(X_initial, U, self.state_space_parameters)

        if self._Y_offset is None:
            Y = np.zeros((self.ny, self.Np + 1))
        else:
            Y = np.tile(self._Y_offset.reshape((self.ny, 1)), (1, self.Np + 1))

        for k in range(X.shape[0]):
            Y[:, k] += self.calculate_measurement_function(
                X[:, k], self.state_space_parameters).flatten()

        Y_limit_penalty = self.calculate_Y_limit_penalty(Y)

        J = 0.0
        for k in range(self.Np):
            e_y_r = Y[:, k] - self.reference_trajectory[:, k]
            J += X[:, k].T @ self.Qx @ X[:, k] + \
                e_y_r.T @ self.Qy @ e_y_r + U[:, k].T @ self.R @ U[:, k] + \
                self.Y_min_max_rho * \
                (Y_limit_penalty[:, k].T @ Y_limit_penalty[:, k])

        eN_y_r = Y[:, self.Np] - self.reference_trajectory[:, self.Np]
        J += X[:, self.Np].T @ self.Px @ X[:, self.Np] + \
            eN_y_r.T @ self.Py @ eN_y_r + \
            self.Y_min_max_rho * \
            (Y_limit_penalty[:, self.Np].T @ Y_limit_penalty[:, self.Np])

        # terminal adjoint
        C_N = self.calculate_measurement_jacobian_x(
            X[:, self.Np], self.state_space_parameters)
        lam_next = (2.0 * self.Px) @ X[:, self.Np] + \
            C_N.T @ (2.0 * self.Py @ eN_y_r +
                     2.0 * self.Y_min_max_rho * Y_limit_penalty[:, self.Np])

        gradient = np.zeros_like(U)
        for k in reversed(range(self.Np)):
            Cx_k = self.calculate_measurement_jacobian_x(
                X[:, k], self.state_space_parameters)
            ek_y = Y[:, k] - self.reference_trajectory[:, k]

            A_k = self.calculate_state_jacobian_x(
                X[:, k], U[:, k], self.state_space_parameters)
            B_k = self.calculate_state_jacobian_u(
                X[:, k], U[:, k], self.state_space_parameters)

            gradient[:, k] = 2.0 * self.R @ U[:, k] + B_k.T @ lam_next

            lam_next = 2.0 * self.Qx @ X[:, k] + 2.0 * \
                Cx_k.T @ (self.Qy @ ek_y +
                          2.0 * self.Y_min_max_rho * Y_limit_penalty[:, k]) + \
                A_k.T @ lam_next

        return J, gradient

    def hvp_analytic(
            self,
            X_initial: np.ndarray,
            U: np.ndarray,
            V: np.ndarray
    ):
        """
        Computes the Hessian-vector product (HVP) analytically
          for a trajectory optimization problem.
        This method performs a forward simulation of the system dynamics,
          computes first-order adjoint variables,
        propagates directional derivatives, and then calculates
          the backward second-order adjoint variables to
        obtain the HVP with respect to the control input direction V.
        Args:
            X_initial (np.ndarray): Initial state vector of shape (nx,).
            U (np.ndarray): Control input trajectory of shape (nu, Np).
            V (np.ndarray): Directional vector for control inputs of shape (nu, Np).
        Returns:
            np.ndarray: Hessian-vector product with respect to control inputs, shape (nu, Np).
        Notes:
            - The method assumes the existence of system dynamics,
              measurement functions, and their derivatives.
            - The computation involves both first- and second-order derivatives
              of the cost and system dynamics.
            - The returned value can be used for second-order optimization algorithms
              such as SQP or Newton-type methods.
        """
        # --- 1) forward states
        X = self.simulate_trajectory(X_initial, U, self.state_space_parameters)
        Y = np.zeros((self.ny, self.Np + 1))
        for k in range(self.Np + 1):
            Y[:, k] = self.calculate_measurement_function(
                X[:, k], self.state_space_parameters).flatten()
        yN = self.calculate_measurement_function(
            X[:, self.Np], self.state_space_parameters)

        eN_y = yN - self.reference_trajectory[:, self.Np].reshape(-1, 1)

        Y_limit_penalty, Y_limit_active = \
            self.calculate_Y_limit_penalty_and_active(Y)

        # --- 2) first-order adjoint (costate lambda) with output terms
        lam = np.zeros((self.nx, self.Np + 1))
        Cx_N = self.calculate_measurement_jacobian_x(
            X[:, self.Np], self.state_space_parameters)

        lam[:, self.Np] = 2.0 * self.Px @ X[:, self.Np] + \
            (Cx_N.T @ ((2.0 * self.Py @ eN_y).flatten() + 2.0 *
                       self.Y_min_max_rho * Y_limit_penalty[:, self.Np])
             ).flatten()

        for k in range(self.Np - 1, -1, -1):
            A_k = self.calculate_state_jacobian_x(
                X[:, k], U[:, k], self.state_space_parameters)
            Cx_k = self.calculate_measurement_jacobian_x(
                X[:, k], self.state_space_parameters)
            ek_y = Y[:, k] - self.reference_trajectory[:, k]

            lam[:, k] = 2.0 * self.Qx @ X[:, k] + \
                Cx_k.T @ (2.0 * self.Qy @ ek_y +
                          2.0 * self.Y_min_max_rho * Y_limit_penalty[:, k]) + \
                A_k.T @ lam[:, k + 1]

        # --- 3) forward directional state: delta_x ---
        dx = np.zeros((self.nx, self.Np + 1))
        for k in range(self.Np):
            A_k = self.calculate_state_jacobian_x(
                X[:, k], U[:, k], self.state_space_parameters)
            B_k = self.calculate_state_jacobian_u(
                X[:, k], U[:, k], self.state_space_parameters)
            dx[:, k + 1] = A_k @ dx[:, k] + B_k @ V[:, k]

        # --- 4) backward second-order adjoint ---
        d_lambda = np.zeros((self.nx, self.Np + 1))

        # Match the treatment of the terminal term phi_xx = l_xx(X_N,·) (currently 2P)
        # Additionally, contributions from pure second-order output and second derivatives of output
        l_xx_dx = self.l_xx(X[:, self.Np], None) @ dx[:, self.Np]
        CX_N_dx = Cx_N @ dx[:, self.Np]

        CX_N_T_Py_Cx_N_dx = Cx_N.T @ (2.0 * self.Py @ CX_N_dx)
        CX_N_T_penalty_CX_N_dx = Cx_N.T @ ((2.0 * self.Y_min_max_rho)
                                           * (Y_limit_active[:, self.Np] * CX_N_dx))

        Hxx_penalty_term_N = self.hxx_lambda_contract(
            X[:, self.Np], self.state_space_parameters,
            2.0 * self.Y_min_max_rho *
            Y_limit_penalty[:, self.Np], dx[:, self.Np]
        )

        d_lambda[:, self.Np] += \
            l_xx_dx.flatten() + \
            CX_N_T_Py_Cx_N_dx.flatten() + \
            Hxx_penalty_term_N.flatten()

        d_lambda[:, self.Np] += CX_N_T_penalty_CX_N_dx.flatten()

        Hu = np.zeros_like(U)
        for k in range(self.Np - 1, -1, -1):
            A_k = self.calculate_state_jacobian_x(
                X[:, k], U[:, k], self.state_space_parameters)
            B_k = self.calculate_state_jacobian_u(
                X[:, k], U[:, k], self.state_space_parameters)
            Cx_k = self.calculate_measurement_jacobian_x(
                X[:, k], self.state_space_parameters)
            ek_y = Y[:, k] - self.reference_trajectory[:, k]

            Cx_dx_k = Cx_k @ dx[:, k]
            term_Qy_GN = Cx_k.T @ (2.0 * self.Qy @ Cx_dx_k)
            term_Qy_hxx = self.hxx_lambda_contract(
                X[:, k], self.state_space_parameters,
                2.0 * self.Qy @ ek_y, dx[:, k]
            )

            term_penalty_GN = Cx_k.T @ ((2.0 * self.Y_min_max_rho)
                                        * (Y_limit_active[:, k] * Cx_dx_k))
            term_penalty_hxx = self.hxx_lambda_contract(
                X[:, k], self.state_space_parameters,
                2.0 * self.Y_min_max_rho * Y_limit_penalty[:, k], dx[:, k]
            )

            term_xx = self.fx_xx_lambda_contract(
                X[:, k], U[:, k], self.state_space_parameters, lam[:, k + 1], dx[:, k])
            term_xu = self.fx_xu_lambda_contract(
                X[:, k], U[:, k], self.state_space_parameters, lam[:, k + 1], V[:, k])

            d_lambda[:, k] = \
                (self.l_xx(X[:, k], U[:, k]) @ dx[:, k]).flatten() + \
                (self.l_xu(X[:, k], U[:, k]) @ V[:, k]).flatten() + \
                (A_k.T @ d_lambda[:, k + 1]).flatten() + \
                term_Qy_GN.flatten() + \
                term_Qy_hxx.flatten() + \
                term_penalty_GN.flatten() + \
                term_penalty_hxx.flatten() + \
                term_xx.flatten() + \
                term_xu.flatten()

            # (HV)_k:
            #   2R V + B^T dlambda_{k+1} + second-order terms from dynamics
            #   (Cu=0 -> no direct contribution from output terms)
            term_ux = self.fu_xx_lambda_contract(
                X[:, k], U[:, k], self.state_space_parameters, lam[:, k + 1], dx[:, k])
            term_uu = self.fu_uu_lambda_contract(
                X[:, k], U[:, k], self.state_space_parameters, lam[:, k + 1], V[:, k])

            Hu[:, k] = \
                (self.l_uu(X[:, k], U[:, k]) @ V[:, k]).flatten() + \
                (self.l_ux(X[:, k], U[:, k]) @ dx[:, k]).flatten() + \
                (B_k.T @ d_lambda[:, k + 1]).flatten() + \
                term_ux.flatten() + term_uu.flatten()

        return Hu
