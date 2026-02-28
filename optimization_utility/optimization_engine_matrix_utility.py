"""
File: optimization_engine_matrix_utility.py

A utility module for constructing and handling matrices and functions
used in PANOC/ALM-based optimization engine for
Nonlinear Model Predictive Control (NMPC).

Similar to sqp_matrix_utility.py but without second-order (Hessian) computations,
as the PANOC/ALM algorithm only requires first-order information
(cost function value and its gradient).

This module provides:
- Code generation for state/measurement equations and their Jacobians.
- Numerical evaluation of state/measurement functions and Jacobians.
- Cost function and gradient computation (via adjoint method) for PANOC.
- Output constraint mapping and its Jacobian transpose for ALM.
"""
import os
import inspect
import numpy as np
import sympy as sp
import importlib

from external_libraries.MCAP_python_control.python_control.control_deploy import ExpressionDeploy

from optimization_utility.sqp_matrix_utility import extract_parameters_from_state_space_equations

# File name suffixes for generated numpy code files
STATE_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX = "oe_state_function.py"
MEASUREMENT_FUNCTION_NUMPY_CODE_FILE_NAME_SUFFIX = "oe_measurement_function.py"

STATE_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX = "oe_state_jacobian_x.py"
STATE_JACOBIAN_U_NUMPY_CODE_FILE_NAME_SUFFIX = "oe_state_jacobian_u.py"
MEASUREMENT_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX = "oe_measurement_jacobian_x.py"


class OptimizationEngine_CostMatrices:
    """
    A class to handle state-space equations, cost matrices,
    and related computations for PANOC/ALM-based NMPC.

    Unlike SQP_CostMatrices_NMPC, this class does not compute or use
    second-order derivatives (Hessians), since the PANOC algorithm
    relies only on first-order gradient information.

    This class provides:
    - Numerical functions for state/measurement equations and their Jacobians.
    - Cost function and its gradient (via adjoint method) for PANOC.
    - Output constraint mapping and its Jacobian transpose for ALM.

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
        state_jacobian_x (sp.Matrix): Jacobian of f w.r.t. x.
        state_jacobian_u (sp.Matrix): Jacobian of f w.r.t. u.
        measurement_jacobian_x (sp.Matrix): Jacobian of h w.r.t. x.
        X_initial (np.ndarray): Current initial state (set before optimization).
        state_space_parameters: Current model parameters (set before optimization).
        reference_trajectory (np.ndarray): Current reference trajectory.
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
            caller_file_name: str = None
    ):
        if caller_file_name is None:
            # inspect arguments
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

        # Input constraints
        self.U_min_matrix = np.ones((self.nu, self.Np)) * -np.inf
        if U_min is not None:
            self.U_min_matrix = np.tile(U_min, (1, self.Np))

        self.U_max_matrix = np.ones((self.nu, self.Np)) * np.inf
        if U_max is not None:
            self.U_max_matrix = np.tile(U_max, (1, self.Np))

        # Output constraints
        self.Y_min_matrix = np.ones((self.ny, self.Np + 1)) * -np.inf
        if Y_min is not None:
            self.Y_min_matrix = np.tile(Y_min, (1, self.Np + 1))

        self.Y_max_matrix = np.ones((self.ny, self.Np + 1)) * np.inf
        if Y_max is not None:
            self.Y_max_matrix = np.tile(Y_max, (1, self.Np + 1))

        # Precompute Jacobians (no Hessians needed for PANOC)
        self.state_jacobian_x = self.f.jacobian(self.x_syms)   # df/dx
        self.state_jacobian_u = self.f.jacobian(self.u_syms)   # df/du
        self.measurement_jacobian_x = self.h.jacobian(self.x_syms)  # dh/dx

        # Create numpy code files for state and measurement functions
        self.state_function_code_file_name, \
            self.measurement_function_code_file_name = \
            self.create_state_measurement_equation_numpy_code(
                file_name_without_ext=caller_file_name_without_ext)

        self.state_jacobian_x_code_file_name, \
            self.state_jacobian_u_code_file_name, \
            self.measurement_jacobian_x_code_file_name = \
            self.create_jacobians_numpy_code(
                file_name_without_ext=caller_file_name_without_ext)

        # Load callable functions from generated code files
        self.state_function_code_file_function = \
            self.get_function_caller_from_python_file(
                self.state_function_code_file_name)

        self.measurement_function_code_file_function = \
            self.get_function_caller_from_python_file(
                self.measurement_function_code_file_name)

        self.state_jacobian_x_code_file_function = \
            self.get_function_caller_from_python_file(
                self.state_jacobian_x_code_file_name)
        self.state_jacobian_u_code_file_function = \
            self.get_function_caller_from_python_file(
                self.state_jacobian_u_code_file_name)
        self.measurement_jacobian_x_code_file_function = \
            self.get_function_caller_from_python_file(
                self.measurement_jacobian_x_code_file_name)

        # Internal state (set before optimization)
        self.state_space_parameters = None
        self.reference_trajectory = None
        self._Y_offset = None
        self.X_initial = None

    # ----------------------------------------------------------------
    # Code generation methods
    # ----------------------------------------------------------------

    def create_state_measurement_equation_numpy_code(
            self,
            file_name_without_ext: str = None
    ):
        """
        Create numpy code files for state and measurement equations.
        Args:
            file_name_without_ext (str): Base file name for generated code files.
        Returns:
            tuple: (state_function_code_file_name, measurement_function_code_file_name)
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
            file_name_without_ext (str): Base file name for generated code files.
        Returns:
            tuple: (state_jacobian_x_name, state_jacobian_u_name,
                     measurement_jacobian_x_name)
        """
        state_jacobian_x_code_file_name = STATE_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX
        state_jacobian_u_code_file_name = STATE_JACOBIAN_U_NUMPY_CODE_FILE_NAME_SUFFIX
        measurement_jacobian_x_code_file_name = \
            MEASUREMENT_JACOBIAN_X_NUMPY_CODE_FILE_NAME_SUFFIX

        if file_name_without_ext is not None:
            state_jacobian_x_code_file_name = file_name_without_ext + \
                "_" + state_jacobian_x_code_file_name
            state_jacobian_u_code_file_name = file_name_without_ext + \
                "_" + state_jacobian_u_code_file_name
            measurement_jacobian_x_code_file_name = file_name_without_ext + \
                "_" + measurement_jacobian_x_code_file_name

        # write code
        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.state_jacobian_x,
            sym_object_name=os.path.splitext(
                state_jacobian_x_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.state_jacobian_u,
            sym_object_name=os.path.splitext(
                state_jacobian_u_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        ExpressionDeploy.write_function_code_from_sympy(
            sym_object=self.measurement_jacobian_x,
            sym_object_name=os.path.splitext(
                measurement_jacobian_x_code_file_name)[0],
            X=self.x_syms, U=self.u_syms
        )

        return state_jacobian_x_code_file_name, \
            state_jacobian_u_code_file_name, \
            measurement_jacobian_x_code_file_name

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

    # ----------------------------------------------------------------
    # Setter methods
    # ----------------------------------------------------------------

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
        Set the output offset matrix for delay compensation.
        Args:
            Y_offset (np.ndarray): Output offset values of shape (ny, 1).
        """
        if Y_offset.shape[0] != self.ny or Y_offset.shape[1] != 1:
            raise ValueError(
                f"Y_offset must have shape ({self.ny}, 1), got {Y_offset.shape}")
        self._Y_offset = Y_offset

    # ----------------------------------------------------------------
    # Numerical evaluation methods
    # ----------------------------------------------------------------

    def calculate_state_function(
            self,
            X: np.ndarray,
            U: np.ndarray,
            Parameters
    ) -> np.ndarray:
        """
        Calculates the next state vector using the state function.
        Args:
            X (np.ndarray): Current state vector of shape (nx,) or (nx, 1).
            U (np.ndarray): Control input vector of shape (nu,) or (nu, 1).
            Parameters: Model parameters.
        Returns:
            np.ndarray: Next state vector of shape (nx, 1).
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
        Calculates the measurement output for a given state.
        Args:
            X (np.ndarray): State vector of shape (nx,) or (nx, 1).
            Parameters: Model parameters.
        Returns:
            np.ndarray: Measurement output of shape (ny, 1).
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
        Calculates the Jacobian of the state function w.r.t. state variables.
        Args:
            X (np.ndarray): State vector.
            U (np.ndarray): Control input vector.
            Parameters: Model parameters.
        Returns:
            np.ndarray: Jacobian matrix of shape (nx, nx).
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
        Calculates the Jacobian of the state function w.r.t. control inputs.
        Args:
            X (np.ndarray): State vector.
            U (np.ndarray): Control input vector.
            Parameters: Model parameters.
        Returns:
            np.ndarray: Jacobian matrix of shape (nx, nu).
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
        Calculates the Jacobian of the measurement function w.r.t. state variables.
        Args:
            X (np.ndarray): State vector.
            Parameters: Model parameters.
        Returns:
            np.ndarray: Jacobian matrix of shape (ny, nx).
        """
        X = X.reshape((self.nx, 1))
        U = np.zeros((self.nu, 1))
        C = self.measurement_jacobian_x_code_file_function(
            X, U, Parameters
        )
        return C.reshape((self.ny, self.nx))

    def simulate_trajectory(
            self,
            X_initial: np.ndarray,
            U_horizon: np.ndarray,
            Parameters
    ):
        """
        Simulates the system trajectory over the prediction horizon.
        Args:
            X_initial (np.ndarray): Initial state vector of shape (nx,).
            U_horizon (np.ndarray): Control input sequence of shape (nu, Np).
            Parameters: Model parameters.
        Returns:
            np.ndarray: State trajectory of shape (nx, Np + 1).
        """
        X_horizon = np.zeros((self.nx, self.Np + 1))
        X_horizon[:, 0] = X_initial.flatten()
        for k in range(self.Np):
            X_horizon[:, k + 1] = self.calculate_state_function(
                X_horizon[:, k], U_horizon[:, k], Parameters).flatten()
        return X_horizon

    def _compute_Y_horizon(self, X_horizon: np.ndarray) -> np.ndarray:
        """
        Computes the output trajectory from a state trajectory.
        Includes Y_offset if set (for delay compensation).
        Args:
            X_horizon (np.ndarray): State trajectory of shape (nx, Np + 1).
        Returns:
            np.ndarray: Output trajectory of shape (ny, Np + 1).
        """
        if self._Y_offset is None:
            Y_horizon = np.zeros((self.ny, self.Np + 1))
        else:
            Y_horizon = np.tile(self._Y_offset.reshape(
                (self.ny, 1)), (1, self.Np + 1))

        for k in range(self.Np + 1):
            Y_horizon[:, k] += self.calculate_measurement_function(
                X_horizon[:, k], self.state_space_parameters).flatten()

        return Y_horizon

    # ----------------------------------------------------------------
    # PANOC interface methods (flat vector interface)
    # ----------------------------------------------------------------

    def compute_cost(self, u_flat: np.ndarray) -> float:
        """
        Computes the cost function value for PANOC/ALM.

        The cost includes state tracking, output tracking, and input penalty terms.
        Output constraint penalties are NOT included here; they are handled
        by ALM when output constraints are present.

        Uses self.X_initial as the current initial state (must be set
        before calling this method).

        Args:
            u_flat (np.ndarray): Flattened control input sequence of shape (nu * Np,).
        Returns:
            float: Cost function value.
        """
        U_horizon = u_flat.reshape((self.nu, self.Np))
        X_horizon = self.simulate_trajectory(
            self.X_initial, U_horizon, self.state_space_parameters)
        Y_horizon = self._compute_Y_horizon(X_horizon)

        J = 0.0
        for k in range(self.Np):
            e_y_r = Y_horizon[:, k] - self.reference_trajectory[:, k]
            J += (X_horizon[:, k].T @ self.Qx @ X_horizon[:, k] +
                  e_y_r.T @ self.Qy @ e_y_r +
                  U_horizon[:, k].T @ self.R @ U_horizon[:, k])

        eN_y_r = Y_horizon[:, self.Np] - self.reference_trajectory[:, self.Np]
        J += (X_horizon[:, self.Np].T @ self.Px @ X_horizon[:, self.Np] +
              eN_y_r.T @ self.Py @ eN_y_r)

        return float(J)

    def compute_gradient(self, u_flat: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the cost function via adjoint method for PANOC/ALM.

        The gradient does NOT include output constraint penalty gradient;
        those terms are handled by ALM when output constraints are present.

        Uses self.X_initial as the current initial state (must be set
        before calling this method).

        Args:
            u_flat (np.ndarray): Flattened control input sequence of shape (nu * Np, 1).
        Returns:
            np.ndarray: Gradient of shape (nu * Np, 1).
        """
        U_horizon = u_flat.reshape((self.nu, self.Np))
        params = self.state_space_parameters

        X_horizon = self.simulate_trajectory(
            self.X_initial, U_horizon, params)
        Y_horizon = self._compute_Y_horizon(X_horizon)

        # Terminal adjoint
        C_N = self.calculate_measurement_jacobian_x(
            X_horizon[:, self.Np], params)
        eN_y_r = Y_horizon[:, self.Np] - self.reference_trajectory[:, self.Np]
        lam_next = 2.0 * (self.Px @ X_horizon[:, self.Np] +
                          C_N.T @ (self.Py @ eN_y_r))

        gradient = np.zeros_like(U_horizon)
        for k in reversed(range(self.Np)):
            Cx_k = self.calculate_measurement_jacobian_x(
                X_horizon[:, k], params)
            ek_y = Y_horizon[:, k] - self.reference_trajectory[:, k]

            A_k = self.calculate_state_jacobian_x(
                X_horizon[:, k], U_horizon[:, k], params)
            B_k = self.calculate_state_jacobian_u(
                X_horizon[:, k], U_horizon[:, k], params)

            gradient[:, k] = 2.0 * self.R @ U_horizon[:, k] + B_k.T @ lam_next

            lam_next = 2.0 * (self.Qx @ X_horizon[:, k] +
                              Cx_k.T @ (self.Qy @ ek_y)) + \
                A_k.T @ lam_next

        return gradient.reshape((-1, 1))

    # ----------------------------------------------------------------
    # ALM output constraint mapping methods
    # ----------------------------------------------------------------

    def compute_output_mapping(self, u_flat: np.ndarray) -> np.ndarray:
        """
        Computes the output constraint mapping F1(u) for ALM.

        F1(u) returns the predicted output trajectory as a flat vector.

        Uses self.X_initial as the current initial state (must be set
        before calling this method).

        Args:
            u_flat (np.ndarray): Flattened control input sequence of shape (nu * Np, 1).
        Returns:
            np.ndarray: Output trajectory of shape (ny * (Np + 1), 1).
        """
        U_horizon = u_flat.reshape((self.nu, self.Np))
        X_horizon = self.simulate_trajectory(
            self.X_initial, U_horizon, self.state_space_parameters)
        Y_horizon = self._compute_Y_horizon(X_horizon)
        return Y_horizon.reshape((-1, 1))

    def compute_output_jacobian_trans(
            self,
            u_flat: np.ndarray,
            d: np.ndarray
    ) -> np.ndarray:
        """
        Computes JF1(u)^T @ d for ALM via adjoint method.

        JF1 is the Jacobian of the output trajectory mapping F1(u).
        This function computes the transpose-Jacobian product efficiently
        using a backward adjoint pass:
            mu_Np = C_Np^T @ D[:, Np]
            For k = Np-1, ..., 0:
                result[:, k] = B_k^T @ mu_{k+1}
                mu_k = C_k^T @ D[:, k] + A_k^T @ mu_{k+1}

        Uses self.X_initial as the current initial state (must be set
        before calling this method).

        Args:
            u_flat (np.ndarray): Flattened control input sequence of shape (nu * Np, 1).
            d (np.ndarray): Dual vector of shape (ny * (Np + 1), 1).
        Returns:
            np.ndarray: JF1(u)^T @ d of shape (nu * Np, 1).
        """
        U_horizon = u_flat.reshape((self.nu, self.Np))
        D = d.reshape((self.ny, self.Np + 1))
        params = self.state_space_parameters

        X_horizon = self.simulate_trajectory(
            self.X_initial, U_horizon, params)

        # Backward adjoint pass
        C_N = self.calculate_measurement_jacobian_x(
            X_horizon[:, self.Np], params)
        mu = C_N.T @ D[:, self.Np]

        result = np.zeros_like(U_horizon)
        for k in reversed(range(self.Np)):
            A_k = self.calculate_state_jacobian_x(
                X_horizon[:, k], U_horizon[:, k], params)
            B_k = self.calculate_state_jacobian_u(
                X_horizon[:, k], U_horizon[:, k], params)

            result[:, k] = B_k.T @ mu

            C_k = self.calculate_measurement_jacobian_x(
                X_horizon[:, k], params)
            mu = C_k.T @ D[:, k] + A_k.T @ mu

        return result.reshape((-1, 1))
