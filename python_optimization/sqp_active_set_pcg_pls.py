"""
File: sqp_active_set_pcg_pls.py

This module implements a Sequential Quadratic Programming (SQP) solver
with an active set strategy, using Preconditioned Conjugate Gradient (PCG)
for solving the quadratic subproblems and Projected Line Search (PLS)
for handling box constraints.
The solver is designed for problems with input bounds and supports
Hessian-vector products (HVP) for efficient optimization.
"""
import numpy as np

from python_optimization.active_set import ActiveSet2D
from python_optimization.active_set import ActiveSet2D_MatrixOperator

RHS_NORM_ZERO_LIMIT_DEFAULT = 1e-12

GRADIENT_NORM_ZERO_LIMIT_DEFAULT = 1e-6

U_NEAR_LIMIT_DEFAULT = 1e-12
GRADIENT_ZERO_LIMIT_DEFAULT = 1e-12

PCG_TOL_DEFAULT = 1e-4
PCG_MAX_ITERATION_DEFAULT = 30
PCG_PHP_MINUS_LIMIT_DEFAULT = 1e-14

LINE_SEARCH_MAX_ITERATION_DEFAULT = 20

ALPHA_SMALL_LIMIT_DEFAULT = 1e-6
ALPHA_DECAY_RATE_DEFAULT = 0.5

SOLVER_MAX_ITERATION_DEFAULT = 100

LAMBDA_FACTOR_DEFAULT = 1e-6


class SQP_ActiveSet_PCG_PLS:
    """
    Sequential Quadratic Programming (SQP) solver with Active Set,
    Preconditioned Conjugate Gradient (PCG), and Projected Line Search
    (PLS) for box-constrained optimization problems.
    """

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
        self._active_set = ActiveSet2D(
            number_of_columns=U_size[0],
            number_of_rows=U_size[1]
        )

        self.U_horizon: np.ndarray = None
        self.X_initial: np.ndarray = None
        self.hvp_function: callable = None

        self._solver_step_iterated_number = 0
        self._pcg_step_iterated_number = 0
        self._line_search_step_iterated_number = 0

        self.J_opt = 0.0

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

    # getter
    def get_solver_step_iterated_number(self):
        return self._solver_step_iterated_number

    def get_pcg_step_iterated_number(self):
        return self._pcg_step_iterated_number

    def get_line_search_step_iterated_number(self):
        return self._line_search_step_iterated_number

    # functions
    def preconditioned_conjugate_gradient(
        self,
        rhs: np.ndarray,
        M_inv=None
    ):
        """
        Solves a linear system using the Preconditioned Conjugate Gradient
          (PCG) method.

        This method iteratively solves for the search direction `d`
          in a quadratic optimization problem,
        applying a preconditioner to accelerate convergence.
          The PCG algorithm is terminated either when
        the residual norm falls below a specified tolerance or when
          a maximum number of iterations is reached.
        Handles negative curvature and semi-definite cases by early termination.

        Args:
            rhs (np.ndarray): The right-hand side vector of the
              linear system to solve.
            M_inv (np.ndarray, optional): The preconditioner matrix
              (inverse or approximation of the Hessian diagonal).
                If None, no preconditioning is applied.

        Returns:
            np.ndarray: The computed search direction vector `d`
              that approximately solves the system.

        Notes:
            - Uses methods from `ActiveSet2D_MatrixOperator`
              for vector operations restricted to the active set.
            - Relies on class attributes such as `_active_set`,
              `_pcg_max_iteration`, `_lambda_factor`,
                `_pcg_php_minus_limit`, and `_pcg_tol`.
            - The Hessian-vector product is computed via `self.hvp_function`.
            - Early termination occurs if negative curvature is detected
              or the residual norm is sufficiently small.
        """
        d = np.zeros_like(rhs)

        rhs_norm = ActiveSet2D_MatrixOperator.norm(rhs, self._active_set)
        if rhs_norm < RHS_NORM_ZERO_LIMIT_DEFAULT:
            return d

        r = rhs.copy()

        # Preconditioning
        z = ActiveSet2D_MatrixOperator.element_wise_product(
            r, M_inv, self._active_set)

        p = z.copy()

        rz = ActiveSet2D_MatrixOperator.vdot(
            r, z, self._active_set)

        for pcg_iteration in range(self._pcg_max_iteration):
            Hp = self.hvp_function(self.X_initial, self.U_horizon, p)
            Hp += self._lambda_factor * p

            denominator = ActiveSet2D_MatrixOperator.vdot(
                p, Hp, self._active_set)

            # Simple handling of negative curvature and semi-definiteness
            self._pcg_step_iterated_number = pcg_iteration + 1
            if denominator <= self._pcg_php_minus_limit:
                break

            alpha = rz / denominator

            d += ActiveSet2D_MatrixOperator.matrix_multiply_scalar(
                p, alpha, self._active_set)

            r -= ActiveSet2D_MatrixOperator.matrix_multiply_scalar(
                Hp, alpha, self._active_set)

            if ActiveSet2D_MatrixOperator.norm(r, self._active_set) <= \
                    self._pcg_tol * rhs_norm:
                break

            z = ActiveSet2D_MatrixOperator.element_wise_product(
                r, M_inv, self._active_set)

            rz_new = ActiveSet2D_MatrixOperator.vdot(
                r, z, self._active_set)

            beta = rz_new / rz

            p = z + ActiveSet2D_MatrixOperator.matrix_multiply_scalar(
                p, beta, self._active_set)

            rz = rz_new

        return d

    def free_mask(self,
                  U_horizon: np.ndarray,
                  gradient: np.ndarray,
                  U_min_matrix: np.ndarray,
                  U_max_matrix: np.ndarray,
                  atol: float = U_NEAR_LIMIT_DEFAULT,
                  gtol: float = GRADIENT_ZERO_LIMIT_DEFAULT):
        """
        Determines the mask of free variables in the optimization horizon
          based on current values, gradients, and bounds.

        This method identifies which variables are at their lower or upper bounds
          within a specified tolerance (`atol`),
        and then checks the gradient to decide if the variable should be considered
          free or active. Variables at their bounds
        with gradients indicating movement away from the bound are marked as not free.
          The active set is updated accordingly.

        Args:
            U_horizon (np.ndarray): Current values of the optimization variables
              over the horizon.
            gradient (np.ndarray): Gradient of the objective function with respect to
              the variables.
            U_min_matrix (np.ndarray): Matrix of lower bounds for the variables.
            U_max_matrix (np.ndarray): Matrix of upper bounds for the variables.
            atol (float, optional): Absolute tolerance for determining if a variable
              is at its bound. Defaults to U_NEAR_LIMIT_DEFAULT.
            gtol (float, optional): Gradient tolerance for determining activity.
              Defaults to GRADIENT_ZERO_LIMIT_DEFAULT.

        Returns:
            np.ndarray: Boolean mask indicating which variables are free
              (True) and which are not (False).
        """
        m = np.ones_like(U_horizon, dtype=bool)
        self._active_set.clear()
        at_lower = np.zeros_like(U_horizon, dtype=bool)
        at_upper = np.zeros_like(U_horizon, dtype=bool)

        for i in range(U_horizon.shape[0]):
            for j in range(U_horizon.shape[1]):

                if (U_horizon[i, j] >= (U_min_matrix[i, j] - atol)) and \
                        (U_horizon[i, j] <= (U_min_matrix[i, j] + atol)):
                    at_lower[i, j] = True

                if (U_horizon[i, j] >= (U_max_matrix[i, j] - atol)) and \
                        (U_horizon[i, j] <= (U_max_matrix[i, j] + atol)):
                    at_upper[i, j] = True

        for i in range(U_horizon.shape[0]):
            for j in range(U_horizon.shape[1]):
                if (at_lower[i, j] and (gradient[i, j] > gtol)) or \
                        (at_upper[i, j] and (gradient[i, j] < -gtol)):
                    m[i, j] = False
                else:
                    self._active_set.push_active(i, j)

        return m

    def solve(
        self,
        U_horizon_initial: np.ndarray,
        cost_and_gradient_function: callable,
        cost_function: callable,
        hvp_function: callable,
        X_initial: np.ndarray,
        U_min_matrix: np.ndarray,
        U_max_matrix: np.ndarray,
    ):
        """
        Solves a constrained optimization problem using
          Sequential Quadratic Programming (SQP)
        with an active set method and preconditioned conjugate gradient (PCG)
          for the search direction.
        Args:
            U_horizon_initial (np.ndarray): Initial guess for the
              control horizon variables.
            cost_and_gradient_function (callable): Function that computes
              the cost and its gradient given state and control variables.
            cost_function (callable): Function that computes the cost
              given state and control variables.
            hvp_function (callable): Function that computes Hessian-vector products
              for the optimization.
            X_initial (np.ndarray): Initial state variables.
            U_min_matrix (np.ndarray): Lower bounds for the control variables.
            U_max_matrix (np.ndarray): Upper bounds for the control variables.
        Returns:
            np.ndarray: Optimized control horizon variables that minimize
              the cost function subject to bounds.
        """
        self.X_initial = X_initial
        U_horizon = U_horizon_initial.copy()

        for solver_iteration in range(self._solver_max_iteration):
            # Calculate cost and gradient
            J, gradient = cost_and_gradient_function(X_initial, U_horizon)

            self._solver_step_iterated_number = solver_iteration + 1
            if np.linalg.norm(gradient) < self._gradient_norm_zero_limit:
                break

            self._mask = self.free_mask(
                U_horizon, gradient, U_min_matrix, U_max_matrix)

            rhs = -gradient
            M_inv = 1.0 / (self._diag_R_full + self._lambda_factor)

            self.U_horizon = U_horizon
            self.hvp_function = hvp_function

            d = self.preconditioned_conjugate_gradient(
                rhs=rhs,
                M_inv=M_inv)

            # line search and projection
            # (No distinction between fixed/free is needed here,
            #  project the whole)
            alpha = 1.0
            U_horizon_new = U_horizon.copy()

            U_updated_flag = False
            for line_search_iteration in range(self._line_search_max_iteration):
                U_candidate = U_horizon + alpha * d

                for i in range(U_candidate.shape[0]):
                    for j in range(U_candidate.shape[1]):
                        if U_candidate[i, j] < U_min_matrix[i, j]:
                            U_candidate[i, j] = U_min_matrix[i, j]
                        elif U_candidate[i, j] > U_max_matrix[i, j]:
                            U_candidate[i, j] = U_max_matrix[i, j]

                J_candidate = cost_function(X_initial, U_candidate)

                self._line_search_step_iterated_number = line_search_iteration + 1
                if J_candidate <= J or alpha < self._alpha_small_limit:
                    U_horizon_new = U_candidate
                    J = J_candidate
                    U_updated_flag = True
                    break

                alpha *= self._alpha_decay_rate

            if True == U_updated_flag:
                U_horizon = U_horizon_new
            else:
                break

        self.J_opt = J

        return U_horizon
