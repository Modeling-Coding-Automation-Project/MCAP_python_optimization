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

        self.U: np.ndarray = None
        self.X_initial: np.ndarray = None
        self.hvp_function: callable = None

        self._solver_step_iterated_number = 0
        self._pcg_step_iterated_number = 0
        self._line_search_step_iterated_number = 0

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
        Solve the system hvp_function(d) = rhs using PCG without matrix.
        hvp_function: function v -> H v
        M_inv: pre-conditioner (None or vector/function).
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
            Hp = self.hvp_function(self.X_initial, self.U, p)
            Hp += self._lambda_factor * p

            denominator = ActiveSet2D_MatrixOperator.vdot(
                p, Hp, self._active_set)

            # Simple handling of negative curvature and semi-definiteness
            if denominator <= self._pcg_php_minus_limit:
                self._pcg_step_iterated_number = pcg_iteration + 1
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
                  U: np.ndarray,
                  gradient: np.ndarray,
                  umin: np.ndarray,
                  umax: np.ndarray,
                  atol: float = U_NEAR_LIMIT_DEFAULT,
                  gtol: float = GRADIENT_ZERO_LIMIT_DEFAULT):
        """
        True = Free, False = Fixed.
        At lower bound g>0 (going outside) -> Fixed.
        At upper bound g<0 (going outside) -> Fixed.
        """

        m = np.ones_like(U, dtype=bool)
        self._active_set.clear()
        at_lower = np.zeros_like(U, dtype=bool)
        at_upper = np.zeros_like(U, dtype=bool)

        for i in range(U.shape[0]):
            for j in range(U.shape[1]):

                if (U[i, j] >= (umin[i, j] - atol)) and \
                        (U[i, j] <= (umin[i, j] + atol)):
                    at_lower[i, j] = True

                if (U[i, j] >= (umax[i, j] - atol)) and \
                        (U[i, j] <= (umax[i, j] + atol)):
                    at_upper[i, j] = True

        for i in range(U.shape[0]):
            for j in range(U.shape[1]):
                if (at_lower[i, j] and (gradient[i, j] > gtol)) or \
                        (at_upper[i, j] and (gradient[i, j] < -gtol)):
                    m[i, j] = False
                else:
                    self._active_set.push_active(i, j)

        return m

    def solve(
        self,
        U_initial: np.ndarray,
        cost_and_gradient_function: callable,
        hvp_function: callable,
        X_initial: np.ndarray,
        u_min: np.ndarray,
        u_max: np.ndarray,
    ):
        """
        General SQP solver
        (Active Set + Preconditioned Conjugate Gradient + Projected Line Search).
        - U_initial: Initial input sequence (N, nu)
        - cost_and_gradient_function(X_initial, U): Function that returns (J, gradient)
        - hvp_function(X_initial, U, V): Function that returns HVP (H*V)
        - u_min, u_max: Input lower and upper bounds (N, nu)
        """
        self.X_initial = X_initial
        U = U_initial.copy()

        for solver_iteration in range(self._solver_max_iteration):
            # Calculate cost and gradient
            J, gradient = cost_and_gradient_function(X_initial, U)

            if np.linalg.norm(gradient) < self._gradient_norm_zero_limit:
                self._solver_step_iterated_number = solver_iteration + 1
                break

            self._mask = self.free_mask(U, gradient, u_min, u_max)

            rhs = -gradient
            M_inv = 1.0 / (self._diag_R_full + self._lambda_factor)

            self.U = U
            self.hvp_function = hvp_function

            d = self.preconditioned_conjugate_gradient(
                rhs=rhs,
                M_inv=M_inv)

            # line search and projection
            # (No distinction between fixed/free is needed here,
            #  project the whole)
            alpha = 1.0
            U_new = U.copy()

            for line_search_iteration in range(self._line_search_max_iteration):
                U_candidate = U + alpha * d

                for i in range(U_candidate.shape[0]):
                    for j in range(U_candidate.shape[1]):
                        if U_candidate[i, j] < u_min[i, j]:
                            U_candidate[i, j] = u_min[i, j]
                        elif U_candidate[i, j] > u_max[i, j]:
                            U_candidate[i, j] = u_max[i, j]

                J_candidate, _ = cost_and_gradient_function(
                    X_initial, U_candidate)

                if J_candidate <= J or alpha < self._alpha_small_limit:
                    U_new = U_candidate
                    J = J_candidate
                    self._line_search_step_iterated_number = line_search_iteration + 1
                    break

                alpha *= self._alpha_decay_rate
            U = U_new

        return U, J
