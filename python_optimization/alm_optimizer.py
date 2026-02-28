"""
File: alm_optimizer.py

Description: This module implements the ALM/PM (Augmented Lagrangian Method /
Penalty Method) algorithm for solving constrained nonlinear optimization problems.

The implementation is based on the Rust optimization-engine library
(src/alm/alm_optimizer.rs and related files) and adapted for Python/NumPy.
It uses the PANOC solver (panoc.py) as the inner solver.

ALM/PM solves problems of the form:

    min  f(u)
     u
    s.t. u ∈ U           (box constraints on decision variables, handled by PANOC)
         F1(u) ∈ C       (ALM-type constraints, e.g., output constraints)
         F2(u) = 0        (PM-type equality constraints, optional)

For nonlinear MPC applications, the typical constraints are:
    - u_min <= u <= u_max  (input box constraints → set U for PANOC)
    - y_min <= Y(u) <= y_max  (output box constraints → F1(u) = Y(u), C = [y_min, y_max])

Algorithm overview (outer loop):
1. y ← Π_Y(y)                           (project Lagrange multipliers onto set Y)
2. u ← argmin_{u∈U} ψ(u; ξ)             (solve inner problem via PANOC, ξ = (c, y))
3. y⁺ ← y + c[F1(u) - Π_C(F1(u) + y/c)] (update Lagrange multipliers)
4. z⁺ ← ||y⁺ - y||, t⁺ ← ||F2(u)||     (compute infeasibility measures)
5. If z⁺ ≤ cδ and t⁺ ≤ δ and ε_nu ≤ ε    → converged, return (u, y⁺)
6. Else if no sufficient decrease         → c ← rho·c  (increase penalty)
7. ε̄ ← max(ε, β·ε̄)                      (shrink inner tolerance)

The augmented cost function is:
    ψ(u; ξ) = f(u) + (c/2)[dist²_C(F1(u) + y/c̄) + ||F2(u)||²]
where c̄ = max(1, c), and its gradient is:
    ∇ψ(u; ξ) = ∇f(u) + c·JF1(u)ᵀ[t(u) - Π_C(t(u))] + c·JF2(u)ᵀF2(u)
where t(u) = F1(u) + y/c.

Module structure:
    - ALM_Factory:       Builds ψ(u; ξ) and ∇ψ(u; ξ) from raw problem data
    - ALM_Problem:       Bundles all problem data for the ALM optimizer
    - ALM_Cache:         Pre-allocated working memory for the algorithm
    - ALM_Optimizer:     Main ALM/PM solver (outer loop with PANOC inner solver)
    - ALM_SolverStatus:  Result returned by ALM_Optimizer.solve()
    - Utility functions: make_box_projection, make_ball_projection

References:
    - optimization-engine: https://github.com/alphaville/optimization-engine
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional
import time

from python_optimization.panoc import (
    ExitStatus,
    SolverStatus as PANOC_SolverStatus,
    PANOC_Cache,
    PANOC_Optimizer,
)

# Maximum number of outer ALM/PM iterations
DEFAULT_MAX_OUTER_ITERATIONS: int = 50
# Maximum number of inner PANOC iterations per outer iteration
DEFAULT_MAX_INNER_ITERATIONS: int = 5000
# Target tolerance for the inner solver (epsilon)
DEFAULT_EPSILON_TOLERANCE: float = 1e-6
# Tolerance for ALM/PM infeasibility (delta)
DEFAULT_DELTA_TOLERANCE: float = 1e-4
# Factor by which the penalty parameter c is multiplied (rho)
DEFAULT_PENALTY_UPDATE_FACTOR: float = 5.0
# Factor by which the inner tolerance is shrunk each iteration (beta)
DEFAULT_EPSILON_UPDATE_FACTOR: float = 0.1
# Sufficient decrease coefficient (theta) for penalty stall check
DEFAULT_INFEAS_SUFFICIENT_DECREASE_FACTOR: float = 0.1
# Initial inner tolerance (epsilon_0)
DEFAULT_INITIAL_TOLERANCE: float = 0.1
# Initial penalty parameter (c_0)
DEFAULT_INITIAL_PENALTY: float = 10.0
# Machine epsilon for numerical comparisons
SMALL_EPSILON: float = 1e-30


def make_box_projection(
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], None]:
    """
    Create an in-place box projection function.

    Returns a callable that projects x in-place onto the box [lower, upper].

    Parameters
    ----------
    lower : np.ndarray or None
        Element-wise lower bounds.  ``None`` means no lower bound (-inf).
    upper : np.ndarray or None
        Element-wise upper bounds.  ``None`` means no upper bound (+inf).

    Returns
    -------
    callable
        Function with signature ``project(x: np.ndarray) -> None``
        that clips x in-place.
    """
    def project(x: np.ndarray) -> None:
        if lower is not None:
            np.maximum(x, lower, out=x)
        if upper is not None:
            np.minimum(x, upper, out=x)
    return project


def make_ball_projection(
    center: Optional[np.ndarray],
    radius: float,
) -> Callable[[np.ndarray], None]:
    """
    Create an in-place Euclidean ball projection function.

    Returns a callable that projects x in-place onto the ball
    {x : ||x - center|| <= radius}.

    Parameters
    ----------
    center : np.ndarray or None
        Center of the ball.  ``None`` means the origin.
    radius : float
        Radius of the ball (must be positive).

    Returns
    -------
    callable
        Function with signature ``project(x: np.ndarray) -> None``
        that projects x in-place.
    """
    assert radius > 0.0, "radius must be positive"

    def project(x: np.ndarray) -> None:
        if center is not None:
            d = x - center
        else:
            d = x.copy()
        norm_d = float(np.linalg.norm(d))
        if norm_d > radius:
            if center is not None:
                x[:] = center + (radius / norm_d) * d
            else:
                x[:] = (radius / norm_d) * d
    return project


# ============================================================================
# ALM Solver Status
# ============================================================================
@dataclass
class ALM_SolverStatus:
    """
    Result returned by :meth:`ALM_Optimizer.solve`.

    Attributes
    ----------
    exit_status : ExitStatus
        Reason the solver terminated.
    num_outer_iterations : int
        Number of outer ALM/PM iterations performed.
    num_inner_iterations : int
        Total number of inner PANOC iterations across all outer iterations.
    last_problem_norm_fpr : float
        Norm of the fixed-point residual of the last inner problem.
    lagrange_multipliers : np.ndarray or None
        Final Lagrange multiplier vector y⁺ (None if no ALM constraints).
    solve_time : float
        Total solve time in seconds.
    penalty : float
        Final value of the penalty parameter c.
    delta_y_norm : float
        ||y⁺ - y|| at termination (ALM infeasibility measure).
    f2_norm : float
        ||F2(u)|| at termination (PM infeasibility measure).
    cost : float
        Original cost f(u) at the solution (without penalty terms).
    """
    exit_status: ExitStatus
    num_outer_iterations: int
    num_inner_iterations: int
    last_problem_norm_fpr: float
    lagrange_multipliers: Optional[np.ndarray]
    solve_time: float
    penalty: float
    delta_y_norm: float
    f2_norm: float
    cost: float

    def has_converged(self) -> bool:
        """Return True if the solver converged to an (ε, δ)-AKKT point."""
        return self.exit_status == ExitStatus.CONVERGED


# ============================================================================
# ALM Cache
# ============================================================================
class ALM_Cache:
    """
    Pre-allocated working memory for the ALM/PM algorithm.

    Create once and reuse across multiple ``solve`` calls to avoid
    repeated memory allocation.

    Parameters
    ----------
    panoc_cache : PANOC_Cache
        Cache for the inner PANOC solver.
    n1 : int
        Dimension of F1 output (number of ALM-type constraints).
        Set to 0 if there are no ALM constraints.
    n2 : int
        Dimension of F2 output (number of PM-type equality constraints).
        Set to 0 if there are no PM constraints.
    """

    def __init__(self, panoc_cache: PANOC_Cache, n1: int, n2: int = 0):
        assert n1 >= 0, "n1 must be non-negative"
        assert n2 >= 0, "n2 must be non-negative"

        self.panoc_cache: PANOC_Cache = panoc_cache
        self.n1: int = n1
        self.n2: int = n2

        # Lagrange multipliers (next iterate, y⁺)
        self.y_plus: Optional[np.ndarray] = (
            np.zeros(n1) if n1 > 0 else None
        )

        # Parameter vector xi = (c, y) ∈ R^{1+n1}
        # xi[0] = c (penalty parameter), xi[1:] = y (Lagrange multipliers)
        if n1 + n2 > 0:
            self.xi: Optional[np.ndarray] = np.zeros(1 + n1)
            self.xi[0] = DEFAULT_INITIAL_PENALTY
        else:
            self.xi = None

        # Auxiliary working vectors
        self.w_alm_aux: Optional[np.ndarray] = (
            np.zeros(n1) if n1 > 0 else None
        )
        self.w_pm: Optional[np.ndarray] = (
            np.zeros(n2) if n2 > 0 else None
        )

        # Infeasibility measures
        self.delta_y_norm: float = 0.0       # ||y⁺ - y|| at current iteration
        self.delta_y_norm_plus: float = np.inf
        self.f2_norm: float = 0.0             # ||F2(u)|| at current iteration
        self.f2_norm_plus: float = np.inf

        # Counters
        self.iteration: int = 0
        self.inner_iteration_count: int = 0
        self.last_inner_problem_norm_fpr: float = -1.0

    def reset(self) -> None:
        """
        Reset the cache to its initial state (called at the start of each solve).
        """
        self.panoc_cache.reset()
        self.iteration = 0
        self.f2_norm = 0.0
        self.f2_norm_plus = 0.0
        self.delta_y_norm = 0.0
        self.delta_y_norm_plus = 0.0
        self.inner_iteration_count = 0


# ============================================================================
# ALM Factory
# ============================================================================
class ALM_Factory:
    """
    Constructs the augmented cost ψ(u; ξ) and its gradient ∇ψ(u; ξ)
    from the raw problem data.

    Given f, ∇f, F1, JF1ᵀ·d, C (and optionally F2, JF2ᵀ·d), it builds:

        ψ(u; ξ) = f(u) + (c/2)[dist²_C(F1(u) + y/c̄) + ||F2(u)||²]

        ∇ψ(u; ξ) = ∇f(u) + c·JF1(u)ᵀ[t(u) - Π_C(t(u))] + c·JF2(u)ᵀF2(u)

    where c̄ = max(1, c), t(u) = F1(u) + y/c̄ for ψ and t(u) = F1(u) + y/c
    for ∇ψ, and ξ = (c, y).

    Parameters
    ----------
    f : callable
        Cost function f(u) -> float.
    df : callable
        Gradient ∇f(u) -> ndarray of shape (n_u,).
    mapping_f1 : callable or None
        F1(u) -> ndarray of shape (n1,).  Required if n1 > 0.
    jacobian_f1_trans : callable or None
        JF1(u)ᵀd -> ndarray of shape (n_u,).
        Signature: jacobian_f1_trans(u, d) -> ndarray.
        Required if mapping_f1 is provided.
    set_c_project : callable or None
        In-place projection onto C.  Signature: project(x) modifies x in-place.
        Required if mapping_f1 is provided.
    mapping_f2 : callable or None
        F2(u) -> ndarray of shape (n2,).  Required if n2 > 0.
    jacobian_f2_trans : callable or None
        JF2(u)ᵀd -> ndarray of shape (n_u,).
        Signature: jacobian_f2_trans(u, d) -> ndarray.
        Required if mapping_f2 is provided.
    n1 : int
        Range dimension of F1 (number of ALM constraints).
    n2 : int
        Range dimension of F2 (number of PM constraints).
    """

    def __init__(
        self,
        f: Callable[[np.ndarray], float],
        df: Callable[[np.ndarray], np.ndarray],
        mapping_f1: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        jacobian_f1_trans: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        set_c_project: Optional[Callable[[np.ndarray], None]] = None,
        mapping_f2: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        jacobian_f2_trans: Optional[
            Callable[[np.ndarray, np.ndarray], np.ndarray]
        ] = None,
        n1: int = 0,
        n2: int = 0,
    ):
        # Validation: F1, JF1ᵀ and C must be provided together
        assert not ((mapping_f1 is None) ^ (set_c_project is None)), \
            "F1 and set C must both be provided or both omitted"
        assert not ((mapping_f1 is None) ^ (jacobian_f1_trans is None)), \
            "F1 and JF1ᵀ must both be provided or both omitted"
        # Validation: F2 and JF2ᵀ must be provided together, iff n2 > 0
        assert not ((mapping_f2 is None) ^ (n2 == 0)), \
            "F2 must be provided if and only if n2 > 0"
        assert not ((mapping_f2 is None) ^ (jacobian_f2_trans is None)), \
            "F2 and JF2ᵀ must both be provided or both omitted"

        self._f = f
        self._df = df
        self._mapping_f1 = mapping_f1
        self._jacobian_f1_trans = jacobian_f1_trans
        self._set_c_project = set_c_project
        self._mapping_f2 = mapping_f2
        self._jacobian_f2_trans = jacobian_f2_trans
        self._n1 = n1
        self._n2 = n2

    def psi(self, u: np.ndarray, xi: np.ndarray) -> float:
        """
        Compute the augmented cost ψ(u; ξ).

        Parameters
        ----------
        u : np.ndarray
            Decision variable.
        xi : np.ndarray
            Parameter vector ξ = (c, y).  May be empty if n1 = n2 = 0.

        Returns
        -------
        float
            Value of ψ(u; ξ).
        """
        cost = self._f(u)
        n_y = len(xi) - 1 if len(xi) > 0 else 0

        # ALM term: (c/2) * dist²_C(F1(u) + y/c̄)
        if self._mapping_f1 is not None and self._set_c_project is not None:
            c = xi[0]
            y = xi[1:]
            c_bar = max(c, 1.0)

            # t = F1(u) + y / c̄
            f1_u = self._mapping_f1(u)
            t = f1_u + y / c_bar

            # s = Π_C(t)
            s = t.copy()
            self._set_c_project(s)

            # dist²_C(t) = ||t - s||²
            diff = t - s
            dist_sq = float(np.dot(diff, diff))
            cost += 0.5 * c * dist_sq

        # PM term: (c/2) * ||F2(u)||²
        if self._mapping_f2 is not None:
            c = xi[0]
            f2_u = self._mapping_f2(u)
            cost += 0.5 * c * float(np.dot(f2_u, f2_u))

        return cost

    def d_psi(self, u: np.ndarray, xi: np.ndarray) -> np.ndarray:
        """
        Compute the gradient ∇ψ(u; ξ).

        Parameters
        ----------
        u : np.ndarray
            Decision variable.
        xi : np.ndarray
            Parameter vector ξ = (c, y).  May be empty if n1 = n2 = 0.

        Returns
        -------
        np.ndarray
            Gradient ∇ψ(u; ξ) of shape (n_u,).
        """
        grad = self._df(u).copy()

        # ALM gradient: c · JF1(u)ᵀ [t(u) - Π_C(t(u))]
        if (self._mapping_f1 is not None
                and self._jacobian_f1_trans is not None
                and self._set_c_project is not None):
            c = xi[0]
            y = xi[1:]

            # t = F1(u) + y/c  (note: uses c, not c̄)
            f1_u = self._mapping_f1(u)
            t = f1_u + y / c

            # s = Π_C(t)
            s = t.copy()
            self._set_c_project(s)

            # d = t - Π_C(t)
            d = t - s

            # grad += c · JF1(u)ᵀ · d
            jf1t_d = self._jacobian_f1_trans(u, d)
            grad += c * jf1t_d

        # PM gradient: c · JF2(u)ᵀ · F2(u)
        if (self._mapping_f2 is not None
                and self._jacobian_f2_trans is not None):
            c = xi[0]
            f2_u = self._mapping_f2(u)
            jf2t_f2u = self._jacobian_f2_trans(u, f2_u)
            grad += c * jf2t_f2u

        return grad


# ============================================================================
# ALM Problem
# ============================================================================
class ALM_Problem:
    """
    Problem definition for ALM/PM optimization.

    Bundles all data required by :class:`ALM_Optimizer`: the parametric
    augmented cost and its gradient (typically built by :class:`ALM_Factory`),
    box constraints on the decision variable, constraint mappings, and
    projection operators.

    Parameters
    ----------
    parametric_cost : callable
        ψ(u, xi) -> float.  Augmented cost function.
    parametric_gradient : callable
        ∇ψ(u, xi) -> ndarray.  Gradient of the augmented cost.
    u_min : np.ndarray or None
        Element-wise lower bounds on u.  At least one of u_min / u_max
        must be provided.
    u_max : np.ndarray or None
        Element-wise upper bounds on u.
    mapping_f1 : callable or None
        F1(u) -> ndarray of shape (n1,).  Required if n1 > 0.
    set_c_project : callable or None
        In-place projection onto set C.  Required if n1 > 0.
    set_y_project : callable or None
        In-place projection onto set Y (for Lagrange multipliers).
        Optional; if None, Lagrange multipliers are not projected.
    mapping_f2 : callable or None
        F2(u) -> ndarray of shape (n2,).  Required if n2 > 0.
    n1 : int
        Dimension of F1 output (number of ALM constraints).
    n2 : int
        Dimension of F2 output (number of PM equality constraints).
    """

    def __init__(
        self,
        parametric_cost: Callable[[np.ndarray, np.ndarray], float],
        parametric_gradient: Callable[[np.ndarray, np.ndarray], np.ndarray],
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        mapping_f1: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        set_c_project: Optional[Callable[[np.ndarray], None]] = None,
        set_y_project: Optional[Callable[[np.ndarray], None]] = None,
        mapping_f2: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        n1: int = 0,
        n2: int = 0,
    ):
        assert u_min is not None or u_max is not None, \
            "At least one of u_min / u_max must be provided"
        # F1 and set_c must both be provided or both omitted
        assert not ((mapping_f1 is None) ^ (set_c_project is None)), \
            "F1 and set C must both be provided or both omitted"
        # set_c is provided iff n1 > 0
        assert not ((set_c_project is None) ^ (n1 == 0)), \
            "set C must be provided if and only if n1 > 0"
        # F2 is provided iff n2 > 0
        assert not ((mapping_f2 is None) ^ (n2 == 0)), \
            "F2 must be provided if and only if n2 > 0"

        self.parametric_cost = parametric_cost
        self.parametric_gradient = parametric_gradient
        self.u_min = u_min
        self.u_max = u_max
        self.mapping_f1 = mapping_f1
        self.set_c_project = set_c_project
        self.set_y_project = set_y_project
        self.mapping_f2 = mapping_f2
        self.n1 = n1
        self.n2 = n2


# ============================================================================
# ALM Optimizer
# ============================================================================
class ALM_Optimizer:
    """
    ALM/PM solver for constrained nonlinear optimization.

    Uses PANOC as the inner solver.  Solves::

        min  f(u)
         u
        s.t. u ∈ U            (box constraints, handled by PANOC)
             F1(u) ∈ C        (ALM-type constraints)
             F2(u) = 0         (PM-type constraints, optional)

    Parameters
    ----------
    alm_cache : ALM_Cache
        Pre-allocated cache (create with :class:`ALM_Cache`).
    alm_problem : ALM_Problem
        Problem definition (create with :class:`ALM_Problem`).
    max_outer_iterations : int
        Maximum number of outer ALM/PM iterations (nu_max).
    max_inner_iterations : int
        Maximum number of inner PANOC iterations per outer iteration.
    epsilon_tolerance : float
        Target tolerance ε for the inner solver.
    delta_tolerance : float
        Tolerance δ for infeasibility.
    penalty_update_factor : float
        Factor rho > 1 to increase penalty parameter c.
    epsilon_update_factor : float
        Factor β ∈ (0, 1) to decrease inner tolerance.
    sufficient_decrease_coeff : float
        Coefficient θ ∈ (0, 1) for sufficient decrease check.
    initial_inner_tolerance : float
        Initial inner tolerance ε₀ (must be >= epsilon_tolerance).
    initial_penalty : float or None
        Initial penalty c₀.  None uses the cache default.
    initial_y : np.ndarray or None
        Initial Lagrange multiplier vector y⁰.  None uses zeros.
    max_duration : float or None
        Maximum allowed solve time in seconds.  None means unlimited.
    """

    def __init__(
        self,
        alm_cache: ALM_Cache,
        alm_problem: ALM_Problem,
        max_outer_iterations: int = DEFAULT_MAX_OUTER_ITERATIONS,
        max_inner_iterations: int = DEFAULT_MAX_INNER_ITERATIONS,
        epsilon_tolerance: float = DEFAULT_EPSILON_TOLERANCE,
        delta_tolerance: float = DEFAULT_DELTA_TOLERANCE,
        penalty_update_factor: float = DEFAULT_PENALTY_UPDATE_FACTOR,
        epsilon_update_factor: float = DEFAULT_EPSILON_UPDATE_FACTOR,
        sufficient_decrease_coeff: float = DEFAULT_INFEAS_SUFFICIENT_DECREASE_FACTOR,
        initial_inner_tolerance: float = DEFAULT_INITIAL_TOLERANCE,
        initial_penalty: Optional[float] = None,
        initial_y: Optional[np.ndarray] = None,
        max_duration: Optional[float] = None,
    ):
        assert max_outer_iterations > 0, "max_outer_iterations must be positive"
        assert max_inner_iterations > 0, "max_inner_iterations must be positive"
        assert epsilon_tolerance > 0.0, "epsilon_tolerance must be positive"
        assert delta_tolerance > 0.0, "delta_tolerance must be positive"
        assert penalty_update_factor > 1.0 + SMALL_EPSILON, \
            "penalty_update_factor must be > 1"
        assert SMALL_EPSILON < epsilon_update_factor < 1.0 - SMALL_EPSILON, \
            "epsilon_update_factor must be in (0, 1)"
        assert SMALL_EPSILON < sufficient_decrease_coeff < 1.0 - SMALL_EPSILON, \
            "sufficient_decrease_coeff must be in (0, 1)"
        assert initial_inner_tolerance >= epsilon_tolerance, \
            "initial_inner_tolerance must be >= epsilon_tolerance"

        self._cache = alm_cache
        self._problem = alm_problem
        self.max_outer_iterations = max_outer_iterations
        self.max_inner_iterations = max_inner_iterations
        self.epsilon_tolerance = epsilon_tolerance
        self.delta_tolerance = delta_tolerance
        self.penalty_update_factor = penalty_update_factor
        self.epsilon_update_factor = epsilon_update_factor
        self.sufficient_decrease_coeff = sufficient_decrease_coeff
        self.initial_inner_tolerance = initial_inner_tolerance
        self.max_duration = max_duration

        # Set initial penalty parameter
        if initial_penalty is not None:
            assert initial_penalty > SMALL_EPSILON, \
                "initial_penalty must be positive"
            if self._cache.xi is not None:
                self._cache.xi[0] = initial_penalty

        # Set initial Lagrange multipliers
        if initial_y is not None:
            assert len(initial_y) == self._problem.n1, \
                "initial_y length must equal n1"
            if self._cache.xi is not None:
                self._cache.xi[1:] = initial_y

        # Set initial inner tolerance
        self._cache.panoc_cache.tolerance = initial_inner_tolerance

    # ----------------------------------------------------------------
    #  Main API
    # ----------------------------------------------------------------
    def solve(self, u: np.ndarray) -> ALM_SolverStatus:
        """
        Solve the ALM/PM problem.

        Parameters
        ----------
        u : np.ndarray
            Initial guess.  Modified **in-place** with the solution on return.

        Returns
        -------
        ALM_SolverStatus
            Solver status including exit condition, iterations, cost, etc.
        """
        tic = time.perf_counter()
        self._cache.reset()
        self._cache.panoc_cache.tolerance = self.initial_inner_tolerance

        num_outer_iterations = 0
        exit_status = ExitStatus.CONVERGED
        should_continue = True
        inner_exit_status = ExitStatus.CONVERGED

        for _ in range(self.max_outer_iterations):
            # Check time limit
            if self.max_duration is not None:
                elapsed = time.perf_counter() - tic
                if elapsed >= self.max_duration:
                    exit_status = ExitStatus.NOT_CONVERGED_OUT_OF_TIME
                    break

            num_outer_iterations += 1
            should_continue, inner_exit_status = self._step(u)

            if inner_exit_status == ExitStatus.NOT_CONVERGED_OUT_OF_TIME:
                exit_status = ExitStatus.NOT_CONVERGED_OUT_OF_TIME
                break

            if not should_continue:
                break

        # Determine final exit status
        if exit_status != ExitStatus.NOT_CONVERGED_OUT_OF_TIME:
            exit_status = inner_exit_status

        if (num_outer_iterations == self.max_outer_iterations
                and should_continue):
            exit_status = ExitStatus.NOT_CONVERGED_ITERATIONS

        # Extract final penalty parameter
        c = self._cache.xi[0] if self._cache.xi is not None else 0.0

        # Compute original cost at solution (penalty terms excluded)
        cost_value = self._compute_cost_at_solution(u)

        # Build result
        lagrange = (
            self._cache.y_plus.copy()
            if self._cache.y_plus is not None
            else None
        )

        return ALM_SolverStatus(
            exit_status=exit_status,
            num_outer_iterations=num_outer_iterations,
            num_inner_iterations=self._cache.inner_iteration_count,
            last_problem_norm_fpr=self._cache.last_inner_problem_norm_fpr,
            lagrange_multipliers=lagrange,
            solve_time=time.perf_counter() - tic,
            penalty=c,
            delta_y_norm=self._cache.delta_y_norm_plus,
            f2_norm=self._cache.f2_norm_plus,
            cost=cost_value,
        )

    # ----------------------------------------------------------------
    #  ALM step (one outer iteration)
    # ----------------------------------------------------------------
    def _step(self, u: np.ndarray):
        """
        Perform one ALM outer iteration.

        Returns
        -------
        tuple (should_continue: bool, inner_exit_status: ExitStatus)
            should_continue is False when an (ε, δ)-AKKT point is found.
        """
        # 1. Project y onto set Y
        self._project_on_set_y()

        # 2. Solve inner problem via PANOC
        inner_status = self._solve_inner_problem(u)
        self._cache.last_inner_problem_norm_fpr = (
            inner_status.norm_fixed_point_residual
        )
        self._cache.inner_iteration_count += inner_status.number_of_iteration
        inner_exit_status = inner_status.exit_status

        # 3. Update Lagrange multipliers:
        #    y⁺ ← y + c·[F1(u) - Π_C(F1(u) + y/c)]
        self._update_lagrange_multipliers(u)

        # 4. Compute infeasibility measures
        self._compute_pm_infeasibility(u)   # ||F2(u)||
        self._compute_alm_infeasibility()   # ||y⁺ - y||

        # 5. Check exit criterion
        if self._is_exit_criterion_satisfied():
            return False, inner_exit_status  # converged

        # 6. Update penalty parameter if insufficient decrease
        if not self._is_penalty_stall_criterion():
            self._update_penalty_parameter()

        # 7. Shrink inner tolerance: ε̄ ← max(ε, β·ε̄)
        self._update_inner_tolerance()

        # 8. Final bookkeeping
        self._final_cache_update()

        return True, inner_exit_status

    # ----------------------------------------------------------------
    #  Private helper methods
    # ----------------------------------------------------------------
    def _project_on_set_y(self) -> None:
        """Project Lagrange multipliers y onto set Y (in-place on xi[1:])."""
        if (self._problem.set_y_project is not None
                and self._cache.xi is not None):
            self._problem.set_y_project(self._cache.xi[1:])

    def _solve_inner_problem(self, u: np.ndarray) -> PANOC_SolverStatus:
        """
        Solve the inner problem ``min_{u ∈ U} ψ(u; ξ)`` using PANOC.

        The parameter vector ξ is captured from the cache by reference.
        """
        xi = self._cache.xi if self._cache.xi is not None else np.array([])
        # Build non-parametric cost/gradient by capturing xi
        xi_ref = xi  # numpy reference (not copy)

        def cost_func(u_: np.ndarray) -> float:
            return self._problem.parametric_cost(u_, xi_ref)

        def grad_func(u_: np.ndarray) -> np.ndarray:
            return self._problem.parametric_gradient(u_, xi_ref)

        solver = PANOC_Optimizer(
            cost_func=cost_func,
            gradient_func=grad_func,
            cache=self._cache.panoc_cache,
            u_min=self._problem.u_min,
            u_max=self._problem.u_max,
            max_iteration=self.max_inner_iterations,
        )
        return solver.solve(u)

    def _update_lagrange_multipliers(self, u: np.ndarray) -> None:
        """
        Update Lagrange multipliers:
            y⁺ = y + c · [F1(u) - Π_C(F1(u) + y/c)]

        Steps:
            1. w = F1(u)
            2. y_plus = w + y/c
            3. y_plus = Π_C(y_plus)
            4. y_plus = y + c · (w - y_plus)
        """
        if self._problem.n1 == 0:
            return

        f1 = self._problem.mapping_f1
        set_c = self._problem.set_c_project
        if f1 is None or set_c is None:
            return
        if self._cache.xi is None or self._cache.y_plus is None:
            return

        c = self._cache.xi[0]
        y = self._cache.xi[1:]

        # Step 1: w = F1(u)
        w = f1(u)
        if self._cache.w_alm_aux is not None:
            self._cache.w_alm_aux[:] = w

        # Step 2: y_plus = F1(u) + y/c
        self._cache.y_plus[:] = w + y / c

        # Step 3: y_plus = Π_C(y_plus)
        set_c(self._cache.y_plus)

        # Step 4: y_plus = y + c · (F1(u) - Π_C(F1(u) + y/c))
        self._cache.y_plus[:] = y + c * (w - self._cache.y_plus)

    def _compute_alm_infeasibility(self) -> None:
        """Compute ALM infeasibility: ||y⁺ - y||."""
        if self._cache.y_plus is not None and self._cache.xi is not None:
            y = self._cache.xi[1:]
            self._cache.delta_y_norm_plus = float(
                np.linalg.norm(self._cache.y_plus - y)
            )

    def _compute_pm_infeasibility(self, u: np.ndarray) -> None:
        """Compute PM infeasibility: ||F2(u)||."""
        if (self._problem.mapping_f2 is not None
                and self._cache.w_pm is not None):
            self._cache.w_pm[:] = self._problem.mapping_f2(u)
            self._cache.f2_norm_plus = float(
                np.linalg.norm(self._cache.w_pm)
            )

    def _is_exit_criterion_satisfied(self) -> bool:
        """
        Check if (ε, δ)-AKKT conditions are satisfied.

        Three criteria must hold simultaneously:
            1. ||Δy|| ≤ c·δ   (or no ALM constraints)
            2. ||F2(u)|| ≤ δ   (or no PM constraints)
            3. ε_nu ≤ ε         (inner tolerance has reached target)
        """
        cache = self._cache
        problem = self._problem

        # Criterion 1: ||Δy|| ≤ c·δ
        if problem.n1 > 0:
            if cache.xi is not None:
                c = cache.xi[0]
                criterion_1 = (
                    cache.iteration > 0
                    and cache.delta_y_norm_plus
                    <= c * self.delta_tolerance + SMALL_EPSILON
                )
            else:
                criterion_1 = True
        else:
            criterion_1 = True

        # Criterion 2: ||F2(u)|| ≤ δ
        criterion_2 = (
            problem.n2 == 0
            or cache.f2_norm_plus
            <= self.delta_tolerance + SMALL_EPSILON
        )

        # Criterion 3: current inner tolerance ≤ target epsilon
        criterion_3 = (
            cache.panoc_cache.tolerance
            <= self.epsilon_tolerance + SMALL_EPSILON
        )

        return criterion_1 and criterion_2 and criterion_3

    def _is_penalty_stall_criterion(self) -> bool:
        """
        Check if penalty update should be skipped (sufficient decrease).

        Returns True if the penalty should NOT be updated (i.e., stall),
        which happens when iteration == 0 or there was sufficient
        decrease in the infeasibility measures.
        """
        cache = self._cache
        problem = self._problem

        if cache.iteration == 0:
            return True

        is_alm = problem.n1 > 0
        is_pm = problem.n2 > 0

        criterion_alm = (
            cache.delta_y_norm_plus
            <= self.sufficient_decrease_coeff * cache.delta_y_norm
            + SMALL_EPSILON
        )
        criterion_pm = (
            cache.f2_norm_plus
            <= self.sufficient_decrease_coeff * cache.f2_norm
            + SMALL_EPSILON
        )

        if is_alm and not is_pm:
            return criterion_alm
        elif not is_alm and is_pm:
            return criterion_pm
        elif is_alm and is_pm:
            return criterion_alm and criterion_pm

        return False

    def _update_penalty_parameter(self) -> None:
        """Multiply penalty parameter c by penalty_update_factor."""
        if self._cache.xi is not None:
            self._cache.xi[0] *= self.penalty_update_factor

    def _update_inner_tolerance(self) -> None:
        """Shrink inner tolerance: ε̄ ← max(ε, β · ε̄)."""
        current = self._cache.panoc_cache.tolerance
        self._cache.panoc_cache.tolerance = max(
            current * self.epsilon_update_factor,
            self.epsilon_tolerance,
        )

    def _final_cache_update(self) -> None:
        """
        End-of-iteration bookkeeping: increment counter, shift
        infeasibility measures, copy y⁺ → y, and reset PANOC cache.
        """
        cache = self._cache
        cache.iteration += 1
        cache.delta_y_norm = cache.delta_y_norm_plus
        cache.f2_norm = cache.f2_norm_plus

        # Copy y⁺ into xi[1:]  (= update y)
        if cache.xi is not None and cache.y_plus is not None:
            cache.xi[1:] = cache.y_plus

        # Reset PANOC cache for next inner solve
        cache.panoc_cache.reset()

    def _compute_cost_at_solution(self, u: np.ndarray) -> float:
        """
        Compute the original cost f(u) at the solution, excluding
        penalty terms.  This is done by temporarily setting c = 0
        in xi (the augmented cost with c = 0 reduces to f(u) because
        the factory uses c̄ = max(1, c) in the denominator, so the
        penalty term becomes 0.5 * 0 * ... = 0).
        """
        if self._cache.xi is not None:
            saved_c = self._cache.xi[0]
            self._cache.xi[0] = 0.0
            cost = self._problem.parametric_cost(u, self._cache.xi)
            self._cache.xi[0] = saved_c
        else:
            cost = self._problem.parametric_cost(u, np.array([]))
        return cost
