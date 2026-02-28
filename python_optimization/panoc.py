"""
File: panoc.py

Description: This module implements the PANOC (Proximal Averaged Newton-type method
for Optimal Control) algorithm for solving constrained nonlinear optimization problems.

The implementation is based on the Rust optimization-engine library and adapted for Python/NumPy
with a simplified interface targeting nonlinear MPC applications with box constraints
(upper/lower bounds on decision variables).

PANOC solves problems of the form:

    min  f(u)
     u
    s.t. u_min <= u <= u_max

where f is a C^{1,1}-smooth cost function (gradient is Lipschitz continuous),
and the constraint set is a rectangle (box) defined by element-wise bounds.

The algorithm combines L-BFGS quasi-Newton directions with a forward-backward
splitting step (projected gradient) and uses a line search to ensure global
convergence. The key idea is to use the Forward-Backward Envelope (FBE) as a
merit function.

Algorithm outline (each iteration):
1. Compute the fixed-point residual (FPR): gamma_fpr = u - proj(u - gamma * grad_f(u))
2. Check convergence: ||gamma_fpr|| < tolerance
3. Update the Lipschitz constant estimate of the gradient
4. Compute an L-BFGS direction d = H * gamma_fpr
5. Perform a line search on tau in [0, 1]:
      u_plus = u - (1 - tau) * gamma_fpr - tau * d
   such that FBE(u_plus) <= FBE(u) - sigma * ||gamma_fpr||^2
6. Update: u <- u_plus

References:
    - optimization-engine: https://github.com/alphaville/optimization-engine
"""
import numpy as np
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Optional
import time

# ============================================================================
# Constants (matching optimization-engine defaults)
# ============================================================================
# Minimum estimated Lipschitz constant (initial estimate floor)
_MIN_L_ESTIMATE: float = 1e-10
# gamma = _GAMMA_L_COEFF / L
_GAMMA_L_COEFF: float = 0.95
# Delta for Lipschitz estimation perturbation
_DELTA_LIPSCHITZ: float = 1e-12
# Epsilon for Lipschitz estimation perturbation
_EPSILON_LIPSCHITZ: float = 1e-6
# Safety parameter for strict inequality in Lipschitz update
_LIPSCHITZ_UPDATE_EPSILON: float = 1e-6
# Maximum iterations for updating the Lipschitz constant
_MAX_LIPSCHITZ_UPDATE_ITERATIONS: int = 10
# Maximum possible Lipschitz constant
_MAX_LIPSCHITZ_CONSTANT: float = 1e9
# Maximum number of line-search iterations
_MAX_LINESEARCH_ITERATIONS: int = 10
# Default maximum PANOC iterations
_MAX_ITER_DEFAULT: int = 100

# L-BFGS defaults
_DEFAULT_SY_EPSILON: float = 1e-10
_DEFAULT_CBFGS_EPSILON: float = 1e-8
_DEFAULT_CBFGS_ALPHA: float = 1.0


# ============================================================================
# Exit status
# ============================================================================
class ExitStatus(Enum):
    """Exit status of the PANOC solver."""
    CONVERGED = auto()
    NOT_CONVERGED_ITERATIONS = auto()
    NOT_CONVERGED_OUT_OF_TIME = auto()
    NOT_FINITE_COMPUTATION = auto()


# ============================================================================
# Solver status (returned after solve)
# ============================================================================
@dataclass
class SolverStatus:
    """Result returned by :meth:`PANOCOptimizer.solve`.

    Attributes
    ----------
    exit_status : ExitStatus
        Reason the solver terminated.
    num_iter : int
        Number of iterations performed.
    solve_time : float
        Wall-clock time in seconds.
    norm_fpr : float
        Norm of the fixed-point residual at the solution (||gamma * FPR||).
    cost_value : float
        Cost function value at the solution.
    """
    exit_status: ExitStatus
    num_iter: int
    solve_time: float
    norm_fpr: float
    cost_value: float

    def has_converged(self) -> bool:
        return self.exit_status == ExitStatus.CONVERGED


# ============================================================================
# L-BFGS buffer  (two-loop recursion with C-BFGS safeguard)
# ============================================================================
class LBFGSBuffer:
    """Limited-memory BFGS buffer with C-BFGS safeguard.

    Stores pairs (s_k, y_k) and computes the product  H * q  using
    the standard two-loop recursion, where H is the L-BFGS approximation
    of the inverse Hessian.

    Parameters
    ----------
    problem_size : int
        Dimension of the decision variable.
    buffer_size : int
        Number of (s, y) pairs to store (L-BFGS memory).
    sy_epsilon : float
        Minimum accepted value of s^T y for an update to be stored.
    cbfgs_alpha : float
        C-BFGS parameter alpha (set <= 0 to disable).
    cbfgs_epsilon : float
        C-BFGS parameter epsilon (set <= 0 to disable).
    """

    def __init__(
        self,
        problem_size: int,
        buffer_size: int,
        sy_epsilon: float = _DEFAULT_SY_EPSILON,
        cbfgs_alpha: float = _DEFAULT_CBFGS_ALPHA,
        cbfgs_epsilon: float = _DEFAULT_CBFGS_EPSILON,
    ):
        assert problem_size > 0
        assert buffer_size > 0
        self._n = problem_size
        self._m = buffer_size
        self.sy_epsilon = sy_epsilon
        self.cbfgs_alpha = cbfgs_alpha
        self.cbfgs_epsilon = cbfgs_epsilon

        # Storage: buffer_size+1 entries (last is temporary workspace)
        self._s = np.zeros((buffer_size + 1, problem_size))
        self._y = np.zeros((buffer_size + 1, problem_size))
        self._rho = np.zeros(buffer_size + 1)
        # workspace for two-loop recursion
        self._alpha_buf = np.zeros(buffer_size)

        self._gamma: float = 1.0  # initial Hessian scaling H0 = gamma * I
        self._active_size: int = 0
        self._old_state = np.zeros(problem_size)
        self._old_g = np.zeros(problem_size)
        self._first_old: bool = True

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Clear the buffer (cheap – just resets flags)."""
        self._active_size = 0
        self._first_old = True

    # ------------------------------------------------------------------
    def update_hessian(self, g: np.ndarray, state: np.ndarray) -> bool:
        """Feed a new (gradient, state) pair to the buffer.

        Parameters
        ----------
        g : np.ndarray
            Current gradient (or FPR) vector.
        state : np.ndarray
            Current iterate (decision variable).

        Returns
        -------
        bool
            True if the pair was accepted, False if rejected.
        """
        if self._first_old:
            self._first_old = False
            self._old_state[:] = state
            self._old_g[:] = g
            return True

        # Compute s = state - old_state, y = g - old_g in the temporary slot
        tmp = self._m  # index of the temporary slot
        np.subtract(state, self._old_state, out=self._s[tmp])
        np.subtract(g, self._old_g, out=self._y[tmp])

        # Validity check
        if not self._new_s_and_y_valid(g, tmp):
            return False

        # Save current as "old"
        self._old_state[:] = state
        self._old_g[:] = g

        # Rotate: move temporary slot to front
        # shift all rows right by 1, temporary row becomes row 0
        self._s = np.roll(self._s, 1, axis=0)
        self._y = np.roll(self._y, 1, axis=0)
        self._rho = np.roll(self._rho, 1)

        # Update H0 scaling: gamma = (s^T y) / (y^T y)
        ys = np.dot(self._s[0], self._y[0])
        yy = np.dot(self._y[0], self._y[0])
        if yy > 0.0:
            self._gamma = ys / yy

        self._active_size = min(self._m, self._active_size + 1)
        return True

    # ------------------------------------------------------------------
    def _new_s_and_y_valid(self, g: np.ndarray, idx: int) -> bool:
        """Check C-BFGS and curvature conditions for the (s, y) pair at *idx*."""
        s = self._s[idx]
        y = self._y[idx]
        ys = float(np.dot(s, y))
        norm_s_sq = float(np.dot(s, s))

        if norm_s_sq <= np.finfo(float).tiny:
            return False
        if self.sy_epsilon > 0.0 and ys <= self.sy_epsilon:
            return False

        self._rho[idx] = 1.0 / ys

        if self.cbfgs_epsilon > 0.0 and self.cbfgs_alpha > 0.0:
            lhs = ys / norm_s_sq
            rhs = self.cbfgs_epsilon * (np.linalg.norm(g) ** self.cbfgs_alpha)
            if lhs <= rhs:
                return False

        return True

    # ------------------------------------------------------------------
    def apply_hessian(self, q: np.ndarray) -> None:
        """Apply the L-BFGS inverse Hessian approximation **in-place**.

        On entry *q* is the gradient (or FPR); on exit it contains H * q.
        Uses the standard two-loop recursion.
        """
        if self._active_size == 0:
            return  # no curvature info yet – return q unchanged

        k = self._active_size
        alpha = self._alpha_buf

        # --- forward pass ---
        for i in range(k):
            alpha[i] = self._rho[i] * np.dot(self._s[i], q)
            q -= alpha[i] * self._y[i]  # q = q - alpha_i * y_i

        # Apply H0 = gamma * I
        q *= self._gamma

        # --- backward pass ---
        for i in range(k - 1, -1, -1):
            beta = self._rho[i] * np.dot(self._y[i], q)
            # q = q + (alpha_i - beta) * s_i
            q += (alpha[i] - beta) * self._s[i]


# ============================================================================
# PANOC Cache (pre-allocated working memory)
# ============================================================================
class PANOCCache:
    """Pre-allocated working arrays for the PANOC algorithm.

    Create once and reuse across multiple ``solve`` calls to avoid
    repeated memory allocation.

    Parameters
    ----------
    problem_size : int
        Dimension of the decision variable vector *u*.
    tolerance : float
        Convergence tolerance on ||gamma * FPR||.
    lbfgs_memory : int
        L-BFGS memory size (number of stored pairs).
    """

    def __init__(self, problem_size: int, tolerance: float, lbfgs_memory: int):
        assert tolerance > 0.0, "tolerance must be positive"
        n = problem_size
        self.tolerance: float = tolerance

        # L-BFGS buffer
        self.lbfgs = LBFGSBuffer(
            n, lbfgs_memory,
            sy_epsilon=_DEFAULT_SY_EPSILON,
            cbfgs_alpha=_DEFAULT_CBFGS_ALPHA,
            cbfgs_epsilon=_DEFAULT_CBFGS_EPSILON,
        )

        # Working arrays
        self.gradient_u = np.zeros(n)
        self.u_half_step = np.zeros(n)
        self.gradient_step = np.zeros(n)
        self.direction_lbfgs = np.zeros(n)
        self.u_plus = np.zeros(n)
        self.gamma_fpr = np.zeros(n)

        # Scalars
        self.gamma: float = 0.0
        self.norm_gamma_fpr: float = np.inf
        self.tau: float = 1.0
        self.lipschitz_constant: float = 0.0
        self.sigma: float = 0.0
        self.cost_value: float = 0.0
        self.rhs_ls: float = 0.0
        self.lhs_ls: float = 0.0
        self.iteration: int = 0

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset the cache to its initial state (called before each solve)."""
        self.lbfgs.reset()
        self.lhs_ls = 0.0
        self.rhs_ls = 0.0
        self.tau = 1.0
        self.lipschitz_constant = 0.0
        self.sigma = 0.0
        self.cost_value = 0.0
        self.iteration = 0
        self.gamma = 0.0
        self.norm_gamma_fpr = np.inf

    # ------------------------------------------------------------------
    def exit_condition(self) -> bool:
        """Check FPR convergence: ||gamma * FPR|| < tolerance."""
        return self.norm_gamma_fpr < self.tolerance


# ============================================================================
# PANOC Optimizer
# ============================================================================
class PANOCOptimizer:
    """PANOC solver for box-constrained nonlinear optimisation.

    Solves::

        min  cost_func(u)
         u
        s.t. u_min <= u <= u_max   (element-wise)

    Parameters
    ----------
    cost_func : callable
        ``cost_func(u) -> float`` – evaluates the cost at *u*.
    gradient_func : callable
        ``gradient_func(u) -> ndarray`` – returns the gradient of the cost at *u*.
    cache : PANOCCache
        Pre-allocated cache (create with :class:`PANOCCache`).
    u_min : np.ndarray or None
        Element-wise lower bounds.  ``None`` means no lower bound.
    u_max : np.ndarray or None
        Element-wise upper bounds.  ``None`` means no upper bound.
    max_iter : int
        Maximum number of PANOC iterations (default 100).
    max_duration : float or None
        Maximum wall-clock time in seconds (``None`` = no limit).
    tolerance : float or None
        Override the convergence tolerance stored in *cache*.

    Example
    -------
    >>> n = 2
    >>> cache = PANOCCache(n, tolerance=1e-6, lbfgs_memory=8)
    >>> cost = lambda u: (1 - u[0])**2 + 200*(u[1] - u[0]**2)**2
    >>> def grad(u):
    ...     g = np.zeros(2)
    ...     g[0] = -2*(1-u[0]) - 800*u[0]*(u[1]-u[0]**2)
    ...     g[1] = 400*(u[1]-u[0]**2)
    ...     return g
    >>> solver = PANOCOptimizer(cost, grad, cache,
    ...                         u_min=np.array([-2.0, -2.0]),
    ...                         u_max=np.array([2.0, 2.0]))
    >>> u0 = np.array([-1.5, 0.9])
    >>> status = solver.solve(u0)
    >>> status.has_converged()
    True
    """

    def __init__(
        self,
        cost_func: Callable[[np.ndarray], float],
        gradient_func: Callable[[np.ndarray], np.ndarray],
        cache: PANOCCache,
        u_min: Optional[np.ndarray] = None,
        u_max: Optional[np.ndarray] = None,
        max_iter: int = _MAX_ITER_DEFAULT,
        max_duration: Optional[float] = None,
        tolerance: Optional[float] = None,
    ):
        assert max_iter > 0, "max_iter must be > 0"
        assert u_min is not None or u_max is not None, \
            "At least one of u_min / u_max must be provided"

        self._cost_func = cost_func
        self._gradient_func = gradient_func
        self._cache = cache
        self._u_min = u_min
        self._u_max = u_max
        self.max_iter = max_iter
        self.max_duration = max_duration

        if tolerance is not None:
            assert tolerance > 0.0
            self._cache.tolerance = tolerance

    # ==================================================================
    #  Public API
    # ==================================================================
    def solve(self, u: np.ndarray) -> SolverStatus:
        """Run PANOC starting from initial guess *u*.

        Parameters
        ----------
        u : np.ndarray
            Initial guess (modified **in-place** with the solution on return).

        Returns
        -------
        SolverStatus
            Information about the solve (convergence, iterations, etc.).
        """
        t_start = time.perf_counter()
        c = self._cache
        c.reset()

        # --- Initialisation (equivalent to PANOCEngine::init) ---
        c.cost_value = self._cost_func(u)
        self._estimate_local_lipschitz(u)  # also fills c.gradient_u
        c.gamma = _GAMMA_L_COEFF / max(c.lipschitz_constant, _MIN_L_ESTIMATE)
        c.sigma = (1.0 - _GAMMA_L_COEFF) / (4.0 * c.gamma)
        self._gradient_step(u)
        self._half_step()

        # --- Main loop ---
        num_iter = 0
        converged = False

        while True:
            # --- One PANOC step ---
            # 1. Compute FPR
            self._compute_fpr(u)

            # 2. Check exit condition
            if c.exit_condition():
                converged = True
                break

            # 3. Update Lipschitz constant
            self._update_lipschitz_constant(u)

            # 4. L-BFGS direction
            self._lbfgs_direction(u)

            # 5. First iteration ⇒ no line search; otherwise ⇒ line search
            if c.iteration == 0:
                self._update_no_linesearch(u)
            else:
                self._linesearch(u)

            c.iteration += 1
            num_iter += 1

            # Iteration limit
            if num_iter >= self.max_iter:
                break

            # Time limit
            if self.max_duration is not None:
                if time.perf_counter() - t_start > self.max_duration:
                    break

        # --- Determine exit status ---
        if not np.all(np.isfinite(u)):
            exit_status = ExitStatus.NOT_FINITE_COMPUTATION
        elif converged:
            exit_status = ExitStatus.CONVERGED
        elif num_iter >= self.max_iter:
            exit_status = ExitStatus.NOT_CONVERGED_ITERATIONS
        else:
            exit_status = ExitStatus.NOT_CONVERGED_OUT_OF_TIME

        # Return the feasible half-step (always satisfies constraints)
        u[:] = c.u_half_step

        elapsed = time.perf_counter() - t_start
        return SolverStatus(
            exit_status=exit_status,
            num_iter=num_iter,
            solve_time=elapsed,
            norm_fpr=c.norm_gamma_fpr,
            cost_value=c.cost_value,
        )

    # ==================================================================
    #  Projection (box constraints)
    # ==================================================================
    def _project(self, x: np.ndarray) -> None:
        """Project *x* onto the box [u_min, u_max] **in-place**."""
        if self._u_min is not None:
            np.maximum(x, self._u_min, out=x)
        if self._u_max is not None:
            np.minimum(x, self._u_max, out=x)

    # ==================================================================
    #  Lipschitz estimation (numerical directional derivative)
    # ==================================================================
    def _estimate_local_lipschitz(self, u: np.ndarray) -> None:
        """Estimate the local Lipschitz constant of the gradient at *u*.

        Also fills ``cache.gradient_u`` with grad f(u).

        The estimate is: L = ||grad(u+h) - grad(u)|| / ||h||
        where h_i = max(delta, epsilon * u_i).
        """
        c = self._cache
        delta = _DELTA_LIPSCHITZ
        epsilon = _EPSILON_LIPSCHITZ

        # Evaluate gradient at u
        c.gradient_u[:] = self._gradient_func(u)

        # Build perturbation h
        h = np.maximum(delta, epsilon * u)
        norm_h = np.linalg.norm(h)

        # Evaluate gradient at u + h
        grad_perturbed = self._gradient_func(u + h)

        # L = ||grad(u+h) - grad(u)|| / ||h||
        c.lipschitz_constant = np.linalg.norm(
            grad_perturbed - c.gradient_u) / norm_h

    # ==================================================================
    #  Core sub-steps
    # ==================================================================
    def _compute_fpr(self, u: np.ndarray) -> None:
        """Compute the fixed-point residual: gamma_fpr = u - u_half_step."""
        c = self._cache
        np.subtract(u, c.u_half_step, out=c.gamma_fpr)
        c.norm_gamma_fpr = np.linalg.norm(c.gamma_fpr)

    def _gradient_step(self, u: np.ndarray) -> None:
        """gradient_step = u - gamma * gradient_u."""
        c = self._cache
        np.subtract(u, c.gamma * c.gradient_u, out=c.gradient_step)

    def _gradient_step_uplus(self) -> None:
        """gradient_step = u_plus - gamma * gradient_u."""
        c = self._cache
        np.subtract(c.u_plus, c.gamma * c.gradient_u, out=c.gradient_step)

    def _half_step(self) -> None:
        """u_half_step = project(gradient_step) onto the constraint set."""
        c = self._cache
        c.u_half_step[:] = c.gradient_step
        self._project(c.u_half_step)

    def _lbfgs_direction(self, u: np.ndarray) -> None:
        """Update L-BFGS buffer and compute direction = H * gamma_fpr."""
        c = self._cache
        c.lbfgs.update_hessian(c.gamma_fpr, u)
        if c.iteration > 0:
            c.direction_lbfgs[:] = c.gamma_fpr
            c.lbfgs.apply_hessian(c.direction_lbfgs)

    # ------------------------------------------------------------------
    #  Lipschitz constant update
    # ------------------------------------------------------------------
    def _lipschitz_check_rhs(self) -> float:
        """RHS of the Lipschitz update condition.

        rhs = f(u) + eps*|f(u)| - <grad, gamma_fpr> + (L_coeff / (2*gamma)) * ||gamma_fpr||^2
        """
        c = self._cache
        inner = float(np.dot(c.gradient_u, c.gamma_fpr))
        return (c.cost_value
                + _LIPSCHITZ_UPDATE_EPSILON * abs(c.cost_value)
                - inner
                + (_GAMMA_L_COEFF / (2.0 * c.gamma)) * c.norm_gamma_fpr ** 2)

    def _update_lipschitz_constant(self, u: np.ndarray) -> None:
        """Update the Lipschitz constant estimate (and gamma, sigma accordingly)."""
        c = self._cache

        cost_half = self._cost_func(c.u_half_step)
        c.cost_value = self._cost_func(u)

        it_lip = 0
        while (cost_half > self._lipschitz_check_rhs()
               and it_lip < _MAX_LIPSCHITZ_UPDATE_ITERATIONS
               and c.lipschitz_constant < _MAX_LIPSCHITZ_CONSTANT):
            c.lbfgs.reset()
            c.lipschitz_constant *= 2.0
            c.gamma /= 2.0

            self._gradient_step(u)
            self._half_step()
            cost_half = self._cost_func(c.u_half_step)
            self._compute_fpr(u)
            it_lip += 1

        c.sigma = (1.0 - _GAMMA_L_COEFF) / (4.0 * c.gamma)

    # ------------------------------------------------------------------
    #  u_plus computation
    # ------------------------------------------------------------------
    def _compute_u_plus(self, u: np.ndarray) -> None:
        """u_plus = u - (1 - tau)*gamma_fpr - tau * direction_lbfgs."""
        c = self._cache
        temp = 1.0 - c.tau
        c.u_plus[:] = u - temp * c.gamma_fpr - c.tau * c.direction_lbfgs

    # ------------------------------------------------------------------
    #  Forward-Backward Envelope helpers
    # ------------------------------------------------------------------
    def _compute_rhs_ls(self) -> None:
        """Compute the RHS of the line-search condition (FBE - sigma * ||fpr||^2)."""
        c = self._cache
        dist_sq = float(np.sum((c.gradient_step - c.u_half_step) ** 2))
        grad_norm_sq = float(np.dot(c.gradient_u, c.gradient_u))
        fbe = c.cost_value - 0.5 * c.gamma * grad_norm_sq + 0.5 * dist_sq / c.gamma
        c.rhs_ls = fbe - c.sigma * c.norm_gamma_fpr ** 2

    def _line_search_condition(self, u: np.ndarray) -> bool:
        """Evaluate the line-search condition.

        Returns True if lhs > rhs (line search should continue).
        Side effects: updates u_plus, cost_value, gradient_u,
        gradient_step, u_half_step, lhs_ls.
        """
        c = self._cache

        # Candidate next iterate
        self._compute_u_plus(u)

        # Evaluate cost and gradient at u_plus
        c.cost_value = self._cost_func(c.u_plus)
        c.gradient_u[:] = self._gradient_func(c.u_plus)

        # Gradient step and half step at u_plus
        self._gradient_step_uplus()
        self._half_step()

        # LHS of line-search condition (FBE at u_plus)
        dist_sq = float(np.sum((c.gradient_step - c.u_half_step) ** 2))
        grad_norm_sq = float(np.dot(c.gradient_u, c.gradient_u))
        c.lhs_ls = c.cost_value - 0.5 * c.gamma * \
            grad_norm_sq + 0.5 * dist_sq / c.gamma

        return c.lhs_ls > c.rhs_ls

    # ------------------------------------------------------------------
    #  Line search / no-line-search update
    # ------------------------------------------------------------------
    def _update_no_linesearch(self, u: np.ndarray) -> None:
        """First-iteration update (no line search): u <- u_half_step."""
        c = self._cache
        u[:] = c.u_half_step
        c.cost_value = self._cost_func(u)
        c.gradient_u[:] = self._gradient_func(u)
        self._gradient_step(u)
        self._half_step()

    def _linesearch(self, u: np.ndarray) -> None:
        """Perform a line search on tau to select the next iterate."""
        c = self._cache
        self._compute_rhs_ls()
        c.tau = 1.0
        num_ls = 0
        while self._line_search_condition(u) and num_ls < _MAX_LINESEARCH_ITERATIONS:
            c.tau /= 2.0
            num_ls += 1

        if num_ls == _MAX_LINESEARCH_ITERATIONS:
            # Fall back to projected gradient step
            c.tau = 0.0
            u[:] = c.u_half_step
        # Accept the candidate
        u[:] = c.u_plus
