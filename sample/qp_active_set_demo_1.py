"""
This script demonstrates how to solve a quadratic programming (QP) problem using the QP_ActiveSetSolver class from the python_optimization.qp_active_set module.
It sets up a simple QP with two variables and four linear inequality constraints, then solves for the optimal solution using the active set method.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import numpy as np

from python_optimization.qp_active_set import QP_ActiveSetSolver

# QP parameters
E = np.eye(2)
L = np.array([[5.0], [8.0]])
M = np.array([[1, 0],   # x <= 4
              [0, 1],   # y <= 6
              [-1, 0],   # x >= 0  (−x <= 0)
              [0, -1]])  # y >= 0  (−y <= 0)

gamma = np.array([[4.0],
                  [6.0],
                  [0.0],
                  [0.0]])

# Run solver
solver = QP_ActiveSetSolver(number_of_variables=E.shape[0],
                            number_of_constraints=M.shape[0]
                            )
x_opt = solver.solve(E, L, M, gamma)

print("Optimal solution: x =", x_opt)
