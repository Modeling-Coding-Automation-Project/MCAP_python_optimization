"""
This script demonstrates how to solve a quadratic programming (QP) problem using the QP_ActiveSetSolver class from the python_optimization.qp_active_set module.
It sets up a QP problem with a quadratic cost matrix, a linear term, constraint matrix, and constraint bounds, then solves for the optimal variable vector.
"""
import os

import sys
sys.path.append(os.getcwd())

import numpy as np

from python_optimization.qp_active_set import QP_ActiveSetSolver

# QP parameters
E = np.eye(3)

L = np.array([[2.0],
              [3.0],
              [1.0]
              ])

M = np.array([[1.0, 1.0, 1.0],
              [3.0, -2.0, -3.0],
              [1.0, -3.0, 2.0]
              ])

gamma = np.array([[1.0],
                  [1.0],
                  [1.0]
                  ])

# Run solver
solver = QP_ActiveSetSolver(number_of_variables=E.shape[0],
                            number_of_constraints=M.shape[0]
                            )
x_opt = solver.solve(E, L, M, gamma)

print("Optimal solution: x =", x_opt)
