"""Regressions for constraint numbers versus positions in an active set."""
import unittest

import numpy as np

from python_optimization.active_set import ActiveSet, ActiveSet2D
from python_optimization.qp_active_set import QP_ActiveSetSolver


class ActiveSetIndicesTest(unittest.TestCase):
    def test_position_removal_preserves_order_and_flags(self):
        active = ActiveSet(6)
        for constraint in (5, 2, 4, 1):
            active.push_active(constraint)
        active.push_active(2)  # Duplicates must not consume positions.
        active.deactivate_at(1)
        self.assertEqual([active.get_active(i) for i in range(3)], [5, 4, 1])
        self.assertFalse(active.is_active(2))
        active.push_inactive(index=4)  # The original API still uses a number.
        active.push_inactive(0)  # Already inactive.
        self.assertEqual([active.get_active(i) for i in range(2)], [5, 1])
        active.deactivate_at(0)
        self.assertEqual(active.get_active(index=0), 1)
        active.deactivate_at(0)
        self.assertEqual(active.get_number_of_active(), 0)
        np.testing.assert_array_equal(active.get_active_indices(), np.zeros(6))
        active.push_active(index=2)
        self.assertEqual(active.get_active(0), 2)

    def test_invalid_positions_do_not_change_the_set(self):
        for count in (0, 1):
            active = ActiveSet(4)
            if count:
                active.push_active(3)
            for position in (-1, count, count + 1):
                for operation in (active.get_active, active.deactivate_at):
                    with self.subTest(count=count, position=position):
                        with self.assertRaises(IndexError):
                            operation(position)
                        self.assertEqual(active.get_number_of_active(), count)
            self.assertEqual(bool(active.is_active(3)), bool(count))
        for operation in (active.push_active, active.push_inactive, active.is_active):
            for constraint in (-1, 4):
                with self.assertRaises(IndexError):
                    operation(constraint)
        empty = ActiveSet(0)
        with self.assertRaises(IndexError):
            empty.get_active(0)

    def test_2d_position_removal_and_coordinate_removal(self):
        active = ActiveSet2D(3, 3)
        for pair in ((2, 1), (0, 2), (1, 0), (2, 2)):
            active.push_active(*pair)
        active.push_active(0, 2)
        active.deactivate_at(1)
        self.assertFalse(active.is_active(0, 2))
        self.assertEqual([active.get_active(i) for i in range(3)],
                         [(2, 1), (1, 0), (2, 2)])
        active.push_inactive(1, 0)
        active.push_inactive(0, 0)
        active.deactivate_at(0)
        self.assertEqual(active.get_active(0), (2, 2))
        active.deactivate_at(0)
        self.assertEqual(active.get_number_of_active(), 0)
        np.testing.assert_array_equal(
            active.get_active_pairs(), np.zeros((9, 2)))
        active.push_active(2, 1)
        self.assertEqual(active.get_active(0), (2, 1))

    def test_2d_invalid_positions_and_coordinates(self):
        active = ActiveSet2D(3, 2)
        for operation in (active.get_active, active.deactivate_at):
            with self.assertRaises(IndexError):
                operation(0)
        active.push_active(2, 1)
        for operation in (active.get_active, active.deactivate_at):
            for position in (-1, 1, 2):
                with self.assertRaises(IndexError):
                    operation(position)
        for operation in (active.push_active, active.push_inactive, active.is_active):
            for pair in ((-1, 0), (3, 0), (0, -1), (0, 2)):
                with self.assertRaises(IndexError):
                    operation(*pair)
        self.assertEqual(active.get_active(0), (2, 1))
        active.clear()
        with self.assertRaises(IndexError):
            active.get_active(0)


class RecordingActiveSet(ActiveSet):
    def __init__(self, count):
        super().__init__(count)
        self.removed_positions = []

    def deactivate_at(self, active_position):
        self.removed_positions.append(
            (active_position, self.get_active(active_position)))
        super().deactivate_at(active_position)


class QPIndicesTest(unittest.TestCase):
    def test_remove_nonzero_constraint_number_from_position_zero(self):
        active = RecordingActiveSet(3)
        active.push_active(2)
        solver = QP_ActiveSetSolver(1, 3, active_set=active, max_iteration=10)
        result = solver.solve(E=np.eye(1), L=np.array([[0.0]]),
                              M=np.array([[-1.0], [1.0], [1.0]]),
                              gamma=np.array([[1.0], [3.0], [1.0]]))
        np.testing.assert_allclose(result, [[0.0]], atol=1e-8)
        self.assertIn((0, 2), active.removed_positions)
        self.assertLess(solver.iteration_count, solver.max_iteration)
        self.assertEqual(active.get_number_of_active(), 0)

    def test_remove_second_active_constraint_and_keep_the_first(self):
        active = RecordingActiveSet(4)
        active.push_active(2)
        active.push_active(3)
        solver = QP_ActiveSetSolver(2, 4, active_set=active, max_iteration=10)
        result = solver.solve(E=np.diag([2.0, 3.0]), L=np.array([[4.0], [1.5]]),
                              M=np.array(
                                  [[-1., 0.], [0., -1.], [1., 0.], [0., 1.]]),
                              gamma=np.array([[0.], [0.], [1.], [1.]]))
        np.testing.assert_allclose(result, [[1.0], [0.5]], atol=1e-8)
        self.assertEqual(active.removed_positions[0], (1, 3))
        self.assertLess(solver.iteration_count, solver.max_iteration)


if __name__ == '__main__':
    unittest.main()
