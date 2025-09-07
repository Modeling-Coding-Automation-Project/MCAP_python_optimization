import numpy as np


class ActiveSet:
    """
    A class that manages an active set of constraints with a
    fixed number of constraints, ensuring safe management of the active set information.

    Attributes
    ----------
    active_flags : np.ndarray of bool
        An array indicating whether each constraint is active (length: number of constraints).
    active_indices : np.ndarray of int
        An array storing the indices of active constraints
        (length: number of constraints, unused parts are set to 0, etc.).
    number_of_active : int
        The current number of active constraints.
    """

    def __init__(self, number_of_constraints: int):
        self.number_of_constraints = number_of_constraints

        self._active_flags = np.zeros(number_of_constraints, dtype=bool)
        self._active_indices = np.zeros(number_of_constraints, dtype=int)
        self._number_of_active = 0

    def push_active(self, index: int):
        """
        Marks the constraint at the specified index as active and adds it to the list of active constraints.

        Args:
            index (int): The index of the constraint to activate.

        Notes:
            - If the constraint at the given index is already active, this method does nothing.
            - Updates the internal flags and indices to reflect the activation.
        """
        if not self._active_flags[index]:
            self._active_flags[index] = True
            self._active_indices[self._number_of_active] = index
            self._number_of_active += 1

    def push_inactive(self, index: int):
        """
        Marks the constraint at the specified index as inactive and removes it from the list of active constraints.
        Args:
            index (int): The index of the constraint to deactivate.
        Notes:
            - If the constraint at the given index is not active, this method does nothing.
            - Updates the internal flags and indices to reflect the deactivation.
        """
        if self._active_flags[index]:
            self._active_flags[index] = False
            found = False
            for i in range(self._number_of_active):
                if not found and self._active_indices[i] == index:
                    found = True
                if found and i < self._number_of_active - 1:
                    self._active_indices[i] = self._active_indices[i + 1]
            if found:
                self._active_indices[self._number_of_active - 1] = 0
                self._number_of_active -= 1

    def get_active(self, index: int):
        """
        Returns the index of the active constraint at the specified position in the active set.
        Args:
            index (int): The position in the active set to retrieve the index from.
        Returns:
            int: The index of the active constraint at the specified position.
        Raises:
            IndexError: If the index is out of bounds for the active set.
        """
        if index < 0 or index >= self._number_of_active:
            raise IndexError("Index out of bounds for active set.")
        return self._active_indices[index]

    def get_active_indices(self):
        """
        Returns the indices of all currently active constraints.
        Returns:
            np.ndarray: An array of indices of active constraints.
        """
        return self._active_indices

    def get_number_of_active(self):
        """
        Returns the number of currently active constraints.
        Returns:
            int: The number of active constraints.
        """
        return self._number_of_active

    def is_active(self, index: int):
        """
        Checks if the constraint at the specified index is currently active.
        Args:
            index (int): The index of the constraint to check.
        Returns:
            bool: True if the constraint is active, False otherwise.
        """
        return self._active_flags[index]


class ActiveSet2D:
    """
    A class that manages active elements in a 2D array (matrix).
    Each active element is represented as a (col, row) pair.

    Attributes:
    number_of_columns, number_of_rows : int
        The number of cols and rows in the target matrix.
    _active_flags : np.ndarray (bool, shape=(number_of_columns, number_of_rows))
        A boolean array indicating the active state of each element.
    _active_pairs : np.ndarray (int, shape=(number_of_columns*number_of_rows, 2))
        An array storing the active (col, row) pairs (unused parts are [0, 0]).
    _number_of_active : int
        The current number of active elements.
    """

    def __init__(self, number_of_columns: int, number_of_rows: int):
        if number_of_columns <= 0 or number_of_rows <= 0:
            raise ValueError(
                "number_of_columns and number_of_rows must be positive.")
        self.number_of_columns = number_of_columns
        self.number_of_rows = number_of_rows

        self._active_flags = np.zeros(
            (number_of_columns, number_of_rows), dtype=bool)
        self._active_pairs = np.zeros(
            (number_of_columns * number_of_rows, 2), dtype=int)
        self._number_of_active = 0

    def _check_bounds(self, col: int, row: int):
        """
        Checks whether the specified column and row indices are
          within the valid bounds of the matrix.
        Args:
            col (int): The column index to check.
            row (int): The row index to check.
        Raises:
            IndexError: If either the column or row index is out of bounds.
        """

        if not (0 <= col < self.number_of_columns) or not (0 <= row < self.number_of_rows):
            raise IndexError("Column/row index out of bounds.")

    def push_active(self, col: int, row: int):
        """
        Activates the element at (col, row) and adds it to the active set.
        Does nothing if it is already active.
        """

        self._check_bounds(col, row)
        if not self._active_flags[col, row]:
            self._active_flags[col, row] = True
            self._active_pairs[self._number_of_active] = (col, row)
            self._number_of_active += 1

    def push_inactive(self, col: int, row: int):
        """
        Deactivates the element at (col, row) and removes it from the active set.
        Does nothing if it is already inactive.
        """

        self._check_bounds(col, row)
        if self._active_flags[col, row]:
            self._active_flags[col, row] = False

            found = False
            for i in range(self._number_of_active):
                if not found and (self._active_pairs[i, 0] == col and self._active_pairs[i, 1] == row):
                    found = True
                if found and i < self._number_of_active - 1:
                    self._active_pairs[i] = self._active_pairs[i + 1]

            if found:
                self._active_pairs[self._number_of_active - 1] = (0, 0)
                self._number_of_active -= 1

    def get_active(self, index: int):
        """
        Returns the (col, row) pair at the specified index in the active set.

        Args:
            index (int): The index in the active set to retrieve the (col, row) pair from.
        Returns:
            tuple: A tuple (col, row) representing the active element at the specified index.
        """

        if index < 0 or index >= self._number_of_active:
            raise IndexError("Index out of bounds for active set.")
        r, c = self._active_pairs[index]
        return int(r), int(c)

    def get_active_pairs(self):
        """
        Returns the entire array of active (col, row) pairs (unused parts remain as [0, 0]).
        Note: This follows the same approach as the 1D version's get_active_indices.
        """
        return self._active_pairs

    def get_number_of_active(self):
        """
        Returns the current number of active elements.
        """
        return self._number_of_active

    def is_active(self, col: int, row: int):
        """
        Returns whether the element at (col, row) is active.
        """
        self._check_bounds(col, row)
        return bool(self._active_flags[col, row])

    def clear(self):
        """
        Clears all active elements, resetting the active set.
        """
        self._active_flags = np.zeros(
            (self.number_of_columns, self.number_of_rows), dtype=bool)
        self._active_pairs = np.zeros(
            (self.number_of_columns * self.number_of_rows, 2), dtype=int)
        self._number_of_active = 0


class ActiveSet2D_MatrixOperator:
    @staticmethod
    def element_wise_product(
        A: np.ndarray,
        B: np.ndarray,
        active_set: ActiveSet2D
    ) -> np.ndarray:
        """
        Returns a matrix where only the elements registered in
          active_set (active pairs) are the product A[i, j] * B[i, j].
        Other elements are zero.
        """
        result = np.zeros_like(A)

        for idx in range(active_set.get_number_of_active()):
            i, j = active_set.get_active(idx)
            result[i, j] = A[i, j] * B[i, j]
        return result

    @staticmethod
    def vdot(
        A: np.ndarray,
        B: np.ndarray,
        active_set: ActiveSet2D
    ) -> float:
        """
        Returns the sum of products A[i, j] * B[i, j] only for
          elements registered in active_set (active pairs).
        Equivalent to numpy.vdot for the active elements.
        """
        total = 0.0

        for idx in range(active_set.get_number_of_active()):
            i, j = active_set.get_active(idx)
            total += A[i, j] * B[i, j]
        return total
