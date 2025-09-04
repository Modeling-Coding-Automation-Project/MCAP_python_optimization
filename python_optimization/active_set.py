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
    Each active element is represented as a (row, col) pair.

    Attributes:
    n_rows, n_cols : int
        The number of rows and columns in the target matrix.
    _active_flags : np.ndarray (bool, shape=(n_rows, n_cols))
        A boolean array indicating the active state of each element.
    _active_pairs : np.ndarray (int, shape=(n_rows*n_cols, 2))
        An array storing the active (row, col) pairs (unused parts are [0, 0]).
    _number_of_active : int
        The current number of active elements.
    """

    def __init__(self, n_rows: int, n_cols: int):
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive.")
        self.n_rows = n_rows
        self.n_cols = n_cols

        self._active_flags = np.zeros((n_rows, n_cols), dtype=bool)
        self._active_pairs = np.zeros((n_rows * n_cols, 2), dtype=int)
        self._number_of_active = 0

    def _check_bounds(self, row: int, col: int):
        """
        Checks whether the specified row and column indices are within the valid bounds of the matrix.
        Args:
            row (int): The row index to check.
            col (int): The column index to check.
        Raises:
            IndexError: If either the row or column index is out of bounds.
        """

        if not (0 <= row < self.n_rows) or not (0 <= col < self.n_cols):
            raise IndexError("Row/column index out of bounds.")

    def push_active(self, row: int, col: int):
        """
        Activates the element at (row, col) and adds it to the active set.
        Does nothing if it is already active.
        """

        self._check_bounds(row, col)
        if not self._active_flags[row, col]:
            self._active_flags[row, col] = True
            self._active_pairs[self._number_of_active] = (row, col)
            self._number_of_active += 1

    def push_inactive(self, row: int, col: int):
        """
        Deactivates the element at (row, col) and removes it from the active set.
        Does nothing if it is already inactive.
        """

        self._check_bounds(row, col)
        if self._active_flags[row, col]:
            self._active_flags[row, col] = False

            found = False
            for i in range(self._number_of_active):
                if not found and (self._active_pairs[i, 0] == row and self._active_pairs[i, 1] == col):
                    found = True
                if found and i < self._number_of_active - 1:
                    self._active_pairs[i] = self._active_pairs[i + 1]

            if found:
                self._active_pairs[self._number_of_active - 1] = (0, 0)
                self._number_of_active -= 1

    def get_active(self, index: int):
        """
        Returns the (row, col) pair at the specified index in the active set.

        Args:
            index (int): The index in the active set to retrieve the (row, col) pair from.
        Returns:
            tuple: A tuple (row, col) representing the active element at the specified index.
        """

        if index < 0 or index >= self._number_of_active:
            raise IndexError("Index out of bounds for active set.")
        r, c = self._active_pairs[index]
        return int(r), int(c)

    def get_active_pairs(self):
        """
        Returns the entire array of active (row, col) pairs (unused parts remain as [0, 0]).
        Note: This follows the same approach as the 1D version's get_active_indices.
        """
        return self._active_pairs

    def get_number_of_active(self):
        """
        Returns the current number of active elements.
        """
        return self._number_of_active

    def is_active(self, row: int, col: int):
        """
        Returns whether the element at (row, col) is active.
        """
        self._check_bounds(row, col)
        return bool(self._active_flags[row, col])
