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
    2次元配列（行列）のアクティブ要素を管理するクラス。
    各アクティブ要素は (row, col) のペアで保持します。

    Attributes
    ----------
    n_rows, n_cols : int
        対象行列の行数・列数。
    _active_flags : np.ndarray (bool, shape=(n_rows, n_cols))
        各要素のアクティブ状態。
    _active_pairs : np.ndarray (int, shape=(n_rows*n_cols, 2))
        アクティブな (row, col) ペアを格納（未使用部分は [0, 0]）。
    _number_of_active : int
        現在のアクティブ要素数。
    """

    def __init__(self, n_rows: int, n_cols: int):
        if n_rows <= 0 or n_cols <= 0:
            raise ValueError("n_rows and n_cols must be positive.")
        self.n_rows = n_rows
        self.n_cols = n_cols

        self._active_flags = np.zeros((n_rows, n_cols), dtype=bool)
        self._active_pairs = np.zeros((n_rows * n_cols, 2), dtype=int)
        self._number_of_active = 0

    # ---- helpers ----
    def _check_bounds(self, row: int, col: int):
        if not (0 <= row < self.n_rows) or not (0 <= col < self.n_cols):
            raise IndexError("Row/column index out of bounds.")

    # ---- core ops ----
    def push_active(self, row: int, col: int):
        """
        (row, col) の要素をアクティブにし、アクティブ集合に追加。
        既にアクティブなら何もしません。
        """
        self._check_bounds(row, col)
        if not self._active_flags[row, col]:
            self._active_flags[row, col] = True
            self._active_pairs[self._number_of_active] = (row, col)
            self._number_of_active += 1

    def push_inactive(self, row: int, col: int):
        """
        (row, col) の要素を非アクティブにし、アクティブ集合から削除。
        既に非アクティブなら何もしません。
        """
        self._check_bounds(row, col)
        if self._active_flags[row, col]:
            self._active_flags[row, col] = False

            found = False
            # 見つけた位置以降を1つずつ前に詰める（1D版と同様の方針）
            for i in range(self._number_of_active):
                if not found and (self._active_pairs[i, 0] == row and self._active_pairs[i, 1] == col):
                    found = True
                if found and i < self._number_of_active - 1:
                    self._active_pairs[i] = self._active_pairs[i + 1]

            if found:
                self._active_pairs[self._number_of_active - 1] = (0, 0)
                self._number_of_active -= 1

    # ---- queries ----
    def get_active(self, index: int):
        """
        アクティブ集合の index 番目にある (row, col) を返します。
        """
        if index < 0 or index >= self._number_of_active:
            raise IndexError("Index out of bounds for active set.")
        r, c = self._active_pairs[index]
        return int(r), int(c)

    def get_active_pairs(self):
        """
        全アクティブペア配列を返します（未使用部分は [0, 0] のまま）。
        ※1D版の get_active_indices と同じ方針。
        """
        return self._active_pairs

    def get_number_of_active(self):
        """
        現在のアクティブ要素数を返します。
        """
        return self._number_of_active

    def is_active(self, row: int, col: int):
        """
        (row, col) がアクティブかどうかを返します。
        """
        self._check_bounds(row, col)
        return bool(self._active_flags[row, col])

    # ---- optional: ユーティリティ ----
    def get_active_flags(self):
        """
        アクティブフラグ行列（bool, shape=(n_rows, n_cols)）を返します。
        """
        return self._active_flags

    def clear(self):
        """
        全要素を非アクティブにして集合を空にします。
        """
        if self._number_of_active > 0:
            # アクティブだった場所のフラグをまとめてFalseへ
            for i in range(self._number_of_active):
                r, c = self._active_pairs[i]
                self._active_flags[r, c] = False
            self._active_pairs[:] = 0
            self._number_of_active = 0
