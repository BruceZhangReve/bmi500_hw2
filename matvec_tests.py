# generate tests for matrix-vector-product function in matvec_multiply.py

# generate tests for dot-product function in matvec_multiply.py

import unittest

from matvec_multiply import dot_product, matvec_multiply


class TestDotProduct(unittest.TestCase):
    """Tests for the dot-product function in matvec_multiply.py."""

    def test_known_vectors(self):
        # 1*4 + 2*5 + 3*6 = 32
        self.assertEqual(dot_product([1, 2, 3], [4, 5, 6]), 32)

    def test_empty_vectors(self):
        # the dot product of two empty vectors is 0
        self.assertEqual(dot_product([], []), 0)

    def test_negative_values(self):
        # (-1)*4 + 2*(-5) + (-3)*6 = -32
        self.assertEqual(dot_product([-1, 2, -3], [4, -5, 6]), -32)

    def test_zero_vector(self):
        # anything dotted with the zero vector is 0
        self.assertEqual(dot_product([1, 2, 3], [0, 0, 0]), 0)

    def test_length_mismatch_raises(self):
        # vectors of different lengths are invalid
        with self.assertRaises(ValueError):
            dot_product([1, 2], [1, 2, 3])

    def test_non_list_input_raises(self):
        # inputs that are not lists are invalid
        with self.assertRaises(TypeError):
            dot_product([1, 2], None)
        with self.assertRaises(TypeError):
            dot_product("ab", "cd")

    def test_non_numeric_entries_raise(self):
        # entries that are not numbers cannot be multiplied
        with self.assertRaises(TypeError):
            dot_product([1, "a"], [2, 3])
        with self.assertRaises(TypeError):
            dot_product([1, 2], [None, 3])


class TestMatVecMultiply(unittest.TestCase):
    """Tests for the matrix-vector-product function in matvec_multiply.py."""

    def test_identity_matrix(self):
        # the identity matrix times a vector returns the vector unchanged
        identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        vector = [5, -2, 7]
        self.assertEqual(matvec_multiply(identity, vector), vector)

    def test_known_product(self):
        # row 0: 1*5 + 2*6 = 17, row 1: 3*5 + 4*6 = 39
        matrix = [[1, 2], [3, 4]]
        vector = [5, 6]
        self.assertEqual(matvec_multiply(matrix, vector), [17, 39])

    def test_single_row(self):
        # a 1x3 matrix times a length-3 vector gives a length-1 result
        self.assertEqual(matvec_multiply([[1, 2, 3]], [4, 5, 6]), [32])

    def test_empty_matrix_raises(self):
        # an empty matrix has no rows to dot with the vector
        with self.assertRaises(ValueError):
            matvec_multiply([], [1, 2])

    def test_row_length_mismatch_raises(self):
        # a row longer than the vector makes the dot product undefined
        with self.assertRaises(ValueError):
            matvec_multiply([[1, 2, 3]], [1, 2])

    def test_matrix_not_a_list_raises(self):
        # a matrix that is not a list is invalid
        with self.assertRaises(TypeError):
            matvec_multiply(None, [1, 2])

    def test_flat_list_matrix_raises(self):
        # a flat list has scalar "rows", so it is not a valid matrix
        with self.assertRaises(TypeError):
            matvec_multiply([1, 2, 3], [1, 2, 3])

    def test_jagged_matrix_raises(self):
        # rows of different lengths make the product undefined
        with self.assertRaises(ValueError):
            matvec_multiply([[1, 2, 3], [4, 5]], [1, 2, 3])

    def test_non_numeric_entry_raises(self):
        # a non-numeric matrix entry cannot be multiplied
        with self.assertRaises(TypeError):
            matvec_multiply([[1, "a"], [3, 4]], [5, 6])

    def test_vector_not_a_list_raises(self):
        # a vector that is not a list is invalid
        with self.assertRaises(TypeError):
            matvec_multiply([[1, 2]], "12")


if __name__ == "__main__":
    unittest.main()
