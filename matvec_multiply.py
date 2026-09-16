import random

# create a function to compute the dot product of two vectors using a for loop
# add comments for the selected function


def dot_product(vector1, vector2):
    """Compute the dot product of two vectors using a for loop.

    The dot product is the sum of the element-wise products of the two
    vectors, i.e. sum(vector1[i] * vector2[i]) over all indices i.

    Args:
        vector1 (list): the first vector
        vector2 (list): the second vector

    Returns:
        float: the dot product of vector1 and vector2

    Raises:
        TypeError: if either argument is not a list, or if any entry is
            not an int or float
        ValueError: if the two vectors do not have the same length
    """
    # guard: both inputs must be lists
    if not isinstance(vector1, list) or not isinstance(vector2, list):
        raise TypeError("Both inputs must be lists of numbers.")

    # guard: a dot product is only defined for vectors of equal length
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length to compute a dot product.")

    result = 0.0
    # accumulate the element-wise products one index at a time
    for i in range(len(vector1)):
        # guard: every entry must be a number before it can be multiplied
        if not isinstance(vector1[i], (int, float)) or not isinstance(
            vector2[i], (int, float)
        ):
            raise TypeError(
                f"Vector entries must be ints or floats; got "
                f"{type(vector1[i]).__name__} and {type(vector2[i]).__name__} "
                f"at index {i}."
            )
        result += vector1[i] * vector2[i]
    return result


# create a function to compute the matrix-vector product using the dot_product function
# add comments for the selected function


def matvec_multiply(matrix, vector):
    """Compute the matrix-vector product using the dot_product function.

    Treats every row of the matrix as a vector and takes its dot product
    with the input vector; the i-th entry of the result is therefore the
    dot product of row i of the matrix with the vector.

    Args:
        matrix (list of list): an n x m matrix stored as a list of n rows
        vector (list): a vector of length m

    Returns:
        list: a vector of length n, the matrix-vector product

    Raises:
        TypeError: if the matrix or vector is not a list, if any row is
            not a list, or if any entry is not an int or float (raised
            by dot_product)
        ValueError: if the matrix is empty, or if any row has a length
            different from the vector
    """
    # guard: the matrix must be a list of rows
    if not isinstance(matrix, list):
        raise TypeError("Matrix must be a list of rows.")

    # guard: an empty matrix has no rows, so the product is undefined
    if len(matrix) == 0:
        raise ValueError("Matrix cannot be empty.")

    # guard: the vector must be a list
    if not isinstance(vector, list):
        raise TypeError("Vector must be a list.")

    # guard: every row must be a list, and its length must match the
    # vector length (a jagged matrix has no well-defined product)
    for i, row in enumerate(matrix):
        if not isinstance(row, list):
            raise TypeError(
                f"Row {i} is not a list; the matrix must be a list of rows."
            )
        if len(row) != len(vector):
            raise ValueError(
                f"Row {i} has length {len(row)}, but the vector has length "
                f"{len(vector)}; every row must match the vector length."
            )

    # dot the input vector with every row of the matrix
    result = []
    for row in matrix:
        result.append(dot_product(row, vector))
    return result


# create a main function to test the matrix-vector product function using randomly generated data of size 1000x1000
# add comments for the selected function


def main():
    """Test matvec_multiply with randomly generated 1000x1000 data.

    Builds a random 1000 x 1000 matrix and a random vector of length
    1000, computes the matrix-vector product, and checks a randomly
    chosen entry of the result against an independent computation.
    """
    size = 1000

    # random matrix with 1000 rows and 1000 columns
    matrix = [[random.random() for _ in range(size)] for _ in range(size)]

    # random vector of length 1000
    vector = [random.random() for _ in range(size)]

    # compute the matrix-vector product with our function
    result = matvec_multiply(matrix, vector)

    # sanity checks on the shape of the result
    assert len(result) == size, "Result should have one entry per matrix row."
    print(f"Result length: {len(result)} (expected {size})")

    # verify one randomly chosen entry against an independent computation
    row_index = random.randrange(size)
    expected = sum(matrix[row_index][j] * vector[j] for j in range(size))
    print(
        f"Row {row_index} check: computed {result[row_index]:.6f}, "
        f"expected {expected:.6f}"
    )
    assert abs(result[row_index] - expected) < 1e-9, "Computed entry does not match."
    print("Matrix-vector product computed correctly.")


if __name__ == "__main__":
    main()
