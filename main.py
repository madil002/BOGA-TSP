import numpy as np

def generate_symmetric_distance_matrix(num_cities):
    np.random.seed(42)

    upper_triangle = np.triu(np.random.randint(10, 100, size=(num_cities, num_cities)), 1)  # Create an upper triangle of random integers
    matrix = upper_triangle + upper_triangle.T  # Mirror the upper triangle to the lower triangle to ensure symmetry

    np.fill_diagonal(matrix, 0)     # Set the diagonal to zero
    print(matrix)
    return matrix

