# linear_transformations.py
"""Volume 1: Linear Transformations.
<Name>
<Class>
<Date>
"""

from random import random
import numpy as np
from matplotlib import pyplot as plt
import math


# Problem 1
def stretch(A, a, b):
    """Scale the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    stretch_matrix = np.array([[a, 0], [0, b]])
    return np.dot(stretch_matrix, A)

def shear(A, a, b):
    """Slant the points in A by a in the x direction and b in the
    y direction.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): scaling factor in the x direction.
        b (float): scaling factor in the y direction.
    """
    shear_matrix = np.array([[1, a], [b, 1]])
    return np.dot(shear_matrix, A)

def reflect(A, a, b):
    """Reflect the points in A about the line that passes through the origin
    and the point (a,b).

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        a (float): x-coordinate of a point on the reflecting line.
        b (float): y-coordinate of the same point on the reflecting line.
    """
    reflect_matrix = np.array([[a**2 - b**2, 2*a*b], [2*a*b, b**2 - a**2]])
    return np.dot((1/((a**2) + (b**2))) * reflect_matrix, A)

def rotate(A, theta):
    """Rotate the points in A about the origin by theta radians.

    Parameters:
        A ((2,n) ndarray): Array containing points in R2 stored as columns.
        theta (float): The rotation angle in radians.
    """
    rotate_matrix = np.array([[math.cos(theta), -1 * math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    return np.dot(rotate_matrix, A)


# Problem 2
def solar_system(T, x_e, x_m, omega_e, omega_m):
    """Plot the trajectories of the earth and moon over the time interval [0,T]
    assuming the initial position of the earth is (x_e,0) and the initial
    position of the moon is (x_m,0).

    Parameters:
        T (int): The final time.
        x_e (float): The earth's initial x coordinate.
        x_m (float): The moon's initial x coordinate.
        omega_e (float): The earth's angular velocity.
        omega_m (float): The moon's angular velocity.
    """
    
    times = np.linspace(0, T, 200)

    earth_positions = []
    moon_positions = []

    earth_times = []
    moon_times = []

    for time in times:
        earth_position = x_e*math.cos(time * omega_e)
        moon_position = (x_m-x_e)*math.cos(time * omega_m) + earth_position

        earth_positions.append(earth_position)
        moon_positions.append(moon_position)

        earth_times.append(x_e * math.sin(time))
        moon_times.append((x_m-x_e) * math.sin(time * omega_m) + x_e * math.sin(time * omega_e))

    plt.axis("equal")
    plt.plot(earth_positions, earth_times)
    plt.plot(moon_positions, moon_times)
    
    
    plt.show()


def random_vector(n):
    """Generate a random vector of length n as a list."""
    return [random() for i in range(n)]

def random_matrix(n):
    """Generate a random nxn matrix as a list of lists."""
    return [[random() for j in range(n)] for i in range(n)]

def matrix_vector_product(A, x):
    """Compute the matrix-vector product Ax as a list."""
    m, n = len(A), len(x)
    return [sum([A[i][k] * x[k] for k in range(n)]) for i in range(m)]

def matrix_matrix_product(A, B):
    """Compute the matrix-matrix product AB as a list of lists."""
    m, n, p = len(A), len(B), len(B[0])
    return [[sum([A[i][k] * B[k][j] for k in range(n)])
                                    for j in range(p) ]
                                    for i in range(m) ]
import time
# Problem 3
def prob3():
    """Use time.time(), timeit.timeit(), or %timeit to time
    matrix_vector_product() and matrix-matrix-mult() with increasingly large
    inputs. Generate the inputs A, x, and B with random_matrix() and
    random_vector() (so each input will be nxn or nx1).
    Only time the multiplication functions, not the generating functions.

    Report your findings in a single figure with two subplots: one with matrix-
    vector times, and one with matrix-matrix times. Choose a domain for n so
    that your figure accurately describes the growth, but avoid values of n
    that lead to execution times of more than 1 minute.
    """
    matrix_matrix_time = []
    matrix_vector_time = []
    n_list = [1, 10, 40, 120, 200, 400]
    for n in n_list:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)
        start_matrix = time.time()
        k = matrix_matrix_product(A, B)
        finish_matrix = time.time()
        matrix_difference = finish_matrix - start_matrix
        matrix_matrix_time.append(matrix_difference)
        start_vector = time.time()
        k = matrix_vector_product(A, x)
        finish_vector = time.time()
        vector_difference = finish_vector - start_vector
        matrix_vector_time.append(vector_difference)
    
    ax1 = plt.subplot(121)
    ax1.plot(n_list, matrix_vector_time, 'g.-')
    ax1.set_title("Matrix-Vector Multiplication")

    ax2 = plt.subplot(122)
    ax2.plot(n_list, matrix_matrix_time, 'b.-')
    ax2.set_title("Matrix-Matrix Multiplication")

    plt.show()




# Problem 4
def prob4():
    """Time matrix_vector_product(), matrix_matrix_product(), and np.dot().

    Report your findings in a single figure with two subplots: one with all
    four sets of execution times on a regular linear scale, and one with all
    four sets of exections times on a log-log scale.
    """
    matrix_matrix_time = []
    matrix_vector_time = []
    dot_matrix_matrix_time = []
    dot_matrix_vector_time = []
    n_list = [1, 10, 40, 120, 200, 400]
    for n in n_list:
        A = random_matrix(n)
        B = random_matrix(n)
        x = random_vector(n)

        start_matrix = time.time()
        k = matrix_matrix_product(A, B)
        finish_matrix = time.time()
        matrix_matrix_time.append(finish_matrix - start_matrix)

        start_vector = time.time()
        k = matrix_vector_product(A, x)
        finish_vector = time.time()
        matrix_vector_time.append(finish_vector - start_vector)

        #--------

        dot_start_matrix = time.time()
        k = np.dot(A, B)
        dot_finish_matrix = time.time()
        dot_matrix_matrix_time.append(dot_finish_matrix - dot_start_matrix)

        dot_start_vector = time.time()
        k = np.dot(A, x)
        dot_finish_vector = time.time()
        dot_matrix_vector_time.append(dot_finish_vector - dot_start_vector)
    
    ax1 = plt.subplot(121)
    ax1.plot(n_list, matrix_vector_time, 'k.-', label="Matrix-Vector")
    ax1.plot(n_list, matrix_matrix_time, 'g.-', label="Matrix-Matrix")
    ax1.plot(n_list, dot_matrix_vector_time, 'r.-', label="NP Matrix-Vector")
    ax1.plot(n_list, dot_matrix_matrix_time, 'b.-', label="NP Matrix-Matrix")
    ax1.legend(loc="upper left")

    ax2 = plt.subplot(122)
    ax2.loglog(n_list, matrix_vector_time, 'k.-', basex=2, basey=2, lw=2, label="Matrix-Vector")
    ax2.loglog(n_list, matrix_matrix_time, 'g.-', basex=2, basey=2, lw=2, label="Matrix-Matrix")
    ax2.loglog(n_list, dot_matrix_vector_time, 'r.-', basex=2, basey=2, lw=2, label="NP Matrix-Vector")
    ax2.loglog(n_list, dot_matrix_matrix_time, 'b.-', basex=2, basey=2, lw=2, label="NP Matrix-Matrix")
    

    plt.show()



if __name__=="__main__":
    prob4()