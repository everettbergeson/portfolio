# numpy_intro.py
"""Python Essentials: Intro to NumPy.
<Name>
<Class>
<Date>
"""

import numpy as np

def prob1():
    """Define the matrices A and B as arrays. Return the matrix product AB."""

    A = np.array([[3, -1, 4], [1, 5, -9]])
    B = np.array([[2, 6, -5, 3], [5, -8, 9, 7], [9, -3, -2, -3]])
    AB = np.dot(A, B)

    return(AB)


def prob2():
    """Define the matrix A as an array. Return the matrix -A^3 + 9A^2 - 15A."""
    A = np.array([[3, 1, 4], [1, 5, 9], [-5, 3, 1]])
    A_2 = np.dot(A, A)
    A_3 = np.dot(A_2, A)
    return ((-1)*A_3) + (9*A_2) - (15*A)

def prob3():
    """Define the matrices A and B as arrays. Calculate the matrix product ABA,
    change its data type to np.int64, and return it.
    """
    A = np.ones((7, 7), dtype=np.int64)
    A = np.triu(A)
    B = np.diag([-5, -5, -5, -5, -5, -5, -5])
    B = B + np.tril((-1) * np.ones((7, 7), dtype=np.int64)) 
    B = B + np.triu(5 * np.ones((7,7), dtype=np.int64))
    
    AB = np.dot(A, B)
    ABA = np.dot(AB, A)
    return np.int64(ABA)

def prob4(A):
    """Make a copy of 'A' and set all negative entries of the copy to 0.
    Return the copy.

    Example:
        >>> A = np.array([-3,-1,3])
        >>> prob4(A)
        array([0, 0, 3])
    """
    
    no_negatives = np.array(A)
    mask = no_negatives < 0
    no_negatives[mask] = 0
    return no_negatives


def prob5():
    """Define the matrices A, B, and C as arrays. Return the block matrix
                                | 0 A^T I |
                                | A  0  0 |,
                                | B  0  C |
    where I is the 3x3 identity matrix and each 0 is a matrix of all zeros
    of the appropriate size.
    """
    A = np.arange(6).reshape((3,2))
    A = np.transpose(A)

    B = np.tril(3 * np.ones((3,3), dtype=np.int64))

    C = np.diag([-2, -2, -2])

    I = np.diag([1, 1, 1])

    block1 = np.hstack((np.zeros((3,3), dtype=np.int64), np.transpose(A), I))
    block2 = np.hstack((A, np.zeros((2,5), dtype=np.int64)))
    block3 = np.hstack((B, np.zeros((3, 2), dtype = np.int64), C))

    full_block = np.vstack((block1, block2, block3))

    return(full_block)

def prob6(A):
    """Divide each row of 'A' by the row sum and return the resulting array.

    Example:
        >>> A = np.array([[1,1,0],[0,1,0],[1,1,1]])
        >>> prob6(A)
        array([[ 0.5       ,  0.5       ,  0.        ],
               [ 0.        ,  1.        ,  0.        ],
               [ 0.33333333,  0.33333333,  0.33333333]])
    """
    row_sto = np.array(A)
    sum_array = row_sto.sum(axis=1)
    sum_array = sum_array.reshape(-1,1)
    result = row_sto / sum_array

    print(result)

def prob7():
    """Given the array stored in grid.npy, return the greatest product of four
    adjacent numbers in the same direction (up, down, left, right, or
    diagonally) in the grid.
    """
    grid = np.load("grid.npy")

    hmax = np.max(grid[:,:-3] * grid[:,1:-2] * grid[:,2:-1] * grid[:,3:])
    vmax = np.max(grid[:-3,:] * grid[1:-2,:] * grid[2:-1,:] * grid[3:,:])
    diagtltobrmax = np.max(grid[:-3,3:] * grid[1:-2,2:-1] * grid[2:-1,1:-2] * grid[3:,:-3])
    diagtrtoblmax = np.max(grid[:-3,:-3] * grid[1:-2,1:-2] * grid[2:-1,2:-1] * grid[3:,3:])
    max = np.max([hmax, vmax, diagtrtoblmax, diagtltobrmax])
    return max


if __name__=="__main__":
    prob1()
    print(prob2())
    print(prob3())
    prob6([[1, 2, 3], [0, 1, 2], [4, 5, 6], [1, 5, 9]])
    print(prob7())