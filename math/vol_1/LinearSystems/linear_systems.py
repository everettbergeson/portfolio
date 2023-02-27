# linear_systems.py
"""Volume 1: Linear Systems.
<Name>
<Class>
<Date>
"""
import numpy as np

# Problem 1
def ref(A):
    """Reduce the square matrix A to REF. You may assume that A is invertible
    and that a 0 will never appear on the main diagonal. Avoid operating on
    entries that you know will be 0 before and after a row operation.

    Parameters:
        A ((n,n) ndarray): The square invertible matrix to be reduced.

    Returns:
        ((n,n) ndarray): The REF of A.
    """
    for i in range(0, len(A[0]) - 1):
        for j in range(i + 1, len(A[0])):
            A[j,i:] -= (A[j,i] / A[i,i]) * A[i,i:] 
    print(A)
    """
    i = 0
        j = 1
            A[1,0:] -= (A[1,0] / A[0,0]) * A[0,0:]
        j = 2    
            A[2,0:] -= (A[2,0] / A[0,0]) * A[0,0:] 
    i = 1
        j = 2    
            A[2,1:] -= (A[2,1] / A[1,1]) * A[1,1:] 
            A[j,i:] -= (A[j,i] / A[i,i]) * A[i,i:] 

    print(A)
    
    start with row 0:
    do the 1st and 2nd rows
    from row 1 from 0 on, subtract A10/A00 * row 0 from 0 on
    from row 2 from 0 on, subtract A20/A00 * row 0 from 0 on

    then do row 1:
    do the 2nd row
    from row 2 from 1 on, subtract A21/A11 * (row 1 from 1 on)
    """

    

# Problem 2
def lu(A):
    """Compute the LU decomposition of the square matrix A. You may
    assume that the decomposition exists and requires no row swaps.

    Parameters:
        A ((n,n) ndarray): The matrix to decompose.

    Returns:
        L ((n,n) ndarray): The lower-triangular part of the decomposition.
        U ((n,n) ndarray): The upper-triangular part of the decomposition.
    """
    m, n = np.shape(A)
    U = np.copy(A)
    L = np.identity(m)
    for j in range (0, n):
        for i in range(j + 1, m):
            L[i, j] = U[i, j] / U[j, j]
            U[i, j:] = U[i, j:] - (L[i, j] * U[j, j:])

    return L, U
    print(U)

# Problem 3
def solve(A, b):
    """Use the LU decomposition and back substitution to solve the linear
    system Ax = b. You may again assume that no row swaps are required.

    Parameters:
        A ((n,n) ndarray)
        b ((n,) ndarray)

    Returns:
        x ((m,) ndarray): The solution to the linear system.
    """
    L_A, U_A = lu(A)

    """
    x_vector = []
    for i in range (0, len(L_A)):
        x = b[i]
        for j in range(0, i):
            x += b[j] * L_A[i, j]
        x_vector.append(x)
    return x_vector
    y_vector = []
    for i in range (0, len(U_A)):
        y = 0
        for j in range(i, len(U_A)):
            y += x_vector[j] * U_A[i, j]
        y_vector.append(y)
    return y_vector
    """

    y_vector = []
    for i in range (0, len(L_A)):
        y = b[i]
        for j in range(0, i):
            y -= y_vector[j] * L_A[i, j]
        y_vector.append(y)

    x_vector = []

    for i in range (len(U_A) - 1, -1, -1):
        x = y_vector[i]
        for j in range (len(U_A) - 1, i, -1):
            x -= U_A[i, j]*y_vector[j]
        x = (1/U_A[i, i]) * x
        x_vector.insert(0, x)
    return x_vector

    
from scipy import linalg as la
import time
from matplotlib import pyplot as plt

# Problem 4
def prob4():
    """Time different scipy.linalg functions for solving square linear systems.

    For various values of n, generate a random nxn matrix A and a random
    n-vector b using np.random.random(). Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Invert A with la.inv() and left-multiply the inverse to b.
        2. Use la.solve().
        3. Use la.lu_factor() and la.lu_solve() to solve the system with the
            LU decomposition.
        4. Use la.lu_factor() and la.lu_solve(), but only time la.lu_solve()
            (not the time it takes to do the factorization).

    Plot the system size n versus the execution times. Use log scales if
    needed.
    """
    inverse_time = []
    solve_time = []
    factor_time = []
    factor_solve_time = []
    n_list = [5, 10, 50, 100, 300, 1000, 2000]
    for n in n_list:
        A = np.random.random((n,n))
        b = np.random.random(n)

        start = time.time()

        inv = la.inv(A)
        x_inv = inv @ b

        time1 = time.time()

        x_solve = la.solve(A, b)

        time2 = time.time()

        L_fac, P_fac = la.lu_factor(A)
        x_factor = la.lu_solve((L_fac, P_fac), b)

        time3 = time.time()

        L_sol, P_sol = la.lu_factor(A)

        time4 = time.time()

        x_fac_solve = la.lu_solve((L_sol, P_sol), b)

        finish = time.time()

        inverse_time.append(time1 - start)
        solve_time.append(time2 - time1)
        factor_time.append(time3 - time2)
        factor_solve_time.append(finish - time4)
    
    plt.plot(n_list, inverse_time, 'k.-', label="Inverse")
    plt.plot(n_list, solve_time, 'g.-', label="Solve")
    plt.plot(n_list, factor_time, 'r.-', label="Factor")
    plt.plot(n_list, factor_solve_time, 'b.-', label="Factor then Solve")
    plt.legend(loc="upper left")

    plt.show() 


from scipy import sparse
# Problem 5
def prob5(n):
    """Let I be the n × n identity matrix, and define
                    [B I        ]        [-4  1            ]
                    [I B I      ]        [ 1 -4  1         ]
                A = [  I . .    ]    B = [    1  .  .      ],
                    [      . . I]        [          .  .  1]
                    [        I B]        [             1 -4]
    where A is (n**2,n**2) and each block B is (n,n).
    Construct and returns A as a sparse matrix.

    Parameters:
        n (int): Dimensions of the sparse matrix B.

    Returns:
        A ((n**2,n**2) SciPy sparse matrix)
    """
    offsets = [-1, 0, 1]
    I = sparse.identity(n)
    #I = sparse.diags([1], [0], shape=(n, n)).toarray()
    B = sparse.diags([1, -4, 1], offsets, shape=(n, n))
    #A = sparse.diags([I, B, I], offsets, shape=(n**2, n**2))

    
    #combo = sparse.bmat([[I, B, I]], format ='bsr').toarray()
    A = sparse.block_diag([B]*n)
    #A.setdiag([I], n)


    
    I = sparse.block_diag([1]*n)
    B = sparse.block_diag([-4]*n)
    B.setdiag([1]*n, 1)
    B.setdiag([1]*n, -1)

    A = sparse.block_diag([B]*n)
    
    plt.spy(A, markersize=1)
    plt.show()

    return A.toarray()



# Problem 6
def prob6():
    """Time regular and sparse linear system solvers.

    For various values of n, generate the (n**2,n**2) matrix A described of
    prob5() and vector b of length n**2. Time how long it takes to solve the
    system Ax = b with each of the following approaches:

        1. Convert A to CSR format and use scipy.sparse.linalg.spsolve()
        2. Convert A to a NumPy array and use scipy.linalg.solve().

    In each experiment, only time how long it takes to solve the system (not
    how long it takes to convert A to the appropriate format). Plot the system
    size n**2 versus the execution times. As always, use log scales where
    appropriate and use a legend to label each line.
    """
    


if __name__=="__main__":
    #print(lu(np.array([[1, 0, 0], [1, 2, 0], [1, 1, 3]], dtype = np.float)))
    #print(solve(np.array([[1, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype = np.float), np.array([6, 2, 15, 4])))
    prob5(30)