# qr_decomposition.py
"""Volume 1: The QR Decomposition.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la

# Problem 1
def qr_gram_schmidt(A):
    """Compute the reduced QR decomposition of A via Modified Gram-Schmidt.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,n) ndarray): An orthonormal matrix.
        R ((n,n) ndarray): An upper triangular matrix.
    """
    m, n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n, n))


    for i in range (0, n):
        R[i, i] = la.norm(Q[:, i])
        Q[:, i] = Q[:, i] / R[i, i]
        for j in range (i + 1, n):
            R[i, j] = Q[:, j].T @ Q[:, i]
            Q[:, j] = Q[:, j] - (R[i, j] * Q[:, i])
    return Q, R
    


# Problem 2
def abs_det(A):
    """Use the QR decomposition to efficiently compute the absolute value of
    the determinant of A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) the absolute value of the determinant of A.
    """
    """m, n = np.shape(A)
    detA = 1
    q, r = la.qr(A)
    for i in range (0, n):
        detA = detA * r[i, i]
    return detA"""

    return np.prod(np.diag(la.qr(A)[1]))

# Problem 3
def solve(A, b):
    """Use the QR decomposition to efficiently solve the system Ax = b.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.
        b ((n, ) ndarray): A vector of length n.

    Returns:
        x ((n, ) ndarray): The solution to the system Ax = b.
    """
    Q, R = la.qr(A)
    y = np.dot(Q.T, b)

    n = len(R)
    x_vector = np.zeros(n)

    for i in range(n-1, -1, -1):
        x = y[i]
        for j in range(n-1, i, -1):
            x -= x_vector[j]*R[i,j]
            
        x_vector[i] = x/R[i,i]

    return x_vector


# Problem 4
def qr_householder(A):
    """Compute the full QR decomposition of A via Householder reflections.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n.

    Returns:
        Q ((m,m) ndarray): An orthonormal matrix.
        R ((m,n) ndarray): An upper triangular matrix.
    """
    m, n = np.shape(A)
    R = np.copy(A)
    Q = np.identity(m)
    sign = lambda x: 1 if x >= 0 else -1

    for k in range (0, n):
        u = np.copy(R[k:, k])
        u[0] = u[0] + (sign(u[0]) * la.norm(u))
        u = u/la.norm(u)
        R[k:, k] = R[k:, k:] - ((2*u) * np.outer(u.T, R[k:, k:]))
        Q[k:, :] = Q[k:, :] - ((2*u) * np.outer(u.T, Q[k:, :]))
    return Q.T, R

# Problem 5
def hessenberg(A):
    """Compute the Hessenberg form H of A, along with the orthonormal matrix Q
    such that A = QHQ^T.

    Parameters:
        A ((n,n) ndarray): An invertible matrix.

    Returns:
        H ((n,n) ndarray): The upper Hessenberg form of A.
        Q ((n,n) ndarray): An orthonormal matrix.
    """
    m, n = np.shape(A)
    H = np.copy(A)
    Q = np.identity(m)

    sign = lambda x: 1 if x >= 0 else -1

    for k in range (0, n-2):
        u = np.copy(H[k + 1:, k])
        u[0] = u[0] + (sign(u[0]) * la.norm(u))
        u = u/la.norm(u)
        H[k+1:, k:] = H[k+1:, k:] - ((2*u) * np.outer(u.T, H[k+1:, k:]))
        H[:, k+1:] = H[:, k+1:] - (2 * np.outer((H[:, k+1:] * u), u.T))
        Q[k+1:, :] = Q[k+1:, :] - ((2*u) * np.outer(u.T, Q[k+1:, :]))

    return H, Q.T


if __name__=="__main__":
    
    x = np.array([[1, -3, 11], [2, 5, 2], [1234, 0, 5]], dtype = np.float)
    
    print("Gram Schmidt")
    Q, R = qr_gram_schmidt(x)
    print(Q, R)
    print(np.allclose(Q.T @ Q, np.identity(3)))
    print(np.allclose(Q @ R, x))
    print(np.allclose(np.triu(R), R))
    
    print("Gram Schmidt")
    q, r = la.qr(x, mode="economic")
    print(q, r)
    print(abs_det(x))
    print(la.det(x))
    b = np.array([-11111, 6, 6])

    print(solve(x, b))
    print(la.solve(x, b))
    print("-------------------")


    A = np.random.random((5, 3))
    Q,R = la.qr(A) # Get the full QR decomposition.
    print(A.shape, Q.shape, R.shape)
    print(np.allclose(Q @ R, A))


    print("-------------------")
    A = np.random.random((8,8))
    H, Q = la.hessenberg(A, calc_q=True)
    # Verify that H has all zeros below the first subdiagonal and QHQ^T = A.
    print(np.allclose(np.triu(H, -1), H))
    print(np.allclose(Q @ H @ Q.T, A))