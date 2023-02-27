# lstsq_eigs.py
"""Volume 1: Least Squares and Computing Eigenvalues.
<Name>
<Class>
<Date>
"""

# (Optional) Import functions from your QR Decomposition lab.
# import sys
# sys.path.insert(1, "../QR_Decomposition")
# from qr_decomposition import qr_gram_schmidt, qr_householder, hessenberg

import numpy as np
from matplotlib import pyplot as plt
from scipy import linalg as la
import cmath


# Problem 1
def least_squares(A, b):
    """Calculate the least squares solutions to Ax = b by using the QR
    decomposition.

    Parameters:
        A ((m,n) ndarray): A matrix of rank n <= m.
        b ((m, ) ndarray): A vector of length m.

    Returns:
        x ((n, ) ndarray): The solution to the normal equations.
    """
    Q, R = la.qr(A, mode="economic")
    return la.solve_triangular(R, np.dot(Q.T, b))

# Problem 2
def line_fit():
    """Find the least squares line that relates the year to the housing price
    index for the data in housing.npy. Plot both the data points and the least
    squares line.
    """

    myarray = np.load('housing.npy')
    x = myarray[:, 0]
    y = myarray[:, 1]

    A = np.column_stack((x, np.ones(len(x))))

    ls = least_squares(A, y)

    slope = ls[0]
    yintercept = ls[1]

    plt.plot(x, (x * slope) + yintercept)
    plt.plot(x, y, 'k*')
    plt.show()

# Problem 3
def polynomial_fit():
    """Find the least squares polynomials of degree 3, 6, 9, and 12 that relate
    the year to the housing price index for the data in housing.npy. Plot both
    the data points and the least squares polynomials in individual subplots.
    """
    myarray = np.load('housing.npy')
    x = myarray[:, 0]
    y = myarray[:, 1]

    yresults = []
    xdomain = np.linspace(0, 16, 100)

    for degree in [3, 6, 9, 12]:
        A = np.vander(x, degree + 1)
        f = np.poly1d(la.lstsq(A, y)[0])

        polyresults = []
        for i in xdomain:
            j = f([i])
            polyresults.append(j)
        yresults.append(polyresults)

    ax1 = plt.subplot(221)
    ax1.plot(xdomain, yresults[0], 'g')
    ax1.plot(x, y, 'k*')

    ax2 = plt.subplot(222)
    ax2.plot(xdomain, yresults[1], 'r')
    ax2.plot(x, y, 'k*')

    ax3 = plt.subplot(223)
    ax3.plot(xdomain, yresults[2], 'c')
    ax3.plot(x, y, 'k*')

    ax4 = plt.subplot(224)
    ax4.plot(xdomain, yresults[3])
    ax4.plot(x, y, 'k*')

    plt.show()

def plot_ellipse(a, b, c, d, e):
    """Plot an ellipse of the form ax^2 + bx + cxy + dy + ey^2 = 1."""
    theta = np.linspace(0, 2*np.pi, 200)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    A = a*(cos_t**2) + c*cos_t*sin_t + e*(sin_t**2)
    B = b*cos_t + d*sin_t
    r = (-B + np.sqrt(B**2 + 4*A)) / (2*A)

    plt.plot(r*cos_t, r*sin_t)
    plt.gca().set_aspect("equal", "datalim")

# Problem 4
def ellipse_fit():
    """Calculate the parameters for the ellipse that best fits the data in
    ellipse.npy. Plot the original data points and the ellipse together, using
    plot_ellipse() to plot the ellipse.
    """
    xk, yk = np.load("ellipse.npy").T
    A = np.column_stack((xk**2, xk, xk*yk, yk, yk**2))
    b = np.ones(len(A))
    c = la.lstsq(A, b)[0]
    plot_ellipse(c[0], c[1], c[2], c[3], c[4])
    plt.plot(xk, yk, '*')
    plt.show()


# Problem 5
def power_method(A, N=20, tol=1e-12):
    """Compute the dominant eigenvalue of A and a corresponding eigenvector
    via the power method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The maximum number of iterations.
        tol (float): The stopping tolerance.

    Returns:
        (float): The dominant eigenvalue of A.
        ((n,) ndarray): An eigenvector corresponding to the dominant
            eigenvalue of A.
    """
    m, n = np.shape(A)
    x0 = np.random.rand(n)
    x0 = x0/la.norm(x0)
    xk = x0
    xk1 = 0
    for k in range(0, N):
        xk1 = A @ xk
        xk1 = xk1 / la.norm(xk1)
        

        if la.norm(xk1 - xk) < tol:
            print("early break")
            return xk1.T @ (A @ xk1), xk1

        xk = xk1

    return xk1.T @ (A @ xk1), xk1


# Problem 6
def qr_algorithm(A, N=50, tol=1e-12):
    """Compute the eigenvalues of A via the QR algorithm.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        N (int): The number of iterations to run the QR algorithm.
        tol (float): The threshold value for determining if a diagonal S_i
            block is 1x1 or 2x2.

    Returns:
        ((n,) ndarray): The eigenvalues of A.
    """
    
    m, n = np.shape(A)
    S = la.hessenberg(A)
    for k in range (0, N):
        Q, R = la.qr(S)
        S = R @ Q
    
    eigs = []
    i = 0
    while i < n:
        if i == (n-1) or S[i + 1, i] <= tol:
            eigs.append(S[i, i])
            
        else:
            a = S[i, i]
            b = S[i, i + 1]
            c = S[i + 1, i]
            d = S[i + 1, i + 1]
            bq = (a + d)
            cq = ((a * d) - (b * c))

            pos_quad = (bq + cmath.sqrt(bq**2 - (4 * cq))) / 2
            neg_quad = (bq - cmath.sqrt(bq**2 - (4 * cq))) / 2
            eigs.append(pos_quad)
            eigs.append(neg_quad)

            i += 1
        i += 1

    return eigs


if __name__=="__main__":
    print(ellipse_fit())