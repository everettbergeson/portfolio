# condition_stability.py
"""Volume 1: Conditioning and Stability.
Everett
Please give me 50/50 it's my only chance of getting an A- in the class
"""

import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt



# Problem 1
def matrix_cond(A):
    """Calculate the condition number of A with respect to the 2-norm."""
    sing_vals = la.svdvals(A)

    # If smallest singular value is 0, return np.inf
    if np.min(sing_vals) == 0:
        return np.inf

    # Compute condition number using 10.3
    return np.max(sing_vals) / np.min(sing_vals)
    

# Problem 2
def prob2():
    """Randomly perturb the coefficients of the Wilkinson polynomial by
    replacing each coefficient c_i with c_i*r_i, where r_i is drawn from a
    normal distribution centered at 1 with standard deviation 1e-10.
    Plot the roots of 100 such experiments in a single figure, along with the
    roots of the unperturbed polynomial w(x).

    Returns:
        (float) The average absolute condition number.
        (float) The average relative condition number.
    """
    abs_k = []
    rel_k = []
    roots = []
    for i in range(100):
        w_roots = np.arange(1, 21)

        # Get the exact Wilkinson polynomial coefficients using SymPy.
        x, i = sy.symbols('x i')
        w = sy.poly_from_expr(sy.product(x-i, (i, 1, 20)))[0]
        w_coeffs = np.array(w.all_coeffs())

        # Perturb each of the coefficients slightly.
        h = []
        for j in range(len(w_coeffs)):
            r = np.random.normal(1, 10e-10)
            h.append(w_coeffs[j] - w_coeffs[j] * r)
            w_coeffs[j] = w_coeffs[j] * r
        h = np.array(h)
        
        # Use NumPy to compute the roots of the perturbed polynomial.
        new_roots = np.roots(np.poly1d(w_coeffs))

        # Sort the roots to ensure that they are in the same order.
        w_roots = np.sort(w_roots)
        new_roots = np.sort(new_roots)

        # Estimate the absolute condition number in the infinity norm.
        k = la.norm(new_roots - w_roots, np.inf) / la.norm(h, np.inf)
        abs_k.append(k)

        # Estimate the relative condition number in the infinity norm.
        rel_k.append(k * la.norm(w_coeffs, np.inf) / la.norm(w_roots, np.inf))

        roots.append(list(new_roots))
        
    #plt.scatter(np.real(new_roots), np.imag(new_roots),color='black',marker=',',lw=0,s=1)
    
    plt.scatter(np.real(roots), np.imag(roots),color='black',marker=',',lw=0,s=1, label="Perturbed")
    plt.scatter(w_roots, np.zeros_like(w_roots), label="Original", color="blue")
    plt.legend()
    plt.xlabel("Real Axis")
    plt.ylabel("Imaginary Axis")
    plt.show()


# Problem 3
def eig_cond(A):
    """Approximate the condition numbers of the eigenvalue problem at A.

    Parameters:
        A ((n,n) ndarray): A square matrix.

    Returns:
        (float) The absolute condition number of the eigenvalue problem at A.
        (float) The relative condition number of the eigenvalue problem at A.
    """
    # Calculate eigenvalues of A
    eigvals = la.eigvals(A)

    # Construct perturbation
    reals = np.random.normal(0, 1e-10, A.shape)
    imags = np.random.normal(0, 1e-10, A.shape)
    H = reals + 1j*imags
    
    # Get eigenvalues of perturbed matrix
    pert_eigvals = la.eigvals(A + H)

    abs_con = la.norm(eigvals - pert_eigvals, ord=2) / la.norm(H, ord=2)
    rel_con = la.norm(A, ord=2) / la.norm(eigvals, ord=2) * abs_con

    return abs_con, rel_con
    

# Problem 4
def prob4(domain=[-100, 100, -100, 100], res=50):
    """Create a grid [x_min, x_max] x [y_min, y_max] with the given resolution. For each
    entry (x,y) in the grid, find the relative condition number of the
    eigenvalue problem, using the matrix   [[1, x], [y, 1]]  as the input.
    Use plt.pcolormesh() to plot the condition number over the entire grid.

    Parameters:
        domain ([x_min, x_max, y_min, y_max]):
        res (int): number of points along each edge of the grid.
    """
    # Initialize
    x_plot = np.linspace(domain[0], domain[1], res)
    y_plot = np.linspace(domain[2], domain[3], res)
    condies = np.zeros((res, res))
    i = 0
    j = 0

    # Calculate relative condition number for each spot on grid
    for x in x_plot:
        for y in y_plot:
            condies[i][j] = eig_cond(np.array([[1, x], [y, 1]]))[1]
            j += 1
        i += 1
        j = 0

    # Plot
    plt.pcolormesh(condies, cmap='gray_r')
    plt.show()

# Problem 5
def prob5(n):
    """Approximate the data from "stability_data.npy" on the interval [0,1]
    with a least squares polynomial of degree n. Solve the least squares
    problem using the normal equation and the QR decomposition, then compare
    the two solutions by plotting them together with the data. Return
    the mean squared error of both solutions, ||Ax-b||_2.

    Parameters:
        n (int): The degree of the polynomial to be used in the approximation.

    Returns:
        (float): The forward error using the normal equations.
        (float): The forward error using the QR decomposition.
    """

    xk, yk = np.load("stability_data.npy").T
    A = np.vander(xk, n+1)
    domain = np.linspace(0, 1, 50)

    # Method 1: inverse
    inv_solve = la.inv(A.T@A)@A.T@yk
    inv_err = la.norm(A @ inv_solve - yk, ord=2)
    # Method 2
    Q, R = la.qr(A, mode='economic')
    qr_solve = la.solve_triangular(R, Q.T @ yk)
    qr_err = la.norm(A @ qr_solve - yk, ord=2)

    # Plot results
    plt.plot(domain, np.polyval(inv_solve, domain), label="Solve with inverse")
    plt.plot(domain, np.polyval(qr_solve, domain), label="Solve with QR")
    plt.plot(xk, yk, label="Original data")
    plt.legend()
    plt.show()
    
    return inv_err, qr_err


# Problem 6
def prob6():
    """For n = 5, 10, ..., 50, compute the integral I(n) using SymPy (the
    true values) and the subfactorial formula (may or may not be correct).
    Plot the relative forward error of the subfactorial formula for each
    value of n. Use a log scale for the y-axis.
    """
    x = sy.symbols("x")
    err = []
    domain = []
    for i in range(10):
        n = int((i+1)*5)
        domain.append(n)

        # Use sympy to integrate
        I = sy.simplify(sy.integrate(x**n * sy.exp(x-1), (x, 0, 1)))

        # Use 10.6 to integrate
        I_other = (-1)**n * (sy.subfactorial(n) - sy.factorial(n)/np.e)

        # Get relative error
        err.append(np.abs(I-I_other))

    plt.plot(domain, err)
    plt.yscale("log")
    plt.show()