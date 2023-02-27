# interior_point_quadratic.py
"""Volume 2: Interior Point for Quadratic Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.sparse import spdiags
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import cvxpy as cp



def startingPoint(G, c, A, b, guess):
    """
    Obtain an appropriate initial point for solving the QP
    .5 x^T Gx + x^T c s.t. Ax >= b.
    Parameters:
        G -- symmetric positive semidefinite matrix shape (n,n)
        c -- array of length n
        A -- constraint matrix shape (m,n)
        b -- array of length m
        guess -- a tuple of arrays (x, y, mu) of lengths n, m, and m, resp.
    Returns:
        a tuple of arrays (x0, y0, l0) of lengths n, m, and m, resp.
    """
    m,n = A.shape
    x0, y0, l0 = guess

    # Initialize linear system
    N = np.zeros((n+m+m, n+m+m))
    N[:n,:n] = G
    N[:n, n+m:] = -A.T
    N[n:n+m, :n] = A
    N[n:n+m, n:n+m] = -np.eye(m)
    N[n+m:, n:n+m] = np.diag(l0)
    N[n+m:, n+m:] = np.diag(y0)
    rhs = np.empty(n+m+m)
    rhs[:n] = -(G.dot(x0) - A.T.dot(l0)+c)
    rhs[n:n+m] = -(A.dot(x0) - y0 - b)
    rhs[n+m:] = -(y0*l0)

    sol = la.solve(N, rhs)
    dx = sol[:n]
    dy = sol[n:n+m]
    dl = sol[n+m:]

    y0 = np.maximum(1, np.abs(y0 + dy))
    l0 = np.maximum(1, np.abs(l0+dl))

    return x0, y0, l0


# Problems 1-2
def qInteriorPoint(Q, c, A, b, guess, niter=20, tol=1e-16, verbose=False):
    """Solve the Quadratic program min .5 x^T Q x +  c^T x, Ax >= b
    using an Interior Point method.

    Parameters:
        Q ((n,n) ndarray): Positive semidefinite objective matrix.
        c ((n, ) ndarray): linear objective vector.
        A ((m,n) ndarray): Inequality constraint matrix.
        b ((m, ) ndarray): Inequality constraint vector.
        guess (3-tuple of arrays of lengths n, m, and m): Initial guesses for
            the solution x and lagrange multipliers y and eta, respectively.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    m, n = A.shape

    # Put together F
    def make_f(x, y, mu):
        Y = np.diag(y)
        M = np.diag(mu)
        first = (Q @ x) - (A.T @ mu) + c
        second = (A @ x) - y - b
        third = Y @ M @ np.ones(m)
        temp = np.append(first, second)
        return np.append(temp, third)

    # Put together DF
    def make_df(x, y, mu):
        Y = np.diag(y)
        M = np.diag(mu)
        top = np.hstack((Q, np.zeros((n,m)), -1*A.T))
        middle = np.hstack((A, -1*np.eye(m,m), np.zeros((m,m))))
        bottom = np.hstack((np.zeros((m,n)), M, Y))
        return np.vstack((top, middle, bottom))

    # Interior point method for QP
    # Choose initial point (x0, y0, mu0)
    x, y, mu = startingPoint(Q, c, A, b, guess)
    k = 0
    finished = False
    while finished == False:
        # Calculate nu, F, DF
        nu = y@mu/m
        print(nu)
        F = make_f(x, y, mu)
        DF = make_df(x, y, mu)

        # Find search direction
        to_solve = -1*F + np.append(np.zeros(n+m), nu/10 * np.ones(m))
        search_dir = la.solve(DF, to_solve)

        # Calculate step length
        change_x = search_dir[:n]
        change_y = search_dir[n:n+m]
        change_mu = search_dir[n+m:]

        mask_b = change_mu < 0
        mask_d = change_y < 0
        beta_max = np.divide(-1*mu, change_mu)[mask_b]
        delta_max = np.divide(-1*y, change_y)[mask_d]
        if len(beta_max) == 0:
            beta_max = 1
        else:
            beta_max = np.min(beta_max)
        if len(delta_max) == 0:
            delta_max = 1
        else:
            delta_max = np.min(delta_max)

        beta = min(1, .95*beta_max)
        delta = min(1, .95*delta_max)
        alpha = min(beta, delta)

        # Take step
        x = x + alpha * change_x
        y = y + alpha * change_y
        mu = mu + alpha * change_mu

        if (k > niter) or (nu < tol):
            finished = True
            break
        k += 1

    return x, x @ Q @ x / 2 + c @ x



def laplacian(n):
    """Construct the discrete Dirichlet energy matrix H for an n x n grid."""
    data = -1*np.ones((5, n**2))
    data[2,:] = 4
    data[1, n-1::n] = 0
    data[3, ::n] = 0
    diags = np.array([-n, -1, 0, 1, n])
    return spdiags(data, diags, n**2, n**2).toarray()


# Problem 3
def circus(n=15):
    """Solve the circus tent problem for grid size length 'n'.
    Display the resulting figure.
    """
    # Initalize tent pole configuration
    L = np.zeros((n,n))
    L[n//2-1:n//2+1,n//2-1:n//2+1] = .5
    m = [n//6-1, n//6, int(5*(n/6.))-1, int(5*(n/6.))]
    mask1, mask2 = np.meshgrid(m, m)
    L[mask1, mask2] = .3
    L = L.ravel()

    # Set initial guesses.
    x = np.ones((n,n)).ravel()
    y = np.ones(n**2)
    mu = np.ones(n**2)

    # Initialize vector c, constraint matrix A and matrix H with the laplacian()
    c = np.ones(n**2) * -1 * (n-1)**(-2)
    A = np.eye(n**2)
    H = laplacian(n)

    # Calculate the solution
    z = qInteriorPoint(H, c, A, L, (x,y,mu))[0].reshape((n,n))

    # Plot the solution
    domain = np.arange(n)
    X, Y = np.meshgrid(domain, domain)
    fig = plt.figure()
    ax1 = fig.add_subplot(111, projection='3d')
    ax1.plot_surface(X, Y, z, rstride=1, cstride=1, color='r')
    plt.show()



# Problem 4
def portfolio(filename="portfolio.txt"):
    """Markowitz Portfolio Optimization

    Parameters:
        filename (str): The name of the portfolio data file.

    Returns:
        (ndarray) The optimal portfolio with short selling.
        (ndarray) The optimal portfolio without short selling.
    """
    data = np.loadtxt(filename)
    values = data[:,1:]
    Q = np.cov(values.T)
    u = np.mean(values, axis=0)
    P = np.eye(values.shape[1])
    R = 1.13

    x = cp.Variable(values.shape[1])
    problem = cp.Problem(cp.Minimize(.5 * cp.quad_form(x,Q)), [cp.sum(x)==1, u@x==R])
    val = problem.solve()
 

    return x.value
if __name__=="__main__":
    """
    Q = np.array([[1, -1],[-1, 2]])
    c = np.array([-2, -6])
    A = np.array([[-1, -1], [1, -2], [-2, -1], [1, 0], [0, 1]])
    m, n = A.shape
    b = np.array([-2, -2, -3, 0, 0])
    guess = (np.array([.5, .5]), np.ones(m), np.ones(m))
    print(qInteriorPoint(Q, c, A, b, guess))
    """
    print(portfolio())
