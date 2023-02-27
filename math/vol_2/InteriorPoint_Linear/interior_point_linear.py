# interior_point_linear.py
"""Volume 2: Interior Point for Linear Programs.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la
from scipy.stats import linregress
from matplotlib import pyplot as plt


# Auxiliary Functions ---------------------------------------------------------
def starting_point(A, b, c):
    """Calculate an initial guess to the solution of the linear program
    min c^T x, Ax = b, x>=0.
    Reference: Nocedal and Wright, p. 410.
    """
    # Calculate x, lam, mu of minimal norm satisfying both
    # the primal and dual constraints.
    B = la.inv(A @ A.T)
    x = A.T @ B @ b
    lam = B @ A @ c
    mu = c - (A.T @ lam)

    # Perturb x and s so they are nonnegative.
    dx = max((-3./2)*x.min(), 0)
    dmu = max((-3./2)*mu.min(), 0)
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    # Perturb x and mu so they are not too small and not too dissimilar.
    dx = .5*(x*mu).sum()/mu.sum()
    dmu = .5*(x*mu).sum()/x.sum()
    x += dx*np.ones_like(x)
    mu += dmu*np.ones_like(mu)

    return x, lam, mu

# Use this linear program generator to test your interior point method.
def randomLP(j,k):
    """Generate a linear program min c^T x s.t. Ax = b, x>=0.
    First generate m feasible constraints, then add
    slack variables to convert it into the above form.
    Parameters:
        j (int >= k): number of desired constraints.
        k (int): dimension of space in which to optimize.
    Returns:
        A ((j, j+k) ndarray): Constraint matrix.
        b ((j,) ndarray): Constraint vector.
        c ((j+k,), ndarray): Objective function with j trailing 0s.
        x ((k,) ndarray): The first 'k' terms of the solution to the LP.
    """
    A = np.random.random((j,k))*20 - 10
    A[A[:,-1]<0] *= -1
    x = np.random.random(k)*10
    b = np.zeros(j)
    b[:k] = A[:k,:] @ x
    b[k:] = A[k:,:] @ x + np.random.random(j-k)*10
    c = np.zeros(j+k)
    c[:k] = A[:k,:].sum(axis=0)/k
    A = np.hstack((A, np.eye(j)))
    return A, b, -c, x


# Problems --------------------------------------------------------------------
def interiorPoint(A, b, c, niter=20, tol=1e-16, verbose=False):
    """Solve the linear program min c^T x, Ax = b, x>=0
    using an Interior Point method.

    Parameters:
        A ((m,n) ndarray): Equality constraint matrix with full row rank.
        b ((m, ) ndarray): Equality constraint vector.
        c ((n, ) ndarray): Linear objective function coefficients.
        niter (int > 0): The maximum number of iterations to execute.
        tol (float > 0): The convergence tolerance.

    Returns:
        x ((n, ) ndarray): The optimal point.
        val (float): The minimum value of the objective function.
    """
    n = len(c)
    m = len(b)
    # Problem 1:
    # Write a function for the vector valued function F described above. 
    # This function should accept x, λ, and µ as parameters and
    # return a 1-dimensional NumPy array with 2n + m entries.
    def _make_f(x, lam, mu):
        M = np.diag(mu)
        first = A.T@lam + mu - c
        second = A@x - b
        third = M@x
        temp = np.append(first, second)
        f = np.append(temp, third)
        return f

    # Problem 2: 
    # write a subroutine to compute the search direction by 
    # solving Equation 16.2. Use σ = 1/10 for the centering parameter.
    def _find_search_dir(f, x, lam, mu):
        # Put together DF
        X = np.diag(x)
        M = np.diag(mu)
        top = np.hstack((np.zeros((n,n)), A.T, np.eye(n)))
        middle = np.hstack((A, np.zeros((m, m+n))))
        bottom = np.hstack((M, np.zeros((n,m)), X))
        DF = np.vstack((top, middle, bottom))
    
        # Create nu, put it in stack
        nu = (x.T @ mu) / n
        nu_and_sigma = np.hstack((np.zeros(len(f)-len(mu)),nu/10 * np.ones(len(mu))))

        b = -f + nu_and_sigma

        # Solve DF(x) = b
        lu, piv = la.lu_factor(DF)
        search_dir = la.lu_solve((lu, piv), b)
        return search_dir, nu
        
    # Problem 3:
    # Write a subroutine to compute the step size after the search
    # direction has been computed.
    def _step_size(f, search_dir, x, lam, mu):
        change_x = search_dir[:n]
        change_mu = search_dir[n+m:]
        mask_a = change_mu < 0
        mask_d = change_x < 0
        alpha_max = np.divide(-1*mu, change_mu)[mask_a]
        delta_max = np.divide(-1*x, change_x)[mask_d]
        if len(alpha_max) == 0:
            alpha_max = 1
        else:
            alpha_max = np.min(alpha_max)
        if len(delta_max) == 0:
            delta_max = 1
        else:
            delta_max = np.min(delta_max)

        alpha = min(1, .95*alpha_max)
        delta = min(1, .95*delta_max)
        return alpha, delta


    # Problem 4: put it all together!!!
    # Get starting point
    x, lam, mu = starting_point(A, b, c)
    finished = False
    iters = 0

    while finished == False:
        # 1. Make f
        f =_make_f(x, lam, mu)

        # 2. Find step direction
        dir, nu = _find_search_dir(f, x, lam, mu)
        
        # 3. Find step size
        alpha, delta = _step_size(f, dir, x, lam, mu)

        # 4. Calculate next x, lam, mu
        x = x + delta*dir[:n]
        lam = lam + alpha*dir[n:n+m]
        mu = mu + alpha*dir[n+m:]

        # 5. See if we are done!
        if nu < tol or iters > niter:
            finished = True
        iters += 1

    return x, c@x




def leastAbsoluteDeviations(filename='simdata.txt'):
    """Generate and show the plot requested in the lab."""
    data = np.loadtxt(filename)
    x_vals = data[:,1]
    y_vals = data[:,0]
    
    m = data.shape[0]
    n = data.shape[1] - 1
    c = np.zeros(3*m + 2*(n + 1))
    c[:m] = 1
    y = np.empty(2*m)
    y[::2] = -data[:, 0]
    y[1::2] = data[:, 0]
    x = data[:, 1:]

    A = np.ones((2*m, 3*m + 2*(n + 1)))
    A[::2, :m] = np.eye(m)
    A[1::2, :m] = np.eye(m)
    A[::2, m:m+n] = -x
    A[1::2, m:m+n] = x
    A[::2, m+n:m+2*n] = x
    A[1::2, m+n:m+2*n] = -x
    A[::2, m+2*n] = -1
    A[1::2, m+2*n+1] = -1
    A[:, m+2*n+2:] = -np.eye(2*m, 2*m)

    sol = interiorPoint(A, y, c, niter=10)[0]

    beta = sol[m:m+n] - sol[m+n:m+2*n]
    beta = beta[0]
    b = sol[m+2*n] - sol[m+2*n+1]

    domain = np.linspace(0,10,200)
    slope, intercept = linregress(x_vals, y_vals)[:2]

    plt.scatter(x_vals, y_vals)
    plt.plot(domain, beta * domain + b, 'r', label="LAD")
    plt.plot(domain, slope * domain + intercept, 'b', label='Least squares')
    plt.legend()
    plt.show()