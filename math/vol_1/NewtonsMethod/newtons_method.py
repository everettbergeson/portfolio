# newtons_method.py
"""Volume 1: Newton's Method.
Everett Bergeson
<Class>
<Date>
"""
import numpy as np
import sympy as sy
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problems 1, 3, and 5
def newton(f, x0, Df, tol=1e-5, maxiter=15, alpha=1.):
    """Use Newton's method to approximate a zero of the function f.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.
        alpha (float): Backtracking scalar (Problem 3).

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    converged = False
    iterations = 0

    # Run the 1D version if x0 is a scalar
    if np.isscalar(x0):
        for k in range(maxiter):            # Iterate at most N times
            x1 = x0 - alpha*(f(x0)/Df(x0))        # Compute next iteration
            iterations += 1                 # Count iterations 
            if abs(x1 - x0) < tol:          # Check for convergence
                converged = True            # If converged, stop
                break
            x0 = x1                         # Otherwise keep going
    # Otherwise run the n-dimensional method
    else:
        for k in range(maxiter):                       # Iterate at most N times
            yk = la.solve(Df(x0), f(x0))               # Instead of computing Df-1, solve for yk
            x1 = x0 - alpha*yk                         # Compute next iteration
            iterations += 1                            # Count iterations 
            if la.norm(x1 - x0) < tol:                 # Check for convergence
                converged = True                       # If converged, stop
                break
            x0 = x1                                    # Otherwise keep going
    
    return x1, converged, iterations

# Problem 2
def prob2(N1, N2, P1, P2):
    """Use Newton's method to solve for the constant r that satisfies

                P1[(1+r)**N1 - 1] = P2[1 - (1+r)**(-N2)].

    Use r_0 = 0.1 for the initial guess.

    Parameters:
        P1 (float): Amount of money deposited into account at the beginning of
            years 1, 2, ..., N1.
        P2 (float): Amount of money withdrawn at the beginning of years N1+1,
            N1+2, ..., N1+N2.
        N1 (int): Number of years money is deposited.
        N2 (int): Number of years money is withdrawn.

    Returns:
        (float): the value of r that satisfies the equation.
    """
    r = sy.symbols('r')

    # Declare f and find its derivative
    f = ((P1 * (((1+r)**N1) - 1)) - (P2 * (1 - ((1+r)**(-1*N2)))))
    Df = sy.diff(f, r)

    # Lambdify it so we can plug it into our Newton's method function
    f = sy.lambdify(r, f)
    Df = sy.lambdify(r, Df)

    # Return the value of r that satisfies the equation
    return newton(f, 0.1, Df)[0]


# Problem 4
def optimal_alpha(f, x0, Df, tol=1e-5, maxiter=15):
    """Run Newton's method for various values of alpha in (0,1].
    Plot the alpha value against the number of iterations until convergence.

    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float): a value for alpha that results in the lowest number of
            iterations.
    """
    iterations = []

    # Try various values of alpha, make list of how many iterations it takes
    for i in range(100):
        iter = newton(f, x0, Df, alpha=(i+1)*.01)[2]
        iterations.append(iter)
    
    # Plot the values of alpha against the number of iterations 
    plt.plot(np.linspace(1, 100, 100), iterations)
    plt.xlabel("Alpha")
    plt.ylabel("Number of iterations")
    plt.title("Number of iterations for various values of alpha")
    plt.show()

    # Return the alpha that had the lowest number of iterations
    return (np.argmin(iterations) + 1) * .01
    

# Problem 6
def prob6():
    """Consider the following Bioremediation system.

                              5xy − x(1 + y) = 0
                        −xy + (1 − y)(1 + y) = 0

    Find an initial point such that Newton’s method converges to either
    (0,1) or (0,−1) with alpha = 1, and to (3.75, .25) with alpha = 0.55.
    Return the intial point as a 1-D NumPy array with 2 entries.
    """
    # Initialize f and its derivative
    f = lambda x: np.array([5*x[0]*x[1] - x[0]*(1+x[1]), -1*x[0]*x[1] + (1-x[1])*(1+x[1])])
    Df = lambda x: np.matrix([[5*x[1] - 1 - x[1], 5*x[0] - x[0]],
                              [-1 * x[1], -1 * x[0] - 2*x[1]]])
    xs = np.linspace(-.25, 0, 20)
    ys = np.linspace(0, .25, 20)

    # Check for the answer with alpha = 1
    for x in xs:
        for y in ys:
            x0 = [x, y]
            check_alpha_1 = newton(f, x0, Df, alpha=1)[0]
            check_alpha_55 = newton(f, x0, Df, alpha=0.55)[0]
            if np.allclose(np.abs(check_alpha_1), np.array([0,1])) and np.allclose(check_alpha_55, np.array([3.75, .25])):
                return x0

# Problem 7
def plot_basins(f, Df, zeros, domain, res=1000, iters=15):
    """Plot the basins of attraction of f on the complex plane.

    Parameters:
        f (function): A function from C to C.
        Df (function): The derivative of f, a function from C to C.
        zeros (ndarray): A 1-D array of the zeros of f.
        domain ([r_min, r_max, i_min, i_max]): A list of scalars that define
            the window limits and grid domain for the plot.
        res (int): A scalar that determines the resolution of the plot.
            The visualized grid has shape (res, res).
        iters (int): The exact number of times to iterate Newton's method.
    """
    # Step 1 construct a res x res grid over the domain
    x_real = np.linspace(domain[0], domain[1], res) # Real parts.
    x_imag = np.linspace(domain[2], domain[3], res) # Imaginary parts.
    X_real, X_imag = np.meshgrid(x_real, x_imag)
    X_0 = X_real + 1j*X_imag
    Y = np.zeros_like(np.real(X_0))

    # Step 2 run Newton's method iter times
    for i in range(iters):
        X_1 = X_0 - f(X_0)/Df(X_0)
        X_0 = X_1

    # Step 3: Calculate y by finding which zero is closest to that point
    @np.vectorize
    def _closest_zero(x):
        return np.argmin(np.abs(zeros - x))
    Y = _closest_zero(X_0)

    # Step 4: visualize
    plt.pcolormesh(x_real, x_imag, Y, cmap="brg")
    plt.show()