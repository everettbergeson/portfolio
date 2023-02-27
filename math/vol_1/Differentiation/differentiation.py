# differentiation.py
"""Volume 1: Differentiation.
<Name>
<Class>
<Date>
"""
import numpy as np
import sympy as sy
from matplotlib import pyplot as plt
from autograd import numpy as anp
from autograd import grad
from autograd import elementwise_grad
import random
import time


# Problem 1
def prob1():
    """Return the derivative of (sin(x) + 1)^sin(cos(x)) using SymPy."""
    x = sy.symbols('x')

    # Create our function and its derivative
    f = (sy.sin(x) + 1)**(sy.sin(sy.cos(x)))
    df = sy.lambdify(x, sy.diff(f))

    return df

# Problem 2
def fdq1(f, x, h=1e-5):
    """Calculate the first order forward difference quotient of f at x."""
    fdq1_result = []
    for x0 in x:
        fdq1_result.append((f(x0 + h) - f(x0))/h)
    return fdq1_result

def fdq2(f, x, h=1e-5):
    """Calculate the second order forward difference quotient of f at x."""
    fdq2_result = []
    for x0 in x:
        fdq2_result.append((-3*f(x0) + 4*f(x0 + h) - f(x0 + 2*h))/(2*h))
    return fdq2_result

def bdq1(f, x, h=1e-5):
    """Calculate the first order backward difference quotient of f at x."""
    bdq1_result = []
    for x0 in x:
        bdq1_result.append((f(x0)-f(x0-h))/h)
    return bdq1_result

def bdq2(f, x, h=1e-5):
    """Calculate the second order backward difference quotient of f at x."""
    bdq2_result = []
    for x0 in x:
        bdq2_result.append((3*f(x0)-4*f(x0 - h) + f(x0 - 2*h))/(2*h))
    return bdq2_result

def cdq2(f, x, h=1e-5):
    """Calculate the second order centered difference quotient of f at x."""
    cdq2_result = []
    for x0 in x:
        cdq2_result.append((f(x0 + h) - f(x0 - h))/(2*h))
    return cdq2_result

def cdq4(f, x, h=1e-5):
    """Calculate the fourth order centered difference quotient of f at x."""
    cdq4_result = []
    for x0 in x:
        cdq4_result.append((f(x0-2*h)-8*f(x0-h)+8*f(x0+h)-f(x0+2*h))/(12*h))
    return cdq4_result


# Problem 3
def prob3(x0):
    """Let f(x) = (sin(x) + 1)^(sin(cos(x))). Use prob1() to calculate the
    exact value of f'(x0). Then use fdq1(), fdq2(), bdq1(), bdq2(), cdq1(),
    and cdq2() to approximate f'(x0) for h=10^-8, 10^-7, ..., 10^-1, 1.
    Track the absolute error for each trial, then plot the absolute error
    against h on a log-log scale.

    Parameters:
        x0 (float): The point where the derivative is being approximated.
    """

    # Use problem 1 to calculate the exact derivative
    # Create our function and its derivative
    x = sy.symbols('x')
    f = sy.lambdify(x, (sy.sin(x) + 1)**(sy.sin(sy.cos(x))))
    df = prob1()

    domain = np.logspace(-8, 0, 9)
    fdq1_list = []
    fdq2_list = []
    bdq1_list = []
    bdq2_list = []
    cdq2_list = []
    cdq4_list = []

    # Calculate the exact value of f'(x0)
    exact = df(x0)
    x0 = np.array([x0])

    # For different values of h, calculate each of the errors
    for h in domain:
        fdq1_list.append(abs(fdq1(f, x0, h) - exact))
        fdq2_list.append(abs(fdq2(f, x0, h) - exact))
        bdq1_list.append(abs(bdq1(f, x0, h) - exact))
        bdq2_list.append(abs(bdq2(f, x0, h) - exact))
        cdq2_list.append(abs(cdq2(f, x0, h) - exact))
        cdq4_list.append(abs(cdq4(f, x0, h) - exact))
    plt.loglog(domain, fdq1_list, "o-", label="Order 1 Forward")
    plt.loglog(domain, fdq2_list, "o-", label="Order 2 Forward")
    plt.loglog(domain, bdq1_list, "o-", label="Order 1 Backward")
    plt.loglog(domain, bdq2_list, "o-", label="Order 2 Backward")
    plt.loglog(domain, cdq2_list, "o-", label="Order 2 Centered")
    plt.loglog(domain, cdq4_list, "o-", label="Order 4 Centered")
    plt.xlabel("h")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.show()


# Problem 4
def prob4():
    """The radar stations A and B, separated by the distance 500m, track a
    plane C by recording the angles alpha and beta at one-second intervals.
    Your goal, back at air traffic control, is to determine the speed of the
    plane.

    Successive readings for alpha and beta at integer times t=7,8,...,14
    are stored in the file plane.npy. Each row in the array represents a
    different reading; the columns are the observation time t, the angle
    alpha (in degrees), and the angle beta (also in degrees), in that order.
    The Cartesian coordinates of the plane can be calculated from the angles
    alpha and beta as follows.

    x(alpha, beta) = a tan(beta) / (tan(beta) - tan(alpha))
    y(alpha, beta) = (a tan(beta) tan(alpha)) / (tan(beta) - tan(alpha))

    Load the data, convert alpha and beta to radians, then compute the
    coordinates x(t) and y(t) at each given t. Approximate x'(t) and y'(t)
    using a first order forward difference quotient for t=7, a first order
    backward difference quotient for t=14, and a second order centered
    difference quotient for t=8,9,...,13. Return the values of the speed at
    each t.
    """
    # Load the data
    data = np.load("plane.npy")
    alpha_list = [t[1] for t in data]
    beta_list = [t[2] for t in data]
    x_list = []
    y_list = []

    # Convert alpha and beta to radians
    alpha_list = np.deg2rad(alpha_list)
    beta_list = np.deg2rad(beta_list)

    # Compute x(t) and y(t)
    for i in range(len(alpha_list)):
        alpha = alpha_list[i]
        beta = beta_list[i]
        bottom = (np.tan(beta) - np.tan(alpha))
        x_list.append(500 * np.tan(beta)/bottom)
        y_list.append(500 * np.tan(beta)*np.tan(alpha)/bottom)

    x_prime = []
    y_prime = []
    # Approximate x'(t) and y'(t) using 
    #   - first order forward difference quotient for t=7
    x_prime.append(x_list[1] - x_list[0])
    y_prime.append(y_list[1] - y_list[0])

    #   - second order centered difference quotient for t=8,9,...,13
    for i in range(1, 7):
        x_prime.append((x_list[i+1] - x_list[i-1])/2)
        y_prime.append((y_list[i+1] - y_list[i-1])/2)

    #   - first order backward difference quotient for t=14
    x_prime.append(x_list[7] - x_list[6])
    y_prime.append(y_list[7] - y_list[6])

    return [np.sqrt(x_prime[i]**2 + y_prime[i]**2) for i in range(len(x_prime))]


# Problem 5
def jacobian_cdq2(f, x, h=1e-5):
    """Approximate the Jacobian matrix of f:R^n->R^m at x using the second
    order centered difference quotient.

    Parameters:
        f (function): the multidimensional function to differentiate.
            Accepts a NumPy (n,) ndarray and returns an (m,) ndarray.
            For example, f(x,y) = [x+y, xy**2] could be implemented as follows.
            >>> f = lambda x: np.array([x[0] + x[1], x[0] * x[1]**2])
        x ((n,) ndarray): the point in R^n at which to compute the Jacobian.
        h (float): the step size in the finite difference quotient.

    Returns:
        ((m,n) ndarray) the Jacobian matrix of f at x.
    """
    n = len(x)
    I = np.identity(n)
    jacobian = []
    for i in range(n):
        jacobian.append((f(x + h*I[i]) - f(x - h*I[i]))/(2*h))
    return jacobian


# Problem 6
def cheb_poly(x, n):
    """Compute the nth Chebyshev polynomial at x.

    Parameters:
        x (autograd.ndarray): the points to evaluate T_n(x) at.
        n (int): The degree of the polynomial.
    """
    # n = 0
    if n == 0:
        return anp.ones_like(x)
    # n = 1
    elif n == 1:
        return x
    # Recursively calculate if n >= 2
    else:
        return 2*x*cheb_poly(x, n-1) - cheb_poly(x, n-2)

def prob6():
    """Use Autograd and cheb_poly() to create a function for the derivative
    of the Chebyshev polynomials, and use that function to plot the derivatives
    over the domain [-1,1] for n=0,1,2,3,4.
    """
    domain = anp.linspace(-1,1,200)
    d_cheb = elementwise_grad(cheb_poly)
    
    # Plot derivatives for degree 0,...,5 chebyshev polynomials
    for n in range(0, 5):
        label_str = "Degree " + str(n)
        plt.plot(domain, d_cheb(domain, n), label=label_str)

    plt.legend()
    plt.title("Derivatives of degree 0 - 5 Chebyshev Polynomials")
    plt.show()

# Problem 7
def prob7(N=200):
    """Let f(x) = (sin(x) + 1)^sin(cos(x)). Perform the following experiment N
    times:

        1. Choose a random value x0.
        2. Use prob1() to calculate the “exact” value of f′(x0). Time how long
            the entire process takes, including calling prob1() (each
            iteration).
        3. Time how long it takes to get an approximation of f'(x0) using
            cdq4(). Record the absolute error of the approximation.
        4. Time how long it takes to get an approximation of f'(x0) using
            Autograd (calling grad() every time). Record the absolute error of
            the approximation.

    Plot the computation times versus the absolute errors on a log-log plot
    with different colors for SymPy, the difference quotient, and Autograd.
    For SymPy, assume an absolute error of 1e-18.
    """
    x = sy.symbols('x')
    f = sy.lambdify(x, (sy.sin(x) + 1)**(sy.sin(sy.cos(x))))
    df = prob1()
    anp_f = lambda x: (anp.sin(x) + 1)**(anp.sin(anp.cos(x)))

    sympy_time = []
    sympy_error = []
    diff_quotients_time = []
    diff_quotients_error = []
    autograd_time = []
    autograd_error = []

    # 1. Generate a random number x0
    random_list = [random.random() for _ in range(N)]
    for x0 in random_list:
        x0_iter = np.array([x0])

        # 2. Calculate exact value of f'(x0), time how long it takes
        t1 = time.time()
        exact_value = df(x0)

        # 3. Time how long it takes to get an approximation
        #    using 4th order centered difference quotient. 
        t2 = time.time()
        d_q_value = cdq4(f, x0_iter)

        # 4. Time how long it takes to get an approximation
        #    using Autograd (caling grad() every time)
        t3 = time.time()
        autograd_f = grad(anp_f)
        autograd_value = autograd_f(x0)
        t4 = time.time()

        # Calculate time
        sympy_time.append(t2-t1)
        diff_quotients_time.append(t3-t2)
        autograd_time.append(t4-t3)

        # Calculate error
        sympy_error.append(1e-18)
        diff_quotients_error.append(abs(d_q_value - exact_value))
        autograd_error.append(abs(autograd_value - exact_value))
        
    # Plot time vs. error
    plt.loglog(sympy_time, sympy_error, "o", alpha = .5, label="Sympy")
    plt.loglog(diff_quotients_time, diff_quotients_error, "o", alpha = .5, label="Difference Quotient")
    plt.loglog(autograd_time, autograd_error, "o", alpha = .5, label="Autograd")
    plt.legend()
    plt.xlabel("Computation Time (seconds)")
    plt.ylabel("Absolute Error")
    plt.show()