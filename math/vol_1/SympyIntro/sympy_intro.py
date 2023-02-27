# sympy_intro.py
"""Python Essentials: Introduction to SymPy.
<Name>
<Class>
<Date>
"""
import sympy as sy
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def prob1():
    """Return an expression for

        (2/5)e^(x^2 - y)cosh(x+y) + (3/7)log(xy + 1).

    Make sure that the fractions remain symbolic.
    """
    x, y = sy.symbols('x, y')
    return sy.Rational(2,5)*sy.exp((x**2)-y) * sy.cosh(x+y) + sy.Rational(3,7)*sy.log((x*y)+1)


# Problem 2
def prob2():
    """Compute and simplify the following expression.

        product_(i=1 to 5)[ sum_(j=i to 5)[j(sin(x) + cos(x))] ]
    """
    x, i, j = sy.symbols('x, i, j')
    return sy.simplify(sy.product(sy.summation(j*(sy.sin(x)+sy.cos(x)), (j, i, 5)), (i, 1, 5)))

# Problem 3
def prob3(N):
    """Define an expression for the Maclaurin series of e^x up to order N.
    Substitute in -y^2 for x to get a truncated Maclaurin series of e^(-y^2).
    Lambdify the resulting expression and plot the series on the domain
    y in [-3,3]. Plot e^(-y^2) over the same domain for comparison.
    """
    # Graph e^(-1y**2)
    domain = np.linspace(-2, 2, 200)
    g = lambda y: np.e**(-1*(y**2))
    plt.plot(domain, g(domain))

    # Calculate, substitute, lambify, plot Maclaurin series
    x, y, n = sy.symbols('x, y, n')
    e_x = sy.simplify(sy.summation((x**n)/sy.factorial(n), (n, 0, N)))
    f = sy.lambdify(y, e_x.subs(x, -1*(y**2)))
    plt.plot(domain, f(domain))
    plt.show()


# Problem 4
def prob4():
    """The following equation represents a rose curve in cartesian coordinates.

    0 = 1 - [(x^2 + y^2)^(7/2) + 18x^5 y - 60x^3 y^3 + 18x y^5] / (x^2 + y^2)^3

    Construct an expression for the nonzero side of the equation and convert
    it to polar coordinates. Simplify the result, then solve it for r.
    Lambdify a solution and use it to plot x against y for theta in [0, 2pi].
    """
    # Initialize
    x, y, r, theta = sy.symbols("x, y, r, theta")
    rose = 1-(((x**2 + y**2)**sy.Rational(7, 2) + (18*(x**5)*y) - 60*((x**3)*(y**3)) + 18*(x*y**5))/(((x**2)+(y**2))**3))
    
    # Substitute to make polar
    rose = sy.simplify(rose.subs({x: r*sy.cos(theta), y: r*sy.sin(theta)}))

    # Solve
    solution = sy.solve(rose, r)

    # Graph
    f = sy.lambdify(theta, solution[0], 'numpy')
    domain = np.linspace(0, 2*np.pi, 200)
    rad = f(domain)
    x_plot = np.multiply(np.cos(domain), rad)
    y_plot = np.multiply(np.sin(domain), rad)
    plt.plot(x_plot, y_plot)
    plt.show()


# Problem 5
def prob5():
    """Calculate the eigenvalues and eigenvectors of the following matrix.

            [x-y,   x,   0]
        A = [  x, x-y,   x]
            [  0,   x, x-y]

    Returns:
        (dict): a dictionary mapping eigenvalues (as expressions) to the
            corresponding eigenvectors (as SymPy matrices).
    """
    # Initialize A and identity matrix
    x, y, lamb = sy.symbols('x y lamb')
    A = sy.Matrix([ [x-y,   x,   0], 
                    [  x, x-y,   x], 
                    [  0,   x, x-y]])
    I = sy.Matrix([ [1,0,0],
                    [0,1,0],
                    [0,0,1],])

    # Calculate eigenvalues
    eig_matrix = sy.det(A - (lamb*I))
    eig_vals = sy.solve(eig_matrix, lamb)
    
    # Get the nullspace of A - lambda*I for each eigenvalue to get the eigenvector
    eig_dict = {}
    for eig in eig_vals:
        M = A - (eig*I)
        eig_dict[eig] = M.nullspace()

    return eig_dict


# Problem 6
def prob6():
    """Consider the following polynomial.

        p(x) = 2*x^6 - 51*x^4 + 48*x^3 + 312*x^2 - 576*x - 100

    Plot the polynomial and its critical points. Determine which points are
    maxima and which are minima.

    Returns:
        (set): the local minima.
        (set): the local maxima.
    """
    local_max = set()
    local_min = set()
    y_max = []
    y_min = []
    # Solve f'(x0) = 0
    x = sy.symbols('x')
    p = 2*x**6 - 51*x**4 + 48*x**3 + 312*x**2 - 576*x - 100
    f = sy.lambdify(x, p)
    dp = sy.diff(p, x)
    crits = sy.solve(dp, x)
    
    # Test those critical points on the second derivative
    d2p = sy.lambdify(x, sy.diff(dp, x))
    for x0 in crits:
        if (d2p(x0) < 0):
            local_max.add(x0)
            y_max.append(f(x0))
        else:
            local_min.add(x0)
            y_min.append(f(x0))
    y_max.reverse()
    y_min.reverse()

    # Plot the function
    domain = np.linspace(-5, 5, 400)
    plt.plot(domain, f(domain), label='p(x)', c='k', alpha=.6)

    # Plot the local maxima and minima:
    plt.scatter(list(local_max), y_max, c='r', s=100, label='Local Maximum')
    plt.scatter(list(local_min), y_min, c='y', s=100, label='Local Minimum')
    plt.legend()
    plt.show()
    
    return y_max, y_min

# Problem 7
def prob7():
    """Calculate the integral of f(x,y,z) = (x^2 + y^2 + z^2)^2 over the
    sphere of radius r. Lambdify the resulting expression and plot the integral
    value for r in [0,3]. Return the value of the integral when r = 2.

    Returns:
        (float): the integral of f over the sphere of radius 2.
    """
    x, y, z, phi, ro, theta, r = sy.symbols('x, y, z, phi, ro, theta, r')
    f = (x**2 + y**2 + z**2)**2
    
    # Calculate determinant of Jacobian
    j = sy.Matrix( [[ro*sy.sin(phi)*sy.cos(theta)], 
                    [ro*sy.sin(phi)*sy.sin(theta)], 
                    [ro*sy.cos(phi)]] )
    J = j.jacobian([ro, phi, theta])
    det_J = sy.simplify(sy.det(J))

    # Substitute h's into f
    h = f.subs({x: ro*sy.sin(phi)*sy.cos(theta), 
                y: ro*sy.sin(phi)*sy.sin(theta), 
                z: ro*sy.cos(phi)})
    to_integrate = sy.simplify(h*det_J)

    # Calculate and plot integral
    integral = sy.integrate(to_integrate, (ro, 0, r), (theta, 0, 2*sy.pi), (phi, 0, sy.pi))
    lamb_int = sy.lambdify(r, integral, 'numpy')
    domain = np.linspace(0, 3, 200)
    plt.plot(domain, lamb_int(domain))
    plt.xlabel("Radius r")
    plt.title("Integral of (x^2+y^2+z^2)^2 over sphere with radius r")
    plt.show()
    return lamb_int(2)

if __name__=="__main__":
    print(prob7())