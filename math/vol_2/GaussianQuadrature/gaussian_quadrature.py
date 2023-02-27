# quassian_quadrature.py
"""Volume 2: Gaussian Quadrature.
Everett Bergeson
"""
import numpy as np
from scipy import linalg as la
from scipy.integrate import quad
from scipy.stats import norm
from matplotlib import pyplot as plt
from scipy.integrate import quad


class GaussianQuadrature:
    """Class for integrating functions on arbitrary intervals using Gaussian
    quadrature with the Legendre polynomials or the Chebyshev polynomials.
    """
    # Problems 1 and 3
    def __init__(self, n, polytype="legendre"):
        """Calculate and store the n points and weights corresponding to the
        specified class of orthogonal polynomial (Problem 3). Also store the
        inverse weight function w(x)^{-1} = 1 / w(x).

        Parameters:
            n (int): Number of points and weights to use in the quadrature.
            polytype (string): The class of orthogonal polynomials to use in
                the quadrature. Must be either 'legendre' or 'chebyshev'.

        Raises:
            ValueError: if polytype is not 'legendre' or 'chebyshev'.
        """
        # Assign a w inverse function depending on the polytype, or raise a valueError
        self.n = n
        if polytype == "legendre":
            self.w_inv = lambda x: 1
        elif polytype == "chebyshev":
            self.w_inv = lambda x: np.sqrt(1-x**2)
        else:
            raise ValueError("Invalid Type")
        
        self.label = polytype
        
        # Store points and weights as attributes
        self.points, self.weights = self.points_weights(self.n)

    # Problem 2
    def points_weights(self, n):
        """Calculate the n points and weights for Gaussian quadrature.

        Parameters:
            n (int): The number of desired points and weights.

        Returns:
            points ((n,) ndarray): The sampling points for the quadrature.
            weights ((n,) ndarray): The weights corresponding to the points.
        """
        # Construct n x n Jacobi matrix J for the polynomial family
        #   indicated in the constructor
        J = np.zeros((self.n,self.n))
        alpha = 0
        beta = []
        measure = np.pi
        # Calculate alpha and beta according to family
        if self.label == "legendre":
            measure = 2
            for k in range(1, self.n+1):
                beta.append((k**2) / (4*(k**2) - 1))
        else:
            for k in range(1, self.n+1):
                if k == 1:
                    beta.append(1/2)
                else:
                    beta.append(1/4)

        # Fill Jacobi Matrix
        for i in range(self.n-1):
            J[i][i] = alpha
            J[i][i+1] = np.sqrt(beta[i])
            J[i+1][i] = np.sqrt(beta[i])
    
        # Use SciPy to compute the eigenvalues and vectors of J
        eigvalues, eigvectors = la.eig(J)

        # Sort the eigenvalues and vectors
        order_eig = np.argsort(eigvalues)
        sorted_eigvalues = [eigvalues[i] for i in order_eig]
        sorted_eigvectors = [eigvectors.T[i] for i in order_eig]

        # Compute the weights
        first_values_squared = [v[0]**2 for v in sorted_eigvectors]
        weights = [measure * v_i for v_i in first_values_squared]

        return np.real(sorted_eigvalues), np.real(weights)
        
        

    # Problem 3
    def basic(self, f):
        """Approximate the integral of a f on the interval [-1,1]."""
        g_x = []
        for x in self.points:
            g_x.append(f(x)*self.w_inv(x))
        return np.inner(g_x, self.weights)

    # Problem 4
    def integrate(self, f, a, b):
        """Approximate the integral of a function on the interval [a,b].

        Parameters:
            f (function): Callable function to integrate.
            a (float): Lower bound of integration.
            b (float): Upper bound of integration.

        Returns:
            (float): Approximate value of the integral.
        """
        h_x = lambda x: f((((b-a)/2)*x) + (a+b)/2)
        return ((b-a)/2) * self.basic(h_x)

    # Problem 6.
    def integrate2d(self, f, a1, b1, a2, b2):
        """Approximate the integral of the two-dimensional function f on
        the interval [a1,b1]x[a2,b2].

        Parameters:
            f (function): A function to integrate that takes two parameters.
            a1 (float): Lower bound of integration in the x-dimension.
            b1 (float): Upper bound of integration in the x-dimension.
            a2 (float): Lower bound of integration in the y-dimension.
            b2 (float): Upper bound of integration in the y-dimension.

        Returns:
            (float): Approximate value of the integral.
        """
        h = lambda x, y: f(((b1-a1)/2)*x + (a1+b1)/2, ((b2-a2)/2)*x + (a2+b2)/2)
        sum = 0
        
        # Compute and return the double sum in 10.5
        for i in range(len(self.points)):
            for j in range(len(self.points)):
                x = self.points[i]
                y = self.points[j]
                g_x = h(x, y) * self.w_inv(x) * self.w_inv(y)
                sum += g_x * self.weights[i] * self.weights[j]
        return (b1-a1)*(b2-a2)*sum/4



        

# Problem 5
def prob5():
    """Use scipy.stats to calculate the "exact" value F of the integral of
    f(x) = (1/sqrt(2 pi))e^((-x^2)/2) from -3 to 2. Then repeat the following
    experiment for n = 5, 10, 15, ..., 50.
        1. Use the GaussianQuadrature class with the Legendre polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
        2. Use the GaussianQuadrature class with the Chebyshev polynomials to
           approximate F using n points and weights. Calculate and record the
           error of the approximation.
    Plot the errors against the number of points and weights n, using a log
    scale for the y-axis. Finally, plot a horizontal line showing the error of
    scipy.integrate.quad() (which doesn’t depend on n).
    """
    # Initialize our function and lists for storing values
    f = lambda x: (1/np.sqrt(2*np.pi))*np.exp((-1*(x**2))/2)
    exact_value = norm.cdf(2) - norm.cdf(-3)
    scipy_approx = quad(f, -3, 2)[0]
    error_legendre = []
    error_chebyshev = []
    error_scipy = []
    
    # For n = 5, 10, ... , 50 calculate error for legendre and chebyshev
    for n in range(1, 11):
        legendre_approx = GaussianQuadrature(n*5, polytype="legendre")
        error_legendre.append(abs(legendre_approx.integrate(f, -3, 2) - exact_value))
        chebyshev_approx = GaussianQuadrature(n*5, polytype="chebyshev")
        error_chebyshev.append(abs(chebyshev_approx.integrate(f, -3, 2) - exact_value))
        error_scipy.append(abs(exact_value - scipy_approx))
    domain = np.linspace(5, 50, 10)

    # Plot the absolute error
    plt.plot(domain, error_legendre, "o-", label="Legendre Approx Error")
    plt.plot(domain, error_chebyshev, "o-", label="Chebyshev Approx Error")
    plt.plot(domain, error_scipy, "o-", label="Chebyshev Approx Error")
    plt.yscale("log")
    plt.legend()
    plt.title("Absolute error of integration")
    plt.ylabel("Absolute error")
    plt.xlabel("n")
    plt.show()