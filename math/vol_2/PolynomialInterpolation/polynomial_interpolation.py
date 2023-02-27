# polynomial_interpolation.py
"""Volume 2: Polynomial Interpolation.
<Name>
<Class>
<Date>
"""
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import barycentric_interpolate
from scipy.interpolate import BarycentricInterpolator
import numpy.linalg as la

# Problems 1 and 2
def lagrange(xint, yint, points):
    """Find an interpolating polynomial of lowest degree through the points
    (xint, yint) using the Lagrange method and evaluate that polynomial at
    the specified points.

    Parameters:
        xint ((n,) ndarray): x values to be interpolated.
        yint ((n,) ndarray): y values to be interpolated.
        points((m,) ndarray): x values at which to evaluate the polynomial.

    Returns:
        ((m,) ndarray): The value of the polynomial at the specified points.
    """
    denominators = []
    # Compute the denominator of each Lj (9.1)
    for j in range(len(xint)):
        temp = np.delete(xint, j)
        denominators.append(np.product(xint[j] - temp))
    
    # Using previous step, evaluate Lj at all points in the computational domain
    # Combine the results into an nxm matrix consisting of
    #   each of the n Lj evaluated at each of the m points in the domain
    Lj_eval_matrix = []
    for p in points:
        eval_at_point = []
        for j in range(len(xint)):
            x_no_j = np.delete(xint, j)
            numerator = np.product(p - x_no_j)
            eval_at_point.append(numerator/denominators[j])
        Lj_eval_matrix.append(eval_at_point)
    
    # Evaluate the interpolating polynomial at each point in the domain
    p_x_list = []
    sum = 0
    for x in Lj_eval_matrix:
        sum_y_L = 0
        for j in range(len(yint)):
            sum_y_L += x[j]*yint[j]
        p_x_list.append(sum_y_L)
    return p_x_list
    

    
# Problems 3 and 4
class Barycentric:
    """Class for performing Barycentric Lagrange interpolation.

    Attributes:
        w ((n,) ndarray): Array of Barycentric weights.
        n (int): Number of interpolation points.
        x ((n,) ndarray): x values of interpolating points.
        y ((n,) ndarray): y values of interpolating points.
    """

    def __init__(self, xint, yint):
        """Calculate the Barycentric weights using initial interpolating points.

        Parameters:
            xint ((n,) ndarray): x values of interpolating points.
            yint ((n,) ndarray): y values of interpolating points.
        """
        self.x = xint
        self.y = yint
        self.n = len(xint)

        # Initialize
        w = np.ones(self.n) 
        C = (np.max(xint) - np.min(xint)) / 4
        shuffle = np.random.permutation(self.n-1)

        # Compute Barycentric weights
        for j in range(self.n):
            temp = (xint[j] - np.delete(xint, j)) / C
            temp = temp[shuffle] 
            w[j] /= np.product(temp)

        self.w = w

    def __call__(self, points):
        """Using the calcuated Barycentric weights, evaluate the interpolating polynomial
        at points.

        Parameters:
            points ((m,) ndarray): Array of points at which to evaluate the polynomial.

        Returns:
            ((m,) ndarray): Array of values where the polynomial has been computed.
        """
        evaluated = []
        for x0 in points:
            top = 0
            bottom = 0
            # Go through each weight, sum it up using p(x) formula on page 4
            for i in range(self.n):
                # Check for divide by zero error
                if (x0 - self.x[i]) != 0:
                    top += (self.w[i] * self.y[i]) / (x0 - self.x[i])
                    bottom += self.w[i] / (x0 - self.x[i])
                else:
                    # If it is dividing by zero, it's an exactly interpolating point
                    top = self.y[i]
                    bottom = 1
                    break
            evaluated.append(top/bottom)
            
        print(self.x)
        print(self.w)
        return evaluated
        

    # Problem 4
    def add_weights(self, xint, yint):
        """Update the existing Barycentric weights using newly given interpolating points
        and create new weights equal to the number of new points.

        Parameters:
            xint ((m,) ndarray): x values of new interpolating points.
            yint ((m,) ndarray): y values of new interpolating points.
        """
        for xi in xint:
            # Calculate our new weight
            new_w = 1/np.product([xi-xk for xk in self.x])

            # Update the other weights
            for j in range(len(self.w)):
                self.w[j] = self.w[j]/(self.x[j]-xi)
            
            # Update our weights to include our new weight
            self.w = np.append(self.w, new_w)

            # Update our x values to include our new x value
            self.x = np.append(self.x, xi)

        # Update y and n
        self.y = np.append(self.y, yint)
        self.n = len(self.x)

# Problem 5
def prob5():
    """For n = 2^2, 2^3, ..., 2^8, calculate the error of intepolating Runge's
    function on [-1,1] with n points using SciPy's BarycentricInterpolator
    class, once with equally spaced points and once with the Chebyshev
    extremal points. Plot the absolute error of the interpolation with each
    method on a log-log plot.
    """
    f = lambda x: 1/(1+25 * x**2)
    domain = np.linspace(-1, 1, 400)
    f_x = f(domain)
    n_list = [2**x for x in range(2, 9)]

    # Method 1: evenly spaced
    error_evenly_spaced = []
    for n in n_list:
        even_points = np.linspace(-1, 1, n)
        # Evaluate domain in interpolating function, calculate error
        even_poly = barycentric_interpolate(even_points, f(even_points), domain)
        error_evenly_spaced.append(la.norm(f_x - even_poly, ord=np.inf))
    plt.loglog(n_list, error_evenly_spaced, label='Evenly Spaced Interpolating Points Error')


    # Method 2: chebyshev extremizers as interpolating points
    error_chebyshev = []
    for n in n_list:
        # Calculate extremizers
        chebyshev_extremizers = [np.cos(i*np.pi/n) for i in range(0, n+1)]
        f_cheby = [f(j) for j in chebyshev_extremizers]

        # Evaluate domain in interpolating function, calculate error
        cheby_poly = barycentric_interpolate(chebyshev_extremizers, f_cheby, domain)
        error_chebyshev.append(la.norm(f_x - cheby_poly, ord=np.inf))
    plt.loglog(n_list, error_chebyshev, label='Chebyshev Extremizer Error')

    plt.title("Error in interpolating polynomials where the points are \nevenly spaced vs. chebyshev extremizers")
    plt.legend()
    plt.show()


# Problem 6
def chebyshev_coeffs(f, n):
    """Obtain the Chebyshev coefficients of a polynomial that interpolates
    the function f at n points.

    Parameters:
        f (function): Function to be interpolated.
        n (int): Number of points at which to interpolate.

    Returns:
        coeffs ((n+1,) ndarray): Chebyshev coefficients for the interpolating polynomial.
    """
    # Calculate coefficients
    y = np.cos((np.pi * np.arange(2*n)) / n)
    samples = f(y)
    # Run it through the fft
    coeffs = np.real(np.fft.fft(samples))[:n+1] / n
    # Scale
    coeffs[0] = coeffs[0]/2
    coeffs[n] = coeffs[n]/2
    return coeffs


# Problem 7
def prob7(n):
    """Interpolate the air quality data found in airdata.npy using
    Barycentric Lagrange interpolation. Plot the original data and the
    interpolating polynomial.

    Parameters:
        n (int): Number of interpolating points to use.
    """
    # Load data
    data = np.load('airdata.npy')

    # Take n + 1 Chebyshev extrema and find the closest match in the non-continuous data
    fx = lambda a, b, n: .5*(a+b + (b-a) * np.cos(np.arange(n+1) * np.pi / n))
    a, b = 0, 366 - 1/24
    domain = np.linspace(0, b, 8784)
    points = fx(a, b, n)
    temp = np.abs(points - domain.reshape(8784, 1))
    temp2 = np.argmin(temp, axis=0)

    # Create barycentric polynomial
    bary = BarycentricInterpolator(domain[temp2], data[temp2])
    poly = bary(domain)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    ax1.scatter(domain, data, label="Data")
    ax2.plot(domain, poly, label="Interpolation")
    ax1.legend()
    ax2.legend()
    plt.show()