# montecarlo_integration.py
"""Volume 1: Monte Carlo Integration.
Everett Bergeson
"""

import numpy as np
from scipy import linalg as la
from scipy import stats
from matplotlib import pyplot as plt


# Problem 1
def ball_volume(n, N=10000):
    """Estimate the volume of the n-dimensional unit ball.

    Parameters:
        n (int): The dimension of the ball. n=2 corresponds to the unit circle,
            n=3 corresponds to the unit sphere, and so on.
        N (int): The number of random points to sample.

    Returns:
        (float): An estimate for the volume of the n-dimensional unit ball.
    """
    # Create an (n,d) array of random numbers between -1 and 1
    points = np.random.uniform(-1, 1, (n,N))
    # Calculate their norms
    lengths = la.norm(points, axis=0)

    # Count how many are inside
    num_within = np.count_nonzero(lengths < 1)
    # Return the proportion of points inside / total volume
    return (num_within * (2**n)/N)



# Problem 2
def mc_integrate1d(f, a, b, N=10000):
    """Approximate the integral of f on the interval [a,b].

    Parameters:
        f (function): the function to integrate. Accepts and returns scalars.
        a (float): the lower bound of interval of integration.
        b (float): the lower bound of interval of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over [a,b].

    Example:
        >>> f = lambda x: x**2
        >>> mc_integrate1d(f, -4, 2)    # Integrate from -4 to 2.
        23.734810301138324              # The true value is 24.
    """
    # Obtain sample of random points
    sample = np.random.uniform(a, b, N)
    # Use 11.2 to estimate
    return ((b-a)/N) * np.sum(f(sample))


# Problem 3
def mc_integrate(f, mins, maxs, N=10000):
    """Approximate the integral of f over the box defined by mins and maxs.

    Parameters:
        f (function): The function to integrate. Accepts and returns
            1-D NumPy arrays of length n.
        mins (list): the lower bounds of integration.
        maxs (list): the upper bounds of integration.
        N (int): The number of random points to sample.

    Returns:
        (float): An approximation of the integral of f over the domain.

    Example:
        # Define f(x,y) = 3x - 4y + y^2. Inputs are grouped into an array.
        >>> f = lambda x: 3*x[0] - 4*x[1] + x[1]**2

        # Integrate over the box [1,3]x[-2,1].
        >>> mc_integrate(f, [1, -2], [3, 1])
        53.562651072181225              # The true value is 54.
    """
    n = len(mins)
    # Find lengths of intervals
    b_a = [maxs[i] - mins[i] for i in range(n)]

    # Calculate volume of region of integration
    V = np.prod(b_a)

    # Randomly sample over region of integration
    samples = np.array([np.random.uniform(0, b_a[i], N) + mins[i] for i in range(n)])
    f_k = [f(k) for k in samples.T]

    # Use 11.2 to estimate integral
    return V*np.sum(f_k) / N
    

# Problem 4
def prob4():
    """Let n=4 and Omega = [-3/2,3/4]x[0,1]x[0,1/2]x[0,1].
    - Define the joint distribution f of n standard normal random variables.
    - Use SciPy to integrate f over Omega.
    - Get 20 integer values of N that are roughly logarithmically spaced from
        10**1 to 10**5. For each value of N, use mc_integrate() to compute
        estimates of the integral of f over Omega with N samples. Compute the
        relative error of estimate.
    - Plot the relative error against the sample size N on a log-log scale.
        Also plot the line 1 / sqrt(N) for comparison.
    """
    # Set up f, calculate true value of F
    n=4
    #f = lambda x: (1/((2*np.pi)**(n/2))) * np.exp((-1/2)*(x.T*x))
    f = lambda x: (1 / (2*np.pi)**2) * np.e**(np.dot(x, x)/-2)
    mins = [-3/2, 0, 0, 0]
    maxs = [3/4, 1, 1/2, 1]
    means = np.zeros(4)
    cov = np.eye(4)
    F = stats.mvn.mvnun(mins,maxs,means,cov)[0]

    # Get 20 integer values of N between 10^1 and 10^5
    N = np.logspace(1, 5, 20)
    N = [int(i) for i in N]
    # Compute f(N) for each value of N, compute relative error
    f_n = [mc_integrate(f, mins, maxs, i) for i in N]
    err = [np.abs(F - j)/np.abs(F) for j in f_n]

    # Plot the relative error against sample size
    # Also plot 1/sqrt(N) for comparison
    plt.loglog(N, err, '-o', label="Relative error")
    plt.loglog(N, 1/np.sqrt(N), '-o', label="1/root(N)")
    plt.legend()
    plt.show()