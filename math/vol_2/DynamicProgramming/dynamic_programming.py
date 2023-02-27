# dynamic_programming.py
"""Volume 2: Dynamic Programming.
<Name>
<Class>
<Date>
"""

import numpy as np
from matplotlib import pyplot as plt

def calc_stopping(N):
    """Calculate the optimal stopping time and expected value for the
    marriage problem.

    Parameters:
        N (int): The number of candidates.

    Returns:
        (float): The maximum expected value of choosing the best candidate.
        (int): The index of the maximum expected value.
    """
    optimal_val = 0
    optimal_ind = N
    V_t = 0
    
    # Do equation 18.1
    for i in range(N-1, 0, -1):
        V_t = max((i/(i+1))*V_t + 1/N, V_t)
        if V_t > optimal_val:
            # Update optimal value and index
            optimal_val = V_t
            optimal_ind = i

    return optimal_val, optimal_ind


# Problem 2
def graph_stopping_times(M):
    """Graph the optimal stopping percentage of candidates to interview and
    the maximum probability against M.

    Parameters:
        M (int): The maximum number of candidates.

    Returns:
        (float): The optimal stopping percent of candidates for M.
    """
    x = []
    y = []
    # Get the optimal values and indices for 3 through M
    domain = range(3, M+1)
    for i in range(3, M+1):
        val, ind = calc_stopping(i)
        x.append(ind/i)
        y.append(val)
    
    # Plot optimal values
    plt.plot(domain, x)
    plt.plot(domain, y)
    plt.show()



# Problem 3
def get_consumption(N, u=lambda x: np.sqrt(x)):
    """Create the consumption matrix for the given parameters.

    Parameters:
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        u (function): Utility function.

    Returns:
        C ((N+1,N+1) ndarray): The consumption matrix.
    """
    # Create partition vector w
    w = []
    for i in range(N+1):
        w.append(i/N)
    w = np.array(w)

    # Create and return consumption matrix
    A = []
    for i in range(N+1):
        col = np.append(np.zeros(i), u(w[:N-i+1]))
        A.append(col)
    
    return np.array(A).T


# Problems 4-6
def eat_cake(T, N, B, u=lambda x: np.sqrt(x)):
    """Create the value and policy matrices for the given parameters.

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        A ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            value of having w_i cake at time j.
        P ((N+1,T+1) ndarray): The matrix where the (ij)th entry is the
            number of pieces to consume given i pieces at time j.
    """
    # Get last column of value function matrix
    w = []
    for i in range(N+1):
        w.append(i/N)
    u_w = u(np.array(w))
    zeros = np.zeros((N+1, T))
    A = np.hstack((zeros, np.reshape(u_w.T, (N+1,1))))

    # Create policy matrix
    P = np.zeros_like(A)
    # Fill in last column of policy matrix
    P[:,-1] = w

    # Calculate value function matrix and fill in policy matrix
    for i in range(T-1, -1, -1):
        # Get CV_t matrix
        CV = get_consumption(N, u)
        # Add beta * the t-1th column of A to it
        CV = np.tril(CV + B*A[:,i+1])

        # update A with the largest value in each row
        A[:,i] = CV.max(axis=1)

        # update P 
        P[:,i] = P[:,-1] - (np.argmax(CV, axis=1)/N)

    return A, P


# Problem 7
def find_policy(T, N, B, u=np.sqrt):
    """Find the most optimal route to take assuming that we start with all of
    the pieces. Show a graph of the optimal policy using graph_policy().

    Parameters:
        T (int): Time at which to end (T+1 intervals).
        N (int): Number of pieces given, where each piece of cake is the
            same size.
        B (float): Discount factor, where 0 < B < 1.
        u (function): Utility function.

    Returns:
        ((T,) ndarray): The matrix describing the optimal percentage to
            consume at each time.
    """
    P = eat_cake(T, N, B, u)[1]
    c = np.zeros(T+1)
    cake_left = 1

    for i in range(T+1):
        # See how many slices we have remaining
        slices_left = int(cake_left * N)
        # See what our policy says to eat at that time interval
        # with that many slices remaining
        to_eat = P[slices_left, i]
        c[i] = to_eat
        # Eat that cake
        cake_left = cake_left - to_eat

    return c