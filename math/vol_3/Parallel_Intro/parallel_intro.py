# iPyParallel - Intro to Parallel Programming
from ipyparallel import Client
from matplotlib import pyplot as plt
import numpy as np
import time

# Problem 1
def initialize():
    """
    Write a function that initializes a Client object, creates a Direct
    View with all available engines, and imports scipy.sparse as spar on
    all engines. Return the DirectView.
    """
    client = Client()
    dview = client[:]
    dview.execute("import scipy.sparse as sparse")
    return dview

# Problem 2
def variables(dx):
    """
    Write a function variables(dx) that accepts a dictionary of variables. Create
    a Client object and a DirectView and distribute the variables. Pull the variables back and
    make sure they haven't changed. Remember to include blocking.
    """
    client = Client()
    dview = client[:]
    dview.block = True
    dview.push(dx) 
    for key in dx.keys():
        print(dview.pull(key))


# Problem 3
def prob3(n=1000000):
    """
    Write a function that accepts an integer n.
    Instruct each engine to make n draws from the standard normal
    distribution, then hand back the mean, minimum, and maximum draws
    to the client. Return the results in three lists.
    
    Parameters:
        n (int): number of draws to make
        
    Returns:
        means (list of float): the mean draws of each engine
        mins (list of float): the minimum draws of each engine
        maxs (list of float): the maximum draws of each engine.
    """
    client = Client()
    dview = client[:]
    dview.block = True
    dview.execute("import numpy as np")
    def draw_normal(n):
        draws = np.random.normal(0, 1, n)
        mean = np.mean(draws)
        min_ = min(draws)
        max_ = max(draws)
        return mean, min_, max_
    results = dview.apply_sync(draw_normal, n)
    means = [result[0] for result in results]
    mins = [result[1] for result in results]
    maxs = [result[2] for result in results]
    return means, mins, maxs

# Problem 4
def prob4():
    """
    Time the process from the previous problem in parallel and serially for
    n = 1000000, 5000000, 10000000, and 15000000. To time in parallel, use
    your function from problem 3 . To time the process serially, run the drawing
    function in a for loop N times, where N is the number of engines on your machine.
    Plot the execution times against n.
    """
    n_list = [1000000, 5000000, 10000000, 15000000]
    times = []
    for n in n_list:
        start = time.time()
        means, mins, maxs = prob3(n)
        times.append(time.time() - start)
    plt.plot(n_list, times, marker='o')

    times = []
    for n in n_list:
        start = time.time()
        for i in range(4):
            draws = np.random.normal(0, 1, n)
            mean = np.mean(draws)
            min_ = min(draws)
            max_ = max(draws)
        times.append(time.time() - start)
    plt.plot(n_list, times, marker='o')
    plt.title("Efficiency of Problem 3")
    plt.xlabel("# of draws")
    plt.ylabel("Time to calculate")
    plt.show()

# Problem 5
def parallel_trapezoidal_rule(f, a, b, n=200):
    """
    Write a function that accepts a function handle, f, bounds of integration,
    a and b, and a number of points to use, n. Split the interval of
    integration among all available processors and use the trapezoidal
    rule to numerically evaluate the integral over the interval [a,b].

    Parameters:
        f (function handle): the function to evaluate
        a (float): the lower bound of integration
        b (float): the upper bound of integration
        n (int): the number of points to use; defaults to 200
    Returns:
        value (float): the approximate integral calculated by the
            trapezoidal rule
    """
    xs = np.linspace(a, b, n)
    client = Client()
    dview = client[:]
    dview.block = True
    dview.scatter('xs', xs)
    dview.push({'f':f})
    dview.execute("f_x = [f(x) for x in xs]")
    results = dview.pull('f_x')
    flattened_list = [i for sublist in results for i in sublist]
    total = sum(flattened_list)
    h = xs[1] - xs[0]
    return (total - (xs[0]/2) - (xs[-1]/2)) * h