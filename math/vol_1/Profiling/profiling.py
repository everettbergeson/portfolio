# profiling.py
"""Python Essentials: Profiling.
<Name>
<Class>
<Date>
"""

# Note: for problems 1-4, you need only implement the second function listed.
# For example, you need to write max_path_fast(), but keep max_path() unchanged
# so you can do a before-and-after comparison.

import numpy as np
from numba import jit
import time
from matplotlib import pyplot as plt


# Problem 1
def max_path(filename="triangle.txt"):
    """Find the maximum vertical path in a triangle of values."""
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]
    def path_sum(r, c, total):
        """Recursively compute the max sum of the path starting in row r
        and column c, given the current total.
        """
        total += data[r][c]
        if r == len(data) - 1:          # Base case.
            return total
        else:                           # Recursive case.
            return max(path_sum(r+1, c,   total),   # Next row, same column
                       path_sum(r+1, c+1, total))   # Next row, next column

    return path_sum(0, 0, 0)            # Start the recursion from the top.

def max_path_fast(filename="triangle_large.txt"):
    """Find the maximum vertical path in a triangle of values."""
    # Process triangle
    with open(filename, 'r') as infile:
        data = [[int(n) for n in line.split()]
                        for line in infile.readlines()]

    height = len(data) - 1
    # Start at the second to last row, going up the triangle
    for i in range(height-1, -1, -1):
        # Replace each entry with the sum of the current entry 
        #    and the greater of the two child entries
        for j in range(len(data[i])):
            data[i][j] = data[i][j] + max(data[i+1][j], data[i+1][j+1])
    
    # Print the brute force results
    """
    %time brute = max_path(filename)
    print("Brute force solution:", brute)
    %time max_path_fast)(filename)
    print("Bottom up solution:", data[0][0])
    """

    # Return the top of the triangle
    return data[0][0]



# Problem 2
def primes(N):
    """Compute the first N primes."""
    primes_list = []
    current = 2
    while len(primes_list) < N:
        isprime = True
        for i in range(2, current): # Check for nontrivial divisors.
            if current % i == 0:
                isprime = False
        if isprime:
            primes_list.append(current)
        current += 1
    return primes_list

import math
def primes_fast(N):
    """Compute the first N primes."""
    primes_list = []
    current = 3
    length = 0

    # Check to see if N is at least 1 so we'll include 2
    if N > 1:
        primes_list.append(2)
        length += 1

    while length < N:
        isprime = True
        # If i divides n, i <= sqrt(n)
        # Also count by 2's since we know every prime is odd besides 2
        for i in range(3, int(math.sqrt(current)+1),2):
            # Break if it has any divisor besides itself
            if current % i == 0:
                isprime = False
                break
        if isprime:
            primes_list.append(current)
            length += 1
        current += 2
    return primes_list


# Problem 3
def nearest_column(A, x):
    """Find the index of the column of A that is closest to x.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    distances = []
    for j in range(A.shape[1]):
        distances.append(np.linalg.norm(A[:,j] - x))
    return np.argmin(distances)

def nearest_column_fast(A, x):
    """Find the index of the column of A that is closest in norm to x.
    Refrain from using any loops or list comprehensions.

    Parameters:
        A ((m,n) ndarray)
        x ((m, ) ndarray)

    Returns:
        (int): The index of the column of A that is closest in norm to x.
    """
    return np.argmin(np.linalg.norm((A.T-x), axis=1))


# Problem 4
def name_scores(filename="names.txt"):
    """Find the total of the name scores in the given file."""
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    for i in range(len(names)):
        name_value = 0
        for j in range(len(names[i])):
            alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            for k in range(len(alphabet)):
                if names[i][j] == alphabet[k]:
                    letter_value = k + 1
            name_value += letter_value
        total += (names.index(names[i]) + 1) * name_value
    return total


def name_scores_fast(filename='names.txt'):
    """Find the total of the name scores in the given file."""
    # Create list of names
    with open(filename, 'r') as infile:
        names = sorted(infile.read().replace('"', '').split(','))
    total = 0
    # Create dictionary of alphabet
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    alph_dict = {alphabet[a]:a+1 for a in range(26)}

    # First half takes 3 milliseconds

    # For each name in the list
    for i, name in enumerate(names):
        # Is there some way to eliminate this loop?
        # This one adds .001 seconds to the process... 
        #total += (i+1) * sum((alph_dict[letter] for letter in name))
        for letter in name:
            total += (i+1) * alph_dict[letter]
    return total

# Problem 5
def fibonacci():
    """Yield the terms of the Fibonacci sequence with F_1 = F_2 = 1."""
    F1 = 1
    yield F1
    F2 = 1
    yield F2
    while True:
        # Calculate F_N, move up F_N-1, F_N-2
        FNext = F1 + F2
        F1 = F2
        F2 = FNext
        yield FNext

def fibonacci_digits(N=1000):
    """Return the index of the first term in the Fibonacci sequence with
    N digits.

    Returns:
        (int): The index.
    """
    for i, x in enumerate(fibonacci()):
        # Check to see if it has N digits
        if x > 10**(N-1):
            return i + 3

# Problem 6
def prime_sieve(N):
    """Yield all primes that are less than N."""
    # 1. Given a cap N, start with all the integers from 2 to N
    num_list = np.arange(2, N+1)

    # 4. Return to step 2 until list is empty
    while len(num_list) != 0:
        
        # 3. Yield the first entry in the list and remove it from the list
        first = num_list[0] 
        yield first

        # 2. Remove all integers that are divisible by the first entry in the list
        num_list = [x for x in num_list if x % first]

# Problem 7
def matrix_power(A, n):
    """Compute A^n, the n-th power of the matrix A."""
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

@jit
def matrix_power_numba(A, n):
    """Compute A^n, the n-th power of the matrix A, with Numba optimization."""
    # Create a copy of A
    product = A.copy()
    temporary_array = np.empty_like(A[0])
    m = A.shape[0]

    # run it n times
    for power in range(1, n):
        for i in range(m):
            for j in range(m):
                total = 0
                for k in range(m):
                    total += product[i,k] * A[k,j]
                temporary_array[j] = total
            product[i] = temporary_array
    return product

def prob7(n=10):
    """Time matrix_power(), matrix_power_numba(), and np.linalg.matrix_power()
    on square matrices of increasing size. Plot the times versus the size.
    """
    # Run matrix_power_numba() once with a small random input so it compiles
    test = np.random.random((2,2))
    comp = matrix_power_numba(test, 2)

    time_no_numba = []
    time_numba = []
    time_linalg = []

    # Generate a random m x m matrix 
    for i in range(2, 8):
        A = np.random.random((2**i,2**i))
        print(np.shape(A))
        t0 = time.time()
        matrix_power(A, n)
        t1 = time.time()
        matrix_power_numba(A, n)
        t2 = time.time()
        np.linalg.matrix_power(A, n)
        t3 = time.time()

        time_no_numba.append(t1 - t0)
        time_numba.append(t2 - t1)
        time_linalg.append(t3 - t2)

    x = np.arange(2, 8)
    plt.loglog(x, time_no_numba, label="Python")
    plt.loglog(x, time_numba, label="Numba")
    plt.loglog(x, time_linalg, label="NumPy")
    plt.xlabel("Size of matrix")
    plt.ylabel("Time")
    plt.legend()
    plt.show()



if __name__=="__main__":
    """
    # Test problem 2
    for i in range(10):
        start = time.time()
        primes_fast(10000)
        finish = time.time()
        print(finish-start)
    
    
    # Test problem 3
    #A = np.array([[1, 2, 5], [1, 2, 3]])
    #x = np.array([4, 3])
    #print(nearest_column_fast(A, x))

    # Test problem 4
    start = time.time()
    print(name_scores())
    finish = time.time()
    print(finish-start)
    total = 0
    for i in range(100):
        start = time.time()
        name_scores_fast()
        finish = time.time()
        total += finish-start
    print(total/100)
    
    # Test problem 6
    start = time.time()
    empty = [p for p in prime_sieve(100)]
    print(empty)
    finish = time.time()
    print(finish-start)
    """
    print(primes_fast(10))