# iterative_solvers.py
"""Volume 1: Iterative Solvers.
<Name>
<Class>
<Date>
"""
import scipy.linalg as la
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse


# Helper function
def diag_dom(n, num_entries=None):
    """Generate a strictly diagonally dominant (n, n) matrix.

    Parameters:
        n (int): The dimension of the system.
        num_entries (int): The number of nonzero values.
            Defaults to n^(3/2)-n.

    Returns:
        A ((n,n) ndarray): A (n, n) strictly diagonally dominant matrix.
    """
    if num_entries is None:
        num_entries = int(n**1.5) - n
    A = np.zeros((n,n))
    rows = np.random.choice(np.arange(0,n), size=num_entries)
    cols = np.random.choice(np.arange(0,n), size=num_entries)
    data = np.random.randint(-4, 4, size=num_entries)
    for i in range(num_entries):
        A[rows[i], cols[i]] = data[i]
    for i in range(n):
        A[i,i] = np.sum(np.abs(A[i])) + 1
    return A

# Problems 1 and 2
def jacobi(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Jacobi Method.

    Parameters:
        A ((n,n) ndarray): A square matrix.
        b ((n ,) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize
    done = False
    iters = 0
    abs_errors = []
    x0 = np.zeros_like(b)
    D_inv = np.diag(1/np.diag(A))

    while done == False:
        # Jacobi's Method (15.2)
        x1 = x0 + D_inv @ (b - A @ x0)

        # Check if converged or reached maxiters
        norm = la.norm(x1 - x0, np.inf)
        abs_errors.append(norm)
        if norm < tol or iters > maxiter:
            done = True

        # Otherwise, iterate again!
        iters += 1
        x0 = x1

    # Show plot of error vs iteration
    if plot == True:
        plt.semilogy(range(iters), abs_errors)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()

    return x1

# Problem 3
def gauss_seidel(A, b, tol=1e-8, maxiter=100, plot=False):
    """Calculate the solution to the system Ax = b via the Gauss-Seidel Method.

    Parameters:
        A ((n, n) ndarray): A square matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.
        plot (bool): If true, plot the convergence rate of the algorithm.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize
    done = False
    iters = 0
    abs_errors = []
    x0 = np.zeros_like(b)
    
    while done == False:
        # Gauss-Seidel Method (15.4)
        x1 = np.zeros_like(b)
        for i in range(len(b)):
            x1[i] = x0[i] + (1/A[i,i] * (b[i] - A[i,:]@x0))

        # Check if converged or reached maxiters
        norm = la.norm(x1 - x0, np.inf)
        abs_errors.append(norm)
        if norm < tol or iters > maxiter:
            done = True

        # Otherwise, iterate again!
        iters += 1
        x0 = np.copy(x1)

    # Show plot of error vs iteration
    if plot == True:
        plt.semilogy(range(iters), abs_errors)
        plt.title("Convergence of Jacobi Method")
        plt.xlabel("Iteration")
        plt.ylabel("Absolute Error of Approximation")
        plt.show()

    return x1


# Problem 4
def gauss_seidel_sparse(A, b, tol=1e-8, maxiter=100):
    """Calculate the solution to the sparse system Ax = b via the Gauss-Seidel
    Method.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse CSR matrix.
        b ((n, ) ndarray): A vector of length n.
        tol (float): The convergence tolerance.
        maxiter (int): the maximum number of iterations to perform.

    Returns:
        x ((n,) ndarray): The solution to system Ax = b.
    """
    # Initialize
    done = False
    iters = 0
    x0 = np.zeros_like(b)
    diag = A.diagonal()
    
    while done == False:
        # Gauss-Seidel Method (15.4)
        x1 = np.zeros_like(b)
        for i in range(len(b)):
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]
            x1[i] = x0[i] + (1/diag[i] * (b[i] - Aix))

        # Check if converged or reached maxiters
        norm = la.norm(x1 - x0, np.inf)
        if norm < tol or iters > maxiter:
            done = True

        # Otherwise, iterate again!
        iters += 1
        x0 = np.copy(x1)

    return x1


# Problem 5
def sor(A, b, omega, tol=1e-8, maxiter=100):
    """Calculate the solution to the system Ax = b via Successive Over-
    Relaxation.

    Parameters:
        A ((n, n) csr_matrix): A (n, n) sparse matrix.
        b ((n, ) Numpy Array): A vector of length n.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The convergence tolerance.
        maxiter (int): The maximum number of iterations to perform.

    Returns:
        ((n,) ndarray): The solution to system Ax = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    # Initialize
    done = False
    converged = False
    iters = 0
    x0 = np.zeros_like(b)
    diag = A.diagonal()
    
    while done == False:
        # SOR Method (15.5)
        x1 = np.zeros_like(b)
        temp_x = np.copy(x0)
        for i in range(len(b)):
            # Get the indices of where the i-th row of A starts and ends if the
            # nonzero entries of A were flattened.
            rowstart = A.indptr[i]
            rowend = A.indptr[i+1]
            # Multiply only the nonzero elements of the i-th row of A with the
            # corresponding elements of x.
            Aix = A.data[rowstart:rowend] @ x0[A.indices[rowstart:rowend]]
            x0[i] = x0[i] + (omega/diag[i] * (b[i] - Aix))

        # Check if converged or reached maxiters
        norm = la.norm(x0 - temp_x, np.inf)
        if norm < tol:
            converged = True
        if converged == True or iters > maxiter:
            done = True
            break

        # Otherwise, iterate again!
        iters += 1

    return x0, converged, iters




# Problem 6
def hot_plate(n, omega, tol=1e-8, maxiter=100, plot=False):
    """Generate the system Au = b and then solve it using sor().
    If show is True, visualize the solution with a heatmap.

    Parameters:
        n (int): Determines the size of A and b.
            A is (n^2, n^2) and b is one-dimensional with n^2 entries.
        omega (float in [0,1]): The relaxation factor.
        tol (float): The iteration tolerance.
        maxiter (int): The maximum number of iterations.
        plot (bool): Whether or not to visualize the solution.

    Returns:
        ((n^2,) ndarray): The 1-D solution vector u of the system Au = b.
        (bool): Whether or not Newton's method converged.
        (int): The number of computed iterations in SOR.
    """

    def _sparse_matrix(n):
        """Parameters:
            n (int): Dimensions of the sparse matrix B.
        Returns:
            A ((n**2,n**2) SciPy sparse matrix)"""
        # Initialize List
        diag_list = []
        # Make the diagonal lists for positioins -1 and 1
        for x in range(n):
            for y in range(n - 1):
                diag_list.append(1)
            if x is not n - 1:
                diag_list.append(0)
        # Make the A Matrix
        A = sparse.diags([1, diag_list, -4, diag_list, 1], [-n, -1, 0, 1, n], shape=(n ** 2, n ** 2))
        return A
        
    # Generate A and b
    A = _sparse_matrix(n)
    b = np.zeros(n)
    b[0] = -100
    b[-1] = -100
    b = np.tile(b, n)

    # Run sor
    u, con, iters = sor(sparse.csr_matrix(A), b, omega, tol=tol, maxiter=maxiter)
    u = u.reshape(n,n)

    # plot
    if plot==True:
        plt.pcolormesh(u, cmap="coolwarm")
        plt.title("Hotplate")
        plt.show()

    return u, con, iters


# Problem 7
def prob7():
    """Run hot_plate() with omega = 1, 1.05, 1.1, ..., 1.9, 1.95, tol=1e-2,
    and maxiter = 1000 with A and b generated with n=20. Plot the iterations
    computed as a function of omega.
    """
    # Run for different values of omega
    omegas = np.linspace(1,1.95,20)
    iterations = []
    for om in omegas:
        u, converged, it = hot_plate(20, om, tol=1e-2, maxiter=1000)
        iterations.append(it)
        
    # Plot iterations as a function of omega
    plt.plot(omegas, iterations)
    plt.title("Iterations as a function of omega")
    plt.xlabel("Omega")
    plt.ylabel("Iterations")
    plt.show()

    # Return the value of omega that minimizes iterations
    least_iters = np.argmin(iterations)
    return omegas[least_iters]