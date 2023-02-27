# cvxpy_intro.py
"""Volume 2: Intro to CVXPY.
<Name>
<Class>
<Date>
"""
import cvxpy as cp
import numpy as np

def prob1():
    """Solve the following convex optimization problem:

    minimize        2x + y + 3z
    subject to      x  + 2y         <= 3
                         y   - 4z   <= 1
                    2x + 10y + 3z   >= 12
                    x               >= 0
                          y         >= 0
                                z   >= 0

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function
    x = cp.Variable(3, nonneg = True)
    c = np.array([2, 1, 3])
    objective = cp.Minimize(c.T @ x)

    # Initialize constraints
    J = np.array([2, 1, 0])
    K = np.array([0, 1, -4])
    L = np.array([2, 10, 3])
    constraints = [J@x<=3, K@x<=1, L@x>=12]

    # Return optimizer and optimal value
    problem = cp.Problem(objective, constraints)
    opt = problem.solve()
    return x.value, opt

# Problem 2
def l1Min(A, b):
    """Calculate the solution to the optimization problem

        minimize    ||x||_1
        subject to  Ax = b

    Parameters:
        A ((m,n) ndarray)
        b ((m, ) ndarray)

    Returns:
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function and constraints
    m, n = A.shape
    x = cp.Variable(n)

    objective = cp.Minimize(cp.norm(x,1))
    constraints = [A@x == b]

    # Return optimizer and optimal value
    problem = cp.Problem(objective, constraints)
    opt = problem.solve()
    return x.value, opt


# Problem 3
def prob3():
    """Solve the transportation problem by converting the last equality constraint
    into inequality constraints.

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """

    # 6 possible ways to move pianos
    # Initialize objective function: minimize transportation costs
    x = cp.Variable(6, nonneg = True)
    c = np.array([4, 7, 6, 8, 8, 9])
    objective = cp.Minimize(c.T @ x)

    #7 pianos at 1, 2 pianos at 2, 4 pianos at 3
    J = np.array([1, 1, 0, 0, 0, 0])
    K = np.array([0, 0, 1, 1, 0, 0])
    L = np.array([0, 0, 0, 0, 1, 1])

    # Supply center 4 needs 5, 5 needs 8
    M = np.array([1, 0, 1, 0, 1, 0])
    N = np.array([0, 1, 0, 1, 0, 1])
    
    constraints = [J@x==7, K@x==2, L@x==4, M@x==5, N@x==8]
    problem = cp.Problem(objective, constraints)
    opt = problem.solve()
    return x.value, opt
            

# Problem 4
def prob4():
    """Find the minimizer and minimum of

    g(x,y,z) = (3/2)x^2 + 2xy + xz + 2y^2 + 2yz + (3/2)z^2 + 3x + z

    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Set up arrays
    Q = np.array([[3, 2, 1], [2, 4, 2], [1, 2, 3]])
    r = np.array([3, 0, 3])
    x = cp.Variable(3)

    # Solve problem
    prob = cp.Problem(cp.Minimize(.5 * cp.quad_form(x, Q) + r.T @ x))
    opt = prob.solve()
    return x.value, opt



# Problem 5
def prob5(A, b):
    """Calculate the solution to the optimization problem
        minimize    ||Ax - b||_2
        subject to  ||x||_1 == 1
                    x >= 0
    Parameters:
        A ((m,n), ndarray)
        b ((m,), ndarray)
        
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """
    # Initialize objective function and constraints
    m, n = A.shape
    x = cp.Variable(n, nonneg=True)

    objective = cp.Minimize(cp.norm(A@x - b, 2))
    constraints = [cp.sum(x)==1]

    # Return optimizer and optimal value
    problem = cp.Problem(objective, constraints)
    opt = problem.solve()
    return x.value, opt


# Problem 6
def prob6():
    """Solve the college student food problem. Read the data in the file 
    food.npy to create a convex optimization problem. The first column is 
    the price, second is the number of servings, and the rest contain
    nutritional information. Use cvxpy to find the minimizer and primal 
    objective.
    
    Returns (in order):
        The optimizer x (ndarray)
        The optimal value (float)
    """	 

    food = np.load("food.npy", allow_pickle=True)

    # Initialize objective function: minimize price of food
    x = cp.Variable(18, nonneg = True)
    objective = cp.Minimize(food[:,0] @ x)

    calo = food[:,2] * food[:,1]
    fat = food[:,3] * food[:,1]
    sug = food[:,4] * food[:,1]
    calc = food[:,5] * food[:,1]
    fib = food[:,6] * food[:,1]
    pro = food[:,7] * food[:,1]
    
    
    constraints = [calo@x<=2000, fat@x<=65, sug@x<=50, calc@x>=1000, fib@x>=25, pro@x>=46]

    problem = cp.Problem(objective, constraints)
    opt = problem.solve()
    return x.value, opt