"""Volume 2: Simplex

<Name>
<Date>
<Class>
"""

import numpy as np


# Problems 1-6
class SimplexSolver(object):
    """Class for solving the standard linear optimization problem

                        minimize        c^Tx
                        subject to      Ax <= b
                                         x >= 0
    via the Simplex algorithm.
    """
    # Problem 1
    def __init__(self, c, A, b):
        """Check for feasibility and initialize the dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.

        Raises:
            ValueError: if the given system is infeasible at the origin.
        """
        # Check if Ax <= b when x = 0
        for i in b:
            if i < 0:
                raise ValueError("System is infeasible at the origin")
        self.c = c
        self.A = A
        self.b = b
        

    # Problem 2
    def _generatedictionary(self, c, A, b):
        """Generate the initial dictionary.

        Parameters:
            c ((n,) ndarray): The coefficients of the objective function.
            A ((m,n) ndarray): The constraint coefficients matrix.
            b ((m,) ndarray): The constraint vector.
        """
        m = np.array(A).shape[0]
        c_bar = np.append(c, np.zeros(m))
        A_bar = np.hstack((A, np.eye(m)))
        right_columns = np.vstack((c_bar, -1*A_bar))
        left_column = np.array([np.append([0], b)])
        self.D = np.hstack((left_column.T, right_columns))

    # Problem 3a
    def _pivot_col(self):
        """Return the column index of the next pivot column.
        """
        # return the first negative element in our row
        for j in range(1, self.D.shape[1]):
            if self.D[0,j] < 0:
                return j

        # unbounded 
        return 0

    # Problem 3b
    def _pivot_row(self, index):
        """Determine the row index of the next pivot row using the ratio test
        (Bland's Rule).
        """
        min_ratio = 1e50
        min_index = 0
        for i in range(1, self.D.shape[0]):
            if self.D[i,index] < 0:
                check_ratio = -1*self.D[i,0]/self.D[i,index]
                if check_ratio < min_ratio:
                    min_ratio = check_ratio
                    min_index = i
        return min_index

    # Problem 4
    def pivot(self):
        """Select the column and row to pivot on. Reduce the column to a
        negative elementary vector.
        """
        m = self.D.shape[0]

        # Select the column and row to pivot on
        col = self._pivot_col()
        row = self._pivot_row(col)
        # Check for unboundedness
        if min(self.D[:,col]) > 0:
            raise ValueError("Unbounded")
        
        # Divide the pivot row by the negative value of the pivot entry
        self.D[row,:] = self.D[row,:] / (-1*self.D[row,col])

        # Use the pivot row to zero out all entries in the pivot column 
        #   above and below the pivot entry
        for i in range(m):
            if i!=row:
                mult = self.D[i,col]/self.D[row,col]
                self.D[i,:] = self.D[i,:] - self.D[row,:]*mult

    # Problem 5
    def solve(self):
        """Solve the linear optimization problem.

        Returns:
            (float) The minimum value of the objective function.
            (dict): The basic variables and their values.
            (dict): The nonbasic variables and their values.
        """
        self._generatedictionary(self.c, self.A, self.b)
        solved = False
        while solved == False:
            if min(self.D[0,1:] >= 0):
                solved = True
                break
            else:
                self.pivot()
        indep_dict = {}
        dep_dict = {}

        for i in range(1, len(self.D[0])):
            if self.D[0][i] == 0:
                for j in range(1, len(self.D)):
                    if self.D[j][i] == -1:
                        indep_dict[i-1]=(self.D[j][0])
            else:
                dep_dict[i-1] = 0

        return (self.D[0,0], indep_dict, dep_dict)


# Problem 6
def prob6(filename='productMix.npz'):
    """Solve the product mix problem for the data in 'productMix.npz'.

    Parameters:
        filename (str): the path to the data file.

    Returns:
        ((n,) ndarray): the number of units that should be produced for each product.
    """
    