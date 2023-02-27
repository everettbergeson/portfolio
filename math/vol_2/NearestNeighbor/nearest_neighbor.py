# nearest_neighbor.py
"""Volume 2: Nearest Neighbor Search.
Everett Bergeson
<Class>
<Date>
"""

import numpy as np
import scipy.stats as st
from scipy import linalg as la
from scipy.spatial import KDTree



# Problem 1
def exhaustive_search(X, z):
    """Solve the nearest neighbor search problem with an exhaustive search.

    Parameters:
        X ((m,k) ndarray): a training set of m k-dimensional points.
        z ((k, ) ndarray): a k-dimensional target point.

    Returns:
        ((k,) ndarray) the element (row) of X that is nearest to z.
        (float) The Euclidean distance from the nearest neighbor to z.
    """
    normedX = la.norm(X-z, axis=1)
    x = np.argmin(normedX)
    d = normedX[x]
    return x, d


# Problem 2: Write a KDTNode class.
class KDTNode:
    """Node class for K-D Trees.

    Attributes:
        left (KDTNode): a reference to this node's left child.
        right (KDTNode): a reference to this node's right child.
        value ((k,) ndarray): a coordinate in k-dimensional space.
        pivot (int): the dimension of the value to make comparisons on.
    """
    def __init__(self, x):
        if isinstance(x, np.ndarray):
            pass
        else:
            raise TypeError('Must be np.ndarray')
        self.value = x
        self.left = None
        self.right = None
        self.pivot = None

# Problems 3 and 4
class KDT:
    """A k-dimensional binary tree for solving the nearest neighbor problem.

    Attributes:
        root (KDTNode): the root node of the tree. Like all other nodes in
            the tree, the root has a NumPy array of shape (k,) as its value.
        k (int): the dimension of the data in the tree.
    """
    def __init__(self):
        """Initialize the root and k attributes."""
        self.root = None
        self.k = None

    def find(self, data):
        """Return the node containing the data. If there is no such node in
        the tree, or if the tree is empty, raise a ValueError.
        """
        def _step(current):
            """Recursively step through the tree until finding the node
            containing the data. If there is no such node, raise a ValueError.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree")
            elif np.allclose(data, current.value):
                return current                      # Base case 2: data found!
            elif data[current.pivot] < current.value[current.pivot]:
                return _step(current.left)          # Recursively search left.
            else:
                return _step(current.right)         # Recursively search right.

        # Start the recursive search at the root of the tree.
        return _step(self.root)

    # Problem 3
    def insert(self, data):
        """Insert a new node containing the specified data.

        Parameters:
            data ((k,) ndarray): a k-dimensional point to insert into the tree.

        Raises:
            ValueError: if data does not have the same dimensions as other
                values in the tree.
            ValueError: if data is already in the tree
        """
        # If tree is empty, create new KDTNode, set pivot to 0
        #   Assign root attribute to new node
        #   Set k attribute as the length of x
        if self.root is None:
            new_root = KDTNode(data)
            new_root.pivot = 0
            self.root = new_root
            self.k = len(new_root.value)
        
        else:
        #   Raise ValueError if data to be inserted is not in Rk
            if len(data) != self.k:
                raise ValueError("Data to be inserted is not in Rk")
        # If tree is non-empty, create a new KDTNode containing x
            new_node = KDTNode(data)
            def _step(compare):
                """Find the existing node that should become its parent
                Determine whether new node will be its parent's left or right
                Link the parent to the new node accordingly
                Set pivot of new node based in its parent's pivot"""
                # Do not allow duplicates in the tree, raise ValueError
                if np.allclose(data, compare.value):
                    raise ValueError("Data is already in the tree")

                piv = compare.pivot
                
                # Check if it should be on the left
                if compare.value[piv] > new_node.value[piv]:
                    if compare.left is None:
                        new_node.pivot = (piv + 1)%self.k
                        compare.left = new_node
                    else:
                        _step(compare.left)

                # Check if it should be on the right
                else:
                    if compare.right is None:
                        new_node.pivot = (piv + 1)%self.k
                        compare.right = new_node
                    else:
                        _step(compare.right)
            _step(self.root)

        


    # Problem 4
    def query(self, z):
        """Find the value in the tree that is nearest to z.

        Parameters:
            z ((k,) ndarray): a k-dimensional target point.

        Returns:
            ((k,) ndarray) the value in the tree that is nearest to z.
            (float) The Euclidean distance from the nearest neighbor to z.
        """
        def KDSearch(current, nearest, d):
            # Base case: dead end
            if current is None:
                return nearest, d
            x = current.value
            i = current.pivot

            # Check if the current is closer to z than nearest
            if la.norm(x-z) < d:
                nearest = current
                d = la.norm(x-z)
            # Search to the left
            if z[i] < x[i]:
                nearest, d = KDSearch(current.left, nearest, d)
                # Search to the right if needed
                if z[i] + d >= x[i]:
                    nearest, d = KDSearch(current.right, nearest, d)
            # Search to the right
            else:
                nearest, d = KDSearch(current.right, nearest, d)
                # Search to the left if needed
                if z[i] - d <= x[i]:
                    nearest, d = KDSearch(current.left, nearest, d)
            return nearest, d

        node, d = KDSearch(self.root, self.root, la.norm(self.root.value-z))
        return node.value, d

    def __str__(self):
        """String representation: a hierarchical list of nodes and their axes.

        Example:                           'KDT(k=2)
                    [5,5]                   [5 5]   pivot = 0
                    /   \                   [3 2]   pivot = 1
                [3,2]   [8,4]               [8 4]   pivot = 1
                    \       \               [2 6]   pivot = 0
                    [2,6]   [7,5]           [7 5]   pivot = 0'
        """
        if self.root is None:
            return "Empty KDT"
        nodes, strs = [self.root], []
        while nodes:
            current = nodes.pop(0)
            strs.append("{}\tpivot = {}".format(current.value, current.pivot))
            for child in [current.left, current.right]:
                if child:
                    nodes.append(child)
        return "KDT(k={})\n".format(self.k) + "\n".join(strs)


# Problem 5: Write a KNeighborsClassifier class.
class KNeighborsClassifier:
    """A k-nearest neighbors classifier that uses SciPy's KDTree to solve
    the nearest neighbor problem efficiently.
    """
    def __init__(self, n_neighbors):
        """
        Parameters: 
            n_neighbors (int): the number of neighbors to include
                in the vote (the k in k-nearest neighbors)
        """
        self.k = n_neighbors
    
    def fit(self, X, y):
        """
        Load a SciPy KDTree with the data in X. Save the tree and labels as
        attributes.
        Parameters:
            X ((m,k) ndarray): training set
            y ((m,) ndarray): training labels
        """
        self.tree = KDTree(X)
        self.labels = y

    def predict(self, z):
        """
        Query the DKTree for the n_neighbors elements of X that are nearest to z
        and return the most common label of those neighbors. If there is a tie
        for the most common label, choose the alphanumerically smallest label.
        Parameters:
            z ((k,) ndarray): 
        Returns:
            mode (str): The most common label of those neighbors
        """
        distances, indices = self.tree.query(z, k=self.k)
        return st.mode(self.labels[indices])[0][0]


# Problem 6
def prob6(n_neighbors, filename="mnist_subset.npz"):
    """Extract the data from the given file. Load a KNeighborsClassifier with
    the training data and the corresponding labels. Use the classifier to
    predict labels for the test data. Return the classification accuracy, the
    percentage of predictions that match the test labels.

    Parameters:
        n_neighbors (int): the number of neighbors to use for classification.
        filename (str): the name of the data file. Should be an npz file with
            keys 'X_train', 'y_train', 'X_test', and 'y_test'.

    Returns:
        (float): the classification accuracy.
    """
    data = np.load("mnist_subset.npz")
    X_test = data["X_test"].astype(np.float) # Test data
    y_test = data["y_test"] 
    X_train = data["X_train"].astype(np.float) # Training data
    y_train = data["y_train"] # Training labels

    classifier = KNeighborsClassifier(n_neighbors)
    classifier.fit(X_train, y_train)
    labs = 0
    
    xlen = len(X_test)
    for i in range(xlen):
        if classifier.predict(X_test[i]) == y_test[i]:
            labs += 1
    return labs/xlen
    