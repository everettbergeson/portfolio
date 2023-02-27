# binary_trees.py
"""Volume 2: Binary Trees.
<Name>
<Class>
<Date>
"""

# These imports are used in BST.draw().
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout


class SinglyLinkedListNode:
    """A node with a value and a reference to the next node."""
    def __init__(self, data):
        self.value, self.next = data, None

class SinglyLinkedList:
    """A singly linked list with a head and a tail."""
    def __init__(self):
        self.head, self.tail = None, None

    def append(self, data):
        """Add a node containing the data to the end of the list."""
        n = SinglyLinkedListNode(data)
        if self.head is None:
            self.head, self.tail = n, n
        else:
            self.tail.next = n
            self.tail = n

    def iterative_find(self, data):
        """Search iteratively for a node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        current = self.head
        while current is not None:
            if current.value == data:
                return current
            current = current.next
        raise ValueError(str(data) + " is not in the list")

    # Problem 1
    def recursive_find(self, data):
        """Search recursively for the node containing the data.
        If there is no such node in the list, including if the list is empty,
        raise a ValueError.

        Returns:
            (SinglyLinkedListNode): the node containing the data.
        """
        def step(current):
            if current.value is None:
                raise ValueError(str(data) + " is not in the list")
            if current.value is data:
                return current
            else:
                return step(current.next)
        return step(self.head)

class BSTNode:
    """A node class for binary search trees. Contains a value, a
    reference to the parent node, and references to two child nodes.
    """
    def __init__(self, data):
        """Construct a new node and set the value attribute. The other
        attributes will be set when the node is added to a tree.
        """
        self.value = data
        self.prev = None        # A reference to this node's parent node.
        self.left = None        # self.left.value < self.value
        self.right = None       # self.value < self.right.value


class BST:
    """Binary search tree data structure class.
    The root attribute references the first node in the tree.
    """
    def __init__(self):
        """Initialize the root attribute."""
        self.root = None

    def find(self, data):
        """Return the node containing the data. If there is no such node
        in the tree, including if the tree is empty, raise a ValueError.
        """

        # Define a recursive function to traverse the tree.
        def _step(current):
            """Recursively step through the tree until the node containing
            the data is found. If there is no such node, raise a Value Error.
            """
            if current is None:                     # Base case 1: dead end.
                raise ValueError(str(data) + " is not in the tree.")
            if data == current.value:               # Base case 2: data found!
                return current
            if data < current.value:                # Recursively search left.
                return _step(current.left)
            else:                                   # Recursively search right.
                return _step(current.right)

        # Start the recursion on the root of the tree.
        return _step(self.root)

    # Problem 2
    def insert(self, data):
        """Insert a new node containing the specified data.

        Raises:
            ValueError: if the data is already in the tree.

        Example:
            >>> tree = BST()                    |
            >>> for i in [4, 3, 6, 5, 7, 8, 1]: |            (4)
            ...     tree.insert(i)              |            / \
            ...                                 |          (3) (6)
            >>> print(tree)                     |          /   / \
            [4]                                 |        (1) (5) (7)
            [3, 6]                              |                  \
            [1, 5, 7]                           |                  (8)
            [8]                                 |
        """
        def _stepinsert(current):
            if data == current.value:
                raise ValueError("Data is already in the tree")
            
            #if our new node is less than the last value
            if data < current.value:
                # check if we have something to the left
                if current.left is not None:
                    #if we do, try again from the new left value
                    _stepinsert(current.left)
                # otherwise, add it
                else:
                    current.left = newnode
                    current.left.prev = current

            if data > current.value:
                if current.right is not None:     
                    _stepinsert(current.right)
                else:
                    current.right = newnode
                    current.right.prev = current
            
        newnode = BSTNode(data)
        if self.root is None:
            self.root = newnode
        else:
            _stepinsert(self.root)


    # Problem 3
    def remove(self, data):
        """Remove the node containing the specified data.

        Raises:
            ValueError: if there is no node containing the data, including if
                the tree is empty.

        Examples:
            >>> print(12)                       | >>> print(t3)
            [6]                                 | [5]
            [4, 8]                              | [3, 6]
            [1, 5, 7, 10]                       | [1, 4, 7]
            [3, 9]                              | [8]
            >>> for x in [7, 10, 1, 4, 3]:      | >>> for x in [8, 6, 3, 5]:
            ...     t1.remove(x)                | ...     t3.remove(x)
            ...                                 | ...
            >>> print(t1)                       | >>> print(t3)
            [6]                                 | [4]
            [5, 8]                              | [1, 7]
            [9]                                 |
                                                | >>> print(t4)
            >>> print(t2)                       | [5]
            [2]                                 | >>> t4.remove(1)
            [1, 3]                              | ValueError: <message>
            >>> for x in [2, 1, 3]:             | >>> t4.remove(5)
            ...     t2.remove(x)                | >>> print(t4)
            ...                                 | []
            >>> print(t2)                       | >>> t4.remove(5)
            []                                  | ValueError: <message>
        """

        target = self.find(data)
        
        # Target is a leaf node
        if target.left is None and target.right is None:
            # Target is the root
            if target is self.root:
                self.root = None
            # Target is to the left of its parent
            
            elif target.prev.left is target:
                target.prev.left = None
            # Target is to the right of its parent
            elif target.prev.right is target:
                target.prev.right = None


        #DONE
        # Target has two children
        elif target.left is not None and target.right is not None:
            # if target is the root, set root as predecessor
            if target is self.root:
                #start looking for predecessor by going one to the left
                predecessor = target.left
                # and then going all the way to the right
                while predecessor.right is not None:
                    predecessor = predecessor.right
                # remove predecessor
                self.remove(predecessor.value)
                #connect new root
                self.root = predecessor
                predecessor.right = target.right
                predecessor.left = target.left
                target.right.prev = predecessor
                if target.left is not None:
                    target.left.prev = predecessor
                        
            # Target is not the root
            else:
                # target is to the left of parent
                if target is target.prev.left:
                    # find the predecessor
                    predecessor = target.left
                    while predecessor.right is not None:
                        predecessor = predecessor.right
                    #remove the predecessor
                    self.remove(predecessor.value)
                    target.prev.left = predecessor
                    predecessor.right = target.right
                    predecessor.left = target.left
                    target.right.prev = predecessor
                    if target.left is not None:
                        target.left.prev = predecessor
                    

                # target is to the right of parent
                if target is target.prev.right:
                    # find the predecessor
                    predecessor = target.left
                    while predecessor.right is not None:
                        predecessor = predecessor.right
                    #remove the predecessor
                    self.remove(predecessor.value)
                    target.prev.right = predecessor
                    predecessor.right = target.right
                    predecessor.left = target.left
                    target.right.prev = predecessor
                    if target.left is not None:
                        target.left.prev = predecessor

            


        # Target has one child
        elif target.left is not None or target.right is not None:
            # Target is the root
            if target is self.root:
                if target.left is not None:
                    self.root = target.left
                    target.left.prev = None
                elif target.right is not None:
                    self.root = target.right
                    target.right.prev = None

            # Target is to the left of its parent
            elif target is target.prev.left:
                # check if there's a child on the right
                if target.right is not None:
                    target.prev.left = target.right
                    target.right.prev = target.prev
                # otherwise check if there's a child on the left
                elif target.left is not None:
                    target.prev.left = target.left
                    target.left.prev = target.prev

            # Target is to the right of its parent
            elif target is target.prev.right:
                if target.right is not None:
                    target.prev.right = target.right
                    target.right.prev = target.prev
                elif target.left is not None:
                    target.prev.right = target.left
                    target.left.prev = target.prev



    def __str__(self):
        """String representation: a hierarchical view of the BST.

        Example:  (3)
                  / \     '[3]          The nodes of the BST are printed
                (2) (5)    [2, 5]       by depth levels. Edges and empty
                /   / \    [1, 4, 6]'   nodes are not printed.
              (1) (4) (6)
        """
        if self.root is None:                       # Empty tree
            return "[]"
        out, current_level = [], [self.root]        # Nonempty tree
        while current_level:
            next_level, values = [], []
            for node in current_level:
                values.append(node.value)
                for child in [node.left, node.right]:
                    if child is not None:
                        next_level.append(child)
            out.append(values)
            current_level = next_level
        return "\n".join([str(x) for x in out])

    def draw(self):
        """Use NetworkX and Matplotlib to visualize the tree."""
        if self.root is None:
            return

        # Build the directed graph.
        G = nx.DiGraph()
        G.add_node(self.root.value)
        nodes = [self.root]
        while nodes:
            current = nodes.pop(0)
            for child in [current.left, current.right]:
                if child is not None:
                    G.add_edge(current.value, child.value)
                    nodes.append(child)

        # Plot the graph. This requires graphviz_layout (pygraphviz).
        nx.draw(G, pos=graphviz_layout(G, prog="dot"), arrows=True,
                with_labels=True, node_color="C1", font_size=8)
        plt.show()


class AVL(BST):
    """Adelson-Velsky Landis binary search tree data structure class.
    Rebalances after insertion when needed.
    """
    def insert(self, data):
        """Insert a node containing the data into the tree, then rebalance."""
        BST.insert(self, data)      # Insert the data like usual.
        n = self.find(data)
        while n:                    # Rebalance from the bottom up.
            n = self._rebalance(n).prev

    def remove(*args, **kwargs):
        """Disable remove() to keep the tree in balance."""
        raise NotImplementedError("remove() is disabled for this class")

    def _rebalance(self,n):
        """Rebalance the subtree starting at the specified node."""
        balance = AVL._balance_factor(n)
        if balance == -2:                                   # Left heavy
            if AVL._height(n.left.left) > AVL._height(n.left.right):
                n = self._rotate_left_left(n)                   # Left Left
            else:
                n = self._rotate_left_right(n)                  # Left Right
        elif balance == 2:                                  # Right heavy
            if AVL._height(n.right.right) > AVL._height(n.right.left):
                n = self._rotate_right_right(n)                 # Right Right
            else:
                n = self._rotate_right_left(n)                  # Right Left
        return n

    @staticmethod
    def _height(current):
        """Calculate the height of a given node by descending recursively until
        there are no further child nodes. Return the number of children in the
        longest chain down.
                                    node | height
        Example:  (c)                  a | 0
                  / \                  b | 1
                (b) (f)                c | 3
                /   / \                d | 1
              (a) (d) (g)              e | 0
                    \                  f | 2
                    (e)                g | 0
        """
        if current is None:     # Base case: the end of a branch.
            return -1           # Otherwise, descend down both branches.
        return 1 + max(AVL._height(current.right), AVL._height(current.left))

    @staticmethod
    def _balance_factor(n):
        return AVL._height(n.right) - AVL._height(n.left)

    def _rotate_left_left(self, n):
        temp = n.left
        n.left = temp.right
        if temp.right:
            temp.right.prev = n
        temp.right = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_right_right(self, n):
        temp = n.right
        n.right = temp.left
        if temp.left:
            temp.left.prev = n
        temp.left = n
        temp.prev = n.prev
        n.prev = temp
        if temp.prev:
            if temp.prev.value > temp.value:
                temp.prev.left = temp
            else:
                temp.prev.right = temp
        if n is self.root:
            self.root = temp
        return temp

    def _rotate_left_right(self, n):
        temp1 = n.left
        temp2 = temp1.right
        temp1.right = temp2.left
        if temp2.left:
            temp2.left.prev = temp1
        temp2.prev = n
        temp2.left = temp1
        temp1.prev = temp2
        n.left = temp2
        return self._rotate_left_left(n)

    def _rotate_right_left(self, n):
        temp1 = n.right
        temp2 = temp1.left
        temp1.left = temp2.right
        if temp2.right:
            temp2.right.prev = temp1
        temp2.prev = n
        temp2.right = temp1
        temp1.prev = temp2
        n.right = temp2
        return self._rotate_right_right(n)

import time
import random
import numpy as np
from matplotlib import pyplot as plt
import math

# Problem 4
def prob4():
    """Compare the build and search times of the SinglyLinkedList, BST, and
    AVL classes. For search times, use SinglyLinkedList.iterative_find(),
    BST.find(), and AVL.find() to search for 5 random elements in each
    structure. Plot the number of elements in the structure versus the build
    and search times. Use log scales where appropriate.
    """
    n_list = []
    sll_build_times = []
    bst_build_times = []
    avl_build_times = []
    
    sll_find_times = []
    bst_find_times = []
    avl_find_times = []

    with open("english.txt", 'r') as myfile:
        contents = myfile.readlines()

    for i in range (3, 11):
        n_list.append(2**i)
    
    for n in n_list:
        randomList = random.choices(contents, k = n)
        sll = SinglyLinkedList()
        bst = BST()
        avl = AVL()

        start = time.time()
        for i in range (0, n):
            sll.append(randomList[i])
        
        time1 = time.time()
        for i in range (0, n):
            bst.insert(randomList[i])
        
        time2 = time.time()
        for i in range (0, n):
            avl.insert(randomList[i])

        time3 = time.time()
        randomitems = random.choices(randomList, k = 5)

        time4 = time.time()
        for item in randomitems:
            sll.iterative_find(item)
        
        time5 = time.time()
        for item in randomitems:
            bst.find(item)

        time6 = time.time()
        for item in randomitems:
            avl.find(item)
        
        time7 = time.time()

        sll_build_times.append(time1-start)
        bst_build_times.append(time2-time1)
        avl_build_times.append(time3-time2)
        
        sll_find_times.append(time5-time4)
        bst_find_times.append(time6-time5)
        avl_find_times.append(time7-time6)

    ax1 = plt.subplot(121)
    ax1.plot(n_list, sll_build_times, 'k.-')
    ax1.plot(n_list, bst_build_times, 'g.-')
    ax1.plot(n_list, avl_build_times, 'r.-')

    ax2 = plt.subplot(122)
    ax2.plt(n_list, sll_find_times, 'k.-')
    ax2.plt(n_list, bst_find_times, 'g.-')
    ax2.plt(n_list, avl_find_times, 'r.-')
    
    plt.show()

if __name__ == "__main__":
    prob4()
    