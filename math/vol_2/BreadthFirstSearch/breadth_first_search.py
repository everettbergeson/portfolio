# breadth_first_search.py
"""Volume 2: Breadth-First Search.
Everett Bergeson
<Class>
<Date>
"""
import collections

# Problems 1-3
class Graph:
    """A graph object, stored as an adjacency dictionary. Each node in the
    graph is a key in the dictionary. The value of each key is a set of
    the corresponding node's neighbors.

    Attributes:
        d (dict): the adjacency dictionary of the graph.
    """
    def __init__(self, adjacency={}):
        """Store the adjacency dictionary as a class attribute"""
        self.d = dict(adjacency)

    def __str__(self):
        """String representation: a view of the adjacency dictionary."""
        return str(self.d)



    # Problem 1
    def add_node(self, n):
        """Add n to the graph (with no initial edges) if it is not already
        present.

        Parameters:
            n: the label for the new node.
        """
        if n not in self.d.keys():
            self.d.update({n:set()})

    # Problem 1
    def add_edge(self, u, v):
        """Add an edge between node u and node v. Also add u and v to the graph
        if they are not already present.

        Parameters:
            u: a node label.
            v: a node label.
        """
        self.add_node(u)
        self.add_node(v)
        # Add u and v to the graph if they are not already present
        if v not in self.d[u] and u not in self.d[v]:
            self.d[u].add(v)
            self.d[v].add(u)

    # Problem 1
    def remove_node(self, n):
        """Remove n from the graph, including all edges adjacent to it.

        Parameters:
            n: the label for the node to remove.

        Raises:
            KeyError: if n is not in the graph.
        """
        # Check to see if n is in the graph
        if n in self.d.keys():
            # Remove each of the edges adjacent to it
            for i in self.d[n]:
                self.d[i].remove(n)
            # Remove n from the graph
            self.d.pop(n)
        else:
            raise KeyError("Node is not in the graph")


    # Problem 1
    def remove_edge(self, u, v):
        """Remove the edge between nodes u and v.

        Parameters:
            u: a node label.
            v: a node label.

        Raises:
            KeyError: if u or v are not in the graph, or if there is no
                edge between u and v.
        """
        # Check to see if the node is in the graph or if there is not edge
        # between u and v
        if u not in self.d.keys() or v not in self.d.keys():
            raise KeyError("Node is not in graph")
        if u not in self.d[v] or v not in self.d[u]:
            raise KeyError("No edge between the nodes")

        self.d[v].remove(u)
        self.d[u].remove(v)


    # Problem 2
    def traverse(self, source):
        """Traverse the graph with a breadth-first search until all nodes
        have been visited. Return the list of nodes in the order that they
        were visited.

        Parameters:
            source: the node to start the search at.

        Returns:
            (list): the nodes in order of visitation.

        Raises:
            KeyError: if the source node is not in the graph.
        """
        # Create a deque
        Q = collections.deque()
        #  Add the node we'll start the search at
        Q.append(source)
        # Create a set of nodes that have been marked to be visited
        M = set()
        M.add(source)
        # The nodes that have been visited in visitation order
        V = []

        
        while len(Q) != 0:
            # Pop a node off of Q, call it the current node
            current = Q.popleft()
            # Visit the current node by appending it to V
            V.append(current)
            # Add the neighbors of the current node that are not in M to Q and M
            for n in self.d[current]:
                if n not in M:
                    M.add(n)
                    Q.append(n)
        return V
        

    # Problem 3
    def shortest_path(self, source, target):
        """Begin a BFS at the source node and proceed until the target is
        found. Return a list containing the nodes in the shortest path from
        the source to the target, including endoints.

        Parameters:
            source: the node to start the search at.
            target: the node to search for.

        Returns:
            A list of nodes along the shortest path from source to target,
                including the endpoints.

        Raises:
            KeyError: if the source or target nodes are not in the graph.
        """
        if source not in self.d or target not in self.d:
            raise KeyError("One of the input nodes are not in the graph")

        # Initialize
        Q = collections.deque()
        Q.append(source)
        M = set()
        M.add(source)
        V = []
        D = dict()

        while len(Q) != 0:
            # Pop a node off of Q, call it the current node
            current = Q.popleft()
            # Visit the current node by appending it to V
            V.append(current)

            for n in self.d[current]:
                # If n has not been marked to visit, add it to the queue
                if n not in M:
                    M.add(n)
                    Q.append(n)
                    D[n] = current
                    
                    # If n is the target, add it to V
                    if n is target:
                        V = [target]
                        while n != source:
                            V.append(D[n])
                            n = D[n]
                        # Send it back in reverse order
                        V.reverse()
                        return V



import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
# Problems 4-6
class MovieGraph:
    """Class for solving the Kevin Bacon problem with movie data from IMDb."""

    # Problem 4
    def __init__(self, filename="movie_data.txt"):
        """Initialize a set for movie titles, a set for actor names, and an
        empty NetworkX Graph, and store them as attributes. Read the speficied
        file line by line, adding the title to the set of movies and the cast
        members to the set of actors. Add an edge to the graph between the
        movie and each cast member.

        Each line of the file represents one movie: the title is listed first,
        then the cast members, with entries separated by a '/' character.
        For example, the line for 'The Dark Knight (2008)' starts with

        The Dark Knight (2008)/Christian Bale/Heath Ledger/Aaron Eckhart/...

        Any '/' characters in movie titles have been replaced with the
        vertical pipe character | (for example, Frost|Nixon (2008)).
        """
        self.graph = nx.Graph()
        self.actors = set()
        self.movies = set()
        myfile = open(filename, 'r', encoding='utf-8', errors='replace')
        
        # Break apart the 
        for line in myfile:
            film = line.split('/')
            self.movies.add(film[0])
            self.actors.update(film[1:])
            for actorname in film[1:]:
                self.graph.add_edge(film[0], actorname)

            


    # Problem 5
    def path_to_actor(self, source, target):
        """Compute the shortest path from source to target and the degrees of
        separation between source and target.

        Returns:
            (list): a shortest path from source to target, including endpoints and movies.
            (int): the number of steps from source to target, excluding movies.
        """
        path = nx.shortest_path(self.graph, source, target)
        length = len(path) // 2
        return path, length


    # Problem 6
    def average_number(self, target):
        """Calculate the shortest path lengths of every actor to the target
        (not including movies). Plot the distribution of path lengths and
        return the average path length.

        Returns:
            (float): the average path length from actor to target.
        """
        # Calculate the lengths of the paths
        lengths = [length//2 for source, length in nx.single_target_shortest_path_length(self.graph, target) if source not in self.movies]

        # Create histogram
        plt.hist(lengths, bins=[i-.5 for i in range(8)])
        plt.xlabel("Degrees of separation")
        plt.ylabel("Frequency")
        plt.title("Kevin Bacon Number")
        plt.show()

        avg = sum(lengths) / len(lengths)

        return avg