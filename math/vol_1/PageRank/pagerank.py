# solutions.py
"""Volume 1: The Page Rank Algorithm.
<Name>
<Class>
<Date>
"""
import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations

# Problems 1-2
class DiGraph:
    """A class for representing directed graphs via their adjacency matrices.

    Attributes:
        (fill this out after completing DiGraph.__init__().)
    """
    # Problem 1
    def __init__(self, A, labels=None):
        """Modify A so that there are no sinks in the corresponding graph,
        then calculate Ahat. Save Ahat and the labels as attributes.

        Parameters:
            A ((n,n) ndarray): the adjacency matrix of a directed graph.
                A[i,j] is the weight of the edge from node j to node i.
            labels (list(str)): labels for the n nodes in the graph.
                If None, defaults to [0, 1, ..., n-1].
        """
        self.n = len(A)
        A = np.array(A, dtype="float")

        # See if there are any sinks
        for i in range(self.n):
            sum = 0
            for j in range(self.n):
                sum += abs(A[j,i])
            # If a node was a sink, replace all entries with 1's
            if sum == 0:
                for j in range(self.n):
                    A[j,i] = 1

        # Normalize columns
        col_sum = np.sum(A, axis=0)
        for k in range(self.n):
            A[:,k] = A[:,k]/col_sum[k]
        self.A = A

        # If there are no labels, create list of labels
        if labels is None:
            self.labels = [str(i) for i in range(self.n)]
        else:
            self.labels = labels


    # Problem 2
    def linsolve(self, epsilon=0.85):
        """Compute the PageRank vector using the linear system method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Returns:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Solve linear system
        left = np.identity(self.n) - (self.A*epsilon)
        right = (1-epsilon)/self.n * np.ones(self.n)
        p = np.linalg.solve(left, right)
        
        # Return dictionary with values
        return dict(zip(self.labels, list(p)))

    # Problem 2
    def eigensolve(self, epsilon=0.85):
        """Compute the PageRank vector using the eigenvalue method.
        Normalize the resulting eigenvector so its entries sum to 1.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Get first eigenvector of B
        B = epsilon*self.A + ((1-epsilon)/self.n)*np.ones_like(self.A)
        w,v = np.linalg.eig(B)
        p = v[:,0]
        # Normalize p
        p = p/np.sum(abs(p))
        
        return dict(zip(self.labels, list(p)))
        
        

    # Problem 2
    def itersolve(self, epsilon=0.85, maxiter=100, tol=1e-12):
        """Compute the PageRank vector using the iterative method.

        Parameters:
            epsilon (float): the damping factor, between 0 and 1.
            maxiter (int): the maximum number of iterations to compute.
            tol (float): the convergence tolerance.

        Return:
            dict(str -> float): A dictionary mapping labels to PageRank values.
        """
        # Initialize
        solved = False
        iters = 0
        diff = 1e50
        p0 = np.ones((self.n)) / self.n

        # Run (13.3) until we go over maxiter or diff is less than tol
        while solved == False:
            p1 = epsilon * (self.A @ p0) + ((1-epsilon)/self.n)*np.ones((self.n))
            diff = np.linalg.norm((p1 - p0), ord=1)
            iters += 1
            p0 = p1
            # Check if we should be done
            if iters >= maxiter or diff < tol:
                solved = True
        
        # Return dictionary with labels
        return dict(zip(self.labels, list(p0)))

# Problem 3
def get_ranks(d):
    """Construct a sorted list of labels based on the PageRank vector.

    Parameters:
        d (dict(str -> float)): a dictionary mapping labels to PageRank values.

    Returns:
        (list) the keys of d, sorted by PageRank value from greatest to least.
    """
    # Sort first
    d = dict(sorted(d.items()))
    return sorted(d, key=d.get, reverse=True)


# Problem 4
def rank_websites(filename="web_stanford.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i if webpage j has a hyperlink to webpage i. Use the DiGraph class
    and its itersolve() method to compute the PageRank values of the webpages,
    then rank them with get_ranks(). If two webpages have the same rank,
    resolve ties by listing the webpage with the larger ID number first.

    Each line of the file has the format
        a/b/c/d/e/f...
    meaning the webpage with ID 'a' has hyperlinks to the webpages with IDs
    'b', 'c', 'd', and so on.

    Parameters:
        filename (str): the file to read from.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of webpage IDs.
    """
    # Read the data
    myfile = open(filename, 'r', encoding='utf-8', errors='replace')
    
    # Get a list of the n unique page IDs in the file (the labels)
    labels = []
    for line in myfile:
        site = line.split('/')
        for link in site:
            link = link.rstrip()
            if link not in labels:
                labels.append(link)
    n = len(labels)

    # after constructing the list of webpage IDs, make a dictionary that maps
    # a webpage ID to its index in the list
    id_to_index = dict(zip(labels, [i for i in range(n)]))
    # Construct the n×n adjacency matrix
    A = np.zeros((n, n))
    myfile = open(filename, 'r', encoding='utf-8')

    for i, line in enumerate(myfile):
        site = line.split('/')
        for link in site[1:]:
            # Get rid of newline characters
            link = link.rstrip()
            link_index = id_to_index[link]
            A[link_index, i] = 1
    
    # Use DiGraph to compute the PageRank values of the webpages
    solution = DiGraph(A, labels)
    # then rank them with your function from Problem 3
    ranks = solution.itersolve(epsilon=0.85)
    
    return get_ranks(ranks)


# Problem 5
def rank_ncaa_teams(filename, epsilon=0.85):
    """Read the specified file and construct a graph where node j points to
    node i with weight w if team j was defeated by team i in w games. Use the
    DiGraph class and its itersolve() method to compute the PageRank values of
    the teams, then rank them with get_ranks().

    Each line of the file has the format
        A,B
    meaning team A defeated team B.

    Parameters:
        filename (str): the name of the data file to read.
        epsilon (float): the damping factor, between 0 and 1.

    Returns:
        (list(str)): The ranked list of team names.
    """
    teams = []
    games = pd.read_csv(filename).to_numpy()
    
    # Create list of unique teams
    for game in games:
        for team in game:
            if team not in teams:
                teams.append(team)

    # Create team dictionary
    n = len(teams)
    team_dict = dict(zip(teams, [i for i in range(n)]))

    # Create adjacency matrix
    A = np.zeros((n,n))
    for i, game in enumerate(games):
        win_ind = team_dict[game[0]]
        los_ind = team_dict[game[1]]
        A[win_ind, los_ind] += 1
    
    # Calculate page rank
    # Use DiGraph to compute the PageRank values of the webpages
    solution = DiGraph(A, teams)
    # then rank them with your function from Problem 3
    ranks = solution.itersolve(epsilon=0.85)

    return get_ranks(ranks)


# Problem 6
def rank_actors(filename="top250movies.txt", epsilon=0.85):
    """Read the specified file and construct a graph where node a points to
    node b with weight w if actor a and actor b were in w movies together but
    actor b was listed first. Use NetworkX to compute the PageRank values of
    the actors, then rank them with get_ranks().

    Each line of the file has the format
        title/actor1/actor2/actor3/...
    meaning actor2 and actor3 should each have an edge pointing to actor1,
    and actor3 should have an edge pointing to actor2.
    """
    myfile = open(filename, 'r', encoding='utf-8', errors='replace')
    DG = nx.DiGraph()
    myfile = [x.strip() for x in myfile.readlines()]

    # Look at each movie
    for movie in myfile:
        actors = movie.split('/')
        actors = actors[1:]
        combs = combinations(actors, 2)
        for comb in combs:
            if DG.has_edge(comb[1], comb[0]):
                DG[comb[1]][comb[0]]["weight"] += 1
            else:
                DG.add_edge(comb[1], comb[0], weight=1)

    ranking = nx.pagerank(DG, alpha=epsilon) 
    return get_ranks(ranking)