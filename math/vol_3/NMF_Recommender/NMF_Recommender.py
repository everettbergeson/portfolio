"""Volume 2: Non-negative Matrix Factorization."""

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from imageio import imread
import warnings
warnings.filterwarnings("ignore")

from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error as mse


#Problems 1-2
class NMFRecommender:

    def __init__(self,random_state=15,rank=3,maxiter=200,tol=1e-3):
        """The parameter values for the algorithm"""
        self.random_state = random_state
        self.rank = rank
        self.maxiter = maxiter
        self.tol = tol
    
    def initialize_matrices(self, m, n):
        """randomly initialize the W and H matrices,"""
        np.random.seed(self.random_state)
        W = np.random.random((m, self.rank))
        H = np.random.random((self.rank, n))
        return W, H
      
    def compute_loss(self, V, W, H):
        """Computes Frobenius norm of V - WH"""
        return np.linalg.norm(V - W@H, ord='fro')
    
    def update_matrices(self, V, W, H):
        """The multiplicative update step to update W and H"""
        H = np.multiply(H, np.divide(W.T @ V, W.T @ W @ H))
        W = np.multiply(W, np.divide(V @ H.T, W @ H @ H.T))
        return W, H
    
    def fit(self, V):
        """Fits W and H weight matrices using CVXPY"""
        iters = 0
        tol = 1e10
        self.W, self.H = self.initialize_matrices(V.shape[0], V.shape[1])
        while iters < self.maxiter or tol < self.tol:
            tol = self.compute_loss(V, self.W, self.H)
            iters += 1
            self.W, self.H = self.update_matrices(V, self.W, self.H)

    def reconstruct(self):
        """Reconstruct V matrix for comparison against the original V"""
        return self.W @ self.H


def prob4():
    """Run NMF recommender on the grocery store example"""
    V = np.array([[0, 1, 0, 1, 2, 2],
                  [2, 3, 1, 1 ,2, 2],
                  [1, 1, 1, 0, 1, 1],
                  [0, 2, 3, 4, 1, 1],
                  [0, 0, 0, 0, 1, 0]])
    nmf = NMFRecommender()
    nmf.fit(V)
    
    return nmf.W, nmf.H, sum(nmf.H[1] > nmf.H[0])


def prob5():
    """
    Calculate the rank and run NMF
    """
    df = pd.read_csv('artist_user.csv').set_index("Unnamed: 0")
    
    # Compute benchmark
    benchmark = np.linalg.norm(df, ord='fro') * .0001

    r = 3
    model = NMF(n_components = r)
    W = model.fit_transform(df)
    H = model.components_
    error = np.sqrt(mse(df, W @ H))
    
    # Iteratively find best rank
    while True:
        if error < benchmark:
            break
        r += 1
        model = NMF(n_components = r)
        W = model.fit_transform(df)
        H = model.components_
        error = np.sqrt(mse(df, W @ H))
    return r, W @ H

def discover_weekly(id):
    """
    Create the recommended weekly 30 list for a given user
    """
    df = pd.read_csv('artist_user.csv').set_index("Unnamed: 0")
    _, V = prob5()
    
    # Get the index of that user to get the correct row in V
    ind = list(df.index.values).index(id)
    listened = df.loc[id]
    scores = V[ind]
    
    # Find which songs are recommended to them the most
    argsorted = list(np.argsort(-scores))
    artists = pd.read_csv('artists.csv')
    
    recs = []
    i = 0
    # Go through the top recommendations, leaving out the songs they've
    # already listened to, add the ones that they haven't heard
    while len(recs) < 30:
        artist_ind = argsorted.index(i)
        if listened.iloc[artist_ind] == 0:
            recs.append(artists.iloc[artist_ind]['name'])
        i += 1
    return recs
