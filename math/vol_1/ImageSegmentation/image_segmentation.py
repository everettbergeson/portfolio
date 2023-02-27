# image_segmentation.py
"""Volume 1: Image Segmentation.
<Name>
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


# Problem 1
def laplacian(A):
    """Compute the Laplacian matrix of the graph G that has adjacency matrix A.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.

    Returns:
        L ((N,N) ndarray): The Laplacian matrix of G.
    """
    L = np.copy(A)
    D = []
    m, n = np.shape(A)

    for i in range (0, m):
        d = 0
        for j in range (0, n):
            d += A[i, j]
        D.append(d)
    
    for k in range (0, n):
        L[k, k] = L[k, k] - D[k]
    
    L = L * (-1)
    return L


# Problem 2
def connectivity(A, tol=1e-8):
    """Compute the number of connected components in the graph G and its
    algebraic connectivity, given the adjacency matrix A of G.

    Parameters:
        A ((N,N) ndarray): The adjacency matrix of an undirected graph G.
        tol (float): Eigenvalues that are less than this tolerance are
            considered zero.

    Returns:
        (int): The number of connected components in G.
        (float): the algebraic connectivity of G.
    """
    connectivity = 0
    L = laplacian(A)
    eigs = np.real(la.eig(L)[0])
    for e in eigs:
        if e <= tol:
            connectivity += 1
    sort = np.sort(eigs)
    
    return connectivity, sort[1]



# Helper function for problem 4.
def get_neighbors(index, radius, height, width):
    """Calculate the flattened indices of the pixels that are within the given
    distance of a central pixel, and their distances from the central pixel.

    Parameters:
        index (int): The index of a central pixel in a flattened image array
            with original shape (radius, height).
        radius (float): Radius of the neighborhood around the central pixel.
        height (int): The height of the original image in pixels.
        width (int): The width of the original image in pixels.

    Returns:
        (1-D ndarray): the indices of the pixels that are within the specified
            radius of the central pixel, with respect to the flattened image.
        (1-D ndarray): the euclidean distances from the neighborhood pixels to
            the central pixel.
    """
    # Calculate the original 2-D coordinates of the central pixel.
    row, col = index // width, index % width

    # Get a grid of possible candidates that are close to the central pixel.
    r = int(radius)
    x = np.arange(max(col - r, 0), min(col + r + 1, width))
    y = np.arange(max(row - r, 0), min(row + r + 1, height))
    X, Y = np.meshgrid(x, y)

    # Determine which candidates are within the given radius of the pixel.
    R = np.sqrt(((X - col)**2 + (Y - row)**2))
    mask = R < radius
    return (X[mask] + Y[mask]*width).astype(np.int), R[mask]


import scipy
from imageio import imread
from matplotlib import pyplot as plt
from scipy.sparse import linalg as sparsela

# Problems 3-6
class ImageSegmenter:
    """Class for storing and segmenting images."""

    # Problem 3
    def __init__(self, filename):
        """Read the image file. Store its brightness values as a flat array."""
        read_image = imread(filename) 
        self.image = read_image / 255
        if self.image.ndim == 3:
            self.brightness = self.image.mean(axis=2)
        else:
            self.brightness = self.image
        self.brightness = np.ravel(self.brightness)

    # Problem 3
    def show_original(self):
        """Display the original image."""
        if len(self.image.shape) == 3:
            plt.imshow(self.image)
        else:
            plt.imshow(self.image, cmap="gray")
        plt.show()
        

    # Problem 4
    def adjacency(self, r=5., sigma_B2=.02, sigma_X2=3.):
        """Compute the Adjacency and Degree matrices for the image graph."""

        size = len(self.brightness)
        m, n = len(self.image), len(self.image[0])

        A = scipy.sparse.lil_matrix((m*n, m*n))
        D = np.zeros(m*n)

        for i in range(m*n):
            indices, distances = get_neighbors(i, r, m, n)
            exp1 = (self.brightness[i] - self.brightness[indices])/sigma_B2
            exp2 = distances/sigma_X2
            weights = np.exp(-abs(exp1) - exp2)
            A[i, indices] = weights
            
        D = A.sum(axis = 0)
        A = scipy.sparse.csc_matrix(A)

        return A, np.array(D)
        

    # Problem 5
    def cut(self, A, D):
        """Compute the boolean mask that segments the image."""
        m, n = self.image.shape[0], self.image.shape[1]
        L = scipy.sparse.csgraph.laplacian(A)
        print(type(D))
        for i in range(0, len(D)):
            D[i] = 1 / np.sqrt(D[i])
        newD = scipy.sparse.diags(D[0])

        eigens = sparsela.eigsh(newD @ L @ newD, which='SM', k=2)[1][:,1]

        eigens = eigens.reshape(m, n)

        mask = eigens > 0

        return mask

    # Problem 6
    def segment(self, r=5., sigma_B=.02, sigma_X=3.):
        """Display the original image and its segments."""
        A, D = self.adjacency()

        pos_mask = self.cut(A, D)
        neg_mask = ~pos_mask

        if len(self.image.shape) == 3:
            pos_mask = np.dstack([pos_mask] * len(self.image[0,0]))
            neg_mask = np.dstack([neg_mask] * len(self.image[0,0]))

            ax1 = plt.subplot('131')
            ax1.imshow(self.image)

            ax2 = plt.subplot('132')
            ax2.imshow(self.image * pos_mask)

            ax3 = plt.subplot('133')
            ax3.imshow(self.image * neg_mask)
        
        else:
            ax1 = plt.subplot('131')
            ax1.imshow(self.image, cmap = 'gray')

            ax2 = plt.subplot('132')
            ax2.imshow(self.image * pos_mask, cmap = 'gray')

            ax3 = plt.subplot('133')
            ax3.imshow(self.image * neg_mask, cmap = 'gray')

        plt.show()



if __name__ == '__main__':
    
    pic = ImageSegmenter("everett.png")
    pic.segment()

#     ImageSegmenter("dream_gray.png").segment()
#     ImageSegmenter("dream.png").segment()
#     ImageSegmenter("monument_gray.png").segment()
#     ImageSegmenter("monument.png").segment()
