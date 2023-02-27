# solutions.py
"""Volume 1: The SVD and Image Compression. Solutions File.
Everett Bergeson
"""

import numpy as np
from scipy import linalg as la
from matplotlib import pyplot as plt

# Problem 1
def compact_svd(A, tol=1e-6):
    """Compute the truncated SVD of A.

    Parameters:
        A ((m,n) ndarray): The matrix (of rank r) to factor.
        tol (float): The tolerance for excluding singular values.

    Returns:
        ((m,r) ndarray): The orthonormal matrix U in the SVD.
        ((r,) ndarray): The singular values of A as a 1-D array.
        ((r,n) ndarray): The orthonormal matrix V^H in the SVD.
    """
    # Create A hermitian
    Ah = A.conj().T
    # Get eigenvalues and eigenvectors of Ah @ A
    lam, V = la.eig(Ah @ A)

    # Get and sort singular values from greatest to least
    sig = np.sqrt(lam)
    indices = np.flip(np.argsort(sig))
    sig = sig[indices]
    # Sort the eigenvectors in the same way
    V = V[:,indices]
    # Count the number of nonzero singular values
    r = 0
    for s in sig:
        if s > tol:
            r += 1
    # Get all the corresponding non-zero parts of sig, V, U
    sig1 = np.real(sig[:r])
    V1 = V[:,:r]
    U1 = np.real(A @ V1 / sig1)
    return U1, sig1, V1.conj().T

# Problem 2
def visualize_svd(A):
    """Plot the effect of the SVD of A as a sequence of linear transformations
    on the unit circle and the two standard basis vectors.
    """
    # Create 200 points around unit circle
    pi = np.pi
    unit_circle = np.linspace(0, 2*pi, 200)
    S = np.zeros((2, 200))
    # Fill S (2 x 200) with x and y coordinates of unit circle
    for u in range(0, 200):
        S[0, u] = np.cos(unit_circle[u])
        S[1, u] = np.sin(unit_circle[u])
    # Create E
    E = np.array([[1, 0, 0], [0, 0, 1]])

    # Find SVD of A
    U, s, Vh = la.svd(A)
    s = np.diag(s)

    # Plot x against y
    ax1 = plt.subplot(221)
    ax1.plot(S[0], S[1])
    ax1.plot(E[0], E[1])

    # Plot VhS, VhE
    ax2 = plt.subplot(222)
    S1 = Vh @ S
    E1 = Vh @ E
    ax2.plot(S1[0], S1[1])
    ax2.plot(E1[0], E1[1])

    # Plot sigVhS, sigVhE
    ax3 = plt.subplot(223)
    S2 = s @ Vh @ S
    E2 = s @ Vh @ E
    ax3.plot(S2[0], S2[1])
    ax3.plot(E2[0], E2[1])

    # Plot UsigVhS, UsigVhE
    ax4 = plt.subplot(224)
    S3 = U @ s @ Vh @ S
    E3 = U @ s @ Vh @ E
    ax4.plot(S3[0], S3[1])
    ax4.plot(E3[0], E3[1])

    plt.show()


# Problem 3
def svd_approx(A, s):
    """Return the best rank s approximation to A with respect to the 2-norm
    and the Frobenius norm, along with the number of bytes needed to store
    the approximation via the truncated SVD.

    Parameters:
        A ((m,n), ndarray)
        s (int): The rank of the desired approximation.

    Returns:
        ((m,n), ndarray) The best rank s approximation of A.
        (int) The number of entries needed to store the truncated SVD.
    """
    m, n = A.shape
    # Compute SVD
    U, sig, Vh = la.svd(A)

    # check to see that s is not greater than the number of
    #nonzero singular values of A
    r = 0
    for singular in sig:
        if singular > 1e-6:
            r += 1
    if s > r:
        raise ValueError("s is greater than rank(A)")

    # Truncate the SVD
    U = U[:, :s]
    sig = sig[:s]
    Vh = Vh[:s, :]

    # Truncated only requires saving a total of mr + r + nr values
    valueSize = s * (m + n + 1)

    # Create As, the best rank s approximation of A
    As = U @ np.diag(sig) @ Vh

    return As, valueSize

# Problem 4
def lowest_rank_approx(A, err):
    """Return the lowest rank approximation of A with error less than 'err'
    with respect to the matrix 2-norm, along with the number of bytes needed
    to store the approximation via the truncated SVD.

    Parameters:
        A ((m, n) ndarray)
        err (float): Desired maximum error.

    Returns:
        A_s ((m,n) ndarray) The lowest rank approximation of A satisfying
            ||A - A_s||_2 < err.
        (int) The number of entries needed to store the truncated SVD.
    """
    m, n = A.shape
    U, sig, Vh = la.svd(A)
    # Choose s such that sigma(s+1) is the largest singular value
    # less than epsilon, then compute As
    indices = np.where(sig > err)
    
    # Check to see if we can approximate A
    if err <= sig[-1]:
        raise ValueError("A cannot be approximated within the tolerance by a matrix of lesser rank")
    
    s = len(indices)

    # Truncate the SVD
    U = U[:, :s]
    sig = sig[:s]
    Vh = Vh[:s, :]

    # Truncated only requires saving a total of mr + r + nr values
    valueSize = s * (m + n + 1)

    # Create As, the best rank s approximation of A
    As = U @ np.diag(sig) @ Vh

    return As, valueSize


from imageio import imread

# Problem 5
def compress_image(filename, s):
    """Plot the original image found at 'filename' and the rank s approximation
    of the image found at 'filename.' State in the figure title the difference
    in the number of entries used to store the original image and the
    approximation.

    Parameters:
        filename (str): Image file path.
        s (int): Rank of new image.
    """
    # Send the RGB values to the interval (0, 1)
    image = imread(filename) / 255

    # Initialize a value for the size of the low rank approximations
    imagevalue = 0
    # Store the size of the original image
    originalsize = image.size

    # Decide if image is B&W or color
    if len(image.shape) == 3:
        # Plot the original image
        ax2 = plt.subplot(122)
        ax2.imshow(image)

        # If color, let R, G, and B be matrices
        # Calculate the low rank approximations for them separately
        Rs, Rvalue = svd_approx(image[:,:,0], s)
        Gs, Gvalue = svd_approx(image[:,:,1], s)
        Bs, Bvalue = svd_approx(image[:,:,2], s)
        # Put them back together in a new 3-dimensional array
        image = np.dstack([Rs, Gs, Bs])
        image = np.clip(image, 0, 1)

        # Add up the sizes of the separate low rank approxs
        imagevalue = Rvalue + Gvalue + Bvalue

        ax1 = plt.subplot(121)
        ax1.imshow(image)

    else:
        # Plot the original image in gray
        ax2 = plt.subplot(122)
        ax2.imshow(image, cmap = 'gray')

        # Grayscale images can be approximated directly
        image, imagevalue = svd_approx(image, s)
        image = np.clip(image, 0, 1)
        ax1 = plt.subplot(121)
        ax1.imshow(image, cmap = 'gray')

    # Calculate difference, turn off axis ticks and labels, display image
    title_string = "Difference: " + str(originalsize - imagevalue)
    plt.suptitle(title_string)
    ax1.axis("off") 
    ax2.axis("off") 
    plt.show()

if __name__=="__main__":
    compress_image("hubble.jpg", 120)