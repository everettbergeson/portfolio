# matplotlib_intro.py
"""Python Essentials: Intro to Matplotlib.
<Name>
<Class>
<Date>
"""
import numpy as np
from matplotlib import pyplot as plt

# Problem 1
def var_of_means(n):
    """Construct a random matrix A with values drawn from the standard normal
    distribution. Calculate the mean value of each row, then calculate the
    variance of these means. Return the variance.

    Parameters:
        n (int): The number of rows and columns in the matrix A.

    Returns:
        (float) The variance of the means of each row.
    """
    
    array = np.random.normal(size=(n, n))
    means = np.mean(array, axis = 1)
    variance = np.var(means)
    return variance
    

def prob1():
    """Create an array of the results of var_of_means() with inputs
    n = 100, 200, ..., 1000. Plot and show the resulting array.
    """
    variance_list = []
    n_list = []
    
    for i in range (1, 11):
        variance_list.append(var_of_means(i * 100))
        n_list.append(i * 100)
    plt.plot(n_list, variance_list)
    plt.show()
    


# Problem 2
def prob2():
    """Plot the functions sin(x), cos(x), and arctan(x) on the domain
    [-2pi, 2pi]. Make sure the domain is refined enough to produce a figure
    with good resolution.
    """
    pi = np.pi
    x = np.linspace(-2*pi, 2*pi, 200)
    plt.plot(x, np.cos(x))
    plt.plot(x, np.sin(x))
    plt.plot(x, np.arctan(x))
    plt.show()


# Problem 3
def prob3():
    """Plot the curve f(x) = 1/(x-1) on the domain [-2,6].
        1. Split the domain so that the curve looks discontinuous.
        2. Plot both curves with a thick, dashed magenta line.
        3. Set the range of the x-axis to [-2,6] and the range of the
           y-axis to [-6,6].
    """
    x1 = np.linspace(-2, 1, 200)
    x2 = np.linspace(1, 6, 200)
    plt.plot(x1, 1/(x1-1), 'm--', lw=4)
    plt.plot(x2, 1/(x2-1), 'm--', lw=4)
    plt.xlim(-2, 6)
    plt.ylim(-6, 6)
    plt.show()


# Problem 4
def prob4():
    """Plot the functions sin(x), sin(2x), 2sin(x), and 2sin(2x) on the
    domain [0, 2pi].
        1. Arrange the plots in a square grid of four subplots.
        2. Set the limits of each subplot to [0, 2pi]x[-2, 2].
        3. Give each subplot an appropriate title.
        4. Give the overall figure a title.
        5. Use the following line colors and styles.
              sin(x): green solid line.
             sin(2x): red dashed line.
             2sin(x): blue dashed line.
            2sin(2x): magenta dotted line.
    """
    pi = np.pi
    x = np.linspace(0, 2*pi, 100)

    ax1 = plt.subplot(221)
    ax1.plot(x, np.sin(x), 'g-')
    ax1.set_title("sin(x)")


    ax2 = plt.subplot(222)
    ax2.plot(x, np.sin(2*x), 'r--')
    ax2.set_title("sin(2x)")


    ax3 = plt.subplot(223)
    ax3.plot(x, 2*(np.sin(x)), 'b--')
    ax3.set_title("2sin(x)")


    ax4 = plt.subplot(224)
    ax4.plot(x, 2*(np.sin(2*x)), 'm:')
    ax4.set_title("2sin(2x)")

    plt.suptitle("Variations of sin(x)")
    plt.axis([0, 2*pi, -2, 2])
    plt.show()


# Problem 5
def prob5():
    """Visualize the data in FARS.npy. Use np.load() to load the data, then
    create a single figure with two subplots:
        1. A scatter plot of longitudes against latitudes. Because of the
            large number of data points, use black pixel markers (use "k,"
            as the third argument to plt.plot()). Label both axes.
        2. A histogram of the hours of the day, with one bin per hour.
            Label and set the limits of the x-axis.
    """
    myarray = np.load('FARS.npy')
    x = myarray[:,0]
    y = myarray[:,1]
    z = myarray[:,2]
    
    ax1 = plt.subplot(121)
    ax1.plot(y, z, 'k,')
    ax1.set_xlabel("Longitude")
    ax1.set_ylabel("Latitude")
    ax1.set_aspect("equal")

    ax2 = plt.subplot(122)
    ax2.hist(x, bins=np.arange(1, 25))

    plt.show()

# Problem 6
def prob6():
    """Plot the function f(x,y) = sin(x)sin(y)/xy on the domain
    [-2pi, 2pi]x[-2pi, 2pi].
        1. Create 2 subplots: one with a heat map of f, and one with a contour
            map of f. Choose an appropriate number of level curves, or specify
            the curves yourself.
        2. Set the limits of each subplot to [-2pi, 2pi]x[-2pi, 2pi].
        3. Choose a non-default color scheme.
        4. Add a colorbar to each subplot.
    """
    pi = np.pi
    x = np.linspace(-2*pi, 2*pi, 200)
    y = x.copy()
    x, y = np.meshgrid(x, y)
    z = (np.sin(x) * np.sin(y)) / (x*y)

    ax1 = plt.subplot(121)
    ax1.pcolormesh(x, y, z, cmap="bone")
    ax1.set_xlim(-2*pi, 2*pi)
    ax1.set_ylim(-2*pi, 2*pi)
    plt.imshow(z, cmap="bone")
    plt.colorbar(shrink=.5)
    

    ax2 = plt.subplot(122)
    ax2.contourf(x, y, z, 10, cmap="coolwarm")
    plt.imshow(z, cmap="coolwarm")
    plt.colorbar(shrink=.5)
    ax2.set_xlim(-2*pi, 2*pi)
    ax2.set_ylim(-2*pi, 2*pi)
    plt.show()


if __name__=="__main__":
    prob6()