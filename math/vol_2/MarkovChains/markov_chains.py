# markov_chains.py
"""Volume 2: Markov Chains.
Everett Bergeson
<Class>
<Date>
"""

import numpy as np
from scipy import linalg as la


class MarkovChain:
    """A Markov chain with finitely many states.

    Attributes:
        (fill this out)
    """
    # Problem 1
    def __init__(self, A, states=None):
        """Check that A is column stochastic and construct a dictionary
        mapping a state's label to its index (the row / column of A that the
        state corresponds to). Save the transition matrix, the list of state
        labels, and the label-to-index dictionary as attributes.

        Parameters:
        A ((n,n) ndarray): the column-stochastic transition matrix for a
            Markov chain with n states.
        states (list(str)): a list of n labels corresponding to the n states.
            If not provided, the labels are the indices 0, 1, ..., n-1.

        Raises:
            ValueError: if A is not square or is not column stochastic.

        Example:
            >>> MarkovChain(np.array([[.5, .8], [.5, .2]], states=["A", "B"])
        corresponds to the Markov Chain with transition matrix
                                   from A  from B
                            to A [   .5      .8   ]
                            to B [   .5      .2   ]
        and the label-to-index dictionary is {"A":0, "B":1}.
        """
        # Get the size of A
        m, n = np.shape(A)
        # I added this because sometimes the addition later on had tiny errors
        tol = .0000000000001
        # Test to see if A is square
        if m != n:
            raise ValueError("A is not square")
        # Test to see if the columns of A sum to 1
        for i in range(0, n):
            # If they don't, raise ValueError
            if abs(sum(A[:,i]) - 1) > tol:
                raise ValueError("A is not column stochastic")
        # Assign the transition matrix as an attribute
        self.transMatrix = A
        # Create an empty dictionary
        dictionary = {}
        # See if they provided a state list for labels
        if states is not None:
            self.labels = states
        # If they didn't, create a list of numbers 0 through (n-1)
        else:
            self.labels = list(range(n))
        # Create a dictionary with the labels and an index
        for i in range (0, len(self.labels)):
                dictionary[self.labels[i]] = i
        # Assign the dictionary as an attribute
        self.dictionary = dictionary


    # Problem 2
    def transition(self, state):
        """Transition to a new state by making a random draw from the outgoing
        probabilities of the state with the specified label.

        Parameters:
            state (str): the label for the current state.

        Returns:
            (str): the label of the state to transitioned to.
        """
        # Find which index our state is
        index = self.dictionary[state]
        # Make a random draw from the outgoing probabilities
        draw = np.random.multinomial(1, self.transMatrix[:,index])
        # Find which index the draw was
        drawIndex = np.argmax(draw)
        # Find which label corresponds to that index
        drawLabel = list(self.dictionary.keys())[drawIndex]
        
        return drawLabel

    # Problem 3
    def walk(self, start, N):
        """Starting at the specified state, use the transition() method to
        transition from state to state N-1 times, recording the state label at
        each step.

        Parameters:
            start (str): The starting state label.

        Returns:
            (list(str)): A list of N state labels, including start.
        """
        # Start with our first one
        nextup = start
        # Make a list with that as our first
        walklist = [nextup]
        # Keep going until we reach the desired number of iterations
        for i in range(1, N):
            # See where it will go next!
            nextup = self.transition(nextup)
            # Add it to the list
            walklist.append(nextup)
        return walklist

    # Problem 3
    def path(self, start, stop):
        """Beginning at the start state, transition from state to state until
        arriving at the stop state, recording the state label at each step.

        Parameters:
            start (str): The starting state label.
            stop (str): The stopping state label.

        Returns:
            (list(str)): A list of state labels from start to stop.
        """
        # Start with our first one
        nextup = start
        # Make a list with that as our first
        pathlist = [nextup]
        # Keep going until we reach the desired point
        while nextup != stop:
            # See where it will go next!
            nextup = self.transition(nextup)
            # Add it to the list
            pathlist.append(nextup)
        return pathlist


    # Problem 4
    def steady_state(self, tol=1e-12, maxiter=40):
        """Compute the steady state of the transition matrix A.

        Parameters:
            tol (float): The convergence tolerance.
            maxiter (int): The maximum number of iterations to compute.

        Returns:
            ((n,) ndarray): The steady state distribution vector of A.

        Raises:
            ValueError: if there is no convergence within maxiter iterations.
        """
        # Make a random vector of the correct size
        n = np.shape(self.transMatrix)[0]
        xCurrent = np.random.rand(n,1)
        # Normalize the vector
        xCurrent = xCurrent / sum(xCurrent)

        k = 0
        while k < maxiter:
            # Find x sub k + 1
            xNext = self.transMatrix @ xCurrent
            # Compare it to x sub k
            if np.linalg.norm(xCurrent - xNext) < tol:
                # If it's small enough return it
                return xCurrent
            # Otherwise keep iterating
            xCurrent = xNext
            k += 1
        # If it takes longer than maxiter times, it doesn't converge
        raise ValueError("A^k does not converge")

class SentenceGenerator(MarkovChain):
    """A Markov-based simulator for natural language.

    Attributes:
        (fill this out)
    """
    # Problem 5
    def __init__(self, filename):
        """Read the specified file and build a transition matrix from its
        contents. You may assume that the file has one complete sentence
        written on each line.
        """
        # Read the training set from the file filename
        myfile = open(filename, 'r', encoding='utf-8', errors='replace')

        # Get the set of unique words in the training set (state labels)
        uniqueWords = []
        # Go through each line
        for line in myfile:
            # Break up each line into sentences
            sentence = line.split()
            # Check each word in the sentence to see if it's in the list
            for word in sentence:
                if word not in uniqueWords:
                    uniqueWords.append(word)
                        
        # add labels $tart and $top to the set of states labels
        uniqueWords.insert(0, "$tart")
        uniqueWords.append("$top")
        # Initialize an appropriately sized square array of zeros to be the transition matrix
        length = len(uniqueWords)
        wordArray = np.zeros((length, length))

        # Re-initialize our file so we can work with each sentence again
        myfile = open(filename, 'r', encoding='utf-8', errors='replace')
        # for each sentence in the training set do:
        for line in myfile:
            # Split the sentence into a list of words
            sentence = line.split()
            # Prepend $tart and append $top to the list of words
            sentence.insert(0, "$tart")
            sentence.append("$top")

            # for each consecutive pair (x, y) of words in the list of words do:
            for i in range (0, len(sentence)-1):
                x = sentence[i]
                y = sentence[i + 1]
                xIndex = 0
                yIndex = 0
                for j in range (0, len(uniqueWords)):
                    # Find the index where x is in uniqueWords
                    if x == uniqueWords[j]:
                        xIndex = j
                    # Find the index where y is in uniqueWords
                    if y == uniqueWords[j]:
                        yIndex = j
                wordArray[yIndex, xIndex] += 1
                
        # Make sure the stop state transitions to itself
        wordArray[-1, -1] = 1
        # Normalize each column by dividing by the column sums
        for i in range(0, len(uniqueWords)):
            columnSum = sum(wordArray[:,i])
            wordArray[:,i] = wordArray[:,i] / columnSum
        MarkovChain.__init__(self, wordArray, uniqueWords)

    # Problem 6
    def babble(self):
        """Create a random sentence using MarkovChain.path().

        Returns:
            (str): A sentence generated with the transition matrix, not
                including the labels for the $tart and $top states.

        Example:
            >>> yoda = SentenceGenerator("yoda.txt")
            >>> print(yoda.babble())
            The dark side of loss is a path as one with you.
        """
        quote = self.path("$tart", "$top")
        quote.remove("$tart")
        quote.remove("$top")
        return ' '.join(quote)

if __name__=="__main__":
    weather2 = np.array([[.7, .6], [.3, .4]])
    weather4 = np.array([[.5, .3, .1, 0], [.3, .3, .3, .3], [.2, .3, .4, .5], [0, .1, .2, .2]])
    states = ["hot", "mild", "cold", "freezing"]
    mark = MarkovChain(weather2, states)
    mark.steady_state()
    
    yoda = SentenceGenerator("trump.txt")
    for i in range (0,40):
        tweet = yoda.babble()
        print(tweet)
        print()
