# object_oriented.py
"""Python Essentials: Object Oriented Programming.
<Name>
<Class>
<Date>
"""


class Backpack:
    """A Backpack object class. Has a name and a list of contents.

    Attributes:
        name (str): the name of the backpack's owner.
        contents (list): the contents of the backpack.
    """

    # Problem 1: Modify __init__() and put(), and write dump().
    def __init__(self, name, color, max_size = 5):
        """Set the name and initialize an empty list of contents.

        Parameters:
            name (str): the name of the backpack's owner.
            contents (list): the list of contents
            color (str): the color of the backpack
            max_size (int): the number of contents allowed
        """
        self.name = name
        self.contents = []
        self.color = color
        self.max_size = max_size


    def put(self, item):
        """Add an item to the backpack's list of contents."""
        if len(self.contents) == self.max_size:
            print("No room!")
        else:
            self.contents.append(item)

    def take(self, item):
        """Remove an item from the backpack's list of contents."""
        self.contents.remove(item)

    def dump(self):
        self.contents = []

    # Magic Methods -----------------------------------------------------------

    # Problem 3: Write __eq__() and __str__().
    def __add__(self, other):
        """Add the number of contents of each Backpack."""
        return len(self.contents) + len(other.contents)

    def __lt__(self, other):
        """Compare two backpacks. If 'self' has fewer contents
        than 'other', return True. Otherwise, return False.
        """
        return len(self.contents) < len(other.contents)
    
    def __eq__(self, other):
        same = False
        if self.name == other.name:
            if self.color == other.color:
                if len(self.contents) == len(other.contents):
                    same = True
        return same
    
    def __str__(self):
        pack_string = "Owner:\t\t" + self.name
        pack_string += "\nColor:\t\t" + self.color
        pack_string += "\nSize:\t\t" + str(len(self.contents))
        pack_string += "\nMax Size:\t" + str(self.max_size)
        pack_string += "\nContents:\t" + str(self.contents)
        return pack_string

# An example of inheritance. You are not required to modify this class.
class Knapsack(Backpack):
    """A Knapsack object class. Inherits from the Backpack class.
    A knapsack is smaller than a backpack and can be tied closed.

    Attributes:
        name (str): the name of the knapsack's owner.
        color (str): the color of the knapsack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the knapsack is tied shut.
    """
    def __init__(self, name, color):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A knapsack only holds 3 item by default.

        Parameters:
            name (str): the name of the knapsack's owner.
            color (str): the color of the knapsack.
            max_size (int): the maximum number of items that can fit inside.
        """
        Backpack.__init__(self, name, color, max_size=3)
        self.closed = True

    def put(self, item):
        """If the knapsack is untied, use the Backpack.put() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.put(self, item)

    def take(self, item):
        """If the knapsack is untied, use the Backpack.take() method."""
        if self.closed:
            print("I'm closed!")
        else:
            Backpack.take(self, item)

    def weight(self):
        """Calculate the weight of the knapsack by counting the length of the
        string representations of each item in the contents list.
        """
        return sum(len(str(item)) for item in self.contents)


# Problem 2: Write a 'Jetpack' class that inherits from the 'Backpack' class.
class Jetpack(Backpack):
    """A jetpack object class. Inherits from the Backpack class.

    Attributes:
        name (str): the name of the jetpack's owner.
        color (str): the color of the jetpack.
        max_size (int): the maximum number of items that can fit inside.
        contents (list): the contents of the backpack.
        closed (bool): whether or not the jetpack is tied shut.
    """
    def __init__(self, name, color, max_size=2, fuel=10):
        """Use the Backpack constructor to initialize the name, color,
        and max_size attributes. A jetpack only holds 2 item by default.

        Parameters:
            name (str): the name of the jetpack's owner.
            color (str): the color of the jetpack.
            max_size (int): the maximum number of items that can fit inside.
            fuel (int): amount of fuel, default is 10
        """
        Backpack.__init__(self, name, color, max_size)
        self.fuel = fuel

    def fly(self, fuel_burned):
        """Accepts an amount of fuel to be burned and decrements the
            fuel attribute by that amount. If the user tries to burn 
            more fuel than remains, print "Not enough fuel!" and do 
            not decrement the fuel.
        """
        if (fuel_burned >= self.fuel):
            print("Not enough fuel!")
        else:
            self.fuel -= fuel_burned
        

    def dump(self):
        """Empty both the contents and the fuel tank.
        """
        self.fuel = 0
        Backpack.dump(self)



# Problem 4: Write a 'ComplexNumber' class.
class ComplexNumber:
    def __init__(self, a, b):
        """a + bi, a = real coefficient, b = imaginary coefficient
        """
        self.real = a
        self.imag = b

    def __str__(self):
        complex_string = str(self.real)
        if self.imag >= 0:
            complex_string += "+" + str(self.imag) + "i"
        else:
            complex_string += str(self.imag) + "i"
        return complex_string
    
    def __abs__(self):
        import math
        return math.sqrt((self.real * self.real) + (self.imag * self.imag))
    
    def __eq__(self, other):
        same = False
        if self.real == other.real:
            if self.imag == other.imag:
                same = True
        return same

    def __add__(self, other):
        return ComplexNumber((self.real + other.real), (self.imag + other.imag))

    def __sub__(self, other):
        return ComplexNumber((self.real - other.real), (self.imag - other.imag))

    def __mul__(self, other):
        f = self.real * other.real
        o = self.real * other.imag
        i = self.imag * other.real
        l = self.imag * other.imag
        real = f + ((-1) * l)
        imag = o + i
        return ComplexNumber(real, imag)

    def __truediv__(self, other):
        other_conj = other.conjugate()
        top = self.__mul__(other_conj)
        bottom = other * other_conj
        return ComplexNumber((top.real / bottom.real), (top.imag / bottom.real))

    def conjugate(self):
        conj_imag = self.imag * (-1)
        conjugate = ComplexNumber(self.real, conj_imag)
        return conjugate
