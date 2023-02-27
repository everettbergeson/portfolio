# python_intro.py
"""Python Essentials: Introduction to Python.
Everett Bergeson
Math 321 Section 2
9/3/2020
"""

def sphere_volume(r):
    """Accept radius, output volume"""
    v = (4 / 3) * 3.14159 * (r**3)
    return v

def isolate(a, b, c, d, e):
    """Print the first three separated by 5
       spaces, then print the rest with a 
       single space between each output"""
    print(a, b, c, sep='     ', end=' ')
    print(d, e)

def first_half(word):
    """Take in a word, return the first half"""
    x = len(word)
    x = x // 2
    return word[:x]

def backward(word1):
    """Reverse the order of a string's characters"""
    word1= word1[::-1]
    return word1

def list_ops():
    """"Do some list ops"""
    my_list = ["bear", "ant", "cat", "dog"]
    my_list.append("eagle")
    my_list[2] = "fox"
    my_list.pop(1)
    my_list = sorted(my_list, reverse=True)
    x = my_list.index("eagle")
    my_list[x] = "hawk"
    my_list[-1] += "hunter"
    return my_list

def pig_latin(word):
    """translate into pig latin"""
    if word[0] in ('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U'):
        vowel = True
    else:
        vowel = False
    if vowel:
        word += "hay"
    else:
        word = word[1:] + word[0] + "ay"
    return word


def palindrome():
    """Find the largest possible palindrome
       of the product of 2 3-digit numbers"""
    max_palindrome = 0
    max_i = 0
    max_j = 0
    i = 1
    # Starting at 1 going to 100 for both i and j
    while i < 1000:
        j = 1
        while j < 1000:
            # convert i * j into string
            test = str(i * j)
            # see if something is the same front to back      
            if test == test[::-1]:
                # compare it to previous max
                if int(test) > max_palindrome:
                    max_palindrome = int(test)
                    max_i = i
                    max_j = j
            j+=1
        i+=1
    return max_palindrome, max_i, max_j
    
def alt_harmonic(n):
    """Use a list comprehension to compute
       the first n terms of the alternating
       harmonic series"""
    sequence = [((-1)**(i+2)) / (i+1) for i in range(n)]
    return sum(sequence)

if __name__ == "__main__":
    print("Hello, world!")
    isolate(1, 2, 3, 4, 5)
    print(sphere_volume(13))
    print(first_half("12345"))
    print(backward("EverettBergeson"))
    print(list_ops())
    print(pig_latin("yeehaw"))
    palindrome()
    alt_harmonic(50000)

