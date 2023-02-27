# standard_library.py
"""Python Essentials: The Standard Library.
Everett Bergeson
<Class>
<Date>
"""
import sys


# Problem 1
def prob1(L):
    """Return the minimum, maximum, and average of the entries of L
    (in that order).
    """
    return(min(L), max(L), (sum(L)/len(L)))


# Problem 2
def prob2():
    """Determine which Python objects are mutable and which are immutable.
    Test numbers, strings, lists, tuples, and sets. Print your results.
    """
    int_mut = 1
    str_mut  = "hello"
    list_mut = [1, 2, 3]
    tuple_mut = ()
    set_mut = {1, 2}


    test_int = int_mut
    test_int += 1
    if test_int == int_mut:
        print("int is mutable")
    else:
        print("int is immutable")
    
    test_str = str_mut
    test_str += "z"
    if test_str == str_mut:
        print("str is mutable")
    else:
        print("str is immutable")

    test_list = list_mut
    test_list[0] = 1234
    if test_list == list_mut:
        print("list is mutable")
    else:
        print("list is immutable")
    
    test_tuple = tuple_mut
    test_tuple += (1,)
    if test_tuple == tuple_mut:
        print("tuple is mutable")
    else:
        print("tuple is immutable")

    test_set = set_mut
    test_set.add("amaaazing")
    if test_set == set_mut:
        print("set is mutable")
    else:
        print("set is immutable")


# Problem 3
def hypot(a, b):
    """Calculate and return the length of the hypotenuse of a right triangle.
    Do not use any functions other than those that are imported from your
    'calculator' module.

    Parameters:
        a: the length one of the sides of the triangle.
        b: the length the other non-hypotenuse side of the triangle.
    Returns:
        The length of the triangle's hypotenuse.
    """

    import calculator as calc
    a_square = calc.product(a, a)
    b_square = calc.product(b, b)
    c_square = calc.sum(a_square, b_square)
    c = calc.math.sqrt(c_square)

    return c


# Problem 4
def power_set(A):
    """Use itertools to compute the power set of A.

    Parameters:
        A (iterable): a str, list, set, tuple, or other iterable collection.

    Returns:
        (list(sets)): The power set of A as a list of sets.
    """
    power_set = [] 
    from itertools import combinations
    for i in range(len(A) + 1):
        subset = list(combinations(A, i))
        for j in range(len(subset)):
            power_set.append(set(subset[j]))
    return power_set



# Problem 5: Implement shut the box.
def shut_the_box(player, timelimit):
    """Play a single game of shut the box."""

    import box
    import time
    start = time.time()
    time_elapsed = round((time.time() - start), 2)

    #create number list
    numbers_left = list(range(1, 10))

    valid_input = True

    # All of this in a while numbers are left loop
    
    while len(numbers_left) > 0:
        time_elapsed = round((time.time() - start), 2)
        time_left = timelimit - time_elapsed
        if(time_left < 0):
            break
        # If the last input was good, roll the dice. Otherwise we'll skip this step.
        if(valid_input == True):
            # Calculate 2 random rolls of 1-6 each, sum the two
            roll = box.roll_dice(numbers_left)
            # Test to see if a roll is valid (check if there exists a combination of numbers that sum to the roll)
            valid_roll = box.isvalid(roll, numbers_left)


        print("\nNumbers Left: ", numbers_left)
        print("Roll: ", roll)
        

        # If the roll is valid, do this
        if (valid_roll):
            
            print("Seconds left:", time_left)
            # Prompt user to input numbers they want to eliminate
            to_elim = input("Numbers to eliminate: ")
            parsed = box.parse_input(to_elim, numbers_left)
            if (sum(parsed) == roll):
                valid_input = True
                for i in range(len(parsed)):
                    numbers_left.remove(parsed[i])
            else:
                valid_input = False
                print("Invalid input")
        else:
            break

    print("Game over!")
    print ("\nScore for player", player, ":", sum(numbers_left), "points")
    print("Time played:", time_elapsed, "seconds")
    print("You kinda suck bye")
    

    
                    
           





if __name__ == "__main__":
    
    a = [1, 2, 3, 4, 5]
    print(prob1(a))
    prob2()
    print(hypot(3, 4))
    
    power_set({1, 2, 3})
    
    shut_the_box(str(sys.argv[1]), int(sys.argv[2]))