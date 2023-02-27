# exceptions_fileIO.py
"""Python Essentials: Exceptions and File Input/Output.
<Name>
<Class>
<Date>
"""

from random import choice


# Problem 1
def arithmagic():
    """
    Takes in user input to perform a magic trick and prints the result.
    Verifies the user's input at each step and raises a
    ValueError with an informative error message if any of the following occur:
    
    The first number step_1 is not a 3-digit number.
    The first number's first and last digits differ by less than $2$.
    The second number step_2 is not the reverse of the first number.
    The third number step_3 is not the positive difference of the first two numbers.
    The fourth number step_4 is not the reverse of the third number.
    """
    
    step_1 = input("Enter a 3-digit number where the first and last "
                                           "digits differ by 2 or more: ")
    if len(step_1) != 3:
        raise ValueError("Not a 3 digit number")
    if abs(int(step_1[0]) - int(step_1[2])) < 2:
        raise ValueError("The first and last digits differ by less than 2.")
    
    step_2 = input("Enter the reverse of the first number, obtained "
                                              "by reading it backwards: ")
    for i in range(0, 3):
        if step_2[i] != step_1[2-i]:
            raise ValueError("That is not the reverse of the first number")
    
    step_3 = input("Enter the positive difference of these numbers: ")
    if int(step_3) != abs(int(step_2)-int(step_1)):
        raise ValueError("Not the positive difference of those numbers")
    
    step_4 = input("Enter the reverse of the previous result: ")
    for i in range(0, 3):
        if step_4[i] != step_3[2-i]:
            raise ValueError("That is not the reverse of the first number")
    
    print(str(step_3), "+", str(step_4), "= 1089 (ta-da!)")


# Problem 2
def random_walk(max_iters=1e12):
    """
    If the user raises a KeyboardInterrupt by pressing ctrl+c while the 
    program is running, the function should catch the exception and 
    print "Process interrupted at iteration $i$".
    If no KeyboardInterrupt is raised, print "Process completed".

    Return walk.
    """
    
    walk = 0
    directions = [1, -1]
    try:
        for i in range(int(max_iters)):
            walk += choice(directions)
    except KeyboardInterrupt:
        print("Process interrupted at iteration", i)
    else: 
        print("Process completed")
    finally:
        return walk


# Problems 3 and 4: Write a 'ContentFilter' class.
    """Class for reading in file
        
    Attributes:
        filename (str): The name of the file
        contents (str): the contents of the file
        
    """
class ContentFilter(object):   
    # Problem 3
    def __init__(self, filename):
        """Read from the specified file. If the filename is invalid, prompt
        the user until a valid filename is given.
        """
        while True:
            try:
                open(filename)
                break
            except:
                filename = input("Please enter a valid file name:")
        
        self.filename = filename
        myfile = open(filename, 'r')
        self.contents = myfile.read()
        myfile.close()

 # Problem 4 ---------------------------------------------------------------
    def check_mode(self, mode):
        """Raise a ValueError if the mode is invalid."""
        if mode == 'w' or 'x' or 'a':
            pass
        else:
            raise ValueError("Mode is invalid")
        
    def uniform(self, outfile, mode='w', case='upper'):
        """Write the data ot the outfile in uniform case."""
        self.check_mode(mode)

        # Open the file, check its case, output accordingly
        with open(outfile, mode) as out:
            if case == 'upper':
                out.write(self.contents.upper())
            elif case == 'lower':
                out.write(self.contents.lower())
            else:
                raise ValueError("Case is invalid")

    def reverse(self, outfile, mode='w', unit='word'):
        """Write the data to the outfile in reverse order."""
        self.check_mode(mode)
        with open(outfile, mode) as out:
            lines = []
            with open(self.filename, 'r') as myfile:
                lines = myfile.readlines()

            # Split by word
            if unit == 'word':
                for i in range(len(lines)):
                    lines[i] = ' '.join(lines[i].split()[::-1])
                lines[-1] += '\n'
                out.write('\n'.join(lines))
            # Split by line
            elif unit == 'line':
                out.write(''.join(lines[::-1]))
            else:
                raise ValueError("Unit is invalid")

    def transpose(self, outfile, mode='w'):
        """Write the transposed version of the data to the outfile."""
        self.check_mode(mode)
        with open(outfile, mode) as out:
            # Create a list of the split up words
            split_words = self.contents.split()
            # find how many rows so we can know how many words per row
            num_lines = self.contents.count('\n')
            new_str = ''
            # Iterate through the list, getting one word per row and moving up one
            interval = int(len(split_words)/num_lines)
            for i in range(interval):
                new_str += ' '.join(split_words[i::interval])
                new_str += '\n'
            out.write(new_str)
        

    def __str__(self):
        """String representation: info about the contents of the file."""
        out_string = ''
        
        sum_alpha = sum(a.isalpha() for a in self.contents)
        sum_num = sum(a.isdigit() for a in self.contents)
        sum_white = sum(a.isspace() for a in self.contents)
        sum_lines = self.contents.count('\n')
        sum_total = sum_alpha+sum_num+sum_white
        out_string += "Source file:\t\t\t" + self.filename + "\n"
        out_string += "Total characters:\t\t" + str(sum_total) + "\n"
        out_string += "Alphabetic characters:\t\t" + str(sum_alpha) + "\n"
        out_string += "Numerical characters:\t\t" + str(sum_num) + "\n"
        out_string += "Whitespace characters:\t\t" + str(sum_white) + "\n"
        out_string += "Number of lines:\t\t" + str(sum_lines) + "\n"
        return out_string

if __name__=="__main__":
    cf = ContentFilter("cf_example1.txt")
    print(cf)
    cf.uniform("uniform.txt", mode='w', case="upper")
    cf.uniform("uniform.txt", mode='a', case="lower")
    cf.reverse("reverse.txt", mode='w', unit="word")
    cf.reverse("reverse.txt", mode='a', unit="line")
    cf.transpose("transpose.txt", mode='w')
