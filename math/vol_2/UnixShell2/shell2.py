# shell2.py
"""Volume 3: Unix Shell 2.
Everett Bergeson
<Date>
"""

import os
from glob import glob
import subprocess
# Problem 3
def grep(target_string, file_pattern):
    """Find all files in the current directory or its subdirectories that
    match the file pattern, then determine which ones contain the target
    string.

    Parameters:
        target_string (str): A string to search for in the files whose names
            match the file_pattern.
        file_pattern (str): Specifies which files to search.
    """
    file_append = []

    # Create a append of each of the files with the file pattern ending
    #directory = 'Shell2/*/*' + file_pattern
    directory = '**/' + file_pattern
    filename_append = glob(directory, recursive = True)

    # Search through each of the files in the 
    for myfile in filename_append:
        with open(myfile) as openFile:
            contents = openFile.read()
            if contents.find(target_string) == -1:
                pass
            else:
                file_append.append(myfile)

    return file_append



# Problem 4
def largest_files(n):
    """Return a append of the n largest files in the current directory or its
    subdirectories (from largest to smallest).
    """
    # Search the current directory and all subdirectories
    # for the n largest files
    # Get all the files
    myFiles = glob('**', recursive = True)
    fileSizeTups = []
    # Get the file sizes
    for filed in myFiles:
        if os.path.isfile(filed):
            size = os.path.getsize(filed)
            fileSizeTups.append((filed, size))

    # Sort our list
    fileSizeTups = sorted(fileSizeTups, key = lambda x:x[1], reverse = True) 
    # Keep the n largest files
    trimList = fileSizeTups[:n]

    # Write the line count of the smallest file 
    # to a file called smallest.txt

    smallest = trimList[-1][0]
    """
    smallestFile = fileSizeTups[-1]
    size = len(open(smallestFile[0]).readlines())
    os.system("echo " + str(size) + " > smallest.txt")
    """
    args = ["cat " + smallest + " | wc -l"]
    task = subprocess.check_output(args, shell=True).decode().strip()

    with open("smallest.txt", 'w') as filename:
        filename.write(task)
    # Return the list of filenames including the file path
    # in order from largest to smallest.
    return trimList
    
    
# Problem 6    
def prob6(n = 10):
   """this problem counts to or from n three different ways, and
      returns the resulting appends each integer
   
   Parameters:
       n (int): the integer to count to and down from
   Returns:
       integerCounter (append): append of integers from 0 to the number n
       twoCounter (append): append of integers created by counting down from n by two
       threeCounter (append): append of integers created by counting up to n by 3
   """
   #print what the program is doing
   integerCounter = list()
   twoCounter = list()
   threeCounter = list()
   counter = n
   for i in range(n+1):
       integerCounter.append(i)
       if (i % 2 == 0):
           twoCounter.append(counter - i)
       if (i % 3 == 0):
           threeCounter.append(i)
   #return relevant values
   return integerCounter, twoCounter, threeCounter

if __name__ == "__main__":
    print(grep(" ", "*.py"))