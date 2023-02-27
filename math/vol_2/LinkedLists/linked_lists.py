# linked_lists.py
"""Volume 2: Linked Lists.
<Name>
<Class>
<Date>
"""


# Problem 1
class Node:
    """A basic node class for storing data."""
    def __init__(self, data):
        """Store the data in the value attribute.
                
        Raises:
            TypeError: if data is not of type int, float, or str.
        """
        if isinstance(data, int) or isinstance(data, str) or isinstance(data, float):
            self.value = data
        else:
            raise TypeError("Data is not of type int, float, or str")
        
        


class LinkedListNode(Node):
    """A node class for doubly linked lists. Inherits from the Node class.
    Contains references to the next and previous nodes in the linked list.
    """
    def __init__(self, data):
        """Store the data in the value attribute and initialize
        attributes for the next and previous nodes in the list.
        """
        Node.__init__(self, data)       # Use inheritance to set self.value.
        self.next = None                # Reference to the next node.
        self.prev = None                # Reference to the previous node.
        self.index = 0

# Problems 2-5
class LinkedList:
    """Doubly linked list data structure class.

    Attributes:
        head (LinkedListNode): the first node in the list.
        tail (LinkedListNode): the last node in the list.
    """
    def __init__(self):
        """Initialize the head and tail attributes by setting
        them to None, since the list is empty initially.
        """
        self.head = None
        self.tail = None
        

    def append(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.index = 0
            
        else:
            # If the list is not empty, place new_node after the tail.
            self.tail.next = new_node               # tail --> new_node
            new_node.prev = self.tail               # tail <-- new_node
            new_node.index = new_node.prev.index + 1
            # Now the last node in the list is new_node, so reassign the tail.
            self.tail = new_node

    # Problem 2
    def find(self, data):
        """Return the first node in the list containing the data.

        Raises:
            ValueError: if the list does not contain the data.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.find('b')
            >>> node.value
            'b'
            >>> l.find('f')
            ValueError: <message>
        """
        # Start at the first value
        # go until the end
        # if the value of each node is equal to the data 
        # return that node
        # if it doesn't return anything, return ValueError
        if(self.head is None):
            raise ValueError("List is empty")
        start = self.head
        while start is not None:
            if start.value is data:
                return start
            else:
                start = start.next
        raise ValueError("List does not contain the data")
        

    # Problem 2
    def get(self, i):
        """Return the i-th node in the list.

        Raises:
            IndexError: if i is negative or greater than or equal to the
                current number of nodes.

        Examples:
            >>> l = LinkedList()
            >>> for x in ['a', 'b', 'c', 'd', 'e']:
            ...     l.append(x)
            ...
            >>> node = l.get(3)
            >>> node.value
            'd'
            >>> l.get(5)
            IndexError: <message>
        """
        start = self.head
        if i < 0 or i >= self.tail.index:
            raise IndexError("i is negative or greater than or equal to the current number of nodes")
        while start is not None:
            if start.index == i:
                return start
            else: 
                start = start.next

    # Problem 3
    def __len__(self):
        """Return the number of nodes in the list.

        Examples:
            >>> l = LinkedList()
            >>> for i in (1, 3, 5):
            ...     l.append(i)
            ...
            >>> len(l)
            3
            >>> l.append(7)
            >>> len(l)
            4
        """
        return self.tail.index + 1

    # Problem 3
    def __str__(self):
        """String representation: the same as a standard Python list.

        Examples:
            >>> l1 = LinkedList()       |   >>> l2 = LinkedList()
            >>> for i in [1,3,5]:       |   >>> for i in ['a','b',"c"]:
            ...     l1.append(i)        |   ...     l2.append(i)
            ...                         |   ...
            >>> print(l1)               |   >>> print(l2)
            [1, 3, 5]                   |   ['a', 'b', 'c']
        """
        my_string = "["
        start = self.head
        while start is not None:
            my_string += repr(start.value)
            if start.next is not None:
                my_string += ", "
            
            start = start.next
        my_string += "]"
        return my_string

    # Problem 4
    def remove(self, data):
        """Remove the first node in the list containing the data.

        Raises:
            ValueError: if the list is empty or does not contain the data.

        Examples:
            >>> print(l1)               |   >>> print(l2)
            ['a', 'e', 'i', 'o', 'u']   |   [2, 4, 6, 8]
            >>> l1.remove('i')          |   >>> l2.remove(10)
            >>> l1.remove('a')          |   ValueError: <message>
            >>> l1.remove('u')          |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.remove(10)
            ['e', 'o']                  |   ValueError: <message>
        """
        
        removenode = self.find(data)
        if removenode.next is None:
            if removenode.prev is None:
                self.tail = None
                self.head = None
            else:
                self.tail = removenode.prev
                removenode.prev.next = None
        elif removenode.prev is None:
            self.head = removenode.next
            removenode.next.prev = None
        else:
            # We're at B, we want A's next to be C
            removenode.prev.next = removenode.next
            # We're at B, we want C's previous to be A
            removenode.next.prev = removenode.prev 
        if self.head is not None:
            self.head.index = 0
            start = self.head
            start = start.next
            while start is not None:
                start.index = start.prev.index + 1
                start = start.next


    # Problem 5
    def insert(self, index, data):
        """Insert a node containing data into the list immediately before the
        node at the index-th location.

        Raises:
            IndexError: if index is negative or strictly greater than the
                current number of nodes.

        Examples:
            >>> print(l1)               |   >>> len(l2)
            ['b']                       |   5
            >>> l1.insert(0, 'a')       |   >>> l2.insert(6, 'z')
            >>> print(l1)               |   IndexError: <message>
            ['a', 'b']                  |
            >>> l1.insert(2, 'd')       |   >>> l3 = LinkedList()
            >>> print(l1)               |   >>> l3.insert(1, 'a')
            ['a', 'b', 'd']             |   IndexError: <message>
            >>> l1.insert(2, 'c')       |
            >>> print(l1)               |
            ['a', 'b', 'c', 'd']        |
        """
        if index == self.tail.index:
            self.append(data)
        elif index > self.tail.index or index < 0:
            raise IndexError("Greater than or negative index...")
        else:
            new_node = LinkedListNode(data)
            if index == 0:
                oldnode = self.head
                oldnode.prev = new_node
                oldnode.next = self.head.next
                new_node.prev = None
                new_node.next = oldnode
                new_node.index = 0
                self.head = new_node
            else:
                oldindex = self.get(index)
                #just inserted C between B and D
                # B's next is C
                oldindex.prev.next = new_node
                # C's previous is B
                new_node.prev = oldindex.prev
                # C's next is D
                new_node.next = oldindex
                # D's previous is C
                oldindex.prev = new_node
                
        start = self.head
        start = start.next
        while start is not None:
            start.index = start.prev.index + 1
            start = start.next


# Problem 6: Deque class.
class Deque(LinkedList):
    def __init__(self):
        """ Inherit everything from LinkedList"""
        LinkedList.__init__(self)

    def pop(self):
        """ Remove and return the last entry
            Reassign new tail
            If empty, raise ValueError """

        if self.head is None and self.tail is None:
            raise ValueError("List is empty")
        else:
            last = self.tail
            if last.prev is None:
                self.tail = None
                self.head = None
            else:
                last.prev.next = None
                self.tail = last.prev
            return last
        

    def popleft(self):
        if self.head is None and self.tail is None:
            raise ValueError("List is empty")
        else:
            first = self.head
            if first.next is None:
                self.tail = None
                self.head = None
            else:
                first.next.prev = None
                self.head = first.next
            return first
        if self.head is not None:
            self.head.index = 0
            start = self.head
            start = start.next
            while start is not None:
                start.index = start.prev.index + 1
                start = start.next
    
    def appendleft(self, data):
        """Append a new node containing the data to the end of the list."""
        # Create a new node to store the input data.
        new_node = LinkedListNode(data)
        if self.head is None:
            # If the list is empty, assign the head and tail attributes to
            # new_node, since it becomes the first and last node in the list.
            self.head = new_node
            self.tail = new_node
            self.index = 0
            
        else:
            # If the list is not empty, place new_node after the tail.
            self.head.prev = new_node               # tail --> new_node
            new_node.next = self.head               # tail <-- new_node
            new_node.index = 0
            # Now the last node in the list is new_node, so reassign the tail.
            self.head = new_node
        
        if self.head is not None:
            self.head.index = 0
            start = self.head
            start = start.next
            while start is not None:
                start.index = start.prev.index + 1
                start = start.next
        
    
    def remove(*args, **kwargs):
        raise NotImplementedError("Use pop() or popleft() for removal")
    
    def insert(*args, **kwargs):
        raise NotImplementedError("Use append() or appendleft() for inserting")


# Problem 7
def prob7(infile, outfile):
    """Reverse the contents of a file by line and write the results to
    another file.

    Parameters:
        infile (str): the file to read from.
        outfile (str): the file to write to.
    """
    toread = open(infile)
    try:
        contents = toread.readlines()
    finally:
        toread.close()
    writestring = contents[::-1]
    with open(outfile, 'w') as writeout:
        writeout.writelines(writestring)
