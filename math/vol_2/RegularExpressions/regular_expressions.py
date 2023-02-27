# regular_expressions.py
"""Volume 3: Regular Expressions.
Everett Bergeson
<Class>
<Date>
"""
import re
# Problem 1
def prob1():
    """Compile and return a regular expression pattern object with the
    pattern string "python".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile("python")

# Problem 2
def prob2():
    """Compile and return a regular expression pattern object that matches
    the string "^{@}(?)[%]{.}(*)[_]{&}$".

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"\^\{@\}\(\?\)\[%\]\{\.\}\(\*\)\[_\]\{&\}\$")

# Problem 3
def prob3():
    """Compile and return a regular expression pattern object that matches
    the following strings (and no other strings).

        Book store          Mattress store          Grocery store
        Book supplier       Mattress supplier       Grocery supplier

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^(Book|Mattress|Grocery) (store|supplier)$")

# Problem 4
def prob4():
    """Compile and return a regular expression pattern object that matches
    any valid Python identifier.

    Returns:
        (_sre.SRE_Pattern): a compiled regular expression pattern object.
    """
    return re.compile(r"^[a-zA-Z_][\w]*[\s]*(=[\s]*[\d]*(\.[\d]*)?|=[\s]*'[^']*'|=\s*[a-zA-Z_][\w]*)?$")


# Problem 5
def prob5(code):
    """Use regular expressions to place colons in the appropriate spots of the
    input string, representing Python code. You may assume that every possible
    colon is missing in the input string.

    Parameters:
        code (str): a string of Python code without any colons.

    Returns:
        (str): code, but with the colons inserted in the right places.
    """
    lines = code.splitlines()
    return_list = []
    # Find something with one of those expressions
    # then replace the newline character with a colon and a newline
    for line in lines:
        # Add a colon after the expression
        add_at_end = re.compile("^ *(if|elif|for|while|try|with|def|class|else|finally|except)")
        if bool(add_at_end.search(line)):
            return_list.append(line + ':')   # Append a colon to the end of the line
        else:
            return_list.append(line)
    return "\n".join(return_list)
    

# Problem 6
def prob6(filename="fake_contacts.txt"):
    """Use regular expressions to parse the data in the given file and format
    it uniformly, writing birthdays as mm/dd/yyyy and phone numbers as
    (xxx)xxx-xxxx. Construct a dictionary where the key is the name of an
    individual and the value is another dictionary containing their
    information. Each of these inner dictionaries should have the keys
    "birthday", "email", and "phone". In the case of missing data, map the key
    to None.

    Returns:
        (dict): a dictionary mapping names to a dictionary of personal info.
    """
    myfile = open(filename, 'r')
    contact_dictionary = {}

    # some have (), others don't
    phone_finder = re.compile(r"[\(\d]{3,4}[\)-]*\d{3}-\d{4}")

    # Match 1 or 2 numbers, slash, 1 or 2 numbers, slash, 2 to 4 numbers
    birthday_finder = re.compile(r"\d{1,2}\/\d{1,2}\/\d{2,4}")

    # has @ in it, preceded and followed by any alphanumeric or period 
    email_finder = re.compile(r"[\.\w]*\@[\.\w]*")

    # Jane C. Doe, middle name optional
    name_finder = re.compile(r"^[a-zA-Z]* [A-Z]?\.? ?[a-zA-Z]*")

    for contact in myfile:
        # Get their name
        full_name = name_finder.findall(contact)
        full_name = full_name[0]

        # Check to see if they have a birthday
        birthday = birthday_finder.findall(contact)
        if len(birthday) == 0:
            birthday = None
        else:
            birthday = birthday[0]

        # Check to see if they have an email
        email = email_finder.findall(contact)
        if len(email) == 0:
            email = None
        else:
            email = email[0]

        phone_number = phone_finder.findall(contact)
        if len(phone_number) == 0:
            phone_number = None
        else:
            phone_number = phone_number[0]
        
        # Edit birthday
        if birthday is not None:
            split_bday = birthday.split("/")
            if len(split_bday[0]) == 1:
                split_bday[0] = "0" + split_bday[0]
            if len(split_bday[1]) == 1:
                split_bday[1] = "0" + split_bday[1]
            if len(split_bday[2]) != 4:
                split_bday[2] = "20" + split_bday[2]
            birthday = "/".join(split_bday)
        
        # Edit phone number
        if phone_number is not None:
            raw_phone = phone_number.replace("(", "")
            raw_phone = raw_phone.replace(")", "")
            raw_phone = raw_phone.replace("-", "")
            phone_number = "(" + raw_phone[0:3] + ")" + raw_phone[3:6] + "-" + raw_phone[6:]

        # Add each item to the inner dictionary
        info_dict = {
            "birthday": birthday,
            "email": email,
            "phone": phone_number
        }

        # Make the name the key and the value the inner dictionary
        contact_dictionary[full_name] = info_dict
    return contact_dictionary