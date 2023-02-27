from object_oriented import ComplexNumber

def create(name, color):
    
    created = ComplexNumber(name, color) 
    return created

def test_ComplexNumber(a, b):
    py_cnum, my_cnum = complex(a, b), ComplexNumber(a, b)
    # Validate the constructor.
    if my_cnum.real != a or my_cnum.imag != b:
        print("__init__() set self.real and self.imag incorrectly")
    # Validate conjugate() by checking the new number's imag attribute.
    if py_cnum.conjugate().imag != my_cnum.conjugate().imag:
        print("conjugate() failed for", py_cnum)
    # Validate __str__().
    if str(py_cnum) != str(my_cnum):
        print("__str__() failed for", py_cnum)
        print(my_cnum)
        print(py_cnum)

if __name__ == "__main__":
    complex1 = create(3, -7)
    print(complex1)
    complex2 = create(2, 6)
    print(complex2)
    print(abs(complex1))
    print(abs(complex2))
    print(complex1 + complex2)
    print(complex1 - complex2)
    print(complex1 * complex2)
    print(complex1 / complex2)