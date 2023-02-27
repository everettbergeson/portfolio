# test_specs.py
"""Python Essentials: Unit Testing.
Everett Bergeson
<Class>
<Date>
"""

import specs
import pytest


def test_add():
    assert specs.add(1, 3) == 4, "failed on positive integers"
    assert specs.add(-5, -7) == -12, "failed on negative integers"
    assert specs.add(-6, 14) == 8

def test_divide():
    assert specs.divide(4,2) == 2, "integer division"
    assert specs.divide(5,4) == 1.25, "float division"
    with pytest.raises(ZeroDivisionError) as excinfo:
        specs.divide(4, 0)
    assert excinfo.value.args[0] == "second input cannot be zero"


# Problem 1: write a unit test for specs.smallest_factor(), then correct it.
def test_smallest():
    assert specs.smallest_factor(4) == 2, "failed on squares of prime numbers"
    assert specs.smallest_factor(1) == 1, "failed on 1"
    assert specs.smallest_factor(2) == 2, "failed on prime numbers"
    assert specs.smallest_factor(6) == 2, "failed on compound numbers"

# Problem 2: write a unit test for specs.month_length().
def test_months():
    assert specs.month_length("April") == 30
    assert specs.month_length("January") == 31
    assert specs.month_length("February") == 28
    assert specs.month_length("February", True) == 29
    assert specs.month_length("Test") == None
    
# Problem 3: write a unit test for specs.operate().
def test_operate():
    assert specs.operate(1, 2, "+") == 3, "failed on addition"
    assert specs.operate(2, 1, "-") == 1, "failed on subtraction"
    assert specs.operate(1, 2, "*") == 2, "failed on multiplication"
    assert specs.operate(2, 1, "/") == 2, "failed on division"
    pytest.raises(TypeError, specs.operate, a=1, b=1, oper=1)
    pytest.raises(ZeroDivisionError, specs.operate, a=1, b=0, oper="/")
    pytest.raises(ValueError, specs.operate, a=1, b=1, oper="1")


# Problem 4: write unit tests for specs.Fraction, then correct it.
@pytest.fixture
def set_up_fractions():
    frac_1_3 = specs.Fraction(1, 3)
    frac_1_2 = specs.Fraction(1, 2)
    frac_n2_3 = specs.Fraction(-2, 3)
    return frac_1_3, frac_1_2, frac_n2_3

def test_fraction_init(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_3.numer == 1
    assert frac_1_2.denom == 2
    assert frac_n2_3.numer == -2
    frac = specs.Fraction(30, 42)
    assert frac.numer == 5
    assert frac.denom == 7
    pytest.raises(ZeroDivisionError, specs.Fraction, numerator = 1, denominator = 0)
    pytest.raises(TypeError, specs.Fraction, numerator = "1", denominator = 1)
    pytest.raises(TypeError, specs.Fraction, numerator = 1, denominator = "1")

def test_fraction_str(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac_1_1 = specs.Fraction(1, 1)
    assert str(frac_1_3) == "1/3"
    assert str(frac_1_2) == "1/2"
    assert str(frac_n2_3) == "-2/3"
    assert str(frac_1_1) == "1"

def test_fraction_float(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert float(frac_1_3) == 1 / 3.
    assert float(frac_1_2) == .5
    assert float(frac_n2_3) == -2 / 3.

def test_fraction_eq(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 == specs.Fraction(1, 2)
    assert frac_1_3 == specs.Fraction(2, 6)
    assert frac_n2_3 == specs.Fraction(8, -12)
    assert frac_n2_3 == -2/3

def test_fraction_add(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 + frac_1_3 == specs.Fraction(5, 6)
    assert frac_1_3 + frac_n2_3 == specs.Fraction(-1, 3)

def test_fraction_sub(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 - frac_1_3 == specs.Fraction(1, 6)
    assert frac_1_3 - frac_n2_3 == specs.Fraction(1, 1)

def test_fraction_mul(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    assert frac_1_2 * frac_1_3 == specs.Fraction(1, 6)
    assert frac_1_3 * frac_n2_3 == specs.Fraction(-2, 9)

def test_fraction_truediv(set_up_fractions):
    frac_1_3, frac_1_2, frac_n2_3 = set_up_fractions
    frac_0_1 = specs.Fraction(0, 1)
    assert frac_1_2 / frac_1_3 == specs.Fraction(3, 2)
    pytest.raises(ZeroDivisionError, specs.Fraction.__truediv__, self = frac_1_3, other = frac_0_1)
    assert frac_1_2 / frac_n2_3 == specs.Fraction(-3, 4)

# Problem 5: Write test cases for Set.

# See if we have 12
def test_set_cardnumber():
    too_many = ["1022", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020", "extra"]
    pytest.raises(ValueError, specs.count_sets, cards = too_many)

# See if unique
# First test 1 and 2, then 2 and 3, then 1 and 3 for each combination
def test_set_unique():
    not_unique1 = ["1022", "1022", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    not_unique2 = ["1022", "1122", "1022", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    not_unique3 = ["1022", "1022", "1022", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    pytest.raises(ValueError, specs.count_sets, cards = not_unique1)
    pytest.raises(ValueError, specs.count_sets, cards = not_unique2)
    pytest.raises(ValueError, specs.count_sets, cards = not_unique3)
    
# See if has 3 or 5 digits
def test_set_not4digits():
    three_digits = ["1022", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "100"]
    pytest.raises(ValueError, specs.count_sets, cards = three_digits)

    five_digits = ["1022", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "10000"]
    pytest.raises(ValueError, specs.count_sets, cards = five_digits)

# See if it has too high of a number or the wrong character
def test_set_wrongcharacter():
    wrong_num = ["1234", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    pytest.raises(ValueError, specs.count_sets, cards = wrong_num)
    
    wrong_alph = ["abcd", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    pytest.raises(ValueError, specs.count_sets, cards = wrong_alph)

# See if it can determine if something is a set
def test_set_ifset():
    assert specs.is_set(0000, 1111, 2222) == True
    assert specs.is_set(0000, 1111, 1100) == False

# See if it actually works
def test_set_numberofsets():
    hand1 = ["1022", "1122", "0100", "2021",
            "0010", "2201", "2111", "0020",
            "1102", "0210", "2110", "1020"]
    assert specs.count_sets(hand1) == 6