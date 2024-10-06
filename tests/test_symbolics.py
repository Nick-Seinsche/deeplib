"""Test suite for the symbolic differentiation module."""

import pytest
import numpy as np
from deeplib.from_scratch.symbolic_differentiation import (
    SimpleVariable,
    Add,
    Subtract,
    Multiply,
    Divide,
    Cos,
    Sin,
    SimpleFunction,
)


@pytest.fixture
def variables():
    """Fixture to create simple variables for testing."""
    x1 = SimpleVariable("x1", (1,))
    x2 = SimpleVariable("x2", (1,))
    return x1, x2


def test_simple_variable_init(variables):
    """Test initialization of SimpleVariable."""
    x1, x2 = variables
    assert x1.name == "x1"
    assert x1.shape == (1,)
    assert x2.name == "x2"
    assert x2.shape == (1,)


def test_simple_variable_evaluate(variables):
    """Test evaluation of SimpleVariable."""
    x1, x2 = variables
    input_values = {x1: 3, x2: 4}
    assert x1._evaluate(input_values) == 3
    assert x2._evaluate(input_values) == 4


def test_simple_variable_derivative(variables):
    """Test derivative of SimpleVariable."""
    x1, x2 = variables
    assert x1._derivative(x1) == 1
    assert x1._derivative(x2) == 0


def test_add_init(variables):
    """Test initialization of Add operation."""
    x1, x2 = variables
    addition = Add(x1, x2)
    assert addition.left == x1
    assert addition.right == x2


def test_add_evaluate(variables):
    """Test evaluation of Add operation."""
    x1, x2 = variables
    addition = Add(x1, x2)
    input_values = {x1: 3, x2: 4}
    assert addition._evaluate(input_values) == 7


def test_add_derivative(variables):
    """Test derivative of Add operation."""
    x1, x2 = variables
    addition = Add(x1, x2)
    assert addition._derivative(x1) == 1
    assert addition._derivative(x2) == 1


def test_subtract_init(variables):
    """Test initialization of Subtract operation."""
    x1, x2 = variables
    subtraction = Subtract(x1, x2)
    assert subtraction.left == x1
    assert subtraction.right == x2


def test_subtract_evaluate(variables):
    """Test evaluation of Subtract operation."""
    x1, x2 = variables
    subtraction = Subtract(x1, x2)
    input_values = {x1: 3, x2: 4}
    assert subtraction._evaluate(input_values) == -1


def test_subtract_derivative(variables):
    """Test derivative of Subtract operation."""
    x1, x2 = variables
    subtraction = Subtract(x1, x2)
    assert subtraction._derivative(x1) == 1
    assert subtraction._derivative(x2) == -1


def test_multiply_init(variables):
    """Test initialization of Multiply operation."""
    x1, x2 = variables
    multiplication = Multiply(x1, x2)
    assert multiplication.left == x1
    assert multiplication.right == x2


def test_multiply_evaluate(variables):
    """Test evaluation of Multiply operation."""
    x1, x2 = variables
    multiplication = Multiply(x1, x2)
    input_values = {x1: 3, x2: 4}
    assert multiplication._evaluate(input_values) == 12


def test_multiply_derivative(variables):
    """Test derivative of Multiply operation."""
    x1, x2 = variables
    multiplication = Multiply(x1, x2)
    assert multiplication._derivative(x1) == x2
    assert multiplication._derivative(x2) == x1


def test_divide_init(variables):
    """Test initialization of Divide operation."""
    x1, x2 = variables
    division = Divide(x1, x2)
    assert division.left == x1
    assert division.right == x2


def test_divide_evaluate(variables):
    """Test evaluation of Divide operation."""
    x1, x2 = variables
    division = Divide(x1, x2)
    input_values = {x1: 4, x2: 2}
    assert division._evaluate(input_values) == 2


def test_divide_derivative(variables):
    """Test derivative of Divide operation."""
    x1, x2 = variables
    division = Divide(x1, x2)
    derivative_x1 = division._derivative(x1)
    derivative_x2 = division._derivative(x2)
    expected_derivative_x1 = 1 / x2
    expected_derivative_x1_alt = x2 / (x2 * x2)
    expected_derivative_x2 = -x1 / (x2 * x2)
    assert (
        derivative_x1 == expected_derivative_x1
        or derivative_x1 == expected_derivative_x1_alt
    )
    assert derivative_x2 == expected_derivative_x2


def test_cos_init(variables):
    """Test initialization of Cos operation."""
    x1, _ = variables
    cosine = Cos(x1)
    assert cosine.term == x1


def test_cos_evaluate(variables):
    """Test evaluation of Cos operation."""
    x1, _ = variables
    cosine = Cos(x1)
    input_values = {x1: 0}
    assert cosine._evaluate(input_values) == 1


def test_cos_derivative(variables):
    """Test derivative of Cos operation."""
    x1, _ = variables
    cosine = Cos(x1)
    assert cosine._derivative(x1) == -Sin(x1)


def test_sin_init(variables):
    """Test initialization of Sin operation."""
    x1, _ = variables
    sine = Sin(x1)
    assert sine.term == x1


def test_sin_evaluate(variables):
    """Test evaluation of Sin operation."""
    x1, _ = variables
    sine = Sin(x1)
    input_values = {x1: 0}
    assert sine._evaluate(input_values) == 0


def test_sin_derivative(variables):
    """Test derivative of Sin operation."""
    x1, _ = variables
    sine = Sin(x1)
    assert sine._derivative(x1) == Cos(x1)


def test_simple_function_init(variables):
    """Test initialization of SimpleFunction."""
    x1, x2 = variables
    term = Add(x1, x2)
    simple_func = SimpleFunction("f", [x1, x2], term)
    var1, var2 = simple_func.variables
    assert simple_func.term == var1 + var2


def test_simple_function_evaluate(variables):
    """Test evaluation of SimpleFunction."""
    x1, x2 = variables
    term = Add(x1, x2)
    simple_func = SimpleFunction("f", [x1, x2], term)
    assert simple_func([3, 4]) == 7


def test_simple_function_jacobian(variables):
    """Test Jacobian of SimpleFunction."""
    x1, x2 = variables
    term = Add(x1, x2)
    jacobian_term = term.jacobian([x1, x2])
    simple_func = SimpleFunction("f", [x1, x2], jacobian_term)
    assert simple_func.term.value[0] == 1
    assert simple_func.term.value[1] == 1


def test_simple_function_hessian(variables):
    """Test Hessian of SimpleFunction."""
    x1, x2 = variables
    term = Add(x1, x2)
    hessian_term = term.hessian([x1, x2])
    simple_func = SimpleFunction("f", [x1, x2], hessian_term)
    expected_hessian = np.array([[0, 0], [0, 0]])
    assert np.all(simple_func.term.value == expected_hessian)
