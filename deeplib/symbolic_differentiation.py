from __future__ import annotations

"""
Module implementing a symbolic mathematical expression framework with support
for basic arithmetic operations, evaluation, differentiation, Jacobians, and
Hessians.
"""

# Standard
from typing import Any, Self, Literal

# Third-Party
import numpy as np


class SimpleTerm:
    """
    Base class representing a symbolic term.

    Support basic arithmetic operations, evaluation, differentiation,
    and matrix calculus (Jacobian and Hessian).
    """

    def __add__(self: Self, other: Self) -> Self | Add:
        """Add two terms together."""
        if other == 0:
            return self
        return Add(self, other)

    def __radd__(self: Self, other: Self) -> Self | Add:
        """Add a term to another term."""
        if other == 0:
            return self
        return Add(self, other)

    def __sub__(self: Self, other: Self) -> Self | Subtract:
        """Subtract two terms."""
        if other == 0:
            return self
        return Subtract(self, other)

    def __rsub__(self: Self, other: Self) -> Self | Subtract:
        """Subtract a term from another term."""
        if other == 0:
            return -self
        return Subtract(other, self)

    def __mul__(self: Self, other: Self) -> Self | Multiply:
        """Multiply two terms."""
        if other == 0:
            return 0
        if other == 1:
            return self
        return Multiply(self, other)

    def __rmul__(self: Self, other: Self) -> Self | Multiply:
        """Multiply a term by another term."""
        if other == 0:
            return 0
        if other == 1:
            return self
        return Multiply(self, other)

    def __truediv__(self: Self, other: Self) -> Self | Divide:
        """Divide two terms."""
        if other == 0:
            raise ZeroDivisionError("Division by zero")
        if other == 1:
            return self
        return Divide(self, other)

    def __rtruediv__(self: Self, other: Self) -> Self | Divide:
        """Divide a term by another term."""
        if other == 0:
            return 0
        return Divide(other, self)

    def __neg__(self: Self) -> Self:
        """Negate the term."""
        return (-1) * self

    def _evaluate(self: Self, input_values: dict[SimpleVariable, (float, int)]) -> None:
        """Evaluate the term given a set of input values.

        Args:
            input_values: Dictionary mapping variables to values.

        Returns:
            The evaluated value of the term.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Note:
            This method should be implemented in a subclass.
        """
        raise NotImplementedError("Evaluation not implemented")

    def _derivative(self: Self, deriving_variable: SimpleVariable) -> None:
        """Compute the derivative of the term.

        Args:
            deriving_variable: The variable with respect to which the derivative
            is computed.

        Returns:
            The derivative of the term.

        Raises:
            NotImplementedError: If the method is not implemented in a subclass.

        Note:
            This method should be implemented in a subclass.
        """
        raise NotImplementedError("Derivative not implemented")

    def _substitute(
        self: Self, substitutions: dict[SimpleVariable, SimpleVariable]
    ) -> Self:
        raise NotImplementedError("Substitution not implemented")

    def jacobian(self, deriving_variables: list[SimpleVariable]) -> MatrixVariable:
        """Compute the Jacobian matrix of the term.

        Args:
            deriving_variables: A list of variables to compute the Jacobian.

        Returns:
            MatrixVariable: A matrix representing the Jacobian.
        """
        return MatrixVariable(
            f"J {self}",
            (len(deriving_variables),),
            value=np.array(
                [
                    self._derivative(deriving_variable)
                    for deriving_variable in deriving_variables
                ]
            ),
        )

    def hessian(self, deriving_variables: list[SimpleVariable]) -> MatrixVariable:
        """Compute the Hessian matrix of the term.

        Args:
            deriving_variables: A list of variables to compute the Hessian.

        Returns:
            MatrixVariable: A matrix representing the Hessian.
        """
        return MatrixVariable(
            f"H {self}",
            (len(deriving_variables), len(deriving_variables)),
            value=np.column_stack(
                [
                    self.jacobian(deriving_variables)._derivative(deriving_variable)
                    for deriving_variable in deriving_variables
                ]
            ),
        )


class SimpleVariable(SimpleTerm):
    """A class representing a single (base) variable within the symbolic system."""

    def __init__(self: Self, name: str, shape: tuple[int] = None) -> None:
        """Initialize the SimpleVariable object."""
        self.name = name
        self.shape = shape

    def __hash__(self: Self) -> int:
        """Return the hash of the variable name."""
        return hash(self.name)

    def __repr__(self) -> str:
        """Return the string representation of the variable."""
        return self.name

    def _evaluate(self, input_values: dict[SimpleVariable, float | int]) -> float | int:
        """Evaluate the variable given input values.

        Args:
            input_values: Dictionary mapping variables to values.

        Returns:
            The value of the variable.
        """
        return input_values[self]

    def _derivative(self: Self, deriving_variable: SimpleVariable) -> int:
        """Compute the derivative of the variable.

        Args:
            deriving_variable: The variable with respect to which the derivative
            is computed.

        Returns:
            The derivative of the variable with respect to the deriving variable.
        """
        return int(self == deriving_variable)

    def _substitute(
        self: Self, substitutions: dict[SimpleVariable, SimpleVariable]
    ) -> Self:
        """Substitute the variable with another variable."""
        return substitutions.get(self, self)


class MatrixVariable(SimpleTerm):
    """
    A class representing a matrix of symbolic variables.

    With support for matrix operations like multiplication and differentiation.
    """

    def __init__(
        self: Self, name: str, shape: tuple[int], value: np.ndarray | None = None
    ) -> None:
        """Initialize the MatrixVariable object."""
        self.name = name
        self.shape = shape
        if value is None:
            self.value = np.array(
                [
                    SimpleVariable(f"{name}_{i}", shape)
                    for i in range(shape[0] * shape[1])
                ]
            ).reshape(shape)
        else:
            self.value = value

    def __mul__(self: Self, other: Self) -> MatrixVariable:
        """
        Multiply the matrix with another matrix.

        Note:
            This overrides the __mul__ method of the SimpleTerm class which assumes
            multiplication to be commutative.
        """
        return np.matmul(self.value, other.value)  # todo

    def __repr__(self: Self) -> str:
        """Return the string representation of the matrix."""
        return self.value.__repr__()

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> np.ndarray:
        """Evaluate the matrix given input values.

        Args:
            input_values: Dictionary mapping variables to values.

        Returns:
            The evaluated matrix.
        """
        if len(self.shape) == 1:
            return np.array(
                [
                    (
                        element._evaluate(input_values)
                        if isinstance(element, SimpleTerm)
                        else element
                    )
                    for element in self.value
                ]
            ).reshape(self.shape)
        elif len(self.shape) == 2:
            return np.array(
                [
                    [
                        (
                            element._evaluate(input_values)
                            if isinstance(element, SimpleTerm)
                            else element
                        )
                        for element in row
                    ]
                    for row in self.value
                ]
            ).reshape(self.shape)
        else:
            raise NotImplementedError("Higher dimensional matrices not implemented")

    def _derivative(self: Self, deriving_variable: SimpleVariable) -> np.ndarray:
        """Compute the derivative of the matrix.

        Args:
            deriving_variable: The variable with respect to which the derivative
            is computed.

        Returns:
            The derivative of the matrix.
        """
        if len(self.shape) == 1:
            return np.array(
                [
                    (
                        element._derivative(deriving_variable)
                        if isinstance(element, SimpleTerm)
                        else 0
                    )
                    for element in self.value
                ]
            ).reshape(self.shape)
        elif len(self.shape) == 2:
            return np.array(
                [
                    [
                        (
                            element._derivative(deriving_variable)
                            if isinstance(element, SimpleTerm)
                            else 0
                        )
                        for element in row
                    ]
                    for row in self.value
                ]
            ).reshape(self.shape)
        else:
            raise NotImplementedError("Higher dimensional matrices not implemented")

    def _substitute(
        self: Self, substitutions: dict[SimpleVariable, SimpleVariable]
    ) -> Self:
        """Substitute the matrix with another matrix."""
        return MatrixVariable(
            self.name,
            self.shape,
            value=np.array(
                [
                    (
                        element._substitute(substitutions)
                        if isinstance(element, SimpleTerm)
                        else element
                    )
                    for element in self.value.flatten()
                ]
            ).reshape(self.shape),
        )


class TwoVariableOperation(SimpleTerm):
    """A class representing an operation between two terms."""

    def __init__(self: Self, left: SimpleTerm, right: SimpleTerm) -> None:
        """Initialize the TwoVariableOperation object."""
        self.left = left
        self.right = right

    def _substitute(self, substitutions: dict[SimpleVariable, SimpleVariable]) -> Self:
        """Substitute the variables in the operation with other variables."""
        return type(self)(
            (
                self.left._substitute(substitutions)
                if isinstance(self.left, SimpleTerm)
                else self.left
            ),
            (
                self.right._substitute(substitutions)
                if isinstance(self.right, SimpleTerm)
                else self.right
            ),
        )

    def __eq__(self: Self, other: Self) -> bool:
        """Check if two operations are equal."""
        return (
            isinstance(self, type(other))
            and self.left == other.left
            and self.right == other.right
        )


class SingleVariableOperation(SimpleTerm):
    """A class representing an operation on a single term."""

    def __init__(self: Self, term: SimpleTerm) -> None:
        """Initialize the SingleVariableOperation object."""
        self.term = term

    def _substitute(self, substitutions: dict[SimpleVariable, SimpleVariable]) -> Self:
        """Substitute the variable in the operation with another variable."""
        return type(self)(
            self.term._substitute(substitutions)
            if isinstance(self.term, SimpleTerm)
            else self.term
        )

    def __eq__(self: Self, other: Self) -> bool:
        """Check if two operations are equal."""
        return isinstance(self, type(other)) and self.term == other.term


class Add(TwoVariableOperation):
    """A class representing the addition of two terms.

    This class models the addition operation between two mathematical terms, which can
    be either symbolic expressions or numeric constants. It provides methods to evaluate
    the addition for given variable values and compute the derivative with respect to
    a specific variable.

    Attributes:
        left (SimpleTerm | float | int): The left term of the addition.
        right (SimpleTerm | float | int): The right term of the addition.

    Methods:
        __repr__(): Returns the string representation of the addition term.
        _evaluate(input_values: dict[SimpleVariable, float | int]) -> float | int:
            Evaluates the addition expression given variable values.
        _derivative(deriving_variable: SimpleVariable) -> SimpleTerm | int | float:
            Computes the derivative of the addition expression with respect to a
            given variable.
    """

    def __init__(
        self: Self, left: SimpleTerm | float | int, right: SimpleTerm | float | int
    ) -> None:
        """
        Initialize the Add object with two terms.

        `left` and `right`, which can be either symbolic terms, integers,
        or floating-point numbers.

        Args:
            left (SimpleTerm | float | int): The left term of the addition.
            right (SimpleTerm | float | int): The right term of the addition.
        """
        self.left = left
        self.right = right

    def __repr__(self: Self) -> str:
        """
        Return the string representation of the addition operation.

        Returns:
            str: A string in the format "(left + right)" where `left` and `right`
            represent the respective terms.
        """
        return f"({self.left} + {self.right})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the addition expression for the given values of the variables.

        If both `left` and `right` are symbolic terms, they are evaluated first,
        then added. If one is a constant, it is added directly. Special handling
        is done when one of the terms is zero.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary mapping
            variables to their corresponding numeric values.

        Returns:
            float | int: The result of the evaluated addition.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return self.left._evaluate(input_values) + self.right._evaluate(
                input_values
            )
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, (int, float)):
            return self.right + self.left._evaluate(input_values)
        if isinstance(self.left, (int, float)) and isinstance(self.right, SimpleTerm):
            return self.left + self.right._evaluate(input_values)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Compute derivative of the addition expression with respect to a given variable.

        The derivative is computed as the sum of the derivatives of the left
        and right terms.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which the
            derivative is to be computed.

        Returns:
            SimpleTerm | int | float: The derivative of the addition expression.
        """
        left_derivative = (
            self.left._derivative(deriving_variable)
            if isinstance(self.left, SimpleTerm)
            else 0
        )

        right_derivative = (
            self.right._derivative(deriving_variable)
            if isinstance(self.right, SimpleTerm)
            else 0
        )

        return left_derivative + right_derivative


class Subtract(TwoVariableOperation):
    """
    A class representing the subtraction of two symbolic terms.

    This class is used to represent the subtraction operation between
    two terms (or constants), providing methods for evaluating the difference
    and computing its derivative.

    Attributes:
        left (SimpleTerm | float | int): The term or constant from which the
            subtraction is performed.
        right (SimpleTerm | float | int): The term or constant to subtract.
    """

    def __init__(
        self: Self, left: SimpleTerm | float | int, right: SimpleTerm | float | int
    ) -> None:
        """
        Initialize the Subtract class with two terms or constants.

        Args:
            left (SimpleTerm | float | int): The term or constant from which the
                subtraction is performed.
            right (SimpleTerm | float | int): The term or constant to subtract.
        """
        self.left = left
        self.right = right

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the subtraction expression.

        Returns:
            str: A string in the form "(left - right)".
        """
        return f"({self.left} - {self.right})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the subtraction of the two terms with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary mapping
                variables to their numerical values.

        Returns:
            float | int: The result of evaluating the subtraction of the two terms.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return self.left._evaluate(input_values) - self.right._evaluate(
                input_values
            )
        if (
            isinstance(self.left, SimpleTerm)
            and isinstance(self.right, (int, float))
            and self.right == 0
        ):
            return self.left._evaluate(input_values)
        if (
            isinstance(self.left, (int, float))
            and isinstance(self.right, SimpleTerm)
            and self.left == 0
        ):
            return -self.right._evaluate(input_values)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the subtraction expression with respect to a given variable.

        The derivative is computed using the difference rule for symbolic terms.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which the
            derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the subtraction expression.
        """
        left_derivative = (
            self.left._derivative(deriving_variable)
            if isinstance(self.left, SimpleTerm)
            else 0
        )

        right_derivative = (
            self.right._derivative(deriving_variable)
            if isinstance(self.right, SimpleTerm)
            else 0
        )

        return left_derivative - right_derivative


class Multiply(TwoVariableOperation):
    """
    A class representing the multiplication of two symbolic terms.

    This class is used to represent the multiplication of two terms (or constants),
    providing methods for evaluating the product and computing its derivative.

    Attributes:
        left (SimpleTerm | float | int): The first term in the multiplication.
        right (SimpleTerm | float | int): The second term in the multiplication.
    """

    def __init__(
        self: Self, left: SimpleTerm | float | int, right: SimpleTerm | float | int
    ) -> None:
        """
        Initialize the Multiply class with two terms or constants.

        Args:
            left (SimpleTerm | float | int): The first term or constant
                to be multiplied.
            right (SimpleTerm | float | int): The second term or constant
                to be multiplied.
        """
        self.left = left
        self.right = right

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the multiplication expression.

        Returns:
            str: A string in the form "(left * right)".
        """
        return f"({self.left} * {self.right})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the multiplication of the two terms with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
                mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the product of the two terms.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return self.left._evaluate(input_values) * self.right._evaluate(
                input_values
            )
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, (int, float)):
            return self.left._evaluate(input_values) * self.right
        if isinstance(self.left, (int, float)) and isinstance(self.right, SimpleTerm):
            return self.left * self.right._evaluate(input_values)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the multiplication expression with respect to a given variable.

        The derivative is computed using the product rule when both terms are symbolic.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the multiplication expression.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return self.left._derivative(
                deriving_variable
            ) * self.right + self.left * self.right._derivative(deriving_variable)
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, (int, float)):
            return self.left._derivative(deriving_variable) * self.right
        if isinstance(self.left, (int, float)) and isinstance(self.right, SimpleTerm):
            return self.left * self.right._derivative(deriving_variable)


class Divide(TwoVariableOperation):
    """
    A class representing the division of two symbolic terms.

    This class is used to represent the division of two terms (or constants),
    allowing for the evaluation of the division and the computation of its derivative.

    Attributes:
        left (SimpleTerm | float | int): The numerator in the division expression.
        right (SimpleTerm | float | int): The denominator in the division expression.
    """

    def __init__(
        self: Self, left: SimpleTerm | float | int, right: SimpleTerm | float | int
    ) -> None:
        """
        Initialize the Divide class with a numerator and denominator.

        Args:
            left (SimpleTerm | float | int): The term or constant to be used as
                the numerator.
            right (SimpleTerm | float | int): The term or constant to be used as
                the denominator.
        """
        self.left = left
        self.right = right

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the division expression.

        Returns:
            str: A string in the form "(left / right)".
        """
        return f"({self.left} / {self.right})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, (float, int)]
    ) -> float | int:
        """
        Evaluate the division expression with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary mapping
                variables to their numerical values.

        Returns:
            float | int: The result of evaluating the division of the two terms.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return self.left._evaluate(input_values) / self.right._evaluate(
                input_values
            )
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, (int, float)):
            return self.left._evaluate(input_values) / self.right
        if isinstance(self.left, (int, float)) and isinstance(self.right, SimpleTerm):
            return self.left / self.right._evaluate(input_values)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the division expression with respect to a given variable.

        The derivative is computed using the quotient rule when both the numerator
        and denominator are symbolic terms.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the division expression.
        """
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, SimpleTerm):
            return (
                self.left._derivative(deriving_variable) * self.right
                - self.left * self.right._derivative(deriving_variable)
            ) / (self.right * self.right)
        if isinstance(self.left, SimpleTerm) and isinstance(self.right, (int, float)):
            return self.left._derivative(deriving_variable) / self.right
        if isinstance(self.left, (int, float)) and isinstance(self.right, SimpleTerm):
            return (-self.left * self.right._derivative(deriving_variable)) / (
                self.right * self.right
            )


class Cos(SingleVariableOperation):
    """
    A class representing the cosine of a symbolic term.

    This class is used to compute the cosine of a given term, allowing for the
    evaluation of the cosine and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        cosine is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Cos class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the cosine function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the cosine expression.

        Returns:
            str: A string in the form "Cos(term)".
        """
        return f"Cos({self.term})"

    def _evaluate(self: Self, input_values: dict[Any, float | int]) -> float | int:
        """
        Evaluate the cosine of the term with the provided variable values.

        Args:
            input_values (dict[Any, float | int]): A dictionary mapping variables
            to their numerical values.

        Returns:
            float | int: The result of evaluating the cosine of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.cos(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return np.cos(self.term)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the cosine of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the cosine, which is the
            negative sine of the term multiplied by the derivative of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return -Sin(self.term) * self.term._derivative(deriving_variable)
        if isinstance(self.term, (int, float)):
            return 0


class Sin(SingleVariableOperation):
    """
    A class representing the sine of a symbolic term.

    This class is used to compute the sine of a given term, allowing for the
    evaluation of the sine and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        sine is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Sin class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the sine function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the sine expression.

        Returns:
            str: A string in the form "Sin(term)".
        """
        return f"Sin({self.term})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the sine of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the sine of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.sin(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return np.sin(self.term)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the sine of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect
            to which the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the sine, which is
            the cosine of the term multiplied by the derivative of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return Cos(self.term) * self.term._derivative(deriving_variable)
        if isinstance(self.term, (int, float)):
            return 0


class Exp(SingleVariableOperation):
    """
    A class representing the exponential of a symbolic term.

    This class is used to compute the exponential of a given term, allowing for the
    evaluation of the exponential and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        exponential is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Exp class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the exponential function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the exponential expression.

        Returns:
            str: A string in the form "Exp(term)".
        """
        return f"Exp({self.term})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the exponential of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the exponential of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.exp(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return np.exp(self.term)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the exponential of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the exponential, which is
            the exponential of the term multiplied by the derivative of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return Exp(self.term) * self.term._derivative(deriving_variable)
        if isinstance(self.term, (int, float)):
            return 0


class Log(SingleVariableOperation):
    """
    A class representing the natural logarithm of a symbolic term.

    This class is used to compute the natural logarithm of a given term, allowing for
    the evaluation of the logarithm and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        natural logarithm is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Log class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the natural logarithm function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the natural logarithm expression.

        Returns:
            str: A string in the form "Log(term)".
        """
        return f"Log({self.term})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the natural logarithm of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the natural logarithm of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.log(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return np.log(self.term)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the natural logarithm of the term with respect to a variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the natural logarithm, which is
            the reciprocal of the term multiplied by the derivative of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return self.term._derivative(deriving_variable) / self.term
        if isinstance(self.term, (int, float)):
            return 0


class Sqrt(SingleVariableOperation):
    """
    A class representing the square root of a symbolic term.

    This class is used to compute the square root of a given term, allowing for the
    evaluation of the square root and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        square root is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Sqrt class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the square root function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the square root expression.

        Returns:
            str: A string in the form "Sqrt(term)".
        """
        return f"Sqrt({self.term})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the square root of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the square root of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.sqrt(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return np.sqrt(self.term)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the square root of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the square root, which is
            the reciprocal of twice the square root of the term multiplied by the
            derivative of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return self.term._derivative(deriving_variable) / (2 * Sqrt(self.term))
        if isinstance(self.term, (int, float)):
            return 0


class RELU(SingleVariableOperation):
    """
    A class representing the maximum of a symbolic term.

    This class is used to compute the maximum of a given term, allowing for the
    evaluation of the maximum and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        maximum is computed.
    """

    def __init__(self: Self, term: SimpleTerm | float | int) -> None:
        """
        Initialize the Max class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the maximum function to.
        """
        self.term = term

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the maximum expression.

        Returns:
            str: A string in the form "Max(term)".
        """
        return f"Max({self.term}, 0)"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the maximum of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the maximum of the term.
        """
        if isinstance(self.term, SimpleTerm):
            return np.max(self.term._evaluate(input_values))
        if isinstance(self.term, (int, float)):
            return self.term

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the maximum of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the maximum, which is
            the derivative of the term if the term is the maximum, and zero
            otherwise.
        """
        if isinstance(self.term, SimpleTerm):
            return self.term._derivative(deriving_variable) * (
                self.term == RELU(self.term)
            )
        if isinstance(self.term, (int, float)):
            return 0


class Norm(SingleVariableOperation):
    """
    A class representing the norm of a symbolic term.

    This class is used to compute the norm of a given term, allowing for the
    evaluation of the norm and the computation of its derivative.

    Attributes:
        term (SimpleTerm | float | int): The term or constant for which the
        norm is computed.
    """

    def __init__(
        self: Self, term: MatrixVariable, p: float | Literal["fro"] = 2
    ) -> None:
        """
        Initialize the Norm class with a given term.

        Args:
            term (SimpleTerm | float | int): The term or constant to apply
            the norm function to.
        """
        self.term = term
        self.p = p

    def __repr__(self: Self) -> str:
        """
        Provide a string representation of the norm expression.

        Returns:
            str: A string in the form "Norm(term)".
        """
        return f"Norm({self.term}, p={self.p})"

    def _evaluate(
        self: Self, input_values: dict[SimpleVariable, float | int]
    ) -> float | int:
        """
        Evaluate the norm of the term with the provided variable values.

        Args:
            input_values (dict[SimpleVariable, float | int]): A dictionary
            mapping variables to their numerical values.

        Returns:
            float | int: The result of evaluating the norm of the term.
        """
        if isinstance(self.term, (SimpleTerm, MatrixVariable)):
            return np.linalg.norm(self.term._evaluate(input_values), ord=self.p)
        if isinstance(self.term, (int, float)):
            return np.linalg.norm(self.term, ord=self.p)

    def _derivative(
        self: Self, deriving_variable: SimpleVariable
    ) -> SimpleTerm | int | float:
        """
        Differentiate the norm of the term with respect to a given variable.

        Args:
            deriving_variable (SimpleVariable): The variable with respect to which
            the derivative is computed.

        Returns:
            SimpleTerm | int | float: The derivative of the norm, which is the
            normalized term multiplied by the derivative of the term.
        """
        if isinstance(self.term, MatrixVariable):
            return sum(self.term._derivative(deriving_variable)) / Norm(
                self.term, self.p
            )
        raise TypeError(f"Norm derivative not implemented for {type(self.term)}")


class SimpleFunction:
    """
    A class representing a mathematical function consisting of symbolic terms.

    This class allows for the evaluation of the function, as well as the computation
    of the Jacobian and Hessian matrices with respect to specified variables.

    Attributes:
        variables (list[SimpleVariable]): List of variables that the
            function depends on.
        term (SimpleTerm): The symbolic expression representing the function's term.
    """

    def __init__(
        self: Self, name: str, variables: list[SimpleVariable], term: SimpleTerm
    ) -> None:
        """
        Initialize the SimpleFunction with variables and a term.

        Args:
            variables (list[SimpleVariable]): A list of symbolic variables
                used in the function.
            term (SimpleTerm): The symbolic term that defines the function.
        """
        self.name = name
        self.variables = [
            SimpleVariable(f"__{variable.name}", variable.shape)
            for variable in variables
        ]

        self.term = term._substitute(
            {var_old: var_new for var_old, var_new in zip(variables, self.variables)}
        )

    def __copy__(self: Self) -> Self:
        """Create a copy of the function."""
        return SimpleFunction(self.variables, self.term.copy())

    def __call__(self: Self, values: list[float | int]) -> np.ndarray | float | int:
        """
        Evaluate the function by substituting the given values for the variables.

        Args:
            values (list[float | int]): A list of numerical values to substitute
            for the variables.

        Returns:
            np.ndarray | float | int: The evaluated result of the function, which
            can be a scalar or a numpy array.
        """
        return self.term._evaluate(
            {variable: value for variable, value in zip(self.variables, values)}
        )

    def __repr__(self: Self) -> str:
        """
        Representation string of the function.

        Returns:
            str: A string representing the function in
            the form "Function(variables -> term)".
        """
        return f"Function({str(self.variables)} -> {self.term.__repr__()})"


if __name__ == "__main__":
    x1 = SimpleVariable("x1", (1,))
    x2 = SimpleVariable("x2", (1,))
    x3 = SimpleVariable("x3", (1,))
    x4 = SimpleVariable("x4", (1,))

    variables = [x1, x2, x3, x4]
    term = (
        Sqrt(x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4)
        + Exp(x1 + x2 + x3 - x4 * x4)
        + Cos(x1)
        + x2
        + Norm(MatrixVariable("v", (3,), np.array([x1, x2, x3])), 2)
    )

    jterm = term.jacobian(variables)
    hterm = term.hessian(variables)

    print(jterm)
    print(hterm)

    f = SimpleFunction("f", variables, term)
    jf = SimpleFunction("jf", variables, jterm)
    hf = SimpleFunction("hf", variables, hterm)

    # evaluate the function
    print(f([1, 2, 3, 4]))
    print(jf([1, 2, 3, 4]))
    print(hf([1, 2, 3, 4]))
