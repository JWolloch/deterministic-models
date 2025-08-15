from pydantic import BaseModel, field_validator, Field
from typing import Literal

class TwoDimensionalLinearConstraint(BaseModel):
    """
    A 2D linear constraint of the form ax + by <= c.
    Args:
        (float) a: The coefficient of x.
        (float) b: The coefficient of y.
        (float) c: The constraints right hand side.
        (str) sign: The sign of the constraint.
        (list[str]) symbols: The symbols to use for the variables, default is ['x', 'y'].
    """
    a: float | int
    b: float | int
    c: float | int
    sign: Literal['<=', '>=', '==']
    symbols: list[str] = Field(default=['x', 'y'], min_length=2, max_length=2)

    @field_validator('symbols')
    @classmethod
    def validate_symbols(cls, v: list[str]):
        if len(v) != 2:
            raise ValueError("Symbols must be a list of two strings")
        return v
    
    def intersection(self, other: 'TwoDimensionalLinearConstraint') -> tuple[float, float]:
        """
        Finds the intersection of the lines defined by the constraints.
        
        Solves the system:
        a1*x + b1*y = c1
        a2*x + b2*y = c2
        
        Returns:
            tuple[float, float]: The (x, y) coordinates of the intersection point
            
        Raises:
            ValueError: If the lines are parallel (no intersection or infinite intersections)
        """
        # Extract coefficients
        a1, b1, c1 = self.a, self.b, self.c
        a2, b2, c2 = other.a, other.b, other.c
        
        # Calculate determinant
        det = a1 * b2 - a2 * b1
        
        # Check if lines are parallel
        if abs(det) < 1e-10:  # Using small epsilon for floating point comparison
            raise ValueError("Lines are parallel - no unique intersection point")
        
        # Use Cramer's rule to solve the system
        x = (c1 * b2 - c2 * b1) / det
        y = (a1 * c2 - a2 * c1) / det
        
        return (x, y)
    
    def satisfies(self, x: float, y: float) -> bool:
        """
        Checks if the constraint is satisfied by the given x and y values.
        """
        lhs = self.a * x + self.b * y
        
        if self.sign == '<=':
            return lhs <= self.c
        elif self.sign == '>=':
            return lhs >= self.c
        elif self.sign == '==':
            return abs(lhs - self.c) < 1e-10  # Use epsilon for floating point equality
        else:
            raise ValueError(f"Unknown sign: {self.sign}")
    
    def x_given_y(self, y: float) -> float:
        """
        Returns the x value for a given y value on the constraint boundary line.
        
        Raises:
            ValueError: If coefficient 'a' is zero (vertical line)
        """
        if abs(self.a) < 1e-10:
            raise ValueError("Cannot solve for x when coefficient 'a' is zero (vertical line)")
        return (self.c - self.b * y) / self.a
    
    def y_given_x(self, x: float) -> float:
        """
        Returns the y value for a given x value on the constraint boundary line.
        
        Raises:
            ValueError: If coefficient 'b' is zero (horizontal line)
        """
        if abs(self.b) < 1e-10:
            raise ValueError("Cannot solve for y when coefficient 'b' is zero (horizontal line)")
        return (self.c - self.a * x) / self.b
    
    def _format_symbol_latex(self, symbol: str) -> str:
        """Convert symbol to LaTeX format."""
        # Handle subscripts like x1 -> x_1, x2 -> x_2, etc.
        if len(symbol) > 1 and symbol[-1].isdigit():
            base = symbol[:-1]
            subscript = symbol[-1]
            return f"${base}_{{{subscript}}}$"
        else:
            return f"${symbol}$"
    
    def __str__(self):
        terms = []
        
        # Convert symbols to LaTeX format
        latex_symbol_0 = self._format_symbol_latex(self.symbols[0])
        latex_symbol_1 = self._format_symbol_latex(self.symbols[1])
        
        # Handle x coefficient (a)
        if self.a != 0:
            if self.a == 1:
                terms.append(latex_symbol_0)
            elif self.a == -1:
                terms.append(f"-{latex_symbol_0}")
            else:
                terms.append(f"{self.a}{latex_symbol_0}")
        
        # Handle y coefficient (b)
        if self.b != 0:
            if self.b == 1:
                if terms:  # If there are previous terms
                    terms.append(f" + {latex_symbol_1}")
                else:
                    terms.append(latex_symbol_1)
            elif self.b == -1:
                if terms:
                    terms.append(f" - {latex_symbol_1}")
                else:
                    terms.append(f"-{latex_symbol_1}")
            else:
                if terms and self.b > 0:
                    terms.append(f" + {self.b}{latex_symbol_1}")
                elif terms and self.b < 0:
                    terms.append(f" - {abs(self.b)}{latex_symbol_1}")
                else:
                    terms.append(f"{self.b}{latex_symbol_1}")
        
        # Handle case where both coefficients are 0
        if not terms:
            terms.append("0")
        
        return f"{''.join(terms)} {self.sign} {self.c}"

    def __repr__(self):
        return f"LinearConstraint(a={self.a}, b={self.b}, c={self.c}, sign='{self.sign}', symbols={self.symbols})"