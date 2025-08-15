"""
2D Linear Programming Plotter Package

This package provides tools for visualizing 2D linear programming problems,
including constraints, feasible regions, intersection points, and optimal solutions.
"""

from .two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from .plotter import TwoDimensionalLinearPlotter

__all__ = ['TwoDimensionalLinearConstraint', 'TwoDimensionalLinearPlotter']
__version__ = '1.0.0'
