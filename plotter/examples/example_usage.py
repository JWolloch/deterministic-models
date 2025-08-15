#!/usr/bin/env python3
"""
Example usage of the TwoDimensionalLinearPlotter class.

This example demonstrates how to:
1. Create linear constraints
2. Set up an objective function
3. Plot with various customization options
4. Save plots in different formats
"""

import sys
import os
# Add the parent directory (deterministic-models) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plotter.two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from plotter.plotter import TwoDimensionalLinearPlotter


def main():
    # Example 1: Basic Linear Programming Problem
    print("Creating Example 1: Basic LP Problem")
    
    # Define constraints:
    # x + y <= 4
    # 2x + y <= 6  
    # x >= 0
    # y >= 0
    constraints = [
        TwoDimensionalLinearConstraint(a=1, b=1, c=4, sign='<='),
        TwoDimensionalLinearConstraint(a=2, b=1, c=6, sign='<='),
        TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>='),  # x >= 0
        TwoDimensionalLinearConstraint(a=0, b=1, c=0, sign='>='),  # y >= 0
    ]
    
    # Define objective function: maximize 3x + 2y
    objective = {'x': 3, 'y': 2}
    
    # Create plotter
    plotter = TwoDimensionalLinearPlotter(
        constraints=constraints,
        objective_coefficients=objective,
        figsize=(12, 8)
    )
    
    # Customize styling
    plotter.set_feasible_region_style(color='lightgreen', alpha=0.4)
    plotter.set_intersection_style(
        feasible_color='darkgreen',
        infeasible_color='red',
        marker='s',  # square markers
        size=80
    )
    
    # Plot with single color constraints and optimal point
    print("Plotting with single color constraints and optimal point...")
    plotter.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,
        show_infeasible_intersections=True,
        show_objective_gradient=True,
        show_optimal_point=True,  # NEW: Show optimal point
        maximize_objective=True,   # NEW: Maximize the objective
        constraint_color_mode='single',
        constraint_single_color='blue',
        title='Example 1: Basic LP Problem',  # NEW: Custom title
        show_objective_function=True,  # NEW: Show objective function
        save_path='plots/example1_single_color_with_optimal_and_title',
        save_format='png',
        show_plot=False  # Don't show, just save
    )
    
    # Plot with multiple colors and optimal point (minimization)
    print("Plotting with multiple color constraints and optimal point (minimization)...")
    plotter.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,
        show_infeasible_intersections=False,
        show_objective_gradient=True,
        show_optimal_point=True,   # NEW: Show optimal point
        maximize_objective=False,  # NEW: Minimize the objective
        constraint_color_mode='multiple',
        save_path='plots/example1_multiple_colors_minimize',
        save_format='png',
        show_plot=False  # Don't show, just save
    )
    
    # Example 2: Custom symbols
    print("\nCreating Example 2: Custom Symbols")
    
    # Define constraints with custom symbols x1, x2:
    # 2*x1 + 3*x2 <= 12
    # x1 + x2 <= 5
    # x1 >= 0
    # x2 >= 0
    constraints2 = [
        TwoDimensionalLinearConstraint(a=2, b=3, c=12, sign='<=', symbols=['x1', 'x2']),
        TwoDimensionalLinearConstraint(a=1, b=1, c=5, sign='<=', symbols=['x1', 'x2']),
        TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>=', symbols=['x1', 'x2']),
        TwoDimensionalLinearConstraint(a=0, b=1, c=0, sign='>=', symbols=['x1', 'x2']),
    ]
    
    objective2 = {'x1': 4, 'x2': 1}
    
    plotter2 = TwoDimensionalLinearPlotter(
        constraints=constraints2,
        objective_coefficients=objective2,
        plot_bounds=(0, 8, 0, 6)  # Custom bounds
    )
    
    # Customize with different line styles
    plotter2.set_constraint_line_style(line_style='--', line_width=3)
    plotter2.set_objective_gradient_style(color='orange', width=4, alpha=0.9)
    
    plotter2.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,
        show_objective_gradient=True,
        show_optimal_point=True,  # NEW: Show optimal point
        maximize_objective=True,  # NEW: Maximize the objective
        constraint_color_mode='multiple',
        save_path='plots/example2_custom_symbols_with_optimal',
        save_format='pgf',  # Save as PGF for LaTeX
        show_plot=False
    )
    
    # Example 3: Infeasible problem
    print("\nCreating Example 3: Infeasible Problem")
    
    # Contradictory constraints:
    # x + y <= 1
    # x + y >= 3
    constraints3 = [
        TwoDimensionalLinearConstraint(a=1, b=1, c=1, sign='<='),
        TwoDimensionalLinearConstraint(a=1, b=1, c=3, sign='>='),
    ]
    
    plotter3 = TwoDimensionalLinearPlotter(constraints=constraints3)
    
    plotter3.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,
        show_infeasible_intersections=True,
        constraint_color_mode='multiple',
        save_path='plots/example3_infeasible',
        save_format='png',
        show_plot=False
    )
    
    # Add example with customized optimal point styling
    print("\nCreating Example 4: Customized Optimal Point Styling")
    
    # Create a simple problem for demonstration
    constraints4 = [
        TwoDimensionalLinearConstraint(a=1, b=1, c=6, sign='<='),
        TwoDimensionalLinearConstraint(a=2, b=1, c=8, sign='<='),
        TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>='),
        TwoDimensionalLinearConstraint(a=0, b=1, c=0, sign='>='),
    ]
    objective4 = {'x': 2, 'y': 3}
    
    plotter4 = TwoDimensionalLinearPlotter(constraints=constraints4, objective_coefficients=objective4)
    
    # Customize optimal point to be a large diamond
    plotter4.set_optimal_point_style(
        color='red',
        marker='D',  # Diamond marker
        size=150,    # Even larger
        edge_color='black',
        edge_width=3
    )
    
    plotter4.plot(
        show_feasible_region=True,
        show_optimal_point=True,
        show_objective_gradient=True,
        constraint_color_mode='multiple',
        save_path='plots/example4_custom_optimal_styling',
        save_format='png',
        show_plot=False
    )
    
    print("\nAll examples completed! Check the generated files in the 'plots/' directory:")
    print("- plots/example1_single_color_with_optimal_and_title.png")
    print("- plots/example1_multiple_colors_minimize.png") 
    print("- plots/example2_custom_symbols_with_optimal.pgf")
    print("- plots/example3_infeasible.png")
    print("- plots/example4_custom_optimal_styling.png")
    print("\nNOTE: Optimal point plotting requires CVXPY. Install with: pip install cvxpy")


if __name__ == "__main__":
    main()