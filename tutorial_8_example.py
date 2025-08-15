#!/usr/bin/env python3
"""
Tutorial 8 Example using the TwoDimensionalLinearPlotter

This recreates the linear programming problem from tutorial_8.ipynb using our plotter class.
Based on the constraints, feasible points, and objective function visible in the notebook.
"""

import sys
import os
# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from plotter.two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from plotter.plotter import TwoDimensionalLinearPlotter


def main():
    print("Creating Tutorial 8 Linear Programming Problem")
    
    # Analyzing the original more carefully:
    # Known feasible points: [(18/7≈2.57,23/7≈3.29), (3,0), (0,2), (0,0), (6,1)]
    # The optimal point is (6,1) and objective is x1 + x2 = 7
    
    # From the lines in original code:
    # y1 = 0.5*x + 2        => x - 2y + 4 = 0    => x - 2y >= -4 (above line)
    # y2 = -0.66666*x + 5   => 2x + 3y - 15 = 0  => 2x + 3y <= 15 (below line)  
    # y3 = 0.33333*x - 1    => x - 3y + 3 = 0    => x - 3y <= -3 (below line)
    
    # Let me work backwards from the known feasible points to get the right constraints:
    # For (6,1) to be optimal and feasible, and (3,0), (0,2), (0,0), (18/7,23/7) to be feasible:
    
    constraints = [
        # From y = 0.5x + 2: x - 2y = -4, feasible region is x - 2y >= -4
        TwoDimensionalLinearConstraint(a=1, b=-2, c=-4, sign='>=', symbols=['x1', 'x2']),
        
        # From y = -2/3*x + 5: 2x + 3y = 15, feasible region is 2x + 3y <= 15
        TwoDimensionalLinearConstraint(a=2, b=3, c=15, sign='<=', symbols=['x1', 'x2']),
        
        # From y = 1/3*x - 1: x - 3y = 3, feasible region is x - 3y <= 3  
        TwoDimensionalLinearConstraint(a=1, b=-3, c=3, sign='<=', symbols=['x1', 'x2']),
        
        # Non-negativity constraints
        TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>=', symbols=['x1', 'x2']),  # x >= 0
        TwoDimensionalLinearConstraint(a=0, b=1, c=0, sign='>=', symbols=['x1', 'x2']),  # y >= 0
    ]
    
    # From the contour lines (c = -x + constant), the objective appears to be: x + y (maximize)
    # The gradient vector is (2, 2) which represents direction of steepest increase
    # This confirms the objective is to maximize x + y
    objective = {'x1': 1, 'x2': 1}
    
    # Verify constraints with known feasible points
    print("Verifying constraints with known feasible points:")
    test_points = [(18/7, 23/7), (3, 0), (0, 2), (0, 0), (6, 1)]
    
    for i, (x, y) in enumerate(test_points):
        print(f"Point {i+1}: ({x:.3f}, {y:.3f})")
        for j, constraint in enumerate(constraints):
            is_satisfied = constraint.satisfies(x, y)
            print(f"  Constraint {j+1}: {constraint} -> {'✓' if is_satisfied else '✗'}")
        print()
    
    # Create plotter with custom bounds to match the original plot
    plotter = TwoDimensionalLinearPlotter(
        constraints=constraints,
        objective_coefficients=objective,
        plot_bounds=(-0.2, 8, -0.2, 6),  # Match the xlim and ylim from original
        figsize=(10, 8)
    )
    
    # Customize styling to match the original look
    plotter.set_feasible_region_style(color='dodgerblue', alpha=0.1)
    plotter.set_optimal_point_style(
        color='crimson',
        marker='*',  # Star marker like in original
        size=150,    # Size 150 like in original
        edge_color='darkred',
        edge_width=1
    )
    plotter.set_objective_gradient_style(color='navy', width=2, alpha=0.8)
    
    # Plot with all features
    plotter.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,  # Show the basic feasible solutions
        show_infeasible_intersections=False,
        show_objective_gradient=True,
        show_optimal_point=True,
        maximize_objective=True,
        constraint_color_mode='single',  # All constraints in black like original
        constraint_single_color='black',
        title='Tutorial 8: Linear Programming Problem',
        show_objective_function=True,
        save_path='plots/tutorial_8_recreation',
        save_format='png',
        show_plot=False  # Don't show to avoid hanging
    )
    
    # Also save as PGF like the original
    plotter.plot(
        show_feasible_region=True,
        show_feasible_intersections=True,
        show_infeasible_intersections=False,
        show_objective_gradient=True,
        show_optimal_point=True,
        maximize_objective=True,
        constraint_color_mode='single',
        constraint_single_color='black',
        title='Tutorial 8: Linear Programming Problem',
        show_objective_function=True,
        save_path='plots/tutorial_8_recreation',
        save_format='pgf',
        show_plot=False
    )
    
    # Print the solution
    result = plotter.solve_lp(maximize=True)
    if result:
        x1_opt, x2_opt, obj_value = result
        print(f"\nOptimal Solution:")
        print(f"x1* = {x1_opt:.3f}")
        print(f"x2* = {x2_opt:.3f}")
        print(f"Optimal Value = {obj_value:.3f}")
        print(f"Point: ({x1_opt:.2f}, {x2_opt:.2f})")
        
        # Verify this matches the expected point (6, 1)
        expected_x, expected_y = 6, 1
        if abs(x1_opt - expected_x) < 0.1 and abs(x2_opt - expected_y) < 0.1:
            print(f"✓ Solution matches expected optimal point ({expected_x}, {expected_y})")
        else:
            print(f"⚠ Solution differs from expected point ({expected_x}, {expected_y})")
    else:
        print("Could not solve the LP problem")
    
    print(f"\nFiles saved:")
    print(f"- plots/tutorial_8_recreation.png")
    print(f"- plots/tutorial_8_recreation.pgf")
    
    # Print feasible points for verification
    print(f"\nFeasible intersection points found:")
    feasible_points = plotter._get_feasible_intersections()
    for i, point in enumerate(feasible_points):
        print(f"  Point {i+1}: ({point[0]:.3f}, {point[1]:.3f})")


if __name__ == "__main__":
    main()
