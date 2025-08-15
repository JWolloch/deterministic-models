#!/usr/bin/env python3
"""
HW1 Q3 Example using the TwoDimensionalLinearPlotter

This recreates the linear programming problem from the notebook using our plotter class.
Based on the constraints and objective function visible in the notebook.
"""

import sys
import os
# Add the parent directory (deterministic-models) to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from plotter.two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from plotter.plotter import TwoDimensionalLinearPlotter


def main():
    print("Creating HW1 Q3 Linear Programming Problem")
    
    # From the notebook, I can see these constraints:
    # -x1 + 2x2 = 7  (constraint line)
    # 6x1 + 3x2 = 40 (constraint line) 
    # -3x1 + 2x2 = -10 (constraint line)
    # x2 = 8 (horizontal line)
    # x1 = 2 (vertical line)
    
    # The feasible region seems to be bounded by these constraints
    # Converting to inequality constraints based on the feasible region shown:
    
    constraints = [
        # -x1 + 2x2 >= 7 (above the line)
        TwoDimensionalLinearConstraint(a=-1, b=2, c=7, sign='>=', symbols=['x1', 'x2']),
        
        # 6x1 + 3x2 <= 40 (below the line)
        TwoDimensionalLinearConstraint(a=6, b=3, c=40, sign='<=', symbols=['x1', 'x2']),
        
        # -3x1 + 2x2 >= -10 (above the line)
        TwoDimensionalLinearConstraint(a=-3, b=2, c=-10, sign='>=', symbols=['x1', 'x2']),
        
        # x2 <= 8 (below the horizontal line)
        TwoDimensionalLinearConstraint(a=0, b=1, c=8, sign='<=', symbols=['x1', 'x2']),
        
        # x1 >= 2 (to the right of the vertical line)
        TwoDimensionalLinearConstraint(a=1, b=0, c=2, sign='>=', symbols=['x1', 'x2']),
    ]
    
    # From the gradient arrow, the objective appears to be: x1 + 2x2 (maximize)
    # The gradient vector is (1, 2) which matches the arrow in the plot
    objective = {'x1': 1, 'x2': 2}
    
    # Create plotter with custom bounds to match the original plot
    plotter = TwoDimensionalLinearPlotter(
        constraints=constraints,
        objective_coefficients=objective,
        plot_bounds=(0, 9, 0, 9),  # Match the xlim and ylim from original
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
        show_feasible_intersections=False,  # Clean look like original
        show_infeasible_intersections=False,
        show_objective_gradient=True,
        show_optimal_point=True,
        maximize_objective=True,
        constraint_color_mode='multiple',  # Different colors for each constraint
        title='HW1 Q3: Linear Programming Problem',  # Custom title
        show_objective_function=True,  # Show objective function
        save_path='plots/hw1_q3_recreation',
        save_format='png',
        show_plot=True
    )
    
    # Also save as PGF like the original
    plotter.plot(
        show_feasible_region=True,
        show_feasible_intersections=False,
        show_infeasible_intersections=False,
        show_objective_gradient=True,
        show_optimal_point=True,
        maximize_objective=True,
        constraint_color_mode='multiple',
        title='HW1 Q3: Linear Programming Problem',  # Custom title
        show_objective_function=True,  # Show objective function
        save_path='plots/hw1_q3_recreation',
        save_format='pgf',
        show_plot=False
    )
    
    # Print the solution
    result = plotter.solve_lp(maximize=True)
    if result:
        x1_opt, x2_opt, obj_value = result
        print("\nOptimal Solution:")
        print(f"x1* = {x1_opt:.3f}")
        print(f"x2* = {x2_opt:.3f}")
        print(f"Optimal Value = {obj_value:.3f}")
        print(f"Point: ({x1_opt:.2f}, {x2_opt:.2f})")
    else:
        print("Could not solve the LP problem")
    
    print("\nAll examples completed! Check the generated files in the 'plots/' directory:")
    print("- plots/hw1_q3_recreation.png")
    print("- plots/hw1_q3_recreation.pgf")


if __name__ == "__main__":
    main()
