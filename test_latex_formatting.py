#!/usr/bin/env python3
"""
Test script to verify LaTeX formatting in the plotter.
"""

from plotter.two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from plotter.plotter import TwoDimensionalLinearPlotter

def test_latex_formatting():
    print("Testing LaTeX formatting...")
    
    # Test constraint string representations
    print("\n=== Constraint String Representations ===")
    
    # Test with subscripted symbols
    constraint1 = TwoDimensionalLinearConstraint(a=2, b=3, c=12, sign='<=', symbols=['x1', 'x2'])
    constraint2 = TwoDimensionalLinearConstraint(a=1, b=-1, c=5, sign='>=', symbols=['x1', 'x2'])
    constraint3 = TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>=', symbols=['x1', 'x2'])
    
    print(f"Constraint 1: {constraint1}")
    print(f"Constraint 2: {constraint2}")
    print(f"Constraint 3: {constraint3}")
    
    # Test with regular symbols
    constraint4 = TwoDimensionalLinearConstraint(a=1, b=2, c=6, sign='<=', symbols=['x', 'y'])
    constraint5 = TwoDimensionalLinearConstraint(a=0, b=1, c=3, sign='==', symbols=['x', 'y'])
    
    print(f"Constraint 4: {constraint4}")
    print(f"Constraint 5: {constraint5}")
    
    print("\n=== Objective Function Formatting ===")
    
    # Test objective function formatting
    constraints = [constraint1, constraint2, constraint3]
    objective = {'x1': 3, 'x2': 2}
    
    plotter = TwoDimensionalLinearPlotter(
        constraints=constraints,
        objective_coefficients=objective
    )
    
    # Test the objective function formatting
    obj_str_max = plotter._format_objective_function(maximize=True)
    obj_str_min = plotter._format_objective_function(maximize=False)
    
    print(f"Maximize objective: {obj_str_max}")
    print(f"Minimize objective: {obj_str_min}")
    
    print("\n=== Creating Test Plot ===")
    
    # Create a plot to test visual LaTeX rendering
    try:
        plotter.plot(
            show_feasible_region=True,
            show_optimal_point=True,
            show_objective_gradient=True,
            constraint_color_mode='multiple',
            title='LaTeX Formatting Test: $x_1$ and $x_2$ Variables',
            show_objective_function=True,
            save_path='plots/latex_test',
            save_format='png',
            show_plot=False
        )
        print("✓ LaTeX test plot created successfully!")
        print("✓ Check 'plots/latex_test.png' to see LaTeX formatting")
    except Exception as e:
        print(f"✗ Error creating plot: {e}")
    
    print("\nLaTeX formatting test completed!")

if __name__ == "__main__":
    test_latex_formatting()
