#!/usr/bin/env python3
"""
Simple test script to verify the plotter package works correctly.
"""

from plotter.two_dimensional_linear_constraint import TwoDimensionalLinearConstraint
from plotter.plotter import TwoDimensionalLinearPlotter

def test_basic_functionality():
    print("Testing basic plotter functionality...")
    
    # Create simple constraints
    constraints = [
        TwoDimensionalLinearConstraint(a=1, b=1, c=5, sign='<='),
        TwoDimensionalLinearConstraint(a=1, b=0, c=0, sign='>='),
        TwoDimensionalLinearConstraint(a=0, b=1, c=0, sign='>='),
    ]
    
    # Create objective
    objective = {'x': 1, 'y': 1}
    
    # Create plotter
    plotter = TwoDimensionalLinearPlotter(
        constraints=constraints,
        objective_coefficients=objective
    )
    
    print("✓ Plotter created successfully")
    
    # Test string representation
    for i, constraint in enumerate(constraints):
        print(f"✓ Constraint {i+1}: {constraint}")
    
    # Test plot method (without showing)
    try:
        plotter.plot(
            show_feasible_region=True,
            show_optimal_point=True,
            title="Test Plot",
            save_path="test_output",
            save_format="png",
            show_plot=False
        )
        print("✓ Plot method executed successfully")
        print("✓ Test file should be saved as 'test_output.png'")
    except Exception as e:
        print(f"✗ Error during plotting: {e}")
    
    print("\nTest completed!")

if __name__ == "__main__":
    test_basic_functionality()
