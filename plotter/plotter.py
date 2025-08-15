import matplotlib.pyplot as plt
import matplotlib
# Enable LaTeX rendering for mathematical notation
matplotlib.rcParams['text.usetex'] = False  # Use mathtext instead of LaTeX for better compatibility
matplotlib.rcParams['mathtext.default'] = 'regular'
import numpy as np
from typing import List, Dict, Optional, Tuple, Literal
import os
from itertools import combinations
from matplotlib.patches import Polygon
import cvxpy as cp

from .two_dimensional_linear_constraint import TwoDimensionalLinearConstraint


class TwoDimensionalLinearPlotter:
    """
    A plotter for 2D linear constraints that can visualize:
    - Multiple linear constraints
    - Feasible region
    - Intersection points (feasible and infeasible)
    - Objective function gradient
    - Customizable styling and colors
    """
    
    def __init__(
        self,
        constraints: List[TwoDimensionalLinearConstraint],
        objective_coefficients: Optional[Dict[str, float]] = None,
        plot_bounds: Optional[Tuple[float, float, float, float]] = None,
        figsize: Tuple[float, float] = (10, 8)
    ):
        """
        Initialize the plotter.
        
        Args:
            constraints: List of 2D linear constraints to plot
            objective_coefficients: Dict mapping symbol names to coefficients for objective function
            plot_bounds: (x_min, x_max, y_min, y_max) bounds for plotting. If None, auto-calculated
            figsize: Figure size as (width, height)
        """
        self.constraints = constraints
        self.objective_coefficients = objective_coefficients
        self.plot_bounds = plot_bounds
        self.figsize = figsize
        
        # Validate inputs
        self._validate_constraints()
        if objective_coefficients:
            self._validate_objective_coefficients()
        
        # Initialize plot parameters
        self.fig = None
        self.ax = None
        
        # Default styling options
        self.constraint_colors = None  # Will be set based on mode
        self.feasible_region_color = 'lightblue'
        self.feasible_region_alpha = 0.3
        self.feasible_intersection_color = 'green'
        self.infeasible_intersection_color = 'orange'  # Changed from red to avoid confusion with optimal point
        self.intersection_marker = 'o'
        self.intersection_size = 50
        self.constraint_line_style = '-'
        self.constraint_line_width = 2
        self.gradient_color = 'purple'
        self.gradient_width = 3
        self.gradient_alpha = 0.8
        
        # Optimal point styling - distinctive red color and larger size
        self.optimal_point_color = 'red'
        self.optimal_point_marker = 'o'
        self.optimal_point_size = 120  # Larger than intersection points
        self.optimal_point_edge_color = 'darkred'
        self.optimal_point_edge_width = 2
    
    def _validate_constraints(self):
        """Validate that all constraints have consistent symbols."""
        if not self.constraints:
            raise ValueError("At least one constraint is required")
        
        # Check symbol consistency
        first_symbols = self.constraints[0].symbols
        for i, constraint in enumerate(self.constraints[1:], 1):
            if constraint.symbols != first_symbols:
                raise ValueError(f"Constraint {i} has symbols {constraint.symbols}, "
                               f"but constraint 0 has symbols {first_symbols}. "
                               f"All constraints must have the same symbols.")
    
    def _validate_objective_coefficients(self):
        """Validate that objective coefficients match constraint symbols."""
        if not self.constraints:
            return
            
        constraint_symbols = set(self.constraints[0].symbols)
        objective_symbols = set(self.objective_coefficients.keys())
        
        if constraint_symbols != objective_symbols:
            raise ValueError(f"Objective coefficients symbols {objective_symbols} "
                           f"must match constraint symbols {constraint_symbols}")
    
    def _calculate_plot_bounds(self) -> Tuple[float, float, float, float]:
        """Calculate appropriate plot bounds based on constraints."""
        if self.plot_bounds:
            return self.plot_bounds
        
        # Find intersection points to get a sense of scale
        intersections = self._find_all_intersections()
        
        if intersections:
            x_coords = [point[0] for point in intersections if point is not None]
            y_coords = [point[1] for point in intersections if point is not None]
            
            if x_coords and y_coords:
                x_min, x_max = min(x_coords), max(x_coords)
                y_min, y_max = min(y_coords), max(y_coords)
                
                # Add some padding
                x_range = x_max - x_min if x_max != x_min else 10
                y_range = y_max - y_min if y_max != y_min else 10
                padding = max(x_range, y_range) * 0.2
                
                return (x_min - padding, x_max + padding, 
                       y_min - padding, y_max + padding)
        
        # Default bounds if no intersections found
        return (-10, 10, -10, 10)
    
    def _find_all_intersections(self) -> List[Optional[Tuple[float, float]]]:
        """Find all intersection points between constraint pairs."""
        intersections = []
        
        for constraint1, constraint2 in combinations(self.constraints, 2):
            try:
                intersection = constraint1.intersection(constraint2)
                intersections.append(intersection)
            except ValueError:
                # Parallel lines
                intersections.append(None)
        
        return intersections
    
    def _is_point_feasible(self, x: float, y: float) -> bool:
        """Check if a point satisfies all constraints."""
        return all(constraint.satisfies(x, y) for constraint in self.constraints)
    
    def _get_feasible_intersections(self) -> List[Tuple[float, float]]:
        """Get all intersection points that are feasible."""
        intersections = self._find_all_intersections()
        feasible = []
        
        for point in intersections:
            if point and self._is_point_feasible(point[0], point[1]):
                feasible.append(point)
        
        return feasible
    
    def _get_infeasible_intersections(self) -> List[Tuple[float, float]]:
        """Get all intersection points that are infeasible."""
        intersections = self._find_all_intersections()
        infeasible = []
        
        for point in intersections:
            if point and not self._is_point_feasible(point[0], point[1]):
                infeasible.append(point)
        
        return infeasible
    
    def solve_lp(self, maximize: bool = True) -> Optional[Tuple[float, float, float]]:
        """
        Solve the linear programming problem using CVXPY.
        
        Args:
            maximize: If True, maximize the objective. If False, minimize.
            
        Returns:
            Tuple of (x_optimal, y_optimal, optimal_value) if solution exists,
            None if problem is infeasible or unbounded
            
        Raises:
            ValueError: If no objective coefficients are provided
        """
        if not self.objective_coefficients:
            raise ValueError("Objective coefficients must be provided to solve LP")
        
        # Get symbols from constraints
        symbols = self.constraints[0].symbols
        
        # Create CVXPY variables
        x = cp.Variable(name=symbols[0])
        y = cp.Variable(name=symbols[1])
        variables = {symbols[0]: x, symbols[1]: y}
        
        # Build constraints
        constraints = []
        for constraint in self.constraints:
            # Build left-hand side of constraint
            lhs = constraint.a * variables[symbols[0]] + constraint.b * variables[symbols[1]]
            
            # Add constraint based on sign
            if constraint.sign == '<=':
                constraints.append(lhs <= constraint.c)
            elif constraint.sign == '>=':
                constraints.append(lhs >= constraint.c)
            elif constraint.sign == '==':
                constraints.append(lhs == constraint.c)
        
        # Build objective function
        obj_expr = (self.objective_coefficients[symbols[0]] * variables[symbols[0]] + 
                   self.objective_coefficients[symbols[1]] * variables[symbols[1]])
        
        if maximize:
            objective = cp.Maximize(obj_expr)
        else:
            objective = cp.Minimize(obj_expr)
        
        # Create and solve problem
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve()
            
            if problem.status == cp.OPTIMAL:
                x_opt = float(x.value)
                y_opt = float(y.value)
                obj_value = float(problem.value)
                return (x_opt, y_opt, obj_value)
            else:
                # Problem is infeasible, unbounded, or solver failed
                print(f"LP Problem status: {problem.status}")
                return None
                
        except Exception as e:
            print(f"Error solving LP: {e}")
            return None
    
    def get_optimal_point(self, maximize: bool = True) -> Optional[Tuple[float, float]]:
        """
        Get the optimal point coordinates.
        
        Args:
            maximize: If True, maximize the objective. If False, minimize.
            
        Returns:
            Tuple of (x_optimal, y_optimal) if solution exists, None otherwise
        """
        result = self.solve_lp(maximize)
        if result:
            return (result[0], result[1])
        return None
    
    def _calculate_feasible_region_vertices(self) -> List[Tuple[float, float]]:
        """Calculate vertices of the feasible region polygon."""
        x_min, x_max, y_min, y_max = self._calculate_plot_bounds()
        
        # Start with a large rectangle representing the plotting area
        boundary_points = [
            (x_min, y_min), (x_max, y_min), 
            (x_max, y_max), (x_min, y_max)
        ]
        
        # Add all feasible intersection points
        feasible_intersections = self._get_feasible_intersections()
        
        # Combine boundary points and intersection points
        candidate_points = boundary_points + feasible_intersections
        
        # Add points where constraints intersect the boundary
        for constraint in self.constraints:
            try:
                # Intersection with left boundary (x = x_min)
                y_left = constraint.y_given_x(x_min)
                if y_min <= y_left <= y_max:
                    candidate_points.append((x_min, y_left))
                
                # Intersection with right boundary (x = x_max)
                y_right = constraint.y_given_x(x_max)
                if y_min <= y_right <= y_max:
                    candidate_points.append((x_max, y_right))
            except ValueError:
                pass  # Vertical line
            
            try:
                # Intersection with bottom boundary (y = y_min)
                x_bottom = constraint.x_given_y(y_min)
                if x_min <= x_bottom <= x_max:
                    candidate_points.append((x_bottom, y_min))
                
                # Intersection with top boundary (y = y_max)
                x_top = constraint.x_given_y(y_max)
                if x_min <= x_top <= x_max:
                    candidate_points.append((x_top, y_max))
            except ValueError:
                pass  # Horizontal line
        
        # Filter to only feasible points
        feasible_points = [
            point for point in candidate_points 
            if self._is_point_feasible(point[0], point[1])
        ]
        
        if not feasible_points:
            return []
        
        # Remove duplicate points
        unique_points = []
        for point in feasible_points:
            is_duplicate = False
            for existing in unique_points:
                if (abs(point[0] - existing[0]) < 1e-10 and 
                    abs(point[1] - existing[1]) < 1e-10):
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_points.append(point)
        
        # Sort points to form a proper polygon (convex hull)
        if len(unique_points) >= 3:
            return self._convex_hull(unique_points)
        else:
            return unique_points
    
    def _convex_hull(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Calculate convex hull of points using Graham scan algorithm."""
        def cross_product(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Find the bottom-most point (and leftmost in case of tie)
        start = min(points, key=lambda p: (p[1], p[0]))
        
        # Sort points by polar angle with respect to start point
        def polar_angle(p):
            dx, dy = p[0] - start[0], p[1] - start[1]
            return np.arctan2(dy, dx)
        
        sorted_points = sorted([p for p in points if p != start], key=polar_angle)
        
        # Build convex hull
        hull = [start]
        
        for point in sorted_points:
            # Remove points that create a right turn
            while (len(hull) > 1 and 
                   cross_product(hull[-2], hull[-1], point) <= 0):
                hull.pop()
            hull.append(point)
        
        return hull
    
    def _format_symbol_latex(self, symbol: str) -> str:
        """Convert symbol to LaTeX format."""
        # Handle subscripts like x1 -> x_1, x2 -> x_2, etc.
        if len(symbol) > 1 and symbol[-1].isdigit():
            base = symbol[:-1]
            subscript = symbol[-1]
            return f"${base}_{{{subscript}}}$"
        else:
            return f"${symbol}$"
    
    def _format_objective_function(self, maximize: bool = True) -> str:
        """Format the objective function as a string with LaTeX notation."""
        if not self.objective_coefficients:
            return ""
        
        symbols = self.constraints[0].symbols if self.constraints else ['x', 'y']
        terms = []
        
        for symbol in symbols:
            coeff = self.objective_coefficients.get(symbol, 0)
            if coeff == 0:
                continue
            
            latex_symbol = self._format_symbol_latex(symbol)
            
            if coeff == 1:
                if not terms:  # First term
                    terms.append(latex_symbol)
                else:
                    terms.append(f" + {latex_symbol}")
            elif coeff == -1:
                if not terms:  # First term
                    terms.append(f"-{latex_symbol}")
                else:
                    terms.append(f" - {latex_symbol}")
            else:
                if not terms:  # First term
                    terms.append(f"{coeff}{latex_symbol}")
                else:
                    if coeff > 0:
                        terms.append(f" + {coeff}{latex_symbol}")
                    else:
                        terms.append(f" - {abs(coeff)}{latex_symbol}")
        
        if not terms:
            return "f() = 0"
        
        objective_str = "".join(terms)
        optimization_type = "maximize" if maximize else "minimize"
        
        # Format function notation with LaTeX
        latex_symbols = [self._format_symbol_latex(s) for s in symbols]
        function_notation = f"f({', '.join(latex_symbols)})"
        
        return f"{optimization_type} {function_notation} = {objective_str}"
    
    def plot(
        self,
        show_feasible_region: bool = True,
        show_feasible_intersections: bool = False,
        show_infeasible_intersections: bool = False,
        show_objective_gradient: bool = False,
        show_optimal_point: bool = False,
        maximize_objective: bool = True,
        constraint_color_mode: Literal['single', 'multiple'] = 'single',
        constraint_single_color: str = 'black',
        title: Optional[str] = None,
        show_objective_function: bool = True,
        save_path: Optional[str] = None,
        save_format: Literal['png', 'pgf'] = 'png',
        show_plot: bool = True
    ):
        """
        Create the complete plot with all specified features.
        
        Args:
            show_feasible_region: Whether to fill and show the feasible region
            show_feasible_intersections: Whether to plot feasible intersection points
            show_infeasible_intersections: Whether to plot infeasible intersection points
            show_objective_gradient: Whether to plot the objective function gradient
            show_optimal_point: Whether to solve LP and plot the optimal point (requires CVXPY)
            maximize_objective: If True, maximize objective; if False, minimize
            constraint_color_mode: 'single' for all black with one legend entry, 
                                  'multiple' for different colors with individual labels
            constraint_single_color: Color to use when constraint_color_mode='single'
            title: Custom title for the plot. If None, auto-generates title
            show_objective_function: Whether to display the objective function in the title
            save_path: Optional path to save the plot (without extension)
            save_format: File format for saving ('png' or 'pgf')
            show_plot: Whether to display the plot
        """
        # Initialize the plot
        self.fig, self.ax = plt.subplots(figsize=self.figsize)
        
        # Calculate plot bounds
        x_min, x_max, y_min, y_max = self._calculate_plot_bounds()
        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(y_min, y_max)
        
        # Plot feasible region first (so it's behind everything else)
        if show_feasible_region:
            self._plot_feasible_region()
        
        # Plot constraints
        self._plot_constraints(constraint_color_mode, constraint_single_color)
        
        # Plot intersection points
        if show_feasible_intersections:
            self._plot_feasible_intersections()
        
        if show_infeasible_intersections:
            self._plot_infeasible_intersections()
        
        # Plot objective gradient
        if show_objective_gradient and self.objective_coefficients:
            self._plot_objective_gradient()
        
        # Plot optimal point
        if show_optimal_point and self.objective_coefficients:
            try:
                self._plot_optimal_point(maximize_objective)
            except ImportError as e:
                print(f"Warning: Could not plot optimal point - {e}")
            except Exception as e:
                print(f"Warning: Error plotting optimal point - {e}")
        
        # Set labels and formatting
        symbols = self.constraints[0].symbols if self.constraints else ['x', 'y']
        self.ax.set_xlabel(self._format_symbol_latex(symbols[0]), fontsize=12)
        self.ax.set_ylabel(self._format_symbol_latex(symbols[1]), fontsize=12)
        self.ax.grid(True, alpha=0.3)
        self.ax.legend()
        
        # Set title
        if title is not None:
            # Use custom title
            plot_title = title
        else:
            # Auto-generate title
            plot_title = 'Linear Programming Problem'
        
        # Add objective function to title if requested
        if show_objective_function and self.objective_coefficients:
            objective_str = self._format_objective_function(maximize_objective)
            if title is not None:
                plot_title = f"{plot_title}\n{objective_str}"
            else:
                plot_title = f"{plot_title}\n{objective_str}"
        
        self.ax.set_title(plot_title, fontsize=14, pad=20)
        
        # Save if requested
        if save_path:
            self._save_plot(save_path, save_format)
        
        # Show plot
        if show_plot:
            plt.show()
    
    def _plot_feasible_region(self):
        """Plot the feasible region as a filled polygon."""
        vertices = self._calculate_feasible_region_vertices()
        
        if len(vertices) >= 3:
            polygon = Polygon(
                vertices, 
                alpha=self.feasible_region_alpha,
                facecolor=self.feasible_region_color,
                edgecolor='none',
                label='Feasible Region'
            )
            self.ax.add_patch(polygon)
    
    def _plot_constraints(self, color_mode: str, single_color: str):
        """Plot constraint lines."""
        x_min, x_max, y_min, y_max = self._calculate_plot_bounds()
        
        if color_mode == 'single':
            colors = [single_color] * len(self.constraints)
            labels = ['Constraints'] + [None] * (len(self.constraints) - 1)
        else:
            # Generate different colors for each constraint
            colors = plt.cm.tab10(np.linspace(0, 1, len(self.constraints)))
            labels = [str(constraint) for constraint in self.constraints]
        
        for i, constraint in enumerate(self.constraints):
            x_vals = np.linspace(x_min, x_max, 1000)
            y_vals = []
            
            try:
                y_vals = [constraint.y_given_x(x) for x in x_vals]
            except ValueError:
                # Vertical line
                if abs(constraint.a) > 1e-10:
                    x_const = constraint.c / constraint.a
                    if x_min <= x_const <= x_max:
                        self.ax.axvline(
                            x=x_const, 
                            color=colors[i],
                            linestyle=self.constraint_line_style,
                            linewidth=self.constraint_line_width,
                            label=labels[i]
                        )
                continue
            
            # Filter y values within plot bounds
            valid_points = [(x, y) for x, y in zip(x_vals, y_vals) 
                          if y_min <= y <= y_max]
            
            if valid_points:
                x_plot, y_plot = zip(*valid_points)
                self.ax.plot(
                    x_plot, y_plot,
                    color=colors[i],
                    linestyle=self.constraint_line_style,
                    linewidth=self.constraint_line_width,
                    label=labels[i]
                )
    
    def _plot_feasible_intersections(self):
        """Plot feasible intersection points."""
        feasible_points = self._get_feasible_intersections()
        
        if feasible_points:
            x_coords, y_coords = zip(*feasible_points)
            self.ax.scatter(
                x_coords, y_coords,
                c=self.feasible_intersection_color,
                marker=self.intersection_marker,
                s=self.intersection_size,
                label='Feasible Intersections',
                zorder=5,
                edgecolors='black',
                linewidth=1
            )
    
    def _plot_infeasible_intersections(self):
        """Plot infeasible intersection points."""
        infeasible_points = self._get_infeasible_intersections()
        
        if infeasible_points:
            x_coords, y_coords = zip(*infeasible_points)
            self.ax.scatter(
                x_coords, y_coords,
                c=self.infeasible_intersection_color,
                marker=self.intersection_marker,
                s=self.intersection_size,
                label='Infeasible Intersections',
                zorder=5,
                edgecolors='black',
                linewidth=1
            )
    
    def _plot_optimal_point(self, maximize: bool = True):
        """Plot the optimal point found by solving the LP."""
        if not self.objective_coefficients:
            return
        
        optimal_point = self.get_optimal_point(maximize)
        
        if optimal_point:
            x_opt, y_opt = optimal_point
            
            # Get the optimal value for the label
            result = self.solve_lp(maximize)
            if result:
                optimal_value = result[2]
                optimization_type = "max" if maximize else "min"
                label = f'Optimal Point ({optimization_type}: {optimal_value:.3f})'
            else:
                label = 'Optimal Point'
            
            self.ax.scatter(
                [x_opt], [y_opt],
                c=self.optimal_point_color,
                marker=self.optimal_point_marker,
                s=self.optimal_point_size,
                label=label,
                zorder=10,  # Highest z-order so it appears on top
                edgecolors=self.optimal_point_edge_color,
                linewidth=self.optimal_point_edge_width
            )
            
            # Add coordinates annotation
            self.ax.annotate(
                f'({x_opt:.3f}, {y_opt:.3f})',
                xy=(x_opt, y_opt),
                xytext=(10, 10),  # Offset from point
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                fontsize=10,
                ha='left'
            )
    
    def _plot_objective_gradient(self):
        """Plot the objective function gradient as an arrow."""
        if not self.objective_coefficients:
            return
        
        symbols = self.constraints[0].symbols
        coeff_x = self.objective_coefficients.get(symbols[0], 0)
        coeff_y = self.objective_coefficients.get(symbols[1], 0)
        
        # Plot arrow from origin
        self.ax.annotate(
            '', 
            xy=(coeff_x, coeff_y), 
            xytext=(0, 0),
            arrowprops=dict(
                arrowstyle='->',
                color=self.gradient_color,
                lw=self.gradient_width,
                alpha=self.gradient_alpha
            )
        )
        
        # Add a legend entry for the gradient (using a line instead of annotation)
        self.ax.plot([], [], color=self.gradient_color, linewidth=self.gradient_width, 
                    label='Objective Gradient')
    
    def _save_plot(self, path: str, format_type: str):
        """Save the plot to file."""
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        
        # Add extension if not present
        if not path.endswith(f'.{format_type}'):
            path = f"{path}.{format_type}"
        
        if format_type == 'pgf':
            plt.savefig(path, format='pgf', bbox_inches='tight', dpi=300)
        else:
            plt.savefig(path, format='png', bbox_inches='tight', dpi=300)
        
        print(f"Plot saved to: {path}")
    
    # Customization methods
    def set_feasible_region_style(self, color: str, alpha: float = 0.3):
        """Set feasible region color and transparency."""
        self.feasible_region_color = color
        self.feasible_region_alpha = alpha
    
    def set_intersection_style(
        self, 
        feasible_color: str = 'green',
        infeasible_color: str = 'red',
        marker: str = 'o',
        size: float = 50
    ):
        """Set intersection point styling."""
        self.feasible_intersection_color = feasible_color
        self.infeasible_intersection_color = infeasible_color
        self.intersection_marker = marker
        self.intersection_size = size
    
    def set_constraint_line_style(self, line_style: str = '-', line_width: float = 2):
        """Set constraint line styling."""
        self.constraint_line_style = line_style
        self.constraint_line_width = line_width
    
    def set_objective_gradient_style(
        self, 
        color: str = 'purple',
        width: float = 3,
        alpha: float = 0.8
    ):
        """Set objective gradient arrow styling."""
        self.gradient_color = color
        self.gradient_width = width
        self.gradient_alpha = alpha
    
    def set_optimal_point_style(
        self,
        color: str = 'red',
        marker: str = 'o',
        size: float = 120,
        edge_color: str = 'darkred',
        edge_width: float = 2
    ):
        """Set optimal point styling."""
        self.optimal_point_color = color
        self.optimal_point_marker = marker
        self.optimal_point_size = size
        self.optimal_point_edge_color = edge_color
        self.optimal_point_edge_width = edge_width
    
    def _save_plot(self, path: str, format_type: str):
        """Save the plot to file."""
        # Ensure directory exists
        directory = os.path.dirname(path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        
        # Add extension if not present
        if not path.endswith(f'.{format_type}'):
            path = f"{path}.{format_type}"
        
        try:
            if format_type == 'pgf':
                plt.savefig(path, format='pgf', bbox_inches='tight', dpi=300)
            else:
                plt.savefig(path, format='png', bbox_inches='tight', dpi=300)
            
            print(f"Plot saved to: {path}")
        except Exception as e:
            print(f"Error saving plot: {e}")