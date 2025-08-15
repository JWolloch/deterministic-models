import numpy as np
import itertools
from numpy import linalg as la

class BasicFeasibleSolutions:
    """
    Class to find Basic Feasible Solutions (BFS) and determine the optimal BFS
    for a given linear programming problem.

    Attributes:
        basicFeasibleSolutions (list): List to store all basic feasible solutions found.
        optimalSolutions (list): List to store optimal solutions among basic feasible solutions.
        bestValue (float): The optimal value achieved by the linear programming problem.
    """

    def __init__(self):
        """
        Initializes BasicFeasibleSolutions with empty lists for solutions and None for bestValue.
        """
        self.basicFeasibleSolutions = []
        self.optimalSolutions = []
        self.bestValue = None


    def BFS(self, A: np.array, b: np.array):
        """
        Finds all Basic Feasible Solutions (BFS) for the given linear system Ax = b.

        Args:
            A (np.array): Coefficient matrix of the linear system.
            b (np.array): Column vector representing the right-hand side of the linear system.

        Returns:
            None
        """
        n = A.shape[1]
        m = A.shape[0]
        for basis in itertools.combinations(range(n), m):
            B = A[:, list(basis)]
            if la.det(B) != 0:
                x = la.solve(B, b)
                if np.all(x >= 0):
                    solution = np.zeros(n)
                    solution[list(basis)] = x.flatten()
                    self.basicFeasibleSolutions.append(solution)


    def optimalBFS(self, c: np.array):
        """
        Determines the optimal basic feasible solutions among all found solutions
        based on the given objective function coefficients.

        Args:
            c (np.array): Coefficients of the objective function.

        Returns:
            None
        """
        values = np.dot(np.array(self.basicFeasibleSolutions), c)
        max_indices = np.where(values == np.max(values))[0]
        self.optimalSolutions = [self.basicFeasibleSolutions[idx] for idx in max_indices]

    def optimalValue(self, c: np.array):
        """
        Computes the optimal value of the objective function for the linear programming problem.

        Args:
            c (np.array): Coefficients of the objective function.

        Returns:
            None
        """
        self.bestValue = np.max(np.dot(np.array(self.basicFeasibleSolutions), c))

    def getBasicFeasibleSolutions(self):
        """
        Returns all basic feasible solutions found.

        Returns:
            np.array: Array containing all basic feasible solutions.
        """
        return np.array(self.basicFeasibleSolutions)

    def getOptimalBFS(self):
        """
        Returns the optimal basic feasible solutions.

        Returns:
            np.array: Array containing the optimal basic feasible solutions.
        """
        return np.array(self.optimalSolutions)

    def getOptimalValue(self):
        """
        Returns the optimal value of the objective function.

        Returns:
            float: The optimal value of the objective function.
        """
        return self.bestValue

    def getAll(self):
        """
        Returns all basic feasible solutions, optimal basic feasible solutions, and the optimal value.

        Returns:
            tuple: A tuple containing all basic feasible solutions, optimal basic feasible solutions,
                   and the optimal value.
        """
        return np.array(self.basicFeasibleSolutions), np.array(self.optimalSolutions), self.bestValue

if __name__ == "__main__":
    # Example usage
    '''A = np.array([[2, 5, 1, -1, 0], [3, 2, 0, 0, -1]])
    b = np.array([[5], [6]])
    c = np.array([[-3], [-1], [-4], [0], [0]])'''

    c1, c2, c3 = 1, 1, 1
    b1, b2 = 4, 2.5

    c = np.array([[c1], [c2], [c3], [0], [0]])
    b = np.array([[b1], [-b2]])
    A = np.array([[2, 0, 3, 1, 0], [-1, 1, -2, 0, 1]])

    bfs_solver = BasicFeasibleSolutions()
    bfs_solver.BFS(A, b)
    bfs = bfs_solver.getBasicFeasibleSolutions()

    bfs_solver.optimalBFS(c)
    bfs_solver.optimalValue(c)

    basicFeasibleSolutions, optimalBFS, optVal = bfs_solver.getAll()

    print("Basic Feasible Solutions:")
    for bfs_solution in basicFeasibleSolutions:
        print(bfs_solution)

    print("\nOptimal Basic Feasible Solutions:")
    for optimal_bfs_solution in optimalBFS:
        print(optimal_bfs_solution)

    print("\nOptimal Value:")
    print(optVal)