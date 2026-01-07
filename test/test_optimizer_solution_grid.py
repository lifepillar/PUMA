import unittest

from optimizer import SolutionGrid

from test.fixtures import DummyFeature, DummySolution


class DummySolutionGrid(unittest.TestCase):
    def setUp(self) -> None:
        # Create features for testing
        self.features = [
            DummyFeature(name="feature1", min_value=0.0, max_value=10.0),
            DummyFeature(name="feature2", min_value=0.0, max_value=10.0),
        ]
        self.shape = (5, 5)

    def test_solution_grid_initialization(self) -> None:
        """Test that SolutionGrid initializes correctly."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        self.assertEqual(grid.features, self.features)
        self.assertEqual(grid.search_grid.shape[0], 5)
        self.assertEqual(grid.search_grid.shape[1], 5)
        self.assertTrue(grid.empty())

    def test_solution_grid_empty(self) -> None:
        """Test that empty() returns True for new grid and False after adding solutions."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        self.assertTrue(grid.empty())

        # Add a solution
        solution = DummySolution(fitness=5.0)
        grid.add(solution)

        self.assertFalse(grid.empty())

    def test_solution_grid_add_and_retrieve(self) -> None:
        """Test adding solutions and retrieving them."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        # Add first solution
        solution1 = DummySolution(fitness=5.0)
        grid.add(solution1)

        self.assertEqual(len(grid.solutions()), 1)
        self.assertEqual(grid.solutions()[0], solution1)

        # Add second solution with higher fitness
        solution2 = DummySolution(fitness=10.0)
        grid.add(solution2)

        self.assertEqual(len(grid.solutions()), 2)
        self.assertEqual(grid.solutions()[1], solution2)

    def test_solution_grid_best_solution(self) -> None:
        """Test retrieving the best solution."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        # Add solutions with different fitness
        solution1 = DummySolution(fitness=5.0)
        solution2 = DummySolution(fitness=10.0)
        solution3 = DummySolution(fitness=7.5)

        grid.add(solution1)
        grid.add(solution2)
        grid.add(solution3)

        # Best solution should be the one with highest fitness
        best = grid.best_solution()
        self.assertEqual(best, solution2)

    def test_solution_grid_top_k(self) -> None:
        """Test retrieving top k solutions."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        # Add solutions with different fitness
        # Choose the fitness so that the three solutions end up in different cells
        solution1 = DummySolution(fitness=5.0)  # => cell(0,0)
        solution2 = DummySolution(fitness=20.0)  # => cell(1,2)
        solution3 = DummySolution(fitness=10.0)  # => cell(0, 1)

        grid.add(solution1)
        grid.add(solution2)
        grid.add(solution3)

        self.assertEqual(3, len(grid.elite))

        top_k = grid.top_k(1)

        self.assertEqual(1, len(top_k))
        self.assertTrue(top_k[0].solution is solution2)

        top_k = grid.top_k(2)

        self.assertEqual(2, len(top_k))
        self.assertTrue(top_k[0].solution is solution2)
        self.assertTrue(top_k[1].solution is solution3)

        top_k = grid.top_k(3)

        self.assertEqual(3, len(top_k))
        self.assertTrue(top_k[0].solution is solution2)
        self.assertTrue(top_k[1].solution is solution3)
        self.assertTrue(top_k[2].solution is solution1)

        top_k = grid.top_k(4)

        self.assertEqual(3, len(top_k))
        self.assertTrue(top_k[0].solution is solution2)
        self.assertTrue(top_k[1].solution is solution3)
        self.assertTrue(top_k[2].solution is solution1)

    def test_solution_grid_clear(self) -> None:
        """Test that clear() resets the solution grid."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        # Add some solutions
        solution1 = DummySolution(fitness=5.0)
        solution2 = DummySolution(fitness=10.0)
        grid.add(solution1)
        grid.add(solution2)

        self.assertFalse(grid.empty())
        self.assertEqual(len(grid.solutions()), 2)

        # Clear the grid
        grid.clear()

        self.assertTrue(grid.empty())
        self.assertEqual(len(grid.solutions()), 0)

    def test_solution_grid_feature_descriptor(self) -> None:
        """Test that feature descriptors are computed correctly."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        solution = DummySolution(fitness=8.0)
        descriptor = grid.feature_descriptor(solution)

        # Each feature should return a value based on fitness
        self.assertEqual(2, len(descriptor))
        self.assertEqual(descriptor[0], 0.8)  # feature1
        self.assertEqual(descriptor[1], 1.6)  # feature2

    def test_solution_grid_select(self) -> None:
        """Test selecting random solutions."""
        grid = SolutionGrid(features=self.features, shape=self.shape)

        # Add multiple solutions
        solution1 = DummySolution(fitness=5.0)
        solution2 = DummySolution(fitness=10.0)
        solution3 = DummySolution(fitness=7.5)

        grid.add(solution1)
        grid.add(solution2)
        grid.add(solution3)

        selected = grid.select(1)
        self.assertEqual(len(selected), 1)

        self.assertIn(selected[0], [solution1, solution2, solution3])


if __name__ == "__main__":
    unittest.main()
