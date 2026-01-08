import unittest

from optimizer import MAPElitesOptimizer
from protocols import GenerationException

from test.fixtures import DummyEvaluator, DummyFeature, DummyGenerator, DummySolution


class TestMAPElitesOptimizer(unittest.TestCase):
    def setUp(self) -> None:
        self.features = [DummyFeature(name="feature1", min_value=0, max_value=10)]
        self.grid_shape = (5,)

        # Create generator and evaluator
        self.generator = DummyGenerator()
        self.evaluator = DummyEvaluator()

    def test_optimize_basic(self) -> None:
        """Test that optimize() finds solutions and improves fitness over time."""
        optimizer = MAPElitesOptimizer(
            features=self.features,
            grid_shape=self.grid_shape,
            generator=self.generator,
            evaluator=self.evaluator,
            n_initial_solutions=5,
        )

        # Run optimization for a few generations
        best_solution = optimizer.optimize(n_generations=2)

        self.assertTrue(len(optimizer.solution_grid.solutions()) > 5)

        # Keep running until fitness converges to zero
        while best_solution.fitness < 0:
            best_solution = optimizer.optimize(n_generations=1)

        self.assertEqual(best_solution.fitness, 0)

    def test_optimize_target_fitness(self) -> None:
        """Test that optimize() stops early when target fitness is reached."""
        optimizer = MAPElitesOptimizer(
            features=self.features,
            grid_shape=self.grid_shape,
            generator=self.generator,
            evaluator=self.evaluator,
            n_initial_solutions=5,
        )

        # Add a solution that already meets the target fitness
        perfect_solution = DummySolution(value=0, fitness=0)  # Perfect score
        optimizer.add_solutions([perfect_solution])

        # Run optimization with target fitness of 0 (should stop immediately)
        best_solution = optimizer.optimize(n_generations=10, target_fitness=0)

        # Should have found the perfect solution
        self.assertTrue(best_solution is perfect_solution)

    def test_optimize_no_solutions(self) -> None:
        """Test that optimize() raises GenerationException when no solutions are found."""
        optimizer = MAPElitesOptimizer(
            features=self.features,
            grid_shape=self.grid_shape,
            generator=self.generator,
            evaluator=self.evaluator,
            n_initial_solutions=0,  # We won't add any initial solution
        )

        with self.assertRaises(GenerationException):
            optimizer.optimize(n_generations=5)

    def test_optimize_reset(self) -> None:
        """Test that reset() clears the optimizer's state."""
        optimizer = MAPElitesOptimizer(
            features=self.features,
            grid_shape=self.grid_shape,
            generator=self.generator,
            evaluator=self.evaluator,
            n_initial_solutions=3,
        )

        # Run some optimization
        optimizer.optimize(n_generations=5)

        # Should have some solutions
        self.assertTrue(len(optimizer.solution_grid.solutions()) > 0)

        # Reset the optimizer
        optimizer.reset()

        # Should be back to empty state
        self.assertTrue(optimizer.solution_grid.empty())
        self.assertEqual([], optimizer.elite)

    def test_optimize_add_solutions(self) -> None:
        """Test that add_solutions() works correctly."""
        optimizer = MAPElitesOptimizer(
            features=self.features,
            grid_shape=self.grid_shape,
            generator=self.generator,
            evaluator=self.evaluator,
        )

        # Create some solutions manually
        solution1 = DummySolution(value=5, fitness=-5)
        solution2 = DummySolution(value=-3, fitness=3)

        # Add them to the optimizer
        optimizer.add_solutions([solution1, solution2])

        # Should have the solutions
        self.assertEqual(len(optimizer.solution_grid.solutions()), 2)

        # Should be able to find the best one
        best = optimizer.solution_grid.best_solution()
        self.assertEqual(best, solution2)  # fitness=3 is better than fitness=-5


if __name__ == "__main__":
    unittest.main()
