import unittest

import numpy as np

from optimizer import Elite

from test.fixtures import DummySolution, DummyFeature


class TestElite(unittest.TestCase):
    def setUp(self) -> None:
        self.feature = DummyFeature(name="test_feature", min_value=0.0, max_value=10.0)

    def test_elite_initialization(self) -> None:
        """Test that Elite initializes correctly with a solution and cell."""
        solution = DummySolution(fitness=5.0)
        cell = (1, 2)

        elite = Elite(solution=solution, cell=cell)

        self.assertEqual(elite.solution, solution)
        self.assertEqual(elite.cell, cell)
        self.assertEqual(elite.fitness, 5.0)

    def test_elite_comparison(self) -> None:
        """Test that Elite comparison operators work correctly."""
        solution1 = DummySolution(fitness=5.0)
        solution2 = DummySolution(fitness=10.0)

        elite1 = Elite(solution=solution1, cell=(1, 2))
        elite2 = Elite(solution=solution2, cell=(3, 4))

        self.assertTrue(elite1 != elite2)
        self.assertTrue(elite1 < elite2)
        self.assertTrue(elite1 <= elite2)
        self.assertTrue(elite1 <= elite1)
        self.assertFalse(elite2 == elite1)
        self.assertFalse(elite2 < elite1)
        self.assertFalse(elite2 <= elite1)

    def test_elite_string_representation(self) -> None:
        """Test that Elite string representations are correct."""
        solution = DummySolution(fitness=7.5)
        elite = Elite(solution=solution, cell=(2, 3))

        expected_str = "value=0 (fitness=7.5)"
        self.assertEqual(str(elite), expected_str)

        expected_repr = "Elite: value=0 (fitness=7.5)"
        self.assertEqual(repr(elite), expected_repr)

    def test_elite_property_accessors(self) -> None:
        """Test that Elite property accessors work correctly."""
        solution = DummySolution(fitness=8.0)
        cell = (2, 3)

        elite = Elite(solution=solution, cell=cell)

        self.assertEqual(elite.solution, solution)
        self.assertEqual(elite.cell, cell)
        self.assertEqual(elite.fitness, solution.fitness)

    def test_elite_edge_case_fitness(self) -> None:
        """Test that Elite handles edge case fitness values correctly."""
        solution_inf = DummySolution(fitness=float("-inf"))
        elite_inf = Elite(solution=solution_inf, cell=(0, 0))
        self.assertEqual(elite_inf.fitness, float("-inf"))

        solution_pos_inf = DummySolution(fitness=float("inf"))
        elite_pos_inf = Elite(solution=solution_pos_inf, cell=(0, 0))
        self.assertEqual(elite_pos_inf.fitness, float("inf"))

        solution_nan = DummySolution(fitness=float("nan"))
        elite_nan = Elite(solution=solution_nan, cell=(0, 0))
        self.assertTrue(np.isnan(elite_nan.fitness))


if __name__ == "__main__":
    unittest.main()
