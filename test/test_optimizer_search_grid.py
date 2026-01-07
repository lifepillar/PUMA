import unittest
import numpy as np

from optimizer import SearchGrid

from test.fixtures import DummyFeature


class TestSearchGrid(unittest.TestCase):
    """Test the SearchGrid class functionality."""

    def setUp(self) -> None:
        self.features = [
            DummyFeature("feature1", min_value=0.0, max_value=10.0),
            DummyFeature("feature2", min_value=5.0, max_value=15.0),
            DummyFeature("feature3", min_value=-5.0, max_value=5.0),
        ]

    def test_initialization_with_int_shape(self) -> None:
        """Test SearchGrid initialization with integer shape."""
        grid = SearchGrid(self.features, shape=5)

        # Check that shape is correctly set
        expected_shape = np.array([5, 5, 5])
        self.assertTrue(np.array_equal(grid.shape, expected_shape))

        # Check min/max values
        expected_min = np.array([0.0, 5.0, -5.0])
        expected_max = np.array([10.0, 15.0, 5.0])
        self.assertTrue(np.array_equal(grid.min, expected_min))
        self.assertTrue(np.array_equal(grid.max, expected_max))

    def test_initialization_with_tuple_shape(self) -> None:
        """Test SearchGrid initialization with tuple shape."""
        grid = SearchGrid(self.features, shape=(4, 6, 2))

        # Check that shape is correctly set
        expected_shape = np.array([4, 6, 2])
        self.assertTrue(np.array_equal(grid.shape, expected_shape))

    def test_initialization_with_invalid_shape(self) -> None:
        """Test that invalid shapes raise appropriate errors."""
        # Shape with wrong length
        with self.assertRaises(AssertionError):
            SearchGrid(self.features, shape=(4, 3))  # Only 2 values for 3 features

        # Shape with single point is too small
        with self.assertRaises(AssertionError):
            SearchGrid(self.features, shape=(2, 1, 2))

    def test_centroids_method(self) -> None:
        """Test the centroids() method."""
        grid = SearchGrid(self.features, (5, 6, 3))

        # Test first dimension
        result = grid.centroids(0)
        expected = np.array([0.0, 2.5, 5.0, 7.5, 10.0])
        np.testing.assert_array_almost_equal(result, expected)

        # Test second dimension
        result = grid.centroids(1)
        expected = np.array([5.0, 7.0, 9.0, 11.0, 13.0, 15.0])
        np.testing.assert_array_almost_equal(result, expected)

        # Test third dimension
        result = grid.centroids(2)
        expected = np.array([-5.0, 0.0, 5.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_at_method(self) -> None:
        """Test the at() method for getting grid points."""
        # feature1: 0.0  2.5  5.0  7.5  10.0
        # feature2: 5.0  7.0  9.0  11.0 13.0 15.0
        # feature3: -5.0  0.0  5.0
        grid = SearchGrid(self.features, (5, 6, 3))

        # Test center point
        result = grid.at((2, 3, 1))
        expected = np.array([5.0, 11.0, 0.0])
        np.testing.assert_array_almost_equal(result, expected)

        # Test with clipping
        result_clipped = grid.at((2, 3, 1), clip=True)
        np.testing.assert_array_almost_equal(result_clipped, expected)

        # Test out-of-bounds without clipping
        result_oo = grid.at((5, 6, 4))
        expected_oo = np.array([12.5, 17.0, 15.0])
        np.testing.assert_array_almost_equal(result_oo, expected_oo)

        # Test out-of-bounds with clipping
        result_clipped_oo = grid.at((5, 6, 4), clip=True)
        expected_clipped_oo = np.array([10.0, 15.0, 5.0])  # Clipped to max values
        np.testing.assert_array_almost_equal(result_clipped_oo, expected_clipped_oo)

    def test_snap_method(self) -> None:
        """Test the snap() method for finding nearest grid points."""
        # feature1: 0.0  2.5  5.0  7.5  10.0
        # feature2: 5.0  7.0  9.0  11.0 13.0 15.0
        # feature3: -5.0  0.0  5.0
        grid = SearchGrid(self.features, (5, 6, 3))

        result = grid.snap(np.array([0.0, 5.0, -5.0]))
        self.assertEqual(result, (0, 0, 0))

        result = grid.snap(np.array([10.0, 15.0, 5.0]))
        self.assertEqual(result, (4, 5, 2))

        # Test snapping to midpoints
        result = grid.snap(np.array([5.0, 10.0, 0.0]))
        self.assertEqual(result, (2, 2, 1))

        # Test snapping with values slightly off grid
        result = grid.snap(np.array([3.1, 7.4, -2.6]))
        self.assertEqual(result, (1, 1, 0))

        # Test snapping with clipping
        result = grid.snap(np.array([-1.0, 16.0, 8.0]), clip=True)
        self.assertEqual(result, (0, 5, 2))


if __name__ == "__main__":
    unittest.main()
