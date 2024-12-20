import unittest
import numpy as np
from process_gNATSGO_5_CONUS import match_coords

class TestMatchCoords(unittest.TestCase):
    def test_basic_matching(self):
        """Test basic nearest neighbor matching"""
        src = np.array([1.0, 2.0, 3.0])
        target = np.array([0.9, 2.1, 3.2])
        expected_indices = np.array([0, 1, 2])
        expected_neighbors = np.array([0.9, 2.1, 3.2])
        
        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_exact_matches(self):
        """Test when source and target have exact matching values"""
        src = np.array([1.0, 2.0, 3.0])
        target = np.array([1.0, 2.0, 3.0])
        expected_indices = np.array([0, 1, 2])
        expected_neighbors = np.array([1.0, 2.0, 3.0])
        
        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_different_sizes(self):
        """Test when source and target arrays have different sizes"""
        src = np.array([1.0, 2.0, 3.0])
        target = np.array([0.9, 1.7, 2.2, 3.1, 3.5])
        expected_indices = np.array([0, 2, 3])
        expected_neighbors = np.array([0.9, 2.2, 3.1])

        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_repeated_values(self):
        """Test when target array has repeated values"""
        src = np.array([1.0, 2.0, 3.0])
        target = np.array([1.1, 1.1, 2.1, 3.1])
        expected_indices = np.array([0, 2, 3])  # Should take first occurrence if equidistant
        expected_neighbors = np.array([1.1, 2.1, 3.1])
        
        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_empty_arrays(self):
        """Test behavior with empty arrays"""
        with self.assertRaises(ValueError):
            match_coords(np.array([]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            match_coords(np.array([1.0, 2.0]), np.array([]))

    def test_single_element(self):
        """Test with single-element arrays"""
        src = np.array([1.0])
        target = np.array([1.1])
        expected_indices = np.array([0])
        expected_neighbors = np.array([1.1])
        
        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

    def test_negative_values(self):
        """Test with negative values"""
        src = np.array([-2.0, -1.0, 0.0])
        target = np.array([-2.2, -1.1, 0.1])
        expected_indices = np.array([0, 1, 2])
        expected_neighbors = np.array([-2.2, -1.1, 0.1])
        
        indices, neighbors = match_coords(src, target)
        np.testing.assert_array_equal(indices, expected_indices)
        np.testing.assert_array_equal(neighbors, expected_neighbors)

if __name__ == '__main__':
    unittest.main()