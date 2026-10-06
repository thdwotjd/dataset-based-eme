"""Regression tests for mode identity preservation across geometry sections."""

import unittest
from types import SimpleNamespace

import numpy as np

from em_simulation.geometry.geometry import Geometry
from em_simulation.geometry.direct_geometry import DirectGeometry


def full_overlap(forward):
    forward = np.asarray(forward, dtype=complex)
    n = len(forward)
    result = np.zeros((2*n, 2*n), dtype=complex)
    result[:n, :n] = forward
    result[n:, n:] = -forward
    return result


class ModeTrackingTests(unittest.TestCase):
    classes = (Geometry, DirectGeometry)

    def test_competing_maxima_use_distinct_predecessors(self):
        overlap = full_overlap([[0.9, 0.85], [0.8, 0.1]])[None]
        for cls in self.classes:
            with self.subTest(geometry=cls.__name__):
                links = cls._generate_mode_links(None, overlap)
                np.testing.assert_array_equal(links, [[1, 0]])

    def test_unambiguous_swapped_order_is_preserved(self):
        overlap = full_overlap([[0.1, 0.92], [0.93, 0.05]])[None]
        for cls in self.classes:
            np.testing.assert_array_equal(cls._generate_mode_links(None, overlap), [[1, 0]])

    def test_below_threshold_is_not_forced(self):
        overlap = full_overlap([[0.5, 0.1], [0.2, 0.9]])[None]
        for cls in self.classes:
            np.testing.assert_array_equal(cls._generate_mode_links(None, overlap), [[-500, 1]])

    def test_threshold_is_applied_before_assignment(self):
        # A full assignment followed by filtering would lose the valid 0.51 link.
        overlap = full_overlap([[0.51, 0.49], [0.49, 0.0]])[None]
        for cls in self.classes:
            np.testing.assert_array_equal(cls._generate_mode_links(None, overlap), [[0, -500]])

    def test_complex_phase_does_not_change_overlap_score(self):
        overlap = full_overlap([[0.95j, 0.1], [0.1j, -0.9]])[None]
        for cls in self.classes:
            np.testing.assert_array_equal(cls._generate_mode_links(None, overlap), [[0, 1]])

    def test_tied_scores_still_have_unique_links(self):
        overlap = full_overlap(np.full((3, 3), 0.8))[None]
        for cls in self.classes:
            links = cls._generate_mode_links(None, overlap)[0]
            self.assertEqual(set(links), {0, 1, 2})

    def test_nonfinite_overlap_is_rejected(self):
        for value in (np.nan, np.inf):
            for cls in self.classes:
                with self.assertRaises(ValueError):
                    cls._generate_mode_links(None, full_overlap([[value]])[None])

    def test_unmatched_mode_keeps_existing_new_id_convention(self):
        links = np.array([[0, -500], [1, 0]])
        for cls in self.classes:
            np.testing.assert_array_equal(cls._generate_tracking_mode_names(None, links),
                                          [[0, 1], [0, 2], [2, 0]])

    def test_duplicate_predecessor_is_rejected(self):
        for cls in self.classes:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                cls._generate_tracking_mode_names(None, np.array([[0, 0]]))

    def test_out_of_range_predecessor_is_rejected(self):
        for cls in self.classes:
            with self.assertRaises(ValueError):
                cls._generate_tracking_mode_names(None, np.array([[0, 2]]))

    def test_reordered_modal_values_preserve_both_directions(self):
        names = np.array([[0, 1], [1, 0]])
        data = np.array([[1, 2, -1, -2], [3, 4, -3, -4]], dtype=complex)
        for cls in self.classes:
            np.testing.assert_array_equal(cls._reorder_data(None, names, data),
                                          [[1, 2, -1, -2], [4, 3, -4, -3]])

    def test_duplicate_names_cannot_overwrite_modal_values(self):
        for cls in self.classes:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                cls._reorder_data(None, np.array([[0, 0]]), np.arange(4)[None])

    def test_overlap_reordering_preserves_all_blocks(self):
        names = np.array([[0, 1], [1, 0]])
        ab = (np.arange(16).reshape(1, 4, 4) + 1j).astype(np.complex64)
        ba = (ab + 30j).copy()
        right = [1, 0, 3, 2]
        for cls in self.classes:
            reordered_ab, reordered_ba = cls._reorder_overlap(None, names, ab, ba)
            np.testing.assert_array_equal(reordered_ab, ab[:, :, right])
            np.testing.assert_array_equal(reordered_ba, ba[:, right, :])

    def test_duplicate_names_cannot_overwrite_overlap(self):
        names = np.array([[0, 0], [0, 1]])
        overlap = full_overlap(np.eye(2))[None]
        for cls in self.classes:
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                cls._reorder_overlap(None, names, overlap, overlap)

    def test_additional_point_uses_one_to_one_linking(self):
        overlap = full_overlap([[0.9, 0.85], [0.8, 0.1]])
        names = np.array([[4, 7]])
        np.testing.assert_array_equal(Geometry._generate_new_tracking_mode_names(None, overlap, names, 0), [7, 4])

    def test_additional_unmatched_mode_stays_minus_one(self):
        overlap = full_overlap([[0.9, 0.1], [0.1, 0.2]])
        np.testing.assert_array_equal(Geometry._generate_new_tracking_mode_names(None, overlap, np.array([[0, 1]]), 0), [0, -1])

    def test_additional_reorder_rejects_duplicate_valid_ids(self):
        overlap = full_overlap(np.eye(2))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            Geometry._reorder_additional_overlap(None, overlap, overlap, np.array([[0, 1]]), np.array([0, 0]), 0)

    def test_additional_reorder_allows_unmatched_mode(self):
        overlap = full_overlap(np.eye(2))
        ab, ba = Geometry._reorder_additional_overlap(None, overlap, overlap,
                                                     np.array([[0, 1]]), np.array([0, -1]), 0)
        self.assertTrue(np.all(ab[:, [1, 3]] == 0))
        self.assertTrue(np.all(ba[[1, 3], :] == 0))


if __name__ == '__main__':
    unittest.main()
