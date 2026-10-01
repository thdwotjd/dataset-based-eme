"""Checks for physical two-axis phase interpolation and local fallback."""

import unittest
import warnings
from itertools import product
from types import SimpleNamespace

import numpy as np

from em_simulation.propagator.multiaxis_neff_interpolation import (
    find_crossings, interpolate_multiaxis_beta,
)


class FakeTwoAxisGeometry:
    parameter_names = ("top_width", "curvature")
    wavelength = 2 * np.pi
    _resolution = 65

    def __init__(self, missing_corner=False, outside=False):
        self.outside = outside
        widths = np.array([0.0, 1.0, 2.0])
        curvatures = np.array([0.0, 1.0])
        neff = {
            (float(w), float(c)): np.array([2 + w + 2*c + 3*w*c,
                                                -(2 + w + 2*c + 3*w*c)])
            for w in widths for c in curvatures
        }
        if missing_corner:
            del neff[(2.0, 1.0)]
        overlap = {point: {} for point in neff}
        for point in neff:
            for neighbor in neff:
                changed = sum(a != b for a, b in zip(point, neighbor))
                adjacent = (abs(point[0] - neighbor[0]) == 1 and point[1] == neighbor[1]) or (
                    point[0] == neighbor[0] and abs(point[1] - neighbor[1]) == 1
                )
                if changed == 1 and adjacent:
                    overlap[point][neighbor] = np.eye(2)
        self.data = SimpleNamespace(
            parameter_grid={"top_width": widths, "curvature": curvatures},
            neff=neff, overlap=overlap,
            get_overlap=lambda a, b: (overlap[a][b], overlap[b][a]),
        )
        self.output_data = {
            "neff": np.array([neff[(1.0, 0.0)], neff[(1.0, 1.0)]]),
            "beta": np.array([neff[(1.0, 0.0)], neff[(1.0, 1.0)]]),
            "mode_present": np.ones((2, 2), dtype=bool),
            "radiation_mode_mask": np.zeros((2, 2), dtype=bool),
            "EME_delta_zs": np.array([1.0]),
            "EME_path": [(1.0, 0.0), (1.0, 1.0)],
            "tracking_mode_names": np.array([[0], [0]]),
        }

    def continuous_parameter_values(self, positions):
        positions = np.asarray(positions)
        width = -0.25 + 2.5*positions if self.outside else 0.5 + positions
        return {"top_width": width, "curvature": positions**2}

    def _get_radiation_mode_mask(self, neff):
        return np.zeros(np.asarray(neff).shape, dtype=bool)


class MultiaxisInterpolationTests(unittest.TestCase):
    def test_constant_parameter_on_grid_has_no_extra_crossings(self):
        geometry = FakeTwoAxisGeometry()
        geometry.continuous_parameter_values = lambda positions: {
            "top_width": 0.5 + np.asarray(positions),
            "curvature": np.zeros_like(np.asarray(positions)),
        }
        roots = find_crossings(
            geometry, geometry.parameter_names,
            geometry.data.parameter_grid,
            {"top_width": 1e-12, "curvature": 1e-12},
            np.array([0.0, 1.0]),
        )
        np.testing.assert_allclose(roots, [0.0, 0.5, 1.0])

    def test_two_axes_and_width_grid_crossing(self):
        beta, info = interpolate_multiaxis_beta(FakeTwoAxisGeometry(),
                                                 ("top_width", "curvature"))
        self.assertAlmostEqual(beta[0, 0].real, 59 / 12, places=9)
        self.assertEqual(info["grid_crossings"], 1)
        self.assertEqual(info["fully_interpolated_mode_steps"], 1)
        self.assertEqual(info["fallback_pieces_by_reason"], {})

    def test_missing_corner_falls_back_only_on_affected_piece(self):
        with warnings.catch_warnings(record=True) as caught:
            beta, info = interpolate_multiaxis_beta(
                FakeTwoAxisGeometry(missing_corner=True),
                ("top_width", "curvature"),
            )
        self.assertEqual(len(caught), 1)
        self.assertAlmostEqual(beta[0, 0].real, 3.067708333333333, places=9)
        self.assertAlmostEqual(info["fallback_mode_length_fraction"], 0.5)
        self.assertIn("missing_corner", info["fallback_pieces_by_reason"])

    def test_outside_grid_retains_only_outside_phase(self):
        with warnings.catch_warnings(record=True) as caught:
            beta, info = interpolate_multiaxis_beta(
                FakeTwoAxisGeometry(outside=True), ("top_width", "curvature")
            )
        self.assertEqual(len(caught), 1)
        self.assertTrue(np.isfinite(beta[0, 0]))
        self.assertGreater(info["fallback_mode_length_fraction"], 0)
        self.assertLess(info["fallback_mode_length_fraction"], 1)
        self.assertIn("out_of_grid", info["fallback_pieces_by_reason"])


class SwappedModeGeometry(FakeTwoAxisGeometry):
    def __init__(self, conflicting=False):
        self.outside = False
        widths = np.array([0.0, 1.0])
        curvatures = np.array([0.0, 1.0])
        points = [(float(w), float(c)) for w in widths for c in curvatures]
        neff = {}
        for point in points:
            w, c = point
            modes = np.array([2 + w + c, 4 + w + c])
            if point == (1.0, 1.0):
                modes = modes[::-1]
            neff[point] = np.r_[modes, -modes]
        overlap = {point: {} for point in points}
        for point in points:
            for neighbor in points:
                if sum(a != b for a, b in zip(point, neighbor)) != 1:
                    continue
                swapped = (point == (1.0, 1.0)) != (neighbor == (1.0, 1.0))
                if conflicting and set((point, neighbor)) == set(((1.0, 0.0), (1.0, 1.0))):
                    swapped = False
                block = np.array([[0, 1], [1, 0]]) if swapped else np.eye(2)
                overlap[point][neighbor] = np.block([
                    [block, np.zeros((2, 2))],
                    [np.zeros((2, 2)), block],
                ])
        self.data = SimpleNamespace(
            parameter_grid={"top_width": widths, "curvature": curvatures},
            neff=neff, overlap=overlap,
            get_overlap=lambda a, b: (overlap[a][b], overlap[b][a]),
        )
        path = [(0.0, 0.0), (0.0, 1.0)]
        self.output_data = {
            "neff": np.array([neff[p] for p in path]),
            "beta": np.array([neff[p] for p in path]),
            "mode_present": np.ones((2, 4), dtype=bool),
            "radiation_mode_mask": np.zeros((2, 4), dtype=bool),
            "EME_delta_zs": np.array([1.0]),
            "EME_path": path,
            "tracking_mode_names": np.array([[0, 1], [0, 1]]),
        }

    def continuous_parameter_values(self, positions):
        positions = np.asarray(positions)
        return {"top_width": 0.25 + 0.5*positions, "curvature": positions}


class ModeMatchingTests(unittest.TestCase):
    def test_corner_raw_mode_swap_preserves_physical_modes(self):
        beta, info = interpolate_multiaxis_beta(
            SwappedModeGeometry(), ("top_width", "curvature")
        )
        self.assertAlmostEqual(beta[0, 0].real, 3.0)
        self.assertAlmostEqual(beta[0, 1].real, 5.0)
        self.assertEqual(info["fallback_pieces_by_reason"], {})

    def test_conflicting_routes_fall_back(self):
        with warnings.catch_warnings(record=True) as caught:
            beta, info = interpolate_multiaxis_beta(
                SwappedModeGeometry(conflicting=True),
                ("top_width", "curvature"),
            )
        self.assertEqual(len(caught), 1)
        self.assertIn("mode_tracking_conflict", info["fallback_pieces_by_reason"])
        self.assertAlmostEqual(beta[0, 0].real, 2.0)


class FakeThreeAxisGeometry(FakeTwoAxisGeometry):
    parameter_names = ("top_width", "curvature", "rotation_angle")

    def __init__(self):
        grids = {
            "top_width": np.array([0.0, 1.0, 2.0]),
            "curvature": np.array([0.0, 1.0]),
            "rotation_angle": np.array([0.0, 1.0]),
        }
        neff = {}
        for w, c, angle in product(*grids.values()):
            value = 2 + w + c + angle + w*c*angle
            neff[(w, c, angle)] = np.array([value, -value])
        overlap = {point: {} for point in neff}
        for point in neff:
            for neighbor in neff:
                if sum(a != b for a, b in zip(point, neighbor)) == 1:
                    overlap[point][neighbor] = np.eye(2)
        self.data = SimpleNamespace(
            parameter_grid=grids, neff=neff, overlap=overlap,
            get_overlap=lambda a, b: (overlap[a][b], overlap[b][a]),
        )
        path = [(1.0, 0.0, 0.0), (1.0, 1.0, 1.0)]
        self.output_data = {
            "neff": np.array([neff[p] for p in path]),
            "beta": np.array([neff[p] for p in path]),
            "mode_present": np.ones((2, 2), dtype=bool),
            "radiation_mode_mask": np.zeros((2, 2), dtype=bool),
            "EME_delta_zs": np.array([1.0]),
            "EME_path": path,
            "tracking_mode_names": np.array([[0], [0]]),
        }

    def continuous_parameter_values(self, positions):
        positions = np.asarray(positions)
        return {"top_width": 0.5 + positions,
                "curvature": positions,
                "rotation_angle": positions}


class ThreeAxisInterpolationTests(unittest.TestCase):
    def test_three_axis_cell_corners(self):
        geometry = FakeThreeAxisGeometry()
        beta, info = interpolate_multiaxis_beta(
            geometry, geometry.parameter_names
        )
        self.assertAlmostEqual(beta[0, 0].real, 53 / 12, places=9)
        self.assertEqual(info["fallback_pieces_by_reason"], {})


if __name__ == "__main__":
    unittest.main()
