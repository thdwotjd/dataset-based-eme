"""Regression checks for the optional one-axis phase interpolation."""

import unittest
from types import SimpleNamespace

import numpy as np

from em_simulation.propagator.neff_interpolation import interpolate_single_axis_beta


class FakeOneAxisGeometry:
    parameter_names = ("top_width", "curvature")
    wavelength = 2 * np.pi
    _resolution = 9

    def __init__(self, varying_width=False, right_guided=True):
        self.varying_width = varying_width
        self.data = SimpleNamespace(
            parameter_grid={
                "top_width": np.array([1.0, 2.0]),
                "curvature": np.array([0.0, 1.0]),
            }
        )
        radiation = np.zeros((2, 2), dtype=bool)
        radiation[1, 0] = not right_guided
        self.output_data = {
            "neff": np.array([[2 + 0j, -2 + 0j], [4 + 0j, -4 + 0j]]),
            "beta": np.array([[2 + 0j, -2 + 0j], [4 + 0j, -4 + 0j]]),
            "mode_present": np.ones((2, 2), dtype=bool),
            "radiation_mode_mask": radiation,
            "EME_delta_zs": np.array([1.0]),
            "EME_path": [(1.0, 0.0), (1.0, 1.0)],
        }

    def continuous_parameter_values(self, positions):
        positions = np.asarray(positions)
        return {
            "top_width": 1.0 + positions if self.varying_width else np.ones_like(positions),
            "curvature": positions**2,
        }


class NeffInterpolationTests(unittest.TestCase):
    def test_weighted_phase_uses_physical_progress(self):
        beta, info = interpolate_single_axis_beta(FakeOneAxisGeometry())
        self.assertAlmostEqual(beta[0, 0].real, 8.0 / 3.0)
        self.assertEqual(info["axis"], "curvature")
        self.assertEqual(info["interpolated_mode_steps"], 1)

    def test_missing_guided_neighbor_uses_existing_phase(self):
        beta, info = interpolate_single_axis_beta(
            FakeOneAxisGeometry(right_guided=False)
        )
        self.assertAlmostEqual(beta[0, 0].real, 2.0)
        self.assertEqual(info["interpolated_mode_steps"], 0)

    def test_two_varying_axes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one varying"):
            interpolate_single_axis_beta(
                FakeOneAxisGeometry(varying_width=True)
            )


if __name__ == "__main__":
    unittest.main()