import sys
import types
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class _RemoteFunction:
    def __init__(self, function):
        self.function = function

    def remote(self, *args, **kwargs):
        return self.function(*args, **kwargs)


def _remote(function=None, **_kwargs):
    if function is None:
        return lambda item: _RemoteFunction(item)
    return _RemoteFunction(function)


if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")
    fake_ray.remote = _remote
    fake_ray.get = lambda value: value
    fake_ray.put = lambda value: value
    fake_ray.is_initialized = lambda: True
    fake_ray.init = lambda **_kwargs: None
    sys.modules["ray"] = fake_ray


from em_simulation import matrix_calculation_tool as mct
from em_simulation.geometry.geometry import Geometry
from em_simulation.propagator.single_propagator.single_eme import SingleEME
from em_simulation.propagator.stability import (
    EMEStabilityConfig,
    mode_reliability_weights,
    new_stability_diagnostics,
    stabilized_inverse,
)


class StabilityTests(unittest.TestCase):
    @staticmethod
    def _make_phase_eme(pml_mode_sink=False, force_unitary=False):
        config = EMEStabilityConfig(pml_mode_sink=pml_mode_sink)
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 4
        eme.section_count = 2
        losses = np.array([1e-6, 1e-3, 5e-3, 5e-3])
        neff = np.vstack([1.5 + 1j * losses, 1.5 + 1j * losses])
        eme.neff_forward = neff
        eme.beta_forward = 2 * np.pi * neff / 1.55e-6
        eme.mode_present = np.array([
            [True, True, True, False],
            [True, True, True, False],
        ])
        eme.output_data = {"EME_delta_zs": np.array([2e-6])}
        eme.stability_config = config
        eme.stability_diagnostics = new_stability_diagnostics()
        eme._mode_weights = mode_reliability_weights(
            neff, eme.mode_present, config
        )
        identity = np.eye(eme.mode_count, dtype=complex)[np.newaxis]
        eme.overlap_forward_ab = identity.copy()
        eme.overlap_forward_ba = identity.copy()
        eme._interface_Smatrix = None
        eme._interface_Tmatrix = None
        eme._is_interface_Smatrix_calculated = False
        eme._is_interface_Tmatrix_calcualted = False
        eme._is_smatrix_calculated = False
        eme._is_tmatrix_calculated = False
        eme._force_unitary = force_unitary
        eme._force_passive = False
        return eme

    def test_pml_mode_sink_is_disabled_by_default(self):
        self.assertFalse(EMEStabilityConfig().pml_mode_sink)

    def test_pml_mode_sink_zeros_only_present_pml_phase_channels(self):
        normal = self._make_phase_eme(pml_mode_sink=False)
        sink = self._make_phase_eme(pml_mode_sink=True)

        normal_phase = normal._calc_phase_propagation_Smatrix()[0]
        sink_phase = sink._calc_phase_propagation_Smatrix()[0]
        normal_diagonal = np.diag(normal_phase)
        sink_diagonal = np.diag(sink_phase)

        np.testing.assert_allclose(sink_diagonal[[0, 1, 3, 4, 5, 7]],
                                   normal_diagonal[[0, 1, 3, 4, 5, 7]])
        np.testing.assert_allclose(sink_diagonal[[2, 6]], 0.0)
        self.assertNotEqual(normal_diagonal[2], 0.0)
        self.assertNotEqual(sink_diagonal[3], 0.0)

        sink.calc_Smatrix()
        self.assertEqual(sink.stability_diagnostics["pml_sink_mode_steps"], 1)
        self.assertEqual(sink.stability_diagnostics["status"], "normal")

    def test_pml_mode_sink_has_priority_over_unitary_projection(self):
        sink = self._make_phase_eme(
            pml_mode_sink=True, force_unitary=True
        )
        expected_phase = sink._calc_phase_propagation_Smatrix()[0]
        projected = np.ones((2, 8, 8), dtype=complex)

        with patch.object(
            mct, "_find_nearest_unitary_3D_ray2", return_value=projected
        ):
            smatrix = sink.calc_Smatrix()

        np.testing.assert_allclose(smatrix[0], expected_phase)

    def test_length_scaled_smatrix_keeps_pml_mode_sink(self):
        sink = self._make_phase_eme(pml_mode_sink=True)
        sink._interface_Smatrix = np.eye(8, dtype=complex)[np.newaxis]
        sink._is_interface_Smatrix_calculated = True

        smatrix = sink._find_Smatrix_new_length(4e-6)

        self.assertEqual(smatrix[0, 2, 2], 0.0)
        self.assertEqual(smatrix[0, 6, 6], 0.0)
        self.assertNotEqual(smatrix[0, 1, 1], 0.0)

    def test_pml_mode_sink_rejects_transfer_matrix_path(self):
        sink = self._make_phase_eme(pml_mode_sink=True)

        with self.assertRaisesRegex(RuntimeError, "direct S-matrix path"):
            sink.calc_Tmatrix()

    def test_unmatched_zero_diagonal_does_not_zero_overlap_row_or_column(self):
        overlap_ab = np.eye(4, dtype=complex)[np.newaxis]
        overlap_ba = np.eye(4, dtype=complex)[np.newaxis]
        overlap_ab[0, 0, 1] = 7.0
        overlap_ba[0, 1, 0] = 9.0
        overlap_ab[0, 1, 1] = 0.0
        overlap_ba[0, 1, 1] = 0.0
        mode_present = np.array([
            [True, True, True, True],
            [True, False, True, False],
        ])

        corrected_ab, corrected_ba, _ = Geometry._equalize_overlap_phase(
            Geometry.__new__(Geometry),
            overlap_ab,
            overlap_ba,
            mode_present,
        )

        self.assertEqual(corrected_ab[0, 0, 1], 7.0)
        self.assertEqual(corrected_ba[0, 1, 0], 9.0)

    def test_loss_weights_use_configured_guided_and_pml_limits(self):
        config = EMEStabilityConfig()
        losses = np.array([
            0.0,
            config.guided_loss,
            np.sqrt(config.guided_loss * config.pml_loss),
            config.pml_loss,
        ])
        neff = 1.5 + 1j * losses

        weights = mode_reliability_weights(
            neff, np.ones(neff.shape, dtype=bool), config
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 0.5, 0.0])

    def test_adaptive_cutoff_preserves_guided_and_drops_pml_direction(self):
        config = EMEStabilityConfig()
        matrix = np.diag([1.0, 1e-4]).astype(complex)

        guided_inverse = stabilized_inverse(
            matrix, np.ones(2), np.ones(2), config
        )
        pml_inverse = stabilized_inverse(
            matrix, np.zeros(2), np.zeros(2), config
        )

        np.testing.assert_allclose(guided_inverse, np.diag([1.0, 1e4]))
        np.testing.assert_allclose(pml_inverse, np.diag([1.0, 0.0]))

    def test_guided_direction_below_absolute_floor_warns_and_is_dropped(self):
        config = EMEStabilityConfig()
        diagnostics = new_stability_diagnostics()
        matrix = np.diag([1.0, 1e-10]).astype(complex)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            inverse = stabilized_inverse(
                matrix,
                np.ones(2),
                np.ones(2),
                config,
                diagnostics,
                interface_index=3,
                direction="12",
            )

        self.assertEqual(inverse[1, 1], 0.0)
        self.assertEqual(len(caught), 1)
        self.assertEqual(diagnostics["status"], "regularized")
        self.assertEqual(
            diagnostics["interface_events"][0]["guided_heavy_dropped"], 1
        )

    def test_direct_interface_smatrix_matches_interface_equations(self):
        overlap_ab = np.array([[1.1 + 0.1j, 0.04], [0.02j, 0.9 - 0.1j]])
        overlap_ba = np.array([[0.95 - 0.03j, -0.02j], [0.03, 1.05 + 0.02j]])
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 2
        eme.section_count = 2
        eme.overlap_forward_ab = overlap_ab[np.newaxis]
        eme.overlap_forward_ba = overlap_ba[np.newaxis]
        eme._mode_weights = np.ones((2, 2))
        eme.stability_config = EMEStabilityConfig()
        eme.stability_diagnostics = new_stability_diagnostics()

        direct = eme._calc_interface_Smatrix()[0]
        transmission_12 = 2 * np.linalg.inv(overlap_ab + overlap_ba.T)
        transmission_21 = 2 * np.linalg.inv(overlap_ba + overlap_ab.T)
        reflection_12 = 0.5 * (overlap_ab.T - overlap_ba) @ transmission_12
        reflection_21 = 0.5 * (overlap_ba.T - overlap_ab) @ transmission_21
        expected = np.block([
            [transmission_12, -reflection_21],
            [reflection_12, transmission_21],
        ])

        np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)

    def test_right_solve_falls_back_without_raising(self):
        diagnostics = new_stability_diagnostics()
        singular = np.array([[1.0, 0.0], [0.0, 0.0]])
        rhs = np.eye(2)

        result = mct._right_solve(rhs, singular, diagnostics, context=7)

        self.assertTrue(np.all(np.isfinite(result)))
        self.assertEqual(diagnostics["status"], "regularized")
        self.assertEqual(diagnostics["feedback_fallbacks"][0]["index"], 7)
        self.assertEqual(
            diagnostics["feedback_fallbacks"][0]["reason"], "singular_solve"
        )

    def test_feedback_solve_drops_only_pml_heavy_small_direction(self):
        config = EMEStabilityConfig()
        diagnostics = new_stability_diagnostics()
        matrix = np.diag([1.0, 1e-4]).astype(complex)
        rhs = np.eye(2)

        pml_result = mct._stabilized_feedback_right_solve(
            rhs,
            matrix,
            mode_weights=np.array([1.0, 0.0]),
            mode_present=np.ones(2, dtype=bool),
            stability_config=config,
            diagnostics=diagnostics,
            context=5,
            feedback_matrix="I-a12_b21",
        )
        mct._stabilized_feedback_right_solve(
            rhs,
            matrix,
            mode_weights=np.array([1.0, 0.0]),
            mode_present=np.ones(2, dtype=bool),
            stability_config=config,
            diagnostics=diagnostics,
            context=5,
            feedback_matrix="I-a12_b21",
        )
        guided_result = mct._stabilized_feedback_right_solve(
            rhs,
            matrix,
            mode_weights=np.ones(2),
            mode_present=np.ones(2, dtype=bool),
            stability_config=config,
        )

        np.testing.assert_allclose(pml_result, np.diag([1.0, 0.0]))
        np.testing.assert_allclose(guided_result, np.diag([1.0, 1e4]))
        self.assertEqual(diagnostics["status"], "regularized")
        self.assertEqual(len(diagnostics["feedback_fallbacks"]), 1)
        self.assertEqual(
            diagnostics["feedback_fallbacks"][0]["reason"], "pml_truncated"
        )
        self.assertEqual(
            diagnostics["feedback_fallbacks"][0]["feedback_matrix"],
            "I-a12_b21",
        )


if __name__ == "__main__":
    unittest.main()
