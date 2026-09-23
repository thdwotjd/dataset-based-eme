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
    def _make_phase_eme(
        pml_mode_sink=False,
        force_unitary=False,
        force_passive=False,
        pml_basis_loss_threshold=None,
    ):
        config = EMEStabilityConfig(
            pml_mode_sink=pml_mode_sink,
            pml_basis_loss_threshold=pml_basis_loss_threshold,
        )
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 4
        eme.section_count = 2
        losses = np.array([
            1e-6,
            1e-3,
            2 * config.pml_sink_loss,
            2 * config.pml_sink_loss,
        ])
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
        eme.basis_mode_present = eme._calc_basis_mode_presence()
        eme._mode_weights = mode_reliability_weights(
            neff, eme.basis_mode_present, config
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
        eme._force_passive = force_passive
        return eme

    def test_pml_mode_sink_is_disabled_by_default(self):
        self.assertFalse(EMEStabilityConfig().pml_mode_sink)

    def test_pml_basis_cutoff_is_disabled_by_default(self):
        self.assertIsNone(EMEStabilityConfig().pml_basis_loss_threshold)

    def test_pml_basis_cutoff_must_be_finite_and_positive(self):
        for invalid in (0.0, -1.0, np.inf, np.nan):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    EMEStabilityConfig(pml_basis_loss_threshold=invalid)

    def test_pml_basis_cutoff_zeros_interface_and_phase_channels(self):
        cutoff = self._make_phase_eme(pml_basis_loss_threshold=5e-3)

        np.testing.assert_array_equal(
            cutoff.basis_mode_present,
            np.array([
                [True, True, False, False],
                [True, True, False, False],
            ]),
        )
        np.testing.assert_allclose(cutoff._mode_weights[:, 2:], 0.0)

        phase = cutoff._calc_phase_propagation_Smatrix()[0]
        interface = cutoff._calc_interface_Smatrix()[0]
        excluded_ports = [2, 3, 6, 7]
        np.testing.assert_allclose(phase[excluded_ports, :], 0.0)
        np.testing.assert_allclose(phase[:, excluded_ports], 0.0)
        np.testing.assert_allclose(interface[excluded_ports, :], 0.0)
        np.testing.assert_allclose(interface[:, excluded_ports], 0.0)
        self.assertNotEqual(phase[0, 0], 0.0)
        self.assertNotEqual(interface[0, 0], 0.0)

    def test_pml_basis_cutoff_survives_local_unitary_projection(self):
        cutoff = self._make_phase_eme(
            force_unitary=True,
            pml_basis_loss_threshold=5e-3,
        )
        smatrix = cutoff.calc_Smatrix()

        excluded_ports = [2, 3, 6, 7]
        np.testing.assert_allclose(smatrix[:, excluded_ports, :], 0.0)
        np.testing.assert_allclose(smatrix[:, :, excluded_ports], 0.0)
        self.assertNotEqual(smatrix[0, 0, 0], 0.0)

    def test_pml_basis_cutoff_rejects_transfer_matrix_path(self):
        cutoff = self._make_phase_eme(pml_basis_loss_threshold=5e-3)

        with self.assertRaisesRegex(RuntimeError, "basis cutoff"):
            cutoff.calc_Tmatrix()

    def test_pretracking_cutoff_ignores_excluded_overlap_candidates(self):
        losses = np.array([
            [1e-4, 2e-2, 2e-4],
            [1e-4, 2e-4, 3e-4],
        ])
        forward_neff = 1.5 + 1j * losses
        neff = np.concatenate((forward_neff, forward_neff), axis=1)
        present = Geometry._generate_raw_mode_presence(neff, 1e-2)
        overlap = np.zeros((1, 6, 6), dtype=complex)
        overlap[0, 0, 0] = 0.8
        overlap[0, 1, 0] = 0.95
        overlap[0, 2, 2] = 1.0

        geometry = Geometry.__new__(Geometry)
        links = geometry._generate_mode_links(overlap, present)
        names = geometry._generate_tracking_mode_names(links, present)

        self.assertEqual(links[0, 0], 0)
        self.assertEqual(names[0, 1], -1)
        self.assertTrue(np.all(names[present] >= 0))
        self.assertEqual(int(names.max() + 1), 3)

    def test_filtered_overlap_reorder_uses_compact_tracked_dimension(self):
        tracking_names = np.array([
            [0, -1],
            [0, 1],
        ])
        overlap_ab = np.arange(16, dtype=float).reshape(1, 4, 4).astype(complex)
        overlap_ba = (100 + np.arange(16, dtype=float)).reshape(1, 4, 4).astype(complex)

        reordered_ab, reordered_ba = Geometry._reorder_filtered_overlap(
            tracking_names,
            overlap_ab,
            overlap_ba,
        )

        self.assertEqual(reordered_ab.shape, (1, 4, 4))
        self.assertEqual(reordered_ba.shape, (1, 4, 4))
        self.assertEqual(reordered_ab[0, 0, 1], overlap_ab[0, 0, 1])
        self.assertEqual(reordered_ba[0, 1, 0], overlap_ba[0, 1, 0])

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

    def test_local_unitary_projection_preserves_pml_sink_propagation(self):
        sink = self._make_phase_eme(
            pml_mode_sink=True, force_unitary=True
        )
        expected_phase = sink._calc_phase_propagation_Smatrix()[0]
        smatrix = sink.calc_Smatrix()

        np.testing.assert_allclose(smatrix[0], expected_phase)

    def test_local_unitary_projection_preserves_lossy_propagation(self):
        eme = self._make_phase_eme(force_unitary=True)
        expected_phase = eme._calc_phase_propagation_Smatrix()[0]

        smatrix = eme.calc_Smatrix()

        np.testing.assert_allclose(smatrix[0], expected_phase)
        self.assertLess(abs(smatrix[0, 1, 1]), 1.0)

    def test_local_passive_projection_preserves_lossy_propagation(self):
        eme = self._make_phase_eme(force_passive=True)
        expected_phase = eme._calc_phase_propagation_Smatrix()[0]

        smatrix = eme.calc_Smatrix()

        np.testing.assert_allclose(smatrix[0], expected_phase)
        self.assertLess(abs(smatrix[0, 1, 1]), 1.0)

    def test_length_change_preserves_lossy_propagation_with_conditioning(self):
        for option in ("unitary", "passive"):
            with self.subTest(option=option):
                eme = self._make_phase_eme(
                    force_unitary=option == "unitary",
                    force_passive=option == "passive",
                )
                expected_phase = eme._calc_phase_propagation_Smatrix(2.0)[0]

                smatrix = eme._find_Smatrix_new_length(4e-6)

                np.testing.assert_allclose(smatrix[0], expected_phase)

    def test_propagation_gain_is_reported_without_rescaling(self):
        eme = self._make_phase_eme()
        eme.beta_forward[0, 0] = (
            eme.beta_forward[0, 0].real - 1j * 1e4
        )

        with self.assertWarnsRegex(RuntimeWarning, "modal gain"):
            phase = eme._calc_phase_propagation_Smatrix()

        self.assertGreater(abs(phase[0, 0, 0]), 1.0)
        self.assertEqual(
            eme.stability_diagnostics["propagation_gain_count"], 1
        )
        self.assertGreater(
            eme.stability_diagnostics["max_propagation_magnitude"], 1.0
        )

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
            np.sqrt(config.guided_loss * config.pml_regularization_loss),
            config.pml_regularization_loss,
        ])
        neff = 1.5 + 1j * losses

        weights = mode_reliability_weights(
            neff, np.ones(neff.shape, dtype=bool), config
        )

        np.testing.assert_allclose(weights, [1.0, 1.0, 0.5, 0.0])

    def test_regularization_and_sink_loss_thresholds_are_independent(self):
        config = EMEStabilityConfig(
            pml_regularization_loss=1e-3,
            pml_sink_loss=1e-2,
            pml_mode_sink=True,
        )
        losses = np.array([1e-4, 2e-3, 2e-2])
        neff = np.vstack([1.5 + 1j * losses, 1.5 + 1j * losses])
        present = np.ones(neff.shape, dtype=bool)

        weights = mode_reliability_weights(neff, present, config)
        np.testing.assert_allclose(weights[:, 1:], 0.0)

        eme = SingleEME.__new__(SingleEME)
        eme.neff_forward = neff
        eme.mode_present = present
        eme.stability_config = config
        eme.basis_mode_present = eme._calc_basis_mode_presence()
        expected_sink_mask = np.array([[False, False, True]])
        np.testing.assert_array_equal(eme._pml_mode_mask(), expected_sink_mask)

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

    def test_stabilized_inverse_supports_rectangular_matrix(self):
        config = EMEStabilityConfig()
        matrix = np.array([
            [1.0 + 0.1j, 0.2, 0.0],
            [0.1, 0.9 - 0.2j, 0.3],
        ])

        inverse = stabilized_inverse(
            matrix,
            np.ones(3),
            np.ones(2),
            config,
        )

        self.assertEqual(inverse.shape, (3, 2))
        np.testing.assert_allclose(
            inverse,
            np.linalg.pinv(matrix),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_local_interface_smatrix_supports_unequal_mode_counts(self):
        overlap_lr = np.array([
            [1.0 + 0.1j, 0.08, 0.02j],
            [0.04, 0.9 - 0.05j, 0.07],
        ])
        overlap_rl = np.array([
            [0.95 - 0.02j, 0.03],
            [0.05j, 1.02 + 0.04j],
            [0.02, 0.06],
        ])
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 3
        eme.section_count = 2
        eme.basis_mode_present = np.array([
            [True, True, False],
            [True, True, True],
        ])
        eme.overlap_forward_ab = np.zeros((1, 3, 3), dtype=complex)
        eme.overlap_forward_ba = np.zeros((1, 3, 3), dtype=complex)
        eme.overlap_forward_ab[0, :2, :3] = overlap_lr
        eme.overlap_forward_ba[0, :3, :2] = overlap_rl
        eme._mode_weights = np.ones((2, 3))
        eme.stability_config = EMEStabilityConfig()
        eme.stability_diagnostics = new_stability_diagnostics()

        local, left_indices, right_indices = (
            eme._calc_local_interface_Smatrix(0)
        )
        transmission_lr = 2 * np.linalg.pinv(overlap_lr + overlap_rl.T)
        transmission_rl = 2 * np.linalg.pinv(overlap_rl + overlap_lr.T)
        reflection_ll = 0.5 * (overlap_rl.T - overlap_lr) @ transmission_lr
        reflection_rr = 0.5 * (overlap_lr.T - overlap_rl) @ transmission_rl
        expected = np.block([
            [transmission_lr, -reflection_rr],
            [reflection_ll, transmission_rl],
        ])

        np.testing.assert_array_equal(left_indices, [0, 1])
        np.testing.assert_array_equal(right_indices, [0, 1, 2])
        self.assertEqual(local.shape, (5, 5))
        np.testing.assert_allclose(local, expected, rtol=1e-12, atol=1e-12)

        embedded = eme._calc_interface_Smatrix()[0]
        global_rows = np.array([0, 1, 2, 3, 4])
        global_columns = np.array([0, 1, 3, 4, 5])
        np.testing.assert_allclose(
            embedded[np.ix_(global_rows, global_columns)],
            expected,
            rtol=1e-12,
            atol=1e-12,
        )
        expected_support = np.zeros((6, 6), dtype=bool)
        expected_support[np.ix_(global_rows, global_columns)] = True
        np.testing.assert_allclose(embedded[~expected_support], 0.0)
        self.assertEqual(
            eme.stability_diagnostics["inactive_leakage_count"], 0
        )
        self.assertEqual(
            eme.stability_diagnostics["max_inactive_leakage"], 0.0
        )

    def test_local_unitary_projection_sets_all_singular_values_to_one(self):
        eme = SingleEME.__new__(SingleEME)
        eme._force_unitary = True
        eme._force_passive = False
        local = np.diag([2.0, 0.5, 0.1]).astype(complex)

        conditioned = eme._condition_local_interface_smatrix(local)

        np.testing.assert_allclose(
            np.linalg.svd(conditioned, compute_uv=False), np.ones(3)
        )

    def test_local_passive_projection_clips_only_singular_values_above_one(self):
        eme = SingleEME.__new__(SingleEME)
        eme._force_unitary = False
        eme._force_passive = True
        local = np.diag([2.0, 0.5, 0.1]).astype(complex)

        conditioned = eme._condition_local_interface_smatrix(local)

        np.testing.assert_allclose(
            np.linalg.svd(conditioned, compute_uv=False), [1.0, 0.5, 0.1]
        )

    def test_local_unitary_projection_has_priority_over_passive(self):
        eme = SingleEME.__new__(SingleEME)
        eme._force_unitary = True
        eme._force_passive = True
        local = np.diag([2.0, 0.5, 0.1]).astype(complex)

        conditioned = eme._condition_local_interface_smatrix(local)

        np.testing.assert_allclose(
            np.linalg.svd(conditioned, compute_uv=False), np.ones(3)
        )

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
        eme.basis_mode_present = np.ones((2, 2), dtype=bool)
        eme.stability_diagnostics = new_stability_diagnostics()

        direct = eme._calc_interface_Smatrix()[0]
        transmission_12 = 2 * np.linalg.inv(overlap_ab + overlap_ba.T)
        transmission_21 = 2 * np.linalg.inv(overlap_ba + overlap_ab.T)
        reflection_12 = 0.5 * (overlap_ba.T - overlap_ab) @ transmission_12
        reflection_21 = 0.5 * (overlap_ab.T - overlap_ba) @ transmission_21
        expected = np.block([
            [transmission_12, -reflection_21],
            [reflection_12, transmission_21],
        ])

        np.testing.assert_allclose(direct, expected, rtol=1e-12, atol=1e-12)

    def test_equal_mode_local_interface_matches_square_interface(self):
        overlap_ab = np.array([[1.1 + 0.1j, 0.04], [0.02j, 0.9 - 0.1j]])
        overlap_ba = np.array([[0.95 - 0.03j, -0.02j], [0.03, 1.05 + 0.02j]])
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 2
        eme.section_count = 2
        eme.overlap_forward_ab = overlap_ab[np.newaxis]
        eme.overlap_forward_ba = overlap_ba[np.newaxis]
        eme.basis_mode_present = np.ones((2, 2), dtype=bool)
        eme._mode_weights = np.ones((2, 2))
        eme.stability_config = EMEStabilityConfig()
        eme.stability_diagnostics = new_stability_diagnostics()

        local, _, _ = eme._calc_local_interface_Smatrix(0)
        embedded = eme._calc_interface_Smatrix()[0]

        np.testing.assert_allclose(local, embedded, rtol=1e-12, atol=1e-12)

    def test_single_plane_wave_interface_matches_fresnel_coefficients(self):
        index_left = 1.0
        index_right = 2.0
        overlap_lr = np.array([[np.sqrt(index_right / index_left)]])
        overlap_rl = np.array([[np.sqrt(index_left / index_right)]])
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 1
        eme.section_count = 2
        eme.overlap_forward_ab = overlap_lr[np.newaxis]
        eme.overlap_forward_ba = overlap_rl[np.newaxis]
        eme.basis_mode_present = np.ones((2, 1), dtype=bool)
        eme._mode_weights = np.ones((2, 1))
        eme.stability_config = EMEStabilityConfig()
        eme.stability_diagnostics = new_stability_diagnostics()

        local, _, _ = eme._calc_local_interface_Smatrix(0)
        modal_transmission = local[0, 0]
        electric_field_transmission = modal_transmission * np.sqrt(
            index_left / index_right
        )
        electric_field_reflection = local[1, 0]

        expected_transmission = 2 * index_left / (index_left + index_right)
        expected_reflection = (index_left - index_right) / (
            index_left + index_right
        )
        np.testing.assert_allclose(
            electric_field_transmission, expected_transmission, rtol=1e-12
        )
        np.testing.assert_allclose(
            electric_field_reflection, expected_reflection, rtol=1e-12
        )

    def test_direct_interface_smatrix_matches_corrected_t_to_s_path(self):
        overlap_ab = np.array([[1.1 + 0.1j, 0.04], [0.02j, 0.9 - 0.1j]])
        overlap_ba = np.array([[0.95 - 0.03j, -0.02j], [0.03, 1.05 + 0.02j]])
        eme = SingleEME.__new__(SingleEME)
        eme.mode_count = 2
        eme.section_count = 2
        eme.overlap_forward_ab = overlap_ab[np.newaxis]
        eme.overlap_forward_ba = overlap_ba[np.newaxis]
        eme.basis_mode_present = np.ones((2, 2), dtype=bool)
        eme._mode_weights = np.ones((2, 2))
        eme.stability_config = EMEStabilityConfig()
        eme.stability_diagnostics = new_stability_diagnostics()

        direct = eme._calc_interface_Smatrix()
        transfer = eme._calc_interface_Tmatrix()
        converted = mct._convert_3Dmatrix_ray(transfer)

        np.testing.assert_allclose(direct, converted, rtol=1e-12, atol=1e-12)

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
