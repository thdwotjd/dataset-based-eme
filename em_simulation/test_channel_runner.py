"""ChannelRunner integration with persisted synthetic MODE fields and real EME propagation."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import warnings

import numpy as np
from matplotlib import pyplot as plt

from em_simulation import (
    ChannelRunner, CoupledDatasetSettings, DataUpdater, EME, EMEStabilityConfig, FDEConfig, Runner,
    SingleStraightWaveguide, StraightGapCoupler, StraightGapSectionModel, WaveguideStack, profile_from_samples,
)
from em_simulation.channels.mapping import ChannelMappingWarning, MappingConfig
from em_simulation.data_updater.channel_data_builder import build_channel_data
from em_simulation.test_channel_dataset import ChannelMode, GAPS
from em_simulation.test_supermode_dataset import device


RING = ("wg1", 1)
BUS_TM = ("wg0", 3)
LENGTH = 3e-6


class _TrackedMode(ChannelMode):
    # Values in spatial basis order. Slot 0 becomes radiation at the output.
    indices = ((2.0, 1.3, 1.8), (1.3, 1.6, 1.85))

    def getdata(self, card, name):
        solve = len(self.solves)
        if name == "neff" and solve in (2, 4):
            local = int(card.rsplit("mode", 1)[1]) - 1
            spatial = self.orders[solve - 1][local]
            return self.indices[(solve - 2) // 2][spatial]
        return super().getdata(card, name)


class ChannelRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        temporary = TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.path = Path(temporary.name) / "dataset"
        cls.model = StraightGapSectionModel(500e-9, 450e-9, WaveguideStack(220e-9, "si", "oxide", "oxide"))
        settings = CoupledDatasetSettings(GAPS, 1.44, FDEConfig(
            1.55e-6, (-2e-6, 2e-6), (-1e-6, 1e-6), (12, 10),
            {"si": "Si material", "oxide": "oxide material"},
            {"coupled": 3, "wg0": 3, "wg1": 1}, "Metal",
        ))
        updater = DataUpdater(cls.path, coupled_settings=settings)
        api = SimpleNamespace(MODE=Mock(return_value=_TrackedMode()))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ChannelMappingWarning)
                build_channel_data(updater, device(updater, cls.model, GAPS), mode_api=api)
        finally:
            updater.close_mode()

    def prepare(self, profile=None, *, limit=0, **channel_options):
        dataset = DataUpdater(self.path, is_testmode=True)
        profile = profile_from_samples([0, LENGTH], GAPS) if profile is None else profile
        geometry = StraightGapCoupler(dataset, self.model, profile, limit_mode_number=limit, verbose=False)
        eme = EME(geometry, stability_config=EMEStabilityConfig(pml_mode_sink=True), neff_interpolation=False)
        runner = Runner(eme)
        return eme, runner, ChannelRunner(runner, **channel_options)

    def test_full_basis_phase_and_radiation_readout_offline(self):
        no_mode = Mock(side_effect=AssertionError("Opened MODE during ChannelRunner execution"))
        with patch("em_simulation.data_updater.data_updater.lumapi", SimpleNamespace(MODE=no_mode), create=True):
            eme, runner, channels = self.prepare(guide_labels={"wg0": "bus", "wg1": "ring"})
            self.assertEqual(channels.available_channels("input"), (("wg0", 1), BUS_TM, RING))
            self.assertEqual(channels.available_channels("output"), (RING,))
            self.assertEqual(channels.available_channels("input", verbose=True)[1]["display_name"], "bus_TM0")
            self.assertEqual(channels.supermode_labels(0, include_polarization=True)[2], "mode 3 (TM)")
            ring, bus = 0.4 + 0.7j, -0.2j
            with patch.object(runner, "propagate_basis_amplitudes", wraps=runner.propagate_basis_amplitudes) as send:
                result = channels.run({RING: ring, BUS_TM: bus})
        no_mode.assert_not_called()

        launch = np.array([-ring, 0, bus])
        np.testing.assert_allclose(send.call_args.args[0], [*launch, 0, 0, 0], atol=1e-14)
        diagnostic = result.diagnostics.channel_mapping
        self.assertEqual(diagnostic.input_endpoint.guided_slots, (0, 2))
        self.assertEqual(diagnostic.output_endpoint.guided_slots, (1, 2))
        self.assertEqual(diagnostic.output_endpoint.forward_slots, (0, 1, 2))
        self.assertEqual(result.output_labels, {RING: "ring_TE0"})
        # Each orthogonal spatial mode has identity interface transmission.
        # The output canonical basis is permuted and slot 0 has a minus gauge.
        indices = np.asarray(_TrackedMode.indices)
        phase = np.exp(1j * 2 * np.pi / eme.propagator.geometry.wavelength * (indices.sum(axis=0) * LENGTH / 2))
        transmitted = phase * launch
        np.testing.assert_allclose(diagnostic.transmitted_forward, transmitted, atol=1e-13)
        B = np.array([-1, 0, 0])
        np.testing.assert_allclose(result.output_amplitudes[RING], B @ transmitted, atol=1e-13)
        guided_only = transmitted.copy()
        guided_only[0] = 0
        self.assertEqual(B @ guided_only, 0)
        self.assertGreater(diagnostic.evaluation.power.P3, 0)

    def test_absent_slots_snapping_clipping_and_legacy_compact_input(self):
        profile = profile_from_samples([0, LENGTH], [GAPS[0] - 10e-9, GAPS[-1] - 10e-9])
        with self.assertWarnsRegex(UserWarning, "profile clipped"):
            _, runner, channels = self.prepare(profile, limit=2)
        result = channels.run({RING: 1})
        diagnostic = result.diagnostics.channel_mapping
        self.assertEqual(diagnostic.input_endpoint.requested_parameters, {"gap": profile(0)})
        self.assertEqual(diagnostic.output_endpoint.requested_parameters, {"gap": profile(LENGTH)})
        self.assertEqual(diagnostic.input_endpoint.node, (GAPS[0],))
        self.assertEqual(diagnostic.output_endpoint.node, (GAPS[-1],))
        self.assertEqual(result.diagnostics.path_clipping.positions, (0.0,))
        self.assertEqual(diagnostic.input_endpoint.forward_slots, (0, 1))
        self.assertEqual(diagnostic.output_endpoint.forward_slots, (1, 2))
        self.assertEqual(diagnostic.transmitted_forward[0], 0)
        np.testing.assert_allclose(runner.propagate_lumped_smatrix([0.7j]),
                                   runner.propagate_basis_amplitudes([0.7j, 0, 0, 0, 0, 0]))
        with self.assertRaisesRegex(ValueError, "6 finite amplitudes"):
            runner.propagate_basis_amplitudes([0, 0])

    def test_input_normalization_invalid_inputs_and_result_snapshots(self):
        _, runner, channels = self.prepare(mapping_config=MappingConfig(condition_warning=1e8))
        raw = channels.run({RING: 2j, BUS_TM: 1})
        normalized = channels.run({RING: 2j, BUS_TM: 1}, input_power=2)
        info = normalized.diagnostics.channel_mapping
        self.assertEqual(info.input_amplitudes.raw.tolist(), [0, 1, 2j])
        np.testing.assert_allclose(info.input_amplitudes.effective, np.array([0, 1, 2j]) * np.sqrt(2 / 5))
        self.assertAlmostEqual(info.evaluation.power.P0, 2)
        self.assertEqual(raw.diagnostics.channel_mapping.input_amplitudes.scale, 1)
        output_snapshot = dict(normalized.output_amplitudes)
        runner.propagator.stability_diagnostics["feedback_fallbacks"].append({"index": -1})
        zero = channels.run({})
        self.assertFalse(zero.diagnostics.channel_mapping.evaluation.input_residuals.E.defined)
        self.assertEqual(normalized.output_amplitudes, output_snapshot)
        self.assertNotIn({"index": -1}, normalized.diagnostics.propagation_stability["feedback_fallbacks"])
        self.assertEqual(normalized.diagnostics.channel_mapping.input_amplitudes.raw.tolist(), [0, 1, 2j])
        for amplitudes, power, message in (
            ({("wg0", 2): 1}, None, "Unknown input channel"),
            ({RING: complex(np.nan, 0)}, None, "finite complex scalar"),
            ({RING: 1}, -1, "positive finite"),
            ({}, 1, "nonpositive or unreliable"),
        ):
            with self.subTest(amplitudes=amplitudes, power=power), self.assertRaisesRegex(ValueError, message):
                channels.run(amplitudes, input_power=power)
        with self.assertRaisesRegex(ValueError, "Duplicate guide labels"):
            ChannelRunner(runner, guide_labels={"wg0": "wg1"})
        warned = ChannelRunner(runner, mapping_config=MappingConfig(e_residual_warning=0.1))
        with self.assertWarnsRegex(ChannelMappingWarning, "relative E residual"):
            unmapped = warned.run({("wg0", 1): 1})
        self.assertTrue(any("relative E residual" in text for text in unmapped.diagnostics.channel_mapping.warnings))

    def test_length_change_refreshes_propagation_and_geometry_rebuild_is_detected(self):
        outside = GAPS[0] - 10e-9
        profile = profile_from_samples([0, LENGTH], [outside, outside])
        with self.assertWarnsRegex(UserWarning, "profile clipped"):
            eme, _, channels = self.prepare(profile)
        first = channels.run({RING: 1})
        repeat = channels.run({RING: 1})
        self.assertEqual(first.output_amplitudes, repeat.output_amplitudes)
        eme.propagator.stability_diagnostics["feedback_fallbacks"].append({"index": -1})
        shorter_length = LENGTH / 2
        eme.change_strucutre_length(shorter_length)
        second = channels.run({RING: 1})
        self.assertNotIn({"index": -1}, second.diagnostics.propagation_stability["feedback_fallbacks"])
        for length, result in ((LENGTH, first), (shorter_length, second)):
            expected = np.exp(1j * 2 * np.pi * 2.0 / eme.propagator.geometry.wavelength * length)
            np.testing.assert_allclose(result.output_amplitudes[RING], expected, atol=1e-13)
            self.assertAlmostEqual(result.diagnostics.channel_mapping.output_endpoint.s_m, length, places=14)
            clipping = result.diagnostics.path_clipping
            np.testing.assert_allclose(clipping.positions, [0, length], rtol=0, atol=1e-15)
            self.assertEqual(clipping.requested_values, (outside, outside))
            self.assertEqual(clipping.applied_values, (GAPS[0], GAPS[0]))
        np.testing.assert_array_equal(first.diagnostics.channel_mapping.output_endpoint.R,
                                      second.diagnostics.channel_mapping.output_endpoint.R)
        eme.propagator.geometry.calc_output_data()
        with self.assertRaisesRegex(ValueError, "Rebuild EME"):
            channels.available_channels("input")

    def test_dataset_reload_preserves_matching_generation(self):
        eme, _, channels = self.prepare()
        dataset = eme.propagator.geometry.data
        first = channels.run({RING: 1})
        dataset.load_data()
        self.assertEqual(channels.run({RING: 1}).output_amplitudes, first.output_amplitudes)

        with patch.dict(dataset.canonical_manifest["file_sha256"], {"overlap": "changed"}):
            with self.assertRaisesRegex(ValueError, "Rebuild EME"):
                channels.available_channels("input")

    def test_single_guide_compact_propagation_and_plots(self):
        # Minimal legacy dataset to exercise the normal single-guide geometry/Runner.
        point = (1e-6, 0.0)
        dataset = DataUpdater.__new__(DataUpdater)
        dataset.parameter_names = ["top_width", "curvature"]
        dataset.parameter_grid = {"top_width": np.array([point[0]]), "curvature": np.array([point[1]])}
        dataset.wavelength, dataset.mode_numbers = 1.55e-6, 1
        dataset.data_info = SimpleNamespace(
            get_cladding_index=lambda: 1.44, get_wavelength=lambda: dataset.wavelength,
        )
        dataset.neff = {point: np.array([2.0, -2.0])}
        dataset.TE_pol = {point: np.array([0.9, 0.9])}
        dataset.calc_data_point_modified = Mock(return_value=False)
        dataset.save_data = Mock()
        geometry = SingleStraightWaveguide(dataset, np.float64(point[0]), LENGTH)
        geometry._verbose = False
        runner = Runner(EME(geometry))
        expected = 0.6j * np.exp(1j * runner.propagator.beta_forward[0, 0] * LENGTH)
        np.testing.assert_allclose(runner.propagate_lumped_smatrix([0.6j]), [expected, 0], atol=1e-13)
        np.testing.assert_allclose(runner.propagate([0.6j])[-1], [expected, 0], atol=1e-7)
        self.addCleanup(plt.close, "all")
        self.assertIn("mode 1", runner.plot_complexplane())
        self.assertIn("mode1", runner.plot_intensity_along_propagation())


if __name__ == "__main__":
    unittest.main()
