"""Commit 4 tests with MODE-shaped synthetic fields and a borrowed mock session."""

from dataclasses import replace
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from em_simulation import (
    CoupledDatasetSettings, DataUpdater, FDEConfig, StraightGapSectionModel, WaveguideStack,
)
from em_simulation.channels import ForwardFields, SectionGrid
from em_simulation.channels.diagnostics import compute_field_products
from em_simulation.channels.mapping import ChannelMappingWarning, MappingConfig, compute_power_products
from em_simulation.data_updater import channel_data_builder, channel_store
from em_simulation.data_updater.channel_data_builder import build_channel_data
from em_simulation.test_supermode_dataset import MockMode, device


GAPS = (150e-9, 250e-9)


class ChannelMode(MockMode):
    def __init__(self, *, fail_at=None, **options):
        super().__init__(orders=((0,), (0, 1, 2), (1, 0, 2), (2, 1, 0), ()), **options)
        self.fail_at = fail_at

    def findmodes(self):
        result = super().findmodes()
        if len(self.solves) == self.fail_at:
            raise RuntimeError("simulated FDE error")
        return result

    def getresult(self, card, name):
        if len(self.solves) == 3 and card.endswith("mode2"):
            raise AssertionError("filtered mode fields must not be read")
        return super().getresult(card, name)

    def getdata(self, card, name):
        if name == "neff" and len(self.solves) in (1, 3):
            # Both guided cutoffs are inclusive. Keep a TM-like candidate too.
            number = int(card.split("mode")[-1])
            return (2.0, 1.44 + 0.000285j, 1.44 - 0.000284j)[number - 1]
        if len(self.solves) == 3 and card.endswith("mode2"):
            raise AssertionError("filtered mode TE fraction must not be read")
        if name == "TE polarization fraction" and card.endswith("mode3"):
            return 0.1
        return super().getdata(card, name)


class ChannelDatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "dataset"
        self.model = StraightGapSectionModel(500e-9, 450e-9, WaveguideStack(220e-9, "si", "oxide", "oxide"))
        self.settings = CoupledDatasetSettings(GAPS, 1.44, FDEConfig(
            1.55e-6, (-2e-6, 2e-6), (-1e-6, 1e-6), (12, 10),
            {"si": "Si material", "oxide": "oxide material"},
            {"coupled": 3, "wg0": 3, "wg1": 1}, "Metal",
        ))

    def setup_build(self, mode):
        updater = DataUpdater(self.path, coupled_settings=self.settings)
        self.addCleanup(updater.close_mode)
        geometry = device(updater, self.model, GAPS)
        api = SimpleNamespace(MODE=Mock(return_value=mode))
        return updater, geometry, api

    def test_single_generation_and_offline_read(self):
        mode = ChannelMode()
        updater, geometry, api = self.setup_build(mode)
        later_writer = DataUpdater(self.path, coupled_settings=self.settings)
        with self.assertWarnsRegex(ChannelMappingWarning, "guided count.*wg0.*from 2.*to 0"):
            manifest = build_channel_data(updater, geometry, mode_api=api)
        api.MODE.assert_called_once_with(hide=False)
        self.assertIs(updater.mode, mode)
        self.assertFalse(mode.closed)
        self.assertEqual(len(mode.solves), 5)
        self.assertEqual(len(manifest["edges"]), 2)
        self.assertEqual([
            (objects["dreme_core_wg0_core"]["enabled"], objects["dreme_core_wg1_core"]["enabled"],
             analysis["number of trial modes"])
            for objects, analysis in mode.solves
        ], [(0, 1, 1), (1, 1, 3), (1, 0, 3), (1, 1, 3), (1, 0, 3)])
        self.assertEqual(mode.analysis["number of trial modes"], 3)
        self.assertTrue(mode.objects["dreme_core_wg1_core"]["enabled"])
        with self.assertRaises(FileExistsError):
            build_channel_data(later_writer, device(later_writer, self.model, GAPS), mode_api=api)
        api.MODE.assert_called_once_with(hide=False)

        offline = DataUpdater(self.path, is_testmode=True)
        data = offline.load_channel_data(self.model)
        self.assertIsNone(offline.mode)
        self.assertEqual(data["generation_completion"], "complete")
        first, second = (data["nodes"][(gap,)] for gap in GAPS)
        self.assertEqual(first["canonical_basis_id"], offline.canonical_basis_ids[(GAPS[0],)])
        self.assertEqual([c["key"] for c in first["channels"]], [("wg0", 1), ("wg0", 3), ("wg1", 1)])
        self.assertEqual(first["channels"][1]["neff"], 1.44 - 0.000284j)
        self.assertEqual(first["channels"][1]["te_fraction"], 0.1)
        self.assertEqual(second["channels"], first["channels"][-1:])
        self.assertEqual(second["guide_counts"]["wg0"]["empty_reason"], "MODE returned no modes")
        np.testing.assert_allclose(np.diag(first["power_products"]["G_V"]), 1, atol=1e-12)
        self.assertEqual(second["field_products"]["H"]["X"].shape, (3, 1))
        self.assertIn("direct_power_w", first["baseline_diagnostics"])

    def test_baseline_keeps_ill_conditioned_solve_with_expected_roundoff(self):
        grid = SectionGrid(0, (0, 1), (0, 1), 1.55e-6, "collocated")
        E = np.zeros((2, 3, 2, 2))
        H = np.zeros_like(E)
        unit = np.sqrt(2)
        E[0, 1, 0, 0] = H[0, 2, 0, 0] = unit
        E[1, 1, 1, 0] = H[1, 2, 1, 0] = unit
        V = ForwardFields(E[1:].copy(), H[1:].copy(), grid)
        # The second coupled mode is almost parallel to the first. Reconstructing
        # V therefore cancels two large coupled coefficients.
        epsilon = 3e-6
        E[1] = (E[0] + epsilon * E[1]) / np.sqrt(1 + epsilon**2)
        H[1] = (H[0] + epsilon * H[1]) / np.sqrt(1 + epsilon**2)
        U = ForwardFields(E, H, grid)
        power, fields = compute_power_products(U, V), compute_field_products(U, V)
        config = MappingConfig(condition_warning=1e9)
        guided = np.ones(2, dtype=bool)
        with self.assertWarnsRegex(ChannelMappingWarning, "roundoff"):
            baseline = channel_data_builder._baseline(U, V, power, fields, guided, config, "test node")
        self.assertEqual(baseline["input_solve"]["method"], "truncated_svd")

        with self.assertWarns(ChannelMappingWarning), \
                patch.object(channel_data_builder, "_direct_power", return_value=10), \
                self.assertRaisesRegex(ValueError, "Poynting power mismatch"):
            channel_data_builder._baseline(U, V, power, fields, guided, config, "test node")

    def test_failed_solve_or_save_preserves_only_the_previous_checkpoint(self):
        for failure in ("solve", "save"):
            with self.subTest(failure=failure):
                self.path = self.path.parent / failure
                mode = ChannelMode(fail_at=5 if failure == "solve" else None)
                updater, geometry, api = self.setup_build(mode)
                write = channel_store._atomic_write

                def save(path, payload):
                    if len(pickle.loads(payload)["nodes"]) == 2:
                        raise OSError("simulated disk error")
                    write(path, payload)

                effect = save if failure == "save" else write
                message = "isolated guide.*wg0.*simulated FDE error" if failure == "solve" else "save channel node.*simulated disk error"
                with patch.object(channel_store, "_atomic_write", side_effect=effect), \
                        self.assertRaisesRegex(RuntimeError, message):
                    build_channel_data(updater, geometry, mode_api=api)
                saved = channel_store.read_channel_checkpoint(self.path)
                self.assertEqual(saved["generation_completion"], "building")
                self.assertEqual(list(saved["nodes"]), [(GAPS[0],)])
                self.assertEqual(saved["nodes"][(GAPS[0],)]["completion"], "complete")
                self.assertEqual(mode.analysis["number of trial modes"], 3)
                self.assertTrue(mode.objects["dreme_core_wg1_core"]["enabled"])
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    DataUpdater(self.path, is_testmode=True)

    def test_shared_grid_mismatch_fails_with_guide_and_node(self):
        updater, geometry, api = self.setup_build(ChannelMode(next_grid_shift=1e-8))
        with self.assertRaisesRegex(RuntimeError, r"isolated guide 'wg1'.*node.*grid mismatch on y"):
            build_channel_data(updater, geometry, mode_api=api)
        self.assertFalse((self.path / channel_store.CHANNEL_FILE).exists())

    def test_incompatible_generation_config_basis_schema_and_missing_file(self):
        updater, geometry, api = self.setup_build(ChannelMode())
        with self.assertWarns(ChannelMappingWarning):
            build_channel_data(updater, geometry, mode_api=api)
        offline = DataUpdater(self.path, is_testmode=True)
        path = self.path / channel_store.CHANNEL_FILE
        with self.assertRaisesRegex(ValueError, "Section model configuration mismatch"):
            offline.load_channel_data(replace(self.model, wg0_width=510e-9))
        original = path.read_bytes()
        for field, error in (("generation_id", "generation_id"), ("settings", "settings"),
                             ("basis", "invalid channel record"), ("schema_version", "schema version"),
                             ("fingerprint", "file fingerprint")):
            with self.subTest(field=field):
                data = pickle.loads(original)
                if field in ("generation_id", "settings"):
                    data["dataset_metadata"][field] = "foreign"
                elif field == "basis":
                    data["nodes"][(GAPS[0],)]["canonical_basis_id"] = "foreign"
                elif field == "schema_version":
                    data[field] = -1
                else:
                    data["nodes"][(GAPS[0],)]["power_products"]["G_V"][0, 0] = 2
                path.write_bytes(pickle.dumps(data))
                with self.assertRaisesRegex(ValueError, error):
                    offline.load_channel_data(self.model)
        path.unlink()
        with self.assertRaisesRegex(FileNotFoundError, "Missing channel data file"):
            offline.load_channel_data(self.model)


if __name__ == "__main__":
    unittest.main()
