"""MODE-free tests of coupled acquisition, persisted basis linkage and EME."""

from copy import deepcopy
import json
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from em_simulation import (
    CoupledDatasetSettings, DataUpdater, EME, EMEStabilityConfig, FDEConfig, Runner,
    StraightGapCoupler, StraightGapSectionModel, WaveguideStack, profile_from_samples,
)
from em_simulation.data_updater.canonical_dataset import MANIFEST
from em_simulation.data_updater.lumerical_section import LumericalSectionAdapter


GAPS = (100e-9, 200e-9, 300e-9)
Y = np.array([-1e-6, 0.0, 1e-6])
Z = np.array([-0.4e-6, 0.4e-6])
NEFF = np.array([1.2, 2.1 - 1e-6j, 1.3])


class MockMode:
    def __init__(self, orders=((0, 1, 2), (1, 0, 2), (2, 0, 1)), *, h_shift=0,
                 next_grid_shift=0, short_count=None, field_vectors=None):
        self.orders = orders
        self.h_shift = h_shift
        self.next_grid_shift = next_grid_shift
        self.short_count = short_count
        self.field_vectors = field_vectors
        self.objects = {}
        self.analysis = {}
        self.solves = []
        self.created = []
        self.closed = False

    def switchtolayout(self):
        pass

    def _add(self, kind):
        self.current = f"new_{len(self.created)}"
        self.objects[self.current] = {}
        self.created.append(kind)

    def addfde(self):
        self._add("fde")
        self.objects["FDE"] = self.objects.pop(self.current)
        self.current = "FDE"

    def addrect(self):
        self._add("rect")

    def addpoly(self):
        self._add("poly")

    def set(self, name, value):
        if name == "name":
            if self.current == "FDE":
                raise ValueError("FDE name is inactive")
            if value in self.objects:
                raise AssertionError(f"duplicate object {value}")
            self.objects[value] = self.objects.pop(self.current)
            self.current = value
        else:
            self.objects[self.current][name] = value

    def select(self, path):
        self.current = path.split("::")[-1]
        if self.current not in self.objects:
            raise AssertionError(f"unknown object {path}")

    def setanalysis(self, name, value):
        self.analysis[name] = value

    def findmodes(self):
        self.solves.append((deepcopy(self.objects), dict(self.analysis)))
        if self.short_count is not None:
            return self.short_count
        return len(self.orders[len(self.solves) - 1])

    def getresult(self, card, name):
        node = len(self.solves) - 1
        mode = int(card.split("mode")[-1]) - 1
        pos = self.orders[node][mode]
        if self.field_vectors is None:
            vector = np.eye(len(Y))[pos] * (-1 if node == 1 and mode == 0 else 1)
        else:
            vector = self.field_vectors[node][name]
        field = np.zeros((1, len(Y), len(Z), 1, 3), dtype=complex)
        field[0, :, :, 0, 1 if name == "E" else 2] = vector[:, None]
        shift = (self.h_shift if name == "H" else 0) + (self.next_grid_shift if node else 0)
        return {"x": np.array([[0.0]]), "y": (Y + shift)[:, None], "z": Z[:, None], name: field}

    def getdata(self, card, name):
        index = int(card.split("mode")[-1]) - 1
        if name == "neff":
            return np.array([[NEFF[self.orders[len(self.solves) - 1][index]]]])
        if name == "TE polarization fraction":
            return np.array([[0.95]])
        raise KeyError(name)

    def close(self):
        self.closed = True


def device(dataset, model, gaps, limit=0, cls=StraightGapCoupler):
    profile = profile_from_samples(np.linspace(0, 3e-6, len(gaps)), gaps)
    return cls(dataset, model, profile, limit_mode_number=limit, verbose=False)


def run_eme(geometry):
    return Runner(EME(geometry, neff_interpolation=False,
                      stability_config=EMEStabilityConfig(pml_mode_sink=False)))


class CoupledDatasetTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = StraightGapSectionModel(
            500e-9, 450e-9, WaveguideStack(220e-9, "si", "oxide", "oxide", slab_thickness=70e-9),
        )

    def settings(self, gaps=GAPS, count=3):
        return CoupledDatasetSettings(gaps, 1.44, FDEConfig(
            1.55e-6, (-2e-6, 2e-6), (-1e-6, 1e-6), (12, 10),
            {"si": "Si material", "oxide": "oxide material"},
            {"coupled": count, "wg0": 2, "wg1": 1}, "Metal",
        ))

    def acquire(self, mode, gaps=GAPS, count=3, folder="dataset"):
        path = self.root / folder
        settings = self.settings(gaps, count)
        updater = DataUpdater(path, coupled_settings=settings)
        geometry = device(updater, self.model, (gaps[0], gaps[-1]))
        factory = Mock(return_value=mode)
        records = []
        self.addCleanup(updater.close_mode)
        manifest = updater.acquire_coupled(
            geometry, mode_api=SimpleNamespace(MODE=factory), on_node=records.append,
        )
        factory.assert_called_once_with(hide=False)
        return updater, manifest, records

    def test_layout_reuses_regions_restores_both_guides_on_error(self):
        session = MockMode()
        settings = self.settings()
        adapter = LumericalSectionAdapter(session, settings.fde)
        a = self.model.describe_cross_section({"gap": GAPS[0]})
        b = self.model.describe_cross_section({"gap": GAPS[2]})
        adapter.prepare(a)
        self.assertEqual(session.objects["FDE"]["solver type"], "2D X normal")
        self.assertEqual(session.objects["FDE"]["mesh cells y"], 12)
        before = deepcopy(session.objects["dreme_stack_slab"])
        with adapter.isolated(a, "wg0"):
            self.assertEqual(session.analysis["number of trial modes"], 2)
            self.assertEqual(session.objects["dreme_core_wg1_core"]["enabled"], 0)
        with self.assertRaisesRegex(RuntimeError, "isolated failure"):
            with adapter.isolated(b, "wg1"):
                self.assertEqual(session.analysis["number of trial modes"], 1)
                self.assertEqual(session.objects["dreme_core_wg0_core"]["enabled"], 0)
                self.assertEqual(session.objects["dreme_core_wg1_core"]["enabled"], 1)
                raise RuntimeError("isolated failure")
        self.assertEqual(session.analysis["number of trial modes"], 3)
        self.assertEqual(session.analysis["wavelength"], 1.55e-6)
        self.assertEqual(session.created.count("poly"), 2)
        self.assertEqual(session.created.count("fde"), 1)
        self.assertEqual(before, session.objects["dreme_stack_slab"])
        for core in b.cores:
            obj = session.objects[f"dreme_core_{core.region_id}"]
            self.assertEqual((obj["enabled"], obj["material"], obj["rotation 1"]), (1, "Si material", 90))
            np.testing.assert_array_equal(np.column_stack((obj["vertices"][:, 1], -obj["vertices"][:, 0])),
                                          core.vertices)
        self.assertEqual(session.solves, [])
        self.assertFalse(session.closed)

    def test_acquires_once_and_persists_directed_links_to_exact_fields(self):
        mode = MockMode()
        updater, manifest, nodes = self.acquire(mode)
        self.assertEqual(len(mode.solves), 3)
        self.assertEqual([node.node for node in nodes], [(g,) for g in GAPS])
        self.assertTrue(all(snapshot["dreme_core_wg0_core"]["enabled"] == 1
                            and snapshot["dreme_core_wg1_core"]["enabled"] == 1
                            and analysis["number of trial modes"] == 3
                            for snapshot, analysis in mode.solves))
        self.assertFalse(mode.closed)
        self.assertEqual(manifest["export_grid"]["y"], tuple(Y))
        self.assertEqual(nodes[0].raw_neff[1], 2.1 - 1e-6j)
        self.assertEqual(nodes[0].neff[1], 2.1)
        self.assertFalse(nodes[0].E.flags.writeable)
        self.assertEqual(set(Path(updater.data_directory).iterdir()), {
            updater.data_directory / name for name in (MANIFEST, "neff.pkl", "TE_pol.pkl", "overlap.pkl")
        })
        disk = DataUpdater(updater.data_directory, is_testmode=True)
        self.assertEqual(disk.canonical_manifest["generation_id"], manifest["generation_id"])
        self.assertEqual(disk.section_model_configuration, self.model.configuration())
        for a, b in zip(nodes, nodes[1:]):
            for source, dest in ((a, b), (b, a)):
                self.assertEqual(disk.canonical_basis_ids[source.node], source.basis_id)
                expected = DataUpdater.calc_overlap(
                    np.concatenate((source.E, source.E.conj())),
                    np.concatenate((dest.H, -dest.H.conj())), Y, Z, 0,
                )
                np.testing.assert_allclose(disk.overlap[source.node][dest.node], expected, atol=1e-14)
        updater.close_mode()
        self.assertTrue(mode.closed)

    def test_runtime_selection_tracking_and_both_directions_from_fields(self):
        updater, _, fields = self.acquire(MockMode())
        disk = DataUpdater(updater.data_directory, is_testmode=True)
        by_node = {node.node: node for node in fields}
        for route in (GAPS, GAPS[::-1], GAPS + GAPS[-2::-1]):
            with self.subTest(route=route):
                geometry = device(disk, self.model, route, limit=2)
                output = geometry.calc_output_data()
                basis = output["runtime_basis"]
                self.assertEqual(geometry.runtime_basis_at(0).node, (route[0],))
                self.assertEqual(geometry.runtime_basis_at(-1).node, (route[-1],))
                self.assertTrue(any(np.any(~section.forward_present) for section in basis))
                runtime_fields = []
                for section in basis:
                    f = by_node[section.node]
                    self.assertEqual(section.basis_id, f.basis_id)
                    self.assertEqual(section.R.shape[0], 3)
                    np.testing.assert_array_equal(section.R[2], 0)
                    np.testing.assert_array_equal(np.any(section.R != 0, axis=0), section.forward_present)
                    E = np.einsum("ms,mcij->scij", section.R, f.E)
                    H = np.einsum("ms,mcij->scij", section.R, f.H)
                    runtime_fields.append((np.concatenate((E, E.conj())), np.concatenate((H, -H.conj()))))
                for i, ((ea, ha), (eb, hb)) in enumerate(zip(runtime_fields, runtime_fields[1:])):
                    np.testing.assert_allclose(output["overlap_ab"][i],
                                               DataUpdater.calc_overlap(ea, hb, Y, Z, 0), atol=1e-14)
                    np.testing.assert_allclose(output["overlap_ba"][i],
                                               DataUpdater.calc_overlap(eb, ha, Y, Z, 0), atol=1e-14)
                result = run_eme(device(disk, self.model, route)).propagate_lumped_smatrix([1])
                self.assertTrue(np.isfinite(result).all())

    def test_one_sign_transform_even_when_directed_diagonals_disagree(self):
        vectors = [
            {"E": np.array([1, 0, 0]), "H": np.array([1, 0, 0])},
            {"E": np.array([1, 2, 0]), "H": np.array([-1, 1, 0])},
        ]
        updater, _, _ = self.acquire(
            MockMode(orders=((1,), (1,)), field_vectors=vectors), gaps=GAPS[:2], count=1,
        )
        geometry = device(updater, self.model, GAPS[:2])
        out = geometry.calc_output_data()
        self.assertEqual(geometry.runtime_basis_at(1).R[0, 0], -1)
        self.assertAlmostEqual(out["overlap_ab"][0, 0, 0], 1)
        self.assertAlmostEqual(out["overlap_ba"][0, 0, 0], -1)

    def test_uniform_complex_phase_repeated_node_and_no_mode_open_or_write(self):
        updater, _, _ = self.acquire(MockMode())
        path = updater.data_directory
        blocked = Mock(side_effect=AssertionError("Offline execution opened MODE"))
        before = {f.name: f.read_bytes() for f in path.iterdir()}

        class SplitUniform(StraightGapCoupler):
            def calc_simulation_parameters(self):
                return [(GAPS[0],)] * 4, np.array([0.5e-6, 1e-6, 1.5e-6])

        with patch("em_simulation.data_updater.data_updater.lumapi", SimpleNamespace(MODE=blocked), create=True):
            disk = DataUpdater(path)
            self.assertIsNone(disk.mode)
            with patch.object(disk, "save_data", side_effect=AssertionError("Offline write")):
                for cls in (StraightGapCoupler, SplitUniform):
                    geometry = device(disk, self.model, (GAPS[0], GAPS[0]), cls=cls)
                    result = run_eme(geometry).propagate_lumped_smatrix([0.4 + 0.7j])
                    self.assertEqual(geometry.runtime_basis_at(0).guided_slots.tolist(), [1])
                    expected = np.zeros(6, complex)
                    expected[1] = (0.4 + 0.7j) * np.exp(1j * 2 * np.pi * 2.1 / disk.wavelength * 3e-6)
                    np.testing.assert_allclose(result, expected, atol=1e-12)
        blocked.assert_not_called()
        self.assertEqual(before, {f.name: f.read_bytes() for f in path.iterdir()})

    def test_missing_node_edge_and_configuration_report_context(self):
        updater, _, _ = self.acquire(MockMode())
        path = updater.data_directory
        disk = DataUpdater(path, is_testmode=True)
        different = StraightGapSectionModel(520e-9, 450e-9, self.model.stack)
        with self.assertRaisesRegex(ValueError, r"guides\.wg0\.width"):
            device(disk, different, GAPS)
        del disk.neff[(GAPS[0],)]
        with self.assertRaisesRegex(ValueError, "neff at node"):
            run_eme(device(disk, self.model, GAPS))
        data = pickle.loads((path / "overlap.pkl").read_bytes())
        del data[(GAPS[1],)][(GAPS[0],)]
        (path / "overlap.pkl").write_bytes(pickle.dumps(data))
        with self.assertRaisesRegex(ValueError, "overlap edge .* ->"):
            DataUpdater(path, is_testmode=True)

    def test_invalid_basis_linkage_and_mixed_generation_fingerprint(self):
        updater, _, _ = self.acquire(MockMode())
        path = updater.data_directory
        manifest = path / MANIFEST
        original = manifest.read_bytes()
        bad = json.loads(original)
        bad["edges"][0]["destination_basis_id"] = "wrong"
        manifest.write_text(json.dumps(bad), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "basis linkage at edge"):
            DataUpdater(path)
        manifest.write_bytes(original)
        edge_path = path / "overlap.pkl"
        edges = pickle.loads(edge_path.read_bytes())
        edges[(GAPS[0],)][(GAPS[1],)][0, 0] += 0.1
        edge_path.write_bytes(pickle.dumps(edges))
        with self.assertRaisesRegex(ValueError, "file fingerprints.*overlap.pkl"):
            DataUpdater(path)

    def test_mode_shortfall_and_incompatible_grids_abort(self):
        for label, mode, pattern in (
            ("short", MockMode(short_count=2), "node .*requested 3.*returned 2"),
            ("eh_grid", MockMode(h_shift=1e-9), "node .*grid mismatch on y.*mode1 H"),
            ("node_grid", MockMode(next_grid_shift=1e-9), "node .*grid mismatch on y.*coupled node"),
        ):
            with self.subTest(label=label):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.acquire(mode, folder=label)
                with self.assertRaisesRegex(ValueError, "is incomplete"):
                    DataUpdater(self.root / label, is_testmode=True)


if __name__ == "__main__":
    unittest.main()
