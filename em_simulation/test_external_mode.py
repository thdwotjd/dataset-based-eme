"""Verify externally acquired MODE data without starting a real MODE session."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np

from em_simulation import CoupledDatasetSettings, DataUpdater, FDEConfig, StraightGapSectionModel, WaveguideStack
from em_simulation.data_updater.external_mode import GROUP, SWEEP, ModeResult, prepare_mode_jobs
from em_simulation.test_supermode_dataset import GAPS, MockMode, device, run_eme


class ProjectMode(MockMode):
    def __init__(self, files):
        super().__init__()
        self.files = files
        self.sweep = {}

    def addstructuregroup(self):
        self._add("group")
        self.objects[self.current].update(x=0, y=0, z=0)

    def adduserprop(self, key, kind, value):
        self.objects[self.current][key] = value

    def addtogroup(self, path):
        self.objects[path.split("::model::", 1)[-1] + "::" + self.current] = self.objects.pop(self.current)

    def getnamed(self, path, prop):
        return self.objects[path.split("::model::", 1)[-1]][prop]

    def setnamed(self, path, prop, value):
        self.objects[path.split("::model::", 1)[-1]][prop] = value

    def getanalysis(self, name):
        return self.analysis[name]

    def runsetup(self):
        script = self.getnamed(GROUP, "script")
        array = re.search(r"dreme_v = \[(.*?)\]", script)[1]
        base = float(re.search(r"gap - \(([^)]+)\)", script)[1])
        name = re.search(r'select\("([^"]+)"\)', script)[1]
        vertices = np.array([[float(value) for value in row.split(",")] for row in array.split(";")])
        vertices[:, 1] += self.getnamed(GROUP, "gap") - base
        self.setnamed(f"{GROUP}::{name}", "vertices", vertices)

    def addsweep(self, kind):
        self.sweep = {"results": []}

    def setsweep(self, name, key, value):
        self.sweep[key] = value

    def addsweepparameter(self, name, param):
        self.sweep["parameter"] = param

    def addsweepresult(self, name, result):
        self.sweep["results"].append(result)

    def save(self, path):
        Path(path).touch()
        self.files[path] = deepcopy((self.objects, self.analysis, self.sweep))

    def load(self, path):
        self.objects, self.analysis, self.sweep = deepcopy(self.files[path])

    def nummodes(self):
        return len(self.orders[self.analysis["result_index"]]) if "result_index" in self.analysis else 0

    def getresult(self, card, name):
        self.solves = [None] * (self.analysis["result_index"] + 1)
        return super().getresult(card, name)

    def getdata(self, card, name):
        self.solves = [None] * (self.analysis["result_index"] + 1)
        return super().getdata(card, name)

    def havesweepresult(self, name, key):
        return key in self.sweep.get("values", {})

    def getsweepresult(self, name, key):
        return self.sweep["values"][key]

    def loadsweep(self, name):
        pass

    def putv(self, name, value):
        self.variable = value

    def loaddata(self, path):
        self.variable = deepcopy(self.files[path])

    def getv(self, name):
        return self.variable

    def findmodes(self):
        raise AssertionError("external workflow must not solve")

    def runsweep(self, name):
        raise AssertionError("external workflow must not run a sweep")


class ExternalModeTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.settings = CoupledDatasetSettings(GAPS, 1.44, FDEConfig(
            1.55e-6, (-2e-6, 2e-6), (-1e-6, 1e-6), (12, 10),
            {"si": "Si material", "oxide": "oxide material"},
            {"coupled": 3, "wg0": 2, "wg1": 1}, "Metal",
        ))
        self.model = StraightGapSectionModel(
            500e-9, 450e-9, WaveguideStack(220e-9, "si", "oxide", "oxide", slab_thickness=70e-9),
        )
        self.files = {}
        self.sessions = []

        def mode_session(hide):
            session = ProjectMode(self.files)
            self.sessions.append(session)
            return session

        self.api = SimpleNamespace(MODE=Mock(side_effect=mode_session))

    def geometry(self, name):
        return device(DataUpdater(self.root / name, coupled_settings=self.settings), self.model, GAPS)

    def prepare(self, workflow, batches=None, folder=None, gap_grid=None):
        geometry = self.geometry("planned")
        if gap_grid is not None:
            subset = DataUpdater(self.root / "subset", coupled_settings=replace(self.settings, gap_grid=gap_grid))
            geometry = device(subset, self.model, (gap_grid[0], gap_grid[-1]))
        plan = prepare_mode_jobs(geometry, self.root / (folder or workflow),
                                 mode_api=self.api, workflow=workflow, batches=batches)
        return plan, json.loads(plan.read_text(encoding="utf-8"))["jobs"]

    def node_result(self, plan, job, point, extension=".ldf", grid_shift=0):
        node = GAPS.index(point["gap_m"])
        raw = MockMode(next_grid_shift=grid_shift)
        raw.solves = [None] * (node + 1)
        if extension == ".lms":
            path = plan.parent / ("renamed_solved_" + str(node) + extension)
            layout, analysis, sweep = deepcopy(self.files[str(plan.parent / job["project"])])
            layout["dreme_gap"]["gap"] = point["gap_m"]
            for obj, prop, value in point["layout"]:
                if obj is not None:
                    layout[obj.split("::model::", 1)[-1]][prop] = value
            analysis["result_index"] = node
            self.files[str(path)] = (layout, analysis, sweep)
        else:
            path = plan.parent / ("reordered_" + str(node) + extension)
            data = {"gap": np.array([[point["gap_m"]]]), "job_id": job["id"], "returned": 3}
            for i, (_, _, value) in enumerate(point["layout"]):
                data[f"p{i}"] = value
            for i in range(1, 4):
                card = f"FDE::data::mode{i}"
                for name in ("E", "H"):
                    data[f"mode{i}_{name}"] = raw.getresult(card, name)
                data[f"mode{i}_neff"] = raw.getdata(card, "neff")
                data[f"mode{i}_TEpol"] = raw.getdata(card, "TE polarization fraction")
            self.files[str(path)] = data
        path.touch()
        return ModeResult(path, plan)

    def sweep_result(self, plan, job):
        path = plan.parent / job["project"]
        _, _, sweep = self.files[str(path)]
        gaps = [p["gap_m"] for p in job["points"]]
        raw = MockMode()
        values = {}
        for i in range(1, 4):
            card = f"FDE::data::mode{i}"
            for kind in ("E", "H", "neff", "TEpol", "x", "y", "z"):
                # MODE result order is explicit and can differ per result.
                order = gaps[::-1] if kind in ("E", "TEpol", "y") else gaps
                key = f"mode{i}_{kind}"
                values[key] = {"gap": np.array(order)[:, None]}
                parts = []
                for gap in order:
                    raw.solves = [None] * (GAPS.index(gap) + 1)
                    if kind in ("neff", "TEpol"):
                        attr = "neff" if kind == "neff" else "TE polarization fraction"
                        parts.append(raw.getdata(card, attr).item())
                    else:
                        result = raw.getresult(card, kind if kind in ("E", "H") else "E")
                        if kind in ("E", "H"):
                            parts.append(result[kind])
                        else:
                            parts.append(result[kind])
                if kind in ("E", "H"):
                    values[key].update({axis: result[axis] for axis in "xyz"})
                    values[key][kind] = np.concatenate(parts, axis=3)
                elif kind in "xyz":
                    values[key][kind] = np.concatenate(parts, axis=1)
                else:
                    values[key][attr] = np.asarray(parts)[:, None]
        sweep["values"] = values
        return ModeResult(path, plan, kind="sweep")

    def import_results(self, name, results):
        geo = self.geometry(name)
        manifest = geo.data.import_coupled(geo, results, mode_api=self.api)
        self.assertTrue(all(session.closed for session in self.sessions))
        disk = DataUpdater(geo.data.data_directory, is_testmode=True)
        return disk, manifest

    def test_sweep_and_files_share_the_python_canonical_dataset(self):
        py = self.geometry("python")
        py.data.acquire_coupled(py, mode_api=SimpleNamespace(MODE=Mock(return_value=MockMode())))
        self.addCleanup(py.data.close_mode)
        file_plan, jobs = self.prepare("files")
        files = [self.node_result(file_plan, job, job["points"][0], extension=ext)
                 for job, ext in zip(jobs, (".ldf", ".lms", ".ldf"))]
        file_data, _ = self.import_results("from_files", files[::-1])

        # Two separate saved Sweep projects plus their independent plan records.
        plan_a, jobs_a = self.prepare("sweep", (GAPS[:2],), "sweep_a")
        plan_b, jobs_b = self.prepare("sweep", folder="sweep_b", gap_grid=GAPS[2:])
        parent_a = self.sweep_result(plan_a, jobs_a[0])
        parent_b = self.sweep_result(plan_b, jobs_b[0])
        sweep_data, manifest = self.import_results("from_sweeps", [parent_b, parent_a])
        self.assertEqual([entry["node"] for entry in manifest["nodes"]], [(g,) for g in GAPS])
        self.assertEqual(len(manifest["edges"]), 4)
        self.assertEqual(len({entry["source"] for entry in manifest["nodes"]}), 2)
        _, _, settings = self.files[str(parent_a.path)]
        self.assertEqual(settings["name"], SWEEP)
        self.assertEqual(settings["parameter"]["Parameter"], f"{GROUP}::gap")
        self.assertEqual(settings["parameter"]["value_2"], GAPS[1])
        self.assertEqual(len(settings["results"]), 3 * 7)

        for data in (file_data, sweep_data):
            for node in py.data.neff:
                np.testing.assert_array_equal(data.neff[node], py.data.neff[node])
                np.testing.assert_array_equal(data.TE_pol[node], py.data.TE_pol[node])
                self.assertEqual(set(data.overlap[node]), set(py.data.overlap[node]))
                for neighbor in py.data.overlap[node]:
                    np.testing.assert_array_equal(data.overlap[node][neighbor], py.data.overlap[node][neighbor])
        for source, destination in ((GAPS[1], GAPS[2]), (GAPS[2], GAPS[1])):
            self.assertIn((destination,), sweep_data.overlap[(source,)])
        a = run_eme(device(file_data, self.model, GAPS)).propagate_lumped_smatrix([1])
        b = run_eme(device(sweep_data, self.model, GAPS)).propagate_lumped_smatrix([1])
        np.testing.assert_allclose(a, b)

    def test_missing_duplicate_wrong_layout_and_grid_are_reported(self):
        plan, jobs = self.prepare("files")
        files = [self.node_result(plan, job, job["points"][0]) for job in jobs]
        for name, results, regex in (
            ("missing", files[:2], "Missing coupled results.*3e-07"),
            ("duplicate", files + files[:1], "duplicate gap.*reordered_0"),
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, regex):
                self.import_results(name, results)
        data = self.files[str(files[0].path)]
        key = next(f"p{i}" for i, (_, prop, _) in enumerate(jobs[0]["points"][0]["layout"])
                   if prop == "mesh cells y")
        original = data[key]
        data[key] = 999
        with self.assertRaisesRegex(ValueError, "reordered_0.*mesh cells y.*999"):
            self.import_results("wrong_mesh", files)
        data[key] = original
        files[-1] = self.node_result(plan, jobs[-1], jobs[-1]["points"][0], grid_shift=1e-9)
        with self.assertRaisesRegex(RuntimeError, "reordered_2.*export grid mismatch"):
            self.import_results("wrong_grid", files)

    def test_invalid_aggregate_gives_child_export_recovery(self):
        plan, jobs = self.prepare("sweep")
        result = self.sweep_result(plan, jobs[0])
        sweep = self.files[str(result.path)][2]["values"]
        E = sweep["mode1_E"]["E"]
        sweep["mode1_E"]["E"] = E.swapaxes(2, 3)
        with self.assertRaisesRegex(ValueError, r"mode1_E shape.*solved sweep child \.lms"):
            self.import_results("wrong_dims", [result])
        sweep["mode1_E"]["E"] = E
        sweep["mode1_H"]["gap"] = np.array(GAPS[:2])
        with self.assertRaisesRegex(ValueError, "mode1_H missing gaps.*prepared export"):
            self.import_results("wrong_gap_axis", [result])
