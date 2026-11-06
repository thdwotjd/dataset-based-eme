"""Solver-free tests for the coupled-waveguide section and longitudinal path."""

from dataclasses import asdict, replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from matplotlib import pyplot as plt
import numpy as np

from em_simulation import DoubleWaveguide, StraightGapCoupler, StraightGapSectionModel, WaveguideStack, profile_from_samples
from em_simulation.geometry.coupled_waveguide.cross_section import CoreRegion
from em_simulation.geometry.geometry import Geometry
from em_simulation.geometry.parameter_path import build_parameter_path


class ParameterPathTests(unittest.TestCase):
    def test_uniform_gap_keeps_length_and_uses_lower_tie(self):
        path = build_parameter_path("gap", [1, 3], 2, length=6)
        self.assertEqual(path.nodes, ((1,), (1,)))
        self.assertEqual(path.node_positions, (0, 6))
        np.testing.assert_array_equal(path.distances, [6])
        path = build_parameter_path("gap", [2], 2, length=6)
        self.assertEqual(path.nodes, ((2,), (2,)))

    def test_approach_hold_departure_crosses_every_node_and_preserves_dwell(self):
        profile = profile_from_samples([0, 2, 7, 9], [4, 0, 0, 4])
        path = build_parameter_path("gap", [0, 1, 2, 3, 4], profile, resolution=2)
        self.assertEqual([node[0] for node in path.nodes], [4, 3, 2, 1, 0, 1, 2, 3, 4, 4])
        np.testing.assert_allclose(path.node_positions, [0, .25, .75, 1.25, 1.75, 7.25, 7.75, 8.25, 8.75, 9])
        self.assertEqual(sum(path.distances), 9)

    def test_reference_plane_ties_preserve_both_endpoints(self):
        profile = profile_from_samples([0, 1, 2], [2, 3, 2])
        path = build_parameter_path("gap", [1, 3], profile)
        self.assertEqual(path.nodes, ((1,), (3,), (1,)))
        np.testing.assert_array_equal(path.distances, [0, 2])
        self.assertEqual(path.values_at(0, "snapped"), 1)
        self.assertEqual(path.values_at(0.5, "snapped"), 3)
        self.assertEqual(path.values_at(2, "snapped"), 1)

    def test_explicit_length_truncates_without_rescaling(self):
        profile = profile_from_samples([0, 10], [1, 3])
        path = build_parameter_path("gap", [1, 2, 3], profile, length=4)
        self.assertEqual(path.sample_positions, (0, 4))
        np.testing.assert_allclose(path.requested_values, [1, 1.8])
        np.testing.assert_allclose(path.node_positions, [0, 2.5, 4])
        with self.assertRaisesRegex(ValueError, "exceeds the sample domain"):
            build_parameter_path("gap", [1, 2, 3], profile, length=11)

    def test_clipping_retains_crossing_locations_and_diagnostics(self):
        with self.assertWarnsRegex(UserWarning, "gap profile clipped"):
            path = build_parameter_path("gap", [1, 2, 3], profile_from_samples([0, 8], [0, 4]))
        np.testing.assert_allclose(path.node_positions, [0, 3, 5, 8])
        self.assertEqual(path.clipping.positions, (0, 8))
        self.assertEqual(path.clipping.requested_values, (0, 4))
        self.assertEqual(path.clipping.applied_values, (1, 3))
        self.assertEqual(path.clipping.bounds, (1, 3))
        self.assertEqual(path.plot_points("clipped"), ([0, 2, 6, 8], [1, 1, 3, 3]))

    def test_smooth_callable_crossing_converges_with_resolution(self):
        def profile(s):
            return s * s

        exact_crossing = np.sqrt(0.5)
        coarse = build_parameter_path("curvature", [0, 1], profile, length=1, resolution=3)
        fine = build_parameter_path("curvature", [0, 1], profile, length=1, resolution=33)
        self.assertLess(abs(fine.node_positions[1] - exact_crossing),
                        abs(coarse.node_positions[1] - exact_crossing))
        self.assertAlmostEqual(fine.node_positions[1], exact_crossing, places=3)

    def test_invalid_samples_and_profile_inputs(self):
        for positions, values in (
            ([1, 2], [1, 2]), ([0, 2, 1], [1, 2, 3]), ([0, 0], [1, 2]),
            ([0, 1], [1]), ([0, np.inf], [1, 2]), ([0, 1], [1, np.nan]),
            ([[0, 1]], [[1, 2]]),
        ):
            with self.subTest(positions=positions, values=values), self.assertRaises(ValueError):
                profile_from_samples(positions, values)
        for profile, length in ((lambda s: np.nan, 2), (np.inf, 2), (1, None), (1, 0), (1, np.inf)):
            with self.subTest(profile=profile, length=length), self.assertRaises(ValueError):
                build_parameter_path("gap", [1, 2], profile, length=length)


class DoubleWaveguideTests(unittest.TestCase):
    def test_section_contract_uses_model_ids_without_gap_or_fixed_axes(self):
        dataset = SimpleNamespace(
            parameter_names=["width"], parameter_grid={"width": [500e-9]},
            wavelength=1.55e-6, crosssection_x="x", crosssection_y="z",
        )
        vertices = ((0.0, 0.0), (500e-9, 0.0), (500e-9, 220e-9), (0.0, 220e-9))
        stack = WaveguideStack(220e-9, "silicon", "oxide", "oxide")
        section = SimpleNamespace(stack=stack, cores=(
            CoreRegion("region_a", "guide_a", "silicon", vertices),
            CoreRegion("region_b", "guide_b", "silicon",
                       tuple((x + 700e-9, z) for x, z in vertices)),
        ))
        model = SimpleNamespace(describe_cross_section=Mock(return_value=section))

        device = DoubleWaveguide(dataset, model)
        self.assertIsInstance(device, Geometry)
        parameters = {"width": 500e-9}
        result = device.describe_cross_section(parameters)
        model.describe_cross_section.assert_called_once_with(parameters)
        self.assertIs(result, section)
        self.assertEqual([(core.guide_id, core.region_id) for core in result.cores],
                         [("guide_a", "region_a"), ("guide_b", "region_b")])


class StraightGapCouplerTests(unittest.TestCase):
    def setUp(self):
        self.stack = WaveguideStack(220e-9, "silicon", "oxide", "oxide", slab_thickness=70e-9)
        self.model = StraightGapSectionModel(
            wg0_width=500e-9, wg1_width=450e-9, stack=self.stack, wg1_center_y=1e-6,
        )
        self.dataset = SimpleNamespace(
            parameter_names=["gap"], parameter_grid={"gap": [100e-9, 200e-9, 300e-9]},
            wavelength=1.55e-6, crosssection_x="y", crosssection_y="z",
        )
        self.addCleanup(plt.close, "all")

    def test_section_and_drawing_share_core_polygon_bounds(self):
        device = StraightGapCoupler(
            self.dataset, self.model,
            gap_profile=profile_from_samples([0, 5e-6], [100e-9, 300e-9]),
        )
        self.assertIsInstance(device, DoubleWaveguide)
        nodes, distances = device.calc_simulation_parameters()
        self.assertEqual(nodes, [(100e-9,), (200e-9,), (300e-9,), (300e-9,)])
        np.testing.assert_allclose(distances, [1.25e-6, 2.5e-6, 1.25e-6], atol=1e-15, rtol=0)
        for s, gap in ((0, 100e-9), (5e-6, 300e-9)):
            section = device.describe_cross_section({"gap": gap})
            self.assertEqual(section, self.model.describe_cross_section({"gap": gap}))
            bus, ring = section.cores
            self.assertEqual((bus.guide_id, ring.guide_id), ("wg0", "wg1"))
            self.assertAlmostEqual(np.mean(ring.lateral_bounds), 1e-6, places=15)
            self.assertAlmostEqual(bus.lateral_bounds[0] - ring.lateral_bounds[1], gap, places=15)
            self.assertAlmostEqual(np.diff(bus.lateral_bounds)[0], 500e-9, places=15)
            self.assertAlmostEqual(np.diff(ring.lateral_bounds)[0], 450e-9, places=15)
            self.assertAlmostEqual(bus.vertices[0][1], 70e-9, places=15)
            self.assertAlmostEqual(bus.vertices[2][1], 220e-9, places=15)
            ax = device.plot_cross_section(s)
            for patch, core in zip(ax.patches, section.cores):
                np.testing.assert_allclose(patch.get_xy()[:-1], core.vertices, atol=1e-15, rtol=0)
        plan_view = device.plot_2D_structure(shape="requested")
        for collection, core in zip(plan_view.collections, device.describe_cross_section({"gap": 300e-9}).cores):
            vertices = collection.get_paths()[0].vertices
            at_output = vertices[vertices[:, 0] == 5e-6, 1]
            np.testing.assert_allclose([min(at_output), max(at_output)], core.lateral_bounds, atol=1e-15, rtol=0)
        self.assertEqual([line.get_label() for line in device.plot_structure_parameters().lines],
                         ["requested", "clipped", "snapped"])

    def test_validates_saved_configuration_and_geometry(self):
        config = json.loads(json.dumps(self.model.configuration()))
        self.assertEqual(config["model_id"], "straight_gap")
        wg0, wg1 = config["guides"]["wg0"], config["guides"]["wg1"]
        self.assertEqual(wg0["role"], "bus")
        self.assertEqual(wg0["placement"], {
            "kind": "edge_gap", "reference_guide_id": "wg1", "direction": "+y", "parameter": "gap",
        })
        self.assertEqual(wg1["role"], "ring")
        self.assertEqual(wg1["placement"], {"kind": "fixed"})
        self.dataset.section_model_configuration = config
        StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)
        with self.assertRaisesRegex(ValueError, r"guides\.wg0\.width: stored=.*supplied="):
            StraightGapCoupler(self.dataset, replace(self.model, wg0_width=400e-9), gap_profile=200e-9, length=5e-6)
        with self.assertRaisesRegex(ValueError, "slab_thickness"):
            replace(self.stack, slab_thickness=self.stack.thickness)
        with self.assertRaisesRegex(ValueError, "wg1_width"):
            replace(self.model, wg1_width=0)
        for grid in ([-1e-9, 200e-9], [200e-9, 100e-9], [200e-9, np.inf]):
            self.dataset.parameter_grid = {"gap": grid}
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)
        self.dataset.parameter_grid = {"gap": [200e-9]}
        self.dataset.crosssection_x = "x"
        with self.assertRaisesRegex(ValueError, "crosssection_x"):
            StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)
        self.dataset.parameter_names = ["width"]
        with self.assertRaisesRegex(ValueError, "'gap' axis"):
            StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)

    def test_legacy_metadata_matches_only_the_original_geometry(self):
        legacy = {
            "model_id": "double_waveguide", "model_version": 1,
            "parameters": {"gap": {"unit": "m"}},
            "coordinates": {
                "unit": "m", "propagation": "+x", "lateral": "y", "vertical": "z",
                "field_component_order": ["x", "y", "z"],
            },
            "guides": {
                "wg0": {"role": "bus", "region_id": "wg0_core", "width": 500e-9},
                "wg1": {"role": "ring", "region_id": "wg1_core", "width": 450e-9, "center_y": 1e-6},
            },
            "stack": asdict(self.stack),
        }
        self.dataset.data_info = SimpleNamespace(section_model_configuration=legacy)
        snapshot = json.dumps(legacy, sort_keys=True)
        StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)
        self.assertEqual(json.dumps(legacy, sort_keys=True), snapshot)
        with self.assertRaisesRegex(ValueError, r"guides\.wg1\.center_y"):
            model = replace(self.model, wg1_center_y=0)
            StraightGapCoupler(self.dataset, model, gap_profile=200e-9, length=5e-6)
        legacy["model_version"] = 2
        with self.assertRaisesRegex(ValueError, "model_version"):
            StraightGapCoupler(self.dataset, self.model, gap_profile=200e-9, length=5e-6)


if __name__ == "__main__":
    unittest.main()
