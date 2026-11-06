"""Solver-free tests for the Commit 3 channel mathematics."""

from dataclasses import replace
import unittest
from unittest.mock import patch

import numpy as np

from em_simulation.channels import ForwardFields, SectionGrid
from em_simulation.channels.diagnostics import (
    FieldNormProducts, FieldProducts, compute_field_products, evaluate_direct_residual,
    evaluate_mapping_diagnostics, reconstruct_fields, transform_field_products,
)
from em_simulation.channels.mapping import (
    ChannelMappingWarning, MappingConfig, PowerProducts, compute_power_products,
    normalize_channel_fields, normalize_input_amplitudes, solve_channel_maps, transform_products,
)
from em_simulation.geometry.runtime_basis import RuntimeBasisSection


RTOL = 1e-10
ATOL = 1e-12
GRID = SectionGrid(0, (0, 0.5, 2), (-1, 0.25), 1.55e-6, "collocated")


def basis(count=3):
    E = np.zeros((count, 3, len(GRID.y), len(GRID.z)), dtype=complex)
    H = np.zeros_like(E)
    for i in range(count):
        # Distinct spatial samples with exact 1 W forward power.
        y, z = divmod(i, len(GRID.z))
        E[i, 1, y, z] = H[i, 2, y, z] = np.sqrt(2 / GRID.weights[y, z])
    return ForwardFields(E, H, GRID)


def mix(fields, transform):
    return ForwardFields(
        np.einsum("mcij,mn->ncij", fields.E, transform),
        np.einsum("mcij,mn->ncij", fields.H, transform), fields.grid,
    )


def power_of(fields, p):
    E = np.tensordot(p, fields.E, axes=1)
    H = np.tensordot(p, fields.H, axes=1)
    return 0.5 * np.sum(np.cross(E, H.conj(), axis=0)[0].real * fields.grid.weights)


def maps_for(power, output=None, **options):
    out = power if output is None else output
    options.setdefault("input_guided", np.ones(power.C.shape[0], dtype=bool))
    options.setdefault("output_present", np.ones(out.C.shape[0], dtype=bool))
    return solve_channel_maps(power, out, **options)


class ChannelMappingTests(unittest.TestCase):
    def test_orthonormal_identity_and_symmetric_coupler_beating(self):
        V = basis(2)
        identity = maps_for(compute_power_products(V, V))
        np.testing.assert_allclose(identity.A, np.eye(2), atol=ATOL)
        np.testing.assert_allclose(identity.B, np.eye(2), atol=ATOL)
        U = mix(V, np.array([[1, 1], [1, -1]]) / np.sqrt(2))
        maps = maps_for(compute_power_products(U, V))
        p = np.array([1, 0])
        for theta in (0, 0.32, np.pi / 4, np.pi / 2):
            phase = np.exp(1j * np.array([theta, -theta]))
            output = maps.B @ (phase * (maps.A @ p))
            np.testing.assert_allclose(output, [np.cos(theta), 1j * np.sin(theta)], atol=ATOL)
            np.testing.assert_allclose(abs(output)**2, [np.cos(theta)**2, np.sin(theta)**2], atol=ATOL)

    def test_nonorthogonal_common_span_solves_full_gram(self):
        U = mix(basis(2), np.array([[1, 0.8 + 0.1j], [0.2j, 0.9]]))
        T = np.array([[1, 0.2j], [-0.35j, 0.6]])
        V = mix(U, T)
        power = compute_power_products(U, V)
        self.assertGreater(abs(power.G_U[0, 1]), 0.2)
        maps = maps_for(power)
        np.testing.assert_allclose(maps.A, T, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(maps.B @ maps.A, np.eye(2), rtol=RTOL, atol=ATOL)
        p = [0.4 + 0.2j, -0.6j]
        for left, right in zip(reconstruct_fields(V, p), reconstruct_fields(U, maps.A @ p)):
            np.testing.assert_allclose(left, right, rtol=RTOL, atol=ATOL)

    def test_one_watt_references_and_coherent_input_power(self):
        fields = mix(basis(2), np.array([[1 + 1j, 0.6 - 0.1j], [0, 0.25j]]))
        V, factors = normalize_channel_fields(fields)
        np.testing.assert_allclose(factors, [1 / np.sqrt(2), 1 / np.sqrt(0.4325)], rtol=RTOL)
        for raw, scaled in zip((fields.E, fields.H), (V.E, V.H)):
            np.testing.assert_allclose(scaled, raw * factors[:, None, None, None], rtol=RTOL, atol=ATOL)
        products = compute_power_products(basis(2), V)
        np.testing.assert_allclose(np.diag(products.G_V), 1, atol=ATOL)
        raw = np.array([1, -0.4j])
        unchanged = normalize_input_amplitudes(raw, products)
        np.testing.assert_array_equal(unchanged.effective, raw)
        self.assertEqual(unchanged.scale, 1)
        expected_raw = power_of(V, raw)
        self.assertGreater(abs(expected_raw - np.sum(abs(raw)**2)), 0.1)
        normalized = normalize_input_amplitudes(raw, products, input_power=2.5)
        np.testing.assert_allclose(normalized.raw_power_w, expected_raw, rtol=RTOL)
        np.testing.assert_allclose(power_of(V, normalized.effective), 2.5, rtol=RTOL)
        np.testing.assert_array_equal(normalized.raw, raw)
        for target in (0, -1, np.inf, np.nan):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, "positive finite"):
                normalize_input_amplitudes(raw, products, input_power=target)
        with self.assertRaisesRegex(ValueError, "raw input reference power"):
            normalize_input_amplitudes([0, 0], products, input_power=1)
        cancel = PowerProducts(np.eye(1), [[1, 1], [1, 1]], [[1, 1]])
        with self.assertRaisesRegex(ValueError, "unreliable"):
            normalize_input_amplitudes([1, -1 + 1e-8], cancel, input_power=1)
        with self.assertRaisesRegex(ValueError, "nonpositive"):
            normalize_channel_fields(ForwardFields(V.E, -V.H, V.grid))

    def test_runtime_guided_input_full_output_and_absent_slot(self):
        U = basis(3)
        V = mix(U, np.eye(3)[:, [0, 2]])
        canonical = compute_power_products(U, V)
        R = np.column_stack((np.eye(3)[:, 2], np.zeros(3), -np.eye(3)[:, 0], np.eye(3)[:, 1]))
        runtime = RuntimeBasisSection((100e-9,), "basis", R,
                                      np.array([True, False, True, True]),
                                      np.array([False, False, True, False]))
        power = transform_products(canonical, runtime.R)
        fields = transform_field_products(compute_field_products(U, V), runtime.R)
        maps = maps_for(power, input_guided=runtime.forward_guided, output_present=runtime.forward_present)
        np.testing.assert_allclose(maps.A @ [2, 3], [0, 0, -2, 0], atol=ATOL)
        np.testing.assert_allclose(maps.B[:, 1], 0, atol=ATOL)
        diag = evaluate_mapping_diagnostics(power, power, fields, fields, maps, [2, 3], [2j, 0, 0, 0])
        np.testing.assert_allclose(diag.output_channels, [0, 2j], atol=ATOL)
        np.testing.assert_allclose([diag.power.P0, diag.power.P1, diag.power.P2, diag.power.P3],
                                   [13, 4, 4, 4], atol=ATOL)
        np.testing.assert_allclose(diag.input_residuals.E.squared_norm, 18, atol=ATOL)
        np.testing.assert_allclose(diag.output_residuals.H.squared_norm, 0, atol=ATOL)

    def test_runtime_transform_and_truncation_recompute_maps(self):
        U = mix(basis(2), np.array([[1, 0.7], [0.1j, 1]]))
        V = mix(U, np.eye(2)[:, [1]])
        products, norms = compute_power_products(U, V), compute_field_products(U, V)
        R = np.array([[0, 2j], [-0.7j, 0]])
        changed = mix(U, R)
        power = transform_products(products, R)
        fields = transform_field_products(norms, R)
        direct_power = compute_power_products(changed, V)
        direct_fields = compute_field_products(changed, V)
        for x, y in zip((power.G_U, power.C, fields.E.X, fields.H.S_U),
                        (direct_power.G_U, direct_power.C, direct_fields.E.X, direct_fields.H.S_U)):
            np.testing.assert_allclose(x, y, rtol=RTOL, atol=ATOL)
        p, a = np.array([0.2 + 0.4j]), np.array([-0.4j, 0.9])
        original = evaluate_mapping_diagnostics(products, products, norms, norms, maps_for(products), p, a)
        runtime = evaluate_mapping_diagnostics(power, power, fields, fields, maps_for(power), p, np.linalg.solve(R, a))
        np.testing.assert_allclose(runtime.output_channels, original.output_channels, rtol=RTOL, atol=ATOL)
        np.testing.assert_allclose(runtime.power.P2, original.power.P2, rtol=RTOL)
        np.testing.assert_allclose(runtime.output_residuals.E.squared_norm,
                                   original.output_residuals.E.squared_norm, atol=ATOL)

        limited = transform_products(products, np.eye(2)[:, [0]])
        self.assertAlmostEqual(abs(maps_for(products).A[0, 0]), 0, places=10)
        self.assertGreater(abs(maps_for(limited).A[0, 0]), 0.5)

    def test_ill_conditioned_gram_uses_truncated_svd_and_reports_residual(self):
        power = PowerProducts([[1, 1], [1, 1 + 1e-14]], [[1]], [[1], [-1]])
        config = MappingConfig(svd_rtol=1e-8)
        with self.assertWarnsRegex(ChannelMappingWarning, "gap 100 nm:.*rank 1/2"):
            maps = maps_for(power, config=config, input_context="gap 100 nm")
        info = maps.input_solve
        self.assertEqual(info.method, "truncated_svd")
        self.assertEqual(info.rank, 1)
        self.assertGreater(info.condition, 1e12)
        self.assertGreater(info.cutoff, 0)
        np.testing.assert_allclose(info.equation_residual, np.linalg.norm(power.G_U @ maps.A - power.C))
        np.testing.assert_allclose(info.relative_equation_residual, 1, rtol=RTOL)
        self.assertEqual(maps.output_solve.method, "solve")
        with patch("numpy.linalg.svd", side_effect=np.linalg.LinAlgError("test")):
            with self.assertRaisesRegex(ValueError, "gap 100 nm:.*did not converge"):
                maps_for(power, input_context="gap 100 nm")

    def test_rejects_incompatible_coordinates_shapes_and_hermiticity(self):
        U = basis(2)
        with self.assertRaisesRegex(ValueError, "collocated"):
            replace(GRID, component_locations="staggered")
        for mismatch, regex in ((replace(GRID, x=1e-6), "grid mismatch on x"),
                                (replace(GRID, z=(-1, 0.3)), "grid mismatch on z"),
                                (replace(GRID, wavelength=1.51e-6), "wavelength mismatch")):
            for compute in (compute_power_products, compute_field_products):
                with self.assertRaisesRegex(ValueError, regex):
                    compute(U, replace(U, grid=mismatch))
        with self.assertRaisesRegex(ValueError, "H has shape"):
            ForwardFields(U.E, U.H[:, :2], GRID)
        with self.assertRaisesRegex(ValueError, "E must be finite"):
            replace(U, E=U.E * np.nan)
        with self.assertRaisesRegex(ValueError, "not Hermitian"):
            maps_for(PowerProducts([[1, 0.1j], [0, 1]], [[1]], [[1], [0]]))
        with self.assertRaisesRegex(ValueError, "input_guided"):
            maps_for(compute_power_products(U, U), input_guided=[0, 1])


class ChannelDiagnosticsTests(unittest.TestCase):
    def test_coherent_direct_and_quadratic_residuals_and_budget(self):
        common = basis(3)
        U = mix(common, np.eye(3)[:, :2])
        V = mix(common, np.array([[1, 0], [0, 1], [1, 1j]]) / np.sqrt(2))
        # Use a different output channel inventory and E/H normalization.
        U_out = ForwardFields(U.E * 1.4, U.H / 1.4, U.grid)
        V_out = mix(common, np.array([[0.4], [0.2j], [0.8]]))
        p0, p1 = compute_power_products(U, V), compute_power_products(U_out, V_out)
        f0, f1 = compute_field_products(U, V), compute_field_products(U_out, V_out)
        maps = maps_for(p0, p1)
        rng = np.random.default_rng(2026)
        coefficients = [np.array([1, np.exp(1j * angle)]) for angle in (0, np.pi, np.pi / 2)]
        coefficients += [rng.normal(size=2) + 1j * rng.normal(size=2) for _ in range(4)]
        errors = []
        for p in coefficients:
            a = rng.normal(size=2) + 1j * rng.normal(size=2)
            report = evaluate_mapping_diagnostics(p0, p1, f0, f1, maps, p, a)
            for computed, direct in (
                (report.input_residuals, evaluate_direct_residual(V, p, U, maps.A @ p)),
                (report.output_residuals, evaluate_direct_residual(U_out, a, V_out, maps.B @ a)),
            ):
                for x, y in ((computed.E, direct.E), (computed.H, direct.H)):
                    np.testing.assert_allclose(x.squared_norm, y.squared_norm, rtol=RTOL, atol=ATOL)
                    np.testing.assert_allclose(x.relative_squared, y.relative_squared, rtol=RTOL, atol=ATOL)
            expected = [power_of(V, p), power_of(U, maps.A @ p),
                        power_of(U_out, a), power_of(V_out, maps.B @ a)]
            budget = report.power
            np.testing.assert_allclose([budget.P0, budget.P1, budget.P2, budget.P3], expected, rtol=RTOL)
            np.testing.assert_allclose([budget.input_mapping.watts, budget.output_mapping.watts],
                                       [expected[0] - expected[1], expected[2] - expected[3]], atol=ATOL)
            L_in = p0.G_V - maps.A.conj().T @ p0.G_U @ maps.A
            L_out = p1.G_U - maps.B.conj().T @ p1.G_V @ maps.B
            np.testing.assert_allclose([budget.input_mapping.watts, budget.output_mapping.watts],
                                       [np.vdot(p, L_in @ p), np.vdot(a, L_out @ a)], atol=ATOL)
            np.testing.assert_allclose(budget.budget_closure_w, 0, atol=ATOL)
            errors.append(report.input_residuals.E.squared_norm)
        self.assertGreater(abs(errors[0] - errors[2]), 1)
        self.assertEqual(f0.components, ("y", "z"))

    def test_negative_signed_difference_and_invalid_relative_reference(self):
        V = basis(1)
        U = ForwardFields(2 * V.E, 0.5 * V.H, V.grid)
        power, fields = compute_power_products(U, V), compute_field_products(U, V)
        config = MappingConfig(e_residual_warning=0.1)
        maps = maps_for(power, config=config, output_context="gap 300 nm")
        with self.assertWarnsRegex(ChannelMappingWarning, "gap 300 nm: relative E residual"):
            report = evaluate_mapping_diagnostics(power, power, fields, fields, maps, [0], [1])
        self.assertEqual(report.power.input_mapping.watts, 0)
        self.assertFalse(report.power.input_mapping.defined)
        self.assertFalse(report.input_residuals.E.defined)
        self.assertIsNotNone(report.input_residuals.E.reason)
        self.assertEqual(report.input_residuals.E.absolute_norm, 0)
        self.assertLess(report.power.output_mapping.watts, 0)
        self.assertLess(report.power.output_mapping.relative, 0)
        self.assertTrue(report.output_residuals.E.threshold_exceeded)
        self.assertEqual(report.output_residuals.E.warning_threshold, 0.1)
        self.assertEqual(report.output_residuals.E.norm_unit, "V")
        self.assertEqual(report.output_residuals.H.norm_unit, "A")

    def test_near_cancellation_keeps_absolute_residual_and_marks_ratio_undefined(self):
        U = basis(1)
        V = mix(U, np.array([[1.0, 1.0]]))
        power, fields = compute_power_products(U, V), compute_field_products(U, V)
        with self.assertWarns(ChannelMappingWarning):
            maps = maps_for(power)
        p = np.array([1.0, -1.0 + 1e-8])
        direct = evaluate_direct_residual(V, p, U, [0])
        self.assertFalse(direct.E.defined)
        self.assertGreater(direct.E.absolute_norm, 0)
        quadratic = evaluate_mapping_diagnostics(power, power, fields, fields, maps, p, [1])
        self.assertFalse(quadratic.input_residuals.E.defined)
        self.assertEqual(maps.output_solve.method, "truncated_svd")
        np.testing.assert_allclose(quadratic.power.P3, 1, atol=ATOL)
        np.testing.assert_allclose(quadratic.power.output_mapping.watts, 0, atol=ATOL)

    def test_power_difference_uses_actual_forward_amplitude_after_cancellation(self):
        power = PowerProducts([[1e-20]], np.eye(2), [[1, 1]])
        norms = FieldNormProducts([[1e-40]], np.eye(2), [[0, 0]])
        fields = FieldProducts(norms, norms)
        maps = maps_for(power)
        report = evaluate_mapping_diagnostics(power, power, fields, fields, maps, [1, -1], [0])
        np.testing.assert_array_equal(report.input_forward, [0])
        self.assertEqual(report.power.P0, 2)
        self.assertEqual(report.power.P1, 0)
        self.assertEqual(report.power.input_mapping.watts, 2)
        self.assertEqual(report.power.input_mapping.relative, 1)
        self.assertEqual(report.power.budget_closure_w, 0)

    def test_field_residual_uses_actual_forward_amplitude_after_cancellation(self):
        common = basis(3)
        U = ForwardFields(1e-12 * common.E[:1], common.H[:1], common.grid)
        V = ForwardFields(common.E[1:] + common.E[:1], common.H[1:], common.grid)
        power, fields = compute_power_products(U, V), compute_field_products(U, V)
        p = [1, -1]
        report = evaluate_mapping_diagnostics(power, power, fields, fields, maps_for(power), p, [0])
        np.testing.assert_array_equal(report.input_forward, [0])
        direct = evaluate_direct_residual(V, p, U, report.input_forward)
        for actual, expected in ((report.input_residuals.E, direct.E),
                                 (report.input_residuals.H, direct.H)):
            np.testing.assert_allclose(actual.squared_norm, expected.squared_norm, rtol=RTOL)
            self.assertEqual(actual.relative_squared, 1)

    def test_negative_squared_norm_is_not_hidden(self):
        U = basis(1)
        power = compute_power_products(U, U)
        valid = compute_field_products(U, U)
        bad = FieldProducts(FieldNormProducts([[1]], [[1]], [[4]]), valid.H)
        with self.assertRaisesRegex(ValueError, "negative squared norm"):
            evaluate_mapping_diagnostics(power, power, bad, valid, maps_for(power), [1], [1])

    def test_empty_output_channels_are_valid_products(self):
        U, V = basis(2), basis(0)
        power, fields = compute_power_products(U, V), compute_field_products(U, V)
        maps = maps_for(power)
        self.assertEqual(maps.A.shape, (2, 0))
        self.assertEqual(maps.B.shape, (0, 2))
        report = evaluate_mapping_diagnostics(power, power, fields, fields, maps, [], [1, 0])
        self.assertEqual(report.output_channels.size, 0)
        self.assertAlmostEqual(report.power.output_mapping.watts, 1)
        self.assertAlmostEqual(report.output_residuals.H.relative_norm, 1)


if __name__ == "__main__":
    unittest.main()
