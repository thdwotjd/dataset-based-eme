"""Recheck Commit 6 from local MODE products and completed Tidy3D files.

Run with env310 from the repository root. This command uses local files only.
python -B -m examples.coupled_waveguide.audit_saved_results DATA_ROOT --output OUTPUT
"""
import argparse
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from em_simulation import DataUpdater
from em_simulation.channels.diagnostics import (
    FieldNormProducts, FieldProducts, compute_field_products, evaluate_mapping_diagnostics,
    transform_field_products,
)
from em_simulation.channels.mapping import (
    MappingConfig, PowerProducts, compute_power_products, solve_channel_maps, transform_products,
)
from em_simulation.data_updater.channel_data_builder import _baseline, validate_baseline
from .tidy3d_comparison import (
    _digest, _load_reference, _read_json, _save_json, compare_saved, prepare_case,
)
from .tidy3d_reference import section_model


# Completed source-aligned cases, including the existing mesh/time checks.
RUNS = (
    ("tidy3d_c6_source_aligned_ppw22", "uniform"),
    ("tidy3d_c6_source_aligned_ppw22_6ps", "uniform"),
    ("tidy3d_c6_source_aligned_ppw22_6ps", "moving"),
    ("tidy3d_c6_source_aligned_mesh12p5_6ps", "uniform"),
    ("tidy3d_c6_source_aligned_mesh12p5_6ps", "moving"),
    ("tidy3d_c6_source_aligned_mesh12p5_12ps", "moving"),
)


def audit_mathematics(dataset, reference, endpoint):
    model = section_model(dataset)
    data = dataset.load_channel_data(model)
    config = MappingConfig(**data["dataset_metadata"]["mapping_solver_config"])
    max_power_error, max_residual_scaled_error = 0.0, 0.0
    for node, record in data["nodes"].items():
        power = PowerProducts(**record["power_products"])
        fields = FieldProducts(*(FieldNormProducts(**record["field_products"][f]) for f in ("E", "H")))
        neff = dataset.neff[node][:power.G_U.shape[0]]
        guided = ((neff.real >= dataset.coupled_settings.cladding_index)
                  & (abs(neff.imag) <= data["dataset_metadata"]["guided_filter"]["max_abs_imag_neff"]))
        maps = solve_channel_maps(power, power, input_guided=guided,
                                  output_present=np.ones(len(neff), bool), config=config)
        baseline = record["baseline_diagnostics"]
        report = evaluate_mapping_diagnostics(power, power, fields, fields, maps,
                                              baseline["input_channels"], baseline["output_forward"])
        validate_baseline(power, fields, report, baseline, f"saved node {node}")
        max_power_error = max(max_power_error, max(abs(np.asarray(baseline["direct_power_w"])
                                   - [getattr(report.power, f"P{i}") for i in range(4)])))
        for computed, expected in zip((report.input_residuals, report.output_residuals),
                                       baseline["direct_residual_squared"]):
            for f, direct in zip(("E", "H"), expected):
                actual = getattr(computed, f)
                max_residual_scaled_error = max(max_residual_scaled_error,
                    abs(actual.squared_norm-direct)/(abs(direct)+actual.reference_squared_norm))

    metadata, U, V, _, indices = _load_reference(reference, dataset, endpoint.node, endpoint, data["nodes"])
    record = data["nodes"][endpoint.node]
    actual_power, actual_fields = compute_power_products(U, V), compute_field_products(U, V)
    saved = record["power_products"]
    power = PowerProducts(saved["G_U"][np.ix_(indices, indices)], saved["G_V"], saved["C"][indices])
    fields = FieldProducts(*(FieldNormProducts(
        record["field_products"][f]["S_U"][np.ix_(indices, indices)],
        record["field_products"][f]["S_V"], record["field_products"][f]["X"][indices],
    ) for f in ("E", "H")))
    errors = {}
    for label, actual, stored in (("power", actual_power, power), ("E", actual_fields.E, fields.E),
                                  ("H", actual_fields.H, fields.H)):
        names = ("G_U", "G_V", "C") if label == "power" else ("S_U", "S_V", "X")
        scale_u = np.sqrt(abs(getattr(stored, names[0]).diagonal().real))
        scale_v = np.sqrt(abs(getattr(stored, names[1]).diagonal().real))
        scales = (np.outer(scale_u, scale_u), np.outer(scale_v, scale_v), np.outer(scale_u, scale_v))
        errors[label] = max(float(np.max(abs(getattr(actual, key)-getattr(stored, key))/scale))
                            for key, scale in zip(names, scales))
        if errors[label] > 2e-6:
            raise ValueError(f"Saved endpoint {label} products disagree with actual MODE fields")
    baseline = _baseline(U, V, power, fields, np.ones(U.count, bool), config, "reloaded guided MODE fields")
    p, a = baseline["input_channels"], baseline["output_forward"]
    # Changing the coordinate basis must preserve the represented field and readout.
    R = np.eye(U.count)[:, ::-1] @ np.diag(
        np.linspace(0.8, 1.2, U.count)*np.exp(1j*np.linspace(0.1, 1.1, U.count)))
    rotated_power, rotated_fields = transform_products(power, R), transform_field_products(fields, R)
    options = dict(input_guided=np.ones(U.count, bool), output_present=np.ones(U.count, bool), config=config)
    maps = solve_channel_maps(power, power, **options)
    rotated_maps = solve_channel_maps(rotated_power, rotated_power, **options)
    original = evaluate_mapping_diagnostics(power, power, fields, fields, maps, p, a)
    rotated = evaluate_mapping_diagnostics(rotated_power, rotated_power, rotated_fields, rotated_fields,
                                           rotated_maps, p, np.linalg.solve(R, a))
    coefficient_error = float(np.max(abs(original.output_channels-rotated.output_channels)))
    power_error = max(abs(getattr(original.power, f"P{i}")-getattr(rotated.power, f"P{i}")) for i in range(4))
    np.testing.assert_allclose(rotated.output_channels, original.output_channels, atol=1e-11, rtol=1e-11)
    for side in ("input_residuals", "output_residuals"):
        for f in ("E", "H"):
            np.testing.assert_allclose(getattr(getattr(rotated, side), f).relative_squared,
                                       getattr(getattr(original, side), f).relative_squared, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(power_error, 0, atol=1e-10)
    return {"saved_node_baselines_rechecked": len(data["nodes"]),
            "saved_baseline_max_direct_matrix_power_error_W": max_power_error,
            "saved_baseline_max_residual_squared_error_over_direct_plus_reference": max_residual_scaled_error,
            "fresh_endpoint_direct_reconstruction": "pass, reloaded guided coupled and all isolated E/H",
            "endpoint_basis_id": metadata["basis_id"],
            "endpoint_guided_mode_indices": indices.tolist(),
            "endpoint_max_scaled_product_errors": errors,
            "phase_permutation_scaling_invariance": "pass",
            "max_output_coefficient_invariance_error": coefficient_error,
            "max_power_invariance_error_W": power_error,
            "scope": "All nodes use direct integrals recorded during acquisition. Fresh E/H is available only for the guided coupled subset at 250 nm."}


def audit_input_replay(data_root, output):
    """Replay the measured forward input through DREME without a new FDTD solve."""
    import tidy3d as td
    from em_simulation import EME, EMEStabilityConfig, Runner
    from .tidy3d_comparison import _combine, _dreme, _field_difference, _plane, _power
    from em_simulation.channels.mapping import _solve

    output = Path(output)
    rows, diagnostics = [], []
    for series, name in (("tidy3d_c6_source_aligned_mesh12p5_6ps", "uniform"),
                          ("tidy3d_c6_source_aligned_mesh12p5_12ps", "moving")):
        folder = Path(data_root)/series/name
        saved = _read_json(folder/"case.json")
        dataset = DataUpdater(saved["dataset"], is_testmode=True)
        model = section_model(dataset)
        geometry, result, _ = _dreme(dataset, model, name, (10, 5, 5, 5), 24, EMEStabilityConfig())
        report = result.diagnostics.channel_mapping
        records = dataset.load_channel_data(model)["nodes"]
        left, right = geometry.runtime_basis_at(0), geometry.runtime_basis_at(-1)
        _, U, V, neff, indices = _load_reference(
            saved["endpoint_reference"], dataset, left.node, report.input_endpoint, records)
        config = MappingConfig(**saved["dreme"]["mapping_config"])
        data = td.SimulationData.from_file(folder/"result.hdf5")
        if data.diverged or data.simulation != td.Simulation.from_file(folder/"simulation.hdf5"):
            raise ValueError(f"Saved simulation mismatch in {folder}")
        measured, incoming = _plane(data, "input", U.grid)
        products = compute_power_products(U, measured)
        fitted, _ = _solve(products.G_U, products.C, config, "measured input onto saved guided MODE fields")
        fitted = fitted[:, 0]
        fitted_fields = _combine(U, fitted)
        canonical = np.zeros(left.R.shape[0], complex)
        canonical[indices] = fitted
        runtime = np.linalg.lstsq(left.R, canonical, rcond=None)[0]
        np.testing.assert_allclose(left.R @ runtime, canonical, rtol=0, atol=1e-12)
        runner = Runner(EME(geometry, neff_interpolation=False))
        propagated = runner.propagate_basis_amplitudes(np.r_[runtime, np.zeros_like(runtime)])
        outgoing = np.where(right.forward_present, propagated[:right.R.shape[1]], 0)
        input_power = transform_products(PowerProducts(**records[left.node]["power_products"]), left.R)
        output_power = transform_products(PowerProducts(**records[right.node]["power_products"]), right.R)
        maps = solve_channel_maps(input_power, output_power, input_guided=left.forward_guided,
                                  output_present=right.forward_present, config=config)
        channels = maps.B @ outgoing
        expected_input = report.input_endpoint.R @ report.evaluation.input_forward
        mapped = _combine(U, expected_input[indices])
        aligned = _combine(fitted_fields, [np.sqrt(_power(mapped)/_power(fitted_fields))])
        original = pd.read_csv(folder/"comparison.csv")
        for index, row in original.iterrows():
            replayed = float(abs(channels[index])**2/incoming["forward_flux_W"])
            rows.append({"case": name, "channel": row.channel,
                         "dreme_original_over_P0": row.dreme_over_P0,
                         "dreme_replayed_over_measured_input": replayed,
                         "tidy_over_incident": row.tidy_over_incident,
                         "tidy_minus_original_pp": row.tidy_minus_dreme_pp,
                         "tidy_minus_replayed_pp": 100*(row.tidy_over_incident-replayed)})
        entry = {
            "case": name, "basis_id": left.basis_id, "measured_forward_W": incoming["forward_flux_W"],
            "represented_guided_input_W": _power(fitted_fields),
            "measured_to_guided_fit_EH_residual": _field_difference(measured, fitted_fields, fit_phase=False),
            "mapped_to_measured_guided_EH_residual_after_power_and_phase_alignment":
                _field_difference(mapped, aligned, fit_phase=True),
            "scope": "Only the stored five guided coupled input modes are replayed. All 24 output modes are retained. This does not reproduce unrepresented incident fields.",
        }
        if name == "uniform":
            expected = expected_input*np.exp(1j*2*np.pi/dataset.wavelength*neff*10e-6)
            actual = report.output_endpoint.R @ report.transmitted_forward
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-11)
            entry["uniform_exact_complex_phase_max_coefficient_error"] = float(np.max(abs(actual-expected)))
        diagnostics.append(entry)
    pd.DataFrame(rows).to_csv(output/"input_replay.csv", index=False)
    _save_json(output/"input_replay_diagnostics.json", diagnostics)
    return diagnostics


def audit_saved_results(data_root, output):
    data_root, output = Path(data_root).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports, tables = [], []
    with patch.object(DataUpdater, "_open_coupled_mode", side_effect=AssertionError("Audit must remain offline")):
        for series, name in RUNS:
            folder = data_root/series/name
            saved = _read_json(folder/"case.json")
            fdtd = saved["fdtd"]
            case = prepare_case(
                name, saved["dataset"], saved["endpoint_reference"], output/"recomputed"/series,
                coarse_dataset=saved["coarse_dataset"]["path"], lengths_um=(10, 5, 5, 5),
                mesh_um=fdtd["mesh_um"], run_time_ps=fdtd["run_time_ps"], source_kind=saved["source"]["kind"],
                source_x_um=saved["source"]["x_um"], lead_um=fdtd["lead_um"],
                retained_modes=saved["dreme"]["retained_modes"], monitor_modes=fdtd["monitor_modes"],
            )
            table, figure, details = compare_saved(case, folder/"result.hdf5")
            plt.close(figure)
            # A stale DREME result or different channel order must fail the audit.
            previous = pd.read_csv(folder/"comparison.csv")
            if table.channel.tolist() != previous.channel.tolist():
                raise ValueError(f"Changed channel order in {folder}")
            np.testing.assert_allclose(table.dreme_W, previous.dreme_W, atol=2e-12, rtol=0)
            np.testing.assert_allclose(table.tidy_over_incident, previous.tidy_over_incident, atol=2e-12, rtol=0)
            table.insert(0, "series", series)
            table.insert(1, "case", name)
            table["fdtd_mesh_nm"] = fdtd["mesh_um"]*1000
            table["fdtd_max_time_ps"] = fdtd["run_time_ps"]
            table["full_minus_guided_readout_pp"] = 100*(table.dreme_W-table.dreme_guided_readout_W)/table.dreme_P0_W
            table["dreme_33_minus_17_nodes_pp"] = 100*(table.dreme_W-table.dreme_coarse_W)/table.dreme_P0_W
            tables.append(table)
            log = (folder/"solver.log").read_text(encoding="utf-8")
            final_decay = float(log.rsplit("field decay:", 1)[1].split()[0])
            reports.append({
                "series": series, "case": name, "result_path": details["hdf5"], "result_sha256": details["hdf5_sha256"],
                "task": _read_json(folder/"task.json"), "source_manifest_sha256": _digest(folder/"case.json"),
                "simulation_sha256": _digest(folder/"simulation.hdf5"),
                "settings": {key: value for key, value in case.manifest.items()
                             if key not in ("endpoint_basis", "path")},
                "grid_path": {key: case.manifest["path"][key] for key in ("grid_x_m", "grid_gap_m")},
                "input": details["input"], "output": details["output"],
                "final_field_decay": final_decay,
                "stopped_below_shutoff": "Field decay smaller than shutoff factor" in log,
            })
            print(f"Rechecked {series}/{name}", flush=True)
        dataset = DataUpdater(reports[-1]["settings"]["dataset"], is_testmode=True)
        from .tidy3d_comparison import _dreme
        from em_simulation import EMEStabilityConfig
        _, result, _ = _dreme(dataset, section_model(dataset), "uniform", (10, 5, 5, 5), 24, EMEStabilityConfig())
        mathematics = audit_mathematics(dataset, reports[-1]["settings"]["endpoint_reference"],
                                       result.diagnostics.channel_mapping.input_endpoint)

    input_replay = audit_input_replay(data_root, output)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(output/"channel_comparison.csv", index=False)
    changes = []
    for label, first, last, name in (
        ("FDTD time 3 to 6 ps", "tidy3d_c6_source_aligned_ppw22", "tidy3d_c6_source_aligned_ppw22_6ps", "uniform"),
        ("FDTD mesh 20.27 to 12.5 nm", "tidy3d_c6_source_aligned_ppw22_6ps", "tidy3d_c6_source_aligned_mesh12p5_6ps", "uniform"),
        ("FDTD mesh 20.27 to 12.5 nm", "tidy3d_c6_source_aligned_ppw22_6ps", "tidy3d_c6_source_aligned_mesh12p5_6ps", "moving"),
        ("FDTD time 6 to 12 ps", "tidy3d_c6_source_aligned_mesh12p5_6ps", "tidy3d_c6_source_aligned_mesh12p5_12ps", "moving"),
    ):
        a = table[(table.series == first) & (table.case == name)].set_index("channel")
        b = table[(table.series == last) & (table.case == name)].set_index("channel")
        for channel, delta in (100*(b.tidy_over_incident-a.tidy_over_incident)).items():
            changes.append({"check": label, "case": name, "channel": channel, "last_minus_first_pp": delta})
    pd.DataFrame(changes).to_csv(output/"convergence.csv", index=False)
    _save_json(output/"mathematics.json", mathematics)
    _save_json(output/"run_manifest.json", {
        "analysis_code_sha256": _digest(Path(__file__)), "cases": reports, "input_replay": input_replay,
        "status": {"mathematical_checks": "pass", "FDTD_initial_comparison": "complete",
                   "quantitative_accuracy_acceptance": "not established"},
        "excluded": ["Earlier tidy3d_c6_baseline has a 200 nm source shift", "Lumerical FDTD is a separate pending run"],
    })
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for ax, name, series in zip(axes, ("uniform", "moving"),
                                ("tidy3d_c6_source_aligned_mesh12p5_6ps", "tidy3d_c6_source_aligned_mesh12p5_12ps")):
        selected = table[(table.series == series) & (table.case == name)].set_index("channel")
        plot = selected[["dreme_over_P0", "tidy_over_incident"]].rename(
            columns={"dreme_over_P0": "DREME / P0", "tidy_over_incident": "Tidy3D / measured input"})
        (100*plot).plot.bar(ax=ax, rot=45)
        ax.set(title=name, ylabel="Reference coefficient power (%)", xlabel="")
    fig.savefig(output/"comparison.png", dpi=150)
    plt.close(fig)
    print(mathematics, flush=True)
    return table, reports, mathematics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    audit_saved_results(arguments.data_root, arguments.output)
