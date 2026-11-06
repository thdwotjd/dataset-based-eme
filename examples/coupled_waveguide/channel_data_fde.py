"""220 nm SOI example. From the checkout, run with python -m examples.coupled_waveguide.channel_data_fde."""

import argparse
import importlib
from pathlib import Path
import sys

import numpy as np

import em_simulation as dreme
from em_simulation.channels.diagnostics import FieldNormProducts, FieldProducts, evaluate_mapping_diagnostics
from em_simulation.channels.mapping import MappingConfig, PowerProducts, solve_channel_maps
from em_simulation.data_updater.channel_data_builder import build_channel_data, validate_baseline


def example(gaps_nm):
    # Same editable smoke geometry as coupled_supermode_fde.ipynb. Isolated
    # budgets are separate from the number of retained coupled modes.
    model = dreme.StraightGapSectionModel(500e-9, 450e-9, dreme.WaveguideStack(220e-9, "si", "oxide", "oxide"))
    settings = dreme.CoupledDatasetSettings(tuple(gap * 1e-9 for gap in gaps_nm), 1.44, dreme.FDEConfig(
        wavelength=1550e-9, y_bounds=(-2e-6, 2.4e-6), z_bounds=(-1.3e-6, 1.3e-6),
        mesh_cells_yz=(176, 104), material_bindings={"si": "Si (Silicon) - Palik", "oxide": "SiO2 (Glass) - Palik"},
        candidate_counts={"coupled": 2, "wg0": 3, "wg1": 3}, boundary="Metal",
    ))
    return model, settings


def acquire(output, model, settings, mode_api):
    updater = dreme.DataUpdater(output, coupled_settings=settings)
    low, high = settings.gap_grid[0], settings.gap_grid[-1]
    geometry = dreme.StraightGapCoupler(
        updater, model, dreme.profile_from_samples([0, 10e-6], [low, high]), verbose=False,
    )

    def report_solve(problem, node, seconds, returned):
        fde, mode = settings.fde, updater.mode
        section = model.describe_cross_section({"gap": node[0]})
        for core in section.cores:
            name = f"::model::dreme_core_{core.region_id}"
            vertices = np.asarray(mode.getnamed(name, "vertices"))
            np.testing.assert_allclose(np.column_stack((vertices[:, 1], -vertices[:, 0])),
                                       core.vertices, rtol=0, atol=1e-15)
            assert mode.getnamed(name, "enabled") == int(problem in ("coupled", core.guide_id))
            assert mode.getnamed(name, "material") == fde.material_bindings[core.material_id]
        assert mode.getanalysis("number of trial modes") == fde.candidate_counts[problem]
        np.testing.assert_allclose(mode.getanalysis("wavelength"), fde.wavelength, rtol=1e-12)
        for axis, cells, bounds in zip("yz", fde.mesh_cells_yz, (fde.y_bounds, fde.z_bounds)):
            assert mode.getnamed("::model::FDE", f"mesh cells {axis}") == cells
            actual = [mode.getnamed("::model::FDE", f"{axis} {edge}") for edge in ("min", "max")]
            np.testing.assert_allclose(actual, bounds, rtol=0, atol=1e-15)
        print(f"{problem} at {node[0] * 1e9:g} nm: {seconds:.2f} s, "
              f"{returned}/{fde.candidate_counts[problem]} candidates; layout verified", flush=True)

    print(f"Starting MODE for {len(settings.gap_grid)} gaps: {output}", flush=True)
    try:
        manifest = build_channel_data(updater, geometry, mode_api=mode_api, on_solve=report_solve)
        print(f"Saved {manifest['generation_id']} in {output}", flush=True)
    finally:
        updater.close_mode()


def verify(output, model):
    updater = dreme.DataUpdater(output, is_testmode=True)
    data = updater.load_channel_data(model)
    config = MappingConfig(**data["dataset_metadata"]["mapping_solver_config"])
    guided_filter = data["dataset_metadata"]["guided_filter"]
    for node, record in data["nodes"].items():
        power = PowerProducts(**record["power_products"])
        fields = FieldProducts(*(FieldNormProducts(**record["field_products"][f]) for f in ("E", "H")))
        k = len(record["canonical_forward_mode_ids"])
        neff = updater.neff[node][:k]
        guided = ((neff.real >= guided_filter["cladding_index"])
                  & (abs(neff.imag) <= guided_filter["max_abs_imag_neff"]))
        maps = solve_channel_maps(power, power, input_guided=guided, output_present=np.ones(k, bool), config=config)
        baseline = record["baseline_diagnostics"]
        report = evaluate_mapping_diagnostics(
            power, power, fields, fields, maps, baseline["input_channels"], baseline["output_forward"],
        )
        validate_baseline(power, fields, report, baseline, f"node {node}")
        np.testing.assert_allclose(np.diag(power.G_V), 1, rtol=1e-10, atol=1e-10)
        off_diagonal = power.G_V - np.diag(np.diag(power.G_V))
        channels = [c["key"] for c in record["channels"]]
        print(f"{node[0] * 1e9:g} nm: {channels}, max |offdiag G_V| = "
              f"{np.max(abs(off_diagonal), initial=0):.6g}")
    assert updater.mode is None
    print(f"Verified {len(data['nodes'])} nodes offline in {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="fresh dataset directory (existing dataset with --offline)")
    parser.add_argument("--gaps-nm", type=float, nargs="+", default=(150, 250))
    parser.add_argument("--mode-api-dir", type=Path, help="directory containing lumapi.py")
    parser.add_argument("--offline", action="store_true", help="read and verify a saved generation without MODE")
    args = parser.parse_args()
    model, settings = example(args.gaps_nm)
    if args.offline:
        verify(args.output, model)
    else:
        if args.mode_api_dir is not None:
            sys.path.insert(0, str(args.mode_api_dir))
        acquire(args.output, model, settings, importlib.import_module("lumapi"))


if __name__ == "__main__":
    main()
