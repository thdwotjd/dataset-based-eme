"""Export phase-aligned MODE endpoint fields for the Tidy3D comparison.

Run from the repository root, using the MODE-capable Python environment:
python -m examples.coupled_waveguide.tidy3d_reference DATASET OUT.npz --mode-api-dir MODE_API
"""
import argparse
import importlib
import json
from pathlib import Path
import sys

import numpy as np

from em_simulation import DataUpdater, StraightGapSectionModel, WaveguideStack
from em_simulation.channels._fields import ForwardFields, SectionGrid
from em_simulation.channels.diagnostics import compute_field_products
from em_simulation.channels.mapping import MappingConfig, compute_power_products, normalize_channel_fields
from em_simulation.data_updater.canonical_dataset import (
    _canonicalize, read_mode_card, read_mode_cards, unpack_mode_fields,
)
from em_simulation.data_updater.lumerical_section import LumericalSectionAdapter


def section_model(dataset):
    config = dataset.canonical_manifest["section_model"]
    bus, ring = (config["guides"][key] for key in ("wg0", "wg1"))
    model = StraightGapSectionModel(bus["width"], ring["width"], WaveguideStack(**config["stack"]), ring["center_y"])
    model.validate_configuration(config)
    return model


def _fields(grid, E, H, wavelength):
    return ForwardFields(E, H, SectionGrid(grid.x, grid.y, grid.z, wavelength, "collocated"))


def _block(a, b, c):
    return np.block([[a, c], [c.conj().T, b]])


def _products(U, V, record):
    power = compute_power_products(U, V)
    field = compute_field_products(U, V)
    gp = record["power_products"]
    yield _block(power.G_U, power.G_V, power.C), _block(gp["G_U"], gp["G_V"], gp["C"])
    for name in ("E", "H"):
        norm = getattr(field, name)
        saved = record["field_products"][name]
        yield _block(norm.S_U, norm.S_V, norm.X), _block(saved["S_U"], saved["S_V"], saved["X"])


def _align(U, V, record):
    pairs = list(_products(U, V, record))
    # Use the strongest dimensionless E/H correlation on each link to infer
    # relative phases. Canonical bilinear normalization permits only signs.
    def correlations(matrix):
        norm = np.sqrt(matrix.diagonal().real)
        return matrix / (norm[:, None] * norm[None, :])

    fresh, saved = zip(*[(correlations(a), correlations(b)) for a, b in pairs[1:]])
    strength = np.minimum(np.abs(fresh), np.abs(saved))
    phases = np.zeros(U.count + V.count, complex)
    phases[0] = 1
    while not np.all(phases):
        known = phases != 0
        links = strength * known[None, :, None] * (~known)[None, None, :]
        product, i, j = np.unravel_index(links.argmax(), links.shape)
        if links[product, i, j] < 1e-9:
            raise ValueError("Endpoint field basis cannot be aligned unambiguously to stored products")
        phase = phases[i] * saved[product][i, j] / fresh[product][i, j]
        if j < U.count:
            sign = 1 if phase.real >= 0 else -1
            np.testing.assert_allclose(phase / abs(phase), sign, rtol=0, atol=1e-5)
            phases[j] = sign
        else:
            phases[j] = phase / abs(phase)

    errors = []
    for actual, expected in pairs:
        norm = np.sqrt(expected.diagonal().real)
        error = (actual * phases.conj()[:, None] * phases[None, :] - expected) / (norm[:, None] * norm[None, :])
        errors.append(float(np.max(np.abs(error))))
    if max(errors) > 2e-6:
        raise ValueError(f"Fresh endpoint fields do not reproduce stored power/E/H products: {errors}")

    def rephase(fields, phase):
        phase = phase[:, None, None, None]
        return ForwardFields(fields.E * phase, fields.H * phase, fields.grid)

    return rephase(U, phases[:U.count]), rephase(V, phases[U.count:]), errors


def export_reference(dataset_path, output, mode_api):
    dataset = DataUpdater(dataset_path, is_testmode=True)
    model = section_model(dataset)
    channel_data = dataset.load_channel_data(model)
    config = dataset.coupled_settings.fde
    manifest = dataset.canonical_manifest
    node = (dataset.coupled_settings.gap_grid[-1],)
    first_gap = dataset.coupled_settings.gap_grid[0]
    record = channel_data["nodes"][node]
    section = model.describe_cross_section({"gap": node[0]})
    count = dataset.mode_numbers
    mapping = MappingConfig(**channel_data["dataset_metadata"]["mapping_solver_config"])
    try:
        mode = dataset._open_coupled_mode(mode_api, hide=True)
        adapter = LumericalSectionAdapter(mode, config)

        def isolated(guide, cross_section):
            channels = [c for c in record["channels"] if c["key"][0] == guide]
            numbers = tuple(c["key"][1] for c in channels)
            adapter.prepare(cross_section, guide)
            if int(mode.findmodes()) < max(numbers):
                raise ValueError(f"Missing isolated {guide} modes {numbers}")
            grid, E, H, neff, te = unpack_mode_fields([read_mode_card(mode, i) for i in numbers], numbers)
            np.testing.assert_allclose(neff, [c["neff"] for c in channels], rtol=0, atol=1e-8)
            np.testing.assert_allclose(te, [c["te_fraction"] for c in channels], rtol=0, atol=1e-7)
            return normalize_channel_fields(_fields(grid, E, H, config.wavelength), config=mapping)[0]

        ring = isolated("wg1", model.describe_cross_section({"gap": first_gap}))
        print("Fixed ring reference solved", flush=True)
        adapter.prepare(section)
        raw = read_mode_cards(mode, int(mode.findmodes()), count)
        canonical = _canonicalize(raw, node, manifest["generation_id"], record["canonical_basis_id"], count)
        np.testing.assert_allclose(canonical.neff, dataset.neff[node][:count], rtol=0, atol=1e-8)
        np.testing.assert_allclose(canonical.te_fraction, dataset.TE_pol[node][:count], rtol=0, atol=1e-7)
        U = _fields(canonical.grid, canonical.E, canonical.H, config.wavelength)
        print(f"Coupled {node[0]*1e9:g} nm reference solved", flush=True)
        bus = isolated("wg0", section)
        for guide in (bus, ring):
            U.grid.require_match(guide.grid)
        V = ForwardFields(np.concatenate((bus.E, ring.E)), np.concatenate((bus.H, ring.H)), U.grid)
        U, V, errors = _align(U, V, record)
        c0 = 299792458.0
        indices = {
            key: complex(np.asarray(mode.getindex(name, c0 / config.wavelength)).item())
            for key, name in config.material_bindings.items()
        }
        solver_version = str(mode.version())
    finally:
        dataset.close_mode()

    guided = (canonical.neff.real >= dataset.coupled_settings.cladding_index) & (
        np.abs(canonical.neff.imag) <= channel_data["dataset_metadata"]["guided_filter"]["max_abs_imag_neff"])
    metadata = {
        "generation_id": manifest["generation_id"], "basis_id": record["canonical_basis_id"],
        "channel_basis_id": record["channel_basis_id"], "node": node,
        "channel_keys": [c["key"] for c in record["channels"]],
        "canonical_forward_mode_ids": record["canonical_forward_mode_ids"],
        "mode_version": solver_version, "material_indices": {k: [v.real, v.imag] for k, v in indices.items()},
        "basis_max_scaled_errors_power_E_H": errors, "wavelength_m": config.wavelength,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez_compressed(f, metadata=json.dumps(metadata), y_m=U.grid.y, z_m=U.grid.z,
                            V_E=V.E, V_H=V.H, guided_indices=np.flatnonzero(guided),
                            U_E=U.E[guided], U_H=U.H[guided], neff=canonical.neff)
    print(json.dumps(metadata, indent=2), flush=True)
    print(output, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--mode-api-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.mode_api_dir))
    export_reference(args.dataset, args.output, importlib.import_module("lumapi"))
