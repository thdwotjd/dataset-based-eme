"""Local SOI comparison and explicit Tidy3D task steps for Commit 6."""
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
import tidy3d as td

from em_simulation import ChannelRunner, DataUpdater, EME, EMEStabilityConfig, Runner, StraightGapCoupler, profile_from_samples
from em_simulation.channels._fields import ForwardFields, SectionGrid
from em_simulation.channels.mapping import (
    MappingConfig, PowerProducts, _solve, compute_power_products, solve_channel_maps, transform_products,
)
from .tidy3d_reference import section_model

UM = 1e-6
TANGENTIAL = ("Ey", "Ez", "Hy", "Hz")


def _json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return [value.real, value.imag]
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _json(value):
    return json.dumps(value, default=_json_value, indent=2, allow_nan=False)


def _save_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(_json(value), encoding="utf-8")
    temporary.replace(path)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(path):
    digest = sha256()
    with path.open('rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _log(directory, message):
    with (directory / "run.log").open("a", encoding="utf-8") as file:
        print(message, file=file)


@dataclass
class Case:
    name: str
    directory: Path
    simulation: td.Simulation
    manifest: dict
    references: ForwardFields
    target: ForwardFields
    config: MappingConfig
    dreme_rows: list


def _load_reference(path, dataset, node, output, records):
    with np.load(path, allow_pickle=False) as raw:
        metadata = json.loads(str(raw["metadata"]))
        keys = [list(c["key"]) for c in records[node]["channels"]]
        if (metadata["generation_id"] != dataset.canonical_manifest["generation_id"]
                or metadata["basis_id"] != output.basis_id or metadata["channel_basis_id"] != output.channel_basis_id
                or tuple(metadata["node"]) != node or metadata["channel_keys"] != keys
                or tuple(metadata["canonical_forward_mode_ids"]) != output.canonical_forward_mode_ids
                or metadata["wavelength_m"] != dataset.wavelength):
            raise ValueError("Endpoint fields do not belong to this DREME generation and basis")
        saved = dataset.canonical_manifest["export_grid"]
        for axis in ("y", "z"):
            np.testing.assert_allclose(raw[f"{axis}_m"], saved[axis], atol=1e-15, rtol=0)
        grid = SectionGrid(saved["x"], saved["y"], saved["z"], dataset.wavelength, "collocated")
        V = ForwardFields(raw["V_E"], raw["V_H"], grid)
        U = ForwardFields(raw["U_E"], raw["U_H"], grid)
        neff = raw["neff"]
        np.testing.assert_allclose(neff, dataset.neff[node][:len(neff)], rtol=0, atol=1e-8)
        guided = raw["guided_indices"]
    np.testing.assert_allclose(compute_power_products(V, V).G_V, records[node]["power_products"]["G_V"],
                               rtol=0, atol=2e-6)
    return metadata, U, V, neff, guided


def _combine(fields, amplitudes):
    return ForwardFields(
        np.einsum("m,mcij->cij", amplitudes, fields.E)[None],
        np.einsum("m,mcij->cij", amplitudes, fields.H)[None], fields.grid,
    )


def _power(fields):
    E, H = fields.E[0], fields.H[0]
    return float(0.5 * np.sum((E[1] * H[2].conj() - E[2] * H[1].conj()) * fields.grid.weights).real)


def _ey_center_y_um(field):
    intensity = abs(field.E[0, 1])**2 * field.grid.weights
    total = np.sum(intensity)
    return float(np.sum(intensity * np.asarray(field.grid.y)[:, None]) / (total * UM)) if total > 0 else None


def _stairs(geometry, extra):
    """Return the exact DREME rectangles, including straight endpoint leads."""
    path = geometry.parameter_path
    propagator_data = geometry.output_data
    runtime_nodes = tuple(basis.node for basis in propagator_data["runtime_basis"])
    if runtime_nodes != tuple(path.nodes) or not np.allclose(propagator_data["EME_delta_zs"], path.distances, atol=1e-15, rtol=0):
        raise ValueError("FDTD staircase must match the actual DREME propagation segments")
    positions = np.r_[-extra, np.asarray(path.node_positions) / UM, path.node_positions[-1] / UM + extra]
    nodes = (path.nodes[0], *path.nodes)
    return [(float(x0), float(x1), geometry.describe_cross_section({"gap": node[0]}))
            for x0, x1, node in zip(positions[:-1], positions[1:], nodes) if x1 > x0]


def _simulation(geometry, fde, metadata, source, *, mesh_um, run_time_ps, source_x_um, lead_um, monitor_modes):
    stack = geometry.section_model.stack
    if (stack.slab_thickness or stack.box_thickness is not None
            or stack.upper_cladding_material_id != stack.lower_cladding_material_id or fde.boundary != "Metal"):
        raise ValueError("This SOI comparison expects the saved fully etched, symmetric-oxide, Metal-wall stack")
    index = metadata["material_indices"]
    if any(abs(value[1]) > 1e-10 for value in index.values()):
        raise ValueError("This single-wavelength comparison expects lossless Si/oxide")
    media = {key: td.Medium(permittivity=value[0] ** 2, name=key) for key, value in index.items()}
    segments = _stairs(geometry, lead_um + 1.0)
    def bounds(section, guide):
        return next(c for c in section.cores if c.guide_id == guide).lateral_bounds

    lo, hi = [], []
    for x0, x1, section in segments:
        y0, y1 = np.asarray(bounds(section, "wg0")) / UM
        lo.extend(((x0, y0), (x1, y0)))
        hi.extend(((x0, y1), (x1, y1)))
    def deduplicate(vertices):
        return [v for i, v in enumerate(vertices) if i == 0 or v != vertices[i - 1]]

    z0 = stack.device_layer_bottom_z / UM
    z1 = z0 + stack.thickness / UM
    y0, y1 = np.asarray(bounds(segments[0][2], "wg1")) / UM
    silicon = media[stack.core_material_id]
    structures = [
        td.Structure(name="bus", medium=silicon, geometry=td.PolySlab(
            vertices=deduplicate(lo + hi[::-1]), axis=2, slab_bounds=(z0, z1))),
        td.Structure(name="ring", medium=silicon, geometry=td.Box(
            center=(0, (y0 + y1)/2, (z0 + z1)/2), size=(td.inf, y1-y0, z1-z0))),
    ]
    L = geometry.calc_total_length() / UM
    yz = [np.asarray(v) / UM for v in (fde.y_bounds, fde.z_bounds)]
    center = tuple(float(np.mean(v)) for v in yz)
    span = tuple(float(np.ptp(v)) for v in yz)
    freq = td.C_0 / (fde.wavelength / UM)
    # CustomFieldSource interprets dataset coordinates relative to its center.
    field_coords = {"x": [0.0], "y": np.asarray(source.grid.y)/UM - center[0],
                    "z": np.asarray(source.grid.z)/UM - center[1], "f": [freq]}
    # Tidy3D uses micrometers for length and field-area integration.
    fields = {name: td.ScalarFieldDataArray((source.E if name[0] == "E" else source.H)
                [0, "xyz".index(name[1])][None, :, :, None] * UM, coords=field_coords)
              for name in TANGENTIAL}
    emitter = td.CustomFieldSource(
        name="incident_EH", center=(source_x_um, *center), size=(0, *span),
        source_time=td.GaussianPulse(freq0=freq, fwidth=freq / 15),
        field_dataset=td.FieldDataset(**fields),
    )
    monitors = []
    for name, x in (("input", 0.0), ("output", L)):
        plane = dict(center=(x, *center), size=(0, *span), freqs=[freq], colocate=True)
        monitors.append(td.FieldMonitor(name=f"{name}_EH", fields=TANGENTIAL, **plane))
        monitors.append(td.FluxMonitor(name=f"{name}_flux", normal_dir="+", **plane))
        monitors.append(td.ModeMonitor(name=f"{name}_directional", store_fields_direction="+",
                                       mode_spec=td.ModeSpec(num_modes=monitor_modes, precision="double"), **plane))
    return td.Simulation(
        center=(L/2, *center), size=(L+2*lead_um, *span), structures=structures,
        medium=media[stack.upper_cladding_material_id], sources=[emitter], monitors=monitors,
        grid_spec=td.GridSpec.uniform(dl=mesh_um), run_time=run_time_ps * 1e-12,
        boundary_spec=td.BoundarySpec(x=td.Boundary.pml(), y=td.Boundary.pec(), z=td.Boundary.pec()),
    )


def _dreme(dataset, model, name, lengths_um, retained_modes, stability):
    lo, hi = dataset.coupled_settings.gap_grid[0], dataset.coupled_settings.gap_grid[-1]
    if name == "uniform":
        length = lengths_um[0]
        s, gaps = [0, length], [hi, hi]
    elif name == "moving":
        s, gaps = np.r_[0, np.cumsum(lengths_um[1:])], [hi, lo, lo, hi]
    else:
        raise ValueError("case must be 'uniform' or 'moving'")
    geometry = StraightGapCoupler(dataset, model, profile_from_samples(np.asarray(s)*UM, gaps),
                                  limit_mode_number=retained_modes, verbose=False)
    runner = Runner(EME(geometry, stability_config=stability, neff_interpolation=False))
    channel = ChannelRunner(runner, guide_labels={"wg0": "bus", "wg1": "ring"})
    bus = next(c["key"] for c in channel.available_channels("input", verbose=True) if c["display_name"] == "bus_TE0")
    return geometry, channel.run({bus: 1}, input_power=1), bus


def prepare_case(name, dataset_path, reference_path, output_root, *, coarse_dataset=None,
                 lengths_um=(10, 5, 5, 5), mesh_um=0.0125, run_time_ps=3.0,
                 source_kind="isolated", source_x_um=-0.125, lead_um=1.25, retained_modes=24, monitor_modes=24):
    """Prepare only local files; return all information needed after an HDF5 is saved."""
    if (len(lengths_um) != 4 or min(lengths_um) <= 0 or not -lead_um < source_x_um < -2*mesh_um
            or not np.isfinite([*lengths_um, mesh_um, run_time_ps, source_x_um, lead_um]).all()
            or min(mesh_um, run_time_ps) <= 0):
        raise ValueError("Check the lengths, mesh, run time and source offset")
    dataset_path, reference_path = Path(dataset_path).resolve(), Path(reference_path).resolve()
    dataset = DataUpdater(dataset_path, is_testmode=True)
    model = section_model(dataset)
    stability = EMEStabilityConfig()
    geometry, result, bus = _dreme(dataset, model, name, lengths_um, retained_modes, stability)
    diagnostic = result.diagnostics.channel_mapping
    left, right = diagnostic.input_endpoint, diagnostic.output_endpoint
    if left.node != right.node or left.basis_id != right.basis_id:
        raise ValueError("Both endpoints must use the single exported reference node")
    data = dataset.load_channel_data(model)
    records = data["nodes"]
    config = MappingConfig(**data["dataset_metadata"]["mapping_solver_config"])
    metadata, U, V, neff, guided = _load_reference(reference_path, dataset, left.node, left, records)
    canonical_a = left.R @ diagnostic.evaluation.input_forward
    absent = np.ones(len(canonical_a), bool)
    absent[guided] = False
    if np.any(abs(canonical_a[absent]) > 1e-12):
        raise ValueError("Input uses canonical modes absent from the guided reference export")
    mapped = _combine(U, canonical_a[guided])
    power = diagnostic.evaluation.power
    np.testing.assert_allclose(_power(mapped), power.P1, rtol=0, atol=2e-6)
    isolated = _combine(V, diagnostic.input_amplitudes.effective)
    direct = _field_difference(isolated, mapped, fit_phase=False)
    for component in ("E", "H"):
        np.testing.assert_allclose(direct[component], getattr(diagnostic.evaluation.input_residuals, component).relative_norm,
                                   rtol=0, atol=2e-6)
    if source_kind == "isolated":
        target = source = isolated
    elif source_kind == "mapped":
        target = mapped
        # Move the desired x=0 field back to the source through the uniform lead.
        phase = np.exp(1j * 2*np.pi / dataset.wavelength * neff[guided] * source_x_um*UM)
        source = _combine(U, canonical_a[guided] * phase)
    else:
        raise ValueError("source_kind must be 'isolated' or 'mapped'")

    record = records[right.node]
    products = transform_products(PowerProducts(**record["power_products"]), right.R)
    out_basis = geometry.runtime_basis_at(-1)
    maps = solve_channel_maps(products, products, input_guided=out_basis.forward_guided,
                              output_present=out_basis.forward_present, config=config)
    guided_only = maps.B @ np.where(out_basis.forward_guided, diagnostic.transmitted_forward, 0)
    coarse_rows = None
    coarse_manifest = None
    if coarse_dataset is not None:
        other = DataUpdater(coarse_dataset, is_testmode=True)
        model.validate_configuration(other.canonical_manifest["section_model"])
        if (other.canonical_manifest["settings"]["fde"] != dataset.canonical_manifest["settings"]["fde"]
                or other.canonical_manifest["export_grid"] != dataset.canonical_manifest["export_grid"]
                or not set(other.coupled_settings.gap_grid).issubset(dataset.coupled_settings.gap_grid)):
            raise ValueError("Coarse dataset must differ only in gap sampling")
        _, coarse, coarse_bus = _dreme(other, model, name, lengths_um, retained_modes, stability)
        if bus != coarse_bus or result.output_labels != coarse.output_labels:
            raise ValueError("Coarse grid channel inventory differs")
        coarse_rows = coarse.output_amplitudes
        coarse_manifest = {"path": str(Path(coarse_dataset).resolve()),
                           "generation_id": other.canonical_manifest["generation_id"],
                           "gap_grid_m": other.coupled_settings.gap_grid}
    rows = []
    for (key, amplitude), guided_amp in zip(result.output_amplitudes.items(), guided_only):
        rows.append({"guide_id": key[0], "fde_mode": key[1], "channel": result.output_labels[key],
                     "dreme_W": abs(amplitude)**2, "dreme_guided_readout_W": abs(guided_amp)**2,
                     "dreme_coarse_W": None if coarse_rows is None else abs(coarse_rows[key])**2})
    simulation = _simulation(geometry, dataset.coupled_settings.fde, metadata, source, mesh_um=mesh_um,
                             run_time_ps=run_time_ps, source_x_um=source_x_um, lead_um=lead_um,
                             monitor_modes=monitor_modes)
    path = geometry.parameter_path
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {
        "case": name, "git_head": sha, "comparison_code_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "python": sys.version.split()[0], "tidy3d": td.__version__, "numpy": np.__version__,
        "dataset": str(dataset_path), "dataset_generation_id": metadata["generation_id"],
        "coarse_dataset": coarse_manifest, "endpoint_reference": str(reference_path),
        "endpoint_reference_sha256": _digest(reference_path), "reference_metadata": metadata,
        "section_model": model.configuration(), "fde": dataset.canonical_manifest["settings"]["fde"],
        "reference_planes_m": {"input": left.s_m, "output": right.s_m},
        "endpoint_basis": {"input": asdict(left), "output": asdict(right)},
        "path": {"requested_x_m": path.sample_positions, "requested_gap_m": path.requested_values,
                 "grid_x_m": path.node_positions, "grid_gap_m": [node[0] for node in path.nodes]},
        "source": {"kind": source_kind, "key": bus, "x_um": source_x_um, "source_power_W": _power(source),
                   "target_power_W": _power(target), "target_Ey_center_y_um": _ey_center_y_um(target)},
        "fdtd": {"mesh_um": mesh_um, "run_time_ps": run_time_ps, "lead_um": lead_um, "monitor_modes": monitor_modes,
                 "boundary": ["PML", "PEC", "PEC"], "material_model": "Palik phase index at 1550 nm; nondispersive",
                 "domain_center_um": simulation.center, "domain_size_um": simulation.size,
                 "grid_cells": simulation.num_cells},
        "dreme": {"retained_modes": retained_modes, "stability_config": asdict(stability),
                  "neff_interpolation": False, "power": asdict(power),
                  "input_residuals": asdict(diagnostic.evaluation.input_residuals), "input_direct_EH_residual": direct,
                  "output_residuals": asdict(diagnostic.evaluation.output_residuals),
                  "mapping_config": asdict(config), "input_solve": asdict(diagnostic.input_solve),
                  "output_solve": asdict(diagnostic.output_solve), "mapping_warnings": diagnostic.warnings,
                  "stability": result.diagnostics.propagation_stability},
    }
    directory = Path(output_root).resolve() / name
    directory.mkdir(parents=True, exist_ok=True)
    path_json = directory / "case.json"
    if path_json.exists():
        saved = _read_json(path_json)
        # Committing the same comparison code after preflight must not orphan the draft/result.
        manifest["git_head"] = saved["git_head"]
        if saved != json.loads(_json(manifest)):
            raise ValueError(f"Case settings changed. Use a new OUTPUT_ROOT: {directory}")
    if not (directory / "simulation.hdf5").exists():
        simulation.to_file(directory / "simulation.hdf5")
    _save_json(path_json, manifest)
    pd.DataFrame(rows).to_csv(directory / "dreme_preflight.csv", index=False)
    _log(directory, "Local preflight complete")
    return Case(name, directory, simulation, manifest, V, target, config, rows)


def plot_case(case):
    sim = case.simulation
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.4), layout="constrained")
    z_mid = sum(sim.structures[0].geometry.slab_bounds)/2
    sim.plot(z=z_mid, ax=axes[0])
    sim.plot(x=0, ax=axes[1])
    for label, position, color in (("source", sim.sources[0].center[0], "C2"),
                                   ("input", 0.0, "C3"), ("output", sim.center[0]*2, "C4")):
        axes[0].axvline(position, ls="--", color=color, label=label)
    axes[0].legend(loc="lower right")
    axes[0].set_title(f"{case.name}: snapped plan, source and planes")
    axes[1].set_title("Input cross-section")
    return fig


def _plane_component(data, component, grid, modal):
    field = getattr(data, component)
    if field is None or len(field.f) != 1 or len(field.x) != 1:
        raise ValueError(f"Missing monochromatic +x plane field {component} in {data.monitor.name}")
    field = field.isel(f=0, x=0)
    dims = ("y", "z", "mode_index") if modal else ("y", "z")
    values = field.transpose(*dims).values
    ys, zs = np.asarray(grid.y)/UM, np.asarray(grid.z)/UM
    yy, zz = np.meshgrid(ys, zs, indexing="ij")
    for src, dst in ((field.y.values, ys), (field.z.values, zs)):
        if len(src) < 2 or max(src[0]-dst[0], dst[-1]-src[-1]) > 1.01 * np.max(np.diff(src)):
            raise ValueError(f"{data.monitor.name} does not cover the FDE projection plane")
    interp = RegularGridInterpolator((field.y.values, field.z.values), values, bounds_error=False, fill_value=0.0)
    result = interp((yy, zz))/UM
    return np.moveaxis(result, -1, 0) if modal else result[None]


def _plane_fields(data, grid, modal):
    ey, ez, hy, hz = (_plane_component(data, c, grid, modal) for c in TANGENTIAL)
    zeros = np.zeros_like(ey)
    return ForwardFields(np.stack((zeros, ey, ez), axis=1), np.stack((zeros, hy, hz), axis=1), grid)


def _plane(data, name, grid):
    modes = data[f"{name}_directional"].grid_corrected_copy
    measured = _plane_fields(data[f"{name}_EH"].grid_corrected_copy, grid, modal=False)
    fields = _plane_fields(modes, grid, modal=True)
    plus, minus = (modes.amps.sel(direction=direction).isel(f=0).transpose("mode_index").values for direction in ("+", "-"))
    forward = _combine(fields, plus)
    reciprocal = ForwardFields(fields.E.conj(), -fields.H.conj(), grid)
    backward = _combine(reciprocal, minus)
    reconstructed = ForwardFields(forward.E + backward.E, forward.H + backward.H, grid)
    native_flux = float(data[f"{name}_flux"].flux.isel(f=0).values)
    reconstructed_flux = _power(reconstructed)
    return forward, {"native_signed_flux_W": native_flux, "measured_signed_flux_W": _power(measured),
                     "raw_Ey_center_y_um": _ey_center_y_um(measured),
                     "reconstructed_signed_flux_W": reconstructed_flux,
                     "reconstructed_over_native_signed_flux": reconstructed_flux/native_flux if abs(native_flux) > 1e-12 else None,
                     "forward_flux_W": _power(forward),
                     "backward_flux_magnitude_W": -_power(backward),
                     "mode_forward_power_W": float(np.sum(abs(plus)**2)),
                     "mode_backward_power_W": float(np.sum(abs(minus)**2)),
                     "raw_EH_reconstruction": _field_difference(measured, reconstructed, fit_phase=False)}


def _field_difference(reference, measured, *, fit_phase):
    weights = reference.grid.weights
    terms = {label: (getattr(reference, label)[0, 1:], getattr(measured, label)[0, 1:]) for label in ("E", "H")}
    norms = {label: np.sum(abs(a)**2 * weights) for label, (a, _) in terms.items()}
    if any(value <= 0 for value in norms.values()):
        raise ValueError("Zero E or H norm at comparison plane")
    # Fit one common phase, preserving the relative E/H phase and impedance.
    cross = sum(np.sum(a.conj()*b*weights) / norms[label] for label, (a, b) in terms.items())
    factor = cross/abs(cross) if fit_phase and abs(cross) else 1.0
    return {label: float(np.sqrt(np.sum(abs(b-factor*a)**2 * weights) / norms[label]))
            for label, (a, b) in terms.items()}


def compare_saved(case, result_path=None):
    """Compare from local HDF5. Neither upload nor solver execution is involved."""
    path = Path(result_path) if result_path else case.directory / "result.hdf5"
    data = td.SimulationData.from_file(path)
    if data.diverged or data.simulation != case.simulation:
        raise ValueError("Result diverged or its simulation differs from case.json/simulation.hdf5")
    V = case.references
    launched, incoming = _plane(data, "input", V.grid)
    outgoing, output = _plane(data, "output", V.grid)
    incident = incoming["forward_flux_W"]
    if incident <= 0 or not np.isfinite(incident):
        raise ValueError("Could not measure a positive incident forward power")
    unit = _combine(launched, [np.sqrt(_power(case.target)/incident)])
    incoming["target_Ey_center_y_um"] = _ey_center_y_um(case.target)
    incoming["target_EH_relative_residual_after_power_and_phase_alignment"] = _field_difference(
        case.target, unit, fit_phase=True)
    products = compute_power_products(outgoing, V)
    # Fit the same local isolated-guide basis with the same SVD policy as ChannelRunner.
    solution, projection_info = _solve(products.G_V, products.C.conj().T, case.config, "Tidy3D output projection")
    coeff = solution[:, 0]
    projected = _combine(V, coeff)
    output["projected_coherent_power_W"] = _power(projected)
    output["projection_EH_relative_residual"] = _field_difference(outgoing, projected, fit_phase=False)
    output["projection_solve"] = asdict(projection_info)

    power = case.manifest["dreme"]["power"]
    kind = case.manifest["source"]["kind"]
    primary = "P1" if kind == "mapped" else "P0"
    table = pd.DataFrame(case.dreme_rows)
    for key in ("P0", "P1"):
        table[f"dreme_{key}_W"] = power[key]
        table[f"dreme_over_{key}"] = table["dreme_W"]/power[key]
    table["tidy_incident_W"] = incident
    table["tidy_coefficient_W"] = abs(coeff)**2
    table["tidy_over_incident"] = table["tidy_coefficient_W"]/incident
    table["dreme_primary_denominator"] = primary
    table["tidy_minus_dreme_pp"] = 100*(table["tidy_over_incident"]-table[f"dreme_over_{primary}"])
    table.to_csv(case.directory / "comparison.csv", index=False)
    details = {"case": case.manifest, "hdf5": str(path.resolve()), "hdf5_sha256": _digest(path),
               "input": incoming, "output": output, "primary_dreme_denominator": primary,
               "tidy_denominator": "input +x modal E/H integrated on the shared plane",
               "channel_power": "unit isolated reference coefficient squared; totals can include cross terms"}
    if (case.directory / "task.json").exists():
        details["task"] = _read_json(case.directory / "task.json")
    _save_json(case.directory / "comparison_diagnostics.json", details)
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    table.set_index("channel")[[f"dreme_over_{primary}", "tidy_over_incident"]].plot.bar(ax=ax, rot=30)
    ax.set(ylabel="Relative coefficient power", title=case.name)
    fig.savefig(case.directory / "comparison.png", dpi=150)
    _log(case.directory, "Offline comparison complete")
    return table, fig, details


def upload_and_estimate(case, *, folder="DREME coupled SOI", solver_version=None):
    """User-triggered draft upload and estimate only."""
    from tidy3d import web

    path = case.directory / "task.json"
    if path.exists():
        task = _read_json(path)
        if task["solver_version_requested"] != solver_version:
            raise ValueError("Existing task uses a different solver version")
    else:
        _log(case.directory, "Requesting draft upload")
        task = {"task_id": web.upload(case.simulation, task_name=f"dreme-soi-{case.name}", folder_name=folder,
                                     solver_version=solver_version, verbose_estimate_cost=False, reduce_simulation=False),
                "solver_version_requested": solver_version}
        _save_json(path, task)
        _log(case.directory, f"Draft uploaded: {task['task_id']}")
    _log(case.directory, f"Requesting cost estimate for {task['task_id']}")
    task["max_flexcredits"] = web.estimate_cost(task["task_id"], solver_version=solver_version)
    _save_json(path, task)
    _log(case.directory, f"Estimated maximum FlexCredits: {task['max_flexcredits']}")
    return task


def run_or_download(case, *, approved_task_id=None):
    """User-triggered start for one explicitly approved draft, or resume/load."""
    from tidy3d import web
    from tidy3d.web.api.states import DRAFT_STATES, POSTPROCESS_STATES, PREPROCESS_STATES, QUEUED_STATES, RUNNING_STATES, SUCCESS_STATES

    result = case.directory / "result.hdf5"
    if result.exists():
        return td.SimulationData.from_file(result)
    path = case.directory / "task.json"
    task = _read_json(path)
    task_id = task["task_id"]
    _log(case.directory, f"Checking task {task_id}")
    info = web.get_info(task_id)
    status = info.status
    if status in DRAFT_STATES:
        estimate = task.get("max_flexcredits")
        if approved_task_id != task_id or estimate is None or not np.isfinite(estimate):
            raise ValueError("Review the maximum cost, then enter this draft's exact task ID to start it")
        _log(case.directory, f"Starting approved draft {task_id}")
        web.start(task_id, solver_version=task["solver_version_requested"])
        web.monitor(task_id)
    elif status in QUEUED_STATES | PREPROCESS_STATES | RUNNING_STATES | POSTPROCESS_STATES:
        web.monitor(task_id)
    elif status not in SUCCESS_STATES:
        raise ValueError(f"Task {task_id} has status {status}; inspect it in Tidy3D")
    info = web.get_info(task_id)
    task["final_status"] = info.status
    task["solver_version"] = getattr(info, "solver_version", None)
    _save_json(path, task)
    if info.status not in SUCCESS_STATES:
        raise ValueError(f"Task {task_id} did not finish successfully: {info.status}")
    partial = case.directory / "result.partial.hdf5"
    data = web.load(task_id, path=partial)
    partial.replace(result)
    if data.log:
        (case.directory / "solver.log").write_text(data.log, encoding="utf-8")
    _log(case.directory, f"Saved result for {task_id}: {result}")
    return data
