"""Build local channel references alongside one fresh canonical coupled generation."""

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Optional
import warnings

import numpy as np

from ..channels import FIELD_CONVENTION, ForwardFields, SectionGrid
from ..channels.diagnostics import (
    compute_field_products, evaluate_direct_residual, evaluate_mapping_diagnostics, reconstruct_fields,
)
from ..channels.mapping import (
    ChannelMappingWarning, MappingConfig, compute_power_products, normalize_channel_fields, solve_channel_maps,
)
from .canonical_dataset import assemble, read_mode_card, read_mode_cards, read_mode_neff, unpack_mode_fields
from .channel_store import (
    SCHEMA_VERSION, save_channel_checkpoint, validate_channel_linkage, validate_channel_node,
)
from .lumerical_section import LumericalSectionAdapter


@dataclass(frozen=True)
class _Isolated:
    fields: Optional[ForwardFields]
    channels: list
    counts: dict


def _section_grid(grid, wavelength):
    # unpack_mode_fields has checked all vector E/H result coordinates on this plane.
    return SectionGrid(grid.x, grid.y, grid.z, wavelength, "collocated")


def _guided(neff, cladding_index, max_abs_imag_neff):
    return (neff.real >= cladding_index) & (abs(neff.imag) <= max_abs_imag_neff)


def _isolation_guides(configuration, sections):
    guides = configuration["guides"]
    fixed = [guide for guide, model in guides.items() if model["placement"]["kind"] == "fixed"]
    if len(guides) != 2 or len(fixed) != 1:
        raise ValueError("channel generation requires exactly one fixed and one moving guide")
    fixed = fixed[0]
    moving, = guides.keys() - {fixed}
    first = next(iter(sections.values()))
    fixed_cores = tuple(core for core in first.cores if core.guide_id == fixed)
    for node, section in sections.items():
        cores = tuple(core for core in section.cores if core.guide_id == fixed)
        if (not fixed_cores or cores != fixed_cores or section.stack != first.stack
                or {core.guide_id for core in section.cores} != guides.keys()):
            raise ValueError(f"fixed guide {fixed!r}, stack or guide inventory changed at node {node}")
    return moving, fixed


def _direct_power(fields, amplitudes):
    E, H = reconstruct_fields(fields, amplitudes)
    return float(0.5 * np.sum((E[1] * H[2].conj() - E[2] * H[1].conj()).real * fields.grid.weights))


def validate_baseline(power, fields, report, baseline, context):
    """Compare matrix results with the direct integrals recorded at acquisition."""
    # Cancellation can leave a small result from large terms. Use a modest
    # multiple of float64 epsilon on the sum of their absolute magnitudes.
    roundoff = 32 * np.finfo(float).eps
    relaxed = []

    def term_scale(left, matrix, right):
        return float(abs(left) @ abs(matrix) @ abs(right))

    def check(label, expected, actual, scale, reference):
        difference = abs(actual - expected)
        strict = 1e-10 * (abs(expected) + reference)
        if not np.isfinite(difference) or not np.isfinite(scale) or difference > max(strict, roundoff * scale):
            raise ValueError(f"{context}: {label} mismatch: {expected:.6g} vs {actual:.6g}")
        if difference > strict:
            relaxed.append(label)

    p, a_out = baseline["input_channels"], baseline["output_forward"]
    a_in, p_out = report.input_forward, report.output_channels
    for name, endpoint, expected, u, v in zip(
        ("input", "output"), (report.input_residuals, report.output_residuals),
        baseline["direct_residual_squared"], (a_in, a_out), (p, p_out),
    ):
        for field, blocks, direct, residual in zip(
            ("E", "H"), (fields.E, fields.H), expected, (endpoint.E, endpoint.H),
        ):
            scale = (term_scale(u, blocks.S_U, u) + term_scale(v, blocks.S_V, v)
                     + 2 * term_scale(u, blocks.X, v))
            check(f"{name} {field} direct/matrix residual", direct, residual.squared_norm,
                  scale, residual.reference_squared_norm)

    for i, (expected, amplitudes, gram) in enumerate(zip(
        baseline["direct_power_w"], (p, a_in, a_out, p_out), (power.G_V, power.G_U, power.G_U, power.G_V),
    )):
        check(f"P{i} direct/matrix Poynting power", expected, getattr(report.power, f"P{i}"),
              term_scale(amplitudes, gram, amplitudes), 1.0)

    if relaxed:
        warnings.warn(f"{context}: direct/matrix differences within coefficient-scaled roundoff for "
                      f"{', '.join(relaxed)}", ChannelMappingWarning, stacklevel=2)


def _baseline(U, V, power, fields, guided, config, context):
    """Compare the products with direct reconstruction using deterministic coherent vectors."""
    maps = solve_channel_maps(
        power, power, input_guided=guided, output_present=np.ones(U.count, dtype=bool),
        config=config, input_context=f"{context} input", output_context=f"{context} output",
    )
    p = 1j ** np.arange(V.count)
    a_out = 1j ** np.arange(U.count)
    report = evaluate_mapping_diagnostics(power, power, fields, fields, maps, p, a_out)
    direct = (
        evaluate_direct_residual(V, p, U, report.input_forward, config=config),
        evaluate_direct_residual(U, a_out, V, report.output_channels, config=config),
    )
    direct_watts = tuple(_direct_power(f, a) for f, a in (
        (V, p), (U, report.input_forward), (U, a_out), (V, report.output_channels),
    ))
    baseline = {
        "input_channels": p, "output_forward": a_out,
        "input_solve": asdict(maps.input_solve), "output_solve": asdict(maps.output_solve),
        "input_residuals": asdict(report.input_residuals), "output_residuals": asdict(report.output_residuals),
        "power": asdict(report.power), "direct_power_w": direct_watts,
        "direct_residual_squared": tuple((endpoint.E.squared_norm, endpoint.H.squared_norm)
                                         for endpoint in direct),
    }
    validate_baseline(power, fields, report, baseline, context)
    return baseline


def build_channel_data(dataset, geometry, *, mode_api=None, hide=False, max_abs_imag_neff=0.000284,
                       mapping_config=MappingConfig(), on_solve=None):
    """Build coupled and isolated products in one DataUpdater-owned MODE session.

    Start with a fresh coupled DataUpdater and a bound geometry. on_solve, if
    supplied, receives (problem, node, findmodes_seconds, returned_count).
    The updater retains ownership and leaves its session open for inspection.
    """
    if (not hasattr(dataset, "coupled_settings") or dataset._is_testmode
            or dataset.canonical_manifest is not None or dataset.mode is not None):
        raise ValueError("channel generation requires a fresh DataUpdater with coupled_settings")
    max_abs_imag_neff = float(max_abs_imag_neff)
    if not np.isfinite(max_abs_imag_neff) or max_abs_imag_neff < 0:
        raise ValueError("max_abs_imag_neff must be finite and nonnegative")
    settings = dataset.coupled_settings
    configuration = geometry.section_model.configuration()
    fde = settings.fde
    guide_budgets = {guide: count for guide, count in fde.candidate_counts.items() if guide != "coupled"}

    def guided(neff):
        return _guided(neff, settings.cladding_index, max_abs_imag_neff)

    channel_data = {
        "schema_version": SCHEMA_VERSION,
        "dataset_metadata": {
            "settings": settings.configuration(), "section_model": configuration,
            "field_convention": FIELD_CONVENTION, "parameter_units": {"gap": "m"},
            "isolated_normalization_w": 1.0, "isolated_phase": "node-local MODE phase",
            "guided_filter": {"cladding_index": settings.cladding_index, "max_abs_imag_neff": max_abs_imag_neff},
            "mapping_solver_config": asdict(mapping_config),
        },
        "nodes": {}, "generation_completion": "building",
    }
    moving = fixed = ring = mode = adapter = None
    previous = None
    sections = {}

    def find_modes(problem, node):
        started = perf_counter()
        result = np.asarray(mode.findmodes()).item()
        elapsed = perf_counter() - started
        returned = int(result)
        if returned < 0 or returned != result:
            raise ValueError(f"invalid MODE mode count {result!r}")
        if on_solve is not None:
            on_solve(problem, node, elapsed, returned)
        return returned

    def solve_isolated(node, section, guide):
        requested = guide_budgets[guide]
        counts = {"requested_count": requested}
        try:
            with adapter.isolated(section, guide):
                returned = find_modes(guide, node)
                neff = np.array([read_mode_neff(mode, i) for i in range(1, returned + 1)], dtype=complex)
                selected = tuple(int(i) + 1 for i in np.flatnonzero(guided(neff)))
                counts.update(returned_count=returned, guided_count=len(selected))
                if not selected:
                    counts["empty_reason"] = ("no returned modes passed the guided filter" if returned
                                              else "MODE returned no modes")
                    return _Isolated(None, [], counts)
                modes = [read_mode_card(mode, i, neff=neff[i - 1]) for i in selected]
                grid, E, H, neff, te = unpack_mode_fields(modes, selected)
                fields = ForwardFields(E, H, _section_grid(grid, fde.wavelength))
                fields, factors = normalize_channel_fields(fields, config=mapping_config)
                channels = [{
                    "key": (guide, i), "neff": complex(index), "te_fraction": float(fraction),
                    "normalization_scale": float(scale),
                } for i, index, fraction, scale in zip(selected, neff, te, factors)]
                return _Isolated(fields, channels, counts)
        except Exception as exc:
            raise RuntimeError(f"isolated guide {guide!r} at node {node} failed: {exc}") from exc

    def open_reader(validated_sections):
        nonlocal moving, fixed, ring, mode, adapter
        sections.update(validated_sections)
        moving, fixed = _isolation_guides(configuration, sections)
        mode = dataset._open_coupled_mode(mode_api, hide)
        adapter = LumericalSectionAdapter(mode, fde)
        first = next(iter(sections))
        ring = solve_isolated(first, sections[first], fixed)

        def read_node(node, section):
            adapter.prepare(section)
            return read_mode_cards(mode, find_modes("coupled", node), dataset.mode_numbers)

        return read_node

    def checkpoint(current):
        nonlocal previous
        node = current.node
        U = ForwardFields(current.E, current.H, _section_grid(current.grid, fde.wavelength))
        bus = solve_isolated(node, sections[node], moving)
        references = {moving: bus, fixed: ring}
        channels, counts, E, H = [], {}, [U.E[:0]], [U.H[:0]]
        for guide, reference in sorted(references.items()):
            if reference.fields is not None:
                try:
                    U.grid.require_match(reference.fields.grid)
                except ValueError as exc:
                    raise ValueError(f"isolated guide {guide!r} at node {node}: {exc}") from exc
                E.append(reference.fields.E)
                H.append(reference.fields.H)
            channels.extend(reference.channels)
            counts[guide] = reference.counts
            if previous is not None:
                before = channel_data["nodes"][previous]["guide_counts"][guide]["guided_count"]
                after = reference.counts["guided_count"]
                if before != after:
                    warnings.warn(f"guided count for {guide!r} changed from {before} at {previous} "
                                  f"to {after} at {node}", ChannelMappingWarning, stacklevel=2)
        V = ForwardFields(np.concatenate(E), np.concatenate(H), U.grid)
        power = compute_power_products(U, V)
        fields = compute_field_products(U, V)
        basis = current.basis_id
        baseline = _baseline(U, V, power, fields, guided(current.neff), mapping_config, f"node {node}")
        if not channel_data["nodes"]:
            channel_data["dataset_metadata"].update(
                generation_id=current.generation_id, export_grid=asdict(current.grid),
            )
        record = {
            "completion": "complete", "canonical_basis_id": basis,
            "canonical_forward_mode_ids": current.mode_numbers, "channel_basis_id": f"{basis}:channels",
            "guide_counts": counts, "channels": channels,
            "power_products": {key: getattr(power, key) for key in ("G_U", "G_V", "C")},
            "field_products": {
                name: {key: getattr(getattr(fields, name), key) for key in ("S_U", "S_V", "X")}
                for name in ("E", "H")
            },
            "baseline_diagnostics": baseline,
        }
        validate_channel_node(record, basis, current.mode_numbers, guide_budgets)
        channel_data["nodes"][node] = record
        save_channel_checkpoint(dataset.data_directory, channel_data, node=node)
        previous = node

    def finish_channels(manifest):
        channel_data["generation_completion"] = "complete"
        validate_channel_linkage(channel_data, manifest)
        return save_channel_checkpoint(dataset.data_directory, channel_data)

    manifest = assemble(dataset, geometry, open_reader, checkpoint, finish_channels=finish_channels)
    return dataset._finish_coupled(manifest)
