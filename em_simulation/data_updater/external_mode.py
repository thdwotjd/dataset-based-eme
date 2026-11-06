"""Prepare and import coupled MODE projects executed outside Python."""

from dataclasses import dataclass
import json
from pathlib import Path
from uuid import uuid4

import numpy as np

from .canonical_dataset import CoupledFields, ModeFields, geometry_sections, read_mode_cards
from .lumerical_section import LumericalSectionAdapter


PLAN = "external_fde_jobs.json"
GROUP = "::model::dreme_gap"
SWEEP = "dreme_gap_sweep"
GAP_ATOL_M = 1e-15
EXPORT_VAR = "dreme_external_result"
RECOVERY = (
    "Keep the solved sweep child .lms files and import them as node results, or run the "
    "prepared export .lsf in each solved child while its mode cards are present and import the .ldf."
)


@dataclass(frozen=True)
class ModeResult:
    """An explicit result file and the preparation plan that identifies its job."""

    path: Path
    plan: Path
    kind: str = "node"


def _scalar(value):
    return np.asarray(value).item()


def _match_gap(value, gaps):
    value = float(_scalar(value))
    matched = [gap for gap in gaps if abs(value - gap) <= GAP_ATOL_M]
    if len(matched) != 1:
        raise ValueError(f"gap {value!r} m does not identify exactly one planned gap in {list(gaps)}")
    return matched[0]


def _vertices(core):
    yz = np.asarray(core.vertices, dtype=float)
    return np.column_stack((-yz[:, 1], yz[:, 0]))


def _number(value):
    return format(float(value), ".17g")


def _quote(value):
    return json.dumps(str(value), ensure_ascii=False)


def _gap_script(moving_core, gap):
    vertices = _vertices(moving_core)
    rows = "; ".join(", ".join(map(_number, row)) for row in vertices)
    object_name = "dreme_core_" + moving_core.region_id
    return (
        f"dreme_v = [{rows}];\n"
        f"dreme_v(:,2) = dreme_v(:,2) + (gap - ({_number(gap)}));\n"
        f"select({_quote(object_name)});\n"
        'set("vertices", dreme_v);\n'
    )


def _group_bus(mode, sections, gaps, job_id, configuration):
    """Move only the translated core into a group driven by the true gap property."""
    base = sections[(gaps[0],)]
    placements = {guide_id: guide["placement"]
                  for guide_id, guide in configuration["guides"].items()}
    moving = [core for core in base.cores if placements[core.guide_id].get("kind") == "edge_gap"]
    if len(base.cores) != 2 or len(moving) != 1:
        raise ValueError("external gap jobs require one fixed core and one gap-driven core")
    bus = moving[0]
    placement = placements[bus.guide_id]
    if (placement.get("direction"), placement.get("parameter")) != ("+y", "gap"):
        raise ValueError("external gap jobs require translation in +y driven by gap")
    for gap in gaps:
        current = sections[(gap,)]
        cores = {core.region_id: core for core in current.cores}
        if current.stack != base.stack or cores.keys() != {core.region_id for core in base.cores}:
            raise ValueError(f"stack or region IDs changed at gap {gap} m")
        for core in base.cores:
            actual = cores[core.region_id]
            if (actual.guide_id, actual.material_id) != (core.guide_id, core.material_id):
                raise ValueError(f"{core.region_id} guide ID or material changed at gap {gap} m")
            offset = gap - gaps[0] if core.region_id == bus.region_id else 0
            _require_value(_vertices(actual), _vertices(core) + [0, offset],
                           f"{core.region_id} polygon at gap {gap} m")
    name = "dreme_core_" + bus.region_id
    script = _gap_script(bus, gaps[0])
    mode.addstructuregroup()
    mode.set("name", GROUP.split("::")[-1])
    mode.set("construction group", 0)
    for axis in "xyz":
        mode.set(axis, 0.0)
    mode.adduserprop("gap", 2, gaps[0])
    mode.adduserprop("dreme_job_id", 1, job_id)
    mode.select(f"::model::{name}")
    mode.addtogroup(GROUP)
    mode.setnamed(GROUP, "script", script)
    mode.runsetup()
    return name


def _capture_layout(mode, adapter, bus_name):
    layout = []
    for name, properties in adapter.object_properties.items():
        obj = f"{GROUP}::{name}" if name == bus_name else f"::model::{name}"
        layout.extend((obj, prop, np.asarray(mode.getnamed(obj, prop)).tolist())
                      for prop in properties)
    layout.extend((GROUP, prop, np.asarray(mode.getnamed(GROUP, prop)).tolist())
                  for prop in ("x", "y", "z", "script"))
    mode.select("::model::FDE")
    layout.extend((None, prop, np.asarray(mode.getanalysis(prop)).tolist())
                  for prop in adapter.analysis_properties)
    return layout


def _require_value(actual, expected, context):
    a, b = np.asarray(actual).squeeze(), np.asarray(expected).squeeze()
    if a.dtype.kind in "biufc" and b.dtype.kind in "biufc":
        same = a.shape == b.shape and np.allclose(a, b, atol=GAP_ATOL_M, rtol=1e-12)
    else:
        same = a.shape == b.shape and np.array_equal(a, b)
    if not same:
        raise ValueError(f"{context} differs from prepared layout: expected {expected!r}, got {actual!r}")


def _sweep_result_properties(count):
    for i in range(1, count + 1):
        card = f"::model::FDE::data::mode{i}"
        for suffix, attribute in (("E", "E"), ("H", "H"), ("neff", "neff"),
                                  ("TEpol", "TE polarization fraction"), ("x", "x"),
                                  ("y", "y"), ("z", "z")):
            yield {"Name": f"mode{i}_{suffix}", "Result": f"{card}::{attribute}"}


def _add_sweep(mode, gaps, count):
    mode.addsweep(0)
    mode.setsweep("sweep", "name", SWEEP)
    mode.setsweep(SWEEP, "type", "values")
    mode.setsweep(SWEEP, "number of points", len(gaps))
    mode.addsweepparameter(SWEEP, {
        "Name": "gap", "Parameter": f"{GROUP}::gap", "Type": "Length",
        **{f"value_{i}": gap for i, gap in enumerate(gaps, 1)},
    })
    for result in _sweep_result_properties(count):
        mode.addsweepresult(SWEEP, result)


def _export_script(path, layout, count):
    lines = [
        f"dreme_export_path = {_quote(path.as_posix())};",
        f"{EXPORT_VAR} = struct;",
        f"{EXPORT_VAR}.gap = getnamed({_quote(GROUP)},\"gap\");",
        f"{EXPORT_VAR}.job_id = getnamed({_quote(GROUP)},\"dreme_job_id\");",
        f"{EXPORT_VAR}.returned = nummodes;",
        'select("::model::FDE");',
    ]
    for i, (obj, prop, _) in enumerate(layout):
        command = f"getanalysis({_quote(prop)})" if obj is None else f"getnamed({_quote(obj)},{_quote(prop)})"
        lines.append(f"{EXPORT_VAR}.p{i} = {command};")
    for i in range(1, count + 1):
        card = _quote(f"FDE::data::mode{i}")
        for attr in ("E", "H"):
            lines.append(f'{EXPORT_VAR}.mode{i}_{attr} = getresult({card},"{attr}");')
        lines.extend((
            f'{EXPORT_VAR}.mode{i}_neff = getdata({card},"neff");',
            f'{EXPORT_VAR}.mode{i}_TEpol = getdata({card},"TE polarization fraction");',
        ))
    lines.append(f"savedata(dreme_export_path, {EXPORT_VAR});")
    lines.append('?("DREME saved " + dreme_export_path);')
    return "\n".join(lines) + "\n"


def prepare_mode_jobs(geometry, folder, *, mode_api, workflow, batches=None):
    """Save projects and export scripts. This never launches a solve or sweep."""
    configuration, sections = geometry_sections(geometry)
    settings = geometry.data.coupled_settings
    gaps = settings.gap_grid
    if workflow not in ("files", "sweep"):
        raise ValueError("workflow must be 'files' or 'sweep'")
    if any(b - a <= 2 * GAP_ATOL_M for a, b in zip(gaps, gaps[1:])):
        raise ValueError("external MODE job gaps are too close to resolve")
    if batches is None:
        batches = [(gap,) for gap in gaps] if workflow == "files" else [gaps]
    batches = [tuple(_match_gap(gap, gaps) for gap in batch) for batch in batches]
    combined = [gap for batch in batches for gap in batch]
    if not combined or len(combined) != len(set(combined)) or any(not batch for batch in batches):
        raise ValueError("job batches must be nonempty and must not repeat a gap")
    if workflow == "files" and any(len(batch) != 1 for batch in batches):
        raise ValueError("file jobs require one gap per batch")
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    jobs = []
    count = settings.get_mode_numbers()
    for i, batch in enumerate(batches, 1):
        project = folder / f"{workflow}_{i:03}.lms"
        job_id = uuid4().hex
        mode = mode_api.MODE(hide=True)
        try:
            adapter = LumericalSectionAdapter(mode, settings.fde)
            adapter.prepare(sections[(batch[0],)])
            bus_name = _group_bus(mode, sections, batch, job_id, configuration)
            layout_by_gap = []
            for gap in batch:
                mode.setnamed(GROUP, "gap", gap)
                mode.runsetup()
                for core in sections[(gap,)].cores:
                    name = "dreme_core_" + core.region_id
                    obj = f"{GROUP}::{name}" if name == bus_name else f"::model::{name}"
                    _require_value(mode.getnamed(obj, "vertices"), _vertices(core), f"{obj} gap {gap}")
                layout = _capture_layout(mode, adapter, bus_name)
                if layout_by_gap:
                    for (obj, prop, value), (_, _, reference) in zip(layout, layout_by_gap[0]["layout"]):
                        if (obj, prop) != (f"{GROUP}::{bus_name}", "vertices"):
                            _require_value(value, reference, f"{obj} {prop} at gap {gap}")
                suffix = f"{workflow}_{i:03}_node_{len(layout_by_gap) + 1:03}"
                export = folder / f"{suffix}_export.lsf"
                result = folder / f"{suffix}.ldf"
                export.write_text(_export_script(result, layout, count), encoding="utf-8")
                layout_by_gap.append({
                    "gap_m": gap, "layout": layout, "export_script": export.name, "result_ldf": result.name,
                })
            mode.setnamed(GROUP, "gap", batch[0])
            mode.runsetup()
            if workflow == "sweep":
                _add_sweep(mode, batch, count)
            mode.save(str(project))
        except Exception as exc:
            raise RuntimeError(f"Could not prepare {project} for gaps {batch}: {exc}") from exc
        finally:
            mode.close()
        jobs.append({"id": job_id, "project": project.name, "workflow": workflow, "points": layout_by_gap})
    plan = folder / PLAN
    plan.write_text(json.dumps({
        "version": 1, "settings": settings.configuration(),
        "section_model": configuration, "jobs": jobs,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return plan


def _point(job, value):
    gap = _match_gap(value, [point["gap_m"] for point in job["points"]])
    return next(point for point in job["points"] if point["gap_m"] == gap)


def _check_layout(point, reader):
    for i, (obj, prop, expected) in enumerate(point["layout"]):
        _require_value(reader(i, obj, prop), expected, f"{obj or 'FDE analysis'} {prop} "
                       f"at gap {point['gap_m']:.17g} m")


def _project_point(mode, job):
    point = _point(job, mode.getnamed(GROUP, "gap"))
    mode.select("::model::FDE")
    _check_layout(point, lambda i, obj, prop: (
        mode.getanalysis(prop) if obj is None else mode.getnamed(obj, prop)
    ))
    return point


def _read_node_export(mode, path, jobs, count):
    mode.putv(EXPORT_VAR, 0)  # A bad export must not reuse the prior file's workspace value.
    mode.loaddata(str(path))
    result = mode.getv(EXPORT_VAR)
    if not isinstance(result, dict):
        raise ValueError("expected a DREME workspace .ldf; run the prepared export .lsf in solved MODE")
    job = jobs[str(_scalar(result["job_id"]))]
    point = _point(job, result["gap"])
    _check_layout(point, lambda i, obj, prop: result[f"p{i}"])
    returned = int(_scalar(result["returned"]))
    if returned < count:
        raise ValueError(f"gap {point['gap_m']} m: requested {count} modes, export contains {returned}")
    modes = tuple(ModeFields(result[f"mode{i}_E"], result[f"mode{i}_H"],
                             result[f"mode{i}_neff"], result[f"mode{i}_TEpol"])
                  for i in range(1, count + 1))
    return (point["gap_m"],), CoupledFields(returned, modes, str(path))


def _result_gaps(result, job, name):
    if "gap" not in result:
        raise ValueError(f"{name} is missing the explicit gap result coordinate")
    gap_vector = np.asarray(result["gap"], dtype=float).squeeze()
    if gap_vector.ndim > 1:
        raise ValueError(f"{name} gap coordinate must be a vector, got {gap_vector.shape}")
    values = [_point(job, value)["gap_m"] for value in np.atleast_1d(gap_vector)]
    expected = [point["gap_m"] for point in job["points"]]
    if len(values) != len(set(values)):
        raise ValueError(f"{name} returned duplicate gaps: {values}")
    if set(values) != set(expected):
        raise ValueError(f"{name} missing gaps: {sorted(set(expected) - set(values))}")
    return values


def _coord(values, axis, n_gaps, index, *, shared=False):
    vector = np.asarray(values).squeeze()
    if axis == "x":
        if vector.ndim == 0 and (shared or n_gaps == 1):
            return vector.item()
        if vector.shape == (n_gaps,):
            return vector[index]
    else:
        if vector.ndim == 1 and (shared or n_gaps == 1):
            return vector
        if vector.ndim == 2 and vector.shape[-1] == n_gaps:
            return vector[:, index]
    raise ValueError(f"{axis} grid {vector.shape} does not identify coordinates for {n_gaps} gaps")


def _read_sweep(mode, path, job, count):
    def fetch(key):
        result = mode.getsweepresult(SWEEP, key)
        if not isinstance(result, dict):
            raise ValueError(f"missing sweep dataset {key}")
        return result, _result_gaps(result, job, key)

    try:
        if not mode.havesweepresult(SWEEP, "mode1_E"):
            mode.loadsweep(SWEEP)  # Restore previously saved results only.
        modes_at = {point["gap_m"]: [] for point in job["points"]}
        for i in range(1, count + 1):
            datasets = {kind: fetch(f"mode{i}_{kind}") for kind in ("E", "H", "neff", "TEpol", "x", "y", "z")}
            n = len(modes_at)
            mode_fields = {}
            for name in ("E", "H"):
                result, order = datasets[name]
                field = np.asarray(result[name]).squeeze()
                # MODE's rectilinear vector result is spatial axes, parameters, components.
                # Remove only singleton axes, preserving the required y,z,gap,component order.
                if n == 1 and field.ndim == 3:
                    field = field[:, :, None, :]
                if field.ndim != 4 or field.shape[2:] != (n, 3):
                    raise ValueError(f"mode{i}_{name} shape {field.shape}; expected (Ny, Nz, {n}, 3)")
                mode_fields[name] = {}
                for j, gap in enumerate(order):
                    coords = {}
                    for axis in "xyz":
                        axis_result, axis_order = datasets[axis]
                        per_gap = _coord(axis_result[axis], axis, n, axis_order.index(gap))
                        embedded = _coord(result[axis], axis, n, j, shared=True)
                        _require_value(embedded, per_gap, f"mode{i}_{name} {axis} export grid at gap {gap}")
                        coords[axis] = per_gap
                    if field.shape[:2] != (len(coords["y"]), len(coords["z"])):
                        raise ValueError(f"mode{i}_{name} at gap {gap} does not fit y/z coordinates")
                    mode_fields[name][gap] = {**coords, name: field[:, :, j, :]}
            scalars = {}
            for name, attr in (("neff", "neff"), ("TEpol", "TE polarization fraction")):
                result, order = datasets[name]
                values = np.atleast_1d(np.asarray(result[attr]).squeeze())
                if values.shape != (n,):
                    raise ValueError(f"mode{i}_{name} shape {values.shape}; expected one scalar per gap")
                scalars[name] = dict(zip(order, values))
            for gap in modes_at:
                modes_at[gap].append(ModeFields(
                    mode_fields["E"][gap], mode_fields["H"][gap],
                    scalars["neff"][gap], scalars["TEpol"][gap],
                ))
        # The MODE number of trial modes is the maximum returned count.
        # All requested cards have to be present and finite at every gap.
        return [((gap,), CoupledFields(count, tuple(modes), f"{path}::{SWEEP}"))
                for gap, modes in modes_at.items()]
    except Exception as exc:
        raise ValueError(f"unsupported or incomplete aggregate Sweep data: {exc}. {RECOVERY}") from exc


def read_external_fields(geometry, results, *, mode_api):
    """Read only saved results and verify their job, layout, and gap coverage."""
    configuration, sections = geometry_sections(geometry)
    expected_settings = geometry.data.coupled_settings.configuration()
    count = geometry.data.mode_numbers
    plans, fields = {}, {}
    results = list(results)
    if not results:
        raise ValueError("Provide at least one MODE result file")
    reader = mode_api.MODE(hide=True)
    try:
        for source in results:
            path, plan = Path(source.path).resolve(), Path(source.plan).resolve()
            try:
                if not path.is_file():
                    raise FileNotFoundError(path)
                if plan not in plans:
                    record = json.loads(plan.read_text(encoding="utf-8"))
                    if record["version"] != 1:
                        raise ValueError(f"unsupported preparation plan version in {plan}")
                    saved_settings = record["settings"].copy()
                    saved_grid = saved_settings.pop("gap_grid")
                    physics = {key: value for key, value in expected_settings.items() if key != "gap_grid"}
                    for key, saved, expected in (
                        ("settings", saved_settings, physics), ("section_model", record["section_model"], configuration),
                    ):
                        if saved != expected:
                            changed = sorted(k for k in set(saved) | set(expected)
                                             if saved.get(k) != expected.get(k))
                            raise ValueError(f"{plan}: {key} mismatch in {changed}")
                    for job in record["jobs"]:
                        for point in job["points"]:
                            _match_gap(point["gap_m"], saved_grid)
                    jobs = {job["id"]: job for job in record["jobs"]}
                    if len(jobs) != len(record["jobs"]):
                        raise ValueError(f"{plan}: duplicate prepared job IDs")
                    plans[plan] = jobs
                jobs = plans[plan]

                if source.kind == "node" and path.suffix.lower() == ".ldf":
                    incoming = [_read_node_export(reader, path, jobs, count)]
                elif source.kind in ("node", "sweep") and path.suffix.lower() == ".lms":
                    reader.load(str(path))
                    job = jobs[str(_scalar(reader.getnamed(GROUP, "dreme_job_id")))]
                    point = _project_point(reader, job)
                    if source.kind == "sweep":
                        if job["workflow"] != "sweep":
                            raise ValueError("this project was not prepared as a Sweep parent")
                        incoming = _read_sweep(reader, path, job, count)
                    else:
                        try:
                            raw = read_mode_cards(reader, reader.nummodes(), count, str(path))
                        except Exception as exc:
                            raise ValueError(f"gap {point['gap_m']} m has no complete saved mode cards: {exc}. "
                                             "Run its prepared export .lsf in solved MODE and import the .ldf.") from exc
                        incoming = [((point["gap_m"],), raw)]
                else:
                    raise ValueError("expected a node .ldf/.lms or a Sweep parent .lms")

                for node, raw in incoming:
                    node = (_match_gap(node[0], expected_settings["gap_grid"]),)
                    if node in fields:
                        raise ValueError(f"duplicate gap {node}: {fields[node].source} and {raw.source}")
                    fields[node] = raw
            except Exception as exc:
                raise ValueError(f"Cannot import {path} using {plan}: {exc}") from exc
    finally:
        reader.close()
    missing = sorted(sections.keys() - fields.keys())
    if missing:
        raise ValueError(f"Missing coupled results for gap nodes {missing}")
    return fields
