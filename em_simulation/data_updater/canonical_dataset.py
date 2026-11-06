"""Canonical coupled acquisition and the offline dataset manifest (schema 1)."""

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
import pickle
from uuid import uuid4

import numpy as np

from ..geometry.parameter_path import validate_parameter_grid
from .lumerical_section import FDEConfig, LumericalSectionAdapter, validate_section
from .overlap_calculation_tool import correct_gain_modified, normalize_field


MANIFEST = "canonical_supermodes.json"
SCHEMA_VERSION = 1
GRID_ATOL_M = 1e-15
FIELD_CONVENTION = {
    "axes": ["y", "z"],
    "components": ["x", "y", "z"],
    "normal": "+x",
    "unit": "m",
    "field_shape": ["mode", "component", "y", "z"],
    "component_locations": "MODE E/H result dataset coordinates (shared rectilinear export)",
    "quadrature": "DREME forward coordinate differences, repeat last step",
    "normalization": "E,H divided by principal sqrt(0.5 integral (E cross H)_x); unconjugated",
    "backward": "E_backward = conj(E_forward), H_backward = -conj(H_forward)",
    "grid_atol_m": GRID_ATOL_M,
}


@dataclass(frozen=True)
class CoupledDatasetSettings:
    """Fixed gap grid and FDE settings. Also supplies DataUpdater dataset info."""

    gap_grid: tuple
    cladding_index: float
    fde: FDEConfig

    def __post_init__(self):
        grid = validate_parameter_grid("gap", self.gap_grid)
        if grid[0] < 0:
            raise ValueError("gap grid must be nonnegative")
        object.__setattr__(self, "gap_grid", tuple(map(float, grid)))
        index = float(self.cladding_index)
        if not np.isfinite(index) or index <= 0:
            raise ValueError("cladding_index must be positive and finite")
        object.__setattr__(self, "cladding_index", index)
        if not isinstance(self.fde, FDEConfig):
            raise TypeError("fde must be an FDEConfig")

    def configuration(self):
        return json.loads(json.dumps(asdict(self), allow_nan=False))

    @classmethod
    def from_configuration(cls, config):
        return cls(config["gap_grid"], config["cladding_index"], FDEConfig(**config["fde"]))

    def get_parameter_names(self):
        return ["gap"]

    def get_parameter_types(self):
        return {"gap": "Length"}

    def get_parameter_grid(self):
        return {"gap": np.asarray(self.gap_grid)}

    def get_mode_numbers(self):
        return self.fde.candidate_counts["coupled"]

    def get_crosssection_x(self):
        return "y"

    def get_crosssection_y(self):
        return "z"

    def get_wavelength(self):
        return self.fde.wavelength

    def get_cladding_index(self):
        return self.cladding_index


@dataclass(frozen=True)
class ExportGrid:
    x: float
    y: tuple
    z: tuple

    def __post_init__(self):
        x = float(self.x)
        if not np.isfinite(x) or abs(x) > GRID_ATOL_M:
            raise ValueError(f"export grid x={x!r} does not match the FDE plane at x=0")
        object.__setattr__(self, "x", x)
        for axis in ("y", "z"):
            coords = validate_parameter_grid(f"export {axis}", getattr(self, axis))
            if len(coords) < 2:
                raise ValueError(f"export grid {axis} must contain at least two coordinates")
            object.__setattr__(self, axis, tuple(map(float, coords)))

    @classmethod
    def from_result(cls, result):
        return cls(
            np.asarray(result["x"]).item(), np.asarray(result["y"]).squeeze(),
            np.asarray(result["z"]).squeeze(),
        )

    def check(self, other, context):
        for axis in ("x", "y", "z"):
            a, b = np.asarray(getattr(self, axis)), np.asarray(getattr(other, axis))
            if a.shape != b.shape or not np.allclose(a, b, atol=GRID_ATOL_M, rtol=0):
                raise ValueError(f"export grid mismatch on {axis} at {context}: "
                                 f"reference shape {a.shape}, actual shape {b.shape}")


@dataclass(frozen=True)
class CanonicalNode:
    """Transient normalized forward E/H for the exact basis used by saved edges."""

    node: tuple
    generation_id: str
    basis_id: str
    mode_numbers: tuple
    requested_count: int
    returned_count: int
    raw_neff: np.ndarray
    neff: np.ndarray
    te_fraction: np.ndarray
    E: np.ndarray
    H: np.ndarray
    grid: ExportGrid

    def metadata(self):
        return {
            "node": self.node,
            "basis_id": self.basis_id,
            "fde_mode_numbers": list(self.mode_numbers),
            "requested_count": self.requested_count,
            "returned_count": self.returned_count,
            "raw_neff_re_im": np.column_stack((self.raw_neff.real, self.raw_neff.imag)).tolist(),
            "neff_re_im": np.column_stack((self.neff.real, self.neff.imag)).tolist(),
            "te_fraction": self.te_fraction.tolist(),
        }


@dataclass(frozen=True)
class ModeFields:
    """One solve's raw E/H rectilinear results and scalars for one FDE card."""

    E: dict
    H: dict
    neff: complex
    te_fraction: float


@dataclass(frozen=True)
class CoupledFields:
    returned_count: int
    modes: tuple
    source: str = ""


def read_mode_neff(mode, number):
    """Read and validate the scalar needed to filter an isolated candidate."""
    card = f"FDE::data::mode{number}"
    try:
        neff = complex(np.asarray(mode.getdata(card, "neff")).item())
        if not np.isfinite(neff):
            raise ValueError("neff is non-finite")
        return neff
    except Exception as exc:
        raise ValueError(f"{card} neff export failed: {exc}") from exc


def read_mode_card(mode, number, *, neff=None):
    """Read one existing FDE card, reusing an effective index already read from it."""
    card = f"FDE::data::mode{number}"
    try:
        return ModeFields(
            mode.getresult(card, "E"), mode.getresult(card, "H"),
            mode.getdata(card, "neff") if neff is None else neff,
            mode.getdata(card, "TE polarization fraction"),
        )
    except Exception as exc:
        raise ValueError(f"{card} export failed: {exc}") from exc


def read_mode_cards(mode, returned, count, source=""):
    """Read existing FDE cards. The caller decides how the solve was obtained."""
    returned = int(np.asarray(returned).item())
    if returned < count:
        raise ValueError(f"requested {count} coupled modes, MODE returned {returned}")
    fields = tuple(read_mode_card(mode, number) for number in range(1, count + 1))
    return CoupledFields(returned, fields, source)


def _canonicalize(raw, node, generation, basis_id, count):
    if raw.returned_count < count or len(raw.modes) != count:
        raise ValueError(f"requested {count} coupled modes, MODE returned {raw.returned_count}, "
                         f"with {len(raw.modes)} complete mode records")
    mode_numbers = tuple(range(1, count + 1))
    grid, E, H, raw_neff, te_fraction = unpack_mode_fields(raw.modes, mode_numbers)
    neff = correct_gain_modified(raw_neff)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        E, H = normalize_field(E[None], H[None], np.asarray(grid.y), np.asarray(grid.z), 0)
    E, H = E[0], H[0]
    for i, (em, hm) in enumerate(zip(E, H), start=1):
        if not np.isfinite(em).all() or not np.isfinite(hm).all():
            raise ValueError(f"mode{i} has invalid DREME normalization (zero or non-finite integral)")
    for array in (E, H, raw_neff, neff, te_fraction):
        array.setflags(write=False)
    return CanonicalNode(node, generation, basis_id, mode_numbers, count, raw.returned_count,
                         raw_neff, neff, te_fraction, E, H, grid)


def unpack_mode_fields(modes, mode_numbers):
    """Validate one nonempty MODE export and unpack its unnormalized fields."""
    raw_neff, te_fraction, electric, magnetic = [], [], [], []
    grid = None
    for i, mode in zip(mode_numbers, modes):
        fields = []
        try:
            for name, result in (("E", mode.E), ("H", mode.H)):
                result_grid = ExportGrid.from_result(result)
                if grid is None:
                    grid = result_grid
                grid.check(result_grid, f"FDE::data::mode{i} {name}")
                values = np.asarray(result[name], dtype=np.complex128).squeeze()
                expected = (len(grid.y), len(grid.z), 3)
                if values.shape != expected or not np.isfinite(values).all():
                    raise ValueError(f"{name} expected finite shape {expected}, got {values.shape}")
                fields.append(np.moveaxis(values, -1, 0))
            raw_neff.append(complex(np.asarray(mode.neff).item()))
            te_fraction.append(float(np.asarray(mode.te_fraction).item()))
        except Exception as exc:
            raise ValueError(f"FDE::data::mode{i} export failed: {exc}") from exc
        electric.append(fields[0])
        magnetic.append(fields[1])

    raw_neff = np.asarray(raw_neff, dtype=np.complex128)
    te_fraction = np.asarray(te_fraction, dtype=float)
    if not np.isfinite(raw_neff).all() or not np.isfinite(te_fraction).all():
        raise ValueError("neff or TE polarization fraction is non-finite")
    if np.any((te_fraction < 0) | (te_fraction > 1)):
        raise ValueError("TE polarization fraction must be in [0, 1]")
    return grid, np.asarray(electric), np.asarray(magnetic), raw_neff, te_fraction


def _atomic_write(path, data):
    # New generation directories have a single writer. Publish the manifest last.
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_bytes(data)
    temp.replace(path)


def _write_manifest(path, record):
    _atomic_write(path, (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8"))


def _files(dataset):
    return {
        "neff": Path(dataset.neff_filename),
        "TE_pol": Path(dataset.TE_pol_filename),
        "overlap": Path(dataset.overlap_filename),
    }


def _expected_edges(nodes):
    return {(a, b) for left, right in zip(nodes, nodes[1:])
            for a, b in ((left, right), (right, left))}


def geometry_sections(geometry):
    """Resolve and validate every configured gap using the shared section model."""
    settings = geometry.data.coupled_settings
    configuration = geometry.section_model.configuration()
    coordinates = configuration["coordinates"]
    if (coordinates["propagation"], coordinates["lateral"], coordinates["vertical"]) != ("+x", "y", "z"):
        raise ValueError("canonical FDE generation requires +x propagation and y/z section coordinates")
    sections = {}
    for gap in settings.gap_grid:
        node = (gap,)
        try:
            section = geometry.describe_cross_section({"gap": node[0]})
            validate_section(section, settings.fde)
        except Exception as exc:
            raise ValueError(f"coupled node {node} preflight failed: {exc}") from exc
        sections[node] = section
    return configuration, sections


def acquire(dataset, geometry, open_mode, on_node):
    """Supply one solve per gap from the DataUpdater-owned MODE session."""
    def open_reader(_sections):
        mode = open_mode()
        adapter = LumericalSectionAdapter(mode, dataset.coupled_settings.fde)

        def read_node(node, section):
            adapter.prepare(section)
            return read_mode_cards(mode, mode.findmodes(), dataset.mode_numbers)

        return read_node

    return assemble(dataset, geometry, open_reader, on_node)


def assemble(dataset, geometry, open_reader, on_node=None, *, finish_channels=None):
    """Normalize and save one fresh generation from any raw coupled field source."""
    if geometry.data is not dataset:
        raise ValueError("geometry must be bound to the DataUpdater assembling this dataset")
    settings = dataset.coupled_settings
    configuration, sections = geometry_sections(geometry)
    folder = Path(dataset.data_directory)
    folder.mkdir(parents=True, exist_ok=False)
    path = folder / MANIFEST
    generation = uuid4().hex
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "building", "generation_id": generation,
        "settings": settings.configuration(), "section_model": configuration,
        "field_convention": FIELD_CONVENTION,
    }
    _write_manifest(path, manifest)
    read_node = open_reader(sections)
    neff, te_pol, overlap = {}, {}, {}
    records, edges = [], []
    previous = reference_grid = None
    for index, (node, section) in enumerate(sections.items()):
        raw = None
        try:
            raw = read_node(node, section)
            current = _canonicalize(raw, node, generation, f"{generation}:{index}",
                                    dataset.mode_numbers)
            if reference_grid is None:
                reference_grid = current.grid
            reference_grid.check(current.grid, f"coupled node {node}")
            neff[node] = np.concatenate((current.neff, -current.neff))
            te_pol[node] = np.tile(current.te_fraction, 2)
            overlap[node] = {}
            if previous is not None:
                for a, b in ((previous, current), (current, previous)):
                    E = np.concatenate((a.E, a.E.conj()))
                    H = np.concatenate((b.H, -b.H.conj()))
                    matrix = dataset.calc_overlap(E, H, np.asarray(reference_grid.y),
                                                  np.asarray(reference_grid.z), 0)
                    if not np.isfinite(matrix).all():
                        raise ValueError(f"non-finite overlap at edge {a.node} -> {b.node}")
                    overlap[a.node][b.node] = matrix
                    edges.append({
                        "source": a.node, "destination": b.node,
                        "source_basis_id": a.basis_id, "destination_basis_id": b.basis_id,
                    })
            entry = current.metadata()
            if raw.source:
                entry["source"] = raw.source
            records.append(entry)
            if on_node is not None:
                on_node(current)
            previous = current
        except Exception as exc:
            source = f" from {raw.source}" if raw is not None and raw.source else ""
            raise RuntimeError(f"canonical coupled node {node}{source} failed: {exc}") from exc

    hashes = {}
    for key, values in (("neff", neff), ("TE_pol", te_pol), ("overlap", overlap)):
        output = _files(dataset)[key]
        payload = pickle.dumps(values, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(output, payload)
        hashes[output.name] = sha256(payload).hexdigest()
    manifest.update(status="complete", nodes=records, edges=edges, export_grid=asdict(reference_grid),
                    file_sha256=hashes)
    if finish_channels is not None:
        manifest["channel_data_sha256"] = finish_channels(manifest)
    _write_manifest(path, manifest)
    return manifest


def read_manifest(folder):
    path = Path(folder) / MANIFEST
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {record['schema_version']!r}")
        if record["status"] != "complete":
            raise ValueError(f"generation {record['generation_id']} is incomplete (status={record['status']!r})")
        if record["field_convention"] != FIELD_CONVENTION:
            raise ValueError("field convention is incompatible")
        settings = CoupledDatasetSettings.from_configuration(record["settings"])
        if settings.configuration() != record["settings"]:
            raise ValueError("FDE settings are incompatible")
        ExportGrid(**record["export_grid"])
        configuration = record["section_model"]
        coords = configuration["coordinates"]
        if (coords["propagation"], coords["lateral"], coords["vertical"]) != ("+x", "y", "z"):
            raise ValueError("section model axes are incompatible")
        return record, settings
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid canonical dataset manifest {path}: {exc}") from exc


def _require_array(values, key, shape, context):
    if not isinstance(values, dict) or key not in values:
        raise ValueError(f"Missing canonical {context}")
    array = values[key]
    if not isinstance(array, np.ndarray) or array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"Invalid canonical {context}: expected finite array {shape}")
    return array


def require_path(dataset, path):
    """Check all path nodes and both directed edges before any EME update."""
    count = 2 * dataset.mode_numbers
    for node in set(path):
        if node not in dataset.canonical_basis_ids:
            raise ValueError(f"Missing canonical basis for node {node}")
        _require_array(dataset.neff, node, (count,), f"neff at node {node}")
        _require_array(dataset.TE_pol, node, (count,), f"TE_pol at node {node}")
        if node not in dataset.overlap:
            raise ValueError(f"Missing canonical overlap record for node {node}")
    for a, b in _expected_edges(path) - {(node, node) for node in path}:
        _require_array(dataset.overlap[a], b, (count, count), f"overlap edge {a} -> {b}")


def load_arrays(dataset, record):
    """Validate IDs, expected adjacency, dimensions and file fingerprints."""
    nodes = [(gap,) for gap in dataset.coupled_settings.gap_grid]
    generation = record["generation_id"]
    count = dataset.mode_numbers
    if not isinstance(generation, str) or not generation:
        raise ValueError("canonical generation_id must be a nonempty string")
    basis_ids = {}
    if len(record["nodes"]) != len(nodes):
        raise ValueError("canonical node metadata does not cover the configured gap grid")
    for i, (node, entry) in enumerate(zip(nodes, record["nodes"])):
        basis = f"{generation}:{i}"
        if (tuple(entry["node"]) != node or entry["basis_id"] != basis
                or entry["fde_mode_numbers"] != list(range(1, count + 1))
                or entry["requested_count"] != count or entry["returned_count"] < count):
            raise ValueError(f"Invalid canonical metadata at node {node}")
        basis_ids[node] = basis
    required_edges = _expected_edges(nodes)
    seen_edges = set()
    for edge in record["edges"]:
        a, b = tuple(edge["source"]), tuple(edge["destination"])
        if ((a, b) not in required_edges or (a, b) in seen_edges
                or edge["source_basis_id"] != basis_ids[a] or edge["destination_basis_id"] != basis_ids[b]):
            raise ValueError(f"Invalid canonical basis linkage at edge {a} -> {b}")
        seen_edges.add((a, b))
    if required_edges - seen_edges:
        a, b = min(required_edges - seen_edges)
        raise ValueError(f"Missing canonical basis linkage at edge {a} -> {b}")

    loaded, actual_hashes = {}, {}
    for name, path in _files(dataset).items():
        try:
            payload = path.read_bytes()
            loaded[name] = pickle.loads(payload)
        except (OSError, pickle.PickleError, EOFError) as exc:
            raise ValueError(f"Cannot read canonical {name} file {path}: {exc}") from exc
        if not isinstance(loaded[name], dict):
            raise ValueError(f"Invalid canonical {name} dictionary in {path}")
        actual_hashes[path.name] = sha256(payload).hexdigest()
    for node, entry in zip(nodes, record["nodes"]):
        values = _require_array(loaded["neff"], node, (2 * count,), f"neff at node {node}")
        pol = _require_array(loaded["TE_pol"], node, (2 * count,), f"TE_pol at node {node}")
        for key in ("raw_neff_re_im", "neff_re_im"):
            parts = np.asarray(entry[key], dtype=float)
            if parts.shape != (count, 2) or not np.isfinite(parts).all():
                raise ValueError(f"Invalid {key} metadata at node {node}")
            if key == "raw_neff_re_im":
                expected = correct_gain_modified(parts[:, 0] + 1j * parts[:, 1])
            elif not np.array_equal(parts[:, 0] + 1j * parts[:, 1], expected):
                raise ValueError(f"Raw/corrected neff mismatch at node {node}")
        if not np.array_equal(values, np.concatenate((expected, -expected))):
            raise ValueError(f"Saved neff does not match the basis at node {node}")
        if not np.array_equal(pol, np.tile(entry["te_fraction"], 2)):
            raise ValueError(f"Saved TE_pol does not match the basis at node {node}")
        if node not in loaded["overlap"]:
            raise ValueError(f"Missing canonical overlap record for node {node}")
    for a, b in sorted(required_edges):
        _require_array(loaded["overlap"][a], b, (2 * count, 2 * count), f"overlap edge {a} -> {b}")
    if actual_hashes != record["file_sha256"]:
        changed = sorted(name for name in actual_hashes
                         if actual_hashes[name] != record["file_sha256"].get(name))
        raise ValueError(f"Canonical generation {generation} has incompatible file fingerprints: {changed}")
    return basis_ids, loaded
