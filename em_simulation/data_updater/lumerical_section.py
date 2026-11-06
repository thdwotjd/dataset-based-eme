"""Apply shared cross-section polygons to a borrowed MODE session."""

from contextlib import contextmanager
from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class FDEConfig:
    """A fixed +x-normal FDE plane at x=0, with dimensions in meters."""

    wavelength: float
    y_bounds: tuple
    z_bounds: tuple
    mesh_cells_yz: tuple
    material_bindings: dict
    candidate_counts: dict
    boundary: str

    def __post_init__(self):
        wavelength = float(self.wavelength)
        if not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("FDE wavelength must be positive and finite")
        object.__setattr__(self, "wavelength", wavelength)
        for name in ("y_bounds", "z_bounds"):
            bounds = tuple(map(float, getattr(self, name)))
            if len(bounds) != 2 or not np.isfinite(bounds).all() or not bounds[0] < bounds[1]:
                raise ValueError(f"FDE {name} must contain two increasing finite coordinates")
            object.__setattr__(self, name, bounds)
        cells = tuple(self.mesh_cells_yz)
        if len(cells) != 2 or any(type(n) is bool or not isinstance(n, Integral) or n < 2 for n in cells):
            raise ValueError("mesh_cells_yz must contain two integer counts of at least 2")
        object.__setattr__(self, "mesh_cells_yz", tuple(map(int, cells)))
        materials = dict(self.material_bindings)
        if not materials or any(
            not isinstance(key, str) or not key.strip()
            or not isinstance(value, str) or not value.strip()
            or value == "<Object defined dielectric>"
            for key, value in materials.items()
        ):
            raise ValueError("material_bindings must map material IDs to named MODE library materials")
        object.__setattr__(self, "material_bindings", materials)
        counts = dict(self.candidate_counts)
        if "coupled" not in counts or any(
            not isinstance(key, str) or not key
            or type(n) is bool or not isinstance(n, Integral) or n < 1
            for key, n in counts.items()
        ):
            raise ValueError("candidate_counts must contain positive integer budgets, including 'coupled'")
        object.__setattr__(self, "candidate_counts", {key: int(n) for key, n in counts.items()})
        if self.boundary not in ("Metal", "PML"):
            raise ValueError("FDE boundary must be 'Metal' or 'PML' on all transverse sides")


def validate_section(section, settings):
    """Check bindings and domain coverage before opening MODE or changing layout."""
    cores = section.cores
    guide_ids = {core.guide_id for core in cores}
    region_ids = {core.region_id for core in cores}
    if len(guide_ids) != 2 or "coupled" in guide_ids or len(region_ids) != len(cores):
        raise ValueError("section requires two distinct guide IDs and unique core region IDs")
    if any(not key or "::" in key for key in region_ids):
        raise ValueError("core region IDs must be nonempty MODE object names without '::'")
    if set(settings.candidate_counts) != guide_ids | {"coupled"}:
        raise ValueError(f"candidate_counts must contain 'coupled' and guide IDs {sorted(guide_ids)}")

    stack = section.stack
    materials = {core.material_id for core in cores} | {
        stack.core_material_id, stack.upper_cladding_material_id, stack.lower_cladding_material_id,
    }
    if stack.substrate_material_id is not None:
        materials.add(stack.substrate_material_id)
    missing = materials - settings.material_bindings.keys()
    if missing:
        raise ValueError(f"missing MODE material bindings for {sorted(missing)}")
    bottom = stack.device_layer_bottom_z
    layer_bottom = bottom if stack.box_thickness is None else bottom - stack.box_thickness
    if not settings.z_bounds[0] < layer_bottom <= bottom < settings.z_bounds[1]:
        raise ValueError("FDE z_bounds must enclose the device layer base and any BOX/substrate interface")
    y0, y1 = settings.y_bounds
    z0, z1 = settings.z_bounds
    for core in cores:
        yz = np.asarray(core.vertices, dtype=float)
        if (yz.ndim != 2 or yz.shape[1] != 2 or len(yz) < 3 or not np.isfinite(yz).all()
                or np.any(yz[:, 0] <= y0) or np.any(yz[:, 0] >= y1)
                or np.any(yz[:, 1] <= z0) or np.any(yz[:, 1] >= z1)):
            raise ValueError(f"core region {core.region_id!r} polygon must be inside FDE y/z bounds")


class LumericalSectionAdapter:
    """Build one layout in a new DataUpdater-owned session and reuse its objects."""

    def __init__(self, mode, config):
        self.mode = mode
        self.config = config
        self._stack = None
        self._guide_by_region = None
        self.object_properties = {}

    def _set(self, properties):
        for name, value in properties.items():
            try:
                self.mode.set(name, value)
            except Exception as exc:
                raise RuntimeError(f"MODE could not set property {name!r}: {exc}") from exc

    def _set_object(self, name, properties, *, create=False):
        if create:
            self._set({"name": name})
        self._set(properties)
        self.object_properties.setdefault(name, {}).update(properties)

    def _material(self, material_id, mesh_order):
        return {
            "material": self.config.material_bindings[material_id],
            "override mesh order from material database": 1,
            "mesh order": mesh_order,
        }

    def _create_layout(self, stack):
        y0, y1 = self.config.y_bounds
        z0, z1 = self.config.z_bounds
        self._x_span = max(y1 - y0, z1 - z0)
        # MODE creates the solver with the fixed name FDE.
        self.mode.addfde()
        self._set_object("FDE", {
            "solver type": "2D X normal", "x": 0.0,
            "y": (y0 + y1) / 2, "y span": y1 - y0,
            "z": (z0 + z1) / 2, "z span": z1 - z0,
            "mesh cells y": self.config.mesh_cells_yz[0],
            "mesh cells z": self.config.mesh_cells_yz[1],
            **{f"{axis} {side} bc": self.config.boundary for axis in "yz" for side in ("min", "max")},
        })

        def layer(name, lower, upper, material, order):
            self.mode.addrect()
            self._set_object(f"dreme_stack_{name}", {
                "x": 0.0, "x span": self._x_span,
                "y": (y0 + y1) / 2, "y span": 2 * (y1 - y0),
                "z": (lower + upper) / 2, "z span": upper - lower,
                **self._material(material, order),
            }, create=True)

        bottom = stack.device_layer_bottom_z
        slab_top = bottom + stack.slab_thickness
        lower_extent = z0 - (z1 - z0)
        upper_extent = z1 + (z1 - z0)
        if stack.box_thickness is None:
            layer("lower", lower_extent, bottom, stack.lower_cladding_material_id, 3)
        else:
            box_bottom = bottom - stack.box_thickness
            layer("substrate", lower_extent, box_bottom, stack.substrate_material_id, 3)
            layer("box", box_bottom, bottom, stack.lower_cladding_material_id, 3)
        if stack.slab_thickness:
            layer("slab", bottom, slab_top, stack.core_material_id, 1)
        layer("upper", slab_top, upper_extent, stack.upper_cladding_material_id, 3)

    def prepare(self, section, guide_id=None):
        """Apply a coupled (None) or isolated layout and explicitly set the search budget."""
        validate_section(section, self.config)
        guide_by_region = {core.region_id: core.guide_id for core in section.cores}
        if guide_id is not None and guide_id not in guide_by_region.values():
            raise ValueError(f"unknown isolated guide ID {guide_id!r}")
        first = self._stack is None
        if not first and (section.stack != self._stack or guide_by_region != self._guide_by_region):
            raise ValueError("section stack or guide/region IDs changed within the fixed FDE layout")

        self.mode.switchtolayout()
        if first:
            self._create_layout(section.stack)
        for core in section.cores:
            name = f"dreme_core_{core.region_id}"
            if first:
                self.mode.addpoly()
                self._set_object(name, {
                    "x": 0.0, "y": 0.0, "z": 0.0, "z span": self._x_span,
                    "first axis": "y", "rotation 1": 90.0,
                }, create=True)
            else:
                self.mode.select(f"::model::{name}")
            # A +90 degree rotation around global y sends local (u, v, w)
            # to global (w, v, -u). The MODE polygon therefore uses (-z, y).
            yz = np.asarray(core.vertices, dtype=float)
            self._set_object(name, {
                "vertices": np.column_stack((-yz[:, 1], yz[:, 0])),
                **self._material(core.material_id, 1),
                "enabled": int(guide_id is None or core.guide_id == guide_id),
            })
        self._stack = section.stack
        self._guide_by_region = guide_by_region
        self.mode.select("::model::FDE")
        self.analysis_properties = {
            "wavelength": self.config.wavelength, "search": "near n", "use max index": 1,
            "number of trial modes": self.config.candidate_counts[guide_id or "coupled"],
        }
        for name, value in self.analysis_properties.items():
            self.mode.setanalysis(name, value)

    @contextmanager
    def isolated(self, section, guide_id):
        """Temporarily isolate one guide and restore the coupled layout and budget."""
        try:
            self.prepare(section, guide_id)
            yield self.mode
        finally:
            self.prepare(section)
