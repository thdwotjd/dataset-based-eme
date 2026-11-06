"""Solver-independent section geometry and a straight gap section model."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def _check_material_id(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty material ID")


@dataclass(frozen=True)
class WaveguideStack:
    """Materials and vertical dimensions, in meters, shared by both guides.

    The device layer runs from the specified bottom up to bottom + thickness.
    An optional unpatterned slab occupies its lower part. A finite lower
    cladding (BOX) requires a substrate material below it.
    """

    thickness: float
    core_material_id: str
    upper_cladding_material_id: str
    lower_cladding_material_id: str
    device_layer_bottom_z: float = 0.0
    slab_thickness: float = 0.0
    box_thickness: Optional[float] = None
    substrate_material_id: Optional[str] = None

    def __post_init__(self):
        for name in ("core_material_id", "upper_cladding_material_id", "lower_cladding_material_id"):
            _check_material_id(getattr(self, name), name)
        for name in ("thickness", "device_layer_bottom_z", "slab_thickness"):
            value = float(getattr(self, name))
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
            object.__setattr__(self, name, value)
        if not 0 <= self.slab_thickness < self.thickness:
            raise ValueError("slab_thickness must be nonnegative and smaller than thickness")
        if (self.box_thickness is None) != (self.substrate_material_id is None):
            raise ValueError("box_thickness and substrate_material_id must be specified together")
        if self.box_thickness is not None:
            box = float(self.box_thickness)
            bottom = self.device_layer_bottom_z
            if not np.isfinite(bottom - box) or not bottom - box < bottom:
                raise ValueError("box_thickness must define a finite, positive lower cladding layer")
            _check_material_id(self.substrate_material_id, "substrate_material_id")
            object.__setattr__(self, "box_thickness", box)
        bottom = self.device_layer_bottom_z
        slab_top = bottom + self.slab_thickness
        layer_top = bottom + self.thickness
        if not np.isfinite(layer_top) or not slab_top < layer_top:
            raise ValueError("stack must define a finite, positive patterned core height")
        if self.slab_thickness and not bottom < slab_top:
            raise ValueError("slab must have a representable positive height")


@dataclass(frozen=True)
class CoreRegion:
    region_id: str
    guide_id: str
    material_id: str
    vertices: tuple

    @property
    def lateral_bounds(self):
        y_values, _ = zip(*self.vertices)
        return min(y_values), max(y_values)


@dataclass(frozen=True)
class WaveguideCrossSection:
    """Straight gap section with extra gap information for shape-specific callers."""

    gap: float
    stack: WaveguideStack
    cores: tuple


def _rectangle(y_min, y_max, z_min, z_max):
    """Four counterclockwise (y, z) vertices with implicit closure."""
    if not np.isfinite((y_min, y_max, z_min, z_max)).all() or not (
        y_min < y_max and z_min < z_max
    ):
        raise ValueError("core polygon must be a finite rectangle of positive area")
    return ((y_min, z_min), (y_max, z_min), (y_max, z_max), (y_min, z_max))


def _configuration_differences(saved, supplied, path=""):
    if isinstance(saved, Mapping) and isinstance(supplied, Mapping):
        for key in sorted(saved.keys() | supplied.keys(), key=str):
            field = f"{path}.{key}" if path else str(key)
            if key not in saved:
                yield f"{field}: stored=<missing>, supplied={supplied[key]!r}"
            elif key not in supplied:
                yield f"{field}: stored={saved[key]!r}, supplied=<missing>"
            else:
                yield from _configuration_differences(saved[key], supplied[key], field)
    elif saved != supplied:
        yield f"{path}: stored={saved!r}, supplied={supplied!r}"


@dataclass(frozen=True)
class StraightGapSectionModel:
    """Two constant-width rectangular cores perpendicular to global +x.

    All lengths are in meters. wg1 stays at wg1_center_y and wg0 moves
    along +y from wg1 to achieve the requested edge-to-edge gap.
    """

    wg0_width: float
    wg1_width: float
    stack: WaveguideStack
    wg1_center_y: float = 0.0

    def __post_init__(self):
        for name in ("wg0_width", "wg1_width", "wg1_center_y"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or (name != "wg1_center_y" and value <= 0):
                raise ValueError(f"invalid {name}: {value}")
            object.__setattr__(self, name, value)
        if not isinstance(self.stack, WaveguideStack):
            raise TypeError("stack must be a WaveguideStack")
        self.describe_cross_section({"gap": 0})

    def configuration(self):
        """Export the JSON-compatible model definition for dataset metadata."""
        return {
            "model_id": "straight_gap",
            "model_version": 1,
            "parameters": {"gap": {"unit": "m"}},
            "coordinates": {
                "unit": "m", "propagation": "+x", "lateral": "y", "vertical": "z",
                "field_component_order": ["x", "y", "z"],
            },
            "guides": {
                "wg0": {
                    "role": "bus", "region_id": "wg0_core", "width": self.wg0_width,
                    "placement": {
                        "kind": "edge_gap", "reference_guide_id": "wg1",
                        "direction": "+y", "parameter": "gap",
                    },
                },
                "wg1": {
                    "role": "ring", "region_id": "wg1_core", "width": self.wg1_width,
                    "center_y": self.wg1_center_y, "placement": {"kind": "fixed"},
                },
            },
            "stack": asdict(self.stack),
        }

    def validate_configuration(self, stored):
        """Compare current metadata or the equivalent legacy double_waveguide v1."""
        expected = self.configuration()
        if (isinstance(stored, Mapping) and stored.get("model_id") == "double_waveguide"
                and stored.get("model_version") == 1):
            # This version used the same placement but did not record it explicitly.
            expected["model_id"] = "double_waveguide"
            for guide in expected["guides"].values():
                del guide["placement"]
        differences = list(_configuration_differences(stored, expected))
        if differences:
            raise ValueError("Section model configuration mismatch: " + "; ".join(differences))

    def describe_cross_section(self, parameters):
        if not isinstance(parameters, Mapping) or set(parameters) != {"gap"}:
            raise ValueError("cross section requires exactly the 'gap' parameter")
        gap = float(parameters["gap"])
        if not np.isfinite(gap) or gap < 0:
            raise ValueError("gap must be finite and nonnegative")
        wg1_min = self.wg1_center_y - self.wg1_width / 2
        wg1_max = self.wg1_center_y + self.wg1_width / 2
        wg0_min = wg1_max + gap
        wg0_max = wg0_min + self.wg0_width
        if gap > 0 and wg0_min == wg1_max:
            raise ValueError("gap is too small to represent at the specified wg1 position")
        bottom = self.stack.device_layer_bottom_z
        z_min = bottom + self.stack.slab_thickness
        z_max = bottom + self.stack.thickness
        material = self.stack.core_material_id
        return WaveguideCrossSection(gap, self.stack, (
            CoreRegion("wg0_core", "wg0", material, _rectangle(wg0_min, wg0_max, z_min, z_max)),
            CoreRegion("wg1_core", "wg1", material, _rectangle(wg1_min, wg1_max, z_min, z_max)),
        ))


def plot_cross_section(section, ax=None):
    """Draw core polygons and, if present, the slab between plot bounds."""
    if ax is None:
        _, ax = plt.subplots()
    for core in section.cores:
        ax.add_patch(Polygon(core.vertices, closed=True, alpha=0.5, label=core.guide_id))
    ax.autoscale_view()
    if section.stack.slab_thickness:
        bottom = section.stack.device_layer_bottom_z
        ax.axhspan(bottom, bottom + section.stack.slab_thickness, alpha=0.2, label="slab")
    ax.set(xlabel="y (m)", ylabel="z (m)", aspect="equal")
    ax.legend()
    return ax
