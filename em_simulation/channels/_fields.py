"""Shared coordinates and array checks for channel fields."""

from dataclasses import dataclass

import numpy as np


FIELD_CONVENTION = {
    "time": "exp(-i omega t)",
    "forward": "exp(i beta s)",
    "normal": "+x",
    "components": ("x", "y", "z"),
    "tangential_components": ("y", "z"),
    "length_unit": "m",
    "quadrature": "DREME forward coordinate differences, repeat last step",
}
GRID_ATOL_M = 1e-15  # Same coordinate comparison as canonical supermode acquisition.


def finite_array(value, name, *, shape=None, ndim=None, dtype=np.complex128):
    if dtype is float and np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


@dataclass(frozen=True)
class SectionGrid:
    """One global y/z plane in meters at a fixed wavelength.

    The caller must confirm that E and H components use the same sample
    locations. Staggered exports must be collocated before entering this API.
    """

    x: float
    y: tuple
    z: tuple
    wavelength: float
    component_locations: str

    def __post_init__(self):
        if self.component_locations != "collocated":
            raise ValueError("channel fields require verified collocated E/H component locations")
        x, wavelength = float(self.x), float(self.wavelength)
        if not np.isfinite(x) or not np.isfinite(wavelength) or wavelength <= 0:
            raise ValueError("section plane must be finite and wavelength must be positive and finite")
        object.__setattr__(self, "x", x)
        object.__setattr__(self, "wavelength", wavelength)
        for axis in ("y", "z"):
            coords = finite_array(getattr(self, axis), axis, ndim=1, dtype=float)
            differences = np.diff(coords)
            if len(coords) < 2 or not np.isfinite(differences).all() or np.any(differences <= 0):
                raise ValueError(f"{axis} requires at least two strictly increasing finite coordinates")
            object.__setattr__(self, axis, tuple(coords))

    @property
    def weights(self):
        dy, dz = np.diff(self.y), np.diff(self.z)
        weights = np.outer(np.append(dy, dy[-1]), np.append(dz, dz[-1]))
        if not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("section quadrature weights must be finite and positive")
        return weights

    def require_match(self, other):
        if self.wavelength != other.wavelength:
            raise ValueError(f"section wavelength mismatch: {self.wavelength} vs {other.wavelength} m")
        for axis in ("x", "y", "z"):
            a, b = np.asarray(getattr(self, axis)), np.asarray(getattr(other, axis))
            if a.shape != b.shape or not np.allclose(a, b, atol=GRID_ATOL_M, rtol=0):
                raise ValueError(f"section grid mismatch on {axis}; shapes {a.shape} and {b.shape}")


@dataclass(frozen=True)
class ForwardFields:
    """E and H in global component order, each (mode, 3, Ny, Nz)."""

    E: np.ndarray
    H: np.ndarray
    grid: SectionGrid

    def __post_init__(self):
        if not isinstance(self.grid, SectionGrid):
            raise TypeError("grid must be a SectionGrid")
        E = finite_array(self.E, "E", ndim=4)
        shape = (len(E), 3, len(self.grid.y), len(self.grid.z))
        if E.shape != shape:
            raise ValueError(f"E has shape {E.shape}; expected {shape}")
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "H", finite_array(self.H, "H", shape=shape))

    @property
    def count(self):
        return len(self.E)
