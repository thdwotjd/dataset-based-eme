"""Convert a physical one-axis profile to adjacent dataset node visits."""

from dataclasses import dataclass
from numbers import Integral
import warnings

import numpy as np


def _finite_vector(values, name, minimum_size):
    array = np.asarray(values)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    array = array.astype(float)
    if array.ndim != 1 or array.size < minimum_size or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite 1D array with at least {minimum_size} values")
    return array


@dataclass(frozen=True)
class SampledProfile:
    """Piecewise-linear profile with an explicit domain, in meters."""

    positions: tuple
    values: tuple

    def __post_init__(self):
        positions = _finite_vector(self.positions, "sample positions", 2)
        values = _finite_vector(self.values, "sample values", 2)
        if positions.shape != values.shape:
            raise ValueError("sample positions and values must have the same length")
        if positions[0] != 0 or np.any(positions[1:] <= positions[:-1]):
            raise ValueError("sample positions must start at 0 and be strictly increasing")
        object.__setattr__(self, "positions", tuple(positions))
        object.__setattr__(self, "values", tuple(values))

    @property
    def length(self):
        return self.positions[-1]

    def __call__(self, s):
        s = np.asarray(s, dtype=float)
        if not np.isfinite(s).all() or np.any((s < 0) | (s > self.length)):
            raise ValueError(f"positions must be in the sample domain [0, {self.length:g}] m")
        return np.interp(s, self.positions, self.values)


def profile_from_samples(s_list, value_list):
    """Create a linear callable whose final sample can supply the device length."""
    return SampledProfile(s_list, value_list)


@dataclass(frozen=True)
class PathClipping:
    parameter_name: str
    bounds: tuple
    positions: tuple
    requested_values: tuple
    applied_values: tuple


@dataclass(frozen=True)
class ParameterPath:
    parameter_name: str
    sample_positions: tuple
    requested_values: tuple
    clipped_values: tuple
    node_positions: tuple
    nodes: tuple
    clipping: PathClipping

    @property
    def distances(self):
        return np.diff(self.node_positions)

    def plot_points(self, shape):
        """Return positions and values, including kinks where clipping starts."""
        if shape == "snapped":
            return self.node_positions, tuple(node[0] for node in self.nodes)
        if shape == "requested":
            return self.sample_positions, self.requested_values
        if shape != "clipped":
            raise ValueError("shape must be 'requested', 'clipped' or 'snapped'")

        positions = [self.sample_positions[0]]
        values = [self.clipped_values[0]]
        for i, (v0, v1) in enumerate(zip(self.requested_values[:-1], self.requested_values[1:])):
            s0, s1 = self.sample_positions[i:i + 2]
            bounds = self.clipping.bounds if v0 < v1 else self.clipping.bounds[::-1]
            for bound in dict.fromkeys(bounds):
                if min(v0, v1) < bound < max(v0, v1):
                    positions.append(_crossing_position(s0, s1, v0, v1, bound))
                    values.append(bound)
            positions.append(s1)
            values.append(self.clipped_values[i + 1])
        return positions, values

    def values_at(self, s, shape):
        """Evaluate the sampled continuous or the piecewise constant snapped shape."""
        if not np.isfinite(s) or not 0 <= s <= self.node_positions[-1]:
            raise ValueError("section position must be in the device domain")
        if shape == "snapped":
            index = np.searchsorted(self.node_positions, s, side="right") - 1
            # The first node is the input reference plane, even at a grid tie.
            return self.nodes[0 if s == 0 else index][0]
        requested = np.interp(s, self.sample_positions, self.requested_values)
        if shape == "requested":
            return requested
        if shape == "clipped":
            return np.clip(requested, *self.clipping.bounds)
        raise ValueError("shape must be 'requested', 'clipped' or 'snapped'")


def validate_parameter_grid(parameter_name, grid):
    """Return a finite, nonempty grid in dataset adjacency order."""
    grid = _finite_vector(grid, f"{parameter_name} grid", 1)
    if np.any(grid[1:] <= grid[:-1]):
        raise ValueError(f"{parameter_name} grid must be strictly increasing")
    return grid


def _profile_samples(profile, length, resolution, parameter_name):
    if not isinstance(resolution, Integral) or isinstance(resolution, bool) or resolution < 2:
        raise ValueError("resolution must be an integer of at least 2")
    sampled = isinstance(profile, SampledProfile)
    if length is None:
        if not sampled:
            raise ValueError("length is required for a callable or constant without a sample domain")
        length = profile.length
    length = float(length)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("length must be positive and finite")
    if sampled and length > profile.length:
        raise ValueError(f"length {length:g} m exceeds the sample domain {profile.length:g} m")

    if sampled:
        positions = np.array([s for s in profile.positions if s < length] + [length])
        values = profile(positions)
    elif callable(profile):
        positions = np.unique(np.linspace(0, length, resolution))
        values = [profile(float(s)) for s in positions]
    else:
        positions = np.array([0, length])
        values = [profile, profile]
    values = _finite_vector(values, f"{parameter_name} profile values", 2)
    if values.shape != positions.shape:
        raise ValueError(f"{parameter_name} profile must return one scalar per position")
    return positions, values


def _crossing_position(s0, s1, v0, v1, boundary):
    fraction = (boundary / 2 - v0 / 2) / (v1 / 2 - v0 / 2)
    return s0 + fraction * (s1 - s0)


def build_parameter_path(parameter_name, grid, profile, *, length=None, resolution=1000):
    """Snap a sampled or scalar profile to the nearest node (ties to the lower node).

    Sampled profiles use their exact knots. Other callables are sampled uniformly
    and treated as linear between samples. Split each segment at every crossed
    grid midpoint. A node supplies the geometry to its right until the next
    node, as in EME. The first and last nodes also represent the reference planes.
    A tie at a reference plane can produce a zero-length endpoint transition.
    """
    grid = validate_parameter_grid(parameter_name, grid)
    positions, values = _profile_samples(profile, length, resolution, parameter_name)
    bounds = (float(grid[0]), float(grid[-1]))
    applied = np.clip(values, *bounds)
    affected = values != applied
    clipping = PathClipping(parameter_name, bounds, tuple(positions[affected]),
                            tuple(values[affected]), tuple(applied[affected]))

    midpoints = grid[:-1] / 2 + grid[1:] / 2
    node_indices = [int(np.searchsorted(midpoints, values[0], side="left"))]
    node_positions = [0.0]

    def visit(s, index):
        if node_indices[-1] != index:
            node_indices.append(index)
            node_positions.append(float(s))

    for s0, s1, v0, v1 in zip(positions[:-1], positions[1:], values[:-1], values[1:]):
        rising = v1 > v0
        index = int(np.searchsorted(midpoints, v0, side="right" if rising else "left"))
        visit(s0, index)
        if v0 == v1:
            continue
        stop = int(np.searchsorted(midpoints, v1, side="left" if rising else "right"))
        direction = 1 if rising else -1
        for current in range(index, stop, direction):
            boundary = midpoints[current if rising else current - 1]
            # Use the original linear segment: clipping its endpoints before
            # interpolating would move crossings elsewhere along the device.
            visit(_crossing_position(s0, s1, v0, v1, boundary), current + direction)

    end_index = int(np.searchsorted(midpoints, values[-1], side="left"))
    node_indices.append(end_index)
    node_positions.append(float(positions[-1]))
    if np.any(affected):
        warnings.warn(
            f"{parameter_name} profile clipped at {np.count_nonzero(affected)} sampled "
            f"position(s) to [{bounds[0]:g}, {bounds[1]:g}]",
            UserWarning, stacklevel=2,
        )
    return ParameterPath(
        parameter_name, tuple(positions), tuple(values), tuple(applied),
        tuple(node_positions), tuple((float(grid[i]),) for i in node_indices), clipping,
    )
