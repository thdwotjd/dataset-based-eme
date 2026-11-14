"""Coupled geometry with a fixed axis, fixed widths, and a variable gap."""

import numpy as np
from matplotlib import pyplot as plt

from ..parameter_path import build_parameter_path, validate_parameter_grid
from .cross_section import StraightGapSectionModel, plot_cross_section
from .double_waveguide import DoubleWaveguide


class StraightGapCoupler(DoubleWaveguide):
    """Two fixed-width guides on a fixed +x propagation axis.

    gap_profile is the edge-to-edge gap in meters, either a constant or a
    callable of longitudinal position s in meters. A sampled profile can
    supply length. The dataset lookup axis is gap, and the same sampled
    path drives both propagation and drawing.
    """

    def __init__(
        self, dataset, section_model, gap_profile, length=None,
        limit_mode_number=0, resolution=1000, verbose=True,
    ):
        if tuple(dataset.parameter_names) != ("gap",) or set(dataset.parameter_grid) != {"gap"}:
            raise ValueError("StraightGapCoupler requires a dataset with exactly the 'gap' axis")
        if not isinstance(section_model, StraightGapSectionModel):
            raise TypeError("section_model must be a StraightGapSectionModel")
        super().__init__(dataset, section_model, limit_mode_number=limit_mode_number, verbose=verbose)
        for field, expected in (("crosssection_x", "y"), ("crosssection_y", "z")):
            actual = getattr(dataset, field, expected)
            if actual != expected:
                raise ValueError(f"StraightGapCoupler requires dataset {field}={expected!r}, got {actual!r}")

        grid = validate_parameter_grid("gap", self.parameter_grid["gap"])
        for gap in grid:
            self.describe_cross_section({"gap": gap})
        self._path = build_parameter_path(
            "gap", grid, gap_profile, length=length, resolution=resolution,
        )
        self._total_length = self._path.node_positions[-1]
        self._resolution = resolution

    @property
    def parameter_path(self):
        """Requested, clipped and snapped geometry plus clipping diagnostics."""
        return self._path

    def calc_total_length(self):
        return self._total_length

    def continuous_parameter_values(self, positions):
        """Return the clipped continuous gap for optional phase interpolation.

        Interface overlaps still use the same snapped node path.
        """
        positions = np.asarray(positions, dtype=float)
        tolerance = 128 * np.finfo(float).eps * self._total_length
        if np.any((positions < -tolerance) | (positions > self._total_length + tolerance)):
            raise ValueError("section position must be in the device domain")
        positions = np.clip(positions, 0.0, self._total_length)
        values = [self._path.values_at(float(s), "clipped") for s in positions.flat]
        return {"gap": np.asarray(values).reshape(positions.shape)}

    def calc_simulation_parameters(self):
        """Return the existing EME node and inter-node distance representation."""
        return list(self.parameter_path.nodes), self.parameter_path.distances

    def plot_cross_section(self, s, *, shape="snapped", ax=None):
        """Draw the section at longitudinal position s (meters)."""
        gap = self.parameter_path.values_at(s, shape)
        return plot_cross_section(self.describe_cross_section({"gap": gap}), ax)

    def plot_2D_structure(self, *, shape="snapped", ax=None):
        """Draw one of the requested, clipped or snapped plan views."""
        positions, gaps = self.parameter_path.plot_points(shape)
        regions_by_section = [
            {core.region_id: core for core in self.describe_cross_section({"gap": gap}).cores}
            for gap in gaps
        ]
        if ax is None:
            _, ax = plt.subplots()
        step = "post" if shape == "snapped" else None
        for region_id, core in regions_by_section[0].items():
            lower, upper = zip(*(regions[region_id].lateral_bounds for regions in regions_by_section))
            ax.fill_between(positions, lower, upper, step=step, alpha=0.5, label=core.guide_id)
        ax.set(xlabel="s = x (m)", ylabel="y (m)")
        ax.legend()
        return ax

    def plot_structure_parameters(self, ax=None):
        """Compare requested, clipped and snapped gap along the device."""
        if ax is None:
            _, ax = plt.subplots()
        for shape in ("requested", "clipped", "snapped"):
            positions, values = self.parameter_path.plot_points(shape)
            if shape == "snapped":
                ax.step(positions, values, where="post", label=shape, marker=".")
            else:
                ax.plot(positions, values, label=shape)
        ax.set(xlabel="s = x (m)", ylabel="gap (m)")
        ax.legend()
        return ax
