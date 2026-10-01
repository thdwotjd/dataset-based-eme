"""Optional single-parameter effective-index interpolation for EME phases.

Only propagation constants are changed. Dataset modes, interface overlaps,
stability weights, and PML sink decisions keep their original values.
"""

from __future__ import annotations

import numpy as np


def interpolate_single_axis_beta(geometry):
    """Integrate tracked guided neff over each EME propagation step.

    Five physical samples per step are combined with Simpson quadrature.
    A mode step retains its original beta if any sample has no valid adjacent
    guided data pair with continuous overlap-based mode tracking.
    """
    if not hasattr(geometry, "continuous_parameter_values"):
        raise TypeError(
            "neff_interpolation requires a geometry that exposes continuous "
            "dataset parameter values along its propagation coordinate"
        )
    output = geometry.output_data
    neff = np.asarray(output["neff"], dtype=np.complex128)
    mode_count = neff.shape[1] // 2
    beta = np.asarray(output["beta"][:, :mode_count], dtype=np.complex128).copy()
    present = np.asarray(output["mode_present"][:, :mode_count], dtype=bool)
    guided = present & ~np.asarray(
        output["radiation_mode_mask"][:, :mode_count], dtype=bool
    )
    lengths = np.asarray(output["EME_delta_zs"], dtype=float)
    if lengths.shape != (len(neff) - 1,) or not np.all(np.isfinite(lengths)) or np.any(lengths <= 0):
        raise ValueError("EME propagation lengths must be positive and match the section path")
    boundaries = np.r_[0.0, np.cumsum(lengths)]
    names = tuple(geometry.parameter_names)
    if not names:
        raise ValueError("Dataset has no parameter names")

    # Inspect the entire physical profile: necks and bends can have equal
    # input and output parameters even though they vary internally.
    probe_count = max(257, min(int(getattr(geometry, "_resolution", 257)), 3001))
    probe = geometry.continuous_parameter_values(
        np.linspace(0.0, boundaries[-1], probe_count)
    )
    varying = []
    for name in names:
        if name not in probe:
            raise NotImplementedError(f"No continuous geometry value for {name}")
        values = np.asarray(probe[name], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Nonfinite physical parameter: {name}")
        grid = np.sort(np.unique(np.asarray(geometry.data.parameter_grid[name], dtype=float)))
        if not len(grid):
            raise ValueError(f"Empty dataset grid for {name}")
        tolerance = max(
            100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(grid)))),
            (float(np.min(np.diff(grid))) * 1e-6) if len(grid) > 1 else 0.0,
        )
        if np.ptp(values) > tolerance:
            varying.append(name)
    if len(varying) > 1:
        raise ValueError(
            "neff_interpolation currently supports one varying dataset "
            f"parameter; found {varying}"
        )
    diagnostics = {
        "enabled": True,
        "axis": varying[0] if varying else None,
        "guided_mode_steps": int(np.count_nonzero(guided[:-1])),
        "interpolated_mode_steps": 0,
        "tracking_conflicts": 0,
    }
    if not varying:
        return beta, diagnostics

    axis = varying[0]
    axis_index = names.index(axis)
    grid = np.sort(np.unique(np.asarray(geometry.data.parameter_grid[axis], dtype=float)))
    if len(grid) < 2 or not np.all(np.isfinite(grid)):
        raise ValueError(f"Interpolation requires at least two finite dataset points on {axis}")
    step_min = float(np.min(np.diff(grid)))
    tolerance = max(
        100 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(grid)))),
        step_min * 1e-9,
    )
    path = [tuple(point) for point in output["EME_path"]]
    path_axis = np.array([float(point[axis_index]) for point in path])
    if len(path) != len(neff):
        raise ValueError("EME path and neff section counts differ")

    fractions = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    sample_s = boundaries[:-1, None] + lengths[:, None] * fractions
    sample_values = np.asarray(
        geometry.continuous_parameter_values(sample_s.ravel())[axis], dtype=float
    ).reshape(sample_s.shape)
    if (not np.all(np.isfinite(sample_values))
            or sample_values.min() < grid[0] - tolerance
            or sample_values.max() > grid[-1] + tolerance):
        raise ValueError(f"Physical {axis} lies outside the dataset grid")
    sample_values = np.clip(sample_values, grid[0], grid[-1])

    # Repeated visits to a point must give the same tracked neff. Ambiguous
    # mode/grid combinations are unavailable for interpolation.
    tracked = np.full((len(grid), mode_count), np.nan + 1j * np.nan)
    conflicts = np.zeros((len(grid), mode_count), dtype=bool)
    grid_index = {float(value): i for i, value in enumerate(grid)}
    for row, value in enumerate(path_axis):
        index = grid_index[float(value)]
        for mode in np.flatnonzero(guided[row]):
            candidate = neff[row, mode]
            if conflicts[index, mode]:
                continue
            previous = tracked[index, mode]
            if np.isfinite(previous):
                if abs(previous - candidate) > 1e-5:
                    tracked[index, mode] = np.nan + 1j * np.nan
                    conflicts[index, mode] = True
            else:
                tracked[index, mode] = candidate
    diagnostics["tracking_conflicts"] = int(np.count_nonzero(conflicts))

    # A rounded EME path may miss the dataset neighbor just beyond a physical
    # maximum. Link it to the nearest visited section with the stored overlap.
    needed_low = max(0, np.searchsorted(grid, sample_values.min(), side="right") - 1)
    needed_high = min(len(grid) - 1, np.searchsorted(grid, sample_values.max(), side="left"))
    visited = set(path_axis)
    for index in range(needed_low, needed_high + 1):
        value = float(grid[index])
        if value in visited:
            continue
        anchor_row = int(np.argmin(np.abs(path_axis - value)))
        anchor_point = path[anchor_row]
        anchor_index = grid_index[float(path_axis[anchor_row])]
        if abs(index - anchor_index) != 1:
            continue
        neighbor = list(anchor_point)
        neighbor[axis_index] = value
        neighbor = tuple(neighbor)
        if neighbor not in geometry.data.neff:
            continue
        try:
            overlap_ab, _ = geometry.data.get_overlap(anchor_point, neighbor)
        except KeyError:
            continue
        raw_count = geometry._tracking_mode_names.shape[1]
        overlaps = np.abs(np.asarray(overlap_ab)[:raw_count, :raw_count])
        raw_neff = np.asarray(geometry.data.neff[neighbor])[:raw_count]
        raw_guided = ~geometry._get_radiation_mode_mask(raw_neff[None, :])[0]
        assignments = {}
        for raw_neighbor in np.flatnonzero(raw_guided):
            raw_anchor = int(np.argmax(overlaps[:, raw_neighbor]))
            if overlaps[raw_anchor, raw_neighbor] <= 0.5:
                continue
            track = int(geometry._tracking_mode_names[anchor_row, raw_anchor])
            assignments.setdefault(track, []).append(raw_neighbor)
        for track, choices in assignments.items():
            if len(choices) == 1 and not conflicts[index, track]:
                tracked[index, track] = raw_neff[choices[0]]

    upper = np.searchsorted(grid, sample_values, side="left")
    upper = np.clip(upper, 0, len(grid) - 1)
    lower = np.maximum(upper - 1, 0)
    exact = np.abs(sample_values - grid[upper]) <= tolerance
    lower[exact] = upper[exact]
    span = grid[upper] - grid[lower]
    weight = np.zeros_like(sample_values)
    np.divide(sample_values - grid[lower], span, out=weight, where=span > 0)
    weight = np.clip(weight, 0.0, 1.0)

    k0 = 2 * np.pi / geometry.wavelength
    for section in range(len(lengths)):
        for mode in np.flatnonzero(guided[section]):
            left = tracked[lower[section], mode]
            right = tracked[upper[section], mode]
            if not (np.all(np.isfinite(left)) and np.all(np.isfinite(right))):
                continue
            local_neff = (1 - weight[section]) * left + weight[section] * right
            mean_neff = (
                local_neff[0] + 4 * local_neff[1] + 2 * local_neff[2]
                + 4 * local_neff[3] + local_neff[4]
            ) / 12
            beta[section, mode] = k0 * mean_neff
            diagnostics["interpolated_mode_steps"] += 1
    return beta, diagnostics