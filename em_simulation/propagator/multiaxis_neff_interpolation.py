"""Multidimensional neff interpolation for EME propagation phases.

The geometry's physical parameter profile chooses interpolation cells. The
stored EME path supplies mode identities and remains the interface path.
"""

from collections import defaultdict, deque
from itertools import product
import warnings

import numpy as np
from scipy.optimize import brentq

FRACTIONS = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
SIMPSON = np.array([1.0, 4.0, 2.0, 4.0, 1.0]) / 12.0


def parameter_values(geometry, positions, names):
    positions = np.atleast_1d(np.asarray(positions, dtype=float))
    values = geometry.continuous_parameter_values(positions)
    result = {}
    for name in names:
        if name not in values:
            raise NotImplementedError(f"No continuous geometry value for {name}")
        result[name] = np.broadcast_to(
            np.asarray(values[name], dtype=float), positions.shape
        )
        if not np.all(np.isfinite(result[name])):
            raise ValueError(f"Nonfinite physical parameter: {name}")
    return result


def grid_options(value, grid, tolerance):
    """Return an exact grid index or the two bracketing indices."""
    if value < grid[0] - tolerance or value > grid[-1] + tolerance:
        return None
    value = float(np.clip(value, grid[0], grid[-1]))
    upper = min(int(np.searchsorted(grid, value, side="left")), len(grid) - 1)
    if abs(value - grid[upper]) <= tolerance:
        return (upper,)
    lower = upper - 1
    if lower >= 0 and abs(value - grid[lower]) <= tolerance:
        return (lower,)
    return (lower, upper) if lower >= 0 else None


def find_crossings(geometry, names, grids, tolerances, boundaries):
    """Find physical z positions where a dataset parameter crosses a grid line."""
    total_length = float(boundaries[-1])
    resolution = int(getattr(geometry, "_resolution", 1000))
    scan = np.unique(np.r_[
        boundaries,
        np.linspace(0.0, total_length, max(1025, min(4 * resolution, 16001))),
    ])
    profiles = parameter_values(geometry, scan, names)
    roots = list(boundaries)
    for name in names:
        grid = grids[name]
        values = profiles[name]
        tolerance = tolerances[name]
        for index in range(len(scan) - 1):
            left, right = values[index:index + 2]
            low, high = min(left, right), max(left, right)
            levels = grid[(grid >= low - tolerance) & (grid <= high + tolerance)]
            for level in levels:
                left_on = abs(left - level) <= tolerance
                right_on = abs(right - level) <= tolerance
                if left_on and not right_on:
                    roots.append(scan[index])
                if right_on and not left_on:
                    roots.append(scan[index + 1])
                if not left_on and not right_on and (left - level) * (right - level) < 0:
                    roots.append(brentq(
                        lambda z, axis=name, target=level:
                        parameter_values(geometry, [z], (axis,))[axis][0] - target,
                        float(scan[index]), float(scan[index + 1]), xtol=1e-16,
                    ))
    roots.sort()
    distinct = [roots[0]]
    z_tolerance = max(1e-16, total_length * 1e-12)
    for root in roots[1:]:
        if root - distinct[-1] > z_tolerance:
            distinct.append(root)
    return np.asarray(distinct)


def make_piece(geometry, names, grids, tolerances, start, end, section):
    nodes = start + (end - start) * FRACTIONS
    values = parameter_values(geometry, nodes, names)
    options = []
    weights = []
    for name in names:
        grid = grids[name]
        tolerance = tolerances[name]
        node_values = values[name]
        if np.any(node_values < grid[0] - tolerance) or np.any(
            node_values > grid[-1] + tolerance
        ):
            return {"section": section, "length": end - start,
                    "reason": "out_of_grid"}
        indices = grid_options(node_values[2], grid, tolerance)
        if indices is None:
            return {"section": section, "length": end - start,
                    "reason": "out_of_grid"}
        if len(indices) == 1:
            if np.any(np.abs(node_values - grid[indices[0]]) > tolerance):
                return {"section": section, "length": end - start,
                        "reason": "unresolved_boundary"}
            axis_weights = (np.ones(5),)
        else:
            low, high = grid[list(indices)]
            if np.any(node_values < low - tolerance) or np.any(
                node_values > high + tolerance
            ):
                return {"section": section, "length": end - start,
                        "reason": "unresolved_boundary"}
            fraction = np.clip((node_values - low) / (high - low), 0.0, 1.0)
            axis_weights = (1.0 - fraction, fraction)
        options.append(indices)
        weights.append(axis_weights)
    corners = []
    for choices in product(*(range(len(indices)) for indices in options)):
        point = tuple(float(grids[name][options[i][choice]])
                      for i, (name, choice) in enumerate(zip(names, choices)))
        weight = np.prod([weights[i][choice] for i, choice in enumerate(choices)], axis=0)
        corners.append((point, weight))
    return {"section": section, "length": end - start, "corners": corners}

def build_mode_graph(geometry, path, corners, names, grids, raw_count):
    """Match guided raw modes on required corners to tracked EME modes."""
    data = geometry.data
    vertices = {point for point in set(path) | set(corners) if point in data.neff}
    grid_index = {
        name: {float(value): i for i, value in enumerate(grids[name])}
        for name in names
    }
    guided = {}
    for point in vertices:
        raw_neff = np.asarray(data.neff[point])[:raw_count]
        guided[point] = ~geometry._get_radiation_mode_mask(raw_neff[None, :])[0]

    adjacency = defaultdict(list)
    missing_edges = 0
    for point in vertices:
        for axis, name in enumerate(names):
            index = grid_index[name].get(float(point[axis]))
            if index is None or index + 1 >= len(grids[name]):
                continue
            neighbor = list(point)
            neighbor[axis] = float(grids[name][index + 1])
            neighbor = tuple(neighbor)
            if neighbor not in vertices:
                continue
            if (point not in data.overlap or
                    neighbor not in data.overlap[point] or
                    neighbor not in data.overlap or
                    point not in data.overlap[neighbor]):
                missing_edges += 1
                continue
            overlap, _ = data.get_overlap(point, neighbor)
            scores = np.abs(np.asarray(overlap)[:raw_count, :raw_count])
            if scores.shape != (raw_count, raw_count):
                missing_edges += 1
                continue
            best_b = np.argmax(scores, axis=1)
            best_a = np.argmax(scores, axis=0)
            forward, reverse = {}, {}
            for raw_a, raw_b in enumerate(best_b):
                if (best_a[raw_b] == raw_a and scores[raw_a, raw_b] > 0.5
                        and guided[point][raw_a] and guided[neighbor][raw_b]):
                    forward[raw_a] = int(raw_b)
                    reverse[int(raw_b)] = raw_a
            adjacency[point].append((neighbor, forward))
            adjacency[neighbor].append((point, reverse))

    labels = defaultdict(lambda: defaultdict(set))
    queue = deque()
    tracking = np.asarray(geometry.output_data["tracking_mode_names"])
    for row, point in enumerate(path):
        if point not in vertices:
            continue
        for raw, track in enumerate(tracking[row]):
            if track < 0 or not guided[point][raw]:
                continue
            if raw not in labels[point][int(track)]:
                labels[point][int(track)].add(raw)
                queue.append((point, int(track), raw))
    while queue:
        point, track, raw = queue.popleft()
        for neighbor, mapping in adjacency[point]:
            linked = mapping.get(raw)
            if linked is not None and linked not in labels[neighbor][track]:
                labels[neighbor][track].add(linked)
                queue.append((neighbor, track, linked))
    return labels, missing_edges


def interpolate_multiaxis_beta(geometry, axes):
    """Integrate multilinear guided neff along each physical EME section."""
    output = geometry.output_data
    names = tuple(geometry.parameter_names)
    neff = np.asarray(output["neff"])
    mode_count = neff.shape[1] // 2
    beta = np.asarray(output["beta"][:, :mode_count], dtype=np.complex128).copy()
    lengths = np.asarray(output["EME_delta_zs"], dtype=float)
    boundaries = np.r_[0.0, np.cumsum(lengths)]
    present = np.asarray(output["mode_present"][:, :mode_count], dtype=bool)
    guided = present & ~np.asarray(
        output["radiation_mode_mask"][:, :mode_count], dtype=bool
    )
    path = [tuple(float(value) for value in point)
            for point in output["EME_path"]]
    tracking = np.asarray(output["tracking_mode_names"])
    if len(path) != len(neff) or tracking.shape[0] != len(path):
        raise ValueError("EME path and mode tracking section counts differ")
    grids, tolerances = {}, {}
    for name in names:
        grid = np.sort(np.unique(np.asarray(geometry.data.parameter_grid[name], dtype=float)))
        if len(grid) == 0 or not np.all(np.isfinite(grid)):
            raise ValueError(f"Invalid dataset grid for {name}")
        grids[name] = grid
        step = float(np.min(np.diff(grid))) if len(grid) > 1 else 0.0
        tolerances[name] = max(
            100 * np.finfo(float).eps * max(1.0, np.max(np.abs(grid))),
            step * 1e-9,
        )

    roots = find_crossings(geometry, names, grids, tolerances, boundaries)
    pieces = []
    required_corners = set()
    z_tolerance = max(1e-16, boundaries[-1] * 1e-12)
    for section, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        within = roots[(roots > start + z_tolerance) & (roots < end - z_tolerance)]
        edges = np.r_[start, within, end]
        previous_count = len(pieces)
        for left, right in zip(edges[:-1], edges[1:]):
            if right - left <= z_tolerance:
                continue
            piece = make_piece(geometry, names, grids, tolerances,
                               float(left), float(right), section)
            pieces.append(piece)
            required_corners.update(point for point, _ in piece.get("corners", ()))
        if len(pieces) == previous_count:
            pieces.append({"section": section, "length": float(end - start),
                           "reason": "unresolved_boundary"})

    labels, missing_edges = build_mode_graph(
        geometry, path, required_corners, names, grids, tracking.shape[1]
    )
    accumulated = np.zeros_like(beta)
    interpolated_length = np.zeros_like(beta.real)
    fallback_lengths = defaultdict(float)
    fallback_pieces = defaultdict(int)
    k0 = 2 * np.pi / geometry.wavelength
    for piece in pieces:
        section = piece["section"]
        piece_length = piece["length"]
        for mode in np.flatnonzero(guided[section]):
            reason = piece.get("reason")
            if reason is None:
                local_neff = np.zeros(5, dtype=np.complex128)
                for point, weight in piece["corners"]:
                    if point not in geometry.data.neff:
                        reason = "missing_corner"
                        break
                    matches = labels[point].get(int(mode), set())
                    if len(matches) > 1:
                        reason = "mode_tracking_conflict"
                        break
                    if not matches:
                        reason = "unmatched_mode"
                        break
                    raw = next(iter(matches))
                    value = geometry.data.neff[point][raw]
                    if not np.isfinite(value):
                        reason = "nonfinite_neff"
                        break
                    local_neff += weight * value
            if reason is None:
                accumulated[section, mode] += (
                    k0 * np.dot(SIMPSON, local_neff) * piece_length
                )
                interpolated_length[section, mode] += piece_length
            else:
                accumulated[section, mode] += beta[section, mode] * piece_length
                fallback_lengths[reason] += piece_length
                fallback_pieces[reason] += 1

    active = guided[:-1]
    beta[:-1] = np.where(active, accumulated[:-1] / lengths[:, None], beta[:-1])
    guided_length = float(np.sum(lengths[:, None] * active))
    fallback_total = sum(fallback_lengths.values())
    diagnostics = {
        "enabled": True,
        "axes": tuple(axes),
        "guided_mode_steps": int(np.count_nonzero(active)),
        "interpolated_mode_steps": int(np.count_nonzero(interpolated_length > 0)),
        "fully_interpolated_mode_steps": int(np.count_nonzero(
            active & np.isclose(interpolated_length[:-1], lengths[:, None],
                                rtol=1e-9, atol=1e-16)
        )),
        "integration_pieces": len(pieces),
        "grid_crossings": max(0, len(roots) - len(boundaries)),
        "missing_overlap_edges": missing_edges,
        "fallback_pieces_by_reason": dict(fallback_pieces),
        "fallback_mode_length_fraction": fallback_total / guided_length if guided_length else 0.0,
    }
    if fallback_total:
        warnings.warn(
            "neff_interpolation retained the original phase over "
            f"{diagnostics['fallback_mode_length_fraction']:.1%} of guided mode length "
            f"({dict(fallback_pieces)})",
            RuntimeWarning,
            stacklevel=2,
        )
    return beta, diagnostics
