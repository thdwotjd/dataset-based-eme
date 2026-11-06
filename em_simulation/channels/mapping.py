"""Hermitian power products and forward channel maps, independent of EME overlaps."""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

from ._fields import ForwardFields, finite_array


class ChannelMappingWarning(UserWarning):
    """A channel map or its reconstruction needs attention."""


@dataclass(frozen=True)
class MappingConfig:
    """Numerical cutoffs for complex128. Residual warning limits are optional."""

    svd_rtol: float = 1e-12
    svd_atol: float = 0.0
    condition_warning: float = 1e12
    denominator_rtol: float = 1e-12
    roundoff_rtol: float = 1e-12
    min_reference_power_w: float = 0.0
    e_residual_warning: Optional[float] = None
    h_residual_warning: Optional[float] = None

    def __post_init__(self):
        for name in ("svd_rtol", "denominator_rtol", "roundoff_rtol"):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 <= value < 1:
                raise ValueError(f"{name} must be finite and in [0, 1)")
        if not np.isfinite(self.condition_warning) or self.condition_warning < 1:
            raise ValueError("condition_warning must be finite and at least 1")
        for name in ("svd_atol", "min_reference_power_w", "e_residual_warning", "h_residual_warning"):
            value = getattr(self, name)
            if name in ("e_residual_warning", "h_residual_warning") and value is None:
                continue
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")


def _product_arrays(U, V, X):
    X = finite_array(X, "cross product", ndim=2)
    k, m = X.shape
    return (finite_array(U, "U Gram", shape=(k, k)),
            finite_array(V, "V Gram", shape=(m, m)), X)


@dataclass(frozen=True)
class PowerProducts:
    """Hermitian power Gram matrices G_U, G_V and cross product C = <U,V>."""

    G_U: np.ndarray
    G_V: np.ndarray
    C: np.ndarray

    def __post_init__(self):
        for name, value in zip(("G_U", "G_V", "C"), _product_arrays(self.G_U, self.G_V, self.C)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class SolveInfo:
    method: str
    rank: int
    condition: float
    cutoff: float
    singular_values: tuple
    equation_residual: float
    relative_equation_residual: float
    context: str


@dataclass(frozen=True)
class ChannelMaps:
    A: np.ndarray  # full forward input slots x input channels
    B: np.ndarray  # output channels x full forward output slots
    input_solve: SolveInfo
    output_solve: SolveInfo
    config: MappingConfig


@dataclass(frozen=True)
class InputAmplitudes:
    raw: np.ndarray
    effective: np.ndarray
    raw_power_w: float
    scale: float
    target_power_w: Optional[float]


def _cross_flux(E, H, weights):
    # x-component of E* x H, with axes in global x,y,z order.
    Ey, Ez = E[:, 1].conj(), E[:, 2].conj()
    return (np.einsum("mij,nij,ij->mn", Ey, H[:, 2], weights, optimize=True)
            - np.einsum("mij,nij,ij->mn", Ez, H[:, 1], weights, optimize=True))


def compute_power_products(U, V):
    """Integrate the symmetric conjugated Poynting product on the same plane."""
    U.grid.require_match(V.grid)
    weights = U.grid.weights
    uu = _cross_flux(U.E, U.H, weights)
    vv = _cross_flux(V.E, V.H, weights)
    uv = _cross_flux(U.E, V.H, weights)
    vu = _cross_flux(V.E, U.H, weights)
    return PowerProducts((uu + uu.conj().T) / 4, (vv + vv.conj().T) / 4, (uv + vu.conj().T) / 4)


def normalize_channel_fields(fields, *, config=MappingConfig()):
    """Scale each isolated reference to 1 W using the same positive real E/H factor.

    Returns the normalized ForwardFields and the applied mode-wise factors.
    Coupled fields retain their original DREME normalization.
    """
    E, H = fields.E, fields.H
    weights = fields.grid.weights
    term_y = E[:, 1] * H[:, 2].conj()
    term_z = E[:, 2] * H[:, 1].conj()
    power = 0.5 * np.sum((term_y - term_z).real * weights, axis=(1, 2))
    scale = 0.5 * np.sum((abs(term_y) + abs(term_z)) * weights, axis=(1, 2))
    threshold = np.maximum(config.min_reference_power_w, config.denominator_rtol * scale)
    invalid = (~np.isfinite(power) | ~np.isfinite(scale)
               | (power <= np.maximum(threshold, np.finfo(float).tiny)))
    if invalid.any():
        index = np.flatnonzero(invalid)[0]
        raise ValueError(f"isolated reference {index} has nonpositive or unreliable forward power "
                         f"({power[index]:.6g} W)")
    factors = 1 / np.sqrt(power)
    factor = factors[:, None, None, None]
    return ForwardFields(E * factor, H * factor, fields.grid), factors


def _transform_blocks(U, V, X, R):
    R = finite_array(R, "forward basis transform R", ndim=2)
    if R.shape[0] != U.shape[0]:
        raise ValueError(f"R has {R.shape[0]} rows; expected {U.shape[0]} canonical modes")
    adjoint = R.conj().T
    return adjoint @ U @ R, V, adjoint @ X


def transform_products(products, R):
    """Apply U_run = U @ R, retaining zeros for absent runtime columns."""
    return PowerProducts(*_transform_blocks(products.G_U, products.G_V, products.C, R))


def _hermitian(matrix, name, config):
    scale = np.max(np.abs(matrix), initial=0.0)
    error = np.max(np.abs(matrix - matrix.conj().T), initial=0.0)
    if error > config.roundoff_rtol * scale:
        raise ValueError(f"{name} is not Hermitian within roundoff tolerance")
    return matrix / 2 + matrix.conj().T / 2


def _quadratic(vector, matrix, name, config):
    value = np.vdot(vector, matrix @ vector)
    magnitudes = np.abs(vector)
    scale = float(magnitudes @ np.abs(matrix) @ magnitudes)
    if not np.isfinite(value) or not np.isfinite(scale):
        raise ValueError(f"{name} is not finite")
    if abs(value.imag) > config.roundoff_rtol * scale:
        raise ValueError(f"{name} has a significant imaginary part: {value!r}")
    return float(value.real), scale


def _usable_denominator(value, scale, config, *, absolute_minimum=0.0):
    return value > max(absolute_minimum, config.denominator_rtol * scale, np.finfo(float).tiny)


def _solve(gram, rhs, config, context):
    size = gram.shape[0]
    if size == 0 or rhs.shape[1] == 0:
        residual = float(np.linalg.norm(rhs))
        return np.zeros_like(rhs), SolveInfo(
            "empty", 0, np.inf, 0.0, (), residual, 1.0 if residual else 0.0, context,
        )
    try:
        left, values, right_h = np.linalg.svd(gram, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"{context}: power Gram SVD did not converge") from exc
    # Ratio first avoids overflow when the minimum singular value is tiny.
    relative_min = float(values[-1] / values[0]) if values[0] else 0.0
    condition = 1.0 / relative_min if relative_min else np.inf
    cutoff = max(config.svd_atol, config.svd_rtol * values[0])
    keep = values > cutoff
    rank = int(np.count_nonzero(keep))
    method = "solve"
    if rank == size and condition <= config.condition_warning:
        try:
            solution = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            method = "truncated_svd"
    else:
        method = "truncated_svd"
    if method == "truncated_svd":
        solution = right_h[keep].conj().T @ ((left[:, keep].conj().T @ rhs) / values[keep, None])
        warnings.warn(f"{context}: power Gram solve used truncated SVD (rank {rank}/{size}, "
                      f"condition {condition:.3g}, cutoff {cutoff:.3g})",
                      ChannelMappingWarning, stacklevel=3)
    residual = float(np.linalg.norm(gram @ solution - rhs))
    rhs_norm = float(np.linalg.norm(rhs))
    relative = residual / rhs_norm if rhs_norm else 0.0
    if not np.isfinite(solution).all() or not np.isfinite(residual) or not np.isfinite(relative):
        raise ValueError(f"{context}: nonfinite power Gram solution or residual")
    return solution, SolveInfo(method, rank, condition, float(cutoff), tuple(map(float, values)),
                               residual, relative, context)


def _slots(mask, size, name):
    mask = np.asarray(mask)
    if mask.dtype.kind != "b" or mask.shape != (size,):
        raise ValueError(f"{name} must be a boolean mask of shape ({size},)")
    return np.flatnonzero(mask)


def solve_channel_maps(input_products, output_products, *, input_guided, output_present,
                       config=MappingConfig(), input_context="input", output_context="output"):
    """Solve guided input and full present forward output maps at their own endpoints.

    Pass Commit 2's forward_guided and forward_present masks after transforming
    each endpoint's canonical products with its actual runtime R.
    """
    guided = _slots(input_guided, input_products.C.shape[0], "input_guided")
    present = _slots(output_present, output_products.C.shape[0], "output_present")
    G_in = _hermitian(input_products.G_U, f"{input_context} G_U", config)
    G_out = _hermitian(output_products.G_V, f"{output_context} G_V", config)
    A_guided, input_info = _solve(G_in[np.ix_(guided, guided)], input_products.C[guided], config, input_context)
    B_present, output_info = _solve(G_out, output_products.C[present].conj().T, config, output_context)
    A = np.zeros_like(input_products.C)
    B = np.zeros_like(output_products.C.T)
    A[guided] = A_guided
    B[:, present] = B_present
    return ChannelMaps(A, B, input_info, output_info, config)


def normalize_input_amplitudes(p_raw, products, *, input_power=None, config=MappingConfig()):
    """Apply an optional total reference-power target once, before the input map."""
    p = finite_array(p_raw, "input channel amplitudes", shape=(products.C.shape[1],)).copy()
    G = _hermitian(products.G_V, "input G_V", config)
    raw_power, magnitude = _quadratic(p, G, "raw input reference power", config)
    if input_power is None:
        return InputAmplitudes(p, p.copy(), raw_power, 1.0, None)
    target = float(input_power)
    if not np.isfinite(target) or target <= 0:
        raise ValueError("input_power must be a positive finite target in watts")
    if not _usable_denominator(raw_power, magnitude, config,
                               absolute_minimum=config.min_reference_power_w):
        raise ValueError(f"raw input reference power {raw_power:.6g} W is nonpositive or unreliable")
    scale = float(np.sqrt(target) / np.sqrt(raw_power))
    effective = finite_array(scale * p, "power-normalized input channel amplitudes")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("input power normalization produced an invalid scale")
    return InputAmplitudes(p, effective, raw_power, scale, target)
