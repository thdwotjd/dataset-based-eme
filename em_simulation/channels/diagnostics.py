"""Coherent tangential-field residuals and signed endpoint power diagnostics."""

from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np

from ._fields import FIELD_CONVENTION, finite_array
from .mapping import (
    ChannelMappingWarning, MappingConfig, _hermitian, _product_arrays,
    _quadratic, _transform_blocks, _usable_denominator,
)


@dataclass(frozen=True)
class FieldNormProducts:
    """One field's weighted norm Grams S_U/S_V and cross term X."""

    S_U: np.ndarray
    S_V: np.ndarray
    X: np.ndarray

    def __post_init__(self):
        for name, value in zip(("S_U", "S_V", "X"), _product_arrays(self.S_U, self.S_V, self.X)):
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class FieldProducts:
    E: FieldNormProducts
    H: FieldNormProducts

    @property
    def components(self):
        return FIELD_CONVENTION["tangential_components"]


@dataclass(frozen=True)
class FieldResidual:
    squared_norm: float
    reference_squared_norm: float
    relative_squared: Optional[float]
    reason: Optional[str]
    norm_unit: str
    warning_threshold: Optional[float]

    @property
    def absolute_norm(self):
        return float(np.sqrt(self.squared_norm))

    @property
    def relative_norm(self):
        return None if self.relative_squared is None else float(np.sqrt(self.relative_squared))

    @property
    def defined(self):
        return self.relative_squared is not None

    @property
    def threshold_exceeded(self):
        return (self.defined and self.warning_threshold is not None
                and self.relative_norm > self.warning_threshold)


@dataclass(frozen=True)
class EndpointResiduals:
    E: FieldResidual
    H: FieldResidual
    context: str


@dataclass(frozen=True)
class SignedPowerDifference:
    watts: float
    relative: Optional[float]
    reason: Optional[str]

    @property
    def defined(self):
        return self.relative is not None


@dataclass(frozen=True)
class PowerBudget:
    """Four checkpoints in watts. Forward throughput change can include reflection."""

    P0: float
    P1: float
    P2: float
    P3: float
    input_mapping: SignedPowerDifference
    output_mapping: SignedPowerDifference
    budget_closure_w: float

    @property
    def forward_throughput_change_w(self):
        return self.P1 - self.P2

    @property
    def total_difference_w(self):
        return self.P0 - self.P3


@dataclass(frozen=True)
class MappingDiagnostics:
    input_residuals: EndpointResiduals
    output_residuals: EndpointResiduals
    power: PowerBudget
    input_forward: np.ndarray
    output_channels: np.ndarray


def _field_inner(a, b, weights):
    return np.einsum("mcij,ncij,ij->mn", a[:, 1:].conj(), b[:, 1:], weights, optimize=True)


def compute_field_products(U, V):
    """Compute separate full E and H norm products for the tangential y/z components."""
    U.grid.require_match(V.grid)
    weights = U.grid.weights
    return FieldProducts(*(
        FieldNormProducts(_field_inner(u, u, weights), _field_inner(v, v, weights),
                          _field_inner(u, v, weights))
        for u, v in ((U.E, V.E), (U.H, V.H))
    ))


def transform_field_products(products, R):
    """Apply U_run = U @ R to both E/H product sets."""
    return FieldProducts(*(
        FieldNormProducts(*_transform_blocks(p.S_U, p.S_V, p.X, R))
        for p in (products.E, products.H)
    ))


def reconstruct_fields(fields, amplitudes):
    """Reconstruct full complex E/H arrays (component, Ny, Nz)."""
    p = finite_array(amplitudes, "field amplitudes", shape=(fields.count,))
    return tuple(np.einsum("m,mcij->cij", p, array, optimize=True)
                 for array in (fields.E, fields.H))


def _ratio(numerator, denominator, scale, config, *, absolute_minimum=0.0):
    if not _usable_denominator(denominator, scale, config, absolute_minimum=absolute_minimum):
        return None, "reference is nonpositive or numerically unreliable"
    value = numerator / denominator
    return (value, None) if np.isfinite(value) else (None, "relative value is nonfinite")


def _nonnegative(value, scale, config, name):
    if value < -config.roundoff_rtol * scale:
        raise ValueError(f"{name} has a negative squared norm: {value:.6g}")
    return max(0.0, value)


def _residual(squared, reference, reference_scale, config, field):
    limit, unit = ((config.e_residual_warning, "V") if field == "E"
                   else (config.h_residual_warning, "A"))
    relative, reason = _ratio(squared, reference, reference_scale, config)
    return FieldResidual(squared, reference, relative, reason, unit, limit)


def _report_residuals(E, H, context):
    for name, residual in (("E", E), ("H", H)):
        if residual.threshold_exceeded:
            warnings.warn(f"{context}: relative {name} residual {residual.relative_norm:.6g} "
                          f"exceeds {residual.warning_threshold:.6g}",
                          ChannelMappingWarning, stacklevel=3)
    return EndpointResiduals(E, H, context)


def evaluate_direct_residual(source, source_amplitudes, target, target_amplitudes, *,
                             config=MappingConfig(), context="direct reconstruction"):
    """Integrate the explicitly reconstructed residual on the shared grid."""
    source.grid.require_match(target.grid)
    weights = source.grid.weights
    source_fields = reconstruct_fields(source, source_amplitudes)
    target_fields = reconstruct_fields(target, target_amplitudes)
    errors = []
    magnitudes = np.abs(np.asarray(source_amplitudes, dtype=np.complex128))
    for name, modes, a, b in zip(("E", "H"), (source.E, source.H), source_fields, target_fields):
        reference = float(np.sum(abs(a[1:])**2 * weights))
        squared = float(np.sum(abs(a[1:] - b[1:])**2 * weights))
        # A small coherent field can come from cancellation of large amplitudes.
        bound = np.einsum("m,mcij->cij", magnitudes, abs(modes[:, 1:]), optimize=True)
        reference_scale = float(np.sum(bound**2 * weights))
        if not np.isfinite([reference, squared, reference_scale]).all():
            raise ValueError(f"{context}: nonfinite {name} field norm")
        errors.append(_residual(squared, reference, reference_scale, config, name))
    return _report_residuals(*errors, context)


def _field_residual(products, source_amplitudes, target_amplitudes, *, direction, config, context):
    errors = []
    for name, blocks in (("E", products.E), ("H", products.H)):
        if direction == "input":
            source, target, cross = blocks.S_V, blocks.S_U, blocks.X.conj().T
        else:
            source, target, cross = blocks.S_U, blocks.S_V, blocks.X
        label = f"{context} {name}"
        source = _hermitian(source, f"{label} source norm Gram", config)
        target = _hermitian(target, f"{label} target norm Gram", config)
        # Apply the map before forming norms so cancellation in the amplitudes is preserved.
        reference, reference_scale = _quadratic(source_amplitudes, source, f"{label} reference", config)
        represented, represented_scale = _quadratic(target_amplitudes, target, f"{label} represented", config)
        cross_real = float(np.vdot(source_amplitudes, cross @ target_amplitudes).real)
        squared = reference + represented - 2.0 * cross_real
        cross_scale = float(abs(source_amplitudes) @ abs(cross) @ abs(target_amplitudes))
        scale = reference_scale + represented_scale + 2.0 * cross_scale
        if not np.isfinite(squared) or not np.isfinite(scale):
            raise ValueError(f"{label} residual is not finite")
        squared = _nonnegative(squared, scale, config, f"{label} residual")
        reference = _nonnegative(reference, reference_scale, config, f"{label} reference")
        errors.append(_residual(squared, reference, reference_scale, config, name))
    return _report_residuals(*errors, context)


def _power_difference(reference, represented, scale, config, label):
    delta = reference - represented
    if not np.isfinite(delta):
        raise ValueError(f"{label} is not finite")
    ratio, reason = _ratio(delta, reference, scale, config,
                           absolute_minimum=config.min_reference_power_w)
    return SignedPowerDifference(delta, ratio, reason)


def evaluate_mapping_diagnostics(input_power, output_power, input_fields, output_fields,
                                 maps, p, a_out):
    """Evaluate the actual A/B maps and amplitudes without any renormalization.

    p is the effective input channel vector, possibly already normalized once.
    a_out is the full transmitted forward vector, including radiation/PML slots.
    """
    config = maps.config
    input_context = maps.input_solve.context
    output_context = maps.output_solve.context
    for direction, power, fields in (("input", input_power, input_fields),
                                     ("output", output_power, output_fields)):
        if fields.E.X.shape != power.C.shape or fields.H.X.shape != power.C.shape:
            raise ValueError(f"{direction} power and field product shapes do not agree")
    A = finite_array(maps.A, "input map A", shape=input_power.C.shape)
    B = finite_array(maps.B, "output map B", shape=output_power.C.T.shape)
    p = finite_array(p, "effective input channels", shape=(A.shape[1],))
    a_out = finite_array(a_out, "transmitted forward amplitudes", shape=(B.shape[1],))
    a_in = finite_array(A @ p, "represented forward amplitudes")
    p_out = finite_array(B @ a_out, "output channel amplitudes")

    g0 = _hermitian(input_power.G_V, f"{input_context} G_V", config)
    g1 = _hermitian(input_power.G_U, f"{input_context} G_U", config)
    g2 = _hermitian(output_power.G_U, f"{output_context} G_U", config)
    g3 = _hermitian(output_power.G_V, f"{output_context} G_V", config)
    P0, s0 = _quadratic(p, g0, "P0 input reference power", config)
    P1, _ = _quadratic(a_in, g1, "P1 represented input power", config)
    P2, s2 = _quadratic(a_out, g2, "P2 transmitted forward power", config)
    P3, _ = _quadratic(p_out, g3, "P3 output reference power", config)
    # Use power after applying the actual maps, so amplitude cancellation happens
    # before power evaluation instead of inside a difference of Gram matrices.
    delta_in = _power_difference(P0, P1, s0, config, f"{input_context} power discrepancy")
    delta_out = _power_difference(P2, P3, s2, config, f"{output_context} power discrepancy")
    closure = (P0 - P3) - (delta_in.watts + (P1 - P2) + delta_out.watts)
    budget = PowerBudget(P0, P1, P2, P3, delta_in, delta_out, closure)
    return MappingDiagnostics(
        _field_residual(input_fields, p, a_in, direction="input", config=config, context=input_context),
        _field_residual(output_fields, a_out, p_out, direction="output", config=config, context=output_context),
        budget, a_in, p_out,
    )
