"""Numerical-stability helpers for eigenmode expansion interfaces."""

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass(frozen=True)
class EMEStabilityConfig:
    """Small set of thresholds used by the stabilized EME path."""

    guided_loss: float = 2.84e-5  # approximately 10 dB/cm at 1550 nm
    pml_loss: float = 2.84e-3  # approximately 10 dB/100 um at 1550 nm
    guided_rcond: float = 1e-5
    pml_rcond: float = 0.05
    feedback_pml_rcond: float = 0.01 #1e-3
    guided_fraction_threshold: float = 0.8
    feedback_pml_fraction_threshold: float = 0.8
    absolute_rcond: float = 1e-8

    def __post_init__(self):
        if not 0 < self.guided_loss < self.pml_loss:
            raise ValueError("guided_loss must be positive and smaller than pml_loss")
        if not 0 < self.guided_rcond <= self.pml_rcond < 1:
            raise ValueError("rcond values must satisfy 0 < guided_rcond <= pml_rcond < 1")
        if not 0 < self.feedback_pml_rcond < 1:
            raise ValueError("feedback_pml_rcond must be between 0 and 1")
        if not 0 <= self.guided_fraction_threshold <= 1:
            raise ValueError("guided_fraction_threshold must be between 0 and 1")
        if not 0 <= self.feedback_pml_fraction_threshold <= 1:
            raise ValueError(
                "feedback_pml_fraction_threshold must be between 0 and 1"
            )
        if not 0 < self.absolute_rcond < 1:
            raise ValueError("absolute_rcond must be between 0 and 1")


def new_stability_diagnostics():
    """Return an empty, deliberately sparse diagnostics record."""

    return {
        "status": "normal",
        "interface_events": [],
        "feedback_fallbacks": [],
    }


def mode_reliability_weights(neff, mode_present, config):
    """Map modal loss to [0, 1] reliability using logarithmic interpolation."""

    loss = np.abs(np.asarray(neff, dtype=np.complex128).imag)
    present = np.asarray(mode_present, dtype=bool)
    if loss.shape != present.shape:
        raise ValueError("neff and mode_present must have the same shape")

    weights = np.ones(loss.shape, dtype=float)
    pml = loss >= config.pml_loss
    leaky = (loss > config.guided_loss) & ~pml
    weights[pml] = 0.0
    if np.any(leaky):
        log_span = np.log(config.pml_loss) - np.log(config.guided_loss)
        weights[leaky] = 1.0 - (
            np.log(loss[leaky]) - np.log(config.guided_loss)
        ) / log_span
    weights[~present] = 0.0
    return np.clip(weights, 0.0, 1.0)


def stabilized_inverse(
    matrix,
    input_weights,
    output_weights,
    config,
    diagnostics=None,
    interface_index=None,
    direction=None,
):
    """Return a mode-aware truncated-SVD inverse of one interface matrix."""

    matrix = np.asarray(matrix, dtype=np.complex128)
    input_weights = np.asarray(input_weights, dtype=float)
    output_weights = np.asarray(output_weights, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    if input_weights.shape != (matrix.shape[1],):
        raise ValueError("input_weights do not match matrix columns")
    if output_weights.shape != (matrix.shape[0],):
        raise ValueError("output_weights do not match matrix rows")

    u, singular_values, vh = np.linalg.svd(matrix, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros_like(matrix)

    sigma_max = singular_values[0]
    normalized = (
        singular_values / sigma_max
        if sigma_max > 0
        else np.zeros_like(singular_values)
    )
    v = vh.conj().T
    guided_fraction = 0.5 * (
        (np.abs(v) ** 2).T @ input_weights
        + (np.abs(u) ** 2).T @ output_weights
    )
    adaptive_cutoff = np.where(
        guided_fraction >= config.guided_fraction_threshold,
        config.guided_rcond,
        config.pml_rcond,
    )
    keep = (
        (normalized >= adaptive_cutoff)
        & (normalized >= config.absolute_rcond)
    )

    inverse_values = np.zeros_like(singular_values)
    inverse_values[keep] = 1.0 / singular_values[keep]
    inverse = (v * inverse_values[np.newaxis, :]) @ u.conj().T

    dropped = ~keep
    guided_heavy_absolute = dropped & (
        guided_fraction >= config.guided_fraction_threshold
    ) & (normalized < config.absolute_rcond)
    reportable_drop = dropped & (normalized >= config.absolute_rcond)
    if diagnostics is not None and (
        np.any(reportable_drop) or np.any(guided_heavy_absolute)
    ):
        diagnostics["status"] = "regularized"
        diagnostics["interface_events"].append({
            "index": interface_index,
            "direction": direction,
            "dropped_count": int(np.count_nonzero(reportable_drop)),
            "guided_heavy_dropped": int(np.count_nonzero(guided_heavy_absolute)),
        })
        if np.any(guided_heavy_absolute):
            warnings.warn(
                "A guided-heavy singular direction was below absolute_rcond "
                f"at interface {interface_index} ({direction}).",
                RuntimeWarning,
                stacklevel=2,
            )

    return inverse
