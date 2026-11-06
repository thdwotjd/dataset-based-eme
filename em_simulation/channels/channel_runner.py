"""Channel launch and readout using the actual tracked EME basis."""

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
from numbers import Integral, Real
import warnings

import numpy as np

from ..geometry.coupled_waveguide.double_waveguide import DoubleWaveguide
from ..geometry.parameter_path import PathClipping
from ..propagator.single_propagator.single_eme import SingleEME
from ..runner.runner import Runner
from .diagnostics import (
    FieldNormProducts, FieldProducts, MappingDiagnostics, evaluate_mapping_diagnostics, transform_field_products,
)
from .mapping import (
    ChannelMappingWarning, InputAmplitudes, MappingConfig, PowerProducts, SolveInfo,
    normalize_input_amplitudes, solve_channel_maps, transform_products,
)


@dataclass(frozen=True)
class ChannelEndpoint:
    """Reference plane and basis. Forward slots are zero-based tracking IDs."""

    node: tuple
    requested_parameters: dict
    s_m: float
    basis_id: str
    channel_basis_id: str
    canonical_forward_mode_ids: tuple
    channel_keys: tuple
    forward_slots: tuple
    guided_slots: tuple
    R: np.ndarray


@dataclass(frozen=True)
class ChannelMappingDiagnostics:
    input_endpoint: ChannelEndpoint
    output_endpoint: ChannelEndpoint
    input_amplitudes: InputAmplitudes
    input_solve: SolveInfo
    output_solve: SolveInfo
    evaluation: MappingDiagnostics
    transmitted_forward: np.ndarray
    warnings: tuple


@dataclass(frozen=True)
class ChannelDiagnostics:
    channel_mapping: ChannelMappingDiagnostics
    path_clipping: PathClipping
    propagation_stability: dict


@dataclass(frozen=True)
class ChannelResult:
    """Complex amplitudes keyed by (guide ID, local 1-based FDE mode number)."""

    output_amplitudes: dict
    output_labels: dict
    diagnostics: ChannelDiagnostics


class ChannelRunner:
    """Launch from the left and read the transmitted channels at the right.

    Labels are for display; inputs and outputs use canonical channel keys.
    The stored generation is loaded once. Maps are solved for the actual
    endpoint basis, mode masks and config on every run.
    """

    def __init__(self, runner, *, guide_labels=None, te_fraction_threshold=0.5, mapping_config=None):
        if not isinstance(runner, Runner) or not isinstance(runner.propagator, SingleEME):
            raise TypeError("ChannelRunner requires a Runner with a single EME propagator")
        geometry = runner.propagator.geometry
        if not isinstance(geometry, DoubleWaveguide) or not hasattr(geometry, "parameter_path"):
            raise TypeError("ChannelRunner requires a double-waveguide geometry with a parameter path")
        self.runner = runner
        self._dataset = geometry.data
        self._dataset_fingerprint = _dataset_fingerprint(self._dataset)
        self._basis_at(0)
        data = self._dataset.load_channel_data(geometry.section_model)
        self._nodes = data["nodes"]
        self._metadata = data["dataset_metadata"]
        if mapping_config is None:
            mapping_config = MappingConfig(**self._metadata["mapping_solver_config"])
        if not isinstance(mapping_config, MappingConfig):
            raise TypeError("mapping_config must be a MappingConfig")
        self._mapping_config = mapping_config

        threshold = float(te_fraction_threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("te_fraction_threshold must be in [0, 1]")
        self._te_threshold = threshold
        guides = self._metadata["section_model"]["guides"]
        labels = {} if guide_labels is None else dict(guide_labels)
        if set(labels) - guides.keys():
            raise ValueError(f"Unknown guide IDs in guide_labels: {set(labels) - guides.keys()}")
        self._guide_labels = {guide: labels.get(guide, guide) for guide in guides}
        names = self._guide_labels.values()
        if any(not isinstance(name, str) or not name.strip() for name in names):
            raise ValueError("Guide labels must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("Duplicate guide labels")

    def _basis_at(self, index):
        propagator = self.runner.propagator
        geometry = propagator.geometry
        if (propagator.output_data is not geometry.output_data or geometry.data is not self._dataset
                or _dataset_fingerprint(self._dataset) != self._dataset_fingerprint):
            raise ValueError("The geometry or dataset changed. Rebuild EME, Runner and ChannelRunner.")
        basis = geometry.runtime_basis_at(index)
        if self._dataset.canonical_basis_ids.get(basis.node) != basis.basis_id:
            raise ValueError(f"Runtime basis at node {basis.node} differs from the dataset")
        return basis

    def _endpoint(self, side):
        if side not in ("input", "output"):
            raise ValueError("side must be 'input' or 'output'")
        index = 0 if side == "input" else -1
        basis = self._basis_at(index)
        record = self._nodes.get(basis.node)
        if (record is None or record["canonical_basis_id"] != basis.basis_id
                or len(record["canonical_forward_mode_ids"]) != basis.R.shape[0]):
            raise ValueError(f"No matching channel basis for the {side} node {basis.node} ({basis.basis_id})")
        return basis, record

    def _family(self, fraction):
        if not isinstance(fraction, Real) or not 0 <= fraction <= 1:
            return None
        return "TE" if fraction >= self._te_threshold else "TM"

    def _channel_details(self, basis, record):
        # Sort within each guide/family for labels, and preserve storage order for use with matrices.
        details = {}
        counts = {}
        for channel in sorted(record["channels"], key=lambda c: (-c["neff"].real, c["key"][1])):
            guide, number = channel["key"]
            family = self._family(channel["te_fraction"])
            if family is None:
                name = f"mode {number}"
            else:
                group = guide, family
                index = counts.get(group, 0)
                name = f"{family}{index}"
                counts[group] = index + 1
            details[channel["key"]] = dict(
                channel, family=family, display_name=f"{self._guide_labels[guide]}_{name}",
                node=basis.node, canonical_basis_id=basis.basis_id, channel_basis_id=record["channel_basis_id"],
                normalization_power_w=self._metadata["isolated_normalization_w"],
                phase_convention=self._metadata["isolated_phase"],
            )
        return tuple(details[channel["key"]] for channel in record["channels"])

    def available_channels(self, side, *, verbose=False):
        """Return endpoint keys in matrix order, or metadata with display names."""
        basis, record = self._endpoint(side)
        if verbose:
            return self._channel_details(basis, record)
        return tuple(channel["key"] for channel in record["channels"])

    def supermode_labels(self, section_index, *, include_polarization=False):
        """Return labels keyed by tracking slot; mode numbers in labels start at 1."""
        basis = self._basis_at(section_index)
        fractions = self.runner.propagator.output_data["TE_pol"][section_index]
        labels = {}
        for slot in basis.forward_slots:
            label = f"mode {slot + 1}"
            family = self._family(fractions[slot]) if include_polarization else None
            labels[int(slot)] = f"{label} ({family})" if family else label
        return labels

    def _snapshot(self, side, basis, record):
        propagator = self.runner.propagator
        path = propagator.geometry.parameter_path
        index = 0 if side == "input" else -1
        return ChannelEndpoint(
            basis.node, {path.parameter_name: float(path.requested_values[index])},
            0.0 if index == 0 else float(np.sum(propagator._lengths_per_matrix)),
            basis.basis_id, record["channel_basis_id"], record["canonical_forward_mode_ids"],
            tuple(channel["key"] for channel in record["channels"]),
            tuple(map(int, basis.forward_slots)), tuple(map(int, basis.guided_slots)), basis.R.copy(),
        )

    def run(self, input_amplitudes, *, input_power=None):
        """Propagate channel amplitudes with optional coherent input normalization in watts."""
        left, input_record = self._endpoint("input")
        right, output_record = self._endpoint("output")
        input_endpoint = self._snapshot("input", left, input_record)
        output_endpoint = self._snapshot("output", right, output_record)
        p = _channel_vector(input_amplitudes, input_endpoint)
        input_gram, input_fields = _runtime_products(input_record, left.R)
        output_gram, output_fields = _runtime_products(output_record, right.R)
        config = self._mapping_config
        normalized = normalize_input_amplitudes(p, input_gram, input_power=input_power, config=config)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ChannelMappingWarning)
            maps = solve_channel_maps(
                input_gram, output_gram, input_guided=left.forward_guided, output_present=right.forward_present,
                config=config, input_context=f"input node {left.node} ({left.basis_id})",
                output_context=f"output node {right.node} ({right.basis_id})",
            )
            a_in = maps.A @ normalized.effective
            # A already has one row per forward slot. The backward half is right incidence.
            full_input = np.concatenate((a_in, np.zeros_like(a_in)))
            full_output = self.runner.propagate_basis_amplitudes(full_input)
            # Absent columns have no physical basis. Present nonguided columns still contribute.
            a_out = np.where(right.forward_present, full_output[:right.R.shape[1]], 0)
            evaluation = evaluate_mapping_diagnostics(
                input_gram, output_gram, input_fields, output_fields, maps, normalized.effective, a_out,
            )
        for item in caught:
            warnings.warn_explicit(item.message, item.category, item.filename, item.lineno)

        diagnostic = ChannelMappingDiagnostics(
            input_endpoint, output_endpoint, normalized, maps.input_solve, maps.output_solve,
            evaluation, a_out, tuple(str(item.message) for item in caught),
        )
        propagator = self.runner.propagator
        path = propagator.geometry.parameter_path
        clipping = path.clipping
        original_length = path.node_positions[-1]
        if clipping.positions and output_endpoint.s_m != original_length:
            scale = output_endpoint.s_m / original_length
            clipping = replace(clipping, positions=tuple(s * scale for s in clipping.positions))
        return ChannelResult(
            dict(zip(output_endpoint.channel_keys, map(complex, evaluation.output_channels))),
            {item["key"]: item["display_name"] for item in self._channel_details(right, output_record)},
            ChannelDiagnostics(diagnostic, clipping,
                               deepcopy(propagator.stability_diagnostics)),
        )


def _dataset_fingerprint(dataset):
    manifest = getattr(dataset, "canonical_manifest", None)
    if manifest is None:
        return None
    return manifest["generation_id"], manifest["file_sha256"].copy(), manifest.get("channel_data_sha256")


def _channel_vector(amplitudes, endpoint):
    if not isinstance(amplitudes, Mapping):
        raise TypeError("input_amplitudes must be a mapping from canonical channel keys to scalars")
    indices = {key: index for index, key in enumerate(endpoint.channel_keys)}
    p = np.zeros(len(indices), dtype=np.complex128)
    for key, value in amplitudes.items():
        if (not isinstance(key, tuple) or len(key) != 2
                or not isinstance(key[1], Integral) or isinstance(key[1], bool) or key not in indices):
            raise ValueError(f"Unknown input channel {key!r} at node {endpoint.node}")
        if np.ndim(value) != 0 or isinstance(value, (str, bytes)):
            raise ValueError(f"Amplitude for input channel {key!r} must be a finite complex scalar")
        try:
            value = complex(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Amplitude for input channel {key!r} must be a finite complex scalar") from exc
        if not np.isfinite(value):
            raise ValueError(f"Amplitude for input channel {key!r} must be a finite complex scalar")
        p[indices[key]] = value
    return p


def _runtime_products(record, R):
    power = PowerProducts(**record["power_products"])
    fields = FieldProducts(*(FieldNormProducts(**record["field_products"][f]) for f in ("E", "H")))
    return transform_products(power, R), transform_field_products(fields, R)
