"""Persistence and basis validation for the channel records of one generation."""

from hashlib import sha256
from pathlib import Path
import pickle

import numpy as np

from ..channels import FIELD_CONVENTION
from .canonical_dataset import ExportGrid, _atomic_write


CHANNEL_FILE = "channel_data.pkl"
SCHEMA_VERSION = 1


def save_channel_checkpoint(directory, data, *, node=None):
    """Atomically publish a completed node or the final channel document."""
    path = Path(directory) / CHANNEL_FILE
    context = f"node {node}" if node is not None else "generation completion"
    try:
        payload = pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL)
        _atomic_write(path, payload)
    except Exception as exc:
        raise RuntimeError(f"Cannot save channel {context} to {path}: {exc}") from exc
    return sha256(payload).hexdigest() if node is None else None


def _read(directory):
    path = Path(directory) / CHANNEL_FILE
    try:
        payload = path.read_bytes()
        data = pickle.loads(payload)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Missing channel data file {path}") from exc
    except (OSError, pickle.PickleError, EOFError, ImportError, AttributeError) as exc:
        raise ValueError(f"Cannot read channel data file {path}: {exc}") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        version = data.get("schema_version") if isinstance(data, dict) else None
        raise ValueError(f"Unsupported channel data schema version {version!r} in {path}")
    if not isinstance(data.get("nodes"), dict) or not isinstance(data.get("dataset_metadata"), dict):
        raise ValueError(f"Invalid channel checkpoint in {path}")
    return data, payload


def read_channel_checkpoint(directory):
    """Inspect persisted nodes of a partial build without treating it as a dataset."""
    return _read(directory)[0]


def _check_products(blocks, shapes, context):
    for name, shape in shapes.items():
        array = blocks[name]
        if (not isinstance(array, np.ndarray) or array.shape != shape
                or array.dtype.kind not in "fc" or not np.isfinite(array).all()):
            raise ValueError(f"{context} {name} must be a finite array with shape {shape}")


def validate_channel_node(record, basis, mode_ids, guide_budgets):
    if (record["completion"] != "complete" or record["canonical_basis_id"] != basis
            or record["canonical_forward_mode_ids"] != mode_ids
            or record["channel_basis_id"] != f"{basis}:channels"):
        raise ValueError("incomplete record or incompatible canonical/channel basis or mode order")

    counts = record["guide_counts"]
    if counts.keys() != guide_budgets.keys():
        raise ValueError("guide inventory differs from configured candidate budgets")
    channels = record["channels"]
    keys = [channel["key"] for channel in channels]
    if keys != sorted(set(keys)) or any(guide not in counts for guide, _ in keys):
        raise ValueError("channel IDs must be unique and ordered by guide and FDE mode number")
    for guide, budget in guide_budgets.items():
        stats = counts[guide]
        returned = stats["returned_count"]
        selected = [number for key, number in keys if key == guide]
        if (stats["requested_count"] != budget or type(returned) is not int or returned < 0
                or stats["guided_count"] != len(selected)
                or any(type(number) is not int or not 1 <= number <= returned for number in selected)):
            raise ValueError(f"invalid channel inventory for guide {guide!r}")

    k, m = len(mode_ids), len(keys)
    _check_products(record["power_products"], {"G_U": (k, k), "G_V": (m, m), "C": (k, m)}, "power")
    for field in ("E", "H"):
        _check_products(record["field_products"][field], {"S_U": (k, k), "S_V": (m, m), "X": (k, m)}, field)


def validate_channel_linkage(data, manifest):
    """Match a complete channel document to the canonical generation and node bases."""
    metadata = data["dataset_metadata"]
    for name in ("generation_id", "settings", "section_model"):
        if metadata.get(name) != manifest[name]:
            raise ValueError(f"channel {name} differs from the canonical generation")
    if metadata.get("field_convention") != FIELD_CONVENTION:
        raise ValueError("incompatible channel field convention")
    if ExportGrid(**metadata["export_grid"]) != ExportGrid(**manifest["export_grid"]):
        raise ValueError("channel export grid differs from the canonical generation")
    if data.get("generation_completion") != "complete":
        raise ValueError(f"channel generation {metadata['generation_id']} is incomplete")

    counts = manifest["settings"]["fde"]["candidate_counts"]
    guide_budgets = {guide: count for guide, count in counts.items() if guide != "coupled"}
    nodes = data["nodes"]
    for canonical in manifest["nodes"]:
        node = tuple(canonical["node"])
        try:
            validate_channel_node(nodes[node], canonical["basis_id"],
                                  tuple(canonical["fde_mode_numbers"]), guide_budgets)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid channel record at node {node}: {exc}") from exc
    if len(nodes) != len(manifest["nodes"]):
        raise ValueError("unexpected channel nodes outside the canonical grid")


def load_channel_data(dataset, section_model):
    """Load complete node products from a trusted local dataset."""
    data, payload = _read(dataset.data_directory)
    digest = sha256(payload).hexdigest()
    del payload
    manifest = getattr(dataset, "canonical_manifest", None)
    if manifest is None or manifest.get("status") != "complete":
        raise ValueError(f"Channel data requires a complete canonical dataset: {dataset.data_directory}")
    section_model.validate_configuration(manifest["section_model"])
    try:
        validate_channel_linkage(data, manifest)
        if digest != manifest.get("channel_data_sha256"):
            raise ValueError("channel file fingerprint differs from the canonical manifest")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid channel data in {dataset.data_directory}: {exc}") from exc
    return data
