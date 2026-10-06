"""Small shared helpers for one-to-one overlap mode tracking."""

import numpy as np
from scipy.optimize import linear_sum_assignment


def match_modes_one_to_one(overlap, link_tolerance=0.5, unmatched=-500):
    """Return predecessor indices by next-mode column, or ``unmatched``.

    Use the existing absolute-overlap score and strict tolerance. Dummy
    assignments allow modes without a valid predecessor to remain unlinked.
    The score is an overlap magnitude, not a normalized probability.
    """
    magnitude = np.abs(np.asarray(overlap))
    if magnitude.ndim != 2 or not np.all(np.isfinite(magnitude)):
        raise ValueError("Mode overlap must be a finite two-dimensional matrix")
    previous_count, next_count = magnitude.shape
    links = np.full(next_count, unmatched, dtype=int)
    if not previous_count or not next_count:
        return links
    valid = magnitude > link_tolerance
    # Rows are next modes; real columns are predecessor modes. Each next mode
    # also has access to an unlinked dummy, so invalid links are never forced.
    scores = np.zeros((next_count, previous_count + next_count), dtype=float)
    scores[:, :previous_count] = np.where(valid.T, magnitude.T, -1.0)
    next_indices, previous_indices = linear_sum_assignment(scores, maximize=True)
    for next_index, previous_index in zip(next_indices, previous_indices):
        if previous_index < previous_count:
            links[next_index] = previous_index
    return links


def validate_tracking_mode_names(names, allow_unmatched=False):
    """Reject duplicate valid IDs before any modal-data assignment can overwrite."""
    names = np.asarray(names)
    if names.ndim not in (1, 2) or not np.issubdtype(names.dtype, np.integer):
        raise ValueError("Tracking mode names must be a one- or two-dimensional integer array")
    rows = names[None, :] if names.ndim == 1 else names
    for section, row in enumerate(rows):
        if not allow_unmatched and np.any(row < 0):
            raise ValueError(f"Negative tracking mode name in section {section}")
        present = row[row >= 0]
        if len(np.unique(present)) != len(present):
            raise ValueError(f"Duplicate tracking mode names in section {section}")
