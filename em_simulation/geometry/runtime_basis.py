"""Runtime basis for canonical datasets, using one transform in both directions."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuntimeBasisSection:
    """U_run = U @ R. R columns are forward slots, including absent slots."""

    node: tuple
    basis_id: str
    R: np.ndarray
    forward_present: np.ndarray
    forward_guided: np.ndarray

    @property
    def forward_slots(self):
        return np.flatnonzero(self.forward_present)

    @property
    def guided_slots(self):
        return np.flatnonzero(self.forward_guided)


def _bidirectional_transform(forward):
    """Conjugate the transform for reciprocal backward E/H."""
    n, k = forward.shape
    full = np.zeros((2 * n, 2 * k), dtype=forward.dtype)
    full[:n, :k] = forward
    full[n:, k:] = forward.conj()
    return full


def canonical_runtime_edges(overlap_ab, overlap_ba, selected, tracking_names, present):
    """Use the selected and tracked fields to transform both directed overlaps.

    The real sign of each linked forward overlap chooses the next section's
    gauge. The same signs are applied to both directions. For the existing
    unconjugated E/H product the matrix rule is transpose, not adjoint.
    """
    count = overlap_ab.shape[1] // 2
    slots = present.shape[1]
    transforms = np.zeros((len(tracking_names), count, slots))
    for R, names in zip(transforms, tracking_names):
        R[selected, names] = 1

    for i, edge in enumerate(overlap_ab):
        a, b = transforms[i:i + 2]
        linked = present[i] & present[i + 1]
        diagonal = np.einsum("mi,mn,ni->i", a, edge[:count, :count], b, optimize=True)
        b[:, linked & (diagonal.real < 0)] *= -1

    full = [_bidirectional_transform(R) for R in transforms]
    runtime_ab = np.stack([a.T @ o @ b for a, o, b in zip(full[:-1], overlap_ab, full[1:])])
    runtime_ba = np.stack([b.T @ o @ a for a, o, b in zip(full[:-1], overlap_ba, full[1:])])
    return transforms, runtime_ab, runtime_ba


def runtime_sections(path, basis_ids, transforms, present, radiation):
    """Freeze the actual basis and masks for consumers of EME output."""
    sections = []
    for node, R, p, rad in zip(path, transforms, present, radiation):
        arrays = (R.copy(), p.copy(), p & ~rad)
        for array in arrays:
            array.setflags(write=False)
        sections.append(RuntimeBasisSection(node, basis_ids[node], *arrays))
    return tuple(sections)
