"""Cross-camera identity gallery.

Each confirmed local track ``(camera, track_id)`` is linked to a *global* identity.
A new track is matched against the stored identity prototypes (mean appearance
embeddings) with the Hungarian algorithm, under two constraints:

* similarity must exceed a threshold, otherwise a new identity is created;
* one identity cannot appear twice in the same camera at the same time.

Once a track is linked, the link is kept for the track's lifetime and its
embedding keeps refining the identity prototype.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import cosine_similarity, l2_normalize


@dataclass
class GlobalIdentityGallery:
    threshold: float = 0.6
    momentum: float = 0.95
    prototypes: dict[int, np.ndarray] = field(default_factory=dict)
    links: dict[tuple[str, int], int] = field(default_factory=dict)
    sightings: dict[int, set[str]] = field(default_factory=dict)
    _next_id: int = 1

    def _new_identity(self, embedding: np.ndarray) -> int:
        gid = self._next_id
        self._next_id += 1
        self.prototypes[gid] = l2_normalize(embedding)
        self.sightings[gid] = set()
        return gid

    def _refine(self, gid: int, embedding: np.ndarray) -> None:
        self.prototypes[gid] = l2_normalize(
            self.momentum * self.prototypes[gid] + (1 - self.momentum) * l2_normalize(embedding)
        )

    def assign(self, camera: str, tracks: list[tuple[int, np.ndarray]]) -> dict[int, int]:
        """Map each ``(track_id, embedding)`` seen in ``camera`` this frame to a global id."""
        result: dict[int, int] = {}
        pending: list[tuple[int, np.ndarray]] = []
        for tid, emb in tracks:
            gid = self.links.get((camera, tid))
            if gid is None:
                pending.append((tid, emb))
            else:
                self._refine(gid, emb)
                result[tid] = gid

        busy = set(result.values())  # identities already visible in this camera now
        candidates = [g for g in self.prototypes if g not in busy]
        if pending and candidates:
            sim = cosine_similarity(np.array([e for _, e in pending]), np.array([self.prototypes[g] for g in candidates]))
            rows, cols = linear_sum_assignment(-sim)
            for r, c in zip(rows, cols):
                if sim[r, c] >= self.threshold:
                    tid, emb = pending[r]
                    result[tid] = candidates[c]
                    self._refine(candidates[c], emb)

        for tid, emb in pending:
            if tid not in result:
                result[tid] = self._new_identity(emb)

        for tid, gid in result.items():
            self.links[(camera, tid)] = gid
            self.sightings[gid].add(camera)
        return result

    def cross_camera_identities(self) -> dict[int, list[str]]:
        """Identities that have been seen by more than one camera."""
        return {g: sorted(c) for g, c in self.sightings.items() if len(c) > 1}
