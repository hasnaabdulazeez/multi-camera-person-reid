"""Per-camera multi-object tracker.

Detections are associated to existing tracks with the Hungarian algorithm on a
cost that blends box overlap (IoU) and appearance (cosine distance between
Re-ID embeddings). Using appearance as well as IoU keeps identities stable
through short occlusions and when people cross paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import count

import numpy as np
from scipy.optimize import linear_sum_assignment

from .geometry import cosine_similarity, iou_matrix, l2_normalize


@dataclass
class Track:
    track_id: int
    box: np.ndarray
    confidence: float
    embedding: np.ndarray | None = None
    hits: int = 1
    misses: int = 0
    age: int = 1

    def update(self, box, confidence, embedding, momentum: float) -> None:
        self.box = np.asarray(box, dtype=float)
        self.confidence = float(confidence)
        if embedding is not None:
            emb = l2_normalize(embedding)
            self.embedding = emb if self.embedding is None else l2_normalize(
                momentum * self.embedding + (1 - momentum) * emb
            )
        self.hits += 1
        self.misses = 0


@dataclass
class TrackerConfig:
    iou_weight: float = 0.5          # 1.0 = IoU only, 0.0 = appearance only
    max_cost: float = 0.7            # reject matches above this blended cost
    min_iou: float = 0.1             # gate: never match boxes that barely overlap
    min_hits: int = 3                # frames before a track is reported
    max_misses: int = 30             # frames a lost track is kept alive
    embedding_momentum: float = 0.9


@dataclass
class Tracker:
    config: TrackerConfig = field(default_factory=TrackerConfig)

    def __post_init__(self) -> None:
        self.tracks: list[Track] = []
        self._ids = count(1)

    def _cost(self, boxes: np.ndarray, embeddings: np.ndarray | None) -> np.ndarray:
        track_boxes = np.array([t.box for t in self.tracks]).reshape(-1, 4)
        iou = iou_matrix(track_boxes, boxes)
        cost = 1.0 - iou
        if embeddings is not None and all(t.embedding is not None for t in self.tracks):
            appearance = 1.0 - cosine_similarity(np.array([t.embedding for t in self.tracks]), embeddings)
            cost = self.config.iou_weight * cost + (1 - self.config.iou_weight) * appearance
        cost[iou < self.config.min_iou] = 1e6
        return cost

    def update(self, detections: np.ndarray, embeddings: np.ndarray | None = None) -> list[Track]:
        """Update with detections (N, 5) = x0, y0, x1, y1, conf. Returns confirmed active tracks."""
        dets = np.asarray(detections, dtype=float).reshape(-1, 5)
        boxes, confs = dets[:, :4], dets[:, 4]
        matched_tracks, matched_dets = set(), set()

        if self.tracks and len(dets):
            cost = self._cost(boxes, embeddings)
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] <= self.config.max_cost:
                    emb = None if embeddings is None else embeddings[c]
                    self.tracks[r].update(boxes[c], confs[c], emb, self.config.embedding_momentum)
                    matched_tracks.add(r)
                    matched_dets.add(c)

        for i, track in enumerate(self.tracks):
            track.age += 1
            if i not in matched_tracks:
                track.misses += 1
        self.tracks = [t for t in self.tracks if t.misses <= self.config.max_misses]

        for c in range(len(dets)):
            if c not in matched_dets:
                emb = None if embeddings is None else l2_normalize(embeddings[c])
                self.tracks.append(Track(next(self._ids), boxes[c], confs[c], emb))

        return [t for t in self.tracks if t.hits >= self.config.min_hits and t.misses == 0]
