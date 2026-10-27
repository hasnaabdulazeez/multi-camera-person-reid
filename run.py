"""Track people across several cameras and give each person one global ID.

    python run.py --sources cam1.mp4 cam2.mp4 cam3.mp4
    python run.py --sources 0 rtsp://user:pass@192.168.1.20/stream --detector yolov8s.engine

Outputs an annotated video per camera plus ``tracks.csv`` with one row per
person per frame (camera, frame, local track id, global id, box).
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import cv2
import numpy as np

from reid.gallery import GlobalIdentityGallery
from reid.models import PersonDetector, ReIDEmbedder
from reid.tracker import Tracker


def color_for(gid: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(gid)
    return tuple(int(c) for c in rng.integers(60, 255, 3))


def mosaic(frames: list[np.ndarray], tile_width: int = 640) -> np.ndarray:
    tiles = [cv2.resize(f, (tile_width, int(f.shape[0] * tile_width / f.shape[1]))) for f in frames]
    height = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, height - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    return np.hstack(tiles)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--detector", default="yolov8n.pt", help=".pt or TensorRT .engine")
    p.add_argument("--reid-model", default="osnet_x1_0")
    p.add_argument("--reid-weights", default=None)
    p.add_argument("--match-threshold", type=float, default=0.6)
    p.add_argument("--out", default="outputs")
    p.add_argument("--show", action="store_true")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    detector = PersonDetector(args.detector)
    embedder = ReIDEmbedder(args.reid_model, args.reid_weights)
    gallery = GlobalIdentityGallery(threshold=args.match_threshold)
    print(f"Re-ID backend: {embedder.backend}")

    cams = {}
    for i, src in enumerate(args.sources):
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise SystemExit(f"Cannot open {src}")
        cams[f"cam{i + 1}"] = {"cap": cap, "tracker": Tracker(), "writer": None,
                               "fps": cap.get(cv2.CAP_PROP_FPS) or 25}

    log = open(out / "tracks.csv", "w", newline="")
    writer = csv.writer(log)
    writer.writerow(["camera", "frame", "track_id", "global_id", "x0", "y0", "x1", "y1"])

    frame_idx, t0 = 0, time.time()
    while cams:
        frames = []
        for name in list(cams):
            cam = cams[name]
            ok, frame = cam["cap"].read()
            if not ok:
                cam["cap"].release()
                if cam["writer"]:
                    cam["writer"].release()
                del cams[name]
                continue

            dets = detector(frame)
            embs = embedder(frame, dets)
            tracks = cam["tracker"].update(dets, embs)
            ids = gallery.assign(name, [(t.track_id, t.embedding) for t in tracks])

            for t in tracks:
                gid = ids[t.track_id]
                x0, y0, x1, y1 = t.box.astype(int)
                cv2.rectangle(frame, (x0, y0), (x1, y1), color_for(gid), 2)
                cv2.putText(frame, f"ID {gid}", (x0, y0 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_for(gid), 2)
                writer.writerow([name, frame_idx, t.track_id, gid, x0, y0, x1, y1])
            cv2.putText(frame, name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if cam["writer"] is None:
                h, w = frame.shape[:2]
                cam["writer"] = cv2.VideoWriter(str(out / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                                cam["fps"], (w, h))
            cam["writer"].write(frame)
            frames.append(frame)

        if args.show and frames:
            cv2.imshow("Multi-camera Re-ID (q to quit)", mosaic(frames))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"frame {frame_idx}  {frame_idx / (time.time() - t0):.1f} it/s  identities={len(gallery.prototypes)}")

    log.close()
    cv2.destroyAllWindows()
    print(f"Identities: {len(gallery.prototypes)}; seen on 2+ cameras: {gallery.cross_camera_identities()}")


if __name__ == "__main__":
    main()
