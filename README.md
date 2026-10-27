# Multi-Camera Person Tracking & Re-Identification

Track people within each camera, then give the **same person the same global ID across cameras** using appearance embeddings. Useful for retail analytics, building security and footfall studies. The detector can run as a **TensorRT** engine for real-time throughput on NVIDIA GPUs.

## Architecture

```
            ┌──────────────── per camera ────────────────┐
frame ──► YOLOv8 person detector ──► OSNet Re-ID embeddings ──► Tracker (Hungarian on IoU + appearance)
                                                                      │ confirmed local tracks
                                                                      ▼
                                              Global identity gallery (cross-camera Hungarian matching)
                                                                      │
                                         annotated videos per camera + tracks.csv (camera, frame, local id, global id, box)
```

**Per-camera tracking** (`reid/tracker.py`) blends IoU and cosine distance into one cost matrix and solves it with the Hungarian algorithm. Appearance keeps IDs attached to the right person when two people cross, and lost tracks stay alive for a configurable number of frames to survive short occlusions. Each track keeps an exponential moving average of its embeddings, which is more robust than a single crop.

**Cross-camera re-identification** (`reid/gallery.py`) links each new local track to a stored identity prototype. It enforces two rules: similarity must pass a threshold, and **one identity can't appear twice in the same camera at once**. That constraint removes a common class of false merges. Links are kept for the lifetime of a track, and its embeddings keep refining the prototype.

## Quick start

```bash
pip install -r requirements.txt
pip install git+https://github.com/KaiyangZhou/deep-person-reid.git   # OSNet (recommended)

python run.py --sources cam1.mp4 cam2.mp4 --show
python run.py --sources 0 rtsp://192.168.1.20/stream --detector yolov8s.engine
```

Without `torchreid`, the embedder falls back to an ImageNet ResNet-50. That works for demos but separates people much less reliably than a Re-ID-trained model.

### TensorRT detector

```bash
yolo export model=yolov8s.pt format=engine half=True imgsz=640   # run on the target GPU
python run.py --sources cam1.mp4 cam2.mp4 --detector yolov8s.engine
```

## Tuning

| Parameter | Where | Effect |
|---|---|---|
| `--match-threshold` | gallery | Higher = fewer wrong merges across cameras, more duplicate IDs |
| `iou_weight` | tracker | Lower = rely more on appearance (crowded scenes) |
| `max_misses` | tracker | How long a lost person is remembered within a camera |
| `min_hits` | tracker | Frames before a track is reported, which suppresses flicker |

## Evaluation

For a quantitative check, convert `tracks.csv` to MOTChallenge format and score per-camera tracking with [TrackEval](https://github.com/JonathonLuiten/TrackEval) (HOTA, IDF1). Score cross-camera ID consistency on a multi-camera benchmark such as EPFL or WILDTRACK.

## Project structure

```
reid/
  geometry.py   # IoU and cosine similarity matrices
  tracker.py    # per-camera tracker (Hungarian, IoU + appearance, occlusion handling)
  gallery.py    # global identity gallery with same-camera exclusivity
  models.py     # YOLOv8 / TensorRT detector, OSNet or ResNet-50 embedder
run.py          # multi-source pipeline, mosaic view, CSV log, per-camera video output
tests/          # tracker and gallery unit tests (no GPU needed)
```

## Privacy

Person re-identification is a surveillance technology. Use it only where people have been informed, local law allows it, and footage and logs are retained no longer than necessary.

## License

MIT
