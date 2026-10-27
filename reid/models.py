"""Person detector (YOLOv8, optionally TensorRT) and Re-ID embedding extractors."""

from __future__ import annotations

import numpy as np
import torch
import torchvision.transforms.functional as TF
from ultralytics import YOLO


class PersonDetector:
    """YOLOv8 person detector. Pass a ``.engine`` file to run a TensorRT export."""

    def __init__(self, weights: str = "yolov8n.pt", conf: float = 0.4, imgsz: int = 640, device=None):
        self.model = YOLO(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        result = self.model.predict(frame, classes=[0], conf=self.conf, imgsz=self.imgsz,
                                    device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return np.zeros((0, 5))
        xyxy = result.boxes.xyxy.cpu().numpy()
        conf = result.boxes.conf.cpu().numpy()[:, None]
        return np.hstack([xyxy, conf])


class ReIDEmbedder:
    """Appearance embeddings for person crops.

    Uses OSNet from ``torchreid`` when installed (trained for person Re-ID on
    Market-1501/DukeMTMC). Otherwise it falls back to an ImageNet ResNet-50, which
    works for demos but separates identities much less reliably.
    """

    def __init__(self, model_name: str = "osnet_x1_0", weights: str | None = None, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            from torchreid.utils import FeatureExtractor

            self._extractor = FeatureExtractor(model_name=model_name, model_path=weights or "", device=self.device)
            self.backend = f"torchreid:{model_name}"
        except ImportError:
            from torchvision.models import ResNet50_Weights, resnet50

            net = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            net.fc = torch.nn.Identity()
            self._net = net.eval().to(self.device)
            self._extractor = None
            self.backend = "torchvision:resnet50"

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) == 0:
            return np.zeros((0, 512))
        h, w = frame.shape[:2]
        crops = []
        for x0, y0, x1, y1 in boxes[:, :4].astype(int):
            crop = frame[max(0, y0):min(h, y1), max(0, x0):min(w, x1), ::-1]  # BGR -> RGB
            crops.append(np.ascontiguousarray(crop if crop.size else np.zeros((256, 128, 3), np.uint8)))

        if self._extractor is not None:
            feats = self._extractor(crops)
        else:
            batch = torch.stack([
                TF.normalize(TF.resize(TF.to_tensor(c), [256, 128], antialias=True),
                             [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                for c in crops
            ]).to(self.device)
            feats = self._net(batch)
        feats = torch.nn.functional.normalize(feats, dim=1)
        return feats.cpu().numpy()
