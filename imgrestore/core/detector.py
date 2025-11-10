from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    """Single face detection result."""
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2) in original image coords
    score: float


class FaceDetector:
    """
    OpenCV DNN face detector (ResNet-SSD Caffe model).

    Expected model files (commonly distributed with OpenCV samples):
      - Prototxt:  'deploy.prototxt'  (network definition)
      - Weights:   'res10_300x300_ssd_iter_140000.caffemodel'  (or *_fp16.caffemodel)

    You can pass explicit paths, or place the files under:
      - <repo_root>/models/
      - <repo_root>/src/models/
    and the detector will try to auto-discover them.
    """

    def __init__(
        self,
        prototxt_path: Optional[str | Path] = None,
        weights_path: Optional[str | Path] = None,
        conf_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        min_size: int = 24,  # minimum face width/height (pixels) after decoding
        use_cuda: bool = False,
    ) -> None:
        self.conf_threshold = float(conf_threshold)
        self.nms_threshold = float(nms_threshold)
        self.min_size = int(min_size)

        if prototxt_path is None or weights_path is None:
            prototxt_path, weights_path = self._autodiscover_model_files()

        self.prototxt_path = Path(prototxt_path)
        self.weights_path = Path(weights_path)

        if not self.prototxt_path.exists():
            raise FileNotFoundError(f"Prototxt not found: {self.prototxt_path}")
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {self.weights_path}")

        self.net = cv2.dnn.readNetFromCaffe(
            str(self.prototxt_path), str(self.weights_path)
        )

        # Backend/target configuration. CUDA requires OpenCV built with CUDA.
        if use_cuda:
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            except Exception:
                # Fallback to CPU if CUDA not available in this build
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # ---------- public API ----------

    def detect(self, image_rgb: np.ndarray) -> List[Detection]:
        """
        Run face detection on an RGB image.

        Returns:
            List[Detection]: list of detections sorted by score (desc).
        """
        if image_rgb is None or image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("detect() expects an RGB image with shape HxWx3.")

        # Convert to BGR for OpenCV DNN preprocessing
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        h, w = image_bgr.shape[:2]

        # Standard SSD face detector preprocessing
        blob = cv2.dnn.blobFromImage(
            image_bgr, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        out = self.net.forward()  # shape: (1, 1, N, 7)

        raw_boxes: List[Tuple[int, int, int, int]] = []
        raw_scores: List[float] = []

        # Decode detections
        if out.ndim != 4 or out.shape[2] == 0 or out.shape[3] != 7:
            return []

        for i in range(out.shape[2]):
            _, class_id, score, x1, y1, x2, y2 = out[0, 0, i]
            # This model has only one class (face); we use the score directly.
            if score < self.conf_threshold:
                continue

            # Convert normalized coords to pixel coords and clamp
            x1i = max(0, min(int(round(x1 * w)), w - 1))
            y1i = max(0, min(int(round(y1 * h)), h - 1))
            x2i = max(0, min(int(round(x2 * w)), w - 1))
            y2i = max(0, min(int(round(y2 * h)), h - 1))

            # Ensure proper ordering
            x1i, x2i = min(x1i, x2i), max(x1i, x2i)
            y1i, y2i = min(y1i, y2i), max(y1i, y2i)

            # Filter by minimum face size
            if (x2i - x1i) < self.min_size or (y2i - y1i) < self.min_size:
                continue

            raw_boxes.append((x1i, y1i, x2i, y2i))
            raw_scores.append(float(score))

        if not raw_boxes:
            return []

        # Apply NMS (expects x,y,w,h)
        boxes_xywh = [
            (x1, y1, (x2 - x1), (y2 - y1)) for (x1, y1, x2, y2) in raw_boxes
        ]
        idxs = cv2.dnn.NMSBoxes(
            bboxes=boxes_xywh,
            scores=raw_scores,
            score_threshold=self.conf_threshold,
            nms_threshold=self.nms_threshold,
        )

        detections: List[Detection] = []
        if len(idxs) > 0:
            # idxs is either a list of indices or Nx1 array depending on OpenCV
            flat = [int(i) for i in np.array(idxs).reshape(-1).tolist()]
            for j in flat:
                detections.append(Detection(box=raw_boxes[j], score=raw_scores[j]))

        # Sort by score desc for convenience
        detections.sort(key=lambda d: d.score, reverse=True)
        return detections

    # ---------- helpers ----------

    def _autodiscover_model_files(self) -> Tuple[Path, Path]:
        """
        Try to find (prototxt, caffemodel) under common repo locations.
        """
        # Common filenames (you can use fp16 variant as well)
        proto_names = ["deploy.prototxt", "deploy_face.prototxt"]
        weight_names = [
            "res10_300x300_ssd_iter_140000.caffemodel",
            "res10_300x300_ssd_iter_140000_fp16.caffemodel",
        ]

        # Search roots: <repo>/models and <repo>/src/models relative to this file
        here = Path(__file__).resolve()
        roots = [
            here.parents[3] / "models",     # <repo_root>/models
            here.parents[2] / "models",     # <repo_root>/src/models (if layout differs)
            here.parent / ".." / ".." / "models",
        ]
        roots = [r.resolve() for r in roots if r is not None]

        protos: List[Path] = []
        weights: List[Path] = []
        for root in roots:
            if root.exists():
                for name in proto_names:
                    protos += list(root.rglob(name))
                for name in weight_names:
                    weights += list(root.rglob(name))

        if not protos or not weights:
            raise FileNotFoundError(
                "Face detector model files not found.\n"
                "Place 'deploy.prototxt' and 'res10_300x300_ssd_iter_140000.caffemodel' "
                "under a 'models/' directory in the repository or pass explicit paths "
                "to FaceDetector(...)."
            )

        # Pick the first found (deterministic ordering due to rglob + list)
        return protos[0], weights[0]
