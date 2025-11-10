from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import time
import numpy as np

from src.core.detector import FaceDetector, Detection
from src.core.align import FaceCropper, CropMeta
from src.core.preprocess import FacePreprocessor
from src.core.prior_gran import GANPriorRestorer
from src.core.blender import FaceBlender
from src.utils.image_io import imread, imwrite


@dataclass(frozen=True)
class FaceDiagnostics:
    index: int
    score: float
    orig_box: Tuple[int, int, int, int]
    padded_box: Tuple[int, int, int, int]
    crop_shape: Tuple[int, int]                 # (h, w)
    t_detect_ms: float
    t_crop_ms: float
    t_pre_ms: float
    t_prior_ms: float
    t_blend_ms: float


@dataclass(frozen=True)
class PipelineDiagnostics:
    num_faces: int
    faces: List[FaceDiagnostics]
    t_total_ms: float



class RestorationPipeline:
    """
    Orchestrates the full face restoration flow:

        detect -> crop -> preprocess -> generative prior -> blend

    Usage:
        pipe = RestorationPipeline(detector, cropper, preproc, prior, blender)
        out_img, diags = pipe.run(image_rgb)
    """

    def __init__(
        self,
        detector: FaceDetector,
        cropper: FaceCropper,
        preproc: FacePreprocessor,
        prior: GANPriorRestorer,
        blender: FaceBlender,
        max_faces: Optional[int] = None,         # limit how many faces to process per image
        process_order: str = "score_desc",       # or "left_to_right", "top_to_bottom"
    ) -> None:
        self.detector = detector
        self.cropper = cropper
        self.preproc = preproc
        self.prior = prior
        self.blender = blender
        self.max_faces = max_faces
        self.process_order = process_order

    def run(self, image_rgb: np.ndarray) -> Tuple[np.ndarray, PipelineDiagnostics]:
        """
        Run the pipeline on a single RGB image.

        Returns:
            (restored_rgb, diagnostics)
        """
        t0 = time.perf_counter()

        td0 = time.perf_counter()
        detections = self.detector.detect(image_rgb)
        td1 = time.perf_counter()

        if not detections:
            return (
                image_rgb,
                PipelineDiagnostics(
                    num_faces=0,
                    faces=[],
                    t_total_ms=round(1000 * (time.perf_counter() - t0), 3),
                ),
            )

        dets_ordered = self._order_detections(detections)

        if self.max_faces is not None and self.max_faces >= 0:
            dets_ordered = dets_ordered[: self.max_faces]

        out_img = image_rgb.copy()
        faces_diags: List[FaceDiagnostics] = []

        for idx, det in enumerate(dets_ordered):
            # --- crop ---
            tc0 = time.perf_counter()
            crop, meta = self.cropper.crop(out_img, det.box)
            tc1 = time.perf_counter()

            tp0 = time.perf_counter()
            crop_pre = self.preproc.run(crop)
            tp1 = time.perf_counter()

            tg0 = time.perf_counter()
            crop_restored = self.prior.restore(crop_pre)
            tg1 = time.perf_counter()

            tb0 = time.perf_counter()
            out_img = self.blender.blend(out_img, crop_restored, meta)
            tb1 = time.perf_counter()

            faces_diags.append(
                FaceDiagnostics(
                    index=idx,
                    score=float(det.score),
                    orig_box=tuple(map(int, det.box)),
                    padded_box=meta.padded_box,
                    crop_shape=meta.crop_shape,
                    t_detect_ms=round(1000 * (td1 - td0), 3),
                    t_crop_ms=round(1000 * (tc1 - tc0), 3),
                    t_pre_ms=round(1000 * (tp1 - tp0), 3),
                    t_prior_ms=round(1000 * (tg1 - tg0), 3),
                    t_blend_ms=round(1000 * (tb1 - tb0), 3),
                )
            )

        diags = PipelineDiagnostics(
            num_faces=len(faces_diags),
            faces=faces_diags,
            t_total_ms=round(1000 * (time.perf_counter() - t0), 3),
        )
        return out_img, diags

    def _order_detections(self, dets: List[Detection]) -> List[Detection]:
        """Order faces by chosen strategy."""
        method = (self.process_order or "score_desc").lower()
        if method == "left_to_right":
            return sorted(dets, key=lambda d: d.box[0])
        if method == "top_to_bottom":
            return sorted(dets, key=lambda d: d.box[1])
        return sorted(dets, key=lambda d: d.score, reverse=True)