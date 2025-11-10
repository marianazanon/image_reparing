from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

Box = Tuple[int, int, int, int]  # (x1, y1, x2, y2)

@dataclass(frozen=True)
class CropMeta:
    """
    Metadata describing how a crop was taken so we can restore it into the
    original image later.
    """
    orig_box: Box           # original detector box (as received)
    padded_box: Box         # square+padded box in image coords (x1,y1,x2,y2)
    inner_rel_box: Box      # orig_box mapped into crop coords (relative to padded crop)
    paste_xy: Tuple[int, int]  # top-left location (x1,y1) for pasting restored crop
    crop_shape: Tuple[int, int]  # (h, w) of the returned crop


class FaceCropper:

    def __init__(self, padding_ratio: float = 0.25, enforce_square: bool = True) -> None:
        if padding_ratio < 0.0:
            raise ValueError("padding_ratio must be >= 0.0")
        self.padding = float(padding_ratio)
        self.square = bool(enforce_square)

    def crop(self, image_rgb: np.ndarray, box: Box) -> Tuple[np.ndarray, CropMeta]:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("crop() expects an RGB image with shape HxWx3.")

        x1, y1, x2, y2 = self._sanitize_box(box)
        H, W = image_rgb.shape[:2]

        if self.square:
            x1, y1, x2, y2 = self._square_box(x1, y1, x2, y2)

        side = max(x2 - x1, y2 - y1)
        padded_side = int(round(side * (1.0 + 2.0 * self.padding)))

        max_side = min(W, H)
        padded_side = max(1, min(padded_side, max_side))

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        px1 = int(round(cx - padded_side * 0.5))
        py1 = int(round(cy - padded_side * 0.5))

        px1 = max(0, min(px1, W - padded_side))
        py1 = max(0, min(py1, H - padded_side))
        px2 = px1 + padded_side
        py2 = py1 + padded_side

        crop = image_rgb[py1:py2, px1:px2].copy()

        rel_x1 = x1 - px1
        rel_y1 = y1 - py1
        rel_x2 = x2 - px1
        rel_y2 = y2 - py1
        inner_rel = (int(rel_x1), int(rel_y1), int(rel_x2), int(rel_y2))

        meta = CropMeta(
            orig_box=(int(box[0]), int(box[1]), int(box[2]), int(box[3])),
            padded_box=(int(px1), int(py1), int(px2), int(py2)),
            inner_rel_box=inner_rel,
            paste_xy=(int(px1), int(py1)),
            crop_shape=(int(crop.shape[0]), int(crop.shape[1])),
        )
        return crop, meta


    @staticmethod
    def _sanitize_box(box: Box) -> Box:
        x1, y1, x2, y2 = map(int, box)
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return x1, y1, x2, y2

    @staticmethod
    def _square_box(x1: int, y1: int, x2: int, y2: int) -> Box:
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        side = int(round(max(x2 - x1, y2 - y1)))
        half = side * 0.5
        sx1 = int(round(cx - half))
        sy1 = int(round(cy - half))
        sx2 = sx1 + side
        sy2 = sy1 + side
        return sx1, sy1, sx2, sy2
