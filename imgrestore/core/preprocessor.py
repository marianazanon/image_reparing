# src/core/preprocessor.py

from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class Preprocessor:

    def __init__(
        self,
        scale: Optional[float] = None,
        target_long_edge: Optional[int] = None,
        blur_ksize: int = 0,
    ) -> None:
        if scale is not None and scale <= 0:
            raise ValueError("scale must be > 0.")
        if target_long_edge is not None and target_long_edge <= 0:
            raise ValueError("target_long_edge must be > 0.")
        if blur_ksize not in (0, 1) and (blur_ksize < 3 or blur_ksize % 2 == 0):
            raise ValueError("blur_ksize must be 0 (off) or an odd integer >= 3.")

        self.scale = scale
        self.target_long_edge = target_long_edge
        self.blur_ksize = 0 if blur_ksize == 1 else blur_ksize

    @staticmethod
    def to_gray(image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")

        if image.ndim == 2:
            gray = image
        elif image.ndim == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif image.ndim == 3 and image.shape[2] == 4:
            # BGRA → GRAY
            gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
        else:
            raise ValueError("Unsupported image shape for grayscale conversion.")

        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    def _compute_new_size(self, h: int, w: int) -> Tuple[int, int]:
        if self.scale is not None:
            new_w = max(1, int(round(w * self.scale)))
            new_h = max(1, int(round(h * self.scale)))
            return new_w, new_h

        if self.target_long_edge is not None:
            long_edge = max(h, w)
            if long_edge == 0:
                return w, h
            factor = self.target_long_edge / float(long_edge)
            if abs(factor - 1.0) < 1e-6:
                return w, h
            new_w = max(1, int(round(w * factor)))
            new_h = max(1, int(round(h * factor)))
            return new_w, new_h

        return w, h  # unchanged

    @staticmethod
    def _pick_interpolation(orig_w: int, orig_h: int, new_w: int, new_h: int) -> int:
        downscale = new_w < orig_w or new_h < orig_h
        return cv2.INTER_AREA if downscale else cv2.INTER_CUBIC

    def resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        if image is None:
            raise ValueError("Input image is None.")

        h, w = image.shape[:2]
        new_w, new_h = self._compute_new_size(h, w)
        if new_w == w and new_h == h:
            return image

        interp = self._pick_interpolation(w, h, new_w, new_h)
        return cv2.resize(image, (new_w, new_h), interpolation=interp)

    def maybe_blur(self, image: np.ndarray) -> np.ndarray:
        if self.blur_ksize == 0:
            return image
        return cv2.GaussianBlur(image, (self.blur_ksize, self.blur_ksize), 0)

    def run(self, image: np.ndarray, want_gray: bool = False) -> np.ndarray:
        img = self.resize_if_needed(image)
        img = self.maybe_blur(img)

        if want_gray:
            img = self.to_gray(img)
        else:
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return img
