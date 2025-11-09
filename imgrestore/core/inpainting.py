from __future__ import annotations

from typing import Optional
import cv2
import numpy as np


class Inpainter:
    _METHODS = {
        "telea": cv2.INPAINT_TELEA,
        "ns": cv2.INPAINT_NS,
        "navier-stokes": cv2.INPAINT_NS,
    }

    def __init__(self, method: str = "telea", radius: float = 3.0) -> None:
        method_key = method.strip().lower()
        if method_key not in self._METHODS:
            raise ValueError(f"Unsupported method '{method}'. Use one of: {list(self._METHODS.keys())}")

        if radius <= 0:
            raise ValueError("radius must be > 0.")

        self.method_flag = self._METHODS[method_key]
        self.method_name = method_key
        self.radius = float(radius)

    @staticmethod
    def _ensure_three_or_one_channel(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None.")

        if img.ndim == 2:
            out = img
        elif img.ndim == 3 and img.shape[2] == 3:
            out = img
        elif img.ndim == 3 and img.shape[2] == 4:
            out = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            raise ValueError("Image must be grayscale or BGR (1 or 3 channels).")

        if out.dtype != np.uint8:
            out = cv2.normalize(out, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        return out

    @staticmethod
    def _ensure_binary_mask(mask: np.ndarray, target_shape_hw: Optional[tuple[int, int]] = None) -> np.ndarray:
        if mask is None:
            raise ValueError("Mask is None.")

        if mask.ndim == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        if target_shape_hw is not None and (mask.shape[0] != target_shape_hw[0] or mask.shape[1] != target_shape_hw[1]):
            # Use nearest neighbor to avoid introducing fractional values
            mask = cv2.resize(mask, (target_shape_hw[1], target_shape_hw[0]), interpolation=cv2.INTER_NEAREST)

        if mask.dtype != np.uint8:
            mask = cv2.normalize(mask, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        out = np.zeros_like(mask, dtype=np.uint8)
        out[mask > 0] = 255
        return out

    def inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        img = self._ensure_three_or_one_channel(image)
        bin_mask = self._ensure_binary_mask(mask, target_shape_hw=img.shape[:2])

        restored = cv2.inpaint(img, bin_mask, self.radius, self.method_flag)
        return restored
