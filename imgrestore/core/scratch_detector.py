from __future__ import annotations

from typing import Tuple
import cv2
import numpy as np


class ScratchDetector:

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 3,
        dilate_iters: int = 1,
        close_iters: int = 1,
        post_erode_iters: int = 0,
        median_ksize: int = 0,
    ) -> None:
        if canny_low < 0 or canny_high <= 0 or canny_high <= canny_low:
            raise ValueError("Invalid Canny thresholds: ensure 0 <= low < high.")
        if morph_kernel_size < 3 or morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be an odd integer >= 3.")
        if median_ksize not in (0, 1) and (median_ksize < 3 or median_ksize % 2 == 0):
            raise ValueError("median_ksize must be 0 (off) or an odd integer >= 3.")

        self.canny_low = canny_low
        self.canny_high = canny_high
        self.morph_kernel_size = morph_kernel_size
        self.dilate_iters = max(0, dilate_iters)
        self.close_iters = max(0, close_iters)
        self.post_erode_iters = max(0, post_erode_iters)
        self.median_ksize = median_ksize if median_ksize != 1 else 0

        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (self.morph_kernel_size, self.morph_kernel_size)
        )

    @staticmethod
    def _to_gray(img: np.ndarray) -> np.ndarray:
        if img is None:
            raise ValueError("Input image is None.")
        if img.ndim == 2:
            gray = img
        elif img.ndim == 3:
            #BGR input (OpenCV default)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image shape for grayscale conversion.")
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def _binarize(mask_like: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask_like, dtype=np.uint8)
        out[mask_like > 0] = 255
        return out

    def build_mask(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)

        #Canny
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        if self.dilate_iters > 0:
            edges = cv2.dilate(edges, self.kernel, iterations=self.dilate_iters)

        if self.close_iters > 0:
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)

        if self.median_ksize and self.median_ksize >= 3:
            edges = cv2.medianBlur(edges, self.median_ksize)

        if self.post_erode_iters > 0:
            edges = cv2.erode(edges, self.kernel, iterations=self.post_erode_iters)

        mask = self._binarize(edges)
        return mask

    def debug_steps(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = self._to_gray(image)
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        if self.dilate_iters > 0:
            edges = cv2.dilate(edges, self.kernel, iterations=self.dilate_iters)
        final_mask = self.build_mask(image)
        return gray, edges, final_mask
