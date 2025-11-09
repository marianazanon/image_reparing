from __future__ import annotations

from typing import Tuple, Optional
import cv2
import numpy as np


class ScratchDetector:
    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 3,
        dilate_iters: int = 0,
        close_iters: int = 0,
        post_erode_iters: int = 1,
        median_ksize: int = 3,
        use_blackhat: bool = True,
        blackhat_len: int = 15,
        blackhat_orientation: str = "vertical",
        blackhat_thresh: Optional[int] = None,
        blackhat_scale: float = 0.7,
        min_area: int = 20,
        max_area_frac: float = 0.05,
        min_aspect_ratio: float = 3.0,
    ) -> None:

        if canny_low < 0 or canny_high <= 0 or canny_high <= canny_low:
            raise ValueError("Invalid Canny thresholds: ensure 0 <= low < high.")

        if morph_kernel_size < 3 or morph_kernel_size % 2 == 0:
            raise ValueError("morph_kernel_size must be odd and >=3.")

        if median_ksize not in (0, 1) and (median_ksize < 3 or median_ksize % 2 == 0):
            raise ValueError("median_ksize must be 0 or odd >=3.")

        if blackhat_len < 3:
            raise ValueError("blackhat_len must be >=3.")

        if blackhat_orientation not in ("vertical", "horizontal"):
            raise ValueError("blackhat_orientation must be 'vertical' or 'horizontal'.")

        if not (0.0 <= blackhat_scale <= 1.0):
            raise ValueError("blackhat_scale must be in [0,1].")

        self.canny_low = canny_low
        self.canny_high = canny_high

        self.morph_kernel_size = morph_kernel_size
        self.dilate_iters = max(0, dilate_iters)
        self.close_iters = max(0, close_iters)
        self.post_erode_iters = max(0, post_erode_iters)
        self.median_ksize = 0 if median_ksize == 1 else median_ksize

        self.use_blackhat = use_blackhat
        self.blackhat_len = blackhat_len
        self.blackhat_orientation = blackhat_orientation
        self.blackhat_thresh = blackhat_thresh
        self.blackhat_scale = blackhat_scale

        self.min_area = max(1, min_area)
        self.max_area_frac = max(0.0, min(1.0, max_area_frac))
        self.min_aspect_ratio = max(1.0, min_aspect_ratio)

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
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Unsupported image shape.")
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return gray

    @staticmethod
    def _binarize(mask_like: np.ndarray) -> np.ndarray:
        out = np.zeros_like(mask_like, dtype=np.uint8)
        out[mask_like > 0] = 255
        return out

    def _blackhat_mask(self, gray: np.ndarray) -> np.ndarray:
        if not self.use_blackhat:
            return np.zeros_like(gray, dtype=np.uint8)

        if self.blackhat_orientation == "vertical":
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (1, self.blackhat_len))
        else:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (self.blackhat_len, 1))

        bh = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, k)

        if self.blackhat_thresh is None:
            _, th = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            thresh = max(0, min(255, int(self.blackhat_thresh)))
            _, th = cv2.threshold(bh, thresh, 255, cv2.THRESH_BINARY)

        return th

    def _filter_components(self, mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape[:2]
        max_area = int(self.max_area_frac * h * w)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros_like(mask, dtype=np.uint8)

        for i in range(1, num):  # skip background 0
            x, y, ww, hh, area = stats[i]
            if area < self.min_area or area > max_area:
                continue
            ar1 = (hh / max(1, ww))
            ar2 = (ww / max(1, hh))
            if max(ar1, ar2) < self.min_aspect_ratio:
                continue
            keep[labels == i] = 255

        return keep

    def build_mask(self, image: np.ndarray) -> np.ndarray:
        gray = self._to_gray(image)

        bh_mask = self._blackhat_mask(gray)

        edges = cv2.Canny(gray, self.canny_low, self.canny_high)

        fused = cv2.addWeighted(bh_mask, self.blackhat_scale, edges, 1.0 - self.blackhat_scale, 0)
        _, fused = cv2.threshold(fused, 0, 255, cv2.THRESH_BINARY)

        if self.median_ksize and self.median_ksize >= 3:
            fused = cv2.medianBlur(fused, self.median_ksize)
        if self.dilate_iters > 0:
            fused = cv2.dilate(fused, self.kernel, iterations=self.dilate_iters)
        if self.close_iters > 0:
            fused = cv2.morphologyEx(fused, cv2.MORPH_CLOSE, self.kernel, iterations=self.close_iters)
        if self.post_erode_iters > 0:
            fused = cv2.erode(fused, self.kernel, iterations=self.post_erode_iters)

        fused = self._filter_components(fused)

        return self._binarize(fused)

    def debug_steps(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = self._to_gray(image)
        bh = self._blackhat_mask(gray) if self.use_blackhat else np.zeros_like(gray)
        final_mask = self.build_mask(image)
        return gray, bh, final_mask
