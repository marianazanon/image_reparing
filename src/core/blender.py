from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

from src.core.align import CropMeta


@dataclass
class BlendConfig:
    """
    Controls how the restored face patch is merged back into the original image.
    """
    method: str = "poisson_mixed"   # "poisson_mixed" | "poisson_normal" | "feather"
    feather_frac: float = 0.12      # feather border as fraction of min(side)
    feather_min_px: int = 8         # minimum feather width in pixels
    mask_erode_px: int = 0          # (Poisson) optional erosion to avoid halos
    fallback_to_feather: bool = True


class FaceBlender:

    def __init__(self, cfg: BlendConfig | None = None) -> None:
        self.cfg = cfg or BlendConfig()


    def blend(self, full_rgb: np.ndarray, restored_crop_rgb: np.ndarray, meta: CropMeta) -> np.ndarray:
        if full_rgb.ndim != 3 or full_rgb.shape[2] != 3:
            raise ValueError("full_rgb must be HxWx3 RGB.")
        if restored_crop_rgb.ndim != 3 or restored_crop_rgb.shape[2] != 3:
            raise ValueError("restored_crop_rgb must be h'xw'x3 RGB.")

        x1, y1, x2, y2 = meta.padded_box
        H, W = full_rgb.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            return full_rgb

        target_w = x2 - x1
        target_h = y2 - y1
        patch = cv2.resize(restored_crop_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        method = (self.cfg.method or "poisson_mixed").lower()
        if method.startswith("poisson"):
            try:
                return self._seamless_clone(full_rgb, patch, (x1, y1, x2, y2), mixed=("mixed" in method))
            except Exception:
                if self.cfg.fallback_to_feather:
                    return self._feather_blend(full_rgb, patch, (x1, y1, x2, y2))
                raise
        else:
            return self._feather_blend(full_rgb, patch, (x1, y1, x2, y2))

    def _feather_blend(self, full_rgb: np.ndarray, patch_rgb: np.ndarray, box: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Simple alpha feathering along the box border:
          - builds a soft mask inside the box
          - composites: out = patch*alpha + full*(1-alpha)
        """
        x1, y1, x2, y2 = box
        H, W = full_rgb.shape[:2]
        out = full_rgb.copy().astype(np.float32)

        overlay = np.zeros_like(out, dtype=np.float32)
        alpha = np.zeros((H, W), dtype=np.float32)

        h, w = (y2 - y1), (x2 - x1)
        if h <= 0 or w <= 0:
            return full_rgb

        overlay[y1:y2, x1:x2] = patch_rgb.astype(np.float32)

        border = max(self.cfg.feather_min_px, int(round(min(h, w) * float(self.cfg.feather_frac))))
        inner_x1 = x1 + border
        inner_y1 = y1 + border
        inner_x2 = x2 - border
        inner_y2 = y2 - border
        if inner_x2 <= inner_x1 or inner_y2 <= inner_y1:
            # If the box is too small for the chosen border, just use a hard mask (no feather)
            alpha[y1:y2, x1:x2] = 1.0
        else:
            alpha[inner_y1:inner_y2, inner_x1:inner_x2] = 1.0
            # Gaussian blur radius proportional to border size (odd kernel)
            k = max(3, int(border * 2) | 1)
            alpha = cv2.GaussianBlur(alpha, (k, k), sigmaX=border * 0.5, borderType=cv2.BORDER_REPLICATE)

        alpha3 = np.repeat(alpha[..., None], 3, axis=2)
        out = overlay * alpha3 + out * (1.0 - alpha3)

        return np.clip(out + 0.5, 0, 255).astype(np.uint8)

    def _seamless_clone(self, full_rgb: np.ndarray, patch_rgb: np.ndarray, box: Tuple[int, int, int, int], mixed: bool) -> np.ndarray:
        """
        Poisson (seamless) cloning using OpenCV:
          - mask: all-ones inside the patch (optionally eroded to avoid halos)
          - center: center of the target box
          - flags: MIXED_CLONE (default) or NORMAL_CLONE
        """
        x1, y1, x2, y2 = box
        target_w = x2 - x1
        target_h = y2 - y1
        patch_rgb = cv2.resize(patch_rgb, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        src_bgr = cv2.cvtColor(patch_rgb, cv2.COLOR_RGB2BGR)
        dst_bgr = cv2.cvtColor(full_rgb, cv2.COLOR_RGB2BGR)

        mask = np.full((target_h, target_w), 255, dtype=np.uint8)
        if self.cfg.mask_erode_px > 0:
            k = max(1, int(self.cfg.mask_erode_px))
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            mask = cv2.erode(mask, kernel, iterations=1)

        center = (int(round((x1 + x2) * 0.5)), int(round((y1 + y2) * 0.5)))

        flags = cv2.MIXED_CLONE if mixed else cv2.NORMAL_CLONE
        blended_bgr = cv2.seamlessClone(src_bgr, dst_bgr, mask, center, flags)

        return cv2.cvtColor(blended_bgr, cv2.COLOR_BGR2RGB)
