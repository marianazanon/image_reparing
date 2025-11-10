from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from src.utils.image_io import ensure_3ch, ensure_uint8


@dataclass
class PreprocessConfig:
    """Config knobs for the pre-clean stage."""
    enable_deblock: bool = True
    deblock_sigma: float = 8.0          # for DCT deblocking (xphoto)
    deblock_psize: int = 8

    enable_denoise: bool = True
    denoise_method: str = "nlm"         # "nlm" | "bilateral"
    nlm_h_color: float = 7.0            # strength for NLMeans chroma
    nlm_h_luma: float = 7.0             # strength for NLMeans luma
    nlm_template: int = 7
    nlm_search: int = 21

    bilateral_d: int = 5
    bilateral_sigma_color: float = 35.0
    bilateral_sigma_space: float = 7.0

    enable_sharpen: bool = True
    usm_amount: float = 0.6             # 0..2 typical
    usm_radius: float = 1.2             # Gaussian sigma (pixels)
    usm_threshold: int = 4              # edge mask threshold (0..255)

    enable_upscale: bool = False
    upscale_method: str = "bicubic"     # "bicubic" | "dnn"
    upscale_scale: float = 2.0
    sr_model_path: Optional[str] = None # e.g. models/EDSR_x2.pb
    sr_model_name: str = "edsr"         # "edsr" | "espcn" | "fsrcnn" | "lapsrn"


class FacePreprocessor:
    """
    Applies a configurable chain of classical OpenCV operations to stabilize
    a face crop before feeding it to the GAN prior.
    """

    def __init__(self, cfg: Optional[PreprocessConfig] = None) -> None:
        self.cfg = cfg or PreprocessConfig()
        self._sr = None

    def run(self, crop_rgb: np.ndarray) -> np.ndarray:
        """
        Apply pre-cleaning to a face crop (RGB uint8 preferred).
        Returns an RGB uint8 image.
        """
        img = ensure_3ch(ensure_uint8(crop_rgb))

        if self.cfg.enable_deblock:
            img = self._deblock(img)

        if self.cfg.enable_denoise:
            img = self._denoise(img)

        if self.cfg.enable_sharpen:
            img = self._unsharp_mask(img)

        if self.cfg.enable_upscale and self.cfg.upscale_scale and self.cfg.upscale_scale != 1.0:
            img = self._upscale(img)

        return img

    def _deblock(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        JPEG deblocking using xphoto.dctDenoising when available, with a
        robust fallback when xphoto is not present.
        """
        if hasattr(cv2, "xphoto") and hasattr(cv2.xphoto, "dctDenoising"):
            out = np.empty_like(img_rgb)
            cv2.xphoto.dctDenoising(
                src=img_rgb, dst=out,
                sigma=float(self.cfg.deblock_sigma),
                psize=int(self.cfg.deblock_psize),
            )
            return out

        ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y = cv2.bilateralFilter(y, d=5, sigmaColor=15, sigmaSpace=5)
        cr = cv2.medianBlur(cr, 3)
        cb = cv2.medianBlur(cb, 3)
        out = cv2.merge([y, cr, cb])
        return cv2.cvtColor(out, cv2.COLOR_YCrCb2RGB)

    def _denoise(self, img_rgb: np.ndarray) -> np.ndarray:
        """Non-local means (default) or bilateral denoising."""
        method = self.cfg.denoise_method.lower()
        if method == "nlm":
            # OpenCV's fastNlMeansDenoisingColored expects BGR; convert there and back.
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            den = cv2.fastNlMeansDenoisingColored(
                src=bgr,
                h=self.cfg.nlm_h_luma,
                hColor=self.cfg.nlm_h_color,
                templateWindowSize=int(self.cfg.nlm_template),
                searchWindowSize=int(self.cfg.nlm_search),
            )
            return cv2.cvtColor(den, cv2.COLOR_BGR2RGB)

        elif method == "bilateral":
            return cv2.bilateralFilter(
                img_rgb,
                d=int(self.cfg.bilateral_d),
                sigmaColor=float(self.cfg.bilateral_sigma_color),
                sigmaSpace=float(self.cfg.bilateral_sigma_space),
            )

        return img_rgb

    def _unsharp_mask(self, img_rgb: np.ndarray) -> np.ndarray:
        blur = cv2.GaussianBlur(img_rgb, ksize=(0, 0), sigmaX=float(self.cfg.usm_radius))
        diff = cv2.absdiff(img_rgb, blur)
        if self.cfg.usm_threshold > 0:
            _, mask = cv2.threshold(
                cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY),
                thresh=int(self.cfg.usm_threshold),
                maxval=255,
                type=cv2.THRESH_BINARY,
            )
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
            mask = (mask > 0).astype(np.uint8)
        else:
            mask = np.ones_like(img_rgb, dtype=np.uint8)

        amount = float(self.cfg.usm_amount)
        sharpen = cv2.addWeighted(img_rgb, 1 + amount, blur, -amount, 0)
        out = (sharpen * mask + img_rgb * (1 - mask)).astype(np.uint8)
        return out

    def _upscale(self, img_rgb: np.ndarray) -> np.ndarray:
        """Upscale by bicubic or dnn_superres if a model is provided."""
        scale = float(self.cfg.upscale_scale)

        if self.cfg.upscale_method.lower() == "dnn" and self.cfg.sr_model_path:
            sr = self._get_sr()
            bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            up = sr.upsample(bgr)
            return cv2.cvtColor(up, cv2.COLOR_BGR2RGB)

        h, w = img_rgb.shape[:2]
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


    def _get_sr(self):
        if self._sr is not None:
            return self._sr

        if not hasattr(cv2, "dnn_superres"):
            raise RuntimeError(
                "OpenCV dnn_superres module not available. "
                "Install opencv-contrib-python and provide a .pb superres model."
            )

        sr = cv2.dnn_superres.DnnSuperResImpl_create()
        model_name = self.cfg.sr_model_name.lower()
        model_path = self.cfg.sr_model_path
        sr.readModel(model_path)

        scale = int(round(self.cfg.upscale_scale))
        sr.setModel(model_name, scale)
        self._sr = sr
        return self._sr
