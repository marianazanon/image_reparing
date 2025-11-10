from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import cv2

try:
    import torch
    from gfpgan import GFPGANer  # type: ignore
except Exception as e:  # pragma: no cover
    _IMPORT_ERROR = e
    torch = None
    GFPGANer = None


@dataclass
class GANPriorConfig:
    """
    Configuration for the GFPGAN prior step.
    """
    model_path: Optional[str] = None      # e.g., "models/GFPGANv1.4.pth"
    arch: str = "clean"                   # "clean" is recommended for v1.4
    channel_multiplier: int = 2           # default for GFPGAN v1.x
    upscale: int = 1                      # keep crop size unless you want a larger face patch
    only_center_face: bool = True         # our crop normally contains a single centered face
    has_aligned: bool = False             # we pass rough crops, not eye-aligned faces
    device: str = "auto"                  # "auto" | "cpu" | "cuda"


class GANPriorRestorer:
    def __init__(self, cfg: Optional[GANPriorConfig] = None) -> None:
        self.cfg = cfg or GANPriorConfig()
        self.device = self._resolve_device(self.cfg.device)
        self.model_path = self._resolve_model_path(self.cfg.model_path)

        if GFPGANer is None or torch is None:
            raise RuntimeError(
                "GFPGAN/torch not available. Ensure 'gfpgan' and 'torch' are in requirements.txt "
                f"and installed correctly. Import error: {_IMPORT_ERROR}"
            )

        self.restorer = GFPGANer(
            model_path=str(self.model_path),
            upscale=int(self.cfg.upscale),
            arch=self.cfg.arch,
            channel_multiplier=int(self.cfg.channel_multiplier),
            bg_upsampler=None,
            device=self.device,
        )


    def restore(self, crop_rgb: np.ndarray) -> np.ndarray:
        """
        Restore a single face crop using GFPGAN.

        Args:
            crop_rgb: HxWx3 RGB uint8 face crop.

        Returns:
            restored_rgb: Hx' x W' x 3 RGB uint8 restored crop.
                          If GFPGAN fails to detect a face, returns the input crop unchanged.
        """
        if crop_rgb is None or crop_rgb.ndim != 3 or crop_rgb.shape[2] != 3:
            raise ValueError("restore() expects an RGB image with shape HxWx3.")
        if crop_rgb.dtype != np.uint8:
            # GFPGAN expects uint8 images in OpenCV BGR; we standardize here.
            crop_rgb = np.clip(crop_rgb, 0, 255).astype(np.uint8)

        # GFPGAN's API expects BGR images (OpenCV convention).
        crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)

        # paste_back=False because we want the restored face patch itself, not full-image compositing.
        # has_aligned=False: we give it a rough crop (not eye-aligned).
        try:
            _cropped, restored_faces, _restored_img = self.restorer.enhance(
                crop_bgr,
                has_aligned=bool(self.cfg.has_aligned),
                only_center_face=bool(self.cfg.only_center_face),
                paste_back=False,
            )
        except Exception:
            # Be resilient: on any runtime error, return the original crop.
            return crop_rgb

        if not restored_faces:
            # No face found inside the crop — return original crop.
            return crop_rgb

        # If only_center_face=True, we expect exactly one face in the list.
        restored_bgr = restored_faces[0]
        restored_rgb = cv2.cvtColor(restored_bgr, cv2.COLOR_BGR2RGB)
        return restored_rgb

    # ---------------- helpers ----------------

    @staticmethod
    def _resolve_device(pref: str) -> str:
        """Choose device based on preference and availability."""
        pref = (pref or "auto").lower()
        if pref == "cpu":
            return "cpu"
        if pref == "cuda":
            # If user forces cuda but it's unavailable, fall back to cpu.
            try:
                import torch  # local import to avoid top-level hard fail
                return "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                return "cpu"
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    @staticmethod
    def _resolve_model_path(model_path: Optional[str]) -> Path:
        """
        Resolve the GFPGAN weights path with fallbacks:
        1) user-specified path,
        2) <repo>/models/GFPGANv1.4.pth,
        3) package resource (gfpgan/weights/GFPGANv1.4.pth) if available.
        """
        if model_path:
            p = Path(model_path).expanduser().resolve()
            if p.exists():
                return p

        local = Path.cwd() / "models" / "GFPGANv1.4.pth"
        if local.exists():
            return local.resolve()

        try:
            import importlib.resources as pkg_res  # py3.9+
            with pkg_res.path("gfpgan", "weights") as wdir:
                p = (wdir / "GFPGANv1.4.pth")
                if p.exists():
                    return p.resolve()
        except Exception:
            pass

        # If still not found, provide a helpful error.
        raise FileNotFoundError(
            "GFPGAN model weights not found. Provide model_path in config or place "
            "'GFPGANv1.4.pth' under <repo>/models/. If installed via pip, make sure "
            "the 'gfpgan' package includes weights or specify the path explicitly."
        )
