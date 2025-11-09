from __future__ import annotations

import os
from pathlib import Path
from typing import Optional
import cv2
import numpy as np


def ensure_parent_dir(path: str | Path) -> None:
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)


def build_output_path(
    input_path: str | Path,
    output_dir: str | Path,
    suffix: str = "_restored",
    force_ext: Optional[str] = None,
) -> Path:
    in_p = Path(input_path)
    out_dir = Path(output_dir)
    stem = in_p.stem  # 'photo' from 'photo.jpg'
    ext = (force_ext if force_ext is not None else in_p.suffix) or ".png"
    if not str(ext).startswith("."):
        ext = "." + str(ext)
    out_name = f"{stem}{suffix}{ext}"
    return out_dir / out_name

_VALID_READ_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_VALID_WRITE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def _check_read_path(path: str | Path) -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Expected a file but got a directory: {p}")
    if p.suffix.lower() not in _VALID_READ_EXTS:
        raise ValueError(f"Unsupported input extension '{p.suffix}'. "
                         f"Use one of: {sorted(_VALID_READ_EXTS)}")
    return p


def read_image(path: str | Path, flags: int = cv2.IMREAD_UNCHANGED) -> np.ndarray:
    p = _check_read_path(path)
    img = cv2.imread(str(p), flags)
    if img is None:
        raise ValueError(f"OpenCV failed to read image: {p}")
    return img


def _pick_imwrite_params(ext: str, quality_jpg: int = 95) -> list[int]:
    ext = ext.lower()
    if ext in (".jpg", ".jpeg"):
        q = max(0, min(100, int(quality_jpg)))
        return [int(cv2.IMWRITE_JPEG_QUALITY), q]
    return []


def write_image(path: str | Path, image: np.ndarray, quality_jpg: int = 95) -> Path:
    p = Path(path)
    if p.suffix.lower() not in _VALID_WRITE_EXTS:
        raise ValueError(f"Unsupported output extension '{p.suffix}'. "
                         f"Use one of: {sorted(_VALID_WRITE_EXTS)}")

    ensure_parent_dir(p)
    params = _pick_imwrite_params(p.suffix, quality_jpg)
    ok = cv2.imwrite(str(p), image, params)
    if not ok:
        raise IOError(f"Failed to write image to: {p}")
    return p
