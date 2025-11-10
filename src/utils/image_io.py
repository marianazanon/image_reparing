from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import List, Sequence, Tuple, Union

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def imread(path: Union[str, Path], as_rgb: bool = True) -> np.ndarray:
    """Read an image from disk. Returns RGB by default (OpenCV loads BGR)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Image not found: {p}")
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {p}")
    if as_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite(
    path: Union[str, Path],
    image: np.ndarray,
    is_rgb: bool = True,
    quality: int = 95,
    create_dirs: bool = True,
) -> None:
    p = Path(path)
    if create_dirs:
        p.parent.mkdir(parents=True, exist_ok=True)

    img = image
    if is_rgb and img.ndim == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    ext = p.suffix.lower()
    params = []
    if ext in {".jpg", ".jpeg"}:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(np.clip(quality, 0, 100))]
    elif ext == ".png":
        compression = max(0, min(9, 9 - round((quality / 100) * 9)))
        params = [cv2.IMWRITE_PNG_COMPRESSION, compression]

    ok = cv2.imwrite(str(p), img, params)
    if not ok:
        raise IOError(f"Failed to write image: {p}")


def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    if img_bgr.ndim == 3 and img_bgr.shape[2] == 3:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    elif img_bgr.ndim == 2:
        return cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    return img_bgr


def to_bgr(img_rgb: np.ndarray) -> np.ndarray:
    if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
        return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    elif img_rgb.ndim == 2:
        return cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2BGR)
    return img_rgb


def ensure_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def ensure_3ch(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.ndim == 3 and img.shape[2] == 4:
        # Drop alpha channel by converting BGRA->BGR->RGB
        bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if img.ndim == 3 and img.shape[2] == 3:
        return img
    raise ValueError(f"Unsupported image shape: {img.shape}")


def list_images(path: Union[str, Path]) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if not p.exists():
        raise FileNotFoundError(f"Path not found: {p}")

    imgs: List[Path] = []
    for ext in IMG_EXTS:
        imgs.extend(p.rglob(f"*{ext}"))
        imgs.extend(p.rglob(f"*{ext.upper()}"))
    # Deduplicate and sort for stable processing order
    return sorted(set(imgs))

def draw_boxes(
    img_rgb: np.ndarray,
    boxes: Sequence[Tuple[int, int, int, int]],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw rectangles (x1,y1,x2,y2) on a copy of an RGB image and return RGB."""
    out = to_bgr(img_rgb.copy())  # OpenCV draws in BGR
    for (x1, y1, x2, y2) in boxes:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness)
    return to_rgb(out)
