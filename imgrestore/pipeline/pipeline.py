# src/pipeline/pipeline.py

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
import numpy as np

from imgrestore.utils.image_io import read_image, write_image, build_output_path
from imgrestore.core.preprocessor import Preprocessor
from imgrestore.core.scratch_detector import ScratchDetector
from imgrestore.core.inpainting import Inpainter


class Pipeline:
    def __init__(
        self,
        resize_scale: Optional[float] = None,
        resize_long_edge: Optional[int] = None,
        blur_ksize: int = 0,
        canny_low: int = 50,
        canny_high: int = 150,
        morph_kernel_size: int = 3,
        dilate_iters: int = 1,
        close_iters: int = 1,
        post_erode_iters: int = 0,
        median_ksize: int = 0,

        inpaint_method: str = "telea",
        inpaint_radius: float = 3.0,
    ) -> None:
        self.pre = Preprocessor(
            scale=resize_scale,
            target_long_edge=resize_long_edge,
            blur_ksize=blur_ksize,
        )
        self.detector = ScratchDetector(
            canny_low=canny_low,
            canny_high=canny_high,
            morph_kernel_size=morph_kernel_size,
            dilate_iters=dilate_iters,
            close_iters=close_iters,
            post_erode_iters=post_erode_iters,
            median_ksize=median_ksize,
        )
        self.inpainter = Inpainter(
            method=inpaint_method,
            radius=inpaint_radius,
        )

    def run(
        self,
        input_path: str | Path,
        output_path: Optional[str | Path] = None,
        output_dir: Optional[str | Path] = None,
        force_ext: Optional[str] = None,
        return_intermediates: bool = False,
    ) -> Tuple[Path, Optional[np.ndarray], Optional[np.ndarray]]:
        bgr = read_image(input_path)

        bgr_proc = self.pre.run(bgr, want_gray=False)
        gray_proc = self.pre.to_gray(bgr_proc)

        mask = self.detector.build_mask(gray_proc)

        restored = self.inpainter.inpaint(bgr_proc, mask)

        if output_path is not None:
            out_path = Path(output_path)
        else:
            if output_dir is None:
                output_dir = Path(input_path).parent
            out_path = build_output_path(input_path, output_dir, suffix="_restored", force_ext=force_ext)

        written = write_image(out_path, restored)

        if return_intermediates:
            return written, gray_proc, mask
        return written, None, None
