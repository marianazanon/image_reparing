# src/main.py

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.pipeline import Pipeline
from utils.image_io import build_output_path, write_image

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Image restoration with Canny + morphology + inpainting (OpenCV)."
    )

    p.add_argument("input_path", type=Path, help="Path to the input image.")

    p.add_argument("-o", "--output-path", type=Path, default=None,
                   help="Exact path to write the restored image.")
    p.add_argument("-d", "--output-dir", type=Path, default=None,
                   help="Directory for the restored image if --output-path is not set.")
    p.add_argument("--force-ext", type=str, default=None,
                   help="Override output extension (e.g., .png, .jpg).")

    p.add_argument("--save-gray", action="store_true",
                   help="Also save the preprocessed grayscale image (as *_gray.png).")
    p.add_argument("--save-mask", action="store_true",
                   help="Also save the generated binary mask (as *_mask.png).")

    p.add_argument("--resize-scale", type=float, default=None,
                   help="Uniform scale factor (e.g., 0.5). If set, overrides --resize-long-edge.")
    p.add_argument("--resize-long-edge", type=int, default=None,
                   help="Resize so the longer side becomes this value (px).")
    p.add_argument("--blur-ksize", type=int, default=0,
                   help="Optional Gaussian blur kernel size (odd; 0 disables).")

    p.add_argument("--canny-low", type=int, default=50, help="Canny lower threshold.")
    p.add_argument("--canny-high", type=int, default=150, help="Canny upper threshold (must be > low).")
    p.add_argument("--morph-kernel", type=int, default=3, help="Morphology kernel size (odd, >=3).")
    p.add_argument("--dilate-iters", type=int, default=1, help="Dilate iterations to thicken edges.")
    p.add_argument("--close-iters", type=int, default=1, help="Closing iterations to connect gaps.")
    p.add_argument("--post-erode-iters", type=int, default=0, help="Erode iterations after closing.")
    p.add_argument("--median-ksize", type=int, default=0,
                   help="Median blur kernel size for mask denoise (odd; 0 disables).")

    p.add_argument("-m", "--method", type=str, default="telea",
                   choices=["telea", "ns", "navier-stokes"],
                   help="Inpainting method (default: telea).")
    p.add_argument("-r", "--radius", type=float, default=3.0,
                   help="Inpainting neighborhood radius in pixels (default: 3.0).")

    return p

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    pipeline = Pipeline(
        resize_scale=args.resize_scale,
        resize_long_edge=args.resize_long_edge,
        blur_ksize=args.blur_ksize,
        canny_low=args.canny_low,
        canny_high=args.canny_high,
        morph_kernel_size=args.morph_kernel,
        dilate_iters=args.dilate_iters,
        close_iters=args.close_iters,
        post_erode_iters=args.post_erode_iters,
        median_ksize=args.median_ksize,
        inpaint_method=args.method,
        inpaint_radius=args.radius,
    )

    written_path, gray_proc, mask = pipeline.run(
        input_path=args.input_path,
        output_path=args.output_path,
        output_dir=args.output_dir,
        force_ext=args.force_ext,
        return_intermediates=(args.save_gray or args.save_mask),
    )

    print(f"[OK] Restored image written to: {written_path}")

    if args.save_gray and gray_proc is not None:
        gray_out = build_output_path(args.input_path, args.output_dir or written_path.parent,
                                     suffix="_gray", force_ext=".png")
        write_image(gray_out, gray_proc)
        print(f"[OK] Preprocessed gray saved to: {gray_out}")

    if args.save_mask and mask is not None:
        mask_out = build_output_path(args.input_path, args.output_dir or written_path.parent,
                                     suffix="_mask", force_ext=".png")
        write_image(mask_out, mask)
        print(f"[OK] Mask saved to: {mask_out}")

if __name__ == "__main__":
    main()
