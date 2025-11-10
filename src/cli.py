from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.core.detector import FaceDetector
from src.core.align import FaceCropper
from src.core.preprocess import FacePreprocessor, PreprocessConfig
from src.core.prior_gran import GANPriorRestorer, GANPriorConfig
from src.core.blender import FaceBlender, BlendConfig
from src.core.colorizer import ImageColorizer, ColorizerConfig
from src.core.pipeline import RestorationPipeline
from src.utils.image_io import imread, imwrite, list_images


class _NoOpPrior:
    """Fallback when GFPGAN isn't available and user allows skipping the GAN step."""
    def restore(self, crop_rgb: np.ndarray) -> np.ndarray:
        return crop_rgb


def _apply_preset(args: argparse.Namespace) -> None:
    """Apply preset configurations to args if not already set by user."""
    if args.preset == "standard":
        # Keep existing defaults
        return
    
    elif args.preset == "enhanced":
        # Enhanced preset for better clarity and quality
        if args.usm_amount == 0.6:  # If still default
            args.usm_amount = 1.2
        if args.usm_radius == 1.2:
            args.usm_radius = 1.5
        if args.usm_threshold == 4:
            args.usm_threshold = 2
        if args.nlm_h_color == 7.0:
            args.nlm_h_color = 10.0
        if args.nlm_h_luma == 7.0:
            args.nlm_h_luma = 10.0
        if args.prior_upscale == 1:
            args.prior_upscale = 2
        if args.blend_method == "poisson_mixed":
            args.blend_method = "feather"
        if args.feather_frac == 0.12:
            args.feather_frac = 0.08
        if args.det_conf == 0.6:
            args.det_conf = 0.5
        if args.jpg_quality == 95:
            args.jpg_quality = 98
    
    elif args.preset == "maximum":
        # Maximum quality preset (slower)
        if args.usm_amount == 0.6:
            args.usm_amount = 1.5
        if args.usm_radius == 1.2:
            args.usm_radius = 2.0
        if args.usm_threshold == 4:
            args.usm_threshold = 1
        if args.nlm_h_color == 7.0:
            args.nlm_h_color = 12.0
        if args.nlm_h_luma == 7.0:
            args.nlm_h_luma = 12.0
        if args.prior_upscale == 1:
            args.prior_upscale = 4
        if args.blend_method == "poisson_mixed":
            args.blend_method = "feather"
        if args.feather_frac == 0.12:
            args.feather_frac = 0.06
        if args.det_conf == 0.6:
            args.det_conf = 0.4
        if args.jpg_quality == 95:
            args.jpg_quality = 100


def _build_pipeline(args: argparse.Namespace) -> RestorationPipeline:
    """Instantiate all components using CLI args and return a ready pipeline."""
    # Detector (OpenCV DNN SSD)
    detector = FaceDetector(
        prototxt_path=args.detector_prototxt,
        weights_path=args.detector_weights,
        conf_threshold=args.det_conf,
        nms_threshold=args.det_nms,
        min_size=args.det_min,
        use_cuda=args.det_cuda,
    )

    # Cropper (square + padding)
    cropper = FaceCropper(
        padding_ratio=args.pad_ratio,
        enforce_square=not args.no_square,
    )

    # Preprocess (OpenCV classical chain)
    pre_cfg = PreprocessConfig(
        enable_deblock=not args.no_deblock,
        deblock_sigma=args.deblock_sigma,
        deblock_psize=args.deblock_psize,

        enable_denoise=not args.no_denoise,
        denoise_method=args.denoise_method,
        nlm_h_color=args.nlm_h_color,
        nlm_h_luma=args.nlm_h_luma,
        nlm_template=args.nlm_template,
        nlm_search=args.nlm_search,
        bilateral_d=args.bilateral_d,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,

        enable_sharpen=not args.no_sharpen,
        usm_amount=args.usm_amount,
        usm_radius=args.usm_radius,
        usm_threshold=args.usm_threshold,

        enable_upscale=args.upscale > 1.0 and not args.no_upscale,
        upscale_method=args.upscale_method,
        upscale_scale=args.upscale,
        sr_model_path=args.sr_model_path,
        sr_model_name=args.sr_model_name,
    )
    preproc = FacePreprocessor(pre_cfg)

    # Prior (GFPGAN)
    prior = None
    if args.skip_gan:
        prior = _NoOpPrior()
    else:
        try:
            prior_cfg = GANPriorConfig(
                model_path=args.gfpgan_model,
                arch=args.gfpgan_arch,
                channel_multiplier=args.gfpgan_chmul,
                upscale=max(1, int(round(args.prior_upscale))),
                only_center_face=True,
                has_aligned=False,
                device=args.device,
            )
            prior = GANPriorRestorer(prior_cfg)
        except Exception as e:
            if args.allow_no_gan:
                logging.warning("GFPGAN unavailable (%s). Proceeding with No-Op prior.", e)
                prior = _NoOpPrior()
            else:
                raise

    # Blender
    blend_cfg = BlendConfig(
        method=args.blend_method,
        feather_frac=args.feather_frac,
        feather_min_px=args.feather_min_px,
        mask_erode_px=args.mask_erode_px,
        fallback_to_feather=True,
    )
    blender = FaceBlender(blend_cfg)

    # Colorizer
    colorizer_cfg = ColorizerConfig(
        method=args.colorize_method,
        enable=not args.no_colorize,
    )
    colorizer = ImageColorizer(colorizer_cfg)

    # Pipe
    pipeline = RestorationPipeline(
        detector=detector,
        cropper=cropper,
        preproc=preproc,
        prior=prior,  # type: ignore[arg-type]
        blender=blender,
        colorizer=colorizer,
        max_faces=args.max_faces,
        process_order=args.order,
    )
    return pipeline


def _dest_path(in_path: Path, in_root: Optional[Path], out_root: Path, mirror: bool) -> Path:
    """Compute output path for a given input image path."""
    if mirror and in_root is not None:
        try:
            rel = in_path.relative_to(in_root)
        except Exception:
            rel = in_path.name
        out_path = out_root / rel
        return out_path
    else:
        return out_root / in_path.name


def _save_diags(diags_json_path: Path, data: Dict) -> None:
    diags_json_path.parent.mkdir(parents=True, exist_ok=True)
    with diags_json_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="face-restore",
        description="Blind Face Restoration (OpenCV pre-clean + GFPGAN prior + controlled blending).",
    )

    # Preset
    p.add_argument(
        "--preset",
        choices=["standard", "enhanced", "maximum"],
        default="enhanced",
        help="Quality preset: standard (fast), enhanced (recommended), maximum (slow but best quality).",
    )

    # I/O
    p.add_argument("--input", "-i", required=True, help="Path to image file or directory.")
    p.add_argument("--output", "-o", required=True, help="Output directory.")
    p.add_argument("--mirror", action="store_true", help="Mirror input directory structure under output.")
    p.add_argument("--jpg-quality", type=int, default=95, help="JPEG save quality (0-100).")
    p.add_argument("--save-diags", action="store_true", help="Save per-image diagnostics as JSON next to outputs.")

    # Device
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Torch device for GFPGAN.")

    # Detector paths (optional; auto-discovery works if left unset)
    p.add_argument("--detector-prototxt", default=None, help="Path to deploy.prototxt for face detector.")
    p.add_argument("--detector-weights", default=None, help="Path to res10_300x300_ssd_iter_140000(.caffemodel).")

    # Detector knobs
    p.add_argument("--det-conf", type=float, default=0.6, help="Detector confidence threshold.")
    p.add_argument("--det-nms", type=float, default=0.3, help="Detector NMS IoU threshold.")
    p.add_argument("--det-min", type=int, default=24, help="Minimum face size (pixels).")
    p.add_argument("--det-cuda", action="store_true", help="Use CUDA DNN backend for detector (if available).")

    # Cropper
    p.add_argument("--pad-ratio", type=float, default=0.25, help="Padding as fraction of square side.")
    p.add_argument("--no-square", action="store_true", help="Do not force square crops (not recommended).")

    # Preprocess
    p.add_argument("--no-deblock", action="store_true", help="Disable JPEG deblocking.")
    p.add_argument("--deblock-sigma", type=float, default=8.0)
    p.add_argument("--deblock-psize", type=int, default=8)

    p.add_argument("--no-denoise", action="store_true", help="Disable denoising.")
    p.add_argument("--denoise-method", choices=["nlm", "bilateral"], default="nlm")
    p.add_argument("--nlm-h-color", type=float, default=7.0)
    p.add_argument("--nlm-h-luma", type=float, default=7.0)
    p.add_argument("--nlm-template", type=int, default=7)
    p.add_argument("--nlm-search", type=int, default=21)
    p.add_argument("--bilateral-d", type=int, default=5)
    p.add_argument("--bilateral-sigma-color", type=float, default=35.0)
    p.add_argument("--bilateral-sigma-space", type=float, default=7.0)

    p.add_argument("--no-sharpen", action="store_true", help="Disable unsharp mask sharpening.")
    p.add_argument("--usm-amount", type=float, default=0.6)
    p.add_argument("--usm-radius", type=float, default=1.2)
    p.add_argument("--usm-threshold", type=int, default=4)

    p.add_argument("--no-upscale", action="store_true", help="Disable upscaling even if scale > 1.")
    p.add_argument("--upscale", type=float, default=1.0, help="Scale factor (e.g., 2.0).")
    p.add_argument("--upscale-method", choices=["bicubic", "dnn"], default="bicubic")
    p.add_argument("--sr-model-path", default=None, help="Path to dnn_superres .pb model (e.g., EDSR_x2.pb).")
    p.add_argument("--sr-model-name", choices=["edsr", "espcn", "fsrcnn", "lapsrn"], default="edsr")

    # Prior (GFPGAN)
    p.add_argument("--gfpgan-model", default=None, help="Path to GFPGANv1.4.pth (optional; has fallbacks).")
    p.add_argument("--gfpgan-arch", default="clean", help="GFPGAN arch (default: clean).")
    p.add_argument("--gfpgan-chmul", type=int, default=2, help="GFPGAN channel multiplier.")
    p.add_argument("--prior-upscale", type=int, default=1, help="Face patch upscale inside GFPGAN (1 keeps size).")
    p.add_argument("--skip-gan", action="store_true", help="Skip GFPGAN and pass crops through unchanged.")
    p.add_argument("--allow-no-gan", action="store_true",
                   help="If GFPGAN import/weights fail, continue with a no-op prior instead of erroring.")

    # Blender
    p.add_argument("--blend-method", choices=["poisson_mixed", "poisson_normal", "feather"], default="poisson_mixed")
    p.add_argument("--feather-frac", type=float, default=0.12)
    p.add_argument("--feather-min-px", type=int, default=8)
    p.add_argument("--mask-erode-px", type=int, default=0)
    
    # Colorizer
    p.add_argument("--no-colorize", action="store_true", help="Disable automatic colorization of grayscale images.")
    p.add_argument("--colorize-method", choices=["opencv", "none"], default="opencv", help="Colorization method.")

    # Pipeline
    p.add_argument("--max-faces", type=int, default=None, help="Max faces per image (None = all).")
    p.add_argument("--order", choices=["score_desc", "left_to_right", "top_to_bottom"], default="score_desc")

    # Misc
    p.add_argument("--verbose", "-v", action="count", default=0, help="-v/-vv for more logs.")

    args = p.parse_args(argv)
    return args


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    
    # Apply preset configurations
    _apply_preset(args)

    # Logging level
    level = logging.WARNING
    if args.verbose == 1:
        level = logging.INFO
    elif args.verbose >= 2:
        level = logging.DEBUG
    logging.basicConfig(format="%(levelname)s: %(message)s", level=level)

    in_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.output).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gather images
    if in_path.is_dir():
        images = list_images(in_path)
        in_root = in_path
    else:
        images = [in_path]
        in_root = None

    if not images:
        logging.error("No images found under: %s", in_path)
        return

    # Build pipeline once
    pipeline = _build_pipeline(args)
    logging.info("Pipeline ready. Processing %d image(s)...", len(images))

    # Process
    for img_path in images:
        try:
            img = imread(img_path, as_rgb=True)
            out_img, diags = pipeline.run(img)

            # Save output image
            out_path = _dest_path(img_path, in_root, out_dir, args.mirror)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            imwrite(out_path, out_img, is_rgb=True, quality=args.jpg_quality)

            # Optional diagnostics JSON
            if args.save_diags:
                diag_path = out_path.with_suffix(out_path.suffix + ".json")
                _save_diags(diag_path, {
                    "input": str(img_path),
                    "output": str(out_path),
                    "diagnostics": asdict(diags),
                })

            logging.info("Processed: %s -> %s", img_path.name, out_path.name)

        except Exception as e:
            logging.error("Failed on %s: %s", img_path, e)


if __name__ == "__main__":
    main()
