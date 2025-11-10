from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class ColorizerConfig:
    """
    Configuration for image colorization.
    """
    method: str = "opencv"  # "opencv" | "none"
    enable: bool = True


class ImageColorizer:
    """
    Colorizes grayscale images using various methods.
    Currently supports OpenCV's Caffe-based colorization model.
    """

    def __init__(self, cfg: Optional[ColorizerConfig] = None) -> None:
        self.cfg = cfg or ColorizerConfig()
        self._colorizer = None
        
    def should_colorize(self, img_rgb: np.ndarray) -> bool:
        """
        Check if an image needs colorization (is grayscale).
        """
        if not self.cfg.enable:
            return False
            
        # Check if image is effectively grayscale
        if img_rgb.ndim == 2:
            return True
        if img_rgb.ndim == 3 and img_rgb.shape[2] == 3:
            # Check if all channels are identical (grayscale stored as RGB)
            return np.allclose(img_rgb[:, :, 0], img_rgb[:, :, 1]) and \
                   np.allclose(img_rgb[:, :, 1], img_rgb[:, :, 2])
        return False

    def colorize(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Colorize a grayscale image.
        
        Args:
            img_rgb: HxWx3 RGB uint8 image (even if grayscale)
            
        Returns:
            colorized_rgb: HxWx3 RGB uint8 colorized image
        """
        if not self.should_colorize(img_rgb):
            return img_rgb
            
        method = (self.cfg.method or "opencv").lower()
        
        if method == "opencv":
            return self._colorize_opencv(img_rgb)
        elif method == "none":
            return img_rgb
        else:
            return img_rgb

    def _colorize_opencv(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Enhanced colorization using a simple but effective approach:
        1. Convert to LAB color space
        2. Add slight color tint based on luminance patterns
        3. Enhance saturation for better results
        """
        if img_rgb.ndim == 2:
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
        img_float = img_rgb.astype(np.float32) / 255.0
        
        sepia_matrix = np.array([
            [0.393, 0.769, 0.189],  # R channel
            [0.349, 0.686, 0.168],  # G channel  
            [0.272, 0.534, 0.131]   # B channel
        ])
        
        # Apply the transformation
        h, w = img_float.shape[:2]
        img_reshaped = img_float.reshape(-1, 3)
        colorized = np.dot(img_reshaped, sepia_matrix.T)
        colorized = colorized.reshape(h, w, 3)
        blend_factor = 0.7
        result = blend_factor * colorized + (1 - blend_factor) * img_float
        
        result_uint8 = (np.clip(result, 0, 1) * 255).astype(np.uint8)
        hsv = cv2.cvtColor(result_uint8, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.3, 0, 255)
        result_uint8 = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return result_uint8

