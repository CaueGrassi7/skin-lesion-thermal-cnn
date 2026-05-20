"""
Thermal image preprocessing pipeline.

Covers raw image loading, noise reduction, normalisation, pseudo-colour
mapping, and saving processed images ready for model input.
"""

from pathlib import Path
from typing import Optional

import numpy as np


def load_thermal_image(path: str | Path) -> np.ndarray:
    """Load a raw thermal image from disk.

    Args:
        path: Path to the image file (e.g. PNG, TIFF, or proprietary format).

    Returns:
        Image as a NumPy array with shape (H, W) or (H, W, C).
    """
    raise NotImplementedError


def denoise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Apply denoising to a thermal image.

    Args:
        image: Input image array.
        method: Denoising method — 'gaussian', 'median', or 'bilateral'.

    Returns:
        Denoised image array.
    """
    raise NotImplementedError


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalise pixel values to the [0, 1] range.

    Args:
        image: Input image array.

    Returns:
        Normalised image array of dtype float32.
    """
    raise NotImplementedError


def apply_colormap(image: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Apply a pseudo-colour map to a single-channel thermal image.

    Args:
        image: Single-channel image array normalised to [0, 1].
        colormap: Matplotlib colormap name (e.g. 'inferno', 'jet', 'hot').

    Returns:
        RGB image array with shape (H, W, 3).
    """
    raise NotImplementedError


def preprocess_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    target_size: Optional[tuple[int, int]] = (224, 224),
) -> None:
    """Run the full preprocessing pipeline over a directory of raw images.

    Loads each image, applies denoising, normalisation, resizing, and saves
    the result to output_dir preserving the original subdirectory structure.

    Args:
        raw_dir: Directory containing raw thermal images.
        output_dir: Destination directory for processed images.
        target_size: (height, width) to resize images; None to skip resizing.
    """
    raise NotImplementedError
