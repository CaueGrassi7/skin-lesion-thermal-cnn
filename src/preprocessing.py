"""
Thermal image preprocessing pipeline.

Covers raw image loading, noise reduction, normalisation, pseudo-colour
mapping, and saving processed images ready for model input.
"""

from pathlib import Path
from typing import Optional

import cv2
import matplotlib.cm as mpl_cm
import numpy as np
from PIL import Image
from tqdm import tqdm


def load_thermal_image(path: str | Path) -> np.ndarray:
    """Load a raw thermal image from disk as a 2-D grayscale array.

    Returns:
        Image as a NumPy array of shape (H, W), dtype uint8.
    """
    img = np.array(Image.open(path))
    if img.ndim == 3:
        img = img[:, :, 0]
    return img


def denoise(image: np.ndarray, method: str = "gaussian") -> np.ndarray:
    """Apply denoising to a thermal image.

    Args:
        image: Input uint8 array of shape (H, W).
        method: 'gaussian' (5×5 kernel), 'median' (5×5), or 'bilateral'.

    Returns:
        Denoised array with the same dtype and shape.
    """
    if method == "gaussian":
        return cv2.GaussianBlur(image, (5, 5), 0)
    if method == "median":
        return cv2.medianBlur(image, 5)
    if method == "bilateral":
        return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    raise ValueError(f"Unknown denoising method: {method!r}")


def normalize(image: np.ndarray) -> np.ndarray:
    """Normalise pixel values to the [0, 1] range.

    Args:
        image: Input uint8 array.

    Returns:
        Float32 array with values in [0, 1].
    """
    return image.astype(np.float32) / 255.0


def apply_colormap(image: np.ndarray, colormap: str = "inferno") -> np.ndarray:
    """Apply a pseudo-colour map to a single-channel thermal image.

    Args:
        image: Float32 array of shape (H, W) normalised to [0, 1].
        colormap: Matplotlib colormap name (e.g. 'inferno', 'jet', 'hot').

    Returns:
        uint8 RGB array of shape (H, W, 3).
    """
    cmap = mpl_cm.get_cmap(colormap)
    rgb_float = cmap(image)[:, :, :3]  # drop alpha
    return (rgb_float * 255).astype(np.uint8)


def preprocess_dataset(
    raw_dir: str | Path,
    output_dir: str | Path,
    target_size: Optional[tuple[int, int]] = (224, 224),
    denoise_method: Optional[str] = "gaussian",
    colormap: Optional[str] = None,
) -> None:
    """Run the full preprocessing pipeline over a directory of raw images.

    Walks raw_dir expecting the structure ``Class/Patient/frame.png``.
    For each frame: load → (optional) denoise → normalise →
    (optional) resize → (optional) colormap → save as PNG.

    Output mirrors the input directory tree under output_dir.

    Args:
        raw_dir: Directory containing raw thermal images.
        output_dir: Destination directory for processed images.
        target_size: (height, width) to resize images; None to keep original.
        denoise_method: Denoising method passed to :func:`denoise`; None to skip.
        colormap: Matplotlib colormap name to produce RGB output; None keeps grayscale.
    """
    raw_dir = Path(raw_dir)
    output_dir = Path(output_dir)

    image_paths = sorted(raw_dir.rglob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG images found under {raw_dir}")

    for src_path in tqdm(image_paths, desc="Preprocessing"):
        rel_path = src_path.relative_to(raw_dir)
        dst_path = output_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        img = load_thermal_image(src_path)

        if denoise_method is not None:
            img = denoise(img, method=denoise_method)

        img_float = normalize(img)

        if target_size is not None:
            h, w = target_size
            img_float = cv2.resize(img_float, (w, h), interpolation=cv2.INTER_AREA)

        if colormap is not None:
            out = apply_colormap(img_float, colormap=colormap)
            Image.fromarray(out).save(dst_path)
        else:
            out = (img_float * 255).astype(np.uint8)
            Image.fromarray(out).save(dst_path)
