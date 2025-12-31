"""
MNIST-style preprocessing for handwritten digit images.

Goal: robust preprocessing for real-world uploaded photos (shadows, noise),
while keeping output compatible with MNIST-trained models:
  - output shape: (28, 28)
  - background: 0.0 (black)
  - digit strokes: >0 (white), normalized to [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PreprocessParams:
    """Tunable parameters for preprocessing."""

    # Adaptive threshold parameters
    adaptive_block_size: int = 31  # must be odd
    adaptive_C: int = 10

    # Morphology
    open_kernel: int = 3
    close_kernel: int = 3
    dilate_iterations: int = 0

    # Crop padding
    pad_px: int = 8

    # Resize target (MNIST style: 20x20 in 28x28 with margins)
    inner_size: int = 20
    out_size: int = 28

    # Deskew
    deskew: bool = True


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    """Convert input image to grayscale uint8 [0..255]."""
    img = np.asarray(image)
    if img.ndim == 3:
        # RGB/BGR/ARGB -> grayscale via average (robust, dependency-free)
        img = img[..., :3].astype(np.float32).mean(axis=2)
    img = img.astype(np.float32)
    # If already in [0,1], scale up
    if img.max() <= 1.5:
        img = img * 255.0
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def _border_median(gray: np.ndarray) -> float:
    h, w = gray.shape
    samples = np.concatenate([gray[0, :], gray[h - 1, :], gray[:, 0], gray[:, w - 1]]).astype(np.float32)
    return float(np.median(samples))


def _center_by_mass(binary_28: np.ndarray, cv2) -> np.ndarray:
    """Translate so center-of-mass is centered at (14,14)."""
    m = cv2.moments(binary_28, binaryImage=True)
    if m["m00"] <= 1e-6:
        return binary_28
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    shift_x = int(round((binary_28.shape[1] / 2) - cx))
    shift_y = int(round((binary_28.shape[0] / 2) - cy))
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted = cv2.warpAffine(binary_28, M, (binary_28.shape[1], binary_28.shape[0]), flags=cv2.INTER_NEAREST, borderValue=0)
    return shifted


def _deskew(binary: np.ndarray, cv2) -> np.ndarray:
    """Deskew using image moments (classic MNIST technique)."""
    m = cv2.moments(binary, binaryImage=True)
    if abs(m["mu02"]) < 1e-6:
        return binary
    skew = m["mu11"] / m["mu02"]
    h, w = binary.shape
    M = np.float32([[1, skew, -0.5 * w * skew], [0, 1, 0]])
    deskewed = cv2.warpAffine(binary, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return deskewed


def preprocess_digit_to_mnist(
    image: np.ndarray,
    *,
    params: Optional[PreprocessParams] = None,
    debug: bool = False,
) -> np.ndarray:
    """
    Robust preprocessing for real photos / scans.

    Returns:
        float32 array shape (28,28), values in [0,1]
    """
    if params is None:
        params = PreprocessParams()

    try:
        import cv2  # type: ignore
    except Exception:
        raise ImportError(
            "opencv-python is required for robust photo preprocessing. "
            "Install it via `pip install -r requirements.txt`."
        )

    gray = _to_gray_uint8(image)

    # Downscale very large images for speed and more stable morphology
    h, w = gray.shape
    max_side = max(h, w)
    if max_side > 1024:
        scale = 1024.0 / max_side
        gray = cv2.resize(gray, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

    # Illumination normalization (helps with shadows on paper)
    blur_bg = cv2.GaussianBlur(gray, (0, 0), sigmaX=21, sigmaY=21)
    norm = cv2.divide(gray, blur_bg, scale=255)

    # Local contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(norm)

    # Decide background polarity using border median
    bg = _border_median(norm)
    light_bg = bg > 127

    # Adaptive threshold for uneven lighting; output should be "digit = white"
    block = params.adaptive_block_size
    if block % 2 == 0:
        block += 1
    block = max(3, block)
    thresh_type = cv2.THRESH_BINARY_INV if light_bg else cv2.THRESH_BINARY
    bw = cv2.adaptiveThreshold(
        norm,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresh_type,
        block,
        params.adaptive_C,
    )

    # Morphology to remove specks and close small gaps
    if params.open_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.open_kernel, params.open_kernel))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    if params.close_kernel > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.close_kernel, params.close_kernel))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    if params.dilate_iterations > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        bw = cv2.dilate(bw, k, iterations=int(params.dilate_iterations))

    # Find the digit region (largest external contour)
    contours, _hier = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: just resize whole image (rare, but prevents crashes)
        resized = cv2.resize(bw, (params.out_size, params.out_size), interpolation=cv2.INTER_AREA)
        out = resized.astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    contour = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(contour)

    # Guard: if contour is tiny, treat as no digit
    if cw * ch < 25:
        resized = cv2.resize(bw, (params.out_size, params.out_size), interpolation=cv2.INTER_AREA)
        out = resized.astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    pad = int(params.pad_px)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(bw.shape[1], x + cw + pad)
    y1 = min(bw.shape[0], y + ch + pad)
    crop = bw[y0:y1, x0:x1]

    if params.deskew:
        crop = _deskew(crop, cv2)

    # Resize preserving aspect ratio into inner_size
    ih, iw = crop.shape
    if ih == 0 or iw == 0:
        resized = cv2.resize(bw, (params.out_size, params.out_size), interpolation=cv2.INTER_AREA)
        out = resized.astype(np.float32) / 255.0
        return np.clip(out, 0.0, 1.0)

    if iw > ih:
        new_w = params.inner_size
        new_h = max(1, int(round(params.inner_size * ih / iw)))
    else:
        new_h = params.inner_size
        new_w = max(1, int(round(params.inner_size * iw / ih)))

    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    out = np.zeros((params.out_size, params.out_size), dtype=np.uint8)
    y_off = (params.out_size - new_h) // 2
    x_off = (params.out_size - new_w) // 2
    out[y_off : y_off + new_h, x_off : x_off + new_w] = resized

    # Final centering by center-of-mass (very important for MNIST-style models)
    out = _center_by_mass(out, cv2)

    # Normalize to [0,1]
    out_f = out.astype(np.float32) / 255.0
    out_f = np.clip(out_f, 0.0, 1.0)

    if debug:
        # Debug output is intentionally minimal to avoid GUI/log noise.
        stroke = int(np.count_nonzero(out > 0))
        print(f"[preprocess] bg={bg:.1f} light_bg={light_bg} crop=({cw}x{ch}) stroke_px={stroke}")

    return out_f

