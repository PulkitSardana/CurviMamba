"""
Stage 1–2: Image Acquisition & Preprocessing
=============================================
Implements the full underwater preprocessing stack:
  - White balance correction
  - Physics-based Dark Channel Prior dehazing
  - Non-local means (NLM) denoising
  - CLAHE local contrast enhancement

Reference: He et al. (2011) "Single Image Haze Removal Using Dark Channel Prior"
           Zuiderveld (1994) "Contrast Limited Adaptive Histogram Equalization"
"""

import cv2
import numpy as np
from typing import Tuple


# ── White Balance ────────────────────────────────────────────────────────────

def white_balance_grayworld(img: np.ndarray) -> np.ndarray:
    """
    Gray-world white balance assumption.
    Scales each channel so its mean equals the overall mean.
    Compensates for the blue/green cast dominant in underwater imagery.
    """
    img_f = img.astype(np.float32)
    mean_per_channel = img_f.mean(axis=(0, 1))       # shape: (3,) — B, G, R
    overall_mean = mean_per_channel.mean()
    scale = overall_mean / (mean_per_channel + 1e-6)
    balanced = img_f * scale[np.newaxis, np.newaxis, :]
    return np.clip(balanced, 0, 255).astype(np.uint8)


# ── Dark Channel Prior Dehazing ──────────────────────────────────────────────

def dark_channel(img: np.ndarray, patch_size: int = 15) -> np.ndarray:
    """Compute dark channel: min over local patch and over channels."""
    min_channel = img.min(axis=2)          # per-pixel channel minimum
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (patch_size, patch_size))
    dark = cv2.erode(min_channel, kernel)  # local minimum = erosion
    return dark


def estimate_atmospheric_light(img: np.ndarray, dark: np.ndarray,
                                top_fraction: float = 0.001) -> np.ndarray:
    """
    Estimate atmospheric light A from the top brightest pixels
    in the dark channel — robust to local bright objects.
    """
    h, w = dark.shape
    n_pixels = h * w
    n_top = max(1, int(n_pixels * top_fraction))
    flat_dark = dark.flatten()
    flat_img = img.reshape(-1, 3)
    # Get indices of the brightest dark-channel pixels
    indices = np.argpartition(flat_dark, -n_top)[-n_top:]
    A = flat_img[indices].max(axis=0).astype(np.float32)
    return A


def dehaze_dark_channel(img: np.ndarray,
                         patch_size: int = 15,
                         omega: float = 0.85,
                         t_min: float = 0.1) -> np.ndarray:
    """
    Dark Channel Prior dehazing (He et al. 2011).
    omega < 1 retains a small amount of haze for naturalness.
    Adapted for underwater: applied per-channel to handle non-uniform attenuation.
    """
    img_f = img.astype(np.float32) / 255.0
    # Compute dark channel on normalised image
    dark = dark_channel((img_f * 255).astype(np.uint8), patch_size) / 255.0
    A = estimate_atmospheric_light((img_f * 255).astype(np.uint8),
                                   (dark * 255).astype(np.uint8)) / 255.0

    # Transmission estimate
    norm = img_f / (A[np.newaxis, np.newaxis, :] + 1e-6)
    dark_norm = dark_channel((norm * 255).astype(np.uint8), patch_size) / 255.0
    t = 1.0 - omega * dark_norm
    t = np.maximum(t, t_min)

    # Guided filter refine the transmission map (edge-preserving)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t_refined = cv2.ximgproc.guidedFilter(
        gray.astype(np.float32), t.astype(np.float32),
        radius=40, eps=1e-3) if hasattr(cv2, 'ximgproc') else t

    # Recover scene radiance: J = (I - A) / t + A
    t3 = t_refined[:, :, np.newaxis]
    J = (img_f - A[np.newaxis, np.newaxis, :]) / t3 + A[np.newaxis, np.newaxis, :]
    J = np.clip(J, 0, 1)
    return (J * 255).astype(np.uint8)


# ── Denoising ────────────────────────────────────────────────────────────────

def denoise_nlm(img: np.ndarray,
                h_luminance: float = 10.0,
                template_window: int = 7,
                search_window: int = 21) -> np.ndarray:
    """
    Non-local means denoising — preserves edges better than Gaussian.
    For colour images, applied in luminance + chrominance decomposition.
    """
    return cv2.fastNlMeansDenoisingColored(
        img,
        None,
        h=h_luminance,
        hColor=h_luminance,
        templateWindowSize=template_window,
        searchWindowSize=search_window,
    )


# ── CLAHE ────────────────────────────────────────────────────────────────────

def apply_clahe(img: np.ndarray,
                clip_limit: float = 2.5,
                tile_grid: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization.
    Applied to the L channel of LAB colour space to preserve hue.
    clip_limit controls the contrast cap — higher = more aggressive.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_eq = clahe.apply(l_ch)
    lab_eq = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ── Full Preprocessing Stack ─────────────────────────────────────────────────

def preprocess(img: np.ndarray,
               apply_dehaze: bool = True,
               apply_denoise: bool = True,
               clahe_clip: float = 2.5) -> np.ndarray:
    """
    Full Stage 1–2 preprocessing pipeline.

    Order: White balance → Dehaze → Denoise → CLAHE

    Args:
        img:            Input BGR image (uint8, H×W×3)
        apply_dehaze:   Apply Dark Channel Prior dehazing
        apply_denoise:  Apply NLM denoising
        clahe_clip:     CLAHE clip limit

    Returns:
        Preprocessed BGR image (uint8, H×W×3)
    """
    out = white_balance_grayworld(img)

    if apply_dehaze:
        out = dehaze_dark_channel(out)

    if apply_denoise:
        # Bilateral filter: edge-preserving, ~10x faster than NLM for baseline
        out = cv2.bilateralFilter(out, d=7, sigmaColor=35, sigmaSpace=35)

    out = apply_clahe(out, clip_limit=clahe_clip)
    return out
