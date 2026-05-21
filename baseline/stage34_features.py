"""
Stage 3–4: Edge & Ridge Detection + Curvilinear Feature Extraction
===================================================================
Implements multi-scale ridge detection and centerline tracing:

  Stage 3 — Edge & ridge detection:
    - Canny edge detection (thin edge map)
    - Frangi vesselness filter (Hessian eigenvalue analysis)
    - Phase congruency (illumination-invariant edges)

  Stage 4 — Curvilinear feature extraction:
    - Steger sub-pixel centerline tracing (2nd-order Taylor expansion)
    - Morphological skeletonisation
    - RORPO-like path-opening approximation (morphological)

Reference: Frangi et al. (1998) "Multiscale vessel enhancement filtering"
           Steger (1998) "An unbiased detector of curvilinear structures"
           Merveille et al. (2016) "Curvilinear structure analysis by ranking
           the orientation responses of path operators"
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from skimage.filters import frangi as skimage_frangi
from skimage.morphology import skeletonize
from skimage.feature import canny
from typing import Tuple, List


# ── Stage 3a: Canny Edge Detection ──────────────────────────────────────────

def detect_edges_canny(gray: np.ndarray,
                        sigma: float = 1.5,
                        low_threshold: float = 0.05,
                        high_threshold: float = 0.15) -> np.ndarray:
    """
    Canny edge detection on a grayscale image.
    Returns binary edge map (bool).
    """
    edges = canny(gray.astype(np.float64) / 255.0,
                  sigma=sigma,
                  low_threshold=low_threshold,
                  high_threshold=high_threshold,
                  use_quantiles=True)
    return edges.astype(np.uint8) * 255


# ── Stage 3b: Frangi Vesselness Filter ──────────────────────────────────────

def frangi_vesselness(gray: np.ndarray,
                       scales: List[float] = None,
                       beta: float = 0.5,
                       gamma: float = 15.0,
                       black_ridges: bool = False) -> np.ndarray:
    """
    Multi-scale Frangi vesselness filter.
    Responds to elongated, tube-like structures at multiple widths.

    scales: list of Gaussian sigmas corresponding to expected structure widths
            e.g. [1, 2, 4] covers cables 2–8 pixels wide

    Returns float32 vesselness map in [0, 1].
    """
    if scales is None:
        scales = [1.0, 1.5, 2.0, 3.0, 5.0]

    gray_f = gray.astype(np.float64) / 255.0
    vesselness = skimage_frangi(
        gray_f,
        sigmas=scales,
        beta=beta,
        gamma=gamma,
        black_ridges=black_ridges,
    )
    # Normalise to [0, 1]
    if vesselness.max() > 0:
        vesselness = vesselness / vesselness.max()
    return vesselness.astype(np.float32)


def threshold_vesselness(vessel_map: np.ndarray,
                          threshold: float = 0.05) -> np.ndarray:
    """Binary threshold on vesselness map."""
    binary = (vessel_map > threshold).astype(np.uint8) * 255
    return binary


# ── Stage 3c: Hessian Eigenvalue Analysis ───────────────────────────────────

def compute_hessian_features(gray: np.ndarray,
                               sigma: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Hessian matrix elements and return:
      - ridge_strength: |lambda1| where lambda1 is the largest-magnitude eigenvalue
      - ridge_direction: angle of principal curvature direction (in radians)

    For a curvilinear structure, one eigenvalue is large (across the structure)
    and the other is small (along it) — this ratio is the vesselness signature.
    """
    g = gray.astype(np.float64) / 255.0
    # Second-order Gaussian derivatives (Hessian elements)
    Ixx = gaussian_filter(g, sigma, order=[0, 2])
    Iyy = gaussian_filter(g, sigma, order=[2, 0])
    Ixy = gaussian_filter(g, sigma, order=[1, 1])

    # Eigenvalues of the 2×2 Hessian: closed-form solution
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    disc = np.sqrt(np.maximum(0, (trace / 2) ** 2 - det))
    lam1 = trace / 2 + disc  # larger eigenvalue
    lam2 = trace / 2 - disc  # smaller eigenvalue

    ridge_strength = np.abs(lam1)
    ridge_direction = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)

    return ridge_strength.astype(np.float32), ridge_direction.astype(np.float32)


# ── Stage 4a: Steger Centerline Detection ───────────────────────────────────

def steger_centerlines(gray: np.ndarray,
                        sigma: float = 2.0,
                        threshold_fraction: float = 0.1) -> np.ndarray:
    """
    Steger's algorithm: sub-pixel centerline detection via second-order
    Taylor expansion along the ridge normal direction.

    A pixel is a centerline point if the gradient perpendicular to the ridge
    direction has a zero crossing — i.e., the pixel lies at the ridge peak.

    Returns binary centerline map (uint8, 0 or 255).
    """
    g = gray.astype(np.float64) / 255.0

    # Smooth image and compute derivatives up to 2nd order
    Ix  = gaussian_filter(g, sigma, order=[0, 1])
    Iy  = gaussian_filter(g, sigma, order=[1, 0])
    Ixx = gaussian_filter(g, sigma, order=[0, 2])
    Iyy = gaussian_filter(g, sigma, order=[2, 0])
    Ixy = gaussian_filter(g, sigma, order=[1, 1])

    # Ridge direction: eigenvector of Hessian for largest |eigenvalue|
    trace = Ixx + Iyy
    det   = Ixx * Iyy - Ixy ** 2
    disc  = np.sqrt(np.maximum(0, (trace / 2) ** 2 - det))
    lam   = trace / 2 + disc  # largest eigenvalue

    # Normal direction (nx, ny) — eigenvector corresponding to lam
    # For symmetric 2x2 matrix, eigenvector is proportional to [Ixy, lam - Ixx]
    nx = Ixy
    ny = lam - Ixx
    norm = np.sqrt(nx ** 2 + ny ** 2) + 1e-10
    nx /= norm
    ny /= norm

    # Sub-pixel offset along normal: t = -(Ix*nx + Iy*ny) / (Ixx*nx²+2Ixy*nx*ny+Iyy*ny²)
    num = -(Ix * nx + Iy * ny)
    den = Ixx * nx ** 2 + 2 * Ixy * nx * ny + Iyy * ny ** 2 + 1e-10
    t   = num / den

    # A pixel is a centerline candidate if |t| ≤ 0.5 (the extremum falls within the pixel)
    centerline_mask = np.abs(t) <= 0.5

    # Filter by ridge strength (largest |eigenvalue|)
    strength = np.abs(lam)
    threshold = threshold_fraction * strength.max()
    strong_ridge = strength > threshold

    result = (centerline_mask & strong_ridge).astype(np.uint8) * 255
    return result


# ── Stage 4b: Morphological Skeletonisation ──────────────────────────────────

def morphological_skeleton(binary_mask: np.ndarray) -> np.ndarray:
    """
    Morphological thinning (Zhang-Suen skeleton) on a binary mask.
    Produces 1-pixel-wide centerline skeleton.
    """
    binary_bool = binary_mask > 127
    skel = skeletonize(binary_bool)
    return skel.astype(np.uint8) * 255


# ── Stage 4c: RORPO-like Path Opening (morphological approximation) ──────────

def path_opening_approximate(binary: np.ndarray,
                               lengths: List[int] = None,
                               n_directions: int = 8) -> np.ndarray:
    """
    Approximate RORPO path-opening using multi-directional linear structuring elements.
    True RORPO requires the standalone C++ library; this approximation captures
    the key property: pixels are kept only if they belong to a long connected
    path in at least one orientation.

    Returns binary map of pixels that survive at least one directional opening.
    """
    if lengths is None:
        lengths = [15, 25, 40]

    h, w = binary.shape
    result = np.zeros_like(binary)

    angles_deg = np.linspace(0, 180, n_directions, endpoint=False)

    for length in lengths:
        for angle_deg in angles_deg:
            angle_rad = np.deg2rad(angle_deg)
            dx = int(round(np.cos(angle_rad) * length))
            dy = int(round(np.sin(angle_rad) * length))

            # Build linear structuring element
            se_pts = []
            for t in np.linspace(-length // 2, length // 2, length):
                px = int(round(t * np.cos(angle_rad)))
                py = int(round(t * np.sin(angle_rad)))
                se_pts.append((py + length // 2, px + length // 2))

            se_size = length + 4
            se = np.zeros((se_size, se_size), dtype=np.uint8)
            for (py, px) in se_pts:
                py_c = np.clip(py, 0, se_size - 1)
                px_c = np.clip(px, 0, se_size - 1)
                se[py_c, px_c] = 1

            # Morphological opening = erosion followed by dilation
            eroded  = cv2.erode(binary,  se)
            dilated = cv2.dilate(eroded, se)
            result  = np.maximum(result, dilated)

    return result


# ── Combined Stage 3–4 Feature Extraction ────────────────────────────────────

def extract_curvilinear_features(img: np.ndarray,
                                  frangi_scales: List[float] = None,
                                  steger_sigma: float = 2.0,
                                  use_path_opening: bool = True) -> dict:
    """
    Full Stage 3–4 feature extraction pipeline.

    Args:
        img:              Preprocessed BGR image
        frangi_scales:    List of sigmas for Frangi filter
        steger_sigma:     Sigma for Steger centerline detection
        use_path_opening: Apply RORPO-like path opening to Frangi output

    Returns dict with keys:
        gray            — grayscale image
        edges           — Canny edge map
        vesselness      — Frangi vesselness float map [0,1]
        vesselness_bin  — Thresholded vesselness binary mask
        steger          — Steger centerline map
        skeleton        — Morphological skeleton of vesselness_bin
        path_opened     — Path-opening filtered binary (if use_path_opening)
        ridge_strength  — Hessian ridge strength map
        ridge_direction — Hessian principal direction map
    """
    if frangi_scales is None:
        frangi_scales = [1.0, 1.5, 2.0, 3.0, 5.0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    edges = detect_edges_canny(gray)
    vessel = frangi_vesselness(gray, scales=frangi_scales, black_ridges=False)
    vessel_bin = threshold_vesselness(vessel, threshold=0.05)
    steger = steger_centerlines(gray, sigma=steger_sigma)
    skeleton = morphological_skeleton(vessel_bin)
    ridge_str, ridge_dir = compute_hessian_features(gray)

    features = {
        "gray":            gray,
        "edges":           edges,
        "vesselness":      vessel,
        "vesselness_bin":  vessel_bin,
        "steger":          steger,
        "skeleton":        skeleton,
        "ridge_strength":  ridge_str,
        "ridge_direction": ridge_dir,
        "path_opened":     None,
    }

    if use_path_opening:
        features["path_opened"] = path_opening_approximate(vessel_bin)

    return features
