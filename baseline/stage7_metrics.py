"""
Stage 7: Post-Processing, Metrics & Evaluation
===============================================
Implements:
  - Geometric filtering (aspect ratio, arc length, curvature continuity)
  - Baseline metrics: centerline precision, recall, F1
  - Connectivity score (topological correctness)
  - FPS measurement
  - Results table generation
"""

import numpy as np
import cv2
from skimage.morphology import skeletonize, label
from scipy.ndimage import distance_transform_edt
from typing import Dict, List, Tuple, Optional
import time


# ═══════════════════════════════════════════════════════════════════════════
# Geometric Filtering (Stage 7a)
# ═══════════════════════════════════════════════════════════════════════════

def filter_by_geometry(skeleton: np.ndarray,
                        min_arc_length: int = 50,
                        min_aspect_ratio: float = 3.0) -> np.ndarray:
    """
    Remove skeleton fragments that are too short or too blob-like.
    Keeps only components that look like genuine curvilinear structures.

    Args:
        skeleton:         Binary skeleton image
        min_arc_length:   Minimum number of skeleton pixels per component
        min_aspect_ratio: Minimum bounding-box aspect ratio (long/short side)

    Returns:
        Filtered skeleton (same dtype as input).
    """
    filtered = np.zeros_like(skeleton)
    n_labels, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8))

    for label_id in range(1, n_labels):
        comp_mask = (labels == label_id).astype(np.uint8)
        arc_length = comp_mask.sum()
        if arc_length < min_arc_length:
            continue

        # Bounding box aspect ratio
        ys, xs = np.where(comp_mask > 0)
        if len(xs) < 2:
            continue
        bbox_w = xs.max() - xs.min() + 1
        bbox_h = ys.max() - ys.min() + 1
        long_side  = max(bbox_w, bbox_h)
        short_side = min(bbox_w, bbox_h)
        aspect_ratio = long_side / (short_side + 1e-6)

        if aspect_ratio >= min_aspect_ratio:
            filtered[comp_mask > 0] = 255

    return filtered


# ═══════════════════════════════════════════════════════════════════════════
# Metrics (Stage 7b)
# ═══════════════════════════════════════════════════════════════════════════

def centerline_precision_recall(pred_skeleton: np.ndarray,
                                 gt_skeleton: np.ndarray,
                                 tolerance_px: int = 3
                                 ) -> Tuple[float, float, float]:
    """
    Compute centerline precision and recall with a pixel tolerance.

    A predicted skeleton pixel is a true positive if it lies within
    tolerance_px pixels of any ground-truth skeleton pixel, and vice versa.

    Args:
        pred_skeleton:   Predicted binary skeleton (uint8, 0/255)
        gt_skeleton:     Ground-truth binary skeleton (uint8, 0/255)
        tolerance_px:    Distance tolerance in pixels

    Returns:
        (precision, recall, f1) as floats in [0, 1]
    """
    pred_bool = pred_skeleton > 0
    gt_bool   = gt_skeleton   > 0

    n_pred = pred_bool.sum()
    n_gt   = gt_bool.sum()

    if n_pred == 0 and n_gt == 0:
        return 1.0, 1.0, 1.0
    if n_pred == 0:
        return 0.0, 0.0, 0.0
    if n_gt == 0:
        return 0.0, 0.0, 0.0

    # Distance transform of the complement — gives distance to nearest GT/pred pixel
    dist_to_gt   = distance_transform_edt(~gt_bool)
    dist_to_pred = distance_transform_edt(~pred_bool)

    tp_pred = (dist_to_gt[pred_bool] <= tolerance_px).sum()   # pred pixels close to GT
    tp_gt   = (dist_to_pred[gt_bool] <= tolerance_px).sum()   # GT pixels close to pred

    precision = tp_pred / n_pred
    recall    = tp_gt   / n_gt
    f1 = (2 * precision * recall / (precision + recall + 1e-10))

    return float(precision), float(recall), float(f1)


def connectivity_score(pred_skeleton: np.ndarray,
                        gt_skeleton: np.ndarray,
                        tolerance_px: int = 5) -> float:
    """
    Connectivity score: measures how well the predicted skeleton maintains
    the topological connectivity of the ground-truth structure.

    For each connected component in the GT skeleton, check whether it is
    "covered" (≥50% of its pixels have a predicted pixel within tolerance).
    Score = fraction of GT components that are covered.

    This captures the core failure mode of classical pipelines: breaking a
    continuous cable into multiple disconnected fragments.

    Args:
        pred_skeleton:  Predicted binary skeleton
        gt_skeleton:    Ground-truth binary skeleton
        tolerance_px:   Coverage threshold distance

    Returns:
        Connectivity score in [0, 1]. 1.0 = all GT components fully covered.
    """
    gt_bool  = gt_skeleton > 0
    pred_bool = pred_skeleton > 0

    n_gt_labels, gt_labels = cv2.connectedComponents(gt_bool.astype(np.uint8))
    n_gt_components = n_gt_labels - 1  # exclude background

    if n_gt_components == 0:
        return 1.0

    dist_to_pred = distance_transform_edt(~pred_bool)

    covered = 0
    for label_id in range(1, n_gt_labels):
        comp_mask = (gt_labels == label_id)
        comp_pixels = comp_mask.sum()
        if comp_pixels == 0:
            continue
        covered_pixels = (dist_to_pred[comp_mask] <= tolerance_px).sum()
        coverage_frac  = covered_pixels / comp_pixels
        if coverage_frac >= 0.5:
            covered += 1

    return covered / n_gt_components


def topological_correctness(pred_skeleton: np.ndarray,
                              gt_skeleton: np.ndarray) -> Dict[str, float]:
    """
    Euler-characteristic based topology metric.

    The Euler number of a binary image = n_components - n_holes (in 2D).
    For an ideal centerline of one cable: Euler number = 1 (one component, no holes).
    Fragmentation increases the Euler number; merging spurious loops decreases it.

    Returns:
        dict with keys: gt_euler, pred_euler, euler_error, n_gt_components, n_pred_components
    """
    from skimage.measure import euler_number, label as sk_label

    def safe_euler(skeleton):
        binary = skeleton > 0
        if not binary.any():
            return 0, 0
        labeled = sk_label(binary, connectivity=2)
        n_comp = labeled.max()
        euler = euler_number(binary, connectivity=2)
        return int(euler), int(n_comp)

    gt_euler, n_gt  = safe_euler(gt_skeleton)
    pr_euler, n_pr  = safe_euler(pred_skeleton)

    return {
        "gt_euler":          gt_euler,
        "pred_euler":        pr_euler,
        "euler_error":       abs(gt_euler - pr_euler),
        "n_gt_components":   n_gt,
        "n_pred_components": n_pr,
    }


# ═══════════════════════════════════════════════════════════════════════════
# FPS Measurement (Stage 7c)
# ═══════════════════════════════════════════════════════════════════════════

class FPSCounter:
    """Measures end-to-end pipeline throughput."""

    def __init__(self):
        self.times: List[float] = []
        self._start: Optional[float] = None

    def start(self):
        self._start = time.perf_counter()

    def stop(self):
        if self._start is not None:
            self.times.append(time.perf_counter() - self._start)
            self._start = None

    @property
    def fps(self) -> float:
        if not self.times:
            return 0.0
        return 1.0 / (np.mean(self.times) + 1e-10)

    @property
    def mean_ms(self) -> float:
        return float(np.mean(self.times) * 1000) if self.times else 0.0

    @property
    def std_ms(self) -> float:
        return float(np.std(self.times) * 1000) if self.times else 0.0

    def summary(self) -> dict:
        return {
            "fps":     round(self.fps, 2),
            "mean_ms": round(self.mean_ms, 1),
            "std_ms":  round(self.std_ms, 1),
            "n_frames": len(self.times),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Per-image Evaluation
# ═══════════════════════════════════════════════════════════════════════════

def evaluate_single(pred_skeleton: np.ndarray,
                     gt_skeleton: np.ndarray,
                     tolerance_px: int = 3) -> Dict[str, float]:
    """Compute all metrics for a single image prediction."""
    prec, rec, f1 = centerline_precision_recall(pred_skeleton, gt_skeleton, tolerance_px)
    conn = connectivity_score(pred_skeleton, gt_skeleton, tolerance_px + 2)
    topo = topological_correctness(pred_skeleton, gt_skeleton)

    return {
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "f1":                round(f1, 4),
        "connectivity":      round(conn, 4),
        "euler_error":       topo["euler_error"],
        "n_gt_components":   topo["n_gt_components"],
        "n_pred_components": topo["n_pred_components"],
    }


def aggregate_metrics(per_image_results: List[Dict]) -> Dict[str, float]:
    """Compute mean ± std across all images."""
    keys = ["precision", "recall", "f1", "connectivity", "euler_error"]
    aggregated = {}
    for k in keys:
        vals = [r[k] for r in per_image_results if k in r]
        aggregated[f"{k}_mean"] = round(float(np.mean(vals)), 4) if vals else 0.0
        aggregated[f"{k}_std"]  = round(float(np.std(vals)),  4) if vals else 0.0
    return aggregated
