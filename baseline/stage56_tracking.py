"""
Stage 5–6: Contour Linking, Temporal Tracking & Shape Model Fitting
====================================================================
Implements:

  Stage 5 — Contour linking & tracking:
    - Graph-based contour fragment linking (minimum-cost path via A*)
    - Kalman filter tracker (NumPy implementation — no filterpy dependency)
    - Lucas-Kanade sparse optical flow for keypoint correspondence

  Stage 6 — Shape model fitting:
    - Active contour (snake) with gradient-based image energy
    - B-spline parametric curve fitting on skeleton points
    - RANSAC-based polynomial curve fitting for robust estimation

Reference: Kass et al. (1988) "Snakes: Active Contour Models"
           Kalman (1960) "A New Approach to Linear Filtering"
"""

import numpy as np
import cv2
from scipy.ndimage import distance_transform_edt
from scipy.interpolate import splprep, splev
from scipy.spatial.distance import cdist
from skimage.graph import route_through_array
from typing import List, Tuple, Optional, Dict


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5a — Graph-Based Contour Linking
# ═══════════════════════════════════════════════════════════════════════════

def extract_skeleton_endpoints(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find endpoints and junction points in a skeleton image.
    An endpoint has exactly 1 neighbour; a junction has ≥3.
    Returns list of (row, col) tuples.
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    # Count 8-connected neighbours for each skeleton pixel
    neighbour_count = cv2.filter2D(
        (skeleton > 0).astype(np.uint8), -1, kernel
    ) - (skeleton > 0).astype(np.uint8)

    skel_bool = skeleton > 0
    endpoints = list(zip(*np.where(skel_bool & (neighbour_count == 1))))
    junctions = list(zip(*np.where(skel_bool & (neighbour_count >= 3))))
    return endpoints, junctions


def link_skeleton_fragments(skeleton: np.ndarray,
                              max_gap_pixels: int = 30) -> np.ndarray:
    """
    Link disconnected skeleton fragments by bridging gaps up to max_gap_pixels.
    Uses minimum-cost path through the distance transform of the complement mask.

    Strategy: find endpoints of each component → connect closest endpoint pairs
    that are within max_gap_pixels → fill the path.

    Returns augmented skeleton with bridges drawn.
    """
    linked = skeleton.copy()
    # Label connected components
    n_labels, labels = cv2.connectedComponents((skeleton > 0).astype(np.uint8))

    if n_labels <= 2:
        return linked  # 0 or 1 components — nothing to link

    # Get endpoints per component
    comp_endpoints = {}
    for label_id in range(1, n_labels):
        comp_mask = (labels == label_id).astype(np.uint8) * 255
        eps, _ = extract_skeleton_endpoints(comp_mask)
        if eps:
            comp_endpoints[label_id] = eps

    if len(comp_endpoints) < 2:
        return linked

    # Build list of all endpoints with component labels
    all_endpoints = []
    all_labels    = []
    for label_id, eps in comp_endpoints.items():
        for ep in eps:
            all_endpoints.append(ep)
            all_labels.append(label_id)

    if len(all_endpoints) < 2:
        return linked

    pts = np.array(all_endpoints)
    lbs = np.array(all_labels)

    # Distance matrix between all endpoints
    dist_matrix = cdist(pts, pts)
    # Mask same-component pairs (set to infinity)
    for i in range(len(lbs)):
        for j in range(len(lbs)):
            if lbs[i] == lbs[j]:
                dist_matrix[i, j] = np.inf

    # Greedily link the closest cross-component endpoint pairs
    visited_components = set()
    while True:
        finite_mask = np.isfinite(dist_matrix)
        if not finite_mask.any():
            break
        idx = np.argmin(dist_matrix)
        i, j = divmod(idx, len(pts))
        gap = dist_matrix[i, j]
        if gap > max_gap_pixels:
            break

        # Draw a line bridge between the two endpoints
        pt_i = (int(pts[i][1]), int(pts[i][0]))  # (x, y) for cv2
        pt_j = (int(pts[j][1]), int(pts[j][0]))
        cv2.line(linked, pt_i, pt_j, 255, 1)

        # Remove all pairs involving these two components from further linking
        mask_i = (lbs == lbs[i])
        mask_j = (lbs == lbs[j])
        dist_matrix[mask_i, :] = np.inf
        dist_matrix[:, mask_i] = np.inf
        dist_matrix[mask_j, :] = np.inf
        dist_matrix[:, mask_j] = np.inf

    return linked


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5b — Kalman Filter Tracker (pure NumPy)
# ═══════════════════════════════════════════════════════════════════════════

class KalmanCurveTracker:
    """
    Kalman filter for tracking a curvilinear object across frames.

    State vector: [cx, cy, vx, vy, length, d_length]
      cx, cy    — centroid of the detected curve
      vx, vy    — velocity of the centroid
      length    — arc length of the curve (pixels)
      d_length  — rate of change of arc length

    Measurement vector: [cx_obs, cy_obs, length_obs]

    This is a simplified tracker — for full curve tracking, each point
    on the B-spline would have its own state, but this centroid+length
    model is sufficient for the baseline pipeline.
    """

    def __init__(self, dt: float = 1.0):
        self.dt = dt
        n = 6   # state dimension
        m = 3   # measurement dimension

        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0,  0,  0 ],
            [0, 1, 0,  dt, 0,  0 ],
            [0, 0, 1,  0,  0,  0 ],
            [0, 0, 0,  1,  0,  0 ],
            [0, 0, 0,  0,  1,  dt],
            [0, 0, 0,  0,  0,  1 ],
        ], dtype=np.float64)

        # Measurement matrix: we observe cx, cy, length
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self.Q = np.diag([2.0, 2.0, 0.5, 0.5, 5.0, 1.0])

        # Measurement noise covariance
        self.R = np.diag([5.0, 5.0, 20.0])

        # State estimate and covariance (initialised on first measurement)
        self.x = None
        self.P = np.eye(n) * 500.0

        self.initialised = False
        self.history: List[np.ndarray] = []

    def initialise(self, cx: float, cy: float, length: float):
        self.x = np.array([cx, cy, 0.0, 0.0, length, 0.0], dtype=np.float64)
        self.initialised = True

    def predict(self) -> np.ndarray:
        """Predict next state."""
        if not self.initialised:
            raise RuntimeError("Tracker not initialised — call initialise() first.")
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.copy()

    def update(self, cx: float, cy: float, length: float) -> np.ndarray:
        """Update with new measurement."""
        z = np.array([cx, cy, length], dtype=np.float64)
        S = self.H @ self.P @ self.H.T + self.R          # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        y = z - self.H @ self.x                            # innovation
        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        self.history.append(self.x.copy())
        return self.x.copy()

    def get_predicted_centroid(self) -> Tuple[float, float]:
        if self.x is None:
            return (0.0, 0.0)
        return (self.x[0], self.x[1])

    def get_predicted_length(self) -> float:
        if self.x is None:
            return 0.0
        return self.x[4]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 5c — Lucas-Kanade Optical Flow Correspondence
# ═══════════════════════════════════════════════════════════════════════════

def track_points_optical_flow(prev_gray: np.ndarray,
                               curr_gray: np.ndarray,
                               points: np.ndarray,
                               win_size: int = 21) -> Tuple[np.ndarray, np.ndarray]:
    """
    Track keypoints from prev_gray to curr_gray using Lucas-Kanade sparse
    optical flow. Returns new point positions and a validity mask.

    Args:
        prev_gray: Previous frame (grayscale, uint8)
        curr_gray: Current frame (grayscale, uint8)
        points:    Nx1x2 float32 array of (x, y) point coordinates
        win_size:  LK window size

    Returns:
        new_pts:  Tracked positions (Nx1x2 float32)
        valid:    Boolean mask of successfully tracked points (N,)
    """
    lk_params = dict(
        winSize=(win_size, win_size),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    pts_f32 = points.astype(np.float32)
    new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts_f32, None, **lk_params
    )
    valid = (status.ravel() == 1)
    return new_pts, valid


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6a — Active Contour (Snake)
# ═══════════════════════════════════════════════════════════════════════════

class ActiveContourSnake:
    """
    Simplified active contour (snake) model.
    Energy = alpha * internal_elasticity + beta * internal_curvature + gamma * image_energy

    The image energy is derived from a pre-computed gradient magnitude map:
    strong gradients pull the snake toward edges.

    This is a gradient-descent implementation — for production, use
    skimage.segmentation.active_contour which uses a more robust solver.
    """

    def __init__(self, alpha: float = 0.01, beta: float = 0.1, gamma: float = 1.0,
                 n_iter: int = 100, step: float = 0.5):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.n_iter = n_iter
        self.step  = step

    def _image_gradient_force(self, gray: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pre-compute gradient force field from image."""
        blur = cv2.GaussianBlur(gray.astype(np.float32), (0, 0), 2.0)
        gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1)
        # Normalise
        mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
        return gx / mag, gy / mag

    def fit(self, gray: np.ndarray,
            init_points: np.ndarray) -> np.ndarray:
        """
        Fit snake to image starting from init_points.

        Args:
            gray:        Grayscale image (uint8 H×W)
            init_points: Nx2 float array of (x, y) initial control points

        Returns:
            Nx2 float array of fitted snake control points
        """
        h, w = gray.shape
        pts = init_points.astype(np.float64).copy()
        n = len(pts)
        if n < 3:
            return pts

        fx, fy = self._image_gradient_force(gray)

        for _ in range(self.n_iter):
            # Internal forces: elasticity (resist stretching)
            elastic = np.zeros_like(pts)
            for i in range(1, n - 1):
                elastic[i] = pts[i - 1] - 2 * pts[i] + pts[i + 1]

            # Internal forces: curvature (resist bending) — 2nd finite difference
            curvature = np.zeros_like(pts)
            for i in range(2, n - 2):
                curvature[i] = pts[i - 2] - 4 * pts[i - 1] + 6 * pts[i] - 4 * pts[i + 1] + pts[i + 2]

            # Image gradient force at current positions
            img_force = np.zeros_like(pts)
            for i in range(n):
                xi = int(np.clip(pts[i, 0], 0, w - 1))
                yi = int(np.clip(pts[i, 1], 0, h - 1))
                img_force[i, 0] = fx[yi, xi]
                img_force[i, 1] = fy[yi, xi]

            # Update
            pts += self.step * (
                self.alpha * elastic
                - self.beta  * curvature
                + self.gamma * img_force
            )
            # Clamp to image bounds
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

        return pts


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6b — B-Spline Parametric Curve Fitting
# ═══════════════════════════════════════════════════════════════════════════

def fit_bspline(skeleton_pts: np.ndarray,
                n_output: int = 200,
                smoothing: float = None) -> Optional[np.ndarray]:
    """
    Fit a B-spline to a set of skeleton points and evaluate it at n_output points.

    Args:
        skeleton_pts: Nx2 array of (x, y) skeleton points
        n_output:     Number of points to evaluate the fitted spline at
        smoothing:    Smoothing factor; None = interpolation (no smoothing)

    Returns:
        n_output × 2 float array of fitted curve points, or None on failure.
    """
    if len(skeleton_pts) < 4:
        return None

    # Remove duplicate consecutive points (spline requires distinct points)
    pts = skeleton_pts.astype(np.float64)
    unique_mask = np.concatenate([[True], np.any(np.diff(pts, axis=0) != 0, axis=1)])
    pts = pts[unique_mask]

    if len(pts) < 4:
        return None

    s = smoothing if smoothing is not None else 0
    try:
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=s, k=3)
        u_new = np.linspace(0, 1, n_output)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Stage 6c — RANSAC Polynomial Curve Fitting
# ═══════════════════════════════════════════════════════════════════════════

def ransac_polynomial_fit(skeleton_pts: np.ndarray,
                           degree: int = 3,
                           n_iter: int = 100,
                           inlier_threshold: float = 3.0,
                           min_inliers_frac: float = 0.5) -> Optional[np.ndarray]:
    """
    RANSAC-based polynomial fitting on skeleton points.
    Robust to outlier edge fragments — fits the dominant curve.

    Returns Nx2 float array of fitted curve points evaluated over the
    x-range of inliers, or None if fitting fails.
    """
    if len(skeleton_pts) < degree + 1:
        return None

    pts = skeleton_pts.astype(np.float64)
    x, y = pts[:, 0], pts[:, 1]
    n = len(x)
    min_inliers = max(degree + 1, int(min_inliers_frac * n))

    best_coeffs = None
    best_inlier_count = 0

    rng = np.random.default_rng(42)

    for _ in range(n_iter):
        # Sample minimal set
        sample_idx = rng.choice(n, size=degree + 1, replace=False)
        try:
            coeffs = np.polyfit(x[sample_idx], y[sample_idx], degree)
        except np.linalg.LinAlgError:
            continue

        # Compute residuals for all points
        y_pred = np.polyval(coeffs, x)
        residuals = np.abs(y - y_pred)
        inliers = residuals < inlier_threshold
        inlier_count = inliers.sum()

        if inlier_count > best_inlier_count:
            best_inlier_count = inlier_count
            best_coeffs = coeffs

    if best_coeffs is None or best_inlier_count < min_inliers:
        return None

    # Refit on all inliers
    y_pred_best = np.polyval(best_coeffs, x)
    final_inliers = np.abs(y - y_pred_best) < inlier_threshold
    if final_inliers.sum() < degree + 1:
        return None

    try:
        final_coeffs = np.polyfit(x[final_inliers], y[final_inliers], degree)
    except np.linalg.LinAlgError:
        return None

    x_range = np.linspace(x[final_inliers].min(), x[final_inliers].max(), 300)
    y_range = np.polyval(final_coeffs, x_range)
    return np.column_stack([x_range, y_range])


# ═══════════════════════════════════════════════════════════════════════════
# Helper: Extract ordered points from skeleton
# ═══════════════════════════════════════════════════════════════════════════

def skeleton_to_ordered_points(skeleton: np.ndarray,
                                 max_points: int = 500) -> np.ndarray:
    """
    Convert a skeleton binary image to an ordered list of (x, y) points
    by following the skeleton from an endpoint.

    Returns Nx2 array of (x, y) float64 points.
    Falls back to unordered points if skeleton traversal fails.
    """
    ys, xs = np.where(skeleton > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float64)

    # Subsample if very long
    if len(xs) > max_points:
        idx = np.linspace(0, len(xs) - 1, max_points).astype(int)
        xs, ys = xs[idx], ys[idx]

    # Simple ordering: sort by x-coordinate (works for mostly-horizontal curves)
    # For more complex topologies, use skeleton traversal
    order = np.argsort(xs)
    return np.column_stack([xs[order], ys[order]]).astype(np.float64)


def compute_arc_length(pts: np.ndarray) -> float:
    """Compute arc length (total path length) of an ordered point array."""
    if len(pts) < 2:
        return 0.0
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.sqrt((diffs ** 2).sum(axis=1))))
