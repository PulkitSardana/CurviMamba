"""
Synthetic Underwater Dataset Generator
=======================================
Generates realistic underwater images containing long, flexible, curvilinear
objects (cables, ropes, pipes) with ground-truth masks and centerline skeletons.

Used as the baseline dataset since URPC/DUO require manual download.
Designed to mimic real underwater degradation effects.
"""

import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from skimage.draw import line_aa
import os
import json
from pathlib import Path


# ── Underwater degradation helpers ──────────────────────────────────────────

def apply_underwater_attenuation(img: np.ndarray, depth: float = 5.0) -> np.ndarray:
    """Simulate wavelength-dependent light attenuation (red channel drops fast)."""
    img = img.astype(np.float32) / 255.0
    # Attenuation coefficients per channel (B, G, R) — red attenuates ~10x faster
    att = np.array([0.02, 0.05, 0.22])
    scale = np.exp(-att * depth)
    img = img * scale[np.newaxis, np.newaxis, :]
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def apply_backscatter(img: np.ndarray, intensity: float = 0.08) -> np.ndarray:
    """Add uniform additive backscatter haze (bluish)."""
    haze = np.array([intensity * 255, intensity * 0.6 * 255, intensity * 0.3 * 255],
                    dtype=np.float32)
    out = img.astype(np.float32) + haze[np.newaxis, np.newaxis, :]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_turbidity(img: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Simulate turbidity as slight Gaussian blur."""
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    return blurred


def add_speckle_noise(img: np.ndarray, std: float = 8.0) -> np.ndarray:
    """Add speckle noise typical of underwater camera sensors."""
    noise = np.random.randn(*img.shape) * std
    out = img.astype(np.float32) + noise
    return np.clip(out, 0, 255).astype(np.uint8)


def generate_background(h: int, w: int, depth: float) -> np.ndarray:
    """Generate a realistic underwater background texture."""
    # Start with a dark blue-green gradient
    bg = np.zeros((h, w, 3), dtype=np.float32)
    y_grad = np.linspace(0.2, 0.05, h)[:, np.newaxis]
    bg[:, :, 0] = y_grad * 80   # Blue
    bg[:, :, 1] = y_grad * 60   # Green
    bg[:, :, 2] = y_grad * 20   # Red (very low)

    # Add low-frequency texture (sediment / caustic patterns)
    noise_b = gaussian_filter(np.random.randn(h, w) * 15, sigma=20)
    noise_g = gaussian_filter(np.random.randn(h, w) * 10, sigma=25)
    bg[:, :, 0] += noise_b
    bg[:, :, 1] += noise_g

    bg = np.clip(bg, 0, 255).astype(np.uint8)
    bg = apply_underwater_attenuation(bg, depth)
    return bg


# ── Curvilinear object generator ─────────────────────────────────────────────

def bezier_curve(p0, p1, p2, p3, n_points: int = 500):
    """Cubic Bézier curve — produces smooth, natural cable shapes."""
    t = np.linspace(0, 1, n_points)[:, np.newaxis]  # (N,1) for broadcasting with (2,) points
    curve = (
        (1 - t) ** 3 * p0 +
        3 * (1 - t) ** 2 * t * p1 +
        3 * (1 - t) * t ** 2 * p2 +
        t ** 3 * p3
    )
    return curve.astype(int)


def draw_curvilinear_object(canvas: np.ndarray, mask: np.ndarray,
                             skeleton: np.ndarray, rng: np.random.Generator,
                             h: int, w: int, obj_type: str = "cable"):
    """
    Draw a single curvilinear object on the canvas with a given type.
    Returns updated canvas, mask, skeleton, and the centerline points.
    """
    # Random control points for Bézier
    margin = 30
    p0 = np.array([rng.integers(margin, w // 3), rng.integers(margin, h - margin)])
    p3 = np.array([rng.integers(2 * w // 3, w - margin), rng.integers(margin, h - margin)])
    p1 = np.array([rng.integers(w // 4, w // 2), rng.integers(0, h)])
    p2 = np.array([rng.integers(w // 2, 3 * w // 4), rng.integers(0, h)])

    centerline_pts = bezier_curve(p0, p1, p2, p3, n_points=600)

    # Object appearance per type
    type_params = {
        "cable":    {"thickness_range": (2, 5),  "color_bgr": (40, 60, 80),   "blur": 1.0},
        "rope":     {"thickness_range": (4, 9),  "color_bgr": (60, 80, 100),  "blur": 1.5},
        "pipeline": {"thickness_range": (8, 18), "color_bgr": (80, 100, 90),  "blur": 2.0},
        "mooring":  {"thickness_range": (3, 7),  "color_bgr": (50, 70, 110),  "blur": 1.2},
    }
    params = type_params.get(obj_type, type_params["cable"])
    thickness = rng.integers(*params["thickness_range"])
    base_color = np.array(params["color_bgr"], dtype=np.float32)

    # Draw thick object on canvas (with slight color variation along length)
    for i in range(len(centerline_pts) - 1):
        pt1 = tuple(np.clip(centerline_pts[i], [0, 0], [w - 1, h - 1]))
        pt2 = tuple(np.clip(centerline_pts[i + 1], [0, 0], [w - 1, h - 1]))
        color_jitter = base_color + rng.uniform(-8, 8, 3)
        color = tuple(int(c) for c in np.clip(color_jitter, 0, 255))
        cv2.line(canvas, pt1, pt2, color, thickness)
        cv2.line(mask, pt1, pt2, 255, thickness)

    # Draw skeleton (1-pixel centerline) — ground truth
    for i in range(len(centerline_pts) - 1):
        pt1 = tuple(np.clip(centerline_pts[i], [0, 0], [w - 1, h - 1]))
        pt2 = tuple(np.clip(centerline_pts[i + 1], [0, 0], [w - 1, h - 1]))
        cv2.line(skeleton, pt1, pt2, 255, 1)

    # Apply slight blur to the object layer for realism
    if params["blur"] > 0:
        object_layer = np.zeros_like(canvas)
        for i in range(len(centerline_pts) - 1):
            pt1 = tuple(np.clip(centerline_pts[i], [0, 0], [w - 1, h - 1]))
            pt2 = tuple(np.clip(centerline_pts[i + 1], [0, 0], [w - 1, h - 1]))
            cv2.line(object_layer, pt1, pt2, tuple(int(c) for c in base_color), thickness)
        blurred_obj = cv2.GaussianBlur(object_layer, (0, 0), params["blur"])
        obj_mask_3ch = np.stack([mask // 255] * 3, axis=-1)
        canvas = np.where(obj_mask_3ch > 0, blurred_obj, canvas)

    return canvas, mask, skeleton, centerline_pts


# ── Main generation function ─────────────────────────────────────────────────

def generate_dataset(output_dir: str, n_images: int = 60,
                     image_size: tuple = (480, 640),
                     seed: int = 42) -> dict:
    """
    Generate a full synthetic underwater curvilinear dataset.

    Structure:
        output_dir/
            images/          Raw underwater images (BGR)
            masks/           Binary instance masks
            skeletons/       Ground-truth 1px centerlines
            annotations.json Metadata per image
    """
    rng = np.random.default_rng(seed)
    h, w = image_size
    out = Path(output_dir)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(exist_ok=True)
    (out / "skeletons").mkdir(exist_ok=True)

    obj_types = ["cable", "rope", "pipeline", "mooring"]
    annotations = []

    for idx in range(n_images):
        depth = rng.uniform(2.0, 15.0)
        turbidity_sigma = rng.uniform(0.5, 2.5)
        noise_std = rng.uniform(4.0, 14.0)
        n_objects = rng.integers(1, 4)  # 1–3 objects per image

        # Background
        bg = generate_background(h, w, depth)

        # Object layers
        canvas = bg.copy()
        mask_combined = np.zeros((h, w), dtype=np.uint8)
        skeleton_combined = np.zeros((h, w), dtype=np.uint8)
        objects_meta = []

        for obj_idx in range(n_objects):
            obj_type = rng.choice(obj_types)
            mask_single = np.zeros((h, w), dtype=np.uint8)
            skel_single = np.zeros((h, w), dtype=np.uint8)
            canvas, mask_single, skel_single, pts = draw_curvilinear_object(
                canvas, mask_single, skel_single, rng, h, w, obj_type
            )
            mask_combined = np.maximum(mask_combined, mask_single)
            skeleton_combined = np.maximum(skeleton_combined, skel_single)
            objects_meta.append({
                "id": obj_idx,
                "type": obj_type,
                "pixel_count": int(mask_single.sum() // 255),
                "skeleton_length": int(skel_single.sum() // 255),
            })

        # Apply underwater degradation pipeline
        canvas = apply_backscatter(canvas, intensity=rng.uniform(0.04, 0.12))
        canvas = apply_turbidity(canvas, sigma=turbidity_sigma)
        canvas = add_speckle_noise(canvas, std=noise_std)

        # Save
        fname = f"{idx:04d}"
        cv2.imwrite(str(out / "images" / f"{fname}.png"), canvas)
        cv2.imwrite(str(out / "masks" / f"{fname}.png"), mask_combined)
        cv2.imwrite(str(out / "skeletons" / f"{fname}.png"), skeleton_combined)

        annotations.append({
            "id": idx,
            "filename": f"{fname}.png",
            "depth": round(float(depth), 2),
            "turbidity_sigma": round(float(turbidity_sigma), 2),
            "noise_std": round(float(noise_std), 2),
            "n_objects": int(n_objects),
            "objects": objects_meta,
        })

    with open(out / "annotations.json", "w") as f:
        json.dump({"dataset": "synthetic_underwater_curvilinear",
                   "version": "1.0",
                   "n_images": n_images,
                   "image_size": list(image_size),
                   "images": annotations}, f, indent=2)

    print(f"[Generator] Created {n_images} images → {output_dir}")
    return {"n_images": n_images, "output_dir": str(out)}


if __name__ == "__main__":
    generate_dataset("dataset", n_images=60)
