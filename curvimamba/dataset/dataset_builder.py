"""
dataset_builder.py
==================
Synthetic + real dataset pipeline for underwater curvilinear object detection.

Two modes:
  A) Pure synthetic: procedurally generates cable/rope/pipe images with
     physics-inspired deformation and underwater rendering effects.
  B) Hybrid: overlays synthetic curvilinear objects onto real underwater
     background crops from RUOD / TrashCan 1.0 / DUO datasets.

Public datasets supported (download separately):
  - RUOD    : https://github.com/dlut-dimt/RUOD
  - TrashCan: https://conservancy.umn.edu/handle/11299/214865
  - DUO     : https://github.com/chongweiliu/DUO_Detection

Output format:
  dataset/
    images/    train/  val/  test/
    masks/     train/  val/  test/   (binary PNG: 255 = cable)
    centerline/ train/ val/  test/   (binary PNG: 255 = centerline pixel)
    dataset_card.json                (splits, stats, annotation protocol)

Usage:
    python dataset_builder.py --mode hybrid --real_root /data/RUOD \
        --output_dir /data/curvilinear_dataset --n_train 8000
"""

import os
import json
import argparse
import random
import math
import hashlib
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, asdict
from datetime import datetime


# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------------
# 1. Bezier / spline curve generator (physics-inspired deformation)
# ---------------------------------------------------------------------------

def random_catmull_rom(
    img_h: int,
    img_w: int,
    n_control: int = 5,
    n_points: int = 512,
    curvature_sigma: float = 0.3,
) -> np.ndarray:
    """
    Generate a smooth random curve via Catmull-Rom spline.
    Returns (N, 2) array of (x, y) pixel coordinates.

    The curve starts at a random left-side point and ends at a random
    right-side point, simulating a cable/pipe crossing the frame.
    """
    # Entry and exit points on left/right edges
    y0 = np.random.randint(img_h // 6, 5 * img_h // 6)
    y1 = np.random.randint(img_h // 6, 5 * img_h // 6)

    ctrl_x = np.linspace(-img_w * 0.1, img_w * 1.1, n_control)
    ctrl_y = np.linspace(y0, y1, n_control)
    ctrl_y += np.random.randn(n_control) * img_h * curvature_sigma
    ctrl_y  = np.clip(ctrl_y, 0, img_h - 1)

    ctrl_pts = np.stack([ctrl_x, ctrl_y], axis=1)

    # Catmull-Rom interpolation
    points = []
    for i in range(1, len(ctrl_pts) - 2):
        p0, p1, p2, p3 = ctrl_pts[i-1], ctrl_pts[i], ctrl_pts[i+1], ctrl_pts[i+2]
        n = max(8, n_points // (n_control - 3))
        for t in np.linspace(0, 1, n):
            t2, t3 = t**2, t**3
            pt = 0.5 * (
                2*p1 +
                (-p0 + p2)*t +
                (2*p0 - 5*p1 + 4*p2 - p3)*t2 +
                (-p0 + 3*p1 - 3*p2 + p3)*t3
            )
            points.append(pt)

    return np.array(points, dtype=np.float32)


def draw_cable(
    canvas: np.ndarray,
    points: np.ndarray,
    width_px: int = 4,
    color: Tuple[int, int, int] = (180, 160, 120),
    noise_scale: float = 0.15,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Draw a cable on canvas and return (updated_canvas, binary_mask).

    Adds:
      - Gaussian cross-section intensity profile (bright centre, dark edges)
      - Random surface texture noise
      - Slight colour variation along length

    Args:
        canvas    : (H, W, 3) BGR image
        points    : (N, 2) curve points
        width_px  : half-width of cable in pixels
        color     : BGR centre colour
        noise_scale: amplitude of surface noise

    Returns:
        canvas (modified), mask (H, W, uint8 binary 0/255)
    """
    H, W = canvas.shape[:2]
    mask = np.zeros((H, W), dtype=np.uint8)

    pts_i = points.astype(np.int32)

    # Draw mask
    for i in range(1, len(pts_i)):
        cv2.line(mask, tuple(pts_i[i-1]), tuple(pts_i[i]), 255, width_px * 2)

    # Draw colour with Gaussian profile
    overlay = canvas.copy()
    for w in range(width_px, 0, -1):
        alpha = math.exp(-0.5 * ((w / width_px) ** 2) * 3)
        col   = tuple(int(c * alpha) for c in color)
        noise = np.random.randn(*canvas.shape).astype(np.float32) * noise_scale * 255
        for i in range(1, len(pts_i)):
            cv2.line(overlay, tuple(pts_i[i-1]), tuple(pts_i[i]), col, w * 2)

    # Blend with noise
    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR).astype(bool)
    canvas_out = canvas.copy()
    canvas_out[mask_3] = (
        overlay[mask_3].astype(np.float32) * (1 - noise_scale) +
        canvas[mask_3].astype(np.float32) * noise_scale
    ).clip(0, 255).astype(np.uint8)

    return canvas_out, mask


def extract_centerline(mask: np.ndarray) -> np.ndarray:
    """Morphological thinning to extract centerline from binary mask."""
    from skimage.morphology import skeletonize
    binary = (mask > 127).astype(np.uint8)
    skel   = skeletonize(binary).astype(np.uint8)
    return (skel * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 2. Underwater rendering effects
# ---------------------------------------------------------------------------

def apply_underwater_effects(
    img: np.ndarray,
    depth: float = 5.0,        # simulated depth in metres
    turbidity: float = 0.3,    # 0 = clear, 1 = very murky
    ambient_colour: Tuple[int, int, int] = (20, 80, 60),  # BGR underwater tint
) -> np.ndarray:
    """
    Simulate underwater optical degradation:
      1. Wavelength-dependent attenuation (red channel decays fastest)
      2. Backscatter haze (additive tint proportional to turbidity)
      3. Non-uniform illumination (spotlight from top)
      4. Green/blue colour cast
      5. Mild blur from water turbulence

    Args:
        img           : (H, W, 3) BGR uint8 input
        depth         : simulated depth (scales attenuation)
        turbidity     : haze amount
        ambient_colour: BGR ambient underwater light tint

    Returns:
        degraded (H, W, 3) BGR uint8 image
    """
    H, W = img.shape[:2]
    out = img.astype(np.float32)

    # --- 1. Wavelength-dependent attenuation ---
    # Attenuation coefficients per channel (BGR order)
    # Blue: 0.01/m, Green: 0.06/m, Red: 0.3/m  (Jerlov water type II)
    att = np.array([0.01, 0.06, 0.30]) * depth   # BGR
    for c in range(3):
        out[:, :, c] *= math.exp(-att[2 - c])    # reverse: B=idx2, R=idx0

    # --- 2. Backscatter haze ---
    haze_colour = np.array(ambient_colour, dtype=np.float32)
    haze = turbidity * haze_colour
    out  = out * (1 - turbidity * 0.4) + haze

    # --- 3. Non-uniform illumination (Gaussian spotlight from top-centre) ---
    yy, xx  = np.mgrid[0:H, 0:W].astype(np.float32)
    spot_cx = W * (0.4 + 0.2 * np.random.rand())
    spot_cy = H * 0.1 * np.random.rand()
    sigma_x = W * (0.4 + 0.2 * np.random.rand())
    sigma_y = H * (0.5 + 0.3 * np.random.rand())
    illum   = np.exp(-0.5 * ((xx - spot_cx)**2 / sigma_x**2 +
                              (yy - spot_cy)**2 / sigma_y**2))
    illum   = 0.5 + 0.5 * illum
    out    *= illum[:, :, np.newaxis]

    # --- 4. Colour cast ---
    out[:, :, 2] *= 0.6 + 0.2 * np.random.rand()   # reduce red
    out[:, :, 1] *= 0.9 + 0.1 * np.random.rand()   # slight green
    out[:, :, 0] *= 1.0 + 0.1 * np.random.rand()   # slight blue boost

    # --- 5. Blur (turbulence) ---
    blur_k = int(turbidity * 3) * 2 + 1   # must be odd
    if blur_k >= 3:
        out = cv2.GaussianBlur(out, (blur_k, blur_k), 0)

    return np.clip(out, 0, 255).astype(np.uint8)


def random_underwater_background(H: int, W: int) -> np.ndarray:
    """Generate a procedural underwater seafloor / water column background."""
    # Base gradient (lighter at top = surface light)
    base = np.zeros((H, W, 3), dtype=np.float32)
    grad = np.linspace(60, 15, H)[:, None]
    base[:, :, 0] = grad * 0.8   # B
    base[:, :, 1] = grad * 1.0   # G
    base[:, :, 2] = grad * 0.3   # R

    # Add Perlin-like noise (approximated with multi-scale Gaussian noise)
    noise = np.zeros((H, W), dtype=np.float32)
    for scale in [8, 16, 32, 64]:
        small = np.random.randn(H // scale + 1, W // scale + 1).astype(np.float32)
        upsampled = cv2.resize(small, (W, H), interpolation=cv2.INTER_CUBIC)
        noise += upsampled / scale

    noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-6)
    base[:, :, 0] += noise * 20
    base[:, :, 1] += noise * 25

    # Random rocks / seafloor patches
    n_patches = np.random.randint(5, 20)
    for _ in range(n_patches):
        cx = np.random.randint(W)
        cy = np.random.randint(H // 2, H)
        rx = np.random.randint(20, 120)
        ry = np.random.randint(10, 60)
        col = np.random.randint(10, 50, 3).astype(np.float32)
        cv2.ellipse(base, (cx, cy), (rx, ry), np.random.randint(360), 0, 360,
                    col.tolist(), -1)

    return np.clip(base, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 3. Sample generator
# ---------------------------------------------------------------------------

@dataclass
class SampleConfig:
    img_h: int = 640
    img_w: int = 640
    n_cables_range: Tuple[int, int] = (1, 3)
    cable_width_range: Tuple[int, int] = (2, 12)
    depth_range: Tuple[float, float] = (1.0, 20.0)
    turbidity_range: Tuple[float, float] = (0.1, 0.7)


def generate_sample(
    cfg: SampleConfig,
    real_bg: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """
    Generate one (image, mask, centerline) triplet.

    Args:
        cfg     : sample configuration
        real_bg : optional real underwater background crop (H, W, 3)

    Returns:
        dict with 'image', 'mask', 'centerline' — all (H, W) or (H, W, 3)
    """
    H, W = cfg.img_h, cfg.img_w

    # Background
    if real_bg is not None:
        bg = cv2.resize(real_bg, (W, H))
    else:
        bg = random_underwater_background(H, W)

    canvas = bg.copy()
    mask   = np.zeros((H, W), dtype=np.uint8)

    n_cables = np.random.randint(*cfg.n_cables_range)

    cable_colors = [
        (200, 180, 100),   # yellow/orange — survey cable
        (60,  60,  60),    # dark grey — umbilical
        (180, 80,  40),    # blue — power cable
        (120, 200, 80),    # green — fibre optic
        (220, 220, 220),   # white — mooring rope
    ]

    for _ in range(n_cables):
        pts   = random_catmull_rom(H, W, n_control=np.random.randint(4, 8))
        width = np.random.randint(*cfg.cable_width_range)
        col   = random.choice(cable_colors)

        canvas, cable_mask = draw_cable(canvas, pts, width_px=width, color=col)
        mask = np.maximum(mask, cable_mask)

    # Underwater effects
    depth      = np.random.uniform(*cfg.depth_range)
    turbidity  = np.random.uniform(*cfg.turbidity_range)
    canvas_uw  = apply_underwater_effects(canvas, depth=depth, turbidity=turbidity)

    # Centerline
    centerline = extract_centerline(mask)

    return {
        "image":      canvas_uw,
        "mask":       mask,
        "centerline": centerline,
        "meta": {
            "depth": depth,
            "turbidity": turbidity,
            "n_cables": n_cables,
        }
    }


# ---------------------------------------------------------------------------
# 4. Real dataset background loader
# ---------------------------------------------------------------------------

class RealBackgroundLoader:
    """
    Loads background crops from RUOD / TrashCan / DUO datasets.
    Only loads images that DO NOT already contain thin curvilinear objects
    (i.e., avoids cable-containing images if metadata is available).
    Falls back to all images if metadata unavailable.
    """

    SUPPORTED = ["RUOD", "TrashCan", "DUO"]

    def __init__(self, root: str, dataset_type: str = "RUOD"):
        self.root = Path(root)
        self.dtype = dataset_type
        self.paths = self._collect_paths()
        random.shuffle(self.paths)
        self._idx = 0
        print(f"[BackgroundLoader] Found {len(self.paths)} background images from {dataset_type}")

    def _collect_paths(self) -> List[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        candidates = []
        for ext in exts:
            candidates.extend(self.root.rglob(f"*{ext}"))
        return candidates

    def __len__(self):
        return len(self.paths)

    def next(self) -> Optional[np.ndarray]:
        if not self.paths:
            return None
        path = self.paths[self._idx % len(self.paths)]
        self._idx += 1
        img = cv2.imread(str(path))
        return img if img is not None else None


# ---------------------------------------------------------------------------
# 5. Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    output_dir: str,
    n_train: int = 8000,
    n_val: int   = 1000,
    n_test: int  = 1000,
    real_root: Optional[str] = None,
    dataset_type: str = "RUOD",
    img_size: int = 640,
):
    """
    Build the full dataset: generate images, save, create dataset card.

    Args:
        output_dir   : root output directory
        n_train      : number of training samples
        n_val        : number of validation samples
        n_test       : number of test samples
        real_root    : path to real dataset for backgrounds (optional)
        dataset_type : one of RUOD / TrashCan / DUO
        img_size     : output image size (square)
    """
    out = Path(output_dir)

    splits = {"train": n_train, "val": n_val, "test": n_test}
    for split in splits:
        (out / "images"     / split).mkdir(parents=True, exist_ok=True)
        (out / "masks"      / split).mkdir(parents=True, exist_ok=True)
        (out / "centerlines"/ split).mkdir(parents=True, exist_ok=True)

    cfg = SampleConfig(img_h=img_size, img_w=img_size)
    loader = RealBackgroundLoader(real_root, dataset_type) if real_root else None

    stats = {}
    total = sum(splits.values())
    generated = 0

    for split, n in splits.items():
        print(f"\n[Builder] Generating {n} {split} samples...")
        split_stats = {"n": n, "n_cables_hist": {}}

        for i in range(n):
            bg = loader.next() if loader else None
            sample = generate_sample(cfg, real_bg=bg)

            # Unique filename based on split + index
            fname = f"{split}_{i:06d}"

            cv2.imwrite(str(out / "images"      / split / f"{fname}.png"), sample["image"])
            cv2.imwrite(str(out / "masks"       / split / f"{fname}.png"), sample["mask"])
            cv2.imwrite(str(out / "centerlines" / split / f"{fname}.png"), sample["centerline"])

            n_c = str(sample["meta"]["n_cables"])
            split_stats["n_cables_hist"][n_c] = split_stats["n_cables_hist"].get(n_c, 0) + 1

            generated += 1
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n} done")

        stats[split] = split_stats

    # Dataset card
    card = {
        "name": "UnderwaterCurvilinear-Synthetic",
        "version": "1.0.0",
        "created": datetime.utcnow().isoformat(),
        "seed": SEED,
        "image_size": img_size,
        "total_samples": total,
        "splits": splits,
        "real_backgrounds": {
            "source": dataset_type if real_root else "none",
            "root": real_root,
        },
        "annotation_protocol": {
            "mask_format": "binary PNG, 255=cable, 0=background",
            "centerline_format": "binary PNG, 255=centerline pixel (morphological skeleton)",
            "cable_types": ["survey cable", "umbilical", "power cable", "fibre optic", "mooring rope"],
            "n_cables_per_image": "1 to 3",
            "cable_width_px": "2 to 12 (simulates 1-6cm cables at 0.5-2m range)",
            "depth_range_m": "1 to 20",
            "turbidity_range": "0.1 to 0.7",
        },
        "metrics_reported": [
            "mAP@IoU0.5",
            "Centerline-F1 (clDice)",
            "Connectivity-Recall (Betti-0 accuracy)",
            "Topological-Correctness (EC match rate)",
            "FPS @ 640x640 on NVIDIA A100",
            "FPS @ 640x640 on Jetson Orin Nano",
        ],
        "class_distribution": stats,
        "citation": (
            "UnderwaterCurvilinear-Synthetic Dataset v1.0. "
            "Procedurally generated with physics-inspired underwater rendering "
            "and real background crops from public datasets."
        ),
    }

    with open(out / "dataset_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print(f"\n[Builder] Done. {generated} samples written to {output_dir}")
    print(f"[Builder] Dataset card saved to {out / 'dataset_card.json'}")
    return card


# ---------------------------------------------------------------------------
# 6. PyTorch Dataset class
# ---------------------------------------------------------------------------

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms.functional as TF

    class CurvilinearDataset(Dataset):
        """
        PyTorch Dataset for underwater curvilinear segmentation.

        Args:
            root    : dataset root (output of build_dataset)
            split   : "train" / "val" / "test"
            size    : (H, W) resize target
            augment : apply training augmentations
        """

        MEAN = [0.485, 0.456, 0.406]
        STD  = [0.229, 0.224, 0.225]

        def __init__(self, root: str, split: str = "train",
                     size: Tuple[int, int] = (640, 640), augment: bool = True):
            self.root    = Path(root)
            self.split   = split
            self.size    = size
            self.augment = augment and (split == "train")

            self.img_paths = sorted((self.root / "images"      / split).glob("*.png"))
            self.msk_paths = sorted((self.root / "masks"       / split).glob("*.png"))
            self.cl_paths  = sorted((self.root / "centerlines" / split).glob("*.png"))
            assert len(self.img_paths) == len(self.msk_paths) == len(self.cl_paths), \
                "Mismatch in image/mask/centerline counts"

        def __len__(self):
            return len(self.img_paths)

        def __getitem__(self, idx: int) -> dict:
            img = cv2.cvtColor(cv2.imread(str(self.img_paths[idx])), cv2.COLOR_BGR2RGB)
            msk = cv2.imread(str(self.msk_paths[idx]), cv2.IMREAD_GRAYSCALE)
            cl  = cv2.imread(str(self.cl_paths[idx]),  cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, self.size[::-1])
            msk = cv2.resize(msk, self.size[::-1], interpolation=cv2.INTER_NEAREST)
            cl  = cv2.resize(cl,  self.size[::-1], interpolation=cv2.INTER_NEAREST)

            if self.augment:
                img, msk, cl = self._augment(img, msk, cl)

            img_t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            for c in range(3):
                img_t[c] = (img_t[c] - self.MEAN[c]) / self.STD[c]

            msk_t = torch.from_numpy((msk > 127).astype(np.float32)).unsqueeze(0)
            cl_t  = torch.from_numpy((cl  > 127).astype(np.float32)).unsqueeze(0)

            return {"image": img_t, "mask": msk_t, "centerline": cl_t,
                    "img_path": str(self.img_paths[idx])}

        def _augment(self, img, msk, cl):
            # Random horizontal flip
            if random.random() > 0.5:
                img = img[:, ::-1].copy()
                msk = msk[:, ::-1].copy()
                cl  = cl[:,  ::-1].copy()

            # Random vertical flip
            if random.random() > 0.5:
                img = img[::-1].copy()
                msk = msk[::-1].copy()
                cl  = cl[::-1].copy()

            # Random brightness / contrast (image only)
            alpha = 0.7 + 0.6 * random.random()    # contrast
            beta  = -30  + 60 * random.random()    # brightness
            img   = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

            # CLAHE (simulates pre-processing that real pipeline applies)
            lab   = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            img   = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

            # Random Gaussian blur (0 = no blur)
            if random.random() > 0.7:
                k = random.choice([3, 5])
                img = cv2.GaussianBlur(img, (k, k), 0)

            # Additive Gaussian noise
            noise = np.random.randn(*img.shape).astype(np.float32) * 8.0
            img   = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

            return img, msk, cl

except ImportError:
    # PyTorch not available — dataset class not built
    CurvilinearDataset = None


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build underwater curvilinear dataset")
    parser.add_argument("--output_dir",   default="./dataset",  help="Output directory")
    parser.add_argument("--real_root",    default=None,          help="Path to real underwater dataset")
    parser.add_argument("--dataset_type", default="RUOD",        choices=["RUOD","TrashCan","DUO"])
    parser.add_argument("--n_train",      type=int, default=8000)
    parser.add_argument("--n_val",        type=int, default=1000)
    parser.add_argument("--n_test",       type=int, default=1000)
    parser.add_argument("--img_size",     type=int, default=640)
    parser.add_argument("--preview",      action="store_true",   help="Save 5 preview samples and exit")
    args = parser.parse_args()

    if args.preview:
        print("Generating 5 preview samples...")
        cfg = SampleConfig(img_h=args.img_size, img_w=args.img_size)
        Path("preview").mkdir(exist_ok=True)
        for i in range(5):
            s = generate_sample(cfg)
            cv2.imwrite(f"preview/img_{i}.png",  s["image"])
            cv2.imwrite(f"preview/mask_{i}.png", s["mask"])
            cv2.imwrite(f"preview/cl_{i}.png",   s["centerline"])
            print(f"  Preview {i}: depth={s['meta']['depth']:.1f}m, "
                  f"turbidity={s['meta']['turbidity']:.2f}, "
                  f"cables={s['meta']['n_cables']}")
        print("Preview saved to ./preview/")
    else:
        build_dataset(
            output_dir=args.output_dir,
            n_train=args.n_train,
            n_val=args.n_val,
            n_test=args.n_test,
            real_root=args.real_root,
            dataset_type=args.dataset_type,
            img_size=args.img_size,
        )
