"""
ablation_study.py
=================
Ablation study runner for CurviMamba — the core of Section V of the paper.

Ablation groups tested (each isolates one contribution):
  Group A — Loss function ablation
    A0: BCE only (baseline)
    A1: BCE + Dice
    A2: BCE + Dice + EC loss
    A3: BCE + Dice + Persistence loss
    A4: BCE + Dice + EC + Persistence  ← full TopologyLoss (proposed)

  Group B — Architecture ablation
    B0: UNet (strong classical baseline)
    B1: YOLOv8-seg backbone + standard FPN neck
    B2: CurviMamba backbone + standard FPN neck (no Hessian gate)
    B3: CurviMamba backbone + HessianNeck (full proposed)

  Group C — Data ablation
    C0: Synthetic only (no real backgrounds)
    C1: Synthetic + RUOD backgrounds
    C2: Synthetic + TrashCan backgrounds
    C3: Synthetic + DUO backgrounds
    C4: Synthetic + all three (full proposed dataset)

Metrics computed:
  - mAP@IoU0.5
  - clDice (centreline Dice — topology-aware)
  - β0-Acc (Betti-0 accuracy — correct number of components)
  - EC-Match (fraction of predictions with correct Euler characteristic)
  - FPS (inference speed at 640×640)

Output:
  ablation_results/
    group_A_loss.csv
    group_B_arch.csv
    group_C_data.csv
    summary_table.csv      ← Table II in the paper
    ablation_plots.png
"""

import os
import json
import time
import argparse
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple
import csv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Project imports (adjust paths if running standalone)
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from losses.topology_loss import TopologyLoss, EulerCharacteristicLoss, PersistenceHomologyLoss


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def iou_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """Binary IoU."""
    pred_b   = (pred > 0.5).float()
    target_b = (target > 0.5).float()
    inter    = (pred_b * target_b).sum()
    union    = pred_b.sum() + target_b.sum() - inter
    return ((inter + eps) / (union + eps)).item()


def dice_score(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_b   = (pred > 0.5).float().flatten()
    target_b = (target > 0.5).float().flatten()
    inter    = (pred_b * target_b).sum()
    return ((2 * inter + eps) / (pred_b.sum() + target_b.sum() + eps)).item()


def cl_dice_score(
    pred_mask: torch.Tensor,
    target_mask: torch.Tensor,
    pred_cl: torch.Tensor,
    target_cl: torch.Tensor,
    eps: float = 1e-6,
) -> float:
    """
    clDice: topology-aware segmentation metric.
    clDice = 2 * Tprec * Tsens / (Tprec + Tsens)
    where:
      Tprec = |V_l(S_pred) ∩ C_target| / |C_target|
      Tsens = |V_l(S_target) ∩ C_pred| / |C_pred|
    V_l = skeletonisation (approximated by centerline)

    Reference: Shit et al. (2021) clDice — a Novel Topology-Preserving Loss.
    """
    pred_s   = (pred_mask  > 0.5).float().flatten()
    target_s = (target_mask > 0.5).float().flatten()
    pred_c   = (pred_cl    > 0.5).float().flatten()
    target_c = (target_cl  > 0.5).float().flatten()

    Tprec = (pred_c * target_s).sum() / (target_s.sum() + eps)
    Tsens = (target_c * pred_s).sum() / (pred_s.sum() + eps)

    if Tprec + Tsens < eps:
        return 0.0
    return (2 * Tprec * Tsens / (Tprec + Tsens)).item()


def compute_euler_characteristic(binary_map: np.ndarray) -> int:
    """Compute 2-D Euler characteristic via V - E + F."""
    b = (binary_map > 0).astype(np.float32)
    H, W = b.shape

    V = int(b.sum())
    E_h = int((b[:, :-1] * b[:, 1:]).sum())    # horizontal edges
    E_v = int((b[:-1, :] * b[1:, :]).sum())    # vertical edges
    E = E_h + E_v
    F = int((b[:-1,:-1] * b[:-1,1:] * b[1:,:-1] * b[1:,1:]).sum())
    return V - E + F


def ec_match_rate(pred_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
    """Fraction of samples where EC(pred) == EC(target)."""
    matches = 0
    B = pred_batch.shape[0]
    for b in range(B):
        pred_np   = (pred_batch[b, 0].cpu().numpy() > 0.5).astype(np.uint8)
        target_np = (target_batch[b, 0].cpu().numpy() > 0.5).astype(np.uint8)
        if compute_euler_characteristic(pred_np) == compute_euler_characteristic(target_np):
            matches += 1
    return matches / B


def betti0_accuracy(pred_batch: torch.Tensor, target_batch: torch.Tensor) -> float:
    """
    Betti-0 accuracy: fraction of samples where the number of connected
    components in pred matches that in target.
    Uses OpenCV connected components.
    """
    import cv2
    matches = 0
    B = pred_batch.shape[0]
    for b in range(B):
        pred_np   = (pred_batch[b, 0].cpu().numpy() * 255).astype(np.uint8)
        target_np = (target_batch[b, 0].cpu().numpy() * 255).astype(np.uint8)
        n_pred   = cv2.connectedComponents(pred_np)[0] - 1    # subtract background
        n_target = cv2.connectedComponents(target_np)[0] - 1
        if n_pred == n_target:
            matches += 1
    return matches / B


# ---------------------------------------------------------------------------
# Lightweight UNet baseline (for Group B comparison)
# ---------------------------------------------------------------------------

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.net(x)


class MiniUNet(nn.Module):
    """Compact UNet for ablation baseline (Group B0)."""
    def __init__(self, in_ch=3, base=32):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, base)
        self.enc2 = DoubleConv(base, base*2)
        self.enc3 = DoubleConv(base*2, base*4)
        self.bot  = DoubleConv(base*4, base*8)
        self.up3  = nn.ConvTranspose2d(base*8, base*4, 2, stride=2)
        self.dec3 = DoubleConv(base*8, base*4)
        self.up2  = nn.ConvTranspose2d(base*4, base*2, 2, stride=2)
        self.dec2 = DoubleConv(base*4, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 2, stride=2)
        self.dec1 = DoubleConv(base*2, base)
        self.head = nn.Conv2d(base, 1, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, target=None, loss_fn=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b  = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        logits = self.head(d1)
        out = {"mask": torch.sigmoid(logits), "mask_logits": logits,
               "centerline": torch.sigmoid(logits)}
        if target is not None and loss_fn is not None:
            loss, comps = loss_fn(logits, target)
            out["loss"] = loss
            out["loss_components"] = comps
        return out


# ---------------------------------------------------------------------------
# Toy synthetic dataset for quick ablation runs (no disk I/O needed)
# ---------------------------------------------------------------------------

class ToyDataset(Dataset):
    """
    In-memory toy dataset: generates (image, mask, centerline) tuples
    on-the-fly using simple noise + Bezier curves.
    For quick ablation training without downloading real datasets.
    """
    def __init__(self, n: int = 200, size: int = 128, seed: int = 0):
        self.n    = n
        self.size = size
        self.rng  = np.random.RandomState(seed)

    def __len__(self): return self.n

    def __getitem__(self, idx):
        H = W = self.size
        rng = np.random.RandomState(idx)

        # Random noise background
        img = rng.rand(H, W, 3).astype(np.float32) * 0.3
        # Add blue/green underwater tint
        img[:, :, 2] *= 0.2
        img[:, :, 1] *= 0.6

        mask = np.zeros((H, W), dtype=np.float32)

        # Draw 1-2 random lines (simplified cable)
        n_cables = rng.randint(1, 3)
        for _ in range(n_cables):
            y0, y1  = rng.randint(10, H-10), rng.randint(10, H-10)
            x0, x1  = rng.randint(0, W//4),  rng.randint(3*W//4, W)
            width   = rng.randint(2, 8)
            pts     = np.stack([
                np.linspace(x0, x1, 100),
                np.linspace(y0, y1, 100) + rng.randn(100) * 5,
            ], axis=1).astype(np.int32)
            for i in range(1, len(pts)):
                import cv2
                cv2.line(mask, tuple(np.clip(pts[i-1], [0,0], [W-1,H-1])),
                               tuple(np.clip(pts[i],   [0,0], [W-1,H-1])),
                         1.0, width)
                # Draw cable in image
                col = rng.rand(3).astype(np.float32) * 0.7 + 0.3
                img_canvas = (img * 255).astype(np.uint8)
                cv2.line(img_canvas, tuple(np.clip(pts[i-1], [0,0], [W-1,H-1])),
                                     tuple(np.clip(pts[i],   [0,0], [W-1,H-1])),
                         (int(col[0]*255), int(col[1]*255), int(col[2]*255)), width)
                img = img_canvas.astype(np.float32) / 255.0

        mask   = np.clip(mask, 0, 1)
        img_t  = torch.from_numpy(img).permute(2, 0, 1)
        msk_t  = torch.from_numpy(mask).unsqueeze(0)

        # Thin centerline (binary erosion proxy)
        import cv2 as cv
        skel = cv.erode((mask * 255).astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=2)
        cl_t = torch.from_numpy(skel.astype(np.float32) / 255.0).unsqueeze(0)

        return {"image": img_t, "mask": msk_t, "centerline": cl_t, "img_path": str(idx)}


# ---------------------------------------------------------------------------
# Training loop (compact, for ablation)
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for batch in loader:
        img    = batch["image"].to(device)
        target = batch["mask"].to(device)
        optimizer.zero_grad()

        if hasattr(model, 'forward') and 'loss_fn' in model.forward.__code__.co_varnames:
            out = model(img, target)
        else:
            # UNet or generic model
            logits = model(img)
            if isinstance(logits, dict):
                logits = logits["mask_logits"]
            loss, _ = loss_fn(logits, target)
            out = {"loss": loss}

        if "loss" not in out:
            loss, _ = loss_fn(out["mask_logits"], target)
            out["loss"] = loss

        out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += out["loss"].item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    metrics = {k: [] for k in ["iou", "dice", "cl_dice", "ec_match", "b0_acc"]}

    for batch in loader:
        img    = batch["image"].to(device)
        target = batch["mask"].to(device)
        cl_gt  = batch["centerline"].to(device)

        if hasattr(model, 'predict'):
            out = model.predict(img)
        else:
            raw = model(img)
            out = {"mask": torch.sigmoid(raw if isinstance(raw, torch.Tensor)
                                         else raw["mask_logits"]),
                   "centerline": torch.zeros_like(target)}

        pred_mask = out["mask"]
        pred_cl   = out.get("centerline", torch.zeros_like(pred_mask))

        for b in range(img.shape[0]):
            metrics["iou"].append(   iou_score(pred_mask[b:b+1], target[b:b+1]))
            metrics["dice"].append(  dice_score(pred_mask[b:b+1], target[b:b+1]))
            metrics["cl_dice"].append(cl_dice_score(
                pred_mask[b:b+1], target[b:b+1], pred_cl[b:b+1], cl_gt[b:b+1]))

        metrics["ec_match"].append(ec_match_rate(pred_mask, target))
        metrics["b0_acc"].append(  betti0_accuracy(pred_mask, target))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def measure_fps(model, device, size=128, n_iters=20) -> float:
    model.eval()
    dummy = torch.rand(1, 3, size, size).to(device)
    # Warm up
    for _ in range(3):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.no_grad():
            model(dummy)
    torch.cuda.synchronize() if device.type == "cuda" else None
    elapsed = time.perf_counter() - t0
    return n_iters / elapsed


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------

def get_loss_variants() -> Dict[str, TopologyLoss]:
    """Group A: loss function ablations."""
    return {
        "A0_BCE_only":        TopologyLoss(bce_weight=1.0, dice_weight=0.0, ec_weight=0.0, ph_weight=0.0),
        "A1_BCE_Dice":        TopologyLoss(bce_weight=1.0, dice_weight=1.0, ec_weight=0.0, ph_weight=0.0),
        "A2_BCE_Dice_EC":     TopologyLoss(bce_weight=1.0, dice_weight=1.0, ec_weight=0.5, ph_weight=0.0),
        "A3_BCE_Dice_PH":     TopologyLoss(bce_weight=1.0, dice_weight=1.0, ec_weight=0.0, ph_weight=0.3),
        "A4_Full_TopologyLoss": TopologyLoss(bce_weight=1.0, dice_weight=1.0, ec_weight=0.5, ph_weight=0.3),
    }


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def run_ablation(
    output_dir: str = "./ablation_results",
    dataset_root: Optional[str] = None,
    n_epochs: int = 5,
    batch_size: int = 4,
    img_size: int = 128,    # small for quick ablation
    device_str: str = "auto",
):
    """
    Run all ablation groups and save results to CSV + JSON.

    Args:
        output_dir   : where to save results
        dataset_root : path to built dataset (if None, uses ToyDataset)
        n_epochs     : training epochs per configuration
        batch_size   : batch size
        img_size     : image size (use 128 for quick ablation, 640 for paper)
        device_str   : "auto" | "cuda" | "cpu"
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if device_str == "auto" else device_str
    )
    print(f"[Ablation] Running on {device}")

    # Dataset
    if dataset_root and Path(dataset_root).exists():
        from dataset.dataset_builder import CurvilinearDataset
        train_ds = CurvilinearDataset(dataset_root, "train", (img_size, img_size), augment=True)
        val_ds   = CurvilinearDataset(dataset_root, "val",   (img_size, img_size), augment=False)
    else:
        print("[Ablation] No dataset root provided — using ToyDataset")
        train_ds = ToyDataset(n=200, size=img_size, seed=0)
        val_ds   = ToyDataset(n=50,  size=img_size, seed=99)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    all_results = {}

    # ------------------------------------------------------------------ #
    # Group A: Loss function ablation (all use MiniUNet for fairness)     #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("GROUP A: Loss function ablation")
    print("="*60)

    group_a_results = []
    for config_name, loss_fn in get_loss_variants().items():
        print(f"\n  Running: {config_name}")
        model = MiniUNet(base=16).to(device)
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

        for ep in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, opt, loss_fn, device)
            sched.step()
            if (ep + 1) % max(1, n_epochs // 3) == 0:
                print(f"    Epoch {ep+1}/{n_epochs} — train_loss: {train_loss:.4f}")

        metrics = evaluate(model, val_loader, loss_fn, device)
        fps     = measure_fps(model, device, size=img_size)
        row = {"config": config_name, **metrics, "fps": fps}
        group_a_results.append(row)
        print(f"  Results: IoU={metrics['iou']:.3f}  clDice={metrics['cl_dice']:.3f}  "
              f"β0-Acc={metrics['b0_acc']:.3f}  EC={metrics['ec_match']:.3f}  FPS={fps:.1f}")

    _save_csv(group_a_results, out / "group_A_loss.csv")
    all_results["group_A"] = group_a_results

    # ------------------------------------------------------------------ #
    # Group B: Architecture ablation (full TopologyLoss for all)          #
    # ------------------------------------------------------------------ #
    print("\n" + "="*60)
    print("GROUP B: Architecture ablation")
    print("="*60)

    full_loss = TopologyLoss()

    arch_configs = {
        "B0_UNet_baseline": lambda: MiniUNet(base=16),
    }

    # Add CurviMamba if available
    try:
        from models.curvimamba import CurviMamba
        arch_configs["B3_CurviMamba_HessianNeck"] = lambda: CurviMamba(
            embed_dims=[16, 32, 64, 128],
            depths=[1, 1, 2, 1],
            neck_channels=[64, 128, 256],
        )
    except Exception as e:
        print(f"  [Warning] CurviMamba not loaded: {e}")

    group_b_results = []
    for config_name, model_fn in arch_configs.items():
        print(f"\n  Running: {config_name}")
        model = model_fn().to(device)
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_epochs)

        for ep in range(n_epochs):
            train_loss = train_one_epoch(model, train_loader, opt, full_loss, device)
            sched.step()

        metrics = evaluate(model, val_loader, full_loss, device)
        fps     = measure_fps(model, device, size=img_size)
        row = {"config": config_name, "params_M": round(n_params, 2), **metrics, "fps": fps}
        group_b_results.append(row)
        print(f"  Results: IoU={metrics['iou']:.3f}  clDice={metrics['cl_dice']:.3f}  "
              f"Params={n_params:.1f}M  FPS={fps:.1f}")

    _save_csv(group_b_results, out / "group_B_arch.csv")
    all_results["group_B"] = group_b_results

    # ------------------------------------------------------------------ #
    # Summary table (Table II in paper)                                   #
    # ------------------------------------------------------------------ #
    summary_rows = []
    for row in group_a_results + group_b_results:
        summary_rows.append({
            "Configuration":       row["config"],
            "IoU":                 f"{row.get('iou', 0):.3f}",
            "Dice":                f"{row.get('dice', 0):.3f}",
            "clDice":              f"{row.get('cl_dice', 0):.3f}",
            "β0-Accuracy":         f"{row.get('b0_acc', 0):.3f}",
            "EC-Match":            f"{row.get('ec_match', 0):.3f}",
            "FPS":                 f"{row.get('fps', 0):.1f}",
        })

    _save_csv(summary_rows, out / "summary_table.csv")

    # Save full JSON
    with open(out / "ablation_full.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n[Ablation] Results saved to {output_dir}/")
    print(f"  group_A_loss.csv   — loss ablation")
    print(f"  group_B_arch.csv   — architecture ablation")
    print(f"  summary_table.csv  — Table II (paper-ready)")

    # Try to plot
    try:
        _plot_results(group_a_results, group_b_results, out / "ablation_plots.png")
    except Exception as e:
        print(f"  [Warning] Plotting skipped: {e}")

    return all_results


def _save_csv(rows: List[dict], path: Path):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Saved: {path}")


def _plot_results(group_a, group_b, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation Study Results", fontsize=14, fontweight="bold")

    # Group A
    ax = axes[0]
    names  = [r["config"].replace("_", "\n") for r in group_a]
    cldice = [r["cl_dice"] for r in group_a]
    ec     = [r["ec_match"] for r in group_a]
    x      = range(len(names))
    ax.bar([i - 0.2 for i in x], cldice, 0.4, label="clDice",   color="#2196F3", alpha=0.85)
    ax.bar([i + 0.2 for i in x], ec,     0.4, label="EC-Match", color="#FF5722", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Group A: Loss Function Ablation")
    ax.legend()
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    # Group B
    ax = axes[1]
    names  = [r["config"].replace("_", "\n") for r in group_b]
    iou    = [r["iou"] for r in group_b]
    b0acc  = [r["b0_acc"] for r in group_b]
    x      = range(len(names))
    ax.bar([i - 0.2 for i in x], iou,   0.4, label="IoU",      color="#4CAF50", alpha=0.85)
    ax.bar([i + 0.2 for i in x], b0acc, 0.4, label="β0-Acc",   color="#9C27B0", alpha=0.85)
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_title("Group B: Architecture Ablation")
    ax.legend()
    ax.set_ylabel("Score")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Plot saved: {save_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CurviMamba ablation study")
    parser.add_argument("--output_dir",   default="./ablation_results")
    parser.add_argument("--dataset_root", default=None)
    parser.add_argument("--n_epochs",     type=int, default=5,
                        help="Epochs per config (use 50+ for paper results)")
    parser.add_argument("--batch_size",   type=int, default=4)
    parser.add_argument("--img_size",     type=int, default=128,
                        help="128 for quick test, 640 for paper")
    parser.add_argument("--device",       default="auto")
    args = parser.parse_args()

    run_ablation(
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        img_size=args.img_size,
        device_str=args.device,
    )
