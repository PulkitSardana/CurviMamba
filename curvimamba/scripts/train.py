"""
train.py
========
Main training script for CurviMamba.

Usage:
    # Quick smoke test (toy data, small model):
    python train.py --mode toy --epochs 10 --img_size 128

    # Full training on built dataset:
    python train.py --mode dataset --data_root /data/curvilinear_dataset \
        --epochs 100 --img_size 640 --batch_size 8

    # Resume from checkpoint:
    python train.py --resume checkpoints/best.pth --epochs 50
"""

import os
import json
import time
import argparse
import random
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(args, device):
    """Build the CurviMamba model."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from models.curvimamba import CurviMamba

    model = CurviMamba(
        in_channels=3,
        embed_dims=args.embed_dims,
        depths=args.depths,
        neck_channels=args.neck_channels,
        num_classes=1,
        loss_weights=dict(
            bce=args.w_bce,
            dice=args.w_dice,
            ec=args.w_ec,
            ph=args.w_ph,
        ),
        sigmas=args.frangi_sigmas,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[Model] CurviMamba — {n_params:.2f}M parameters")
    return model


def build_dataloaders(args):
    """Build train/val dataloaders."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))

    if args.mode == "toy":
        from experiments.ablation_study import ToyDataset
        train_ds = ToyDataset(n=500,  size=args.img_size, seed=0)
        val_ds   = ToyDataset(n=100,  size=args.img_size, seed=42)
    else:
        from dataset.dataset_builder import CurvilinearDataset
        train_ds = CurvilinearDataset(args.data_root, "train",
                                       (args.img_size, args.img_size), augment=True)
        val_ds   = CurvilinearDataset(args.data_root, "val",
                                       (args.img_size, args.img_size), augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=True)

    print(f"[Data] Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


@torch.no_grad()
def validate(model, loader, device) -> dict:
    from experiments.ablation_study import (
        iou_score, dice_score, cl_dice_score, ec_match_rate, betti0_accuracy
    )
    model.eval()
    metrics = {"val_loss": [], "iou": [], "dice": [], "cl_dice": [], "ec_match": [], "b0_acc": []}

    for batch in loader:
        img    = batch["image"].to(device)
        target = batch["mask"].to(device)
        cl_gt  = batch["centerline"].to(device)

        out = model(img, target)
        metrics["val_loss"].append(out["loss"].item())

        pred = out["mask"]
        pred_cl = out["centerline"]

        for b in range(img.shape[0]):
            metrics["iou"].append(iou_score(pred[b:b+1], target[b:b+1]))
            metrics["dice"].append(dice_score(pred[b:b+1], target[b:b+1]))
            metrics["cl_dice"].append(cl_dice_score(
                pred[b:b+1], target[b:b+1], pred_cl[b:b+1], cl_gt[b:b+1]))

        metrics["ec_match"].append(ec_match_rate(pred, target))
        metrics["b0_acc"].append(betti0_accuracy(pred, target))

    return {k: float(np.mean(v)) for k, v in metrics.items()}


def train(args):
    set_seed(args.seed)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
        if args.device == "auto" else args.device
    )
    print(f"[Train] Device: {device}")

    out_dir  = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir  = out_dir / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir))

    model = build_model(args, device)
    train_loader, val_loader = build_dataloaders(args)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Resume
    start_epoch = 0
    best_cldice = 0.0
    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_cldice = ckpt.get("best_cldice", 0.0)
        print(f"[Resume] From epoch {start_epoch}, best clDice={best_cldice:.3f}")

    # Save config
    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n[Train] Starting — {args.epochs} epochs")
    print(f"  Loss weights: BCE={args.w_bce}  Dice={args.w_dice}  "
          f"EC={args.w_ec}  PH={args.w_ph}")

    for epoch in range(start_epoch, args.epochs):
        # ---------- Train ----------
        model.train()
        epoch_loss   = 0.0
        loss_comps   = {"bce": 0., "dice": 0., "ec": 0., "ph": 0.}
        t0 = time.perf_counter()

        for step, batch in enumerate(train_loader):
            img    = batch["image"].to(device)
            target = batch["mask"].to(device)

            optimizer.zero_grad()
            out = model(img, target)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += out["loss"].item()
            for k in loss_comps:
                loss_comps[k] += out["loss_components"].get(k, 0.0)

        scheduler.step()

        n_steps   = len(train_loader)
        epoch_loss /= n_steps
        for k in loss_comps:
            loss_comps[k] /= n_steps
        t_epoch = time.perf_counter() - t0

        # ---------- Validate ----------
        val_metrics = validate(model, val_loader, device)

        # ---------- Logging ----------
        writer.add_scalar("train/loss_total", epoch_loss, epoch)
        for k, v in loss_comps.items():
            writer.add_scalar(f"train/loss_{k}", v, epoch)
        for k, v in val_metrics.items():
            writer.add_scalar(f"val/{k}", v, epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        improved = val_metrics["cl_dice"] > best_cldice
        if improved:
            best_cldice = val_metrics["cl_dice"]
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "best_cldice": best_cldice,
                "val_metrics": val_metrics,
            }, ckpt_dir / "best.pth")

        # Save latest
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_cldice": best_cldice,
        }, ckpt_dir / "latest.pth")

        marker = " ← best" if improved else ""
        print(
            f"Ep {epoch+1:3d}/{args.epochs} | "
            f"loss={epoch_loss:.4f} "
            f"(bce={loss_comps['bce']:.3f} "
            f"dice={loss_comps['dice']:.3f} "
            f"ec={loss_comps['ec']:.3f} "
            f"ph={loss_comps['ph']:.3f}) | "
            f"IoU={val_metrics['iou']:.3f} "
            f"clDice={val_metrics['cl_dice']:.3f} "
            f"EC={val_metrics['ec_match']:.3f} | "
            f"{t_epoch:.1f}s"
            f"{marker}"
        )

    writer.close()
    print(f"\n[Train] Complete. Best clDice: {best_cldice:.3f}")
    print(f"  Checkpoints: {ckpt_dir}")
    print(f"  TensorBoard: tensorboard --logdir {log_dir}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()

    # Mode
    p.add_argument("--mode",       default="toy",    choices=["toy", "dataset"])
    p.add_argument("--data_root",  default=None,     help="Dataset root (for --mode dataset)")

    # Model
    p.add_argument("--embed_dims", nargs=4, type=int, default=[64, 128, 256, 512])
    p.add_argument("--depths",     nargs=4, type=int, default=[2, 2, 6, 2])
    p.add_argument("--neck_channels", nargs=3, type=int, default=[256, 512, 1024])
    p.add_argument("--frangi_sigmas", nargs="+", type=float, default=[1.0, 2.0, 4.0])

    # Loss weights
    p.add_argument("--w_bce",  type=float, default=1.0)
    p.add_argument("--w_dice", type=float, default=1.0)
    p.add_argument("--w_ec",   type=float, default=0.5)
    p.add_argument("--w_ph",   type=float, default=0.3)

    # Training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=8)
    p.add_argument("--img_size",     type=int,   default=640)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)

    # Misc
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--device",     default="auto")
    p.add_argument("--output_dir", default="./runs/curvimamba")
    p.add_argument("--resume",     default=None)

    args = p.parse_args()
    train(args)
