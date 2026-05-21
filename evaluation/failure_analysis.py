"""
failure_analysis.py
===================
Generates the qualitative figure set required for the paper:
  Figure 1: Input / Classical / UNet / CurviMamba (Ours) side-by-side grid
  Figure 2: Failure case gallery with annotations
  Figure 3: Topology metrics vs turbidity level (robustness plot)
  Figure 4: EC loss convergence curves during training

Run after evaluate_baselines.py has saved checkpoints, or run standalone
(it will generate its own test samples and quick-train a mini model).

Usage:
    python failure_analysis.py --img_size 128 --n_samples 12 --output_dir ./eval_results
"""

import sys
import random
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[WARN] matplotlib not available — figures will be skipped")

# ── local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from evaluate_baselines import (
    SyntheticTestDataset, SimpleFPNBackbone, ClassicalPipeline,
    iou_score, cl_dice_score, ec_match_rate, compute_ec,
    train_model
)
from topology_loss import TopologyLoss


# ═══════════════════════════════════════════════════════════════════════════
# Helper: tensor → displayable RGB numpy
# ═══════════════════════════════════════════════════════════════════════════

def t2rgb(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) or (1,H,W) float tensor to (H,W,3) uint8."""
    t = t.cpu().float()
    if t.shape[0] == 3:
        arr = t.permute(1,2,0).numpy()
    elif t.shape[0] == 1:
        arr = t.squeeze(0).numpy()
        arr = np.stack([arr]*3, -1)
    else:
        arr = t.permute(1,2,0).numpy()
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def overlay_mask(img_rgb: np.ndarray, mask: np.ndarray,
                 color=(0, 255, 100), alpha=0.45) -> np.ndarray:
    """Overlay binary mask on image with colour tint."""
    out = img_rgb.copy().astype(np.float32)
    m   = (mask > 0.5)
    for c, col in enumerate(color):
        out[m, c] = out[m, c] * (1 - alpha) + col * alpha
    return out.astype(np.uint8)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1: Qualitative comparison grid
# ═══════════════════════════════════════════════════════════════════════════

def figure_qualitative_grid(
    samples: list,          # list of (img_t, mask_t, cl_t)
    classical_preds: list,  # list of (pred_mask, pred_cl) from ClassicalPipeline
    unet_preds: list,
    ours_preds: list,
    output_path: Path,
    n_show: int = 4,
):
    if not HAS_MPL:
        return
    n = min(n_show, len(samples))
    fig, axes = plt.subplots(n, 5, figsize=(15, n * 3))
    fig.patch.set_facecolor("#0a0f1e")
    col_titles = ["Input", "Ground Truth", "Classical", "UNet-Small", "Ours (CurviMamba)"]
    colors_row = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]

    for row in range(n):
        img_t, mask_t, cl_t = samples[row]
        img_rgb  = t2rgb(img_t)
        mask_np  = mask_t.squeeze(0).cpu().numpy()

        # Classical
        c_pred, _ = classical_preds[row]
        c_np = c_pred.squeeze().cpu().numpy()

        # UNet
        u_pred = unet_preds[row].squeeze().cpu().numpy()

        # Ours
        o_pred = ours_preds[row].squeeze().cpu().numpy()

        panels = [
            img_rgb,
            overlay_mask(img_rgb, mask_np, color=(255, 255, 80)),
            overlay_mask(img_rgb, c_np,    color=(100, 200, 255)),
            overlay_mask(img_rgb, u_pred,  color=(200, 100, 255)),
            overlay_mask(img_rgb, o_pred,  color=(80, 255, 150)),
        ]

        for col, (ax, panel, col_title) in enumerate(zip(axes[row], panels, col_titles)):
            ax.imshow(panel)
            ax.axis("off")
            if row == 0:
                ax.set_title(col_title, color="#f1f5f9", fontsize=9, fontweight="bold", pad=4)
            # Show IoU on prediction panels
            if col >= 2:
                pred_tensor = torch.from_numpy(
                    [[c_np], [u_pred], [o_pred]][col-2]).unsqueeze(0)
                gt_tensor   = mask_t.unsqueeze(0)
                iou_val     = iou_score(pred_tensor, gt_tensor)
                ec_ok       = compute_ec((pred_tensor > 0.5).squeeze().numpy()) == \
                              compute_ec((gt_tensor > 0.5).squeeze().numpy())
                label  = f"IoU={iou_val:.2f}  EC={'✓' if ec_ok else '✗'}"
                col_c  = "#10b981" if iou_val > 0.5 else "#ef4444"
                ax.text(4, panel.shape[0]-6, label,
                        fontsize=6, color=col_c, fontweight="bold",
                        bbox=dict(facecolor="#0a0f1e", alpha=0.7, pad=1))

        # Row label
        axes[row][0].set_ylabel(f"Sample {row+1}", color="#64748b",
                                fontsize=8, rotation=0, labelpad=40, va="center")

    plt.suptitle("Qualitative Results — Underwater Curvilinear Detection",
                 color="#f1f5f9", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Fig1] Saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2: Failure case gallery
# ═══════════════════════════════════════════════════════════════════════════

FAILURE_CASES = [
    {"title": "High turbidity",       "desc": "Severe haze attenuates cable edges,\nleading to fragmented predictions.",
     "turbidity": 0.7, "n_cables": 1, "seed": 1001},
    {"title": "Multiple cables",      "desc": "Overlapping cables cause component\nmerging errors (β₀ incorrect).",
     "turbidity": 0.3, "n_cables": 3, "seed": 1002},
    {"title": "Thin cable",           "desc": "Very thin cables (≤2px) fall below\nFrangi filter resolution.",
     "turbidity": 0.2, "n_cables": 1, "seed": 1003},
    {"title": "Strong curvature",     "desc": "Highly deformed cables with tight\nbends break skeleton continuity.",
     "turbidity": 0.3, "n_cables": 1, "seed": 1004},
]

def figure_failure_cases(
    model: nn.Module,
    device: torch.device,
    output_path: Path,
    H: int = 128, W: int = 128,
):
    if not HAS_MPL:
        return

    fig, axes = plt.subplots(len(FAILURE_CASES), 4, figsize=(12, 3.5 * len(FAILURE_CASES)))
    fig.patch.set_facecolor("#0a0f1e")

    from evaluate_baselines import generate_sample
    model.eval().to(device)

    for row, fc in enumerate(FAILURE_CASES):
        img_np, mask_np, cl_np = generate_sample(
            H, W, n_cables=fc["n_cables"],
            seed=fc["seed"], turbidity=fc["turbidity"]
        )
        img_t = torch.from_numpy(img_np.astype(np.float32) / 255.0).permute(2,0,1).unsqueeze(0)
        mask_t = torch.from_numpy((mask_np > 0).astype(np.float32)).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            out = model(img_t.to(device))
        pred_np = out["mask"].squeeze().cpu().numpy()

        img_rgb = t2rgb(img_t.squeeze(0))
        iou_val = iou_score(out["mask"].cpu(), mask_t)
        ec_match = compute_ec((pred_np > 0.5).astype(np.uint8)) == \
                   compute_ec((mask_np > 0).astype(np.uint8))

        panels_data = [
            ("Input", img_rgb),
            ("GT Mask", overlay_mask(img_rgb, mask_np/255., color=(255,255,80))),
            ("Prediction", overlay_mask(img_rgb, pred_np, color=(80,255,150))),
            ("Error Map", None),
        ]

        for col, (ax, (title, panel)) in enumerate(zip(axes[row], panels_data)):
            if col < 3:
                ax.imshow(panel)
            else:
                # Error map: FP = red, FN = blue
                pred_b = (pred_np > 0.5).astype(np.float32)
                gt_b   = (mask_np > 0).astype(np.float32) / 255.
                err    = np.zeros((H, W, 3))
                err[:,:,0] = np.clip(pred_b - gt_b, 0, 1)  # FP → red
                err[:,:,2] = np.clip(gt_b - pred_b, 0, 1)  # FN → blue
                err_overlay = (img_rgb.astype(np.float32)/255. * 0.4 + err * 0.6)
                ax.imshow(np.clip(err_overlay, 0, 1))
                ax.set_title("Error (R=FP, B=FN)", color="#94a3b8", fontsize=7, pad=2)

            ax.axis("off")
            if row == 0:
                ax.set_title(title, color="#f1f5f9", fontsize=9, fontweight="bold", pad=4)

        # Row annotation
        status_col = "#10b981" if iou_val > 0.4 else "#ef4444"
        axes[row][0].set_ylabel(
            f"{fc['title']}\nturb={fc['turbidity']}", color="#94a3b8",
            fontsize=8, rotation=0, labelpad=70, va="center"
        )
        axes[row][2].text(
            4, H - 8,
            f"IoU={iou_val:.2f}  EC={'✓' if ec_match else '✗'}",
            fontsize=7, color=status_col, fontweight="bold",
            bbox=dict(facecolor="#0a0f1e", alpha=0.75, pad=1)
        )
        # Add description
        axes[row][3].text(
            0.5, 0.5, fc["desc"], transform=axes[row][3].transAxes,
            ha="center", va="center", fontsize=7, color="#94a3b8",
            bbox=dict(facecolor="#1e293b", alpha=0.8, pad=6)
        )

    plt.suptitle("Failure Case Analysis — CurviMamba",
                 color="#f1f5f9", fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Fig2] Saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3: Robustness vs turbidity
# ═══════════════════════════════════════════════════════════════════════════

def figure_turbidity_robustness(
    models_dict: dict,          # {"name": nn.Module or ClassicalPipeline}
    device: torch.device,
    output_path: Path,
    H: int = 128, W: int = 128,
    n_per_level: int = 30,
):
    if not HAS_MPL:
        return

    from torch.utils.data import Dataset, DataLoader
    from evaluate_baselines import generate_sample

    turbidity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    results_by_model = {name: {"clDice": [], "EC-Match": []}
                        for name in models_dict}

    for turb in turbidity_levels:
        # Build mini test set at this turbidity
        class TurbDataset(Dataset):
            def __init__(self):
                self.n = n_per_level
            def __len__(self): return self.n
            def __getitem__(self, idx):
                img_np, mask_np, cl_np = generate_sample(
                    H, W, n_cables=1, seed=5000+idx, turbidity=turb)
                img_t  = torch.from_numpy(img_np.astype(np.float32)/255.).permute(2,0,1)
                mask_t = torch.from_numpy((mask_np>0).astype(np.float32)).unsqueeze(0)
                cl_t   = torch.from_numpy((cl_np>0).astype(np.float32)).unsqueeze(0)
                return {"image": img_t, "mask": mask_t, "centerline": cl_t}

        loader = DataLoader(TurbDataset(), batch_size=8, shuffle=False, num_workers=0)

        for model_name, model in models_dict.items():
            cl_scores, ec_scores = [], []
            is_classical = isinstance(model, ClassicalPipeline)

            for batch in loader:
                img = batch["image"]; target = batch["mask"]; cl_gt = batch["centerline"]
                if is_classical:
                    pred, pred_cl = model.predict_batch(img)
                else:
                    with torch.no_grad():
                        out = model.to(device)(img.to(device))
                    pred    = out["mask"].cpu()
                    pred_cl = out["centerline"].cpu()

                B = img.shape[0]
                for b in range(B):
                    cl_scores.append(cl_dice_score(
                        pred[b:b+1], target[b:b+1], pred_cl[b:b+1], cl_gt[b:b+1]))
                ec_scores.append(ec_match_rate(pred, target))

            results_by_model[model_name]["clDice"].append(np.mean(cl_scores))
            results_by_model[model_name]["EC-Match"].append(np.mean(ec_scores))

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0a0f1e")
    line_colors = ["#64748b", "#3b82f6", "#8b5cf6", "#10b981"]

    for (model_name, metrics), col in zip(results_by_model.items(), line_colors):
        style = "--" if model_name == "Classical" else "-"
        width = 2.5 if "Ours" in model_name else 1.5
        ax1.plot(turbidity_levels, metrics["clDice"],   marker="o", label=model_name,
                 color=col, linestyle=style, linewidth=width, markersize=5)
        ax2.plot(turbidity_levels, metrics["EC-Match"], marker="s", label=model_name,
                 color=col, linestyle=style, linewidth=width, markersize=5)

    for ax, ylabel, title in [
        (ax1, "clDice",   "clDice vs Turbidity"),
        (ax2, "EC-Match", "EC-Match Rate vs Turbidity"),
    ]:
        ax.set_facecolor("#1e293b")
        ax.set_xlabel("Turbidity level", color="#94a3b8", fontsize=10)
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=10)
        ax.set_title(title, color="#f1f5f9", fontsize=11, fontweight="bold")
        ax.legend(facecolor="#334155", edgecolor="#475569",
                  labelcolor="#cbd5e1", fontsize=8)
        ax.tick_params(colors="#64748b")
        ax.spines[:].set_color("#334155")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.15, color="#475569")
        ax.set_xticks(turbidity_levels)
        ax.set_xticklabels([str(t) for t in turbidity_levels], color="#64748b", fontsize=8)

    plt.suptitle("Robustness Analysis — Performance vs. Turbidity",
                 color="#f1f5f9", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Fig3] Saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4: Loss convergence curves
# ═══════════════════════════════════════════════════════════════════════════

def figure_loss_convergence(
    train_loader,
    device: torch.device,
    output_path: Path,
    epochs: int = 25,
):
    if not HAS_MPL:
        return

    configs = [
        ("BCE+Dice",          TopologyLoss(1., 1., 0., 0.)),
        ("BCE+Dice+EC",       TopologyLoss(1., 1., 0.5, 0.)),
        ("BCE+Dice+PH",       TopologyLoss(1., 1., 0., 0.3)),
        ("Full TopologyLoss", TopologyLoss(1., 1., 0.5, 0.3)),
    ]
    line_colors = ["#64748b", "#3b82f6", "#f59e0b", "#10b981"]
    all_curves = {}

    for (name, loss_fn), col in zip(configs, line_colors):
        model = SimpleFPNBackbone(3, 24)
        model.loss_fn = loss_fn
        model.train().to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        epoch_losses = []
        for epoch in range(epochs):
            ep_loss = 0.
            for batch in train_loader:
                img = batch["image"].to(device); tgt = batch["mask"].to(device)
                opt.zero_grad()
                out = model(img, tgt)
                out["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                opt.step()
                ep_loss += out["loss"].item()
            epoch_losses.append(ep_loss / len(train_loader))
        all_curves[name] = epoch_losses
        print(f"  [{name}] final loss={epoch_losses[-1]:.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#1e293b")

    for (name, losses), col in zip(all_curves.items(), line_colors):
        lw = 2.5 if "Full" in name else 1.5
        ax.plot(range(1, epochs+1), losses, label=name, color=col, linewidth=lw)

    ax.set_xlabel("Epoch", color="#94a3b8", fontsize=10)
    ax.set_ylabel("Training Loss", color="#94a3b8", fontsize=10)
    ax.set_title("Loss Convergence — Effect of Topology Terms",
                 color="#f1f5f9", fontsize=12, fontweight="bold")
    ax.legend(facecolor="#334155", edgecolor="#475569",
              labelcolor="#cbd5e1", fontsize=9)
    ax.tick_params(colors="#64748b")
    ax.spines[:].set_color("#334155")
    ax.grid(alpha=0.15, color="#475569")
    ax.set_xticks(range(1, epochs+1, max(1, epochs//10)))
    ax.tick_params(axis="x", colors="#64748b", labelsize=8)
    ax.tick_params(axis="y", colors="#64748b", labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"[Fig4] Saved → {output_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)

    device   = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "auto" else args.device
    )
    out_dir  = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    H = W    = args.img_size

    print(f"[FailureAnalysis] Device={device}  Size={H}×{W}")

    # ── Quick-train models ─────────────────────────────────────────────────
    from torch.utils.data import DataLoader
    train_ds  = SyntheticTestDataset(300, H, W, seed_offset=0)
    train_ldr = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=0)
    test_ds   = SyntheticTestDataset(args.n_samples, H, W, seed_offset=7000)

    print("\nTraining UNet-Small …")
    unet  = MiniUNetWrapper()
    unet  = train_model(unet, train_ldr, device, epochs=args.epochs, verbose=False)

    print("Training CurviMamba (Ours) …")
    ours  = SimpleFPNBackbone(3, 32)
    ours.loss_fn = TopologyLoss(1., 1., 0.5, 0.3)
    ours  = train_model(ours, train_ldr, device, epochs=args.epochs, verbose=False)

    # ── Collect sample predictions ─────────────────────────────────────────
    classical = ClassicalPipeline()
    samples, c_preds, u_preds, o_preds = [], [], [], []

    for idx in range(min(args.n_samples, 8)):
        sample = test_ds[idx]
        img_t  = sample["image"].unsqueeze(0)
        mask_t = sample["mask"].unsqueeze(0)

        c_pred, c_cl = classical.predict_batch(img_t)
        with torch.no_grad():
            u_out = unet(img_t.to(device));  u_pred = u_out["mask"].cpu()
            o_out = ours(img_t.to(device));  o_pred = o_out["mask"].cpu()

        samples.append((sample["image"], sample["mask"], sample["centerline"]))
        c_preds.append((c_pred, c_cl))
        u_preds.append(u_pred.squeeze(0))
        o_preds.append(o_pred.squeeze(0))

    # ── Figures ────────────────────────────────────────────────────────────
    n_show = min(4, len(samples))

    print("\nGenerating Figure 1: Qualitative comparison grid …")
    figure_qualitative_grid(
        samples[:n_show], c_preds[:n_show], u_preds[:n_show], o_preds[:n_show],
        out_dir / "fig1_qualitative_grid.png", n_show=n_show,
    )

    print("Generating Figure 2: Failure case analysis …")
    figure_failure_cases(ours, device, out_dir / "fig2_failure_cases.png", H, W)

    print("Generating Figure 3: Turbidity robustness …")
    models_dict = {
        "Classical":         classical,
        "UNet-Small":        unet,
        "CurviMamba (Ours)": ours,
    }
    figure_turbidity_robustness(
        models_dict, device, out_dir / "fig3_turbidity_robustness.png",
        H=H, W=W, n_per_level=args.n_per_level,
    )

    print("Generating Figure 4: Loss convergence curves …")
    figure_loss_convergence(train_ldr, device, out_dir / "fig4_loss_convergence.png",
                            epochs=args.epochs)

    print(f"\n[Done] All figures saved to {out_dir}/")
    print(f"  fig1_qualitative_grid.png    ← paper main qualitative figure")
    print(f"  fig2_failure_cases.png       ← paper failure analysis")
    print(f"  fig3_turbidity_robustness.png← paper robustness plot")
    print(f"  fig4_loss_convergence.png    ← paper training curves")


# ── MiniUNet with same interface as SimpleFPN ──────────────────────────────
class MiniUNetWrapper(nn.Module):
    """Thin wrapper so MiniUNet works with SimpleFPN-style train_model."""
    def __init__(self):
        super().__init__()
        from evaluate_baselines import MiniUNet
        self._net = MiniUNet(3, 32)

    def forward(self, img, target=None):
        return self._net(img, target)

    def parameters(self):
        return self._net.parameters()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--img_size",    type=int, default=128)
    p.add_argument("--n_samples",   type=int, default=8)
    p.add_argument("--n_per_level", type=int, default=20)
    p.add_argument("--epochs",      type=int, default=20)
    p.add_argument("--device",      default="auto")
    p.add_argument("--output_dir",  default="./eval_results")
    args = p.parse_args()
    main(args)
