"""
Classical 7-Stage Pipeline — Main Runner
=========================================
Chains all pipeline stages end-to-end and produces:
  1. Per-image visualisation grids
  2. Metrics table (precision, recall, F1, connectivity, FPS)
  3. Summary report (JSON + printed table)

Usage:
    python run_pipeline.py --dataset_dir dataset --results_dir results
    python run_pipeline.py --dataset_dir dataset --results_dir results --n_images 20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Local imports ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from pipeline.stage12_preprocess  import preprocess
from pipeline.stage34_features    import extract_curvilinear_features
from pipeline.stage56_tracking    import (
    link_skeleton_fragments,
    skeleton_to_ordered_points,
    fit_bspline,
    ransac_polynomial_fit,
    KalmanCurveTracker,
    compute_arc_length,
    ActiveContourSnake,
)
from pipeline.stage7_metrics import (
    filter_by_geometry,
    evaluate_single,
    aggregate_metrics,
    FPSCounter,
)


# ═══════════════════════════════════════════════════════════════════════════
# Single-frame pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_single_frame(img: np.ndarray) -> dict:
    """
    Run the full 7-stage classical pipeline on one frame.

    Returns a dict with intermediate results and the final prediction.
    """
    # ── Stage 1–2: Preprocessing ──────────────────────────────────────────
    preprocessed = preprocess(img, apply_dehaze=True, apply_denoise=True)

    # ── Stage 3–4: Feature Extraction ────────────────────────────────────
    features = extract_curvilinear_features(
        preprocessed,
        frangi_scales=[1.5, 3.0, 5.0],   # 3 scales: good speed/quality tradeoff
        steger_sigma=2.0,
        use_path_opening=False,
    )

    # ── Stage 5: Contour Linking ──────────────────────────────────────────
    # Combine Steger + Frangi skeleton votes
    combined_skel = np.maximum(features["steger"], features["skeleton"])
    linked_skel = link_skeleton_fragments(combined_skel, max_gap_pixels=25)

    # ── Stage 6: Shape Model Fitting ─────────────────────────────────────
    pts = skeleton_to_ordered_points(linked_skel, max_points=400)
    fitted_curve = None
    if len(pts) >= 4:
        fitted_curve = fit_bspline(pts, n_output=300, smoothing=len(pts) * 0.5)
        if fitted_curve is None:
            fitted_curve = ransac_polynomial_fit(pts, degree=3)

    # ── Stage 7: Post-processing ──────────────────────────────────────────
    final_skeleton = filter_by_geometry(linked_skel, min_arc_length=10,
                                         min_aspect_ratio=1.5)

    return {
        "preprocessed":   preprocessed,
        "vesselness":      features["vesselness"],
        "vesselness_bin":  features["vesselness_bin"],
        "edges":           features["edges"],
        "steger":          features["steger"],
        "combined_skel":   combined_skel,
        "linked_skel":     linked_skel,
        "final_skeleton":  final_skeleton,
        "fitted_curve":    fitted_curve,
        "ordered_pts":     pts,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def visualise_result(img: np.ndarray, result: dict, gt_skeleton: np.ndarray,
                      metrics: dict, save_path: str):
    """
    Save a 3×3 visualisation grid showing each pipeline stage.
    """
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.2)

    def show(ax, data, title, cmap="gray", vmin=None, vmax=None):
        ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 0
    show(fig.add_subplot(gs[0, 0]), cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
         "1. Input (underwater)", cmap=None)
    show(fig.add_subplot(gs[0, 1]),
         cv2.cvtColor(result["preprocessed"], cv2.COLOR_BGR2RGB),
         "2. Preprocessed\n(WB + Dehaze + CLAHE)", cmap=None)
    show(fig.add_subplot(gs[0, 2]), result["vesselness"],
         "3. Frangi Vesselness", cmap="hot")

    # Row 1
    show(fig.add_subplot(gs[1, 0]), result["edges"],
         "3b. Canny Edges")
    show(fig.add_subplot(gs[1, 1]), result["steger"],
         "4. Steger Centerlines")
    show(fig.add_subplot(gs[1, 2]), result["linked_skel"],
         "5. Linked Skeleton")

    # Row 2: GT overlay, final prediction, fitted curve
    ax_gt = fig.add_subplot(gs[2, 0])
    gt_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    gt_overlay[gt_skeleton > 0] = [0, 255, 0]
    ax_gt.imshow(gt_overlay); ax_gt.set_title("GT Skeleton (green)", fontsize=9, fontweight="bold"); ax_gt.axis("off")

    ax_pred = fig.add_subplot(gs[2, 1])
    pred_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    pred_overlay[result["final_skeleton"] > 0] = [255, 80, 80]
    ax_pred.imshow(pred_overlay); ax_pred.set_title("6–7. Final Prediction (red)", fontsize=9, fontweight="bold"); ax_pred.axis("off")

    ax_fit = fig.add_subplot(gs[2, 2])
    fit_overlay = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()
    if result["fitted_curve"] is not None:
        curve = result["fitted_curve"].astype(int)
        h, w = fit_overlay.shape[:2]
        for i in range(len(curve) - 1):
            x0, y0 = np.clip(curve[i],   [0, 0], [w-1, h-1])
            x1, y1 = np.clip(curve[i+1], [0, 0], [w-1, h-1])
            cv2.line(fit_overlay, (x0, y0), (x1, y1), (80, 200, 255), 2)
    ax_fit.imshow(fit_overlay)
    ax_fit.set_title("6. B-Spline Fit (cyan)", fontsize=9, fontweight="bold")
    ax_fit.axis("off")

    # Metrics annotation
    m_text = (f"P={metrics['precision']:.3f}  R={metrics['recall']:.3f}  "
              f"F1={metrics['f1']:.3f}  Conn={metrics['connectivity']:.3f}")
    fig.suptitle(f"Classical 7-Stage Pipeline  |  {m_text}", fontsize=11,
                 fontweight="bold", y=0.98)

    plt.savefig(save_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics Table Visualisation
# ═══════════════════════════════════════════════════════════════════════════

def plot_metrics_table(per_image_results: list, agg: dict,
                        fps_summary: dict, save_path: str):
    """Save a publication-quality metrics summary figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics_of_interest = ["precision", "recall", "f1", "connectivity"]
    labels = ["Precision", "Recall", "F1", "Connectivity"]
    means = [agg[f"{m}_mean"] for m in metrics_of_interest]
    stds  = [agg[f"{m}_std"]  for m in metrics_of_interest]

    # Bar chart
    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, means, yerr=stds, capsize=5,
                  color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"],
                  alpha=0.85, width=0.55)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score"); ax.set_title("Baseline Metrics (mean ± std)", fontweight="bold")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{mean:.3f}", ha="center", va="bottom", fontsize=9)

    # Per-image F1 distribution
    ax2 = axes[1]
    f1_vals = [r["f1"] for r in per_image_results]
    ax2.hist(f1_vals, bins=15, color="#4C72B0", alpha=0.75, edgecolor="white")
    ax2.axvline(np.mean(f1_vals), color="#C44E52", lw=2, ls="--",
                label=f"Mean={np.mean(f1_vals):.3f}")
    ax2.set_xlabel("F1 Score"); ax2.set_ylabel("Count")
    ax2.set_title("Per-Image F1 Distribution", fontweight="bold")
    ax2.legend(fontsize=9)

    # Connectivity vs F1 scatter
    ax3 = axes[2]
    conn_vals = [r["connectivity"] for r in per_image_results]
    ax3.scatter(f1_vals, conn_vals, alpha=0.6, color="#55A868", s=40)
    ax3.set_xlabel("F1 Score"); ax3.set_ylabel("Connectivity Score")
    ax3.set_title("Connectivity vs F1\n(each dot = one image)", fontweight="bold")
    ax3.set_xlim(0, 1); ax3.set_ylim(0, 1)

    fps_text = f"Throughput: {fps_summary['fps']:.1f} FPS  |  {fps_summary['mean_ms']:.0f} ms/frame"
    fig.suptitle(f"Classical Pipeline Baseline Results  |  {fps_text}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Classical 7-stage underwater curvilinear pipeline")
    parser.add_argument("--dataset_dir", default="dataset",  help="Root of the dataset")
    parser.add_argument("--results_dir", default="results",  help="Where to save results")
    parser.add_argument("--n_images",    type=int, default=None, help="Limit number of images")
    parser.add_argument("--vis_every",   type=int, default=5,    help="Save visualisation every N images")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    results_dir = Path(args.results_dir)
    (results_dir / "vis").mkdir(parents=True, exist_ok=True)

    image_dir    = dataset_dir / "images"
    skeleton_dir = dataset_dir / "skeletons"

    image_paths = sorted(image_dir.glob("*.png"))
    if args.n_images:
        image_paths = image_paths[:args.n_images]

    if not image_paths:
        print(f"[ERROR] No images found in {image_dir}")
        sys.exit(1)

    print(f"[Pipeline] Processing {len(image_paths)} images...")

    fps_counter = FPSCounter()
    per_image_results = []

    tracker = KalmanCurveTracker()  # shared across frames (video mode)

    for i, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Could not read {img_path}")
            continue

        gt_path = skeleton_dir / img_path.name
        gt_skel = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE) if gt_path.exists() else \
                  np.zeros(img.shape[:2], dtype=np.uint8)

        # ── Time the full pipeline ─────────────────────────────────────────
        fps_counter.start()
        result = run_single_frame(img)
        fps_counter.stop()

        # ── Kalman update ──────────────────────────────────────────────────
        pts = result["ordered_pts"]
        if len(pts) > 1:
            centroid_x = float(pts[:, 0].mean())
            centroid_y = float(pts[:, 1].mean())
            arc_len = compute_arc_length(pts)
            if not tracker.initialised:
                tracker.initialise(centroid_x, centroid_y, arc_len)
            else:
                tracker.predict()
                tracker.update(centroid_x, centroid_y, arc_len)

        # ── Evaluate ──────────────────────────────────────────────────────
        metrics = evaluate_single(result["final_skeleton"], gt_skel)
        metrics["image_id"] = img_path.stem
        per_image_results.append(metrics)

        # ── Visualise (every N frames) ────────────────────────────────────
        if i % args.vis_every == 0:
            vis_path = str(results_dir / "vis" / f"{img_path.stem}_pipeline.png")
            visualise_result(img, result, gt_skel, metrics, vis_path)
            print(f"  [{i+1}/{len(image_paths)}] {img_path.stem} | "
                  f"F1={metrics['f1']:.3f} Conn={metrics['connectivity']:.3f} | "
                  f"Saved vis → {vis_path}")
        else:
            print(f"  [{i+1}/{len(image_paths)}] {img_path.stem} | "
                  f"F1={metrics['f1']:.3f} Conn={metrics['connectivity']:.3f}")

    # ── Aggregate ─────────────────────────────────────────────────────────
    agg = aggregate_metrics(per_image_results)
    fps_summary = fps_counter.summary()

    print("\n" + "═" * 60)
    print("  BASELINE METRICS SUMMARY")
    print("═" * 60)
    print(f"  Images evaluated : {len(per_image_results)}")
    print(f"  Precision        : {agg['precision_mean']:.4f} ± {agg['precision_std']:.4f}")
    print(f"  Recall           : {agg['recall_mean']:.4f} ± {agg['recall_std']:.4f}")
    print(f"  F1 Score         : {agg['f1_mean']:.4f} ± {agg['f1_std']:.4f}")
    print(f"  Connectivity     : {agg['connectivity_mean']:.4f} ± {agg['connectivity_std']:.4f}")
    print(f"  Euler Error      : {agg['euler_error_mean']:.2f} ± {agg['euler_error_std']:.2f}")
    print(f"  Throughput       : {fps_summary['fps']:.1f} FPS  ({fps_summary['mean_ms']:.0f} ms/frame)")
    print("═" * 60)

    # ── Save metrics table figure ─────────────────────────────────────────
    metrics_fig_path = str(results_dir / "metrics_table.png")
    plot_metrics_table(per_image_results, agg, fps_summary, metrics_fig_path)
    print(f"\n[Pipeline] Metrics figure saved → {metrics_fig_path}")

    # ── Save JSON report ──────────────────────────────────────────────────
    report = {
        "pipeline": "classical_7stage_baseline",
        "n_images": len(per_image_results),
        "aggregate_metrics": agg,
        "fps": fps_summary,
        "per_image": per_image_results,
    }
    report_path = results_dir / "baseline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[Pipeline] Report saved → {report_path}")

    return report


if __name__ == "__main__":
    main()
