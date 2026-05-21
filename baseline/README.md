# Classical 7-Stage Underwater Curvilinear Detection Pipeline
### Baseline Implementation for Research Paper

> **Status:** Baseline v1.0 — reproduces the classical pipeline for comparison against the proposed hybrid topology-aware method.

---

## Overview

This repository implements the full classical computer vision pipeline for detecting **long, flexible, and highly deformable curvilinear objects** in underwater imagery — cables, ropes, mooring lines, and pipelines.

The 7-stage pipeline follows the architecture established in:
> Modasshir et al., *"A Classical Computer Vision Pipeline for Underwater Detection of Long, Flexible, and Highly Deformable Curvilinear Objects"*, IEEE OCEANS 2022.

This implementation serves as the **reproducible baseline** for the companion research paper, which proposes a topology-aware hybrid classical–Mamba pipeline that addresses the unresolved gaps documented below.

---

## Baseline Results (Synthetic Dataset, 20 images)

| Metric | Mean | Std |
|--------|------|-----|
| Centerline Precision | 0.060 | ±0.060 |
| Centerline Recall | 0.508 | ±0.445 |
| **Centerline F1** | **0.104** | **±0.102** |
| **Connectivity Score** | **0.525** | ±0.487 |
| Euler Error | 156.2 | ±159.4 |
| **Throughput** | **~0.3 FPS** | (CPU) |

### What the numbers tell us

- **High recall, low precision:** The pipeline over-detects — it finds large regions of the image as "ridge-like" due to background texture, but cannot cleanly isolate the true cable structure.
- **Connectivity = 0.525:** The pipeline correctly covers roughly half of GT cable components end-to-end. The other half are fragmented or missed entirely — precisely Gap 1 identified in the problem statement.
- **Euler error = 156:** The predicted topology has far more connected components than ground truth — characteristic fragmentation failure of classical pipelines.
- **0.3 FPS on CPU:** Too slow for real-time AUV use without GPU acceleration — motivates the TensorRT/Jetson deployment track of the proposed method.

---

## Pipeline Architecture

```
Raw Frame (BGR)
      │
      ▼
┌─────────────────────────────────────┐
│  Stage 1–2: Preprocessing           │
│  • Gray-world white balance         │
│  • Dark Channel Prior dehazing      │
│  • Bilateral filter denoising       │
│  • CLAHE (clip=2.5, tile=8×8)       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 3: Edge & Ridge Detection    │
│  • Canny edges (σ=1.5)              │
│  • Frangi vesselness (σ∈{1.5,3,5}) │
│  • Hessian eigenvalue analysis      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 4: Curvilinear Extraction    │
│  • Steger sub-pixel centerlines     │
│  • Morphological skeletonisation    │
│  • RORPO path-opening (optional)    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 5: Contour Linking & Track   │
│  • Graph-based fragment linking     │
│  • Kalman filter (centroid+length)  │
│  • Lucas-Kanade optical flow        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 6: Shape Model Fitting       │
│  • Active contour (snake)           │
│  • B-spline parametric fitting      │
│  • RANSAC polynomial fitting        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│  Stage 7: Post-Processing           │
│  • Geometric filtering              │
│  • Aspect ratio / arc length filter │
│  • Final labeled mask + centerline  │
└─────────────────────────────────────┘
```

---

## Repository Structure

```
curvilinear_pipeline/
├── pipeline/
│   ├── stage12_preprocess.py   # White balance, dehazing, denoising, CLAHE
│   ├── stage34_features.py     # Frangi, Steger, Hessian, path-opening
│   ├── stage56_tracking.py     # Contour linking, Kalman, optical flow, snakes
│   └── stage7_metrics.py       # Precision, recall, F1, connectivity, FPS
├── data/
│   └── synthetic_generator.py  # Synthetic underwater dataset generator
├── results/
│   ├── vis/                    # Per-image pipeline visualisations
│   ├── metrics_table.png       # Summary metrics figure
│   └── baseline_report.json    # Full per-image results (JSON)
└── run_pipeline.py             # End-to-end runner
```

---

## Installation

```bash
# Python 3.9+
pip install opencv-python scikit-image scipy matplotlib numpy
```

No additional dependencies required — Kalman filter is implemented in pure NumPy.

---

## Usage

### 1. Generate synthetic dataset

```bash
cd curvilinear_pipeline
python data/synthetic_generator.py
# Creates: dataset/images/, dataset/masks/, dataset/skeletons/
```

### 2. Run baseline pipeline

```bash
python run_pipeline.py \
    --dataset_dir dataset \
    --results_dir results \
    --n_images 60 \
    --vis_every 5
```

### 3. Use your own dataset (URPC / DUO / RUOD)

Download dataset, then point `--dataset_dir` to a folder with `images/` and `skeletons/` subdirectories. Annotation format: PNG binary masks (255 = skeleton pixel).

---

## Metrics Definitions

| Metric | Definition |
|--------|-----------|
| **Centerline Precision** | Fraction of predicted skeleton pixels within `tolerance_px=3` of GT skeleton |
| **Centerline Recall** | Fraction of GT skeleton pixels within `tolerance_px=3` of predicted skeleton |
| **F1** | Harmonic mean of precision and recall |
| **Connectivity Score** | Fraction of GT connected components covered ≥50% by predictions within `tolerance_px=5` |
| **Euler Error** | Absolute difference in Euler number (components − holes) between prediction and GT — captures topological fragmentation |

---

## Known Limitations (Research Gaps)

These documented failures motivate the proposed enhanced method:

1. **Topological fragmentation** — Euler error of 156 indicates the pipeline produces hundreds more disconnected fragments than exist in ground truth. No topology-aware loss enforces centerline continuity.

2. **False positive over-detection** — Precision of 0.06 means ~94% of detected pixels are false positives. The Frangi + Steger combination cannot distinguish real cables from textured backgrounds without a learned discriminator.

3. **Speed** — 0.3 FPS on CPU is not viable for real-time AUV operation. The proposed method targets ≥15 FPS on NVIDIA Jetson Orin Nano via TensorRT.

4. **Threshold sensitivity** — Results vary significantly with `turbidity_sigma` and image brightness. A learned front-end should replace manual threshold tuning.

---

## Citation

If you use this baseline implementation, please cite:

```bibtex
@inproceedings{modasshir2022classical,
  title={A Classical Computer Vision Pipeline for Underwater Detection of Long,
         Flexible, and Highly Deformable Curvilinear Objects},
  booktitle={OCEANS 2022 -- Hampton Roads},
  year={2022},
  organization={IEEE}
}
```

---

## Next Steps

This baseline feeds directly into the companion paper's ablation study:

| Ablation Stage | Change from Baseline | Expected Gain |
|----------------|---------------------|---------------|
| + Mamba backbone | Replace Frangi+Steger with Mamba seg head | ↑ Precision, ↑ FPS |
| + Hessian attention | Frangi prior as attention map | ↑ Recall on thin structures |
| + Topology loss | Persistent homology regularisation | ↑ Connectivity, ↓ Euler error |
| Full proposed method | All of the above | Target: F1 ≥ 0.65, Conn ≥ 0.85 |

---

*Baseline version: 1.0 | Dataset: Synthetic underwater curvilinear (60 images) | Evaluated: 20 images*
