# CurviMamba 🐍〰️

> **Mamba-SSM + Hessian Gate + TopologyLoss for underwater detection of long, flexible, highly deformable curvilinear objects**

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Journal-blue)](paper/CurviMamba_IEEE_Paper.docx)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](requirements.txt)

---

## Results at a glance

| Method | mAP@0.5 | clDice | β₀-Acc | EC-Match | FPS |
|--------|---------|--------|--------|----------|-----|
| Classical Pipeline | 0.421 | 0.384 | 0.631 | 0.587 | 28.4 |
| UNet-Small | 0.613 | 0.571 | 0.712 | 0.689 | 84.3 |
| SimpleFPN (YOLO) | 0.671 | 0.634 | 0.743 | 0.711 | 112.7 |
| CurviMamba-noTopo | 0.694 | 0.651 | 0.761 | 0.728 | 97.2 |
| **CurviMamba (Ours)** | **0.741** | **0.724** | **0.819** | **0.803** | **89.6** |

---

## Repository structure

```
CurviMamba/
│
├── README.md                        ← you are here
├── requirements.txt                 ← pip dependencies
│
├── configs/
│   └── default.yaml                 ← training hyperparameters
│
├── baseline/                        ← classical 7-stage pipeline (comparison)
│   ├── README.md                    ← baseline-specific docs
│   ├── run_pipeline.py              ← end-to-end runner
│   ├── stage12_preprocess.py        ← CLAHE, dehazing, denoising
│   ├── stage34_features.py          ← Frangi, Steger centerline, RORPO
│   ├── stage56_tracking.py          ← graph linking, Kalman filter, optical flow
│   ├── stage7_metrics.py            ← centerline F1, connectivity, EC error
│   └── synthetic_generator.py      ← synthetic test frames
│
├── curvimamba/                      ← proposed model (pip-installable package)
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── curvimamba.py            ← full model: Mamba + HessianNeck + head
│   │   └── hessian_attention.py    ← Frangi gate + HessianNeck (drop-in FPN)
│   ├── losses/
│   │   ├── __init__.py
│   │   └── topology_loss.py        ← BCE + Dice + EC + PH composite loss
│   ├── dataset/
│   │   ├── __init__.py
│   │   └── dataset_builder.py      ← synthetic + RUOD/TrashCan/DUO hybrid
│   ├── experiments/
│   │   ├── __init__.py
│   │   └── ablation_study.py       ← Groups A / B / C ablations → paper tables
│   └── scripts/
│       └── train.py                ← main training entry point
│
├── evaluation/
│   ├── evaluate_baselines.py        ← runs all 5 methods, produces Table I
│   ├── failure_analysis.py          ← qualitative figures + failure cases
│   ├── evaluation_summary.json     ← raw numerical results
│   ├── table1_baseline_comparison.csv
│   ├── table2_loss_ablation.csv
│   └── table3_arch_ablation.csv
│
├── paper/
│   ├── CurviMamba_IEEE_Paper.docx  ← full draft paper (IEEE format)
│   └── figures/
│       ├── Fig1_Architecture.png   ← system architecture (300 DPI)
│       ├── Fig2_Qualitative.png    ← input/baseline/ours grid
│       ├── Fig3_Quantitative.png   ← bar + radar comparison charts
│       ├── Fig4_Ablation.png       ← ablation study charts
│       ├── Fig5_FailureCases.png   ← failure analysis
│       ├── Fig6_Training.png       ← loss convergence + turbidity robustness
│       ├── fig_table1_baselines.png
│       ├── fig_table2_ablation.png
│       ├── fig_turbidity_robustness.png
│       └── fig_loss_convergence.png
│
└── data/                            ← gitignored, populated by dataset_builder.py
    ├── raw/                         ← original ROV footage / RUOD / TrashCan
    └── processed/                   ← images/ masks/ splits
```

---

## Quick start

```bash
# 1. Install
git clone https://github.com/YOUR_USERNAME/CurviMamba
cd CurviMamba
pip install -r requirements.txt

# 2. Generate synthetic dataset (no downloads needed)
python curvimamba/dataset/dataset_builder.py --preview
python curvimamba/dataset/dataset_builder.py \
    --output_dir ./data/processed \
    --n_train 8000 --n_val 1000 --n_test 1000

# 3. Train CurviMamba (smoke test first)
python curvimamba/scripts/train.py --mode toy --epochs 10 --img_size 128

# 4. Full training
python curvimamba/scripts/train.py \
    --mode dataset --data_root ./data/processed \
    --epochs 100 --img_size 640 --batch_size 8

# 5. Evaluate all baselines → reproduces Table I
python evaluation/evaluate_baselines.py

# 6. Run ablation study → reproduces Tables II & III
python curvimamba/experiments/ablation_study.py \
    --output_dir ./ablation_results --n_epochs 50

# 7. Run classical baseline for comparison
python baseline/run_pipeline.py \
    --dataset_dir ./data/processed --results_dir ./baseline_results
```

---

## Three novel contributions

| # | Module | File | Paper section |
|---|--------|------|---------------|
| 1 | **TopologyLoss** (BCE + Dice + EC + PH) | `curvimamba/losses/topology_loss.py` | §III-E |
| 2 | **HessianNeck** (Frangi → learned gate) | `curvimamba/models/hessian_attention.py` | §III-C |
| 3 | **CurviMamba** (Mamba backbone + above) | `curvimamba/models/curvimamba.py` | §III-D |

---

## Citation

```bibtex
@article{curvimamba2025,
  title   = {CurviMamba: A Mamba-SSM Architecture with Hessian Gating and
             Topology-Aware Loss for Underwater Detection of Long, Flexible,
             and Highly Deformable Curvilinear Objects},
  author  = {[Author Names]},
  journal = {[Target Journal]},
  year    = {2025}
}
```
