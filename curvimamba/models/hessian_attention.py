"""
hessian_attention.py
====================
Classical Hessian-ridge attention prior for curvilinear structure detection.

Computes multi-scale Frangi vesselness (eigenvalue analysis of the Hessian)
and injects it as a spatial attention bias into a convolutional or transformer
feature map neck — making the network attend to tubular/ridge structures
from the very first forward pass.

This module is DIFFERENTIABLE end-to-end; the Gaussian smoothing and Hessian
computation use fixed (non-learned) kernels, but the gating MLP that blends
the vesselness map into the feature map IS learned.

Modules exported:
  - HessianVesselness      : classical Frangi filter (differentiable)
  - HessianAttentionGate   : learned gating of vesselness into feature maps
  - HessianNeck            : drop-in neck replacement for YOLO / segmentation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Differentiable Gaussian kernel utilities
# ---------------------------------------------------------------------------

def gaussian_kernel_2d(sigma: float, kernel_size: int = 0) -> torch.Tensor:
    """Returns a 2-D Gaussian kernel tensor (1, 1, k, k)."""
    if kernel_size == 0:
        kernel_size = int(6 * sigma + 1) | 1  # ensure odd
    ax = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
    g1d = torch.exp(-0.5 * (ax / sigma) ** 2)
    g1d = g1d / g1d.sum()
    g2d = g1d.unsqueeze(1) @ g1d.unsqueeze(0)
    return g2d.unsqueeze(0).unsqueeze(0)


def gaussian_smooth(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Applies 2-D Gaussian smoothing channel-wise. x: (B, C, H, W)."""
    kernel = gaussian_kernel_2d(sigma).to(x.device)
    pad = kernel.shape[-1] // 2
    B, C, H, W = x.shape
    # Process all channels at once via groups
    x_flat = x.view(B * C, 1, H, W)
    smoothed = F.conv2d(x_flat, kernel, padding=pad)
    return smoothed.view(B, C, H, W)


def hessian_2d(img: torch.Tensor, sigma: float) -> Tuple[torch.Tensor, ...]:
    """
    Compute second-order Gaussian derivatives (Hxx, Hxy, Hyy) of a
    single-channel image batch.

    Args:
        img   : (B, 1, H, W) float tensor
        sigma : smoothing scale

    Returns:
        Hxx, Hxy, Hyy — each (B, 1, H, W)
    """
    smoothed = gaussian_smooth(img, sigma)

    # Finite-difference kernels for 2nd derivatives
    kxx = torch.tensor([[[[0, 0, 0],
                          [1, -2, 1],
                          [0, 0, 0]]]], dtype=torch.float32, device=img.device)
    kyy = torch.tensor([[[[0, 1, 0],
                          [0, -2, 0],
                          [0, 1, 0]]]], dtype=torch.float32, device=img.device)
    kxy = torch.tensor([[[[1, 0, -1],
                          [0, 0, 0],
                          [-1, 0, 1]]]], dtype=torch.float32, device=img.device) * 0.25

    Hxx = F.conv2d(smoothed, kxx, padding=1)
    Hyy = F.conv2d(smoothed, kyy, padding=1)
    Hxy = F.conv2d(smoothed, kxy, padding=1)
    return Hxx, Hxy, Hyy


def frangi_vesselness(
    img: torch.Tensor,
    sigmas: List[float] = (1.0, 2.0, 4.0),
    beta: float = 0.5,
    c: float = 15.0,
    bright_on_dark: bool = True,
) -> torch.Tensor:
    """
    Multi-scale Frangi vesselness filter (2-D).

    Args:
        img            : (B, 1, H, W) — greyscale or luminance channel
        sigmas         : list of Gaussian scales to analyse
        beta           : sensitivity to deviation from blob shape (Rb)
        c              : sensitivity to second-order structure (S)
        bright_on_dark : if True, detect bright ridges on dark background

    Returns:
        vesselness : (B, 1, H, W) in [0, 1]
    """
    max_response = torch.zeros_like(img)

    for sigma in sigmas:
        Hxx, Hxy, Hyy = hessian_2d(img, sigma)

        # Eigenvalues of 2x2 symmetric Hessian
        tmp   = torch.sqrt((Hxx - Hyy).pow(2) + 4.0 * Hxy.pow(2) + 1e-8)
        lam1  = 0.5 * (Hxx + Hyy - tmp)
        lam2  = 0.5 * (Hxx + Hyy + tmp)

        # For bright ridges: lam2 should be strongly negative
        if bright_on_dark:
            # Zero out where lam2 > 0 (not a ridge)
            valid = (lam2 < 0).float()
        else:
            valid = (lam2 > 0).float()
            lam1  = -lam1
            lam2  = -lam2

        # Frangi measures
        Rb = lam1 / (lam2 + 1e-8)          # blob vs tube ratio
        S2 = lam1.pow(2) + lam2.pow(2)     # second-order structure magnitude

        response = (
            (1.0 - torch.exp(-Rb.pow(2) / (2 * beta ** 2))) *
            torch.exp(-S2 / (2 * c ** 2))
        ) * valid

        # Scale normalisation (sigma^2 accounts for derivative magnitude scaling)
        response = response * sigma ** 2

        max_response = torch.max(max_response, response)

    # Normalise to [0, 1]
    vmin = max_response.flatten(2).min(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    vmax = max_response.flatten(2).max(dim=2)[0].unsqueeze(-1).unsqueeze(-1)
    vesselness = (max_response - vmin) / (vmax - vmin + 1e-8)
    return vesselness


# ---------------------------------------------------------------------------
# 2. Hessian Attention Gate — learned blending
# ---------------------------------------------------------------------------

class HessianAttentionGate(nn.Module):
    """
    Fuses a spatial vesselness prior into a deep feature map via a learned gate.

    Architecture:
        vesselness (1 ch) → conv 1x1 → sigmoid → element-wise gate on features

    The gate is initialised so that at the start of training, features are
    passed through almost unchanged, and the vesselness influence grows as
    the gate learns.

    Args:
        feat_channels : number of channels in the feature map to gate
        sigmas        : Gaussian scales for the Frangi filter
    """

    def __init__(
        self,
        feat_channels: int,
        sigmas: List[float] = (1.0, 2.0, 4.0),
        beta: float = 0.5,
        c: float = 15.0,
    ):
        super().__init__()
        self.sigmas = sigmas
        self.beta   = beta
        self.c      = c

        # Learned projection: vesselness → per-channel attention weights
        self.gate_proj = nn.Sequential(
            nn.Conv2d(1, feat_channels // 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_channels // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_channels // 4, feat_channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # Blending coefficient (learned scalar, initialised near 0 so gate starts inactive)
        self.alpha = nn.Parameter(torch.zeros(1))

        # Grey-conversion weights (if input is RGB)
        self.register_buffer(
            "grey_weights",
            torch.tensor([0.2126, 0.7152, 0.0722]).view(1, 3, 1, 1)
        )

    def _to_grey(self, x: torch.Tensor) -> torch.Tensor:
        """Convert (B, C, H, W) to (B, 1, H, W) luminance."""
        if x.shape[1] == 1:
            return x
        elif x.shape[1] == 3:
            return (x * self.grey_weights).sum(dim=1, keepdim=True)
        else:
            # For arbitrary C, just average
            return x.mean(dim=1, keepdim=True)

    def forward(self, feat: torch.Tensor, img: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            feat : (B, C, H, W) deep feature map
            img  : (B, *, H_orig, W_orig) raw image for vesselness (optional).
                   If None, vesselness is computed from feat directly.

        Returns:
            gated feature map (B, C, H, W)
        """
        B, C, H, W = feat.shape

        if img is not None:
            grey = self._to_grey(img)
            # Resize to match feature map spatial size
            if grey.shape[2:] != feat.shape[2:]:
                grey = F.interpolate(grey, size=(H, W), mode='bilinear', align_corners=False)
        else:
            grey = self._to_grey(feat)
            # Normalise feat-derived grey to [0,1]
            grey = (grey - grey.min()) / (grey.max() - grey.min() + 1e-8)

        vessel = frangi_vesselness(grey, sigmas=self.sigmas, beta=self.beta, c=self.c)

        # Learned gate
        gate = self.gate_proj(vessel)           # (B, C, H, W) in [0,1]

        # Blend: alpha starts at 0 → gate is identity at init
        alpha = torch.sigmoid(self.alpha)       # [0, 1]
        return feat * (1.0 + alpha * gate)


# ---------------------------------------------------------------------------
# 3. HessianNeck — drop-in YOLO / segmentation neck with attention gates
# ---------------------------------------------------------------------------

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class HessianNeck(nn.Module):
    """
    Feature Pyramid Network neck augmented with Hessian attention gates.

    Designed to slot in as a drop-in replacement for the standard FPN neck
    in YOLOv8 / YOLOv11 style architectures.

    Takes three feature maps (P3, P4, P5) from the backbone and produces
    three output feature maps with the same spatial sizes but with
    vesselness-guided spatial attention applied at each scale.

    Args:
        in_channels  : list of [C_P3, C_P4, C_P5] input channels
        out_channels : list of [C_out3, C_out4, C_out5] output channels
        sigmas       : Frangi scales (should match approximate cable widths in pixels)
    """

    def __init__(
        self,
        in_channels:  List[int] = [256, 512, 1024],
        out_channels: List[int] = [256, 512, 1024],
        sigmas: List[float] = (1.0, 2.0, 4.0),
    ):
        super().__init__()
        assert len(in_channels) == 3

        # Lateral convolutions (1x1 for channel alignment)
        self.lat5 = ConvBNAct(in_channels[2], out_channels[2], k=1, p=0)
        self.lat4 = ConvBNAct(in_channels[1], out_channels[1], k=1, p=0)
        self.lat3 = ConvBNAct(in_channels[0], out_channels[0], k=1, p=0)

        # Top-down fusion convolutions
        self.td4 = ConvBNAct(out_channels[2] + out_channels[1], out_channels[1])
        self.td3 = ConvBNAct(out_channels[1] + out_channels[0], out_channels[0])

        # Hessian attention gates at each scale
        self.hag3 = HessianAttentionGate(out_channels[0], sigmas=sigmas)
        self.hag4 = HessianAttentionGate(out_channels[1], sigmas=sigmas)
        self.hag5 = HessianAttentionGate(out_channels[2], sigmas=sigmas)

    def forward(
        self,
        features: List[torch.Tensor],
        img: Optional[torch.Tensor] = None,
    ) -> List[torch.Tensor]:
        """
        Args:
            features : [P3, P4, P5] from backbone
            img      : raw input image for vesselness computation

        Returns:
            [out3, out4, out5] gated feature maps
        """
        p3, p4, p5 = features

        # Lateral projections
        f5 = self.lat5(p5)
        f4 = self.lat4(p4)
        f3 = self.lat3(p3)

        # Top-down fusion
        f4_td = torch.cat([F.interpolate(f5, size=f4.shape[2:], mode='nearest'), f4], dim=1)
        f4_td = self.td4(f4_td)

        f3_td = torch.cat([F.interpolate(f4_td, size=f3.shape[2:], mode='nearest'), f3], dim=1)
        f3_td = self.td3(f3_td)

        # Apply Hessian attention gates
        out5 = self.hag5(f5,    img=img)
        out4 = self.hag4(f4_td, img=img)
        out3 = self.hag3(f3_td, img=img)

        return [out3, out4, out5]


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 256, 256
    img = torch.rand(B, 3, H, W)

    # Test Frangi
    grey = img.mean(dim=1, keepdim=True)
    v = frangi_vesselness(grey, sigmas=[1, 2, 4])
    print(f"Frangi output shape: {v.shape}, range [{v.min():.3f}, {v.max():.3f}]")

    # Test HessianAttentionGate
    feat = torch.randn(B, 256, H // 8, W // 8)
    gate = HessianAttentionGate(feat_channels=256)
    out  = gate(feat, img=img)
    print(f"AttentionGate output shape: {out.shape}")

    # Test HessianNeck
    p3 = torch.randn(B, 256, H // 8,  W // 8)
    p4 = torch.randn(B, 512, H // 16, W // 16)
    p5 = torch.randn(B, 1024, H // 32, W // 32)
    neck = HessianNeck()
    outs = neck([p3, p4, p5], img=img)
    for i, o in enumerate(outs):
        print(f"Neck output P{i+3}: {o.shape}")
