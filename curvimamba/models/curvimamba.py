"""
curvimamba.py
=============
CurviMamba: Mamba-backbone segmentation model for underwater curvilinear
object detection, with Hessian attention neck and topology-aware loss.

Architecture overview:
    Input image (B, 3, H, W)
        │
        ▼
    PatchEmbed + Mamba encoder blocks (Vision Mamba-style, simplified)
        │  ← 4 stages, outputting multi-scale feature maps
        ▼
    HessianNeck (FPN with vesselness attention gates)
        │
        ▼
    CurvilinearHead (lightweight decoder + segmentation mask)
        │
        ▼
    TopologyLoss (BCE + Dice + EC + Persistence homology)

NOTE: The Mamba blocks here use a simplified SSM approximation that runs
without the mamba-ssm CUDA extension, making the code pip-installable
and runnable on any GPU/CPU. For production, swap MambaBlock with the
official mamba_ssm.Mamba class.

For YOLO-based deployment, CurviMamba can be used as a backbone feature
extractor feeding into any detection/segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple

from .hessian_attention import HessianNeck
from losses.topology_loss import TopologyLoss


# ---------------------------------------------------------------------------
# Simplified SSM block (no CUDA extension required)
# ---------------------------------------------------------------------------

class SimplifiedSSM(nn.Module):
    """
    Lightweight structured state-space model approximation.

    Uses a 1-D causal depthwise convolution + selective gating to approximate
    the sequence-modelling behaviour of Mamba without requiring the
    mamba-ssm custom CUDA kernels.

    For the full Mamba implementation, replace this with:
        from mamba_ssm import Mamba
        Mamba(d_model=dim, d_state=16, d_conv=4, expand=2)

    Args:
        dim      : feature dimension
        d_state  : SSM state dimension
        d_conv   : depthwise conv width
        expand   : inner expansion factor
    """

    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.inner_dim = int(dim * expand)

        self.norm   = nn.LayerNorm(dim)
        self.in_proj = nn.Linear(dim, self.inner_dim * 2, bias=False)

        # Depthwise conv approximating the SSM scan
        self.dw_conv = nn.Conv1d(
            self.inner_dim, self.inner_dim,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.inner_dim, bias=True,
        )
        self.act    = nn.SiLU()
        self.out_proj = nn.Linear(self.inner_dim, dim, bias=False)

        # Selective gate: maps input to per-position gate
        self.gate_proj = nn.Sequential(
            nn.Linear(dim, d_state, bias=False),
            nn.SiLU(),
            nn.Linear(d_state, self.inner_dim, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, dim)"""
        residual = x
        x = self.norm(x)

        gate = self.gate_proj(x)                    # (B, L, inner_dim)

        xz   = self.in_proj(x)                      # (B, L, 2*inner_dim)
        x_in, z = xz.chunk(2, dim=-1)               # each (B, L, inner_dim)

        # Causal depthwise conv along sequence dimension
        x_in_t = x_in.transpose(1, 2)               # (B, inner_dim, L)
        x_conv  = self.dw_conv(x_in_t)[..., :x_in_t.shape[-1]]  # causal trim
        x_conv  = self.act(x_conv).transpose(1, 2)  # (B, L, inner_dim)

        y = x_conv * self.act(z) * gate
        y = self.out_proj(y)
        return y + residual


class MambaBlock(nn.Module):
    """SSM block + MLP, as used in Vision Mamba."""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, **ssm_kwargs):
        super().__init__()
        self.ssm = SimplifiedSSM(dim, **ssm_kwargs)
        mlp_dim  = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_dim, bias=False),
            nn.GELU(),
            nn.Linear(mlp_dim, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ssm(x)
        x = x + self.mlp(x)
        return x


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Image → patch tokens."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 96, patch_size: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """Returns (tokens: B,L,C), H_patches, W_patches"""
        B, C, H, W = x.shape
        x = self.proj(x)                   # (B, embed_dim, H/p, W/p)
        Hp, Wp = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)   # (B, L, embed_dim)
        x = self.norm(x)
        return x, Hp, Wp


# ---------------------------------------------------------------------------
# Vision Mamba encoder (4-stage)
# ---------------------------------------------------------------------------

class VisionMambaEncoder(nn.Module):
    """
    Hierarchical Vision Mamba encoder producing multi-scale features.

    Stage output spatial sizes (for H=W=512):
      P2: H/4,  W/4   — stride 4  (high-res, thin cable details)
      P3: H/8,  W/8   — stride 8
      P4: H/16, W/16  — stride 16
      P5: H/32, W/32  — stride 32 (low-res, global context)

    Args:
        in_channels  : input image channels (3 for RGB)
        embed_dims   : list of 4 channel sizes per stage
        depths       : number of Mamba blocks per stage
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 256, 512],
        depths: List[int] = [2, 2, 6, 2],
    ):
        super().__init__()
        assert len(embed_dims) == 4 and len(depths) == 4

        # Patch embed for first stage
        self.patch_embed = PatchEmbed(in_channels, embed_dims[0], patch_size=4)

        # Downsampling between stages
        self.downsample = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(embed_dims[i]),
                nn.Linear(embed_dims[i], embed_dims[i+1], bias=False),
            )
            for i in range(3)
        ])

        # Mamba blocks per stage
        self.stages = nn.ModuleList([
            nn.Sequential(*[MambaBlock(embed_dims[i]) for _ in range(depths[i])])
            for i in range(4)
        ])

        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in embed_dims])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Returns:
            List of 4 feature maps [P2, P3, P4, P5] in (B, C, H, W) format
        """
        B = x.shape[0]
        outs = []

        tokens, H, W = self.patch_embed(x)          # stage 0

        for i, stage in enumerate(self.stages):
            tokens = stage(tokens)
            tokens = self.norms[i](tokens)
            feat = tokens.transpose(1, 2).reshape(B, -1, H, W)
            outs.append(feat)

            if i < 3:
                # Downsample tokens for next stage
                tokens = self.downsample[i](tokens)
                H, W = H // 2, W // 2

        # outs: [P2(stride4), P3(stride8), P4(stride16), P5(stride32)]
        return outs


# ---------------------------------------------------------------------------
# Curvilinear segmentation head
# ---------------------------------------------------------------------------

class CurvilinearHead(nn.Module):
    """
    Lightweight decoder that upsamples multi-scale features to
    produce a full-resolution segmentation mask.

    Input: 3 FPN feature maps [P3, P4, P5]
    Output: (B, 1, H, W) binary mask logits (before sigmoid)

    Also outputs a (B, num_classes, H/4, W/4) detection map
    for bounding-box style downstream use.
    """

    def __init__(
        self,
        in_channels: List[int] = [256, 512, 1024],
        mid_channels: int = 128,
        num_classes: int = 1,
    ):
        super().__init__()

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(sum(in_channels), mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
        )

        self.decode_blocks = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, mid_channels // 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels // 2),
            nn.SiLU(inplace=True),
        )

        self.seg_head = nn.Conv2d(mid_channels // 2, num_classes, 1)
        self.cls_head = nn.Conv2d(mid_channels // 2, num_classes, 1)

        # Centerline head — predicts thin skeleton probability
        self.centerline_head = nn.Conv2d(mid_channels // 2, 1, 1)

    def forward(
        self, features: List[torch.Tensor], target_size: Tuple[int, int]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features    : [P3, P4, P5] from neck
            target_size : (H, W) of original image

        Returns:
            mask_logits       : (B, 1, H, W)
            cls_logits        : (B, num_cls, H/8, W/8)
            centerline_logits : (B, 1, H, W)
        """
        p3, p4, p5 = features

        # Upsample all to P3 resolution
        p4_up = F.interpolate(p4, size=p3.shape[2:], mode='bilinear', align_corners=False)
        p5_up = F.interpolate(p5, size=p3.shape[2:], mode='bilinear', align_corners=False)

        fused = self.fuse_conv(torch.cat([p3, p4_up, p5_up], dim=1))
        feats = self.decode_blocks(fused)

        # Upsample to original size
        feats_full = F.interpolate(feats, size=target_size, mode='bilinear', align_corners=False)

        mask_logits       = self.seg_head(feats_full)
        centerline_logits = self.centerline_head(feats_full)
        cls_logits        = self.cls_head(feats)

        return mask_logits, cls_logits, centerline_logits


# ---------------------------------------------------------------------------
# Full CurviMamba model
# ---------------------------------------------------------------------------

class CurviMamba(nn.Module):
    """
    CurviMamba: End-to-end underwater curvilinear object segmentation model.

    Novel contributions:
      1. Mamba (SSM) backbone for linear-complexity global context on
         high-resolution images (critical for thin, long objects)
      2. Hessian vesselness attention gates in the FPN neck
         (classical physics prior injected into the learned neck)
      3. Topology-aware loss head (EC loss + Persistence homology loss)
         penalising fragmented or looped predictions

    Args:
        in_channels   : input image channels
        embed_dims    : Mamba encoder channel sizes
        depths        : Mamba blocks per stage
        neck_channels : FPN output channels
        num_classes   : number of object classes
        loss_weights  : dict with keys bce, dice, ec, ph
        sigmas        : Frangi filter scales
    """

    def __init__(
        self,
        in_channels: int = 3,
        embed_dims: List[int] = [64, 128, 256, 512],
        depths: List[int] = [2, 2, 6, 2],
        neck_channels: List[int] = [256, 512, 1024],
        num_classes: int = 1,
        loss_weights: Optional[dict] = None,
        sigmas: List[float] = [1.0, 2.0, 4.0],
    ):
        super().__init__()

        if loss_weights is None:
            loss_weights = dict(bce=1.0, dice=1.0, ec=0.5, ph=0.3)

        # --- Backbone ---
        self.encoder = VisionMambaEncoder(in_channels, embed_dims, depths)

        # Align backbone output channels to neck input channels
        self.chan_align = nn.ModuleList([
            nn.Conv2d(embed_dims[i+1], neck_channels[i], 1)
            for i in range(3)
        ])

        # --- Neck ---
        self.neck = HessianNeck(
            in_channels=neck_channels,
            out_channels=neck_channels,
            sigmas=sigmas,
        )

        # --- Head ---
        self.head = CurvilinearHead(
            in_channels=neck_channels,
            mid_channels=128,
            num_classes=num_classes,
        )

        # --- Loss ---
        self.loss_fn = TopologyLoss(
            bce_weight=loss_weights.get("bce", 1.0),
            dice_weight=loss_weights.get("dice", 1.0),
            ec_weight=loss_weights.get("ec", 0.5),
            ph_weight=loss_weights.get("ph", 0.3),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)

    def forward(
        self,
        img: torch.Tensor,
        target: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Args:
            img    : (B, 3, H, W) input image
            target : (B, 1, H, W) binary mask, required during training

        Returns dict with keys:
            'mask'       : (B, 1, H, W) sigmoid probabilities
            'centerline' : (B, 1, H, W) sigmoid centerline probabilities
            'loss'       : scalar (only if target provided)
            'loss_components' : dict (only if target provided)
        """
        B, C, H, W = img.shape

        # Backbone: get [P2, P3, P4, P5]
        enc_feats = self.encoder(img)

        # Use P3, P4, P5 (skip P2 for memory) and align channels
        neck_in = [self.chan_align[i](enc_feats[i+1]) for i in range(3)]

        # Neck with Hessian attention
        neck_out = self.neck(neck_in, img=img)

        # Segmentation head
        mask_logits, cls_logits, cl_logits = self.head(neck_out, (H, W))

        out = {
            "mask":       torch.sigmoid(mask_logits),
            "centerline": torch.sigmoid(cl_logits),
            "mask_logits": mask_logits,
        }

        if target is not None:
            loss, components = self.loss_fn(mask_logits, target)
            out["loss"] = loss
            out["loss_components"] = components

        return out

    @torch.no_grad()
    def predict(self, img: torch.Tensor, threshold: float = 0.5) -> dict:
        """Inference-time forward with thresholded binary mask."""
        self.eval()
        out  = self.forward(img)
        pred = (out["mask"] > threshold).float()
        cl   = (out["centerline"] > threshold).float()
        return {"mask": pred, "centerline": cl, "mask_prob": out["mask"]}


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, W = 1, 256, 256   # use small size for quick test

    model = CurviMamba(
        embed_dims=[32, 64, 128, 256],
        depths=[1, 1, 2, 1],
        neck_channels=[128, 256, 512],
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"CurviMamba parameters: {total_params:.1f}M")

    img    = torch.rand(B, 3, H, W)
    target = torch.zeros(B, 1, H, W)
    target[:, 0, H//2-3:H//2+3, 10:W-10] = 1.0  # horizontal cable

    out = model(img, target)
    print(f"Mask output shape:       {out['mask'].shape}")
    print(f"Centerline output shape: {out['centerline'].shape}")
    print(f"Total loss:              {out['loss'].item():.4f}")
    print("Loss components:")
    for k, v in out['loss_components'].items():
        print(f"  {k:>6}: {v:.4f}")
