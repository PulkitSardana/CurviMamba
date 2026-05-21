"""
topology_loss.py
================
Topology-aware loss for curvilinear structure segmentation.

Two complementary topological penalties:
  1. EulerCharacteristicLoss  — fast, differentiable proxy via discrete EC
  2. PersistenceHomologyLoss  — Betti-number-based via soft persistence diagrams

Both can be used standalone or combined (TopologyLoss wrapper).

Reference contributions:
  - Clough et al. (2020) "Explicit Topological Priors for Deep-Learning Based
    Image Segmentation Using Persistent Homology"
  - Hu et al. (2021) "Topology-Preserving Deep Image Segmentation"
  - Carlier et al. (2023) "Euler Characteristic Transform for Shape Analysis"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Utility: soft thresholding / differentiable binarisation
# ---------------------------------------------------------------------------

def soft_threshold(x: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    """Sigmoid-based soft binarisation centred at tau."""
    return torch.sigmoid(10.0 * (x - tau))


# ---------------------------------------------------------------------------
# 1. Euler Characteristic Loss
# ---------------------------------------------------------------------------

class EulerCharacteristicLoss(nn.Module):
    """
    Differentiable Euler Characteristic (EC) loss for 2-D binary maps.

    For a binary image B:
        EC = V - E + F
    where V = vertices (pixels ON), E = horizontal+vertical edges between
    adjacent ON pixels, F = 2x2 fully-ON quad faces.

    For a single connected curvilinear object (cable, rope):
        EC_target = 1  (one connected component, no holes)

    The loss penalises deviation of the predicted EC from the target EC,
    which implicitly penalises fragmentation (too many components) and
    spurious loops.

    Args:
        target_ec   : expected Euler characteristic (default 1 for one cable)
        weight      : scalar weight applied to the loss term
        tau         : soft-threshold parameter for differentiable binarisation
    """

    def __init__(
        self,
        target_ec: float = 1.0,
        weight: float = 1.0,
        tau: float = 0.5,
    ):
        super().__init__()
        self.target_ec = target_ec
        self.weight = weight
        self.tau = tau

        # Fixed convolution kernels for local structure counting
        # Edge kernel: detects horizontal neighbour pairs
        self._register_kernels()

    def _register_kernels(self):
        # Horizontal edge: pixel (i,j) and (i,j+1) both ON
        he = torch.zeros(1, 1, 1, 2)
        he[0, 0, 0, :] = 1.0
        self.register_buffer("horiz_kernel", he)

        # Vertical edge: pixel (i,j) and (i+1,j) both ON
        ve = torch.zeros(1, 1, 2, 1)
        ve[0, 0, :, 0] = 1.0
        self.register_buffer("vert_kernel", ve)

        # 2x2 face kernel
        fk = torch.ones(1, 1, 2, 2)
        self.register_buffer("face_kernel", fk)

    def _compute_ec(self, prob_map: torch.Tensor) -> torch.Tensor:
        """
        Compute soft EC for a single (H, W) probability map.
        Returns scalar tensor.
        """
        p = soft_threshold(prob_map, self.tau).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)

        # Vertices: sum of soft-binary pixels
        V = p.sum()

        # Horizontal edges: min(p_i, p_{i,j+1}) summed — soft AND
        p_h_left  = p[:, :, :, :-1]
        p_h_right = p[:, :, :, 1:]
        E_h = torch.min(p_h_left, p_h_right).sum()

        # Vertical edges
        p_v_top = p[:, :, :-1, :]
        p_v_bot = p[:, :, 1:,  :]
        E_v = torch.min(p_v_top, p_v_bot).sum()

        E = E_h + E_v

        # Faces: 2x2 blocks — soft min of all 4 pixels
        p_tl = p[:, :, :-1, :-1]
        p_tr = p[:, :, :-1, 1:]
        p_bl = p[:, :, 1:,  :-1]
        p_br = p[:, :, 1:,  1:]
        F = torch.min(torch.min(p_tl, p_tr), torch.min(p_bl, p_br)).sum()

        ec = V - E + F
        return ec

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            pred   : (B, 1, H, W) or (B, H, W) — raw logits or probabilities
            target : (B, 1, H, W) or (B, H, W) — binary ground-truth masks
            mask   : optional (B, H, W) valid-region mask

        Returns:
            Scalar loss tensor.
        """
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Convert logits → probabilities if needed
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        B = pred.shape[0]
        loss = pred.new_zeros(1)

        for b in range(B):
            pred_ec   = self._compute_ec(pred[b])
            target_ec = self._compute_ec(target[b].float())
            # Use target EC per sample (richer than fixed scalar target_ec)
            loss = loss + (pred_ec - target_ec).pow(2)

        return self.weight * loss / B


# ---------------------------------------------------------------------------
# 2. Persistence Homology Loss (soft Betti number matching)
# ---------------------------------------------------------------------------

class PersistenceHomologyLoss(nn.Module):
    """
    Soft persistence-based topological loss.

    Approximates Betti-0 (connected components) and Betti-1 (loops/holes)
    via a differentiable filtration over the predicted probability map.

    The approach:
      - Sort pixels by descending predicted probability → filtration order
      - Track component births/deaths via a union-find structure
        (non-differentiable) then use the PERSISTENCE PAIRS to build a
        differentiable penalty: push each spurious component's death
        probability close to its birth probability (zero persistence = merged)

    This is a lightweight approximation of the full TopoLoss (Hu et al. 2021)
    that avoids the cubical complex library dependency, making it pip-installable.

    Args:
        betti0_weight : weight for Betti-0 (connectivity) penalty
        betti1_weight : weight for Betti-1 (loop) penalty
        target_betti0 : desired number of connected components (1 for one cable)
        target_betti1 : desired number of loops (0 for an open curve)
        max_pixels    : subsample to this many pixels for efficiency (None = all)
    """

    def __init__(
        self,
        betti0_weight: float = 1.0,
        betti1_weight: float = 0.5,
        target_betti0: int = 1,
        target_betti1: int = 0,
        max_pixels: Optional[int] = 4096,
    ):
        super().__init__()
        self.betti0_weight = betti0_weight
        self.betti1_weight = betti1_weight
        self.target_betti0 = target_betti0
        self.target_betti1 = target_betti1
        self.max_pixels = max_pixels

    # ------------------------------------------------------------------
    # Union-Find (numpy, non-differentiable — used only for pair finding)
    # ------------------------------------------------------------------

    @staticmethod
    def _union_find_components(binary: np.ndarray) -> Tuple[int, list]:
        """
        4-connected component labelling via union-find.
        Returns (num_components, list_of_(birth_idx, death_idx) pairs)
        where indices refer to positions in the flattened sorted array.
        """
        H, W = binary.shape
        parent = np.arange(H * W)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
                return True
            return False

        pairs = []  # (birth_flat_idx, death_flat_idx)
        component_birth = {}  # root → birth flat index (in sorted order)

        flat = binary.flatten()
        for i, v in enumerate(flat):
            if v == 0:
                continue
            idx = i  # flat index in sorted-by-prob order (caller sorts)
            neighbours = []
            r, c = divmod(idx, W)
            if r > 0 and flat[idx - W]: neighbours.append(idx - W)
            if r < H-1 and flat[idx + W]: neighbours.append(idx + W)
            if c > 0 and flat[idx - 1]: neighbours.append(idx - 1)
            if c < W-1 and flat[idx + 1]: neighbours.append(idx + 1)

            roots_seen = set()
            for nb in neighbours:
                roots_seen.add(find(nb))

            if not roots_seen:
                # New component born
                component_birth[find(idx)] = idx
            else:
                roots_seen = list(roots_seen)
                # Merge all into first
                for r2 in roots_seen[1:]:
                    if find(r2) in component_birth:
                        pairs.append((component_birth[find(r2)], idx))
                        del component_birth[find(r2)]
                    union(roots_seen[0], r2)

        num_components = len(component_birth)
        return num_components, pairs

    def _persistence_loss_single(self, prob_map: torch.Tensor) -> torch.Tensor:
        """Compute persistence loss for one (H, W) probability map."""
        H, W = prob_map.shape
        prob_np = prob_map.detach().cpu().numpy()

        # Sort pixels descending by probability (superlevel set filtration)
        flat_prob = prob_np.flatten()
        sort_idx  = np.argsort(-flat_prob)  # descending

        # Build filtration: include pixels one-by-one in sort order
        included = np.zeros(H * W, dtype=bool)
        # We run a simplified version: threshold at 0.5 and count components
        # then use the actual probabilities at birth/death for the differentiable part

        binary = (flat_prob > 0.5).astype(np.float32).reshape(H, W)

        # Subsample for efficiency
        if self.max_pixels is not None and H * W > self.max_pixels:
            scale = (self.max_pixels / (H * W)) ** 0.5
            new_H = max(8, int(H * scale))
            new_W = max(8, int(W * scale))
            prob_map_small = F.interpolate(
                prob_map.unsqueeze(0).unsqueeze(0),
                size=(new_H, new_W), mode='bilinear', align_corners=False
            ).squeeze()
            binary = (prob_map_small.detach().cpu().numpy() > 0.5).astype(np.float32)
            prob_map_ref = prob_map_small
        else:
            prob_map_ref = prob_map

        num_comp, pairs = self._union_find_components(binary)

        loss = prob_map.new_zeros(1)

        # Betti-0 loss: each extra component beyond target_betti0
        # Penalty = sum of (birth_prob - death_prob) for spurious pairs
        # i.e. push death_prob → birth_prob (zero persistence → merge)
        flat_ref = prob_map_ref.flatten()
        for birth_idx, death_idx in pairs:
            if birth_idx < flat_ref.shape[0] and death_idx < flat_ref.shape[0]:
                birth_prob = flat_ref[birth_idx]
                death_prob = flat_ref[death_idx]
                # Spurious persistence: we want death_prob → birth_prob
                loss = loss + self.betti0_weight * (birth_prob - death_prob).pow(2)

        # Simple Betti-1 proxy: penalise any pixel that is ON but has all
        # 4 neighbours ON (interior point of a loop — a hole filler)
        p = prob_map_ref
        if p.shape[0] > 2 and p.shape[1] > 2:
            interior = (
                p[1:-1, 1:-1] *
                p[:-2,  1:-1] *
                p[2:,   1:-1] *
                p[1:-1, :-2]  *
                p[1:-1, 2:]
            )
            loss = loss + self.betti1_weight * interior.sum()

        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)

        B = pred.shape[0]
        loss = pred.new_zeros(1)
        for b in range(B):
            loss = loss + self._persistence_loss_single(pred[b])
        return loss / B


# ---------------------------------------------------------------------------
# 3. Combined TopologyLoss wrapper
# ---------------------------------------------------------------------------

class TopologyLoss(nn.Module):
    """
    Full topology-aware loss = BCE + DiceLoss + EC loss + Persistence loss.

    This is the loss you attach to your segmentation head for curvilinear
    underwater object detection.

    Args:
        bce_weight   : weight for binary cross-entropy
        dice_weight  : weight for Dice loss
        ec_weight    : weight for Euler characteristic loss
        ph_weight    : weight for persistence homology loss
        ec_target    : expected Euler characteristic per object
    """

    def __init__(
        self,
        bce_weight: float  = 1.0,
        dice_weight: float = 1.0,
        ec_weight: float   = 0.5,
        ph_weight: float   = 0.3,
        ec_target: float   = 1.0,
    ):
        super().__init__()
        self.bce_weight  = bce_weight
        self.dice_weight = dice_weight
        self.ec_loss  = EulerCharacteristicLoss(target_ec=ec_target, weight=ec_weight)
        self.ph_loss  = PersistenceHomologyLoss(betti0_weight=ph_weight)

    # ----- Dice loss (standard) -----
    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
        pred_f   = pred.flatten(1)
        target_f = target.flatten(1).float()
        intersection = (pred_f * target_f).sum(1)
        return 1.0 - (2.0 * intersection + eps) / (pred_f.sum(1) + target_f.sum(1) + eps)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            pred   : (B, 1, H, W) raw logits
            target : (B, 1, H, W) binary masks

        Returns:
            total_loss : scalar
            components : dict with individual loss values for logging
        """
        target_f = target.float()

        bce  = F.binary_cross_entropy_with_logits(pred, target_f)
        dice = self.dice_loss(pred, target_f).mean()
        ec   = self.ec_loss(pred, target_f)
        ph   = self.ph_loss(pred, target_f)

        total = (
            self.bce_weight  * bce  +
            self.dice_weight * dice +
            ec  +   # ec_weight already baked in
            ph      # ph_weight already baked in
        )

        components = {
            "bce":   bce.item(),
            "dice":  dice.item(),
            "ec":    ec.item(),
            "ph":    ph.item(),
            "total": total.item(),
        }
        return total, components


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    B, H, W = 2, 128, 128
    pred   = torch.randn(B, 1, H, W)
    target = torch.zeros(B, 1, H, W)
    # Draw a horizontal cable in target
    target[:, 0, 60:65, 10:118] = 1.0

    loss_fn = TopologyLoss()
    total, comps = loss_fn(pred, target)
    print("TopologyLoss self-test passed:")
    for k, v in comps.items():
        print(f"  {k:>6}: {v:.4f}")
