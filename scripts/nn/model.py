"""
ArrowDetector — MobileNetV2 backbone + FPN neck + three prediction heads.

Heads:
  tip_hm    : (B, 1, 128, 128) Gaussian heatmap of arrow tip positions
  nock_hm   : (B, 1, 128, 128) Gaussian heatmap of nock positions
  score_map : (B, 11, 128, 128) score logits at every spatial position
              (used at detected tip locations; 0 = miss, 1–10 = score)

Input: (B, 4, 512, 512) — normalised RGB + radial distance channel.

The first conv of MobileNetV2 is extended from 3→4 input channels; the new
channel's weights are initialised to zero so pretrained behaviour is preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

# ── Feature-map indices in MobileNetV2.features for a 512×512 input ──────────
# Index 3  → stride 4  → 128 × 128,  24 ch
# Index 6  → stride 8  →  64 ×  64,  32 ch
# Index 13 → stride 16 →  32 ×  32,  96 ch
# Index 18 → stride 32 →  16 ×  16, 1280 ch

C2_IDX = 3
C3_IDX = 6
C4_IDX = 13
C5_IDX = 18

C2_CH  = 24
C3_CH  = 32
C4_CH  = 96
C5_CH  = 1280

FPN_CH = 128     # channel width used throughout the FPN
N_SCORES = 11    # classes: 0 = miss, 1-10 = archery score


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class FPN(nn.Module):
    """Lightweight top-down Feature Pyramid Network.

    Takes a dict of feature maps {stride: tensor} and returns P2
    (the highest-resolution level, 128 × 128 for a 512 input).
    """

    def __init__(self):
        super().__init__()
        # Lateral 1×1 convs to project backbone channels → FPN_CH
        self.lat5 = nn.Conv2d(C5_CH, FPN_CH, 1)
        self.lat4 = nn.Conv2d(C4_CH, FPN_CH, 1)
        self.lat3 = nn.Conv2d(C3_CH, FPN_CH, 1)
        self.lat2 = nn.Conv2d(C2_CH, FPN_CH, 1)

        # 3×3 smoothing after each merge
        self.smooth4 = ConvBnRelu(FPN_CH, FPN_CH)
        self.smooth3 = ConvBnRelu(FPN_CH, FPN_CH)
        self.smooth2 = ConvBnRelu(FPN_CH, FPN_CH)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        return p2   # (B, FPN_CH, 128, 128)


class HeatmapHead(nn.Sequential):
    """Small conv head: (B, FPN_CH, H, W) → (B, 1, H, W) in [0, 1]."""

    def __init__(self):
        super().__init__(
            ConvBnRelu(FPN_CH, 64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )


class ScoreHead(nn.Sequential):
    """Dense score classification head: (B, FPN_CH, H, W) → (B, 11, H, W).

    Logits (no softmax) — use cross-entropy loss at GT tip positions.
    """

    def __init__(self):
        super().__init__(
            ConvBnRelu(FPN_CH, 64),
            nn.Conv2d(64, N_SCORES, 1),
        )


class ArrowDetector(nn.Module):
    """Full model: backbone + FPN + tip/nock/score heads."""

    def __init__(self):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────────────
        base = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
        feats = base.features

        # Extend first conv to accept 4 channels (RGB + radial distance).
        # Weights for channels 0-2 come from pretrained; channel 3 init to 0.
        old_conv = feats[0][0]
        new_conv = nn.Conv2d(
            4, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight
            new_conv.weight[:, 3:] = 0.0
        feats[0][0] = new_conv

        # Split backbone into four segments ending at the tap points
        self.stage2 = nn.Sequential(*feats[:C2_IDX + 1])   # → (B,  24, 128, 128)
        self.stage3 = nn.Sequential(*feats[C2_IDX + 1: C3_IDX + 1])  # → (B,  32,  64,  64)
        self.stage4 = nn.Sequential(*feats[C3_IDX + 1: C4_IDX + 1])  # → (B,  96,  32,  32)
        self.stage5 = nn.Sequential(*feats[C4_IDX + 1: C5_IDX + 1])  # → (B,1280,  16,  16)

        # ── neck + heads ──────────────────────────────────────────────────
        self.fpn        = FPN()
        self.tip_head   = HeatmapHead()
        self.nock_head  = HeatmapHead()
        self.score_head = ScoreHead()

    def forward(self, x):
        """
        Args:
            x: (B, 4, 512, 512)
        Returns:
            tip_hm    : (B, 1, 128, 128)
            nock_hm   : (B, 1, 128, 128)
            score_map : (B, 11, 128, 128)
        """
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p2 = self.fpn(c2, c3, c4, c5)

        tip_hm    = self.tip_head(p2)
        nock_hm   = self.nock_head(p2)
        score_map = self.score_head(p2)

        return tip_hm, nock_hm, score_map


# ── post-processing helpers (used at inference) ───────────────────────────────

def heatmap_nms(hm: torch.Tensor, kernel: int = 5) -> torch.Tensor:
    """Zero out non-maxima in a heatmap. Input: (H, W) or (1, H, W)."""
    if hm.dim() == 2:
        hm = hm.unsqueeze(0).unsqueeze(0)
    else:
        hm = hm.unsqueeze(0)
    pad   = kernel // 2
    local_max = F.max_pool2d(hm, kernel_size=kernel, stride=1, padding=pad)
    keep  = (hm == local_max).float()
    return (hm * keep).squeeze()


def decode_heatmap(tip_hm: torch.Tensor, score_map: torch.Tensor,
                   threshold: float = 0.35,
                   scale: float = 1.0, pad_x: int = 0, pad_y: int = 0,
                   input_size: int = 512, heatmap_size: int = 128):
    """Extract arrow tips from heatmap and map back to original image coords.

    Args:
        tip_hm     : (1, H, W) or (H, W) heatmap after NMS
        score_map  : (11, H, W) score logits
        threshold  : minimum heatmap value to accept a peak
        scale, pad_x, pad_y: letterbox parameters from dataset.letterbox()

    Returns:
        list of dicts: {'tip': [x, y], 'score': int, 'confidence': float}
    """
    hm   = heatmap_nms(tip_hm.squeeze())
    ys, xs = torch.where(hm > threshold)
    if len(xs) == 0:
        return []

    ratio = input_size / heatmap_size   # 4.0

    results = []
    for hx, hy in zip(xs.tolist(), ys.tolist()):
        confidence = hm[hy, hx].item()

        # Map heatmap coords → original image coords
        lbx = (hx + 0.5) * ratio   # letterboxed 512-space
        lby = (hy + 0.5) * ratio
        orig_x = (lbx - pad_x) / scale
        orig_y = (lby - pad_y) / scale

        # Score classification at this spatial location
        score_logits = score_map[:, hy, hx]
        score = int(score_logits.argmax().item())

        results.append({
            'tip':        [round(orig_x), round(orig_y)],
            'score':      score,
            'confidence': confidence,
        })

    return results
