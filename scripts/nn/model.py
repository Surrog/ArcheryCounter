"""
ArrowDetector — MobileNetV2 backbone + FPN neck + two prediction heads.

Heads:
  tip_hm    : (B, 1, 128, 128) Gaussian heatmap of arrow tip positions
  score_map : (B, 11, 128, 128) score logits at every spatial position
              (used at detected tip locations; 0 = miss, 1–10 = score)

Input: (B, 4, 512, 512) — normalised RGB + radial distance channel.

The first conv of MobileNetV2 is extended from 3→4 input channels; the new
channel's weights are initialised to zero so pretrained behaviour is preserved.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import (
    mobilenet_v2, MobileNet_V2_Weights,
    mobilenet_v3_large, MobileNet_V3_Large_Weights,
)

# ── Backbone tap configs ───────────────────────────────────────────────────────
# Indices into model.features; channels at those indices; valid for 640×640 input.

BACKBONE_CONFIGS = {
    'mobilenet_v2': {
        'tap_idxs': (3, 6, 13, 18),   # strides 4, 8, 16, 32
        'tap_chs':  (24, 32, 96, 1280),
        'weights':  MobileNet_V2_Weights.IMAGENET1K_V1,
        'builder':  lambda w: mobilenet_v2(weights=w),
    },
    'mobilenet_v3_large': {
        'tap_idxs': (3, 6, 12, 16),   # strides 4, 8, 16, 32
        'tap_chs':  (24, 40, 112, 960),
        'weights':  MobileNet_V3_Large_Weights.IMAGENET1K_V1,
        'builder':  lambda w: mobilenet_v3_large(weights=w),
    },
}

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

    Takes four feature maps (c2..c5) and returns P2 (highest resolution).
    """

    def __init__(self, c2_ch, c3_ch, c4_ch, c5_ch):
        super().__init__()
        self.lat5 = nn.Conv2d(c5_ch, FPN_CH, 1)
        self.lat4 = nn.Conv2d(c4_ch, FPN_CH, 1)
        self.lat3 = nn.Conv2d(c3_ch, FPN_CH, 1)
        self.lat2 = nn.Conv2d(c2_ch, FPN_CH, 1)

        self.smooth4 = ConvBnRelu(FPN_CH, FPN_CH)
        self.smooth3 = ConvBnRelu(FPN_CH, FPN_CH)
        self.smooth2 = ConvBnRelu(FPN_CH, FPN_CH)

    def forward(self, c2, c3, c4, c5):
        p5 = self.lat5(c5)
        p4 = self.smooth4(self.lat4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode='bilinear', align_corners=False))
        p3 = self.smooth3(self.lat3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode='bilinear', align_corners=False))
        p2 = self.smooth2(self.lat2(c2) + F.interpolate(p3, size=c2.shape[-2:], mode='bilinear', align_corners=False))
        return p2


class HeatmapHead(nn.Sequential):
    """Small conv head: (B, FPN_CH, H, W) → (B, 1, H, W) in [0, 1]."""

    # Prior probability of a positive pixel (~5 tips spread over 128×128 heatmap).
    # Initialising the bias to log(prior/(1-prior)) makes sigmoid outputs start
    # near `prior` rather than 0.5.  Without this, the 16 000 negative pixels
    # drive the bias sharply negative in the first epoch, collapsing all outputs
    # below the 0.35 recall threshold before the positive signal can compete.
    _PRIOR = 0.01

    def __init__(self):
        super().__init__(
            ConvBnRelu(FPN_CH, 64),
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.constant_(self[-2].bias, math.log(self._PRIOR / (1 - self._PRIOR)))


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
    """Full model: backbone + FPN + tip/score heads."""

    def __init__(self, in_channels: int = 4, backbone: str = 'mobilenet_v2'):
        """
        Args:
            in_channels: 4 = RGB + radial (default), 3 = RGB only.
            backbone: 'mobilenet_v2' (default) or 'mobilenet_v3_large'.
        """
        super().__init__()

        cfg   = BACKBONE_CONFIGS[backbone]
        c2i, c3i, c4i, c5i = cfg['tap_idxs']
        c2ch, c3ch, c4ch, c5ch = cfg['tap_chs']

        # ── backbone ──────────────────────────────────────────────────────
        base  = cfg['builder'](cfg['weights'])
        feats = base.features

        old_conv = feats[0][0]
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                new_conv.weight[:, :3] = old_conv.weight
                new_conv.weight[:, 3:] = 0.0
            feats[0][0] = new_conv

        self.stage2 = nn.Sequential(*feats[:c2i + 1])
        self.stage3 = nn.Sequential(*feats[c2i + 1: c3i + 1])
        self.stage4 = nn.Sequential(*feats[c3i + 1: c4i + 1])
        self.stage5 = nn.Sequential(*feats[c4i + 1: c5i + 1])

        # ── neck + heads ──────────────────────────────────────────────────
        self.fpn        = FPN(c2ch, c3ch, c4ch, c5ch)
        self.tip_head   = HeatmapHead()
        self.score_head = ScoreHead()

    def forward(self, x):
        """
        Args:
            x: (B, 4, 512, 512)
        Returns:
            tip_hm    : (B, 1, 128, 128)
            score_map : (B, 11, 128, 128)
        """
        c2 = self.stage2(x)
        c3 = self.stage3(c2)
        c4 = self.stage4(c3)
        c5 = self.stage5(c4)

        p2 = self.fpn(c2, c3, c4, c5)

        tip_hm    = self.tip_head(p2)
        score_map = self.score_head(p2)

        return tip_hm, score_map


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
                   threshold: float = 0.85,
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
