"""
BoundaryDetector — ResNet50 encoder + UNet-style decoder for binary paper
boundary segmentation.

Input:  (B, 1, 512, 512)  grayscale, normalised
Output: (B, 1, 512, 512)  sigmoid probability map  (paper = 1, background = 0)

The ResNet50 first conv is patched from 3→1 input channels; weights are
averaged across the RGB channels so pretrained ImageNet behaviour is preserved.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights


class ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample 2× + skip connection + two ConvBnRelu."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Sequential(
            ConvBnRelu(in_ch + skip_ch, out_ch),
            ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # Crop skip to match x if sizes differ (can happen with odd input dims)
        if skip.shape[-2:] != x.shape[-2:]:
            skip = skip[:, :, :x.shape[2], :x.shape[3]]
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BoundaryDetector(nn.Module):
    """
    Encoder: ResNet50 (pretrained ImageNet, patched for 1-channel input).
    Decoder: 4-level UNet upsampling path.
    Output:  (B, 1, 512, 512) logits (apply sigmoid for probabilities).
    """

    # ResNet50 feature channels at strides 2, 4, 8, 16, 32
    ENC_CHS = (64, 256, 512, 1024, 2048)

    def __init__(self):
        super().__init__()

        # ── Encoder (ResNet50) ────────────────────────────────────────────────
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        # Patch first conv: 3→1 channels, average the RGB weights
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        with torch.no_grad():
            new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
        backbone.conv1 = new_conv

        # Expose encoder stages as named sub-modules
        self.enc0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2  → 64 ch
        self.pool = backbone.maxpool                                              # /4
        self.enc1 = backbone.layer1   # /4  → 256 ch
        self.enc2 = backbone.layer2   # /8  → 512 ch
        self.enc3 = backbone.layer3   # /16 → 1024 ch
        self.enc4 = backbone.layer4   # /32 → 2048 ch

        # ── Decoder ───────────────────────────────────────────────────────────
        self.dec4 = DecoderBlock(2048, 1024, 512)   # /32 → /16
        self.dec3 = DecoderBlock( 512,  512, 256)   # /16 → /8
        self.dec2 = DecoderBlock( 256,  256, 128)   # /8  → /4
        self.dec1 = DecoderBlock( 128,   64, 64)    # /4  → /2

        # Final upsampling /2 → /1 (no skip at this level)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ConvBnRelu(64, 32),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        e0 = self.enc0(x)       # (B, 64,   H/2,  W/2)
        e1 = self.enc1(self.pool(e0))  # (B, 256,  H/4,  W/4)
        e2 = self.enc2(e1)      # (B, 512,  H/8,  W/8)
        e3 = self.enc3(e2)      # (B, 1024, H/16, W/16)
        e4 = self.enc4(e3)      # (B, 2048, H/32, W/32)

        # Decoder
        d = self.dec4(e4, e3)   # (B, 512,  H/16, W/16)
        d = self.dec3(d,  e2)   # (B, 256,  H/8,  W/8)
        d = self.dec2(d,  e1)   # (B, 128,  H/4,  W/4)
        d = self.dec1(d,  e0)   # (B, 64,   H/2,  W/2)
        return self.final_up(d) # (B, 1,    H,    W)   — logits
