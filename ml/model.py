# FILE: ml/model.py
"""
3D U-Net for BraTS 2021 multi-class brain tumour segmentation.

Architecture:
  • Encoder: 4 down-sampling stages (residual double-conv + max-pool)
  • Bottleneck: residual double-conv with dropout
  • Decoder: 4 up-sampling stages (transposed conv + skip + residual double-conv)
  • Output: 1×1×1 conv → NUM_CLASSES channels → softmax (or raw logits for CE loss)

Input:  (B, 4, H, W, D)   – four MRI modalities
Output: (B, 4, H, W, D)   – per-voxel class logits {background, NCR, ED, ET}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


NUM_CLASSES = 4  # 0=BG, 1=NCR, 2=ED, 3=ET


# ─── BUILDING BLOCKS ──────────────────────────────────────────────────────────

class ConvBnRelu3d(nn.Module):
    """3D Conv → InstanceNorm → LeakyReLU."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.InstanceNorm3d(out_ch, affine=True),
            nn.LeakyReLU(0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResDoubleConv(nn.Module):
    """
    Two ConvBnRelu blocks with an optional residual projection.
    Residual connection improves gradient flow in deep networks.
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = ConvBnRelu3d(in_ch,  out_ch)
        self.conv2 = ConvBnRelu3d(out_ch, out_ch)
        # 1×1×1 projection if channel counts differ
        self.proj = (
            nn.Conv3d(in_ch, out_ch, 1, bias=False)
            if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.conv1(x)
        out = self.conv2(out)
        return F.leaky_relu(out + residual, 0.01, inplace=True)


class Down(nn.Module):
    """ResDoubleConv → MaxPool3d (2×2×2)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = ResDoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        skip  = self.conv(x)
        pooled = self.pool(skip)
        return pooled, skip   # pooled goes forward; skip saved for decoder


class Up(nn.Module):
    """TransposedConv3d (2×2×2 stride 2) → concat skip → ResDoubleConv."""
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose3d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        self.conv = ResDoubleConv(in_ch // 2 + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Handle odd-sized volumes by padding if necessary
        diff = [skip.size(i) - x.size(i) for i in range(2, 5)]
        x = F.pad(x, [0, diff[2], 0, diff[1], 0, diff[0]])
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


# ─── 3D U-NET ─────────────────────────────────────────────────────────────────

class UNet3D(nn.Module):
    """
    Fully volumetric 3D U-Net.

    Args:
        in_channels:  Number of input modalities (4 for BraTS).
        num_classes:  Number of output segmentation classes (4 for BraTS).
        base_filters: Feature maps in the first encoder stage.
                      Doubles at each down-sampling step.
        dropout:      Dropout probability applied at the bottleneck.

    Memory:
        base_filters=32, input=(1,4,128,128,128) → ~7 GB GPU RAM.
        Reduce to base_filters=16 for 6 GB cards.
    """

    def __init__(
        self,
        in_channels:  int   = 4,
        num_classes:  int   = NUM_CLASSES,
        base_filters: int   = 32,
        dropout:      float = 0.2,
    ):
        super().__init__()
        f = base_filters  # shorthand

        # ── Encoder ──────────────────────────────────────────────────────────
        self.enc1 = Down(in_channels, f)      # (B, f,   H/2, W/2, D/2)
        self.enc2 = Down(f,     f * 2)        # (B, 2f,  H/4, ...)
        self.enc3 = Down(f * 2, f * 4)        # (B, 4f,  H/8, ...)
        self.enc4 = Down(f * 4, f * 8)        # (B, 8f,  H/16, ...)

        # ── Bottleneck ───────────────────────────────────────────────────────
        self.bottleneck = nn.Sequential(
            ResDoubleConv(f * 8, f * 16),
            nn.Dropout3d(dropout),
        )                                      # (B, 16f, H/16, ...)

        # ── Decoder ──────────────────────────────────────────────────────────
        self.dec4 = Up(f * 16, f * 8,  f * 8)
        self.dec3 = Up(f * 8,  f * 4,  f * 4)
        self.dec2 = Up(f * 4,  f * 2,  f * 2)
        self.dec1 = Up(f * 2,  f,      f)

        # ── Segmentation head ────────────────────────────────────────────────
        self.head = nn.Conv3d(f, num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm3d) and m.affine:
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ── Encode ──────────────────────────────────────────────────────────
        x, s1 = self.enc1(x)   # s1: skip from stage 1
        x, s2 = self.enc2(x)
        x, s3 = self.enc3(x)
        x, s4 = self.enc4(x)

        # ── Bottleneck ───────────────────────────────────────────────────────
        x = self.bottleneck(x)

        # ── Decode ───────────────────────────────────────────────────────────
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)

        return self.head(x)   # (B, num_classes, H, W, D) – raw logits


# ─── LOSS FUNCTIONS ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft multi-class Dice loss.
    Combines well with CrossEntropyLoss for segmentation tasks.
    """
    def __init__(self, smooth: float = 1e-5, ignore_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (B, C, H, W, D) – raw model output
        targets: (B, H, W, D)    – integer labels
        """
        num_classes = logits.shape[1]
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets: (B, C, H, W, D)
        targets_oh = F.one_hot(targets, num_classes).permute(0, 4, 1, 2, 3).float()

        start_cls = 1 if self.ignore_bg else 0
        dice_sum = 0.0
        count    = 0

        for c in range(start_cls, num_classes):
            p = probs[:, c]
            g = targets_oh[:, c]
            intersection = (p * g).sum()
            dice_sum += (2 * intersection + self.smooth) / (p.sum() + g.sum() + self.smooth)
            count += 1

        return 1.0 - dice_sum / max(count, 1)


class CombinedLoss(nn.Module):
    """Dice + Cross-Entropy with configurable weighting."""
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5):
        super().__init__()
        self.dice = DiceLoss(ignore_background=True)
        self.ce   = nn.CrossEntropyLoss()
        self.dw   = dice_weight
        self.cew  = ce_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.dw * self.dice(logits, targets) + self.cew * self.ce(logits, targets)


# ─── METRICS ──────────────────────────────────────────────────────────────────

def dice_per_class(
    logits:      torch.Tensor,
    targets:     torch.Tensor,
    num_classes: int   = NUM_CLASSES,
    smooth:      float = 1e-5,
) -> List[float]:
    """
    Compute per-class hard Dice score.
    Returns list of length num_classes (index 0 = background).
    """
    preds   = logits.argmax(dim=1)  # (B, H, W, D)
    scores  = []
    for c in range(num_classes):
        pred_c   = (preds == c).float()
        target_c = (targets == c).float()
        inter    = (pred_c * target_c).sum().item()
        union    = pred_c.sum().item() + target_c.sum().item()
        scores.append((2 * inter + smooth) / (union + smooth))
    return scores


# ─── MODEL SUMMARY ────────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> str:
    total  = sum(p.numel() for p in model.parameters())
    train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return f"Total: {total:,}  |  Trainable: {train:,}"


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = UNet3D(in_channels=4, num_classes=4, base_filters=32).to(device)
    print(count_parameters(model))

    # Dry run with tiny input (memory check)
    x = torch.randn(1, 4, 64, 64, 64).to(device)
    with torch.no_grad():
        out = model(x)
    print(f"Input:  {tuple(x.shape)}")
    print(f"Output: {tuple(out.shape)}")   # (1, 4, 64, 64, 64)
