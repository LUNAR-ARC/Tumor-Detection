# FILE: ml/dataset.py
"""
BraTS 2021 Task 1 – Custom PyTorch DataLoader
Streams heavy 3D NIfTI volumes from disk with minimal RAM footprint.
Designed for batch_size=1 or 2 on consumer-grade hardware.
"""

import os
import json
import random
import numpy as np
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List, Dict


# ─── LABEL MAP ────────────────────────────────────────────────────────────────
# BraTS 2021 raw labels → model output channels
# 0 = background, 1 = NCR (necrotic core), 2 = ED (edema), 4 = ET (enhancing)
# We remap 4 → 3 so labels are contiguous: {0,1,2,3}
LABEL_MAP = {0: 0, 1: 1, 2: 2, 4: 3}
NUM_CLASSES = 4  # background, NCR, ED, ET


# ─── AUGMENTATION HELPERS ─────────────────────────────────────────────────────

def random_flip(img, seg):
    # Randomly choose a spatial axis: 1 (H), 2 (W), or 3 (D) for the 4D image
    axis = np.random.choice([1, 2, 3])
    
    # Flip the 4D image (Channels, H, W, D)
    img = np.flip(img, axis=axis).copy()
    
    # Flip the 3D mask (H, W, D). 
    # Because it lacks the channel dimension, its spatial axes are 0, 1, 2.
    seg = np.flip(seg, axis=axis-1).copy() 
    
    return img, seg


def random_intensity_shift(img: np.ndarray, shift_range=0.1, scale_range=0.1) -> np.ndarray:
    """Per-channel random brightness shift and contrast scale."""
    for c in range(img.shape[0]):
        shift = random.uniform(-shift_range, shift_range)
        scale = random.uniform(1 - scale_range, 1 + scale_range)
        img[c] = img[c] * scale + shift
    return img


def random_crop(img: np.ndarray, seg: np.ndarray,
                crop_size: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """Random spatial crop. Falls back to centre-crop if volume is smaller."""
    _, H, W, D = img.shape
    ch, cw, cd = crop_size
    h0 = random.randint(0, max(H - ch, 0))
    w0 = random.randint(0, max(W - cw, 0))
    d0 = random.randint(0, max(D - cd, 0))
    return (
        img[:, h0:h0+ch, w0:w0+cw, d0:d0+cd],
        seg[   h0:h0+ch, w0:w0+cw, d0:d0+cd],
    )


def centre_crop(img: np.ndarray, seg: np.ndarray,
                crop_size: Tuple[int,int,int]) -> Tuple[np.ndarray, np.ndarray]:
    """Deterministic centre crop."""
    _, H, W, D = img.shape
    ch, cw, cd = crop_size
    h0, w0, d0 = (H - ch) // 2, (W - cw) // 2, (D - cd) // 2
    h0, w0, d0 = max(h0, 0), max(w0, 0), max(d0, 0)
    return (
        img[:, h0:h0+ch, w0:w0+cw, d0:d0+cd],
        seg[   h0:h0+ch, w0:w0+cw, d0:d0+cd],
    )


def zscore_normalise(img: np.ndarray) -> np.ndarray:
    """Z-score normalisation per channel, masking background (zero) voxels."""
    for c in range(img.shape[0]):
        channel = img[c]
        mask    = channel > 0
        if mask.sum() == 0:
            continue
        mu  = channel[mask].mean()
        std = channel[mask].std() + 1e-8
        img[c] = np.where(mask, (channel - mu) / std, 0.0)
    return img


def remap_labels(seg: np.ndarray) -> np.ndarray:
    """Map BraTS raw labels {0,1,2,4} → {0,1,2,3}."""
    out = np.zeros_like(seg)
    for src, dst in LABEL_MAP.items():
        out[seg == src] = dst
    return out


# ─── DATASET ──────────────────────────────────────────────────────────────────

class BraTS2021Dataset(Dataset):
    """
    Streams BraTS 2021 Task 1 cases from disk.

    Expected directory layout:
        brats_root/
          BraTS2021_00000/
            BraTS2021_00000_t1.nii.gz
            BraTS2021_00000_t1ce.nii.gz
            BraTS2021_00000_t2.nii.gz
            BraTS2021_00000_flair.nii.gz
            BraTS2021_00000_seg.nii.gz   ← absent during inference
          BraTS2021_00001/
            ...

    Args:
        root_dir:   Path to the dataset root.
        mode:       'train' | 'val' | 'test'
        split_file: Optional JSON with {"train":[...], "val":[...]} case IDs.
                    If None, an 80/20 split is generated automatically.
        crop_size:  Spatial crop applied to each volume (H, W, D).
        augment:    Apply random flips + intensity jitter during training.
        cache_meta: Keep a small metadata dict per case (no voxels cached).
    """

    MODALITIES = ['t1', 't1ce', 't2', 'flair']

    def __init__(
        self,
        root_dir:   str,
        mode:       str = 'train',
        split_file: Optional[str] = None,
        crop_size:  Tuple[int,int,int] = (128, 128, 128),
        augment:    bool = True,
        cache_meta: bool = True,
    ):
        super().__init__()
        self.root      = Path(root_dir)
        self.mode      = mode
        self.crop_size = crop_size
        self.augment   = augment and (mode == 'train')
        self.cases: List[Path] = []

        # Discover all cases
        all_cases = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not all_cases:
            raise FileNotFoundError(f"No subdirectories found in {root_dir}")

        # Load or build split
        if split_file and os.path.exists(split_file):
            with open(split_file) as f:
                splits = json.load(f)
            ids = splits.get(mode, [])
            self.cases = [self.root / i for i in ids if (self.root / i).exists()]
        else:
            random.seed(42)
            random.shuffle(all_cases)
            n = len(all_cases)
            split_idx = int(n * 0.8)
            if mode == 'train':
                self.cases = all_cases[:split_idx]
            elif mode == 'val':
                self.cases = all_cases[split_idx:]
            else:  # test – use all
                self.cases = all_cases

        if not self.cases:
            raise ValueError(f"No cases found for mode='{mode}'")

        print(f"[BraTS2021Dataset] mode={mode}  cases={len(self.cases)}  "
              f"crop={crop_size}  augment={self.augment}")

    def __len__(self) -> int:
        return len(self.cases)

    def _load_nifti(self, path: Path) -> np.ndarray:
        """Load a NIfTI file and return float32 ndarray."""
        nib_img = nib.load(str(path))
        return nib_img.get_fdata(dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case_dir  = self.cases[idx]
        case_name = case_dir.name

        # ── Load modalities ──────────────────────────────────────────────────
        imgs = []
        for mod in self.MODALITIES:
            path = case_dir / f"{case_name}_{mod}.nii.gz"
            if not path.exists():
                path = case_dir / f"{case_name}_{mod}.nii"
            arr = self._load_nifti(path)   # (H, W, D)
            imgs.append(arr)

        img = np.stack(imgs, axis=0).astype(np.float32)  # (4, H, W, D)

        # ── Load segmentation mask (optional) ───────────────────────────────
        seg_path = case_dir / f"{case_name}_seg.nii.gz"
        if not seg_path.exists():
            seg_path = case_dir / f"{case_name}_seg.nii"

        has_seg = seg_path.exists()
        if has_seg:
            seg = self._load_nifti(seg_path).astype(np.int64)  # (H, W, D)
            seg = remap_labels(seg)
        else:
            seg = np.zeros(img.shape[1:], dtype=np.int64)

        # ── Normalise ────────────────────────────────────────────────────────
        img = zscore_normalise(img)

        # ── Spatial crop ─────────────────────────────────────────────────────
        if self.augment:
            img, seg = random_crop(img, seg, self.crop_size)
            img, seg = random_flip(img, seg)
            img       = random_intensity_shift(img)
        else:
            img, seg = centre_crop(img, seg, self.crop_size)

        # ── To tensors ───────────────────────────────────────────────────────
        img_t = torch.from_numpy(img)           # (4, H, W, D)  float32
        seg_t = torch.from_numpy(seg).long()    # (H, W, D)     int64

        return {
            'image':     img_t,
            'label':     seg_t,
            'case_name': case_name,
            'has_seg':   has_seg,
        }


# ─── DATALOADER FACTORY ───────────────────────────────────────────────────────

def get_dataloader(
    root_dir:    str,
    mode:        str       = 'train',
    batch_size:  int       = 1,
    num_workers: int       = 2,
    split_file:  Optional[str] = None,
    crop_size:   Tuple[int,int,int] = (128, 128, 128),
    pin_memory:  bool      = True,
) -> DataLoader:
    """
    Returns a DataLoader configured for streaming large 3D NIfTI data.

    Keep batch_size=1 or 2 to avoid OOM on consumer GPUs.
    num_workers=2 is usually optimal — each worker decompresses gzip.
    """
    dataset = BraTS2021Dataset(
        root_dir   = root_dir,
        mode       = mode,
        split_file = split_file,
        crop_size  = crop_size,
        augment    = (mode == 'train'),
    )
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = (mode == 'train'),
        num_workers = num_workers,
        pin_memory  = pin_memory and torch.cuda.is_available(),
        persistent_workers = (num_workers > 0),
        prefetch_factor    = 2 if num_workers > 0 else None,
        drop_last   = (mode == 'train'),
    )


# ─── QUICK SANITY CHECK ───────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else './brats2021'
    loader = get_dataloader(root, mode='train', batch_size=1, num_workers=0)
    batch  = next(iter(loader))
    print('image shape:', batch['image'].shape)   # (1, 4, 128, 128, 128)
    print('label shape:', batch['label'].shape)   # (1, 128, 128, 128)
    print('unique labels:', batch['label'].unique())
    print('case:', batch['case_name'])
