# FILE: ml/predict.py
"""
CLI inference — segment a single BraTS case without the API.

Usage:
    python predict.py \
        --t1    /data/case/case_t1.nii.gz \
        --t1ce  /data/case/case_t1ce.nii.gz \
        --t2    /data/case/case_t2.nii.gz \
        --flair /data/case/case_flair.nii.gz \
        --checkpoint ../ml/runs/exp1/best_model.pth \
        --output ./case_seg.nii.gz
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from model import UNet3D, NUM_CLASSES

CROP_SIZE  = (128, 128, 128)
LABEL_UNMAP = {0: 0, 1: 1, 2: 2, 3: 4}


def zscore(arr):
    mask = arr > 0
    if mask.sum() == 0:
        return arr
    mu  = arr[mask].mean()
    std = arr[mask].std() + 1e-8
    return np.where(mask, (arr - mu) / std, 0.0).astype(np.float32)


def pad_or_crop(arr, target):
    H, W, D = arr.shape; th, tw, td = target
    h0 = max((H - th) // 2, 0); w0 = max((W - tw) // 2, 0); d0 = max((D - td) // 2, 0)
    arr = arr[h0:h0+th, w0:w0+tw, d0:d0+td]
    ph  = max(th - arr.shape[0], 0); pw = max(tw - arr.shape[1], 0); pd = max(td - arr.shape[2], 0)
    return np.pad(arr, [(0,ph),(0,pw),(0,pd)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--t1',         required=True)
    p.add_argument('--t1ce',       required=True)
    p.add_argument('--t2',         required=True)
    p.add_argument('--flair',      required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output',     default='./segmentation.nii.gz')
    p.add_argument('--base_filters', type=int, default=32)
    p.add_argument('--device',     default='auto')
    args = p.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # Load model
    model = UNet3D(in_channels=4, num_classes=NUM_CLASSES,
                   base_filters=args.base_filters).to(device)
    ck = torch.load(args.checkpoint, map_location=device)
    state = ck.get('model_state', ck)
    state = {k.replace('module.', ''): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    print("Model loaded.")

    # Load volumes
    mods = {'t1': args.t1, 't1ce': args.t1ce, 't2': args.t2, 'flair': args.flair}
    vols = {}
    for mod, path in mods.items():
        nib_img = nib.load(path)
        vols[mod] = nib_img.get_fdata(dtype=np.float32)
        print(f"  {mod}: {vols[mod].shape}")

    affine     = nib.load(args.flair).affine
    orig_shape = vols['flair'].shape

    # Preprocess
    arrays = [pad_or_crop(zscore(vols[m]), CROP_SIZE) for m in ['t1','t1ce','t2','flair']]
    x = torch.from_numpy(np.stack(arrays)[np.newaxis]).float().to(device)

    # Infer
    print("Running inference…")
    with torch.no_grad():
        logits = model(x)   # (1, 4, 128, 128, 128)
    seg = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Remap labels
    out = np.zeros_like(seg)
    for src, dst in LABEL_UNMAP.items():
        out[seg == src] = dst

    # Resize to original shape
    seg_t = torch.from_numpy(out.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    seg_resized = F.interpolate(seg_t, size=orig_shape, mode='nearest') \
                    .squeeze().numpy().astype(np.uint8)

    # Save
    nib.save(nib.Nifti1Image(seg_resized, affine), args.output)
    print(f"Saved → {args.output}")
    labels, counts = np.unique(seg_resized, return_counts=True)
    for l, c in zip(labels, counts):
        names = {0:'Background', 1:'NCR', 2:'ED', 4:'ET'}
        print(f"  Label {l} ({names.get(l,'?')}): {c:,} voxels")


if __name__ == '__main__':
    main()
