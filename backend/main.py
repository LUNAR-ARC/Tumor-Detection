# FILE: backend/main.py
"""
brAIn — FastAPI Inference Backend (local)

Run from the project root:
    cd backend
    uvicorn main:app --host 127.0.0.1 --port 8000 --reload

Install deps:
    pip install fastapi uvicorn[standard] python-multipart nibabel torch numpy
"""

import io
import os
import sys
import uuid
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'ml'))
from model import UNet3D, NUM_CLASSES

# ── Config ────────────────────────────────────────────────────────────────────

MODEL_PATH   = os.getenv('MODEL_PATH', '../ml/runs/exp1/best_model.pth')
BASE_FILTERS = int(os.getenv('BASE_FILTERS', '32'))
CROP_SIZE    = (128, 128, 128)
MAX_FILE_MB  = 600
LABEL_UNMAP  = {0: 0, 1: 1, 2: 2, 3: 4}

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s  %(levelname)-8s  %(message)s')
log = logging.getLogger('brAIn')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Device: {DEVICE}")

# ── Model ─────────────────────────────────────────────────────────────────────

def load_model(path: str) -> Optional[UNet3D]:
    if not os.path.exists(path):
        log.warning(f"Checkpoint not found at '{path}' — running in demo mode.")
        return None
    try:
        model = UNet3D(in_channels=4, num_classes=NUM_CLASSES,
                       base_filters=BASE_FILTERS).to(DEVICE)
        ck    = torch.load(path, map_location=DEVICE)
        state = {k.replace('module.', ''): v
                 for k, v in ck.get('model_state', ck).items()}
        model.load_state_dict(state)
        model.eval()
        log.info(f"Model loaded from {path}")
        return model
    except Exception as e:
        log.error(f"Failed to load model: {e}")
        return None

MODEL: Optional[UNet3D] = load_model(MODEL_PATH)

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title='brAIn — MRI Tumour Segmentation', version='1.0.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost', 'http://127.0.0.1',
                   'null',            # file:// origin
                   '*'],
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── Helpers ───────────────────────────────────────────────────────────────────

async def _read(f: UploadFile) -> bytes:
    data = await f.read()
    if len(data) / 1024**2 > MAX_FILE_MB:
        raise HTTPException(413, f"{f.filename} exceeds {MAX_FILE_MB} MB limit.")
    return data

def _load_nifti(data: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        tmp.write(data); path = tmp.name
    try:
        return nib.load(path).get_fdata(dtype=np.float32)
    finally:
        os.unlink(path)

def _zscore(arr: np.ndarray) -> np.ndarray:
    mask = arr > 0
    if not mask.any(): return arr
    mu, std = arr[mask].mean(), arr[mask].std() + 1e-8
    return np.where(mask, (arr - mu) / std, 0.0)

def _pad_or_crop(arr: np.ndarray, target: tuple) -> np.ndarray:
    H, W, D = arr.shape; th, tw, td = target
    h0 = max((H-th)//2, 0); w0 = max((W-tw)//2, 0); d0 = max((D-td)//2, 0)
    arr = arr[h0:h0+th, w0:w0+tw, d0:d0+td]
    arr = np.pad(arr, [(0, max(th-arr.shape[0], 0)),
                       (0, max(tw-arr.shape[1], 0)),
                       (0, max(td-arr.shape[2], 0))])
    return arr

def _unmap(seg: np.ndarray) -> np.ndarray:
    out = np.zeros_like(seg)
    for s, d in LABEL_UNMAP.items(): out[seg == s] = d
    return out

def _demo_mask(shape: tuple) -> np.ndarray:
    H, W, D = shape; vol = np.zeros(shape, dtype=np.uint8)
    cx, cy, cz = H*.55, W*.48, D*.50
    for z in range(D):
        for y in range(W):
            for x in range(H):
                dx=(x-cx)/H; dy=(y-cy)/W; dz=(z-cz)/D
                r = np.sqrt(dx*dx*3 + dy*dy*3 + dz*dz*5)
                if   r < .07: vol[x,y,z] = 4
                elif r < .10: vol[x,y,z] = 1
                elif r < .15: vol[x,y,z] = 2
    return vol

def _to_nifti_bytes(seg: np.ndarray) -> bytes:
    img = nib.Nifti1Image(seg.astype(np.uint8), np.eye(4))
    img.header.set_data_dtype(np.uint8)
    with tempfile.NamedTemporaryFile(suffix='.nii.gz', delete=False) as tmp:
        nib.save(img, tmp.name); path = tmp.name
    try:
        with open(path, 'rb') as f: return f.read()
    finally:
        os.unlink(path)

# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(volumes: dict) -> np.ndarray:
    orig_shape = volumes['flair'].shape
    arrays = [_pad_or_crop(_zscore(volumes[m]), CROP_SIZE)
              for m in ['t1', 't1ce', 't2', 'flair']]
    x = torch.from_numpy(np.stack(arrays)[np.newaxis]).float().to(DEVICE)

    if MODEL is None:
        seg_crop = _demo_mask(CROP_SIZE)
    else:
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
                logits = MODEL(x)
            seg_crop = logits.argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        seg_crop = _unmap(seg_crop)

    seg_t = torch.from_numpy(seg_crop.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    return F.interpolate(seg_t, size=orig_shape, mode='nearest') \
             .squeeze().numpy().astype(np.uint8)

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get('/')
def root():
    return {'service': 'brAIn', 'device': str(DEVICE),
            'model_loaded': MODEL is not None}

@app.get('/health')
def health():
    return {'status': 'ok', 'model_ready': MODEL is not None}

@app.post('/segment')
async def segment(
    t1:         UploadFile = File(...),
    t1ce:       UploadFile = File(...),
    t2:         UploadFile = File(...),
    flair:      UploadFile = File(...),
    patient_id: str        = Form(default='UNKNOWN'),
):
    for f, lbl in [(t1,'T1'),(t1ce,'T1ce'),(t2,'T2'),(flair,'FLAIR')]:
        if not (f.filename or '').lower().endswith(('.nii', '.nii.gz')):
            raise HTTPException(400, f"{lbl}: only .nii / .nii.gz accepted.")

    job_id  = str(uuid.uuid4())[:8]
    t_start = time.time()
    log.info(f"[{job_id}] patient={patient_id}")

    try:
        vols_raw = {
            't1':    await _read(t1),
            't1ce':  await _read(t1ce),
            't2':    await _read(t2),
            'flair': await _read(flair),
        }
    except HTTPException: raise
    except Exception as e: raise HTTPException(400, f"File read error: {e}")

    try:
        volumes = {k: _load_nifti(v) for k, v in vols_raw.items()}
    except Exception as e:
        raise HTTPException(422, f"NIfTI parsing failed: {e}")

    try:
        seg = run_inference(volumes)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(507, "GPU out of memory — try CPU or reduce volume size.")
    except Exception as e:
        log.exception(f"[{job_id}] Inference error")
        raise HTTPException(500, f"Inference failed: {e}")

    elapsed = time.time() - t_start
    log.info(f"[{job_id}] done in {elapsed:.1f}s  labels={np.unique(seg).tolist()}")

    try:
        seg_bytes = _to_nifti_bytes(seg)
    except Exception as e:
        raise HTTPException(500, f"NIfTI encoding failed: {e}")

    return StreamingResponse(
        io.BytesIO(seg_bytes),
        media_type='application/gzip',
        headers={
            'X-Job-Id':        job_id,
            'X-Patient-Id':    patient_id,
            'X-Elapsed-Sec':   f'{elapsed:.2f}',
            'Content-Disposition': f'attachment; filename="{patient_id}_seg.nii.gz"',
        },
    )

@app.post('/reload-model')
def reload_model():
    global MODEL
    MODEL = load_model(MODEL_PATH)
    return {'model_loaded': MODEL is not None, 'path': MODEL_PATH}

@app.get('/model-info')
def model_info():
    if MODEL is None:
        return {'loaded': False, 'mode': 'demo'}
    return {
        'loaded': True, 'path': MODEL_PATH, 'device': str(DEVICE),
        'parameters': sum(p.numel() for p in MODEL.parameters()),
        'base_filters': BASE_FILTERS, 'num_classes': NUM_CLASSES,
        'crop_size': CROP_SIZE,
    }