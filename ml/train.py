# FILE: brAIn/ml/train.py
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader, NUM_CLASSES
from model import UNet3D, CombinedLoss, dice_per_class, count_parameters


# ─── LOGGING ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(message)s',
    datefmt='%H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── ARGS ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='BraTS 2021 3D U-Net Trainer')
    p.add_argument('--data_dir',    required=True, help='BraTS dataset root')
    p.add_argument('--output_dir',  default='./runs/exp1')
    p.add_argument('--split_file',  default=None,  help='Optional JSON split file')
    p.add_argument('--epochs',      type=int,   default=300)
    p.add_argument('--batch_size',  type=int,   default=1)
    p.add_argument('--num_workers', type=int,   default=2)
    p.add_argument('--base_filters',type=int,   default=32)
    p.add_argument('--dropout',     type=float, default=0.2)
    p.add_argument('--lr',          type=float, default=1e-4)
    p.add_argument('--weight_decay',type=float, default=1e-5)
    p.add_argument('--warmup_epochs',type=int,  default=5)
    p.add_argument('--dice_weight', type=float, default=0.5)
    p.add_argument('--ce_weight',   type=float, default=0.5)
    p.add_argument('--crop',  nargs=3, type=int, default=[128,128,128],
                   metavar=('H','W','D'), help='Crop size')
    p.add_argument('--val_every',   type=int, default=5,  help='Validate every N epochs')
    p.add_argument('--save_every',  type=int, default=25, help='Checkpoint every N epochs')
    p.add_argument('--patience',    type=int, default=20, help='Epochs to wait before early stopping')
    p.add_argument('--amp',         action='store_true', default=True)
    p.add_argument('--resume',      default=None, help='Path to checkpoint to resume from')
    p.add_argument('--seed',        type=int, default=42)
    return p.parse_args()


# ─── LR SCHEDULER ─────────────────────────────────────────────────────────────

class PolyLRWithWarmup:
    """
    Polynomial decay from lr_max → lr_min over (total_epochs - warmup) steps,
    preceded by linear warm-up.
    """
    def __init__(self, optimizer, total_epochs: int, warmup_epochs: int = 5,
                 lr_max: float = 1e-4, lr_min: float = 1e-6, power: float = 0.9):
        self.opt           = optimizer
        self.total         = total_epochs
        self.warmup        = warmup_epochs
        self.lr_max        = lr_max
        self.lr_min        = lr_min
        self.power         = power
        self.current_epoch = 0

    def step(self):
        e = self.current_epoch
        if e < self.warmup:
            lr = self.lr_max * (e + 1) / self.warmup
        else:
            prog = (e - self.warmup) / max(self.total - self.warmup, 1)
            lr   = self.lr_min + (self.lr_max - self.lr_min) * (1 - prog) ** self.power
        for pg in self.opt.param_groups:
            pg['lr'] = lr
        self.current_epoch += 1
        return lr


# ─── TRAIN ONE EPOCH ──────────────────────────────────────────────────────────

def train_epoch(
    model:     nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    device:    torch.device,
    use_amp:   bool,
    epoch:     int,
) -> float:
    model.train()
    total_loss = 0.0
    n_batches  = len(loader)

    for step, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)  # (B,4,H,W,D)
        labels = batch['label'].to(device, non_blocking=True)  # (B,H,W,D)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(images)                              # (B,4,H,W,D)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        if step % 10 == 0:
            log.info(f"  Ep {epoch:03d} [{step+1:3d}/{n_batches}]  "
                     f"loss={loss.item():.4f}")

    return total_loss / n_batches


# ─── VALIDATE ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(
    model:   nn.Module,
    loader,
    device:  torch.device,
    use_amp: bool,
) -> Tuple[float, Dict[str, float]]:
    """
    Returns (mean_dice_across_tumour_classes, {et, ed, ncr} dice dict).
    BraTS evaluation regions:
        ET  = label 3 (mapped from 4)
        TC  = ET + NCR  (labels 3 + 1)
        WT  = ET + NCR + ED  (all tumour labels)
    We report per-class Dice for {NCR, ED, ET}.
    """
    model.eval()
    all_dices: Dict[str, List[float]] = {'ncr': [], 'ed': [], 'et': []}

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        if not batch['has_seg'][0]:
            continue

        with autocast(enabled=use_amp):
            logits = model(images)

        dices = dice_per_class(logits, labels, num_classes=NUM_CLASSES)
        # indices: 0=BG, 1=NCR, 2=ED, 3=ET
        all_dices['ncr'].append(dices[1])
        all_dices['ed'].append(dices[2])
        all_dices['et'].append(dices[3])

    mean_ncr = float(np.mean(all_dices['ncr'])) if all_dices['ncr'] else 0.0
    mean_ed  = float(np.mean(all_dices['ed']))  if all_dices['ed']  else 0.0
    mean_et  = float(np.mean(all_dices['et']))  if all_dices['et']  else 0.0
    mean_all = (mean_ncr + mean_ed + mean_et) / 3.0

    return mean_all, {'ncr': mean_ncr, 'ed': mean_ed, 'et': mean_et}


# ─── CHECKPOINTING ────────────────────────────────────────────────────────────

def save_checkpoint(
    path:      Path,
    epoch:     int,
    model:     nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler:    GradScaler,
    best_dice: float,
    args,
):
    torch.save({
        'epoch':      epoch,
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'scaler_state': scaler.state_dict(),
        'best_dice':  best_dice,
        'args':       vars(args),
    }, str(path))
    log.info(f"  Saved checkpoint → {path}")


def load_checkpoint(path: str, model, optimizer, scaler):
    ck = torch.load(path, map_location='cpu')
    model.load_state_dict(ck['model_state'])
    optimizer.load_state_dict(ck['optim_state'])
    scaler.load_state_dict(ck['scaler_state'])
    return ck['epoch'], ck.get('best_dice', 0.0)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"Device: {device}")

    # ── Data ──────────────────────────────────────────────────────────────────
    crop = tuple(args.crop)
    train_loader = get_dataloader(
        args.data_dir, mode='train',
        batch_size=args.batch_size, num_workers=args.num_workers,
        split_file=args.split_file, crop_size=crop,
    )
    val_loader = get_dataloader(
        args.data_dir, mode='val',
        batch_size=1, num_workers=args.num_workers,
        split_file=args.split_file, crop_size=crop,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = UNet3D(
        in_channels=4, num_classes=NUM_CLASSES,
        base_filters=args.base_filters, dropout=args.dropout,
    ).to(device)
    log.info(f"Model parameters: {count_parameters(model)}")

    # Multi-GPU
    if torch.cuda.device_count() > 1:
        log.info(f"Using {torch.cuda.device_count()} GPUs (DataParallel)")
        model = nn.DataParallel(model)

    # ── Optimiser & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler    = GradScaler(enabled=args.amp and device.type == 'cuda')
    scheduler = PolyLRWithWarmup(
        optimizer, total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, lr_max=args.lr,
    )
    criterion = CombinedLoss(
        dice_weight=args.dice_weight, ce_weight=args.ce_weight
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_dice   = 0.0
    epochs_no_improve = 0
    if args.resume:
        log.info(f"Resuming from {args.resume}")
        start_epoch, best_dice = load_checkpoint(
            args.resume, model, optimizer, scaler
        )
        start_epoch += 1  # continue from next epoch
        for _ in range(start_epoch):
            scheduler.step()

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(str(outdir / 'tb'))

    # ── Training loop ─────────────────────────────────────────────────────────
    log.info("═" * 60)
    log.info(f"Starting training: {args.epochs} epochs, batch={args.batch_size}")
    log.info("═" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # LR step
        current_lr = scheduler.step()
        writer.add_scalar('LR', current_lr, epoch)

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, args.amp, epoch,
        )
        writer.add_scalar('Loss/train', train_loss, epoch)
        elapsed = time.time() - t0

        log.info(f"Epoch {epoch:03d}/{args.epochs}  "
                 f"train_loss={train_loss:.4f}  "
                 f"lr={current_lr:.2e}  "
                 f"time={elapsed:.1f}s")

        # Validate
        if (epoch + 1) % args.val_every == 0 or epoch == args.epochs - 1:
            mean_dice, dices = validate(model, val_loader, device, args.amp)
            writer.add_scalar('Dice/mean',  mean_dice,      epoch)
            writer.add_scalar('Dice/ET',    dices['et'],    epoch)
            writer.add_scalar('Dice/ED',    dices['ed'],    epoch)
            writer.add_scalar('Dice/NCR',   dices['ncr'],   epoch)

            log.info(f"  Val Dice — mean={mean_dice:.4f}  "
                     f"ET={dices['et']:.4f}  "
                     f"ED={dices['ed']:.4f}  "
                     f"NCR={dices['ncr']:.4f}")

            # Save best & Early Stopping Logic
            if mean_dice > best_dice:
                best_dice = mean_dice
                epochs_no_improve = 0  # Reset patience counter
                save_checkpoint(
                    outdir / 'best_model.pth', epoch,
                    model, optimizer, scaler, best_dice, args,
                )
                log.info(f"  ★ New best Dice: {best_dice:.4f}")
            else:
                epochs_no_improve += args.val_every
                log.info(f"  Early stopping counter: {epochs_no_improve}/{args.patience}")

            # Check if patience has run out
            if epochs_no_improve >= args.patience:
                log.info(f"🛑 Early stopping triggered. Validation Dice hasn't improved in {epochs_no_improve} epochs.")
                break  # Exit the training loop entirely

        # Periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                outdir / f'checkpoint_ep{epoch:04d}.pth', epoch,
                model, optimizer, scaler, best_dice, args,
            )

    writer.close()
    log.info(f"Training complete. Best val Dice: {best_dice:.4f}")
    log.info(f"Best model saved to: {outdir / 'best_model.pth'}")


if __name__ == '__main__':
    main()