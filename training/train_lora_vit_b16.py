import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import os
from tqdm import tqdm
import torch.nn.functional as F
import math
from functools import partial

# Import official modules
from models import get_model
from losses import DACELoss

from my_dataset import custom_collate, create_train_val_split



def calculate_metrics(pred_counts, gt_counts):
    """Calculate MAE and RMSE"""
    mae = np.mean(np.abs(pred_counts - gt_counts))
    rmse = np.sqrt(np.mean((pred_counts - gt_counts) ** 2))
    return {'mae': mae, 'rmse': rmse}


def train_one_epoch(model, dataloader, criterion, optimizer, grad_scaler, device, epoch, max_count=17):
    """Train for one epoch"""
    model.train()

    epoch_loss = 0.0
    epoch_ce_loss = 0.0
    epoch_count_loss = 0.0

    hist_bins = max_count + 2
    hist_counts = torch.zeros(hist_bins, dtype=torch.int64, device=device)
    total_cells = 0
    cell_sum_acc = torch.tensor(0.0, device=device)
    cell_max = torch.tensor(0.0, device=device)

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, points, densities) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        densities = densities.to(device, non_blocking=True)
        points = [p.to(device, non_blocking=True) for p in points]

        with torch.no_grad():
            flat = densities.flatten()
            bucket = flat.clamp(min=0, max=max_count + 1).long()
            hist_counts += torch.bincount(bucket, minlength=hist_bins)
            total_cells += flat.numel()
            cell_sum_acc += flat.sum()
            cell_max = torch.maximum(cell_max, flat.max())

        optimizer.zero_grad(set_to_none=True)

        with autocast('cuda'):
            # Model returns (logits, density) during training
            logits, pred_density = model(images)

            # Compute DACE loss
            loss, loss_info = criterion(
                pred_class=logits,
                pred_density=pred_density,
                target_density=densities,
                target_points=points
            )

        # Backward pass with gradient clipping
        grad_scaler.scale(loss).backward()
        grad_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], max_norm=1.0
        )
        grad_scaler.step(optimizer)
        grad_scaler.update()

        # Accumulate metrics
        epoch_loss += loss_info['loss'].item()
        epoch_ce_loss += loss_info['ce_loss'].item()

        # Handle different count loss names (dmcount returns ot_loss/tv_loss/count_loss)
        for key in ['count_loss', 'mae_loss', 'mse_loss']:
            if key in loss_info:
                epoch_count_loss += loss_info[key].item()
                break

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss_info['loss'].item():.4f}",
            'ce': f"{loss_info['ce_loss'].item():.4f}",
        })

    hist_cpu = hist_counts.cpu().numpy()
    total = int(hist_cpu.sum())
    nonzero = total - int(hist_cpu[0])
    mean_per_cell = float(cell_sum_acc.item() / max(total_cells, 1))
    max_per_cell = float(cell_max.item())

    def percentile_from_hist(counts, p):
        target = p / 100.0 * counts.sum()
        cum = 0
        for i, c in enumerate(counts):
            cum += c
            if cum >= target:
                return i
        return len(counts) - 1

    p50 = percentile_from_hist(hist_cpu, 50)
    p90 = percentile_from_hist(hist_cpu, 90)
    p99 = percentile_from_hist(hist_cpu, 99)
    p999 = percentile_from_hist(hist_cpu, 99.9)
    frac_saturated = float(hist_cpu[max_count]) / max(total, 1)
    frac_overflow = float(hist_cpu[max_count + 1]) / max(total, 1)

    print(f"\n  [per-cell density histogram — epoch {epoch}, max_count={max_count}]")
    print(f"    total cells: {total_cells:,}  nonzero: {nonzero:,} ({100*nonzero/max(total,1):.2f}%)")
    print(f"    mean/cell: {mean_per_cell:.3f}   max/cell (raw): {max_per_cell:.1f}")
    print(f"    p50={p50}  p90={p90}  p99={p99}  p99.9={p999}")
    print(f"    frac cells == max_count ({max_count}): {frac_saturated*100:.3f}%   above max_count: {frac_overflow*100:.3f}%")
    for i in range(min(hist_bins, max_count + 2)):
        bar = '#' * int(50 * hist_cpu[i] / max(hist_cpu.max(), 1))
        label = f"{i}" if i <= max_count else f">{max_count}"
        print(f"    {label:>4}  {hist_cpu[i]:>12,}  {bar}")

    # Return average metrics
    num_batches = len(dataloader)
    return {
        'loss': epoch_loss / num_batches,
        'ce_loss': epoch_ce_loss / num_batches,
        'count_loss': epoch_count_loss / num_batches,
        'hist': hist_cpu.tolist(),
        'mean_per_cell': mean_per_cell,
        'max_per_cell': max_per_cell,
    }


@torch.no_grad()
def _sliding_window_count(model, image_tensor, window_size=224, stride=224):
    """Sum predicted density over a native-resolution image via non-overlapping
    sliding windows. Zero-pads the bottom/right to a multiple of stride.
    Returns a Python float (total count)."""
    _, _, h, w = image_tensor.shape
    pad_h = max(0, window_size - h) + ((stride - (max(h, window_size) - window_size) % stride) % stride)
    pad_w = max(0, window_size - w) + ((stride - (max(w, window_size) - window_size) % stride) % stride)
    if pad_h or pad_w:
        image_tensor = F.pad(image_tensor, (0, pad_w, 0, pad_h), value=0.0)
    _, _, H, W = image_tensor.shape

    total = 0.0
    for top in range(0, H - window_size + 1, stride):
        for left in range(0, W - window_size + 1, stride):
            window = image_tensor[:, :, top:top + window_size, left:left + window_size]
            out = model(window)
            if isinstance(out, tuple):
                _, out = out
            total += float(out.sum().item())
    return total


@torch.no_grad()
def evaluate(model, val_subset, device, window_size=224, stride=224):
    """Sliding-window validation at NATIVE resolution. Matches inference
    scale used by count_in_region.py, so val MAE is a real proxy for
    deployment accuracy."""
    model.eval()
    from collections import defaultdict

    pred_counts, gt_counts = [], []
    per_ds_errs = defaultdict(list)

    val_ds = val_subset.dataset  # underlying CrowdDataset
    indices = list(val_subset.indices)

    for i in tqdm(indices, desc="Evaluating"):
        img_tensor, points, tag = val_ds.get_raw_sample(i)
        img_tensor = img_tensor.unsqueeze(0).to(device, non_blocking=True)
        with autocast('cuda'):
            pred_count = _sliding_window_count(model, img_tensor, window_size, stride)
        gt_count = float(len(points))
        pred_counts.append(pred_count)
        gt_counts.append(gt_count)
        per_ds_errs[tag].append(abs(pred_count - gt_count))

    pred_counts = np.array(pred_counts)
    gt_counts = np.array(gt_counts)
    metrics = calculate_metrics(pred_counts, gt_counts)

    # Per-dataset MAE — so we can see whether drone improved without hurting others
    print("  Per-dataset MAE:")
    for tag in sorted(per_ds_errs.keys()):
        errs = np.array(per_ds_errs[tag])
        print(f"    {tag:<18}  n={len(errs):>4}  MAE={errs.mean():.2f}")
    metrics['per_dataset_mae'] = {k: float(np.mean(v)) for k, v in per_ds_errs.items()}
    return metrics




def main():
    # ================= CONFIGURATION =================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Each dir holds images/ + mats/ (see training/README in the repo root README).
    # site_data = proprietary mosque camera frames (not included); the rest are public:
    # drone_data = DroneCrowd + UP-COUNT, hacrowd_data = HAJJv2, *_converted = UCF-QNRF / JHU-Crowd++ / NWPU-Crowd.
    # Drop './site_data' to train on public data only.
    DATA_DIRS = ['./site_data', './drone_data', './hacrowd_data', './qnrf_converted', './jhu_converted', './nwpu_converted']
    PRETRAINED_WEIGHTS = 'best_mae_b16.pth'
    # v3 = v2 + wider bins (MAX_COUNT=10, 7 bins) for dense-scene coverage
    CHECKPOINT_DIR = './checkpoints_b16_all_datasets_v3_wider_bins'
    RESUME_CHECKPOINT = ''
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Native-scale training config (audit-driven; see compute_bins.py)
    SCALE_MIN, SCALE_MAX = 0.5, 3.0
    MAX_CROP_FRAC = 0.9         # cap crop at 90% of min(h,w) -> no upsample fallback
    MAX_COUNT = 10              # compute_bins.py: clips 0.04% of cells, 2x per-cell ceiling
    DRONE_FRACTION = 0.30       # target share of drone_data in each batch
    DRONE_TAG = 'drone_data'

    # Model configuration
    backbone_name = 'clip_vit_b_16'
    input_size = 224
    reduction = 8

    # LoRA configuration (set USE_LORA=True to inject LoRA into attention)
    USE_LORA = True
    lora_rank = 16

    # Training configuration
    batch_size = 32
    lr = 1e-4
    epochs = 150
    num_workers = 8
    warmup_epochs = 5
    
    # CLIP-EBC bins — data-driven from compute_bins.py (FULL dataset, 10,482
    # samples x 2 crops = 16,435,776 cells) under clamped RandomResizedCrop
    # [0.5, 3.0]. MAX_COUNT=10 clips 0.00265% of cells (436/16.4M). Anchors are
    # the mean of clipped cells in each bin range (CLIP-EBC "average" convention).
    bins = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 5), (6, 8), (9, float('inf'))]
    anchor_points = [0.0, 1.0, 2.0, 3.0, 4.2793, 6.5886, 9.7472]
    
    # VPT configuration to match pretrained weights
    clip_args = {
        'prompt_type': 'word',
        'num_vpt': 32,        # ← Match pretrained (32 prompts)
        'vpt_drop': 0.0,      # ← No dropout
        'deep_vpt': True      # ← Deep VPT (prompts at all layers)
    }
    
    print(f"\n{'='*70}")
    print(f"CLIP-EBC LoRA + VPT Fine-tuning for Mosque Crowd Counting")
    print(f"{'='*70}")
    print(f"Backbone:      {backbone_name}")
    print(f"LoRA:          {'ON (rank={})'.format(lora_rank) if USE_LORA else 'OFF'}")
    print(f"VPT Tokens:    {clip_args['num_vpt']} (Deep)")
    print(f"Bins:          {bins}")
    print(f"Anchor Points: {anchor_points}")
    print(f"Batch Size:    {batch_size}")
    print(f"Learning Rate: {lr}")
    print(f"Epochs:        {epochs}")
    print(f"Warmup:        {warmup_epochs} epochs (linear ramp 1e-6 -> {lr})")
    print(f"Loss:          DACE (CE + DMCount)")
    print(f"Normalization: ImageNet")
    print(f"{'='*70}\n")
    
    # =================================================
    # 1. Initialize Model
    print("Initializing CLIP-EBC model...")
    model = get_model(
        backbone=backbone_name,
        input_size=input_size,
        reduction=reduction,
        bins=bins,
        anchor_points=anchor_points,
        **clip_args
    )
    
    # 2. Load Pretrained Weights


    def resize_positional_embedding_for_clip_ebc(state_dict, model, device):
        key = "image_encoder.positional_embedding"
        if key not in state_dict:
            print(f"  {key} not found in checkpoint; skipping pos-emb resize.")
            return state_dict

        pos_old = state_dict[key]
        pos_new = model.image_encoder.positional_embedding

        if pos_old.shape == pos_new.shape:
            print("Positional embedding shapes already match; no resize needed.")
            return state_dict

        print(f"Resizing positional embedding: {pos_old.shape} -> {pos_new.shape}")

        num_tokens_old, dim = pos_old.shape
        num_tokens_new, _   = pos_new.shape

        cls_token_old = pos_old[:1, :]
        grid_old      = pos_old[1:, :]

        hw_old = num_tokens_old - 1
        h_old = w_old = int(hw_old ** 0.5)

        hw_new = num_tokens_new - 1
        h_new = w_new = int(hw_new ** 0.5)

        grid_old = grid_old.view(1, h_old, w_old, dim).permute(0, 3, 1, 2)
        grid_new = F.interpolate(grid_old, size=(h_new, w_new), mode="bilinear", align_corners=False)
        grid_new = grid_new.permute(0, 2, 3, 1).reshape(1, hw_new, dim)[0]

        pos_resized = torch.cat([cls_token_old, grid_new], dim=0)

        if pos_resized.shape != pos_new.shape:
            print(f"  Resized pos-emb shape {pos_resized.shape} != model {pos_new.shape}, using model default.")
            return state_dict

        state_dict[key] = pos_resized.to(device)
        print("  Positional embedding resized successfully.")
        return state_dict

    resume_epoch = 0
    resume_data = None
    lora_injected = False
    if RESUME_CHECKPOINT and os.path.exists(RESUME_CHECKPOINT):
        print(f"Resuming from checkpoint: {RESUME_CHECKPOINT}")
        resume_data = torch.load(RESUME_CHECKPOINT, map_location=device, weights_only=False)
        has_lora_keys = any('lora_' in k for k in resume_data['model_state_dict'].keys())

        if USE_LORA and has_lora_keys:
            # Resuming LoRA training — inject LoRA first so keys match
            from lora_utils import inject_lora_into_clip_vit, prepare_model_for_lora_training
            print(f"  LoRA checkpoint detected — injecting LoRA before loading weights...")
            model = inject_lora_into_clip_vit(model, rank=lora_rank)
            model.load_state_dict(resume_data['model_state_dict'])
            model = prepare_model_for_lora_training(model, device=str(device))
            lora_injected = True
            # New phase (decoder unfrozen) — reset epoch, fresh optimizer
            print(f"  Loaded LoRA weights (decoder phase — epoch reset to 0)")
            resume_data = None
        else:
            model.load_state_dict(resume_data['model_state_dict'])
            if USE_LORA:
                print(f"  Loaded VPT weights (new LoRA phase — epoch reset to 0)")
                resume_data = None
            else:
                resume_epoch = resume_data['epoch']
                print(f"  Resumed model at epoch {resume_epoch}")
    elif os.path.exists(PRETRAINED_WEIGHTS):
        print(f"Loading pre-trained weights from {PRETRAINED_WEIGHTS}...")
        checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        new_state_dict = resize_positional_embedding_for_clip_ebc(new_state_dict, model, device)

        model.load_state_dict(new_state_dict, strict=False)
        print("Weights loaded.")
    else:
        print(f"WARNING: '{PRETRAINED_WEIGHTS}' NOT found! Training from scratch.")

    # 3. Optional LoRA injection (skip if already injected during resume)
    if USE_LORA and not lora_injected:
        from lora_utils import inject_lora_into_clip_vit, prepare_model_for_lora_training
        print(f"\n  Injecting LoRA (rank={lora_rank})...")
        model = inject_lora_into_clip_vit(model, rank=lora_rank)
        model = prepare_model_for_lora_training(model, device=str(device))

    model = model.to(device)

    # Multi-GPU with DataParallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"\n  Using {num_gpus} GPUs with DataParallel")
        model = torch.nn.DataParallel(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable: {trainable_params_count:,} / {total_params:,} ({100*trainable_params_count/total_params:.2f}%)")
    
    # 4. Setup Training Components
    print("\n  Setting up training components...")
    
    # Loss function — DMCount (OT + TV + count)
    criterion = DACELoss(
        bins=bins,
        reduction=reduction,
        weight_count_loss=1.0,
        count_loss="dmcount",
        input_size=input_size
    ).to(device)
    print("   Loss: DACE (Cross-Entropy + DMCount)")

    # Optimizer
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    print(f"   Optimizer: AdamW (lr={lr}, weight_decay=1e-4)")

    # Learning rate scheduler — author's cosine annealing with warm restarts
    def cosine_annealing_warm_restarts(epoch, base_lr, warmup_epochs, warmup_lr, T_0, T_mult, eta_min):
        if epoch < warmup_epochs:
            lr_val = warmup_lr + (base_lr - warmup_lr) * epoch / warmup_epochs
        else:
            epoch_shifted = epoch - warmup_epochs
            if T_mult == 1:
                T_cur = epoch_shifted % T_0
                T_i = T_0
            else:
                n = int(math.log((epoch_shifted / T_0 * (T_mult - 1) + 1), T_mult))
                T_cur = epoch_shifted - T_0 * (T_mult ** n - 1) / (T_mult - 1)
                T_i = T_0 * T_mult ** n
            lr_val = eta_min + (base_lr - eta_min) * (1 + math.cos(math.pi * T_cur / T_i)) / 2
        return lr_val / base_lr

    scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=partial(
            cosine_annealing_warm_restarts,
            base_lr=lr,
            warmup_epochs=warmup_epochs,
            warmup_lr=1e-6,
            T_0=5,
            T_mult=2,
            eta_min=1e-7,
        ),
    )
    print(f"   Scheduler: CosineAnnealingWarmRestarts (warmup={warmup_epochs}, T_0=5, T_mult=2)")
    
    # Gradient scaler for mixed precision
    grad_scaler = GradScaler()

    # Restore optimizer/scheduler state if resuming
    if resume_data is not None:
        if 'optimizer_state_dict' in resume_data:
            optimizer.load_state_dict(resume_data['optimizer_state_dict'])
            print(f"   Optimizer state restored")
        if 'scheduler_state_dict' in resume_data:
            scheduler.load_state_dict(resume_data['scheduler_state_dict'])
            print(f"   Scheduler state restored")
    
    # 5. Setup Data Loaders
    print("\n  Loading datasets...")
    train_dataset, val_dataset = create_train_val_split(
        DATA_DIRS,
        val_fraction=0.05,
        seed=42,
        input_size=input_size,
        reduction=reduction,
        max_count=MAX_COUNT,
        num_crops=2,
        scale_min=SCALE_MIN,
        scale_max=SCALE_MAX,
        max_crop_frac=MAX_CROP_FRAC,
    )

    # Drone oversampling via WeightedRandomSampler.
    # Target drone share = DRONE_FRACTION of each batch. Each index gets weight
    #   w_drone = (p / (1-p)) * (n_other / n_drone)    (p = DRONE_FRACTION)
    # so that expected drone share under weighted sampling equals p exactly.
    # Train subset indices reference the underlying CrowdDataset.samples list,
    # whose 3rd field is the dataset tag.
    train_tags = [train_dataset.dataset.samples[i][2] for i in train_dataset.indices]
    n_drone = sum(1 for t in train_tags if t == DRONE_TAG)
    n_other = len(train_tags) - n_drone
    if n_drone == 0 or n_other == 0:
        print(f"   Warning: drone oversampling disabled (n_drone={n_drone}, n_other={n_other}).")
        sampler = None
        shuffle_train = True
    else:
        p = DRONE_FRACTION
        w_drone = (p / (1.0 - p)) * (n_other / n_drone)
        weights = torch.tensor(
            [w_drone if t == DRONE_TAG else 1.0 for t in train_tags],
            dtype=torch.double,
        )
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=weights, num_samples=len(train_tags), replacement=True
        )
        shuffle_train = False
        print(f"   Drone oversampling: n_drone={n_drone}, n_other={n_other}, "
              f"w_drone={w_drone:.3f} -> expected {p*100:.0f}% drone per batch")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=custom_collate,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0
    )
    print(f"   Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"   Validation: sliding-window at NATIVE resolution (window={input_size}, stride={input_size})")
    
    # Helper to get unwrapped model state dict (handles DataParallel)
    def get_state_dict():
        if hasattr(model, 'module'):
            return model.module.state_dict()
        return model.state_dict()

    # 6. Training Loop
    start_epoch = resume_epoch + 1
    print(f"\n  Starting training from epoch {start_epoch}...\n")
    best_mae = float('inf')
    if resume_data is not None and 'mae' in resume_data:
        best_mae = resume_data['mae']
        print(f"   Resuming with best MAE: {best_mae:.2f}")

    for epoch in range(start_epoch, epochs + 1):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'='*70}")
        
        # Train
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, grad_scaler, device, epoch, max_count=MAX_COUNT
        )
        
        # Print training metrics
        print(f"\n  Training Metrics:")
        print(f"   Loss:       {train_metrics['loss']:.4f}")
        print(f"   CE Loss:    {train_metrics['ce_loss']:.4f}")
        print(f"   Count Loss: {train_metrics['count_loss']:.4f}")
        
        # Update learning rate (every epoch, including warmup — author's pattern)
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"   Learning Rate: {current_lr:.6f}")
        
        # Evaluate frequently early (catch divergence), then every 5 epochs
        if val_dataset is not None and (epoch <= 10 or epoch % 5 == 0):
            val_metrics = evaluate(model, val_dataset, device)
            print(f"\n  Validation Metrics:")
            print(f"   MAE:  {val_metrics['mae']:.2f}")
            print(f"   RMSE: {val_metrics['rmse']:.2f}")
            
            # Save best model
            if val_metrics['mae'] < best_mae:
                best_mae = val_metrics['mae']
                save_path = os.path.join(CHECKPOINT_DIR, "best_model.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': get_state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'mae': best_mae,
                    'rmse': val_metrics['rmse'],
                    'bins': bins,
                    'anchor_points': anchor_points,
                    'config': {
                        'backbone': backbone_name,
                        'input_size': input_size,
                        'reduction': reduction,
                        'scale_min': SCALE_MIN,
                        'scale_max': SCALE_MAX,
                        'max_crop_frac': MAX_CROP_FRAC,
                        'max_count': MAX_COUNT,
                        'drone_fraction': DRONE_FRACTION,
                        'pipeline': 'v2_nativescale',
                    }
                }, save_path)
                print(f"   Best model saved! MAE: {best_mae:.2f}")
        
        # Save regular checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': get_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
            }, save_path)
            print(f"   Checkpoint saved: {save_path}")
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"   Best Validation MAE: {best_mae:.2f}")
    print(f"   Checkpoints saved in: {CHECKPOINT_DIR}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
