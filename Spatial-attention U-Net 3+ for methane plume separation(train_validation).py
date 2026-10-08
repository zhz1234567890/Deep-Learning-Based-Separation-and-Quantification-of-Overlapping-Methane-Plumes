"""
Build a U-Net 3+ model with spatial attention for methane plume source
separation. The script includes model training and validation procedures
for model optimization and checkpoint selection.

Input: Combined methane enhancement maps and per‑source contribution labels (`training_dataset`, `validation_dataset`)
Output: Best checkpoint
"""

import os
import glob
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import binary_dilation

# Root directory for training and validation datasets
DATA_ROOT = r"...\overlapping_plumes(2-8)-V2"

OUT_ROOT = '<path_to_output_directory>'

COMBINED_DIRNAME = 'combined_plumes'
ALPHA_SIGNAL_SUM_EPS = 0.001
SIGNAL_MASK_DILATE_ITERS = 1
PRETRAIN_CKPT = None
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 42
KMAX = 8
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 0.0001
ALPHA_MASK_THR = 0.05
EPOCHS = 100
BATCH = 4
LR = 1e-05
NUM_WORKERS = 0
LAMBDA_RATIO = 1.0
LAMBDA_MASK = 1.0
OUT_CKPT = os.path.join(OUT_ROOT, 'your_output_path')


def get_experiment_config_dict():
    """Return the training configuration."""
    return {
        'DATA_ROOT': DATA_ROOT,
        'OUT_ROOT': OUT_ROOT,
        'OUT_CKPT': OUT_CKPT,
        'PRETRAIN_CKPT': PRETRAIN_CKPT,
        'DEVICE': DEVICE,
        'SEED': SEED,
        'KMAX': KMAX,
        'EARLY_STOP_PATIENCE': EARLY_STOP_PATIENCE,
        'EARLY_STOP_MIN_DELTA': EARLY_STOP_MIN_DELTA,
        'ALPHA_MASK_THR': ALPHA_MASK_THR,
        'COMBINED_DIRNAME': COMBINED_DIRNAME,
        'ALPHA_SIGNAL_SUM_EPS': ALPHA_SIGNAL_SUM_EPS,
        'SIGNAL_MASK_DILATE_ITERS': SIGNAL_MASK_DILATE_ITERS,
        'EPOCHS': EPOCHS,
        'BATCH': BATCH,
        'LR': LR,
        'NUM_WORKERS': NUM_WORKERS,
        'LAMBDA_RATIO': LAMBDA_RATIO,
        'LAMBDA_MASK': LAMBDA_MASK,
    }


def print_experiment_config():
    """Print the training configuration."""
    cfg = get_experiment_config_dict()
    print('\n================ EXPERIMENT CONFIG ================')
    for k, v in cfg.items():
        print(f'{k}: {v}')
    print('===================================================\n')


def save_experiment_config(out_dir):
    """Save the training configuration as JSON."""
    cfg = get_experiment_config_dict()
    path = os.path.join(out_dir, 'exp_config.json')
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f'Experiment config saved to: {path}')


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pairs(split_dir):
    """Match input and label files by filename."""
    cdir = os.path.join(split_dir, COMBINED_DIRNAME)
    ldir = os.path.join(split_dir, 'labels')
    if not os.path.isdir(cdir):
        raise FileNotFoundError(f'Missing {COMBINED_DIRNAME} directory: {cdir}')
    if not os.path.isdir(ldir):
        raise FileNotFoundError(f'Missing labels directory: {ldir}')
    xs = sorted(glob.glob(os.path.join(cdir, '*.npz')))
    pairs = []
    for x in xs:
        base = os.path.basename(x)
        y = os.path.join(ldir, base)
        if os.path.exists(y):
            pairs.append((x, y))
    return pairs


def load_pretrained_model_only(model, ckpt_path, device):
    """Load model weights without restoring the optimizer."""
    if ckpt_path is None or not os.path.exists(ckpt_path):
        print('No pretrained checkpoint loaded.')
        return model
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    if any((k.startswith('module.') for k in state.keys())):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f'Loaded pretrained model from: {ckpt_path}')
    print(f'Missing keys: {missing}')
    print(f'Unexpected keys: {unexpected}')
    return model


class PlumeDataset(Dataset):
    """Load plume maps and construct training targets."""

    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]
        xnp = np.load(x_path)
        if 'combined' in xnp.files:
            combined = xnp['combined'].astype(np.float32)
        elif 'combined_clean' in xnp.files:
            combined = xnp['combined_clean'].astype(np.float32)
        else:
            raise KeyError(f"{x_path} contains neither 'combined' nor 'combined_clean'")
        combined_t = torch.from_numpy(combined)[None, ...]
        ynp = np.load(y_path, allow_pickle=True)
        alpha = ynp['alpha_list'].astype(np.float32)
        N = int(ynp['N'])
        H, W = combined.shape
        if alpha.ndim != 3:
            raise ValueError(f'alpha_list must be three-dimensional: {y_path}, got shape={alpha.shape}')
        if alpha.shape[1] != H or alpha.shape[2] != W:
            raise ValueError(f'Spatial dimensions of combined and alpha_list do not match: {x_path} vs {y_path}, combined={combined.shape}, alpha={alpha.shape}')
        alpha_sum = alpha[:N].sum(axis=0)
        signal_mask = alpha_sum > ALPHA_SIGNAL_SUM_EPS
        signal_mask = binary_dilation(signal_mask, structure=np.ones((3, 3), dtype=bool), iterations=SIGNAL_MASK_DILATE_ITERS)
        signal_mask = signal_mask.astype(np.float32)
        signal_t = torch.from_numpy(signal_mask)
        N_eff = min(N, KMAX)
        alpha_pad = np.zeros((KMAX, H, W), dtype=np.float32)
        alpha_pad[:N_eff] = alpha[:N_eff]
        alpha_t = torch.from_numpy(alpha_pad)
        mask_gt = (alpha_pad > ALPHA_MASK_THR).astype(np.float32)
        mask_gt_t = torch.from_numpy(mask_gt)
        return {
            'combined': combined_t,
            'alpha': alpha_t,
            'mask_gt': mask_gt_t,
            'signal': signal_t,
            'N': N_eff,
        }


class SpatialAttention(nn.Module):
    """Apply spatial attention using channel mean and maximum maps."""

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        a = torch.cat([avg, mx], dim=1)
        att = torch.sigmoid(self.conv(a))
        return x * att


class SkipAttn(nn.Module):
    """Apply spatial attention to a skip branch."""

    def __init__(self, ch, spatial_ks=7):
        super().__init__()
        self.sa = SpatialAttention(kernel_size=spatial_ks)

    def forward(self, x):
        return self.sa(x)


class ConvBNReLU(nn.Module):
    """Apply convolution, batch normalization, and ReLU."""

    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    """Apply two convolution blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(ConvBNReLU(in_ch, out_ch, 3, 1), ConvBNReLU(out_ch, out_ch, 3, 1))

    def forward(self, x):
        return self.net(x)


def _resize_to(x, size_hw):
    """Resize features using max pooling or bilinear interpolation."""
    H, W = (x.shape[-2], x.shape[-1])
    th, tw = size_hw
    if H == th and W == tw:
        return x
    if H > th or W > tw:
        return F.adaptive_max_pool2d(x, output_size=(th, tw))
    return F.interpolate(x, size=(th, tw), mode='bilinear', align_corners=False)


class FullScaleFusion(nn.Module):
    """Fuse multiscale features with spatial attention."""

    def __init__(self, in_channels_list, cat_ch, out_ch, spatial_ks=7):
        super().__init__()
        self.proj = nn.ModuleList([ConvBNReLU(cin, cat_ch, k=3, p=1) for cin in in_channels_list])
        self.attn = nn.ModuleList([SkipAttn(cat_ch, spatial_ks=spatial_ks) for _ in in_channels_list])
        self.fuse = DoubleConv(cat_ch * len(in_channels_list), out_ch)

    def forward(self, feats_list, target_hw):
        outs = []
        for x, proj, att in zip(feats_list, self.proj, self.attn):
            x = proj(x)
            x = _resize_to(x, target_hw)
            x = att(x)
            outs.append(x)
        x = torch.cat(outs, dim=1)
        x = self.fuse(x)
        return x


class UNet3PlusAttn(nn.Module):
    """Predict source contribution weights and mask logits with U-Net 3+."""

    def __init__(self, in_ch=1, base=32, kmax=8, cat_ch=None):
        super().__init__()
        self.kmax = kmax
        if cat_ch is None:
            cat_ch = base
        c1, c2, c3, c4, c5 = (base, base * 2, base * 4, base * 8, base * 16)
        self.enc1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = DoubleConv(c4, c5)
        d4_ch = base * 8
        d3_ch = base * 4
        d2_ch = base * 2
        d1_ch = base
        self.fuse_d4 = FullScaleFusion(in_channels_list=[c1, c2, c3, c4, c5], cat_ch=cat_ch, out_ch=d4_ch)
        self.fuse_d3 = FullScaleFusion(in_channels_list=[c1, c2, c3, c4, c5, d4_ch], cat_ch=cat_ch, out_ch=d3_ch)
        self.fuse_d2 = FullScaleFusion(in_channels_list=[c1, c2, c3, c4, c5, d3_ch], cat_ch=cat_ch, out_ch=d2_ch)
        self.fuse_d1 = FullScaleFusion(in_channels_list=[c1, c2, c3, c4, c5, d2_ch], cat_ch=cat_ch, out_ch=d1_ch)
        self.ratio_head = nn.Conv2d(d1_ch, kmax, 1)
        self.mask_head = nn.Conv2d(d1_ch, kmax, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        e5 = self.enc5(self.pool4(e4))
        d4 = self.fuse_d4([e1, e2, e3, e4, e5], target_hw=(e4.shape[-2], e4.shape[-1]))
        d3 = self.fuse_d3([e1, e2, e3, e4, e5, d4], target_hw=(e3.shape[-2], e3.shape[-1]))
        d2 = self.fuse_d2([e1, e2, e3, e4, e5, d3], target_hw=(e2.shape[-2], e2.shape[-1]))
        d1 = self.fuse_d1([e1, e2, e3, e4, e5, d2], target_hw=(e1.shape[-2], e1.shape[-1]))
        ratio_logits = self.ratio_head(d1)
        mask_logits = self.mask_head(d1)
        w_pos = F.softplus(ratio_logits)
        w = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-08)
        return (w, mask_logits)


def soft_iou_np(a, b, mask=None, eps=1e-08):
    """Compute soft IoU within an optional signal mask."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
        if a.size == 0:
            return 0.0
    inter = np.sum(a * b)
    union = np.sum(a + b - a * b)
    return float(inter / (union + eps))


def hungarian_match_softiou(w_b, alpha_b, signal_b, N):
    """Match predicted slots to sources using soft IoU."""
    w_np = w_b.detach().cpu().numpy().astype(np.float32)
    a_np = alpha_b.detach().cpu().numpy().astype(np.float32)
    sig = signal_b.detach().cpu().numpy() > 0.5
    w_np = np.clip(w_np, 0.0, None)
    a_np = np.clip(a_np, 0.0, None)
    a_np = a_np[:N]
    K = w_np.shape[0]
    cost = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        wk = w_np[k]
        for j in range(N):
            aj = a_np[j]
            iou = soft_iou_np(wk, aj, mask=sig)
            cost[k, j] = 1.0 - iou
    r, c = linear_sum_assignment(cost)
    pairs = [(int(rr), int(cc)) for rr, cc in zip(r.tolist(), c.tolist()) if cc < N]
    pairs.sort(key=lambda x: x[1])
    return pairs[:N]


def _mean_or_zero(loss_list, device):
    """Average losses or return zero for an empty list."""
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_list).mean()


class LossFn(nn.Module):
    """Compute matched contribution and mask losses."""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, combined, w, mask_logits, alpha_gt, mask_gt, N_list, signal_mask):
        device = combined.device
        sig = (signal_mask > 0.5).float()
        ratio_losses = []
        mask_losses = []
        for b in range(w.shape[0]):
            N = int(N_list[b])
            pairs = hungarian_match_softiou(w[b], alpha_gt[b], sig[b], N)
            for pk, gj in pairs:
                diff = F.smooth_l1_loss(w[b, pk], alpha_gt[b, gj], reduction='none')
                ratio_losses.append((diff * sig[b]).sum() / (sig[b].sum() + 1e-08))
                mask_losses.append(self.bce(mask_logits[b, pk], mask_gt[b, gj]))
        L_ratio = _mean_or_zero(ratio_losses, device)
        L_mask = _mean_or_zero(mask_losses, device)
        total = torch.tensor(0.0, device=device)
        total = total + LAMBDA_RATIO * L_ratio
        total = total + LAMBDA_MASK * L_mask
        info = {
            'L_total': total.detach().item(),
            'L_ratio': L_ratio.detach().item(),
            'L_mask': L_mask.detach().item(),
        }
        return (total, info)


def run_epoch(model, loader, optimizer, loss_fn, train=True):
    """Run one training or validation epoch."""
    model.train(train)
    pbar = tqdm(loader, ncols=150)
    losses = []
    for batch in pbar:
        combined = batch['combined'].to(DEVICE)
        alpha_gt = batch['alpha'].to(DEVICE)
        mask_gt = batch['mask_gt'].to(DEVICE)
        N_list = batch['N']
        signal = batch['signal'].to(DEVICE)
        w, mask_logits = model(combined)
        loss, info = loss_fn(combined, w, mask_logits, alpha_gt, mask_gt, N_list, signal)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        losses.append(info['L_total'])
        sig_ratio = float(signal.mean().detach().cpu().item())
        pbar.set_description(
            f"{'train' if train else 'val'} "
            f"tot={info['L_total']:.4f} "
            f"ratio={info['L_ratio']:.4f} "
            f"mask={info['L_mask']:.4f} "
            f"| sig={sig_ratio * 100:.1f}%"
        )

    return float(np.mean(losses))


def main():
    """Train the model and save configurations and checkpoints."""
    seed_all(SEED)
    os.makedirs(OUT_CKPT, exist_ok=True)
    print_experiment_config()
    save_experiment_config(OUT_CKPT)
    train_pairs = list_pairs(os.path.join(DATA_ROOT, 'training_dataset'))
    val_pairs = list_pairs(os.path.join(DATA_ROOT, 'validation_dataset'))
    print(f'Train samples: {len(train_pairs)}')
    print(f'Val samples:   {len(val_pairs)}')
    train_ds = PlumeDataset(train_pairs)
    val_ds = PlumeDataset(val_pairs)
    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    model = UNet3PlusAttn(in_ch=1, base=32, kmax=KMAX, cat_ch=32).to(DEVICE)
    model = load_pretrained_model_only(model, PRETRAIN_CKPT, DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0001)
    loss_fn = LossFn().to(DEVICE)
    best_val = 1000000000.0
    no_improve = 0
    for epoch in range(1, EPOCHS + 1):
        print(f'\n===== Epoch {epoch}/{EPOCHS} =====')
        tr = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        va = run_epoch(model, val_loader, optimizer, loss_fn, train=False)
        ckpt_path = os.path.join(OUT_CKPT, f'epoch_{epoch:03d}.pt')
        torch.save({
            'epoch': epoch,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': va,
        }, ckpt_path)
        improved = va < best_val - EARLY_STOP_MIN_DELTA
        if improved:
            best_val = va
            no_improve = 0
            best_path = os.path.join(OUT_CKPT, 'best.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'val_loss': va,
            }, best_path)
            print(f' Best updated: val={va:.6f} -> {best_path}')
        else:
            no_improve += 1
            print(f'No improvement: {no_improve}/{EARLY_STOP_PATIENCE} (best={best_val:.6f})')
        print(f'Epoch done: train={tr:.6f}, val={va:.6f}')
        if no_improve >= EARLY_STOP_PATIENCE:
            print(f'Early stop triggered at epoch {epoch}. Best val={best_val:.6f}')
            break
    print('\nTraining finished.')
    print('Best ckpt:', os.path.join(OUT_CKPT, 'best.pt'))


if __name__ == '__main__':
    main()
