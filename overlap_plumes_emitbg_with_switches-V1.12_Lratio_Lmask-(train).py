# -*- coding: utf-8 -*-
#只用三个损失函数进行实验，搞太多了不好写啊
# USE_LOSS_RATIO
# USE_LOSS_REG
#USE_LOSS_MASK

# 这个是上面三个损失函数的消融对比试验，这个版本取消了# USE_LOSS_REG

"""
羽流信号区采用的是sum.alpha>0的方式获取，并经过一次膨胀操作。
别的都不动，保留各个损失函数和注意力机制的开关

Train: UNet3+ + 可开关注意力机制 + 可开关损失函数
+ Hungarian matching (cost = 1 - softIoU)

用途：
1. 同一份代码支持注意力机制消融：
   - USE_ECA
   - USE_SPATIAL

2. 同一份代码支持损失函数消融：
   - USE_LOSS_RATIO
   - USE_LOSS_MASK
   - USE_LOSS_REC
   - USE_LOSS_SSIM
   - USE_LOSS_REG
   - USE_LOSS_MSE
   - USE_LOSS_UNUSED_MASK
   - USE_LOSS_UNUSED_W

3. 自动根据当前开关生成实验输出目录，避免每次手动改代码覆盖结果

说明：
- 如果 PRETRAIN_CKPT 与当前模型结构不完全一致（比如注意力模块开关不同），
  仍可通过 strict=False 做部分加载，这适合做 fine-tune 初始化，
  但做论文解释时请说明是“部分兼容加载”。

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


# =========================
# Config
# =========================

# -------------------------
# 数据与输出路径
# -------------------------
# DATA_ROOT = r"E:\Dataset\plums_emitbg_combined\02_plumes_library\overlapping_plumes(2-8)-V2"
DATA_ROOT = r"E:\Dataset\plums_emitbg_combined\02_plumes_library\overlapping_plumes(5-8)_wind4to9_256_20k_V2（closer）"

OUT_ROOT  = r"E:\CH4-Plume-Separation\switch_runs(spatial_attention+Lratio_Lmask+closer)"

# 训练时读取普通 combined_plumes
COMBINED_DIRNAME = "combined_plumes"

# signal_mask 由 alpha_sum 在线生成
SIGNAL_MASK_SOURCE = "alpha_sum"

# sum(alpha) 的判定阈值
ALPHA_SIGNAL_SUM_EPS = 1e-3

# 保留一次 3x3 核膨胀
SIGNAL_MASK_DILATE = True
SIGNAL_MASK_DILATE_ITERS = 1

# -------------------------
# 预训练权重
# 修改用意：
# - 若想从头训练：设为 None
# - 若想在已有模型基础上继续训练：填写 best.pt 路径
# -------------------------
PRETRAIN_CKPT = r"E:\CH4-Plume-Separation\checkpoints_(random_points_2)\best.pt"
RESUME_OPTIMIZER = False   # 这里只保留接口，目前默认不恢复旧 optimizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
KMAX = 8

# ===== Early Stopping =====
EARLY_STOP_ENABLE = True
EARLY_STOP_PATIENCE = 10
EARLY_STOP_MIN_DELTA = 1e-4

# ===== GT mask from alpha =====
# 修改用意：
# alpha > ALPHA_MASK_THR 的区域视为对应源的有效掩码真值
ALPHA_MASK_THR = 0.05

# ===== Training =====
EPOCHS = 100
BATCH  = 4
LR     = 1e-5
NUM_WORKERS = 0

# =========================================================
# Ablation switches: Attention
# 修改用意：
# 用于注意力模块消融
# =========================================================
USE_ECA = False
# USE_SPATIAL = False   #既然Lssim的对比就包含了空间注意力，这里你咋又关了？你所有损失函数要对比的是包含空间注意力和所有损失函数的模型啊。不是没有任何注意力的模型，这样对比就乱套了啊
USE_SPATIAL = True

# =========================================================
# Ablation switches: Losses
# 修改用意：
# 这些开关控制某个损失项是否进入总损失
# total = sum( 1[USE_LOSS_i] * lambda_i * L_i )
# =========================================================
USE_LOSS_RATIO       = True
USE_LOSS_MASK        = True
USE_LOSS_REG         = False  #去掉该损失函数的对比实验

USE_LOSS_REC         = False
USE_LOSS_SSIM        = False
USE_LOSS_MSE         = False
USE_LOSS_UNUSED_MASK = False
USE_LOSS_UNUSED_W    = False  

# ===== Loss weights =====
LAMBDA_RATIO = 1.0
LAMBDA_MASK  = 1.0
LAMBDA_REG   = 1.0

LAMBDA_REC   = 2.0
LAMBDA_SSIM  = 1.0  #没啥用
LAMBDA_MSE   = 1.0

# ===== Unmatched slots regularization =====
LAMBDA_UNUSED_MASK = 0.3
LAMBDA_UNUSED_W    = 0.05


# =========================
# Experiment naming / logging
# =========================
def build_exp_name():
    """
    修改用意：
    自动把当前实验的注意力组合 + 损失组合写进文件夹名，
    避免每次手动改 OUT_CKPT，且方便回溯。
    """
    attn_name = f"eca{int(USE_ECA)}_spatial{int(USE_SPATIAL)}"

    loss_items = []
    if USE_LOSS_RATIO:
        loss_items.append("ratio")
    if USE_LOSS_MASK:
        loss_items.append("mask")
    if USE_LOSS_REC:
        loss_items.append("rec")
    if USE_LOSS_SSIM:
        loss_items.append("ssim")
    if USE_LOSS_REG:
        loss_items.append("tv")
    if USE_LOSS_MSE:
        loss_items.append("mse")
    if USE_LOSS_UNUSED_MASK:
        loss_items.append("unusedmask")
    if USE_LOSS_UNUSED_W:
        loss_items.append("unusedw")

    loss_name = "none" if len(loss_items) == 0 else "-".join(loss_items)
    return f"{attn_name}__loss_{loss_name}"


OUT_CKPT = os.path.join(OUT_ROOT, build_exp_name())


def get_experiment_config_dict():
    return {
        "DATA_ROOT": DATA_ROOT,
        "OUT_ROOT": OUT_ROOT,
        "OUT_CKPT": OUT_CKPT,
        "PRETRAIN_CKPT": PRETRAIN_CKPT,
        "DEVICE": DEVICE,
        "SEED": SEED,
        "KMAX": KMAX,

        "EARLY_STOP_ENABLE": EARLY_STOP_ENABLE,
        "EARLY_STOP_PATIENCE": EARLY_STOP_PATIENCE,
        "EARLY_STOP_MIN_DELTA": EARLY_STOP_MIN_DELTA,

        "ALPHA_MASK_THR": ALPHA_MASK_THR,

        "COMBINED_DIRNAME": COMBINED_DIRNAME,
        "SIGNAL_MASK_SOURCE": SIGNAL_MASK_SOURCE,

        "ALPHA_SIGNAL_SUM_EPS": ALPHA_SIGNAL_SUM_EPS,
        "SIGNAL_MASK_DILATE": SIGNAL_MASK_DILATE,
        "SIGNAL_MASK_DILATE_ITERS": SIGNAL_MASK_DILATE_ITERS,

        "EPOCHS": EPOCHS,
        "BATCH": BATCH,
        "LR": LR,
        "NUM_WORKERS": NUM_WORKERS,

        "USE_ECA": USE_ECA,
        "USE_SPATIAL": USE_SPATIAL,

        "USE_LOSS_RATIO": USE_LOSS_RATIO,
        "USE_LOSS_MASK": USE_LOSS_MASK,
        "USE_LOSS_REC": USE_LOSS_REC,
        "USE_LOSS_SSIM": USE_LOSS_SSIM,
        "USE_LOSS_REG": USE_LOSS_REG,
        "USE_LOSS_MSE": USE_LOSS_MSE,
        "USE_LOSS_UNUSED_MASK": USE_LOSS_UNUSED_MASK,
        "USE_LOSS_UNUSED_W": USE_LOSS_UNUSED_W,

        "LAMBDA_RATIO": LAMBDA_RATIO,
        "LAMBDA_MASK": LAMBDA_MASK,
        "LAMBDA_REC": LAMBDA_REC,
        "LAMBDA_SSIM": LAMBDA_SSIM,
        "LAMBDA_REG": LAMBDA_REG,
        "LAMBDA_MSE": LAMBDA_MSE,
        "LAMBDA_UNUSED_MASK": LAMBDA_UNUSED_MASK,
        "LAMBDA_UNUSED_W": LAMBDA_UNUSED_W,
    }


def print_experiment_config():
    cfg = get_experiment_config_dict()
    print("\n================ EXPERIMENT CONFIG ================")
    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("===================================================\n")

    if not any([
        USE_LOSS_RATIO, USE_LOSS_MASK, USE_LOSS_REC, USE_LOSS_SSIM,
        USE_LOSS_REG, USE_LOSS_MSE, USE_LOSS_UNUSED_MASK, USE_LOSS_UNUSED_W
    ]):
        raise ValueError("你把所有损失项都关掉了，total loss 恒为 0，训练没有意义。")


def save_experiment_config(out_dir):
    """
    修改用意：
    将当前实验的配置保存下来，便于后面看结果时知道当时到底开了哪些项
    """
    cfg = get_experiment_config_dict()
    path = os.path.join(out_dir, "exp_config.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"Experiment config saved to: {path}")


# =========================
# Utils
# =========================
def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_pairs(split_dir):
    """
    适配当前新数据目录，不再死依赖 sample_*.npz 命名
    修改用意：
    - 增加目录存在性检查，避免路径配错时静默返回空列表
    - 自动匹配 combined_plumes 与 labels 中同名文件
    """
    cdir = os.path.join(split_dir, COMBINED_DIRNAME)
    ldir = os.path.join(split_dir, "labels")

    if not os.path.isdir(cdir):
        raise FileNotFoundError(f"未找到 {COMBINED_DIRNAME} 目录: {cdir}")
    if not os.path.isdir(ldir):
        raise FileNotFoundError(f"未找到 labels 目录: {ldir}")

    xs = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    pairs = []
    for x in xs:
        base = os.path.basename(x)
        y = os.path.join(ldir, base)
        if os.path.exists(y):
            pairs.append((x, y))
    return pairs


def load_pretrained_model_only(model, ckpt_path, device):
    """
    只加载旧模型参数做 fine-tune 初始化：
    - 不恢复 optimizer
    - 兼容 ckpt["model"] / 直接 state_dict
    - 兼容 DataParallel 的 module. 前缀
    - strict=False 允许结构轻微变化（如注意力消融）
    """
    if (ckpt_path is None) or (not os.path.exists(ckpt_path)):
        print("No pretrained checkpoint loaded.")
        return model

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Loaded pretrained model from: {ckpt_path}")
    print(f"Missing keys: {missing}")
    print(f"Unexpected keys: {unexpected}")
    return model


def _dynamic_thr_np(single_hw: np.ndarray, q_percent: float, min_thr: float) -> float:
    nz = single_hw[single_hw > 0]
    if nz.size == 0:
        return float(min_thr)
    thr = float(np.percentile(nz, q_percent))
    if not np.isfinite(thr):
        thr = float(min_thr)
    return max(float(min_thr), thr)


def debug_print_thr_candidates(combined_t: torch.Tensor,
                               q_list=(0.5, 1.0, 2.0),
                               min_thr=1e-5,
                               fixed_thr=1e-5,
                               tag="train"):
    arr = combined_t.detach().cpu().numpy()[:, 0]  # (B,H,W)
    B, H, W = arr.shape
    total_pix = H * W

    fixed_ratios = [((arr[b] > fixed_thr).sum() / total_pix) for b in range(B)]
    print(f"\n[DYN_THR][{tag}] fixed_thr={fixed_thr:.3e}  signal_ratio="
          f"{np.mean(fixed_ratios)*100:.2f}% (min={np.min(fixed_ratios)*100:.2f}%, max={np.max(fixed_ratios)*100:.2f}%)")

    q_used = float(DYN_THR_Q)
    thrs_u, ratios_u = [], []
    for b in range(B):
        thr_u = _dynamic_thr_np(arr[b], q_percent=q_used, min_thr=min_thr)
        thrs_u.append(thr_u)
        ratios_u.append(((arr[b] > thr_u).sum() / total_pix))

    print(f"[DYN_THR][{tag}] USED q={q_used:>4.1f}%  thr≈{np.mean(thrs_u):.3e} "
          f"(min={np.min(thrs_u):.3e}, max={np.max(thrs_u):.3e})  "
          f"signal_ratio≈{np.mean(ratios_u)*100:.2f}% "
          f"(min={np.min(ratios_u)*100:.2f}%, max={np.max(ratios_u)*100:.2f}%)")

    for q in q_list:
        thrs, ratios = [], []
        for b in range(B):
            thr = _dynamic_thr_np(arr[b], q_percent=float(q), min_thr=min_thr)
            thrs.append(thr)
            ratios.append(((arr[b] > thr).sum() / total_pix))

        print(f"[DYN_THR][{tag}] q={float(q):>4.1f}%  thr≈{np.mean(thrs):.3e} "
              f"(min={np.min(thrs):.3e}, max={np.max(thrs):.3e})  "
              f"signal_ratio≈{np.mean(ratios)*100:.2f}% "
              f"(min={np.min(ratios)*100:.2f}%, max={np.max(ratios)*100:.2f}%)")

    print(f"[DYN_THR][{tag}] ==> using q={DYN_THR_Q}% , min_thr={min_thr:.3e}\n")


# =========================
# Dataset
# =========================
class PlumeDataset(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        x_path, y_path = self.pairs[idx]

        xnp = np.load(x_path)

        # 修改用意：
        # 兼容两种数据格式：
        # - 新数据（带 EMIT 背景）：combined
        # - 旧数据（无背景）：combined_clean
        if "combined" in xnp.files:
            combined = xnp["combined"].astype(np.float32)
        elif "combined_clean" in xnp.files:
            combined = xnp["combined_clean"].astype(np.float32)
        else:
            raise KeyError(f"{x_path} 中既没有 'combined' 也没有 'combined_clean'")

        combined_t = torch.from_numpy(combined)[None, ...]  # (1,H,W)

        # signal_mask 直接使用前处理脚本已经保存好的最终掩膜 rough_mask_paper
        # =====================================================
        # 构造训练时使用的 signal_mask
        # 作用：
        # 1) 使用真实标签 alpha_list 在线生成羽流信号区
        # 2) 对前 N 个真实羽流分量的 alpha 在通道维求和
        # 3) 将 sum(alpha) > 0.001 的区域定义为初始羽流信号区
        # 4) 再做一次 3x3 核膨胀，使边缘略微放宽
        # =====================================================
        if SIGNAL_MASK_SOURCE != "alpha_sum":
            raise ValueError(f"当前代码只允许 SIGNAL_MASK_SOURCE='alpha_sum'，收到: {SIGNAL_MASK_SOURCE}")

        ynp = np.load(y_path, allow_pickle=True)
        alpha = ynp["alpha_list"].astype(np.float32)  # (N,H,W)
        N = int(ynp["N"])

        H, W = combined.shape
        if alpha.ndim != 3:
            raise ValueError(f"alpha_list 不是三维数组: {y_path}, got shape={alpha.shape}")
        if alpha.shape[1] != H or alpha.shape[2] != W:
            raise ValueError(
                f"combined 与 alpha_list 空间尺寸不一致: {x_path} vs {y_path}, "
                f"combined={combined.shape}, alpha={alpha.shape}"
            )

        alpha_sum = alpha[:N].sum(axis=0)  # (H,W)
        signal_mask = (alpha_sum > ALPHA_SIGNAL_SUM_EPS)

        if SIGNAL_MASK_DILATE:
            from scipy.ndimage import binary_dilation
            signal_mask = binary_dilation(
                signal_mask,
                structure=np.ones((3, 3), dtype=bool),
                iterations=SIGNAL_MASK_DILATE_ITERS
            )

        signal_mask = signal_mask.astype(np.float32)
        signal_t = torch.from_numpy(signal_mask)

        # 占位值，仅为保持 dataloader 返回字段一致
        thr_t = torch.tensor(-999.0, dtype=torch.float32)


        H, W = combined.shape
        if alpha.ndim != 3:
            raise ValueError(f"alpha_list 不是三维数组: {y_path}, got shape={alpha.shape}")
        if alpha.shape[1] != H or alpha.shape[2] != W:
            raise ValueError(
                f"combined 与 alpha_list 空间尺寸不一致: {x_path} vs {y_path}, "
                f"combined={combined.shape}, alpha={alpha.shape}"
            )

        N_eff = min(N, KMAX)
        alpha_pad = np.zeros((KMAX, H, W), dtype=np.float32)
        alpha_pad[:N_eff] = alpha[:N_eff]
        alpha_t = torch.from_numpy(alpha_pad)  # (KMAX,H,W)

        # 修改用意：
        # mask_gt 只用于 mask 分支监督
        # 比例极小的 alpha 区域不作为 mask 正样本
        mask_gt = (alpha_pad > ALPHA_MASK_THR).astype(np.float32)
        mask_gt_t = torch.from_numpy(mask_gt)

        return {
            "combined": combined_t,  # (1,H,W)
            "alpha": alpha_t,        # (KMAX,H,W)
            "mask_gt": mask_gt_t,    # (KMAX,H,W)
            "signal": signal_t,      # (H,W)
            "thr": thr_t,            # ()
            "N": N_eff
        }


# =========================
# Attention: ECA + Spatial
# =========================
class ECALayer(nn.Module):
    """
    Efficient Channel Attention (ECA)
    """
    def __init__(self, channels, gamma=2, b=1):
        super().__init__()
        t = int(abs((np.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        k = max(3, k)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1d = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)                   # (B,C,1,1)
        y = y.squeeze(-1).transpose(-1, -2)   # (B,1,C)
        y = self.conv1d(y)                    # (B,1,C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B,C,1,1)
        return x * y


class SpatialAttention(nn.Module):
    """
    CBAM-like spatial attention
    """
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
    """
    修改用意：
    对 skip 分支的注意力机制做成可开关：
    - use_eca=True  -> 使用通道注意力
    - use_spatial=True -> 使用空间注意力
    两者都关掉则为恒等映射
    """
    def __init__(self, ch, spatial_ks=7, use_eca=True, use_spatial=True):
        super().__init__()
        self.use_eca = use_eca
        self.use_spatial = use_spatial

        if self.use_eca:
            self.eca = ECALayer(ch)
        if self.use_spatial:
            self.sa = SpatialAttention(kernel_size=spatial_ks)

    def forward(self, x):
        if self.use_eca:
            x = self.eca(x)
        if self.use_spatial:
            x = self.sa(x)
        return x


# =========================
# UNet3+ blocks
# =========================
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def _resize_to(x, size_hw):
    """
    size_hw: (H,W) of target
    - bigger -> adaptive maxpool down
    - smaller -> bilinear up
    - same -> passthrough
    """
    H, W = x.shape[-2], x.shape[-1]
    th, tw = size_hw
    if H == th and W == tw:
        return x
    if H > th or W > tw:
        return F.adaptive_max_pool2d(x, output_size=(th, tw))
    return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)


class FullScaleFusion(nn.Module):
    """
    UNet3+ 的 full-scale fusion:
    - 每个输入分支先投影到 cat_ch
    - resize 到 target size
    - 分支上加可选 SkipAttn (ECA / Spatial)
    - concat 后再 conv 融合成 out_ch
    """
    def __init__(self, in_channels_list, cat_ch, out_ch,
                 spatial_ks=7, use_eca=True, use_spatial=True):
        super().__init__()
        self.proj = nn.ModuleList([
            ConvBNReLU(cin, cat_ch, k=3, p=1) for cin in in_channels_list
        ])
        self.attn = nn.ModuleList([
            SkipAttn(cat_ch, spatial_ks=spatial_ks,
                     use_eca=use_eca, use_spatial=use_spatial)
            for _ in in_channels_list
        ])
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
    """
    Encoder: 4 downs + bottleneck (e1..e5)
    Decoder:
      d4: [e1,e2,e3,e4,e5]
      d3: [e1,e2,e3,e4,e5,d4]
      d2: [e1,e2,e3,e4,e5,d3]
      d1: [e1,e2,e3,e4,e5,d2]
    """
    def __init__(self, in_ch=1, base=32, kmax=8, cat_ch=None,
                 use_eca=True, use_spatial=True):
        super().__init__()
        self.kmax = kmax
        if cat_ch is None:
            cat_ch = base

        c1, c2, c3, c4, c5 = base, base*2, base*4, base*8, base*16

        self.enc1 = DoubleConv(in_ch, c1)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(c1, c2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(c2, c3)
        self.pool3 = nn.MaxPool2d(2)
        self.enc4 = DoubleConv(c3, c4)
        self.pool4 = nn.MaxPool2d(2)
        self.enc5 = DoubleConv(c4, c5)

        d4_ch = base*8
        d3_ch = base*4
        d2_ch = base*2
        d1_ch = base

        # 修改用意：
        # 这里把注意力开关一路传入 full-scale fusion
        self.fuse_d4 = FullScaleFusion(
            in_channels_list=[c1, c2, c3, c4, c5],
            cat_ch=cat_ch, out_ch=d4_ch,
            use_eca=use_eca, use_spatial=use_spatial
        )
        self.fuse_d3 = FullScaleFusion(
            in_channels_list=[c1, c2, c3, c4, c5, d4_ch],
            cat_ch=cat_ch, out_ch=d3_ch,
            use_eca=use_eca, use_spatial=use_spatial
        )
        self.fuse_d2 = FullScaleFusion(
            in_channels_list=[c1, c2, c3, c4, c5, d3_ch],
            cat_ch=cat_ch, out_ch=d2_ch,
            use_eca=use_eca, use_spatial=use_spatial
        )
        self.fuse_d1 = FullScaleFusion(
            in_channels_list=[c1, c2, c3, c4, c5, d2_ch],
            cat_ch=cat_ch, out_ch=d1_ch,
            use_eca=use_eca, use_spatial=use_spatial
        )

        self.ratio_head = nn.Conv2d(d1_ch, kmax, 1)
        self.mask_head  = nn.Conv2d(d1_ch, kmax, 1)

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

        ratio_logits = self.ratio_head(d1)  # (B,K,H,W)
        mask_logits  = self.mask_head(d1)   # (B,K,H,W)

        # 修改用意：
        # ratio 分支输出经过 softplus 保证非负，再做逐像素归一化
        w_pos = F.softplus(ratio_logits)
        w = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-8) #公式（6）

        return w, mask_logits


# =========================
# SSIM (simple differentiable)
# =========================
def _gaussian_kernel(window_size=11, sigma=1.5, device="cpu"):
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    kernel1d = g[None, :]
    kernel2d = kernel1d.t() @ kernel1d
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d


def ssim_loss(x, y, window_size=11, sigma=1.5, C1=1e-4, C2=9e-4):
    device = x.device
    k = _gaussian_kernel(window_size, sigma, device=device).view(1, 1, window_size, window_size)

    mu_x = F.conv2d(x, k, padding=window_size//2)
    mu_y = F.conv2d(y, k, padding=window_size//2)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x*x, k, padding=window_size//2) - mu_x2
    sigma_y2 = F.conv2d(y*y, k, padding=window_size//2) - mu_y2
    sigma_xy = F.conv2d(x*y, k, padding=window_size//2) - mu_xy

    ssim_map = ((2*mu_xy + C1) * (2*sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2) + 1e-8
    )
    return 1.0 - ssim_map.mean(dim=[1, 2, 3])  # (B,)


# =========================
# TV regularization
# =========================
def tv_loss(w):
    dh = torch.abs(w[:, :, 1:, :] - w[:, :, :-1, :]).mean()
    dw = torch.abs(w[:, :, :, 1:] - w[:, :, :, :-1]).mean()
    return dh + dw


# =========================
# soft-IoU + Hungarian
# =========================
def soft_iou_np(a, b, mask=None, eps=1e-8):
    if mask is not None:
        a = a[mask]
        b = b[mask]
        if a.size == 0:
            return 0.0
    inter = np.sum(a * b)
    union = np.sum(a + b - a * b)
    return float(inter / (union + eps))


def hungarian_match_softiou(w_b, alpha_b, signal_b, N):
    """
    修改用意：
    对 K 个预测 slot 与 N 个真值源做 Hungarian matching
    cost = 1 - softIoU
    """
    w_np = w_b.detach().cpu().numpy().astype(np.float32)      # (K,H,W)
    a_np = alpha_b.detach().cpu().numpy().astype(np.float32)  # (K,H,W)
    sig = (signal_b.detach().cpu().numpy() > 0.5)             # (H,W) bool

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


# =========================
# Loss wrapper
# =========================
def _mean_or_zero(loss_list, device):
    if len(loss_list) == 0:
        return torch.tensor(0.0, device=device)
    return torch.stack(loss_list).mean()


class LossFn(nn.Module):
    """
    修改用意：
    将所有损失项做成可开关形式。

    总损失形式：
    L_total = Σ 1[USE_i] * λ_i * L_i

    注意：
    - 关闭某个损失项，表示“移除该项监督”
    - 不代表删除对应网络分支
    """
    def __init__(self,
                 use_ratio=True,
                 use_mask=True,
                 use_rec=True,
                 use_ssim=True,
                 use_reg=True,
                 use_mse=True,
                 use_unused_mask=True,
                 use_unused_w=True):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")
        self.mse = nn.MSELoss(reduction="none")

        self.use_ratio = use_ratio
        self.use_mask = use_mask
        self.use_rec = use_rec
        self.use_ssim = use_ssim
        self.use_reg = use_reg
        self.use_mse = use_mse
        self.use_unused_mask = use_unused_mask
        self.use_unused_w = use_unused_w

    def forward(self, combined, w, mask_logits, alpha_gt, mask_gt, N_list, signal_mask):
        B, K, H, W = w.shape
        device = combined.device

        # 修改用意：
        # sig 是“信号区域掩码”，用于限制 ratio/rec/mse 等损失只在信号区计算
        sig = (signal_mask > 0.5).float()  # (B,H,W)

        # 修改用意：
        # gate 不必是 0/1，它是可微分的软门控
        gate = torch.sigmoid(mask_logits)   # (B,K,H,W)
        comp_pred = w * gate * combined     # (B,K,H,W)
        comp_gt   = alpha_gt * combined     # (B,K,H,W)

        # -------------------------
        # Reconstruction loss
        # 修改用意：
        # 用于约束所有 slot 重构之和逼近 combined
        # -------------------------
        if self.use_rec:
            recon = comp_pred.sum(dim=1)    # (B,H,W)
            target = combined[:, 0]         # (B,H,W)
            res = torch.abs(recon - target)
            L_rec = (res * sig).sum() / (sig.sum() + 1e-8)
        else:
            L_rec = torch.tensor(0.0, device=device)

        ratio_losses = []
        mask_losses = []
        ssim_losses = []
        mse_losses = []
        unused_mask_losses = []
        unused_w_losses = []

        need_matching = any([
            self.use_ratio,
            self.use_mask,
            self.use_ssim,
            self.use_mse,
            self.use_unused_mask,
            self.use_unused_w
        ])

        # -------------------------
        # Hungarian matching
        # 修改用意：
        # 只有依赖 slot 对应关系的损失才需要 matching
        # -------------------------
        if need_matching:
            for b in range(B):
                N = int(N_list[b])
                pairs = hungarian_match_softiou(w[b], alpha_gt[b], sig[b], N)
                matched_pred = set([pk for pk, _ in pairs])
                zero = torch.zeros((H, W), device=device, dtype=combined.dtype)

                # unmatched slot regularization
                if self.use_unused_mask or self.use_unused_w:
                    for k in range(K):
                        if k in matched_pred:
                            continue

                        if self.use_unused_mask:
                            unused_mask_losses.append(self.bce(mask_logits[b, k], zero))

                        if self.use_unused_w and sig[b].sum() > 0:
                            unused_w_losses.append(w[b, k][sig[b] > 0].mean())

                # matched losses
                for pk, gj in pairs:
                    if self.use_ratio:
                        diff = F.smooth_l1_loss(
                            w[b, pk], alpha_gt[b, gj], reduction="none"
                        )
                        ratio_losses.append((diff * sig[b]).sum() / (sig[b].sum() + 1e-8))

                    if self.use_mask:
                        mask_losses.append(self.bce(mask_logits[b, pk], mask_gt[b, gj]))

                    if self.use_ssim:
                        cp = comp_pred[b, pk].unsqueeze(0).unsqueeze(0)
                        ct = comp_gt[b, gj].unsqueeze(0).unsqueeze(0)
                        ssim_losses.append(ssim_loss(cp, ct).mean())

                    if self.use_mse:
                        mse_map = self.mse(comp_pred[b, pk], comp_gt[b, gj])
                        mse_losses.append((mse_map * sig[b]).sum() / (sig[b].sum() + 1e-8))

        L_ratio = _mean_or_zero(ratio_losses, device) if self.use_ratio else torch.tensor(0.0, device=device)
        L_mask  = _mean_or_zero(mask_losses, device) if self.use_mask else torch.tensor(0.0, device=device)
        L_ssim  = _mean_or_zero(ssim_losses, device) if self.use_ssim else torch.tensor(0.0, device=device)
        L_mse   = _mean_or_zero(mse_losses, device) if self.use_mse else torch.tensor(0.0, device=device)
        L_unused_mask = _mean_or_zero(unused_mask_losses, device) if self.use_unused_mask else torch.tensor(0.0, device=device)
        L_unused_w    = _mean_or_zero(unused_w_losses, device) if self.use_unused_w else torch.tensor(0.0, device=device)
        L_reg = tv_loss(w) if self.use_reg else torch.tensor(0.0, device=device)

        # -------------------------
        # 总损失：带开关控制
        # 修改用意：
        # total = Σ 1[USE_i] * λ_i * L_i
        # -------------------------
        total = torch.tensor(0.0, device=device)

        if self.use_ratio:
            total = total + LAMBDA_RATIO * L_ratio
        if self.use_mask:
            total = total + LAMBDA_MASK * L_mask
        if self.use_rec:
            total = total + LAMBDA_REC * L_rec
        if self.use_ssim:
            total = total + LAMBDA_SSIM * L_ssim
        if self.use_mse:
            total = total + LAMBDA_MSE * L_mse
        if self.use_reg:
            total = total + LAMBDA_REG * L_reg
        if self.use_unused_mask:
            total = total + LAMBDA_UNUSED_MASK * L_unused_mask
        if self.use_unused_w:
            total = total + LAMBDA_UNUSED_W * L_unused_w

        info = {
            "L_total": total.detach().item(),
            "L_ratio": L_ratio.detach().item(),
            "L_mask":  L_mask.detach().item(),
            "L_rec":   L_rec.detach().item(),
            "L_ssim":  L_ssim.detach().item(),
            "L_reg":   L_reg.detach().item(),
            "L_unused_mask": L_unused_mask.detach().item(),
            "L_unused_w":    L_unused_w.detach().item(),
            "L_mse": L_mse.detach().item(),
        }
        return total, info


# =========================
# Train / Val
# =========================
def run_epoch(model, loader, optimizer, loss_fn, train=True, epoch=1):
    model.train(train)
    pbar = tqdm(loader, ncols=150)
    losses = []

    for step, batch in enumerate(pbar):
        combined = batch["combined"].to(DEVICE)     # (B,1,H,W)
        alpha_gt = batch["alpha"].to(DEVICE)        # (B,K,H,W)
        mask_gt  = batch["mask_gt"].to(DEVICE)      # (B,K,H,W)
        N_list   = batch["N"]
        signal   = batch["signal"].to(DEVICE)       # (B,H,W)
        thr_batch = batch["thr"].to(DEVICE)         # (B,)



        w, mask_logits = model(combined)
        loss, info = loss_fn(combined, w, mask_logits, alpha_gt, mask_gt, N_list, signal)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        losses.append(info["L_total"])

        sig_ratio = float(signal.mean().detach().cpu().item())

        pbar.set_description(
            f"{'train' if train else 'val'} "
            f"tot={info['L_total']:.4f} "
            f"ratio={info['L_ratio']:.4f} "
            f"mask={info['L_mask']:.4f} "
            f"rec={info['L_rec']:.4f} "
            f"ssim={info['L_ssim']:.4f} "
            f"mse={info['L_mse']:.4f} "
            f"tv={info['L_reg']:.4f} "
            f"um={info['L_unused_mask']:.3f} "
            f"uw={info['L_unused_w']:.3f} "
            f"| sig={sig_ratio * 100:.1f}% masksrc=alpha_sum(sum>{ALPHA_SIGNAL_SUM_EPS:.1e},dil{SIGNAL_MASK_DILATE_ITERS})"
        )

    return float(np.mean(losses))


# =========================
# Main
# =========================
def main():
    seed_all(SEED)
    os.makedirs(OUT_CKPT, exist_ok=True)

    print_experiment_config()
    save_experiment_config(OUT_CKPT)

    train_pairs = list_pairs(os.path.join(DATA_ROOT, "training_dataset"))
    val_pairs   = list_pairs(os.path.join(DATA_ROOT, "validation_dataset"))

    print(f"Train samples: {len(train_pairs)}")
    print(f"Val samples:   {len(val_pairs)}")

    train_ds = PlumeDataset(train_pairs)
    val_ds   = PlumeDataset(val_pairs)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=False
    )

    # 修改用意：
    # 将注意力开关传入模型
    model = UNet3PlusAttn(
        in_ch=1,
        base=32,
        kmax=KMAX,
        cat_ch=32,
        use_eca=USE_ECA,
        use_spatial=USE_SPATIAL
    ).to(DEVICE)

    model = load_pretrained_model_only(model, PRETRAIN_CKPT, DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    # 修改用意：
    # 将损失开关传入 LossFn
    loss_fn = LossFn(
        use_ratio=USE_LOSS_RATIO,
        use_mask=USE_LOSS_MASK,
        use_rec=USE_LOSS_REC,
        use_ssim=USE_LOSS_SSIM,
        use_reg=USE_LOSS_REG,
        use_mse=USE_LOSS_MSE,
        use_unused_mask=USE_LOSS_UNUSED_MASK,
        use_unused_w=USE_LOSS_UNUSED_W
    ).to(DEVICE)

    best_val = 1e9
    no_improve = 0

    for epoch in range(1, EPOCHS + 1):
        print(f"\n===== Epoch {epoch}/{EPOCHS} =====")
        tr = run_epoch(model, train_loader, optimizer, loss_fn, train=True,  epoch=epoch)
        va = run_epoch(model, val_loader,   optimizer, loss_fn, train=False, epoch=epoch)

        ckpt_path = os.path.join(OUT_CKPT, f"epoch_{epoch:03d}.pt")
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "val_loss": va
        }, ckpt_path)

        improved = (va < best_val - EARLY_STOP_MIN_DELTA)
        if improved:
            best_val = va
            no_improve = 0
            best_path = os.path.join(OUT_CKPT, "best.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "val_loss": va
            }, best_path)
            print(f"✅ Best updated: val={va:.6f} -> {best_path}")
        else:
            no_improve += 1
            print(f"⏳ No improvement: {no_improve}/{EARLY_STOP_PATIENCE} (best={best_val:.6f})")

        print(f"Epoch done: train={tr:.6f}, val={va:.6f}")

        if EARLY_STOP_ENABLE and no_improve >= EARLY_STOP_PATIENCE:
            print(f"🛑 Early stop triggered at epoch {epoch}. Best val={best_val:.6f}")
            break

    print("\nTraining finished.")
    print("Best ckpt:", os.path.join(OUT_CKPT, "best.pt"))


if __name__ == "__main__":
    main()