"""
Test a trained plume separation model using a labeled test set and its best
model checkpoint.

Inputs:best model checkpoint and a labeled test set(test_dataset)
Outputs: comparison figures for source counts 2-8 in ALL_GRIDS, data for
later IME-based source emission-rate estimation in RESULT_BUNDLES_FOR_IME,
and metrics_all.csv, evaluation_report.txt, and evaluation_report.xlsx.
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from scipy.ndimage import binary_dilation, label, binary_fill_holes
from scipy.optimize import linear_sum_assignment
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Root directory of the dataset. This is the parent folder of training and validation sets,
# NOT the path pointing directly to the training/validation dataset.
# Example: If training set path is E:\Dataset\plums_emitbg_combined\02_plumes_library\overlapping_plumes(2‑8)‑V2\training_dataset
# Set this value to: E:\Dataset\plums_emitbg_combined\02_plumes_library\overlapping_plumes(2‑8)‑V2
DATA_ROOT = r"<path_to_dataset_root_containing_test_dataset>"

CKPT_PATH = r"<path_to_trained_best_model/best.pt>"
OUT_DIR = r"<path_to_test_output_directory>"


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
KMAX = 8

COMBINED_THR = 1e-5


COMBINED_DIRNAME = "combined_plumes"


ALPHA_SIGNAL_SUM_EPS = 1e-3


SIGNAL_MASK_DILATE_ITERS = 1

CMAP = "plasma"


ALL_GRIDS_DIRNAME = "ALL_GRIDS"


RESULT_BUNDLES_DIRNAME = "RESULT_BUNDLES_FOR_IME"
RESULT_BUNDLE_SUFFIX = "__bundle.npz"


VMAX_Q = 99.8
DPI_GRID = 140


SOURCE_COORD_SYSTEM = "plot_xy"

ABS_THR = 1e-5
REL_THR = 0.01

MASK_DILATE_ITERS = 1

MIN_COMPONENT_PIXELS = 9
MAX_COMPONENT_GAP_PIXELS = 10.0
MAX_SOURCE_DISTANCE_PIXELS = 100.0


def seed_all(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def list_npz(split_dir):
    """Match test inputs and labels by filename."""
    cdir = os.path.join(split_dir, COMBINED_DIRNAME)
    ldir = os.path.join(split_dir, "labels")

    if not os.path.isdir(cdir):
        raise FileNotFoundError(f"Missing {COMBINED_DIRNAME} directory: {cdir}")
    if not os.path.isdir(ldir):
        raise FileNotFoundError(f"Missing labels directory: {ldir}")

    xs = sorted(glob.glob(os.path.join(cdir, "*.npz")))
    pairs = []
    for x in xs:
        base = os.path.basename(x)
        y = os.path.join(ldir, base)
        if os.path.exists(y):
            sid = os.path.splitext(base)[0]
            pairs.append((x, y, sid))
    return pairs


def src_to_id_str(src_name: str) -> str:
    """Convert a source name to a display ID."""
    s = str(src_name).replace("800_", "")
    parts = s.split("_")
    if len(parts) == 1:
        return parts[0]
    return "".join(parts)


def build_alpha_signal_mask(alpha, N, eps=1e-3, do_dilate=True, dil_iters=1):
    """Build the signal mask from source contribution labels."""
    alpha_sum = alpha[:N].sum(axis=0)
    signal_mask = (alpha_sum > eps)

    if do_dilate:
        signal_mask = binary_dilation(
            signal_mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=dil_iters
        )

    return signal_mask.astype(np.uint8)


def extract_markers_and_ids_from_ynp(ynp, N):
    """Extract source coordinates and IDs from labels."""
    markers = None
    ids_for_cols = None
    source_xy_save = None
    source_id_save = None

    if ("src_xy_big_plot" in ynp.files) and ("src_id" in ynp.files):
        mxy = ynp["src_xy_big_plot"].astype(np.float32)
        mid = [str(x) for x in ynp["src_id"].tolist()]
        n_use = min(N, mxy.shape[0], len(mid))
        markers = [(float(mxy[i, 0]), float(mxy[i, 1]), mid[i]) for i in range(n_use)]
        ids_for_cols = (mid + ["unk"] * N)[:N]
        source_xy_save = mxy[:n_use].astype(np.float32)
        source_id_save = np.asarray(mid[:n_use], dtype=object)

    elif ("src_xy_small_plot" in ynp.files) and ("src_id" in ynp.files):
        mxy = ynp["src_xy_small_plot"].astype(np.float32)
        mid = [str(x) for x in ynp["src_id"].tolist()]
        n_use = min(N, mxy.shape[0], len(mid))
        markers = [(float(mxy[i, 0]), float(mxy[i, 1]), mid[i]) for i in range(n_use)]
        ids_for_cols = (mid + ["unk"] * N)[:N]
        source_xy_save = mxy[:n_use].astype(np.float32)
        source_id_save = np.asarray(mid[:n_use], dtype=object)

    elif ("src_xy" in ynp.files) and ("src_id" in ynp.files):
        mxy = ynp["src_xy"].astype(np.float32)
        mid = [str(x) for x in ynp["src_id"].tolist()]
        n_use = min(N, mxy.shape[0], len(mid))
        markers = [(float(mxy[i, 0]), float(mxy[i, 1]), mid[i]) for i in range(n_use)]
        ids_for_cols = (mid + ["unk"] * N)[:N]
        source_xy_save = mxy[:n_use].astype(np.float32)
        source_id_save = np.asarray(mid[:n_use], dtype=object)

    elif ("markers_xy" in ynp.files) and ("markers_id" in ynp.files):
        mxy = ynp["markers_xy"].astype(np.float32)
        mid = [str(x) for x in ynp["markers_id"].tolist()]
        n_use = min(N, mxy.shape[0], len(mid))
        markers = [(float(mxy[i, 0]), float(mxy[i, 1]), mid[i]) for i in range(n_use)]
        ids_for_cols = (mid + ["unk"] * N)[:N]
        source_xy_save = mxy[:n_use].astype(np.float32)
        source_id_save = np.asarray(mid[:n_use], dtype=object)

    else:
        if "sources" in ynp.files:
            sources = [str(s) for s in ynp["sources"].tolist()]
            ids_for_cols = [src_to_id_str(s) for s in sources[:N]]
            source_id_save = np.asarray(ids_for_cols, dtype=object)
        else:
            ids_for_cols = [str(i + 1) for i in range(N)]
            source_id_save = np.asarray(ids_for_cols, dtype=object)

    return markers, ids_for_cols, source_xy_save, source_id_save


def save_result_bundle_npz(bundle_path, sid, x_path, y_path,
                           xnp, ynp,
                           combined, alpha, N,
                           signal_mask,
                           gt_components, pred_components,
                           w, gate,
                           pairs_h, pk_for_j,
                           ids_for_cols,
                           markers=None,
                           source_xy_save=None,
                           source_id_save=None):
    """Save inputs, matched predictions, and metadata for later IME estimation."""
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)

    save_dict = {}

    for k in xnp.files:
        save_dict[k] = xnp[k]

    for k in ynp.files:
        if k in save_dict:
            save_dict[f"label__{k}"] = ynp[k]
        else:
            save_dict[k] = ynp[k]

    pred_components_arr = np.stack(pred_components, axis=0).astype(np.float32)
    gt_components_arr = np.asarray(gt_components, dtype=np.float32)
    signal_mask_arr = np.asarray(signal_mask).astype(np.uint8)
    pairs_h_arr = np.asarray(pairs_h, dtype=np.int32).reshape(-1, 2)
    pk_for_j_arr = np.asarray(pk_for_j, dtype=np.int32)

    save_dict.update({
        "sample_id": np.asarray(str(sid)),
        "orig_x_path": np.asarray(str(x_path)),
        "orig_y_path": np.asarray(str(y_path)),
        "N_eval": np.asarray(int(N), dtype=np.int32),

        "combined_eval": np.asarray(combined, dtype=np.float32),
        "alpha_list_eval": np.asarray(alpha, dtype=np.float32),
        "signal_mask": signal_mask_arr,
        "gt_components": gt_components_arr,
        "pred_components": pred_components_arr,

        "w_pred": np.asarray(w, dtype=np.float32),
        "gate_pred": np.asarray(gate, dtype=np.float32),

        "match_pairs_pred_gt": pairs_h_arr,
        "matched_pred_slot_for_gt": pk_for_j_arr,

        "ids_for_cols": np.asarray(ids_for_cols, dtype=object),
    })

    if source_xy_save is not None:
        save_dict["source_xy_for_ime"] = np.asarray(source_xy_save, dtype=np.float32)

    if source_id_save is not None:
        save_dict["source_id_for_ime"] = np.asarray(source_id_save, dtype=object)

    if markers is not None and len(markers) > 0:
        save_dict["markers_xy_eval"] = np.asarray([[m[0], m[1]] for m in markers], dtype=np.float32)
        save_dict["markers_id_eval"] = np.asarray([str(m[2]) for m in markers], dtype=object)

    np.savez_compressed(bundle_path, **save_dict)


class SpatialAttention(nn.Module):
    """Apply spatial attention to feature maps."""
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
    def __init__(self, spatial_ks=7):
        super().__init__()

        self.sa = SpatialAttention(kernel_size=spatial_ks)

    def forward(self, x):
        x = self.sa(x)
        return x


class ConvBNReLU(nn.Module):
    """Apply convolution, batch normalization, and ReLU."""
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
    """Apply two convolution blocks."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            ConvBNReLU(in_ch, out_ch, 3, 1),
            ConvBNReLU(out_ch, out_ch, 3, 1),
        )

    def forward(self, x):
        return self.net(x)


def _resize_to(x, size_hw):
    """Resize features by pooling or bilinear interpolation."""
    H, W = x.shape[-2], x.shape[-1]
    th, tw = size_hw
    if H == th and W == tw:
        return x
    if H > th or W > tw:
        return F.adaptive_max_pool2d(x, output_size=(th, tw))
    return F.interpolate(x, size=(th, tw), mode="bilinear", align_corners=False)


class FullScaleFusion(nn.Module):
    """Fuse multiscale features with spatial attention."""
    def __init__(self, in_channels_list, cat_ch, out_ch,
                 spatial_ks=7):
        super().__init__()
        self.proj = nn.ModuleList([
            ConvBNReLU(cin, cat_ch, k=3, p=1) for cin in in_channels_list
        ])
        self.attn = nn.ModuleList([
            SkipAttn(spatial_ks=spatial_ks)
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
    """Predict source contribution weights and mask logits."""
    def __init__(self, in_ch=1, base=32, kmax=8, cat_ch=None):
        super().__init__()
        self.kmax = kmax
        if cat_ch is None:
            cat_ch = base

        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16

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

        self.fuse_d4 = FullScaleFusion(
            [c1, c2, c3, c4, c5],
            cat_ch=cat_ch, out_ch=d4_ch,

        )
        self.fuse_d3 = FullScaleFusion(
            [c1, c2, c3, c4, c5, d4_ch],
            cat_ch=cat_ch, out_ch=d3_ch,

        )
        self.fuse_d2 = FullScaleFusion(
            [c1, c2, c3, c4, c5, d3_ch],
            cat_ch=cat_ch, out_ch=d2_ch,

        )
        self.fuse_d1 = FullScaleFusion(
            [c1, c2, c3, c4, c5, d2_ch],
            cat_ch=cat_ch, out_ch=d1_ch,

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

        ratio_logits = self.ratio_head(d1)
        mask_logits  = self.mask_head(d1)

        w_pos = F.softplus(ratio_logits)
        w = w_pos / (w_pos.sum(dim=1, keepdim=True) + 1e-8)
        return w, mask_logits


def load_model_from_ckpt(ckpt_path, device, kmax=8, base=32, cat_ch=32):
    """Load the trained model for evaluation."""
    model = UNet3PlusAttn(
        in_ch=1,
        base=base,
        kmax=kmax,
        cat_ch=cat_ch,

    ).to(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and ("model" in ckpt) else ckpt

    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def soft_iou(a, b, mask=None, eps=1e-8):
    """Compute soft IoU within an optional mask."""
    if mask is not None:
        a = a[mask]
        b = b[mask]
        if a.size == 0:
            return 0.0
    inter = np.sum(a * b)
    union = np.sum(a + b - a * b)
    return float(inter / (union + eps))


def hungarian_match_softiou(w_pred, alpha_gt, N, signal_mask=None):
    """Match predicted slots to labeled sources using soft IoU."""
    K = w_pred.shape[0]

    if signal_mask is None:
        raise ValueError("hungarian_match_softiou requires an explicit signal_mask")

    signal_mask = np.asarray(signal_mask).astype(bool)

    w_use = np.clip(w_pred, 0.0, None)
    a_use = np.clip(alpha_gt, 0.0, None)


    cost = np.zeros((K, N), dtype=np.float32)
    for k in range(K):
        wk = w_use[k]
        for j in range(N):
            aj = a_use[j]
            iou = soft_iou(wk, aj, mask=signal_mask)
            cost[k, j] = 1.0 - iou

    r, c = linear_sum_assignment(cost)
    pairs = []
    for rr, cc in zip(r.tolist(), c.tolist()):
        if cc < N:
            pairs.append((rr, cc))
    pairs.sort(key=lambda x: x[1])
    return pairs[:N]


def source_xy_to_rowcol(x, y, H, coord_system="plot_xy"):
    """Convert source coordinates to array row and column coordinates."""
    x = float(x)
    y = float(y)

    if coord_system == "plot_xy":
        col = x
        row = (H - 1) - y
    elif coord_system == "array_xy":
        col = x
        row = y
    else:
        raise ValueError(f"Unknown coordinate system: {coord_system}")

    return float(row), float(col)


def rowcol_inside_image(row, col, H, W):
    """Check whether coordinates lie within the image."""
    return (0.0 <= row <= H - 1) and (0.0 <= col <= W - 1)


def build_initial_component_mask(component_kgm2, abs_thr, rel_thr, total_mask=None):
    """Threshold a source component within an optional total mask."""
    comp = np.asarray(component_kgm2, dtype=np.float32)
    comp = np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0)
    comp = np.clip(comp, 0.0, None)

    peak = float(np.max(comp))
    if peak <= 0:
        return np.zeros_like(comp, dtype=np.uint8), 0.0

    thr = max(float(abs_thr), float(rel_thr) * peak)
    mask0 = comp > thr

    if total_mask is not None:
        mask0 = np.logical_and(mask0, total_mask > 0)

    return mask0.astype(np.uint8), float(thr)


def _component_min_dist2_to_source(rr, cc, source_row, source_col):
    """Compute the minimum squared distance to a source."""
    if rr.size == 0:
        return np.inf
    return float(np.min((rr - source_row) ** 2 + (cc - source_col) ** 2))


def _component_min_gap_pixels(rr1, cc1, rr2, cc2):
    """Compute the minimum distance between connected components."""
    if rr1.size == 0 or rr2.size == 0:
        return np.inf

    pts1 = np.stack([rr1, cc1], axis=1).astype(np.float32)
    pts2 = np.stack([rr2, cc2], axis=1).astype(np.float32)

    best = np.inf
    chunk = 512
    for s in range(0, len(pts1), chunk):
        a = pts1[s:s + chunk]
        d2 = np.sum((a[:, None, :] - pts2[None, :, :]) ** 2, axis=2)
        cur = float(np.sqrt(np.min(d2)))
        if cur < best:
            best = cur
            if best <= 0:
                return 0.0
    return best


def choose_connected_components_for_one_source(mask0,
                                               source_row,
                                               source_col,
                                               min_component_pixels=6,
                                               max_gap_pixels=8.0,
                                               max_source_distance_pixels=80.0,
                                               prefer_source_containing_component=True):
    """Select the main source component and nearby components."""
    structure = np.ones((3, 3), dtype=np.uint8)
    lab, num = label(mask0.astype(np.uint8), structure=structure)

    if num == 0:
        return np.zeros_like(mask0, dtype=np.uint8)

    comps = []
    for lab_id in range(1, num + 1):
        rr, cc = np.where(lab == lab_id)
        if rr.size == 0:
            continue

        area = int(rr.size)
        d2_src = _component_min_dist2_to_source(rr, cc, source_row, source_col)

        contains_source = False
        sr = int(round(source_row))
        sc = int(round(source_col))
        if 0 <= sr < lab.shape[0] and 0 <= sc < lab.shape[1]:
            contains_source = (lab[sr, sc] == lab_id)

        comps.append({
            "lab_id": lab_id,
            "rr": rr,
            "cc": cc,
            "area": area,
            "d2_src": d2_src,
            "contains_source": contains_source,
        })

    if len(comps) == 0:
        return np.zeros_like(mask0, dtype=np.uint8)

    valid_comps = [c for c in comps if c["area"] >= int(min_component_pixels)]
    if len(valid_comps) == 0:
        valid_comps = comps

    main_comp = None

    if prefer_source_containing_component:
        source_hit = [c for c in valid_comps if c["contains_source"]]
        if len(source_hit) > 0:
            source_hit = sorted(source_hit, key=lambda c: (-c["area"], c["d2_src"]))
            main_comp = source_hit[0]

    if main_comp is None:
        valid_comps = sorted(valid_comps, key=lambda c: (c["d2_src"], -c["area"]))
        main_comp = valid_comps[0]

    final_mask = (lab == main_comp["lab_id"])
    main_rr = main_comp["rr"]
    main_cc = main_comp["cc"]

    for comp in comps:
        if comp["lab_id"] == main_comp["lab_id"]:
            continue

        if comp["area"] < int(min_component_pixels):
            continue

        src_dist = float(np.sqrt(comp["d2_src"]))
        if src_dist > float(max_source_distance_pixels):
            continue

        gap = _component_min_gap_pixels(main_rr, main_cc, comp["rr"], comp["cc"])
        if gap <= float(max_gap_pixels):
            final_mask = np.logical_or(final_mask, lab == comp["lab_id"])

    return final_mask.astype(np.uint8)


def postprocess_mask(mask_in, fill_holes=True, dilate_iters=0):
    """Fill mask holes and apply dilation."""
    mask = (mask_in > 0)

    if fill_holes:
        mask = binary_fill_holes(mask)

    if dilate_iters > 0:
        mask = binary_dilation(
            mask,
            structure=np.ones((3, 3), dtype=bool),
            iterations=int(dilate_iters)
        )

    return mask.astype(np.uint8)


def get_single_source_mask_from_component(component_kgm2,
                                          source_xy,
                                          source_coord_system="plot_xy",
                                          total_mask=None,
                                          abs_thr=1e-5,
                                          rel_thr=0.01,
                                          fill_holes=True,
                                          dilate_iters=0):
    """Extract a plume mask for one source component."""
    comp = np.asarray(component_kgm2, dtype=np.float32)
    H, W = comp.shape

    x, y = float(source_xy[0]), float(source_xy[1])
    source_row, source_col = source_xy_to_rowcol(
        x=x, y=y, H=H, coord_system=source_coord_system
    )

    if not rowcol_inside_image(source_row, source_col, H, W):
        mask = np.zeros((H, W), dtype=np.uint8)
        info = {
            "source_x": x,
            "source_y": y,
            "source_row": float(source_row),
            "source_col": float(source_col),
            "mask_pixels": 0,
            "thr_used": np.nan,
            "note": "source_outside"
        }
        return mask, info

    mask0, thr_used = build_initial_component_mask(
        component_kgm2=comp,
        abs_thr=abs_thr,
        rel_thr=rel_thr,
        total_mask=total_mask
    )

    mask_cc = choose_connected_components_for_one_source(
        mask0=mask0,
        source_row=source_row,
        source_col=source_col,
        min_component_pixels=MIN_COMPONENT_PIXELS,
        max_gap_pixels=MAX_COMPONENT_GAP_PIXELS,
        max_source_distance_pixels=MAX_SOURCE_DISTANCE_PIXELS,
        prefer_source_containing_component=True
    )

    mask_final = postprocess_mask(
        mask_in=mask_cc,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )

    info = {
        "source_x": x,
        "source_y": y,
        "source_row": float(source_row),
        "source_col": float(source_col),
        "mask_pixels": int(mask_final.sum()),
        "thr_used": float(thr_used),
        "note": "ok" if int(mask_final.sum()) > 0 else "mask_empty"
    }
    return mask_final.astype(np.uint8), info


def prf_iou_dice_from_masks(pred_mask: np.ndarray, gt_mask: np.ndarray):
    """Compute precision, recall, F1, IoU, and Dice from binary masks."""
    pred_m = np.asarray(pred_mask).astype(bool)
    gt_m = np.asarray(gt_mask).astype(bool)

    tp = np.logical_and(pred_m, gt_m).sum()
    fp = np.logical_and(pred_m, np.logical_not(gt_m)).sum()
    fn = np.logical_and(np.logical_not(pred_m), gt_m).sum()

    precision = tp / (tp + fp + 1e-12)
    recall    = tp / (tp + fn + 1e-12)
    f1        = 2 * precision * recall / (precision + recall + 1e-12)
    iou       = tp / (tp + fp + fn + 1e-12)
    dice      = 2 * tp / (2 * tp + fp + fn + 1e-12)

    return float(f1), float(precision), float(recall), float(iou), float(dice)


def evaluate_one_component_by_masks(pred_c, gt_c, source_xy, total_mask=None):
    """Evaluate a matched prediction using extracted plume masks."""
    pred_mask, pred_info = get_single_source_mask_from_component(
        component_kgm2=pred_c,
        source_xy=source_xy,
        source_coord_system=SOURCE_COORD_SYSTEM,
        total_mask=total_mask,
        abs_thr=ABS_THR,
        rel_thr=REL_THR,
        fill_holes=True,
        dilate_iters=MASK_DILATE_ITERS
    )

    gt_mask, gt_info = get_single_source_mask_from_component(
        component_kgm2=gt_c,
        source_xy=source_xy,
        source_coord_system=SOURCE_COORD_SYSTEM,
        total_mask=total_mask,
        abs_thr=ABS_THR,
        rel_thr=REL_THR,
        fill_holes=True,
        dilate_iters=MASK_DILATE_ITERS
    )

    f1, prec, rec, iou, dice = prf_iou_dice_from_masks(pred_mask, gt_mask)

    union_mask = np.logical_or(pred_mask > 0, gt_mask > 0).astype(np.uint8)

    return {
        "pred_thr": float(pred_info["thr_used"]) if np.isfinite(pred_info["thr_used"]) else np.nan,
        "gt_thr": float(gt_info["thr_used"]) if np.isfinite(gt_info["thr_used"]) else np.nan,
        "pred_mask_pixels": int(pred_info["mask_pixels"]),
        "gt_mask_pixels": int(gt_info["mask_pixels"]),
        "union_mask_pixels": int(union_mask.sum()),
        "F1": f1,
        "Precision": prec,
        "Recall": rec,
        "mIoU": iou,
        "Dice": dice,
    }


def append_rows_for_one_component_mask_based(all_rows, N, sid, src_id, pred_c, gt_c,
                                             source_xy, total_mask=None):
    """Append metrics for one matched source component."""
    rr = evaluate_one_component_by_masks(
        pred_c=pred_c,
        gt_c=gt_c,
        source_xy=source_xy,
        total_mask=total_mask
    )

    all_rows.append({
        "N": int(N),
        "sample": str(sid),
        "source": str(src_id),

        "pred_thr": rr["pred_thr"],
        "gt_thr": rr["gt_thr"],
        "pred_mask_pixels": rr["pred_mask_pixels"],
        "gt_mask_pixels": rr["gt_mask_pixels"],
        "union_mask_pixels": rr["union_mask_pixels"],

        "F1": rr["F1"],
        "Precision": rr["Precision"],
        "Recall": rr["Recall"],
        "mIoU": rr["mIoU"],
        "Dice": rr["Dice"],
    })


def save_reports_by_N_mask_based(df: pd.DataFrame, out_dir: str):
    """Save detailed metrics and summaries by source count."""
    os.makedirs(out_dir, exist_ok=True)

    need_cols = [
        "N", "sample", "source",
        "pred_thr", "gt_thr",
        "pred_mask_pixels", "gt_mask_pixels", "union_mask_pixels",
        "F1", "Precision", "Recall", "mIoU", "Dice"
    ]
    for c in need_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required DataFrame column: {c}")

    N_ORDER = list(range(2, 9))

    df = df.copy()
    df["N"] = df["N"].astype(int)
    df = df.sort_values(["N", "sample", "source"], ascending=True)


    csv_path = os.path.join(out_dir, "metrics_all.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")


    metric_cols = ["F1", "Precision", "Recall", "mIoU", "Dice"]
    aux_cols = ["pred_thr", "gt_thr", "pred_mask_pixels", "gt_mask_pixels", "union_mask_pixels"]

    g = df.groupby(["N"], sort=False)

    summary_metrics = g[metric_cols].agg(["mean", "std"])
    summary_metrics.columns = [f"{a}_{b}" for a, b in summary_metrics.columns]

    summary_aux = g[aux_cols].agg(["mean", "std", "min", "max"])
    summary_aux.columns = [f"{a}_{b}" for a, b in summary_aux.columns]

    summary = pd.concat([summary_metrics, summary_aux], axis=1)
    summary["count_pairs"] = g.size()
    summary = summary.reset_index()

    summary = summary.set_index("N").reindex(N_ORDER).reset_index()


    txt_path = os.path.join(out_dir, "evaluation_report.txt")

    def _f3(x):
        if x is None:
            return "nan"
        try:
            x = float(x)
        except Exception:
            return "nan"
        if np.isnan(x) or np.isinf(x):
            return "nan"
        return f"{x:.3f}"

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== Summary by source count N (IME-style plume masks) ===\n")
        headers = [
            ("N", 4), ("Pairs", 8),
            ("PredThrMean/1e-3", 18),
            ("GTThrMean/1e-3", 16),
            ("PredPixMean", 14),
            ("GTPixMean", 12),
            ("UnionPixMean", 14),
            ("F1Mean", 10), ("F1Std", 10),
            ("PrecMean", 12), ("PrecStd", 12),
            ("RecallMean", 12), ("RecallStd", 12),
            ("mIoUMean", 10), ("mIoUStd", 10),
            ("DiceMean", 10), ("DiceStd", 10),
        ]
        f.write("".join([f"{h:<{w}}" for h, w in headers]) + "\n")
        f.write("-" * sum(w for _, w in headers) + "\n")

        for _, r in summary.iterrows():
            n = int(r["N"])
            cnt = int(r["count_pairs"]) if pd.notna(r["count_pairs"]) else 0

            pred_thr_mean_x1e3 = float(r["pred_thr_mean"]) * 1000.0 if pd.notna(r["pred_thr_mean"]) else np.nan
            gt_thr_mean_x1e3   = float(r["gt_thr_mean"]) * 1000.0 if pd.notna(r["gt_thr_mean"]) else np.nan

            f.write(
                f"{n:<4}{cnt:<8}"
                f"{_f3(pred_thr_mean_x1e3):<18}"
                f"{_f3(gt_thr_mean_x1e3):<16}"
                f"{_f3(r.get('pred_mask_pixels_mean')):<14}"
                f"{_f3(r.get('gt_mask_pixels_mean')):<12}"
                f"{_f3(r.get('union_mask_pixels_mean')):<14}"
                f"{_f3(r.get('F1_mean')):<10}{_f3(r.get('F1_std')):<10}"
                f"{_f3(r.get('Precision_mean')):<12}{_f3(r.get('Precision_std')):<12}"
                f"{_f3(r.get('Recall_mean')):<12}{_f3(r.get('Recall_std')):<12}"
                f"{_f3(r.get('mIoU_mean')):<10}{_f3(r.get('mIoU_std')):<10}"
                f"{_f3(r.get('Dice_mean')):<10}{_f3(r.get('Dice_std')):<10}\n"
            )


    xlsx_path = os.path.join(out_dir, "evaluation_report.xlsx")

    df_xlsx = df.copy()
    if "pred_thr" in df_xlsx.columns:
        df_xlsx["pred_thr_x1e3"] = pd.to_numeric(df_xlsx["pred_thr"], errors="coerce") * 1000.0
    if "gt_thr" in df_xlsx.columns:
        df_xlsx["gt_thr_x1e3"] = pd.to_numeric(df_xlsx["gt_thr"], errors="coerce") * 1000.0

    for c in ["pred_thr_x1e3", "gt_thr_x1e3", "F1", "Precision", "Recall", "mIoU", "Dice"]:
        if c in df_xlsx.columns:
            df_xlsx[c] = pd.to_numeric(df_xlsx[c], errors="coerce").round(3)

    summary_xlsx = summary.copy()
    for c in summary_xlsx.columns:
        if c != "N":
            summary_xlsx[c] = pd.to_numeric(summary_xlsx[c], errors="coerce").round(3)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        df_xlsx.to_excel(writer, sheet_name="Details(mask-based)", index=False)
        summary_xlsx.to_excel(writer, sheet_name="By_N(mask-based)", index=False)

    print("\nResults saved to:", out_dir)
    print("Detailed CSV:", csv_path)
    print("TXT report:", txt_path)
    print("XLSX report:", xlsx_path)


def save_comparison_grid_png(save_path, combined, gt_components, pred_components, ids_for_cols,
                             title_main, markers=None):
    """Save a comparison figure with combined, true, and predicted plumes."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = combined.shape
    N = len(gt_components)
    assert len(pred_components) == N
    assert len(ids_for_cols) == N

    TOP_GT_GAP   = 0.2
    GT_PRED_GAP  = 0.2
    COL_WSPACE   = 0.01
    CBAR_SIZE    = "3.5%"
    CBAR_PAD     = 0.12
    LEFT, RIGHT, TOP, BOTTOM = 0.02, 0.98, 0.95, 0.02

    fig_w = 2.55 * max(N, 3) + 0.8
    fig_h = 8.8

    v = combined[combined > COMBINED_THR]
    if v.size > 10:
        vmax = float(np.percentile(v, VMAX_Q))
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = float(v.max())
        vmax = max(vmax, 1e-12)
    else:
        vmax = float(combined.max()) if combined.max() > 0 else 1.0
    vmin = 0.0

    def _ticks(step, L):
        L = int(L)
        if L <= 1:
            return [0]
        step = int(step)
        ticks = list(range(0, L, step))
        last = L - 1
        if last not in ticks:
            ticks.append(last)
        if len(ticks) >= 2 and (ticks[-1] - ticks[-2]) <= 10:
            ticks.pop(-2)
        return ticks

    def _apply_grid(ax, show_labels=False):
        major_step = 50
        minor_step = 25

        ax.set_xticks(_ticks(major_step, W))
        ax.set_yticks(_ticks(major_step, H))
        ax.set_xticks(_ticks(minor_step, W), minor=True)
        ax.set_yticks(_ticks(minor_step, H), minor=True)

        ax.grid(True, which="major", linestyle="--", linewidth=0.35, alpha=0.35, color="white")
        ax.grid(True, which="minor", linestyle=":",  linewidth=0.20, alpha=0.22, color="white")

        for sp in ax.spines.values():
            sp.set_linewidth(0.6)

        if not show_labels:
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.tick_params(which="both", length=0)
        else:
            ax.tick_params(labelsize=8)

    def _snap_center(v):
        frac = v - np.floor(v)
        if abs(frac - 0.5) < 1e-3:
            return v - 0.5
        return v

    def _marker_xy_for_plot(x, y):
        x0 = _snap_center(float(x))
        y0 = _snap_center(float(y))
        xp = np.clip(x0, 0, W - 1)
        yp = np.clip(y0, 0, H - 1)
        return float(xp), float(yp)

    def _disp(img):
        return np.flipud(img)

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI_GRID)

    outer = fig.add_gridspec(
        2, 1,
        height_ratios=[1.25, 2.05],
        hspace=TOP_GT_GAP
    )

    ax0 = fig.add_subplot(outer[0])
    im0 = ax0.imshow(_disp(combined), cmap=CMAP, origin="lower",
                     interpolation="bilinear", vmin=vmin, vmax=vmax)
    ax0.set_title(title_main, fontsize=12, pad=10)
    ax0.set_xlabel("X (pixel)", labelpad=4)
    ax0.set_ylabel("Y (pixel)", labelpad=4)
    _apply_grid(ax0, show_labels=True)

    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.ax.tick_params(labelsize=8, pad=2)
    cbar.ax.yaxis.tick_right()
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label(r"$\Delta$XCH$_4$ (kg/m$^2$)", rotation=90, labelpad=8, fontsize=9)

    if markers:
        from matplotlib.transforms import Bbox

        pts = np.array([_marker_xy_for_plot(x, y) for (x, y, _) in markers], dtype=np.float32)
        for (xp, yp) in pts:
            ax0.scatter([float(xp)], [float(yp)],
                        s=40, facecolors="none",
                        edgecolors="white", linewidths=1.2, zorder=6)

        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        ax_bbox = ax0.get_window_extent(renderer)

        reserved_bboxes = []
        r_px = 6
        for (xp, yp) in pts:
            px, py = ax0.transData.transform((float(xp), float(yp)))
            reserved_bboxes.append(Bbox.from_extents(px - r_px, py - r_px, px + r_px, py + r_px))

        def _overlap(b1: Bbox, b2: Bbox, pad=2.0) -> bool:
            return not (b1.x1 + pad < b2.x0 or b1.x0 - pad > b2.x1 or
                        b1.y1 + pad < b2.y0 or b1.y0 - pad > b2.y1)

        def _inside(b: Bbox, container: Bbox, margin=1.0) -> bool:
            return (b.x0 >= container.x0 + margin and b.x1 <= container.x1 - margin and
                    b.y0 >= container.y0 + margin and b.y1 <= container.y1 - margin)

        offsets = [
            (-6, 0),(-6, 6),(-10, 6),(0, 6),(6,6),(8, 8),
            (6, 0),(6, -6),(8, -8),(0,-8),
            (-6, -6),(-9, -7),(-10, -10),(-12, -12),
        ]

        placed_bboxes = []
        for (x, y, lab), (xp, yp) in zip(markers, pts):
            xp, yp = float(xp), float(yp)
            lab = str(lab)

            placed = False
            for (dx, dy) in offsets:
                ha = "left" if dx >= 0 else "right"
                va = "bottom" if dy >= 0 else "top"

                txt = ax0.annotate(
                    lab,
                    xy=(xp, yp),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha=ha, va=va,
                    color="white",
                    fontsize=8,
                    fontweight="normal",
                    zorder=7,
                    path_effects=[pe.withStroke(linewidth=0.9, foreground="black")]
                )

                fig.canvas.draw()
                bb = txt.get_window_extent(renderer)
                bb = Bbox.from_extents(bb.x0 - 2, bb.y0 - 1, bb.x1 + 2, bb.y1 + 1)

                if not _inside(bb, ax_bbox, margin=1.0):
                    txt.remove()
                    continue
                if any(_overlap(bb, rb, pad=1.0) for rb in reserved_bboxes):
                    txt.remove()
                    continue
                if any(_overlap(bb, pb, pad=1.0) for pb in placed_bboxes):
                    txt.remove()
                    continue

                placed_bboxes.append(bb)
                placed = True
                break

            if not placed:
                ax0.annotate(
                    lab,
                    xy=(xp, yp),
                    xytext=(18, 10),
                    textcoords="offset points",
                    ha="left", va="bottom",
                    color="white",
                    fontsize=7,
                    fontweight="normal",
                    zorder=7,
                    path_effects=[pe.withStroke(linewidth=0.9, foreground="black")]
                )

    bottom = outer[1].subgridspec(2, N, hspace=GT_PRED_GAP, wspace=COL_WSPACE)

    for i in range(N):
        ax = fig.add_subplot(bottom[0, i])
        ax.imshow(_disp(gt_components[i]), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT {ids_for_cols[i]}", fontsize=10, pad=4)
        _apply_grid(ax, show_labels=False)

    for i in range(N):
        ax = fig.add_subplot(bottom[1, i])
        ax.imshow(_disp(pred_components[i]), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred {ids_for_cols[i]}", fontsize=10, pad=4)
        _apply_grid(ax, show_labels=False)

    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    fig.savefig(save_path, pad_inches=0.02)
    plt.close(fig)


def export_one_grid(model, x_path, y_path, sid, out_root=None, bundle_root=None,
                    save_png=True, save_bundle=True):
    """Export a sample comparison figure and its IME result bundle."""
    xnp = np.load(x_path, allow_pickle=True)
    if "combined" in xnp.files:
        combined = xnp["combined"].astype(np.float32)
    elif "combined_clean" in xnp.files:
        combined = xnp["combined_clean"].astype(np.float32)
    else:
        raise KeyError(f"{x_path} contains neither 'combined' nor 'combined_clean'")

    ynp = np.load(y_path, allow_pickle=True)
    alpha = ynp["alpha_list"].astype(np.float32)
    N = int(ynp["N"])
    if not (2 <= N <= 8):
        return None, None

    signal_mask = build_alpha_signal_mask(
        alpha=alpha,
        N=N,
        eps=ALPHA_SIGNAL_SUM_EPS,
        do_dilate=True,
        dil_iters=SIGNAL_MASK_DILATE_ITERS
    )

    gt_components = (alpha * combined[None, :, :]).astype(np.float32)

    inp = torch.from_numpy(combined)[None, None, ...].to(DEVICE)
    with torch.no_grad():
        w_t, mask_logits_t = model(inp)

    w = w_t[0].detach().cpu().numpy().astype(np.float32)
    gate = torch.sigmoid(mask_logits_t)[0].detach().cpu().numpy().astype(np.float32)

    pairs_h = hungarian_match_softiou(
        w, alpha, N,
        signal_mask=signal_mask
    )

    pred_components = []
    pk_for_j = [-1] * N
    for pk, gj in pairs_h:
        pk_for_j[gj] = pk

    for j in range(N):
        pk = pk_for_j[j]
        if pk >= 0:
            pred_components.append((w[pk] * gate[pk] * combined).astype(np.float32))
        else:
            pred_components.append(np.zeros_like(combined, dtype=np.float32))

    markers, ids_for_cols, source_xy_save, source_id_save = extract_markers_and_ids_from_ynp(ynp, N)

    title_main = f"{sid} | N={N}"

    png_path = None
    bundle_path = None

    if save_png and (out_root is not None):
        out_dir_n = os.path.join(out_root, f"N_{N}")
        os.makedirs(out_dir_n, exist_ok=True)
        png_path = os.path.join(out_dir_n, f"{sid}__grid.png")

        save_comparison_grid_png(
            save_path=png_path,
            combined=combined,
            gt_components=[gt_components[j] for j in range(N)],
            pred_components=pred_components,
            ids_for_cols=ids_for_cols,
            title_main=title_main,
            markers=markers
        )

    if save_bundle and (bundle_root is not None):
        bundle_dir_n = os.path.join(bundle_root, f"N_{N}")
        os.makedirs(bundle_dir_n, exist_ok=True)
        bundle_path = os.path.join(bundle_dir_n, f"{sid}{RESULT_BUNDLE_SUFFIX}")

        save_result_bundle_npz(
            bundle_path=bundle_path,
            sid=sid,
            x_path=x_path,
            y_path=y_path,
            xnp=xnp,
            ynp=ynp,
            combined=combined,
            alpha=alpha,
            N=N,
            signal_mask=signal_mask,
            gt_components=gt_components,
            pred_components=pred_components,
            w=w,
            gate=gate,
            pairs_h=pairs_h,
            pk_for_j=pk_for_j,
            ids_for_cols=ids_for_cols,
            markers=markers,
            source_xy_save=source_xy_save,
            source_id_save=source_id_save
        )

    return png_path, bundle_path


def main():
    """Evaluate the test set and export figures, IME bundles, and reports."""
    seed_all(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)

    test_dir = os.path.join(DATA_ROOT, "test_dataset")
    print("test_dir:", test_dir)

    pairs = list_npz(test_dir)
    print("test samples:", len(pairs))

    model = load_model_from_ckpt(CKPT_PATH, DEVICE, kmax=KMAX, base=32, cat_ch=32)
    print("Loaded:", CKPT_PATH)


    all_rows = []
    pbar = tqdm(pairs, desc="Evaluating FULL test set")

    for (x_path, y_path, sid) in pbar:
        xnp = np.load(x_path)
        if "combined" in xnp.files:
            combined = xnp["combined"].astype(np.float32)
        elif "combined_clean" in xnp.files:
            combined = xnp["combined_clean"].astype(np.float32)
        else:
            raise KeyError(f"{x_path} contains neither 'combined' nor 'combined_clean'")

        ynp = np.load(y_path, allow_pickle=True)
        alpha = ynp["alpha_list"].astype(np.float32)
        N = int(ynp["N"])
        if not (2 <= N <= 8):
            continue

        signal_mask = build_alpha_signal_mask(
            alpha=alpha,
            N=N,
            eps=ALPHA_SIGNAL_SUM_EPS,
            do_dilate=True,
            dil_iters=SIGNAL_MASK_DILATE_ITERS
        )

        gt_components = alpha * combined[None, :, :]

        inp = torch.from_numpy(combined)[None, None, ...].to(DEVICE)
        with torch.no_grad():
            w_t, mask_logits_t = model(inp)

        w = w_t[0].detach().cpu().numpy().astype(np.float32)
        gate = torch.sigmoid(mask_logits_t)[0].detach().cpu().numpy().astype(np.float32)

        pairs_h = hungarian_match_softiou(w, alpha, N, signal_mask=signal_mask)

        _, ids_for_cols, source_xy_save, _ = extract_markers_and_ids_from_ynp(ynp, N)

        if source_xy_save is None or len(source_xy_save) < N:
            raise ValueError(
                f"{sid} lacks source coordinates required for plume mask extraction. "
                f"Provide src_xy_big_plot, src_xy_small_plot, src_xy, or markers_xy in the labels."
            )

        total_mask_for_metric = signal_mask


        for (pk, gj) in pairs_h:
            pred_c = (w[pk] * gate[pk] * combined).astype(np.float32)

            gt_c = gt_components[gj].astype(np.float32)
            src_id = ids_for_cols[gj] if (gj < len(ids_for_cols)) else f"gt{gj}"
            src_xy = source_xy_save[gj]

            append_rows_for_one_component_mask_based(
                all_rows=all_rows,
                N=N,
                sid=sid,
                src_id=src_id,
                pred_c=pred_c,
                gt_c=gt_c,
                source_xy=src_xy,
                total_mask=total_mask_for_metric
            )

    df = pd.DataFrame(all_rows)
    save_reports_by_N_mask_based(df, OUT_DIR)

    csv_path  = os.path.join(OUT_DIR, "metrics_all.csv")
    txt_path  = os.path.join(OUT_DIR, "evaluation_report.txt")
    xlsx_path = os.path.join(OUT_DIR, "evaluation_report.xlsx")


    grid_root = os.path.join(OUT_DIR, ALL_GRIDS_DIRNAME)
    bundle_root = os.path.join(OUT_DIR, RESULT_BUNDLES_DIRNAME)

    os.makedirs(grid_root, exist_ok=True)
    os.makedirs(bundle_root, exist_ok=True)

    pbar2 = tqdm(pairs, desc="Exporting ALL grids / result bundles")
    for (x_path, y_path, sid) in pbar2:
        try:
            png_path, bundle_path = export_one_grid(
                model,
                x_path, y_path, sid,
                out_root=grid_root,
                bundle_root=bundle_root,
                save_png=True,
                save_bundle=True
            )

            show_name = None
            if bundle_path is not None:
                show_name = os.path.basename(bundle_path)
            elif png_path is not None:
                show_name = os.path.basename(png_path)

            if show_name is not None:
                pbar2.set_postfix({"saved": show_name})

        except Exception as e:
            pbar2.set_postfix({"error": str(e)[:60]})
            continue

    print("\nResults saved to:", OUT_DIR)
    print("Detailed CSV:", csv_path)
    print("TXT report:", txt_path)
    print("XLSX report:", xlsx_path)

    print("Comparison grids:", os.path.join(OUT_DIR, ALL_GRIDS_DIRNAME))

    print("Result bundles for IME:", os.path.join(OUT_DIR, RESULT_BUNDLES_DIRNAME))


if __name__ == "__main__":
    main()
