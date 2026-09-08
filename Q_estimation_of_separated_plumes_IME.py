"""
Estimate source emission rates from separated methane plumes using IME.

Inputs: RESULT_BUNDLES_FOR_IME exported by the test script, and matching
test-set labels containing 10m wind speeds(overlapping_plumes(2-8)-V2__Add_S10\test_dataset\labels).
Outputs: plume masks with emission-rate annotations in PER_SAMPLE_MASK_PREVIEW,
per-sample separation and quantification figures in PER_SAMPLE_Q_FIGS,
and CSV statistics and a text report in the output directory.
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import (
    label,
    binary_fill_holes,
    binary_dilation,
)

# The bundle directory exported by the test script. Fill with the path of the output RESULT_BUNDLES_FOR_IME folder.
INPUT_BUNDLE_DIR = r"...\RESULT_BUNDLES_FOR_IME"

# the labels directory of the test set with U10
TEST_LABEL_DIR = r"...\overlapping_plumes(2-8)-V2__Add_S10\test_dataset\labels"

OUTPUT_DIR = r"<path_to_IME_output_directory>"

MASK_PREVIEW_DIRNAME = "PER_SAMPLE_MASK_PREVIEW"
MASK_OVERLAY_ALPHA = 0.22
MASK_CONTOUR_LINEWIDTH = 1.0
MASK_MARKER_SIZE = 35

BASE_GT_Q_KGH = 3600.0

GT_SCALE_KEY = "scale_list"

PRED_COMPONENT_KEY = "pred_components"
GT_COMPONENT_KEY   = "gt_components"
COMBINED_KEY       = "combined_eval"
SIGNAL_MASK_KEY    = "signal_mask"
SOURCE_XY_KEY      = "source_xy_for_ime"
SOURCE_ID_KEY      = "source_id_for_ime"
IDS_FOR_COLS_KEY   = "ids_for_cols"
N_KEY              = "N_eval"
SAMPLE_ID_KEY      = "sample_id"


SOURCE_COORD_SYSTEM = "plot_xy"


PIXEL_SIZE_M = 60.0
PIXEL_AREA_M2 = PIXEL_SIZE_M * PIXEL_SIZE_M


ABS_THR = 1e-5
REL_THR = 0.01

USE_TOTAL_MASK_IF_AVAILABLE = True
MASK_FILL_HOLES = True
MASK_DILATE_ITERS = 1


MIN_COMPONENT_PIXELS = 9
MAX_COMPONENT_GAP_PIXELS = 10.0
MAX_SOURCE_DISTANCE_PIXELS = 100.0
PREFER_SOURCE_CONTAINING_COMPONENT = True


U10_KEY = "u10_list"


CMAP = "plasma"
COMBINED_THR = 1e-5
VMAX_Q = 99.8
DPI_GRID = 140

PER_SAMPLE_FIG_DIRNAME = "PER_SAMPLE_Q_FIGS"


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
        a = pts1[s:s+chunk]
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


def extract_single_source_plume_mask(component_kgm2,
                                     source_row,
                                     source_col,
                                     abs_thr=1e-5,
                                     rel_thr=0.01,
                                     total_mask=None,
                                     fill_holes=True,
                                     dilate_iters=0):
    """Extract and refine a plume mask for one source."""
    mask0, thr_used = build_initial_component_mask(
        component_kgm2=component_kgm2,
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
        prefer_source_containing_component=PREFER_SOURCE_CONTAINING_COMPONENT
    )

    mask_final = postprocess_mask(
        mask_in=mask_cc,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )

    return mask_final.astype(np.uint8), float(thr_used)


def get_single_source_mask_from_component(component_kgm2,
                                          source_xy,
                                          source_coord_system="plot_xy",
                                          total_mask=None,
                                          abs_thr=1e-5,
                                          rel_thr=0.01,
                                          fill_holes=True,
                                          dilate_iters=0):
    """Extract a source mask and its metadata."""
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

    mask, thr_used = extract_single_source_plume_mask(
        component_kgm2=comp,
        source_row=source_row,
        source_col=source_col,
        abs_thr=abs_thr,
        rel_thr=rel_thr,
        total_mask=total_mask,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )

    info = {
        "source_x": x,
        "source_y": y,
        "source_row": float(source_row),
        "source_col": float(source_col),
        "mask_pixels": int(mask.sum()),
        "thr_used": float(thr_used),
        "note": "ok" if int(mask.sum()) > 0 else "mask_empty"
    }
    return mask.astype(np.uint8), info


def compute_plume_length_from_area_m(mask, pixel_area_m2):
    """Compute plume length as the square root of its area."""
    m = (mask > 0)
    mask_pixels = int(m.sum())

    plume_area_m2 = float(mask_pixels) * float(pixel_area_m2)
    L_m = float(np.sqrt(plume_area_m2)) if plume_area_m2 > 0 else 0.0

    return L_m, plume_area_m2, mask_pixels


def compute_ime_kg(component_kgm2, mask, pixel_area_m2):
    """Integrate positive methane enhancement within the plume mask."""
    comp = np.asarray(component_kgm2, dtype=np.float32)
    m = (mask > 0)

    mass_map = np.clip(comp, 0.0, None) * float(pixel_area_m2)
    ime_kg = float(np.sum(mass_map[m]))

    return max(ime_kg, 0.0)


def estimate_one_source_Q_ime(component_kgm2,
                              source_xy,
                              ueff_mps,
                              pixel_size_m=60.0,
                              pixel_area_m2=None,
                              source_coord_system="plot_xy",
                              total_mask=None,
                              abs_thr=1e-5,
                              rel_thr=0.01,
                              fill_holes=True,
                              dilate_iters=0,
                              source_id="source"):
    """Estimate one source emission rate using IME and effective wind speed."""
    comp = np.asarray(component_kgm2, dtype=np.float32)
    H, W = comp.shape

    if pixel_area_m2 is None:
        pixel_area_m2 = float(pixel_size_m) * float(pixel_size_m)

    x, y = float(source_xy[0]), float(source_xy[1])
    source_row, source_col = source_xy_to_rowcol(
        x=x, y=y, H=H, coord_system=source_coord_system
    )

    if not rowcol_inside_image(source_row, source_col, H, W):
        return {
            "source_id": str(source_id),
            "source_x": x,
            "source_y": y,
            "source_row": source_row,
            "source_col": source_col,
            "ueff_mps": float(ueff_mps),
            "mask_pixels": 0,
            "mask_area_m2": 0.0,
            "ime_kg": 0.0,
            "L_m": 0.0,
            "Q_kg_s": 0.0,
            "Q_kg_h": 0.0,
            "note": "source_outside"
        }

    mask, thr_used = extract_single_source_plume_mask(
        component_kgm2=comp,
        source_row=source_row,
        source_col=source_col,
        abs_thr=abs_thr,
        rel_thr=rel_thr,
        total_mask=total_mask,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )

    if int(mask.sum()) == 0:
        return {
            "source_id": str(source_id),
            "source_x": x,
            "source_y": y,
            "source_row": source_row,
            "source_col": source_col,
            "ueff_mps": float(ueff_mps),
            "mask_pixels": 0,
            "mask_area_m2": 0.0,
            "ime_kg": 0.0,
            "L_m": 0.0,
            "Q_kg_s": 0.0,
            "Q_kg_h": 0.0,
            "note": "mask_empty",
            "thr_used": float(thr_used)
        }

    ime_kg = compute_ime_kg(
        component_kgm2=comp,
        mask=mask,
        pixel_area_m2=pixel_area_m2
    )

    L_m, plume_area_m2, mask_pixels = compute_plume_length_from_area_m(
        mask=mask,
        pixel_area_m2=pixel_area_m2
    )

    if L_m <= 1e-12:
        Q_kg_s = 0.0
        note = "L_too_small"
    else:
        Q_kg_s = float(ueff_mps) * float(ime_kg) / float(L_m)
        note = "ok"

    Q_kg_h = Q_kg_s * 3600.0

    return {
        "source_id": str(source_id),
        "source_x": x,
        "source_y": y,
        "source_row": float(source_row),
        "source_col": float(source_col),
        "ueff_mps": float(ueff_mps),
        "mask_pixels": int(mask_pixels),
        "mask_area_m2": float(plume_area_m2),
        "ime_kg": float(ime_kg),
        "L_m": float(L_m),
        "Q_kg_s": float(Q_kg_s),
        "Q_kg_h": float(Q_kg_h),
        "note": note,
        "thr_used": float(thr_used)
    }


def calc_ueff_from_u10(u10_array):
    """Convert 10 m wind speed to effective wind speed."""
    u10 = np.asarray(u10_array, dtype=np.float32)

    if np.any(~np.isfinite(u10)):
        raise ValueError(f"Non-finite values in u10_list: {u10}")


    ueff = 0.4535 * u10 + 0.6541

    return np.asarray(ueff, dtype=np.float32)


def list_bundle_npz(bundle_root):
    """List result bundles grouped by source count."""
    if not os.path.isdir(bundle_root):
        raise FileNotFoundError(f"Bundle directory not found: {bundle_root}")

    xs = sorted(glob.glob(os.path.join(bundle_root, "N_*", "*.npz")))
    if len(xs) == 0:
        raise FileNotFoundError(f"No bundle NPZ files found under {bundle_root}")

    return xs


def load_test_label_by_sample_id(sample_id, label_dir):
    """Load the matching test label file."""
    label_path = os.path.join(label_dir, f"{sample_id}.npz")
    if not os.path.exists(label_path):
        raise FileNotFoundError(f"Matching test label file not found: {label_path}")

    return np.load(label_path, allow_pickle=True), label_path


def load_bundle_for_q(bundle_path):
    """Load separated plumes, source metadata, reference rates, and wind speeds."""
    if not os.path.exists(bundle_path):
        raise FileNotFoundError(f"Bundle not found: {bundle_path}")

    z = np.load(bundle_path, allow_pickle=True)


    if SAMPLE_ID_KEY in z.files:
        sample_id = str(z[SAMPLE_ID_KEY].tolist())
    else:
        sample_id = os.path.splitext(os.path.basename(bundle_path))[0]

    if sample_id.endswith("__bundle"):
        sample_id = sample_id[:-8]


    if N_KEY in z.files:
        N = int(z[N_KEY])
    else:
        N = int(z[PRED_COMPONENT_KEY].shape[0])


    for k in [PRED_COMPONENT_KEY, GT_COMPONENT_KEY, COMBINED_KEY, SIGNAL_MASK_KEY]:
        if k not in z.files:
            raise KeyError(f"{bundle_path} is missing required field: {k}")

    if SOURCE_XY_KEY not in z.files:
        raise KeyError(f"{bundle_path} is missing source coordinate field: {SOURCE_XY_KEY}")

    if GT_SCALE_KEY not in z.files:
        raise KeyError(
            f"{bundle_path} is missing GT scaling field {GT_SCALE_KEY}. "
            f"Ensure the test script preserves the original input NPZ fields in the bundle."
        )

    pred_components = z[PRED_COMPONENT_KEY].astype(np.float32)
    gt_components = z[GT_COMPONENT_KEY].astype(np.float32)
    combined = z[COMBINED_KEY].astype(np.float32)
    signal_mask = (z[SIGNAL_MASK_KEY] > 0).astype(np.uint8)
    source_xy = z[SOURCE_XY_KEY].astype(np.float32)


    label_npz, label_path = load_test_label_by_sample_id(sample_id, TEST_LABEL_DIR)

    if U10_KEY not in label_npz.files:
        raise KeyError(f"{label_path} is missing field: {U10_KEY}")

    u10_list = np.asarray(label_npz[U10_KEY], dtype=np.float32).reshape(-1)

    if u10_list.size < N:
        raise ValueError(
            f"{label_path} contains too few values in {U10_KEY}. "
            f"N={N}, but u10_list.size={u10_list.size}"
        )

    u10_list = u10_list[:N]
    ueff_list = calc_ueff_from_u10(u10_list)


    if SOURCE_ID_KEY in z.files:
        source_ids = [str(x) for x in z[SOURCE_ID_KEY].tolist()]
    elif IDS_FOR_COLS_KEY in z.files:
        source_ids = [str(x) for x in z[IDS_FOR_COLS_KEY].tolist()]
    else:
        source_ids = [f"source_{i + 1}" for i in range(N)]


    scale_list = np.asarray(z[GT_SCALE_KEY], dtype=np.float32).reshape(-1)
    if scale_list.size < N:
        raise ValueError(
            f"{bundle_path} contains too few values in {GT_SCALE_KEY}. "
            f"N={N}, but scale_list.size={scale_list.size}"
        )

    scale_list = scale_list[:N]
    q_gt_true_kgh = BASE_GT_Q_KGH * scale_list


    source_ids = source_ids[:N]
    source_xy = source_xy[:N]
    pred_components = pred_components[:N]
    gt_components = gt_components[:N]

    return {
        "bundle_path": bundle_path,
        "sample_id": sample_id,
        "N": N,
        "combined": combined,
        "signal_mask": signal_mask,
        "pred_components": pred_components,
        "gt_components": gt_components,
        "source_xy": source_xy,
        "source_ids": source_ids,
        "scale_list": scale_list,
        "Q_gt_true_kgh": q_gt_true_kgh,
        "u10_list": u10_list,
        "ueff_list": ueff_list,
    }


def build_all_pred_masks_from_one_bundle(bundle_info,
                                         source_coord_system="plot_xy",
                                         abs_thr=1e-5,
                                         rel_thr=0.01,
                                         fill_holes=True,
                                         dilate_iters=0):
    """Extract masks for predicted source components."""
    pred_components = bundle_info["pred_components"]
    source_xy = bundle_info["source_xy"]
    total_mask = bundle_info["signal_mask"] if USE_TOTAL_MASK_IF_AVAILABLE else None
    N = int(bundle_info["N"])

    return build_all_masks_from_components(
        components=pred_components,
        source_xy=source_xy,
        total_mask=total_mask,
        N=N,
        source_coord_system=source_coord_system,
        abs_thr=abs_thr,
        rel_thr=rel_thr,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )


def build_all_gt_masks_from_one_bundle(bundle_info,
                                       source_coord_system="plot_xy",
                                       abs_thr=1e-5,
                                       rel_thr=0.01,
                                       fill_holes=True,
                                       dilate_iters=0):
    """Extract masks for ground-truth source components."""
    gt_components = bundle_info["gt_components"]
    source_xy = bundle_info["source_xy"]
    total_mask = bundle_info["signal_mask"] if USE_TOTAL_MASK_IF_AVAILABLE else None
    N = int(bundle_info["N"])

    return build_all_masks_from_components(
        components=gt_components,
        source_xy=source_xy,
        total_mask=total_mask,
        N=N,
        source_coord_system=source_coord_system,
        abs_thr=abs_thr,
        rel_thr=rel_thr,
        fill_holes=fill_holes,
        dilate_iters=dilate_iters
    )


def build_all_masks_from_components(components,
                                    source_xy,
                                    total_mask,
                                    N,
                                    source_coord_system="plot_xy",
                                    abs_thr=1e-5,
                                    rel_thr=0.01,
                                    fill_holes=True,
                                    dilate_iters=0):
    """Extract masks and metadata for all source components."""
    masks = []
    infos = []

    for i in range(N):
        m, info = get_single_source_mask_from_component(
            component_kgm2=components[i],
            source_xy=source_xy[i],
            source_coord_system=source_coord_system,
            total_mask=total_mask,
            abs_thr=abs_thr,
            rel_thr=rel_thr,
            fill_holes=fill_holes,
            dilate_iters=dilate_iters
        )
        masks.append(m)
        infos.append(info)

    masks = np.stack(masks, axis=0).astype(np.uint8)
    return masks, infos


def estimate_all_sources_from_one_bundle(bundle_info,
                                         pixel_size_m=60.0,
                                         pixel_area_m2=None,
                                         source_coord_system="plot_xy",
                                         abs_thr=1e-5,
                                         rel_thr=0.01,
                                         fill_holes=True,
                                         dilate_iters=0):
    """Estimate source emission rates and assemble detailed results."""
    pred_components = bundle_info["pred_components"]
    source_xy = bundle_info["source_xy"]
    source_ids = bundle_info["source_ids"]
    total_mask = bundle_info["signal_mask"] if USE_TOTAL_MASK_IF_AVAILABLE else None
    q_gt_true_kgh = bundle_info["Q_gt_true_kgh"]

    N = int(bundle_info["N"])

    if pixel_area_m2 is None:
        pixel_area_m2 = float(pixel_size_m) * float(pixel_size_m)

    ueffs = np.asarray(bundle_info["ueff_list"], dtype=np.float32)
    if ueffs.size != N:
        raise ValueError(
            f"Bundle ueff_list length does not match N: len={ueffs.size}, N={N}"
        )

    rows = []
    pred_q_kgh = []

    for i in range(N):
        row = estimate_one_source_Q_ime(
            component_kgm2=pred_components[i],
            source_xy=source_xy[i],
            ueff_mps=ueffs[i],
            pixel_size_m=pixel_size_m,
            pixel_area_m2=pixel_area_m2,
            source_coord_system=source_coord_system,
            total_mask=total_mask,
            abs_thr=abs_thr,
            rel_thr=rel_thr,
            fill_holes=fill_holes,
            dilate_iters=dilate_iters,
            source_id=source_ids[i]
        )

        row["sample_id"] = bundle_info["sample_id"]
        row["bundle_path"] = bundle_info["bundle_path"]
        row["N"] = N
        row["source_idx"] = i
        row["scale_factor"] = float(bundle_info["scale_list"][i])
        row["u10_mps"] = float(bundle_info["u10_list"][i])
        row["Q_gt_true_kgh"] = float(q_gt_true_kgh[i])

        rows.append(row)
        pred_q_kgh.append(float(row["Q_kg_h"]))

    df = pd.DataFrame(rows)
    return df, pred_q_kgh


def estimate_all_sources_from_components_and_masks(components,
                                                   masks,
                                                   source_xy,
                                                   source_ids,
                                                   ueff_list,
                                                   sample_id,
                                                   bundle_path,
                                                   N,
                                                   scale_list,
                                                   q_gt_true_kgh,
                                                   pixel_size_m=60.0,
                                                   pixel_area_m2=None):
    """Compute IME and emission rates from source components and masks."""
    if pixel_area_m2 is None:
        pixel_area_m2 = float(pixel_size_m) * float(pixel_size_m)

    rows = []
    q_list = []
    ime_list = []

    for i in range(N):
        comp = np.asarray(components[i], dtype=np.float32)
        mask = (np.asarray(masks[i]) > 0).astype(np.uint8)

        x, y = float(source_xy[i][0]), float(source_xy[i][1])
        source_row, source_col = source_xy_to_rowcol(
            x=x, y=y, H=comp.shape[0], coord_system=SOURCE_COORD_SYSTEM
        )

        if int(mask.sum()) == 0:
            ime_kg = 0.0
            L_m = 0.0
            mask_area_m2 = 0.0
            mask_pixels = 0
            q_kg_s = 0.0
            q_kg_h = 0.0
            note = "mask_empty"
        else:
            ime_kg = compute_ime_kg(
                component_kgm2=comp,
                mask=mask,
                pixel_area_m2=pixel_area_m2
            )

            L_m, mask_area_m2, mask_pixels = compute_plume_length_from_area_m(
                mask=mask,
                pixel_area_m2=pixel_area_m2
            )

            if L_m <= 1e-12:
                q_kg_s = 0.0
                q_kg_h = 0.0
                note = "L_too_small"
            else:
                q_kg_s = float(ueff_list[i]) * float(ime_kg) / float(L_m)
                q_kg_h = q_kg_s * 3600.0
                note = "ok"

        row = {
            "sample_id": sample_id,
            "bundle_path": bundle_path,
            "N": int(N),
            "source_idx": i,
            "source_id": str(source_ids[i]),
            "source_x": x,
            "source_y": y,
            "source_row": float(source_row),
            "source_col": float(source_col),
            "ueff_mps": float(ueff_list[i]),
            "u10_mps": float(ueff_list[i]),
            "mask_pixels": int(mask_pixels),
            "mask_area_m2": float(mask_area_m2),
            "ime_kg": float(ime_kg),
            "L_m": float(L_m),
            "Q_kg_s": float(q_kg_s),
            "Q_kg_h": float(q_kg_h),
            "scale_factor": float(scale_list[i]),
            "Q_gt_true_kgh": float(q_gt_true_kgh[i]),
            "note": note,
        }
        rows.append(row)
        q_list.append(float(q_kg_h))
        ime_list.append(float(ime_kg))

    df = pd.DataFrame(rows)
    return df, q_list, ime_list


def _fmt_q(q):
    """Format emission rates for figure titles."""
    q = float(q)
    if q >= 1000:
        return f"{q:.0f}"
    if q >= 100:
        return f"{q:.1f}"
    return f"{q:.2f}"


def save_comparison_grid_with_q(save_path, combined, gt_components, pred_components,
                                ids_for_cols, gt_qs, pred_qs, markers=None, title_main=""):
    """Save per-sample separation figures with emission rates."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = combined.shape
    N = len(gt_components)

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
        ticks = list(range(0, L, int(step)))
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
        ax.grid(True, which="minor", linestyle=":", linewidth=0.20, alpha=0.22, color="white")

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

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size=CBAR_SIZE, pad=CBAR_PAD)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.ax.tick_params(labelsize=8, pad=2)
    cbar.ax.yaxis.tick_right()
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("$\\Delta XCH_4$ (kg/m²)", rotation=270, labelpad=12, fontsize=9)


    if markers is not None and len(markers) > 0:
        pts = np.array([_marker_xy_for_plot(x, y) for (x, y, _) in markers], dtype=np.float32)
        for (xp, yp) in pts:
            ax0.scatter([float(xp)], [float(yp)],
                        s=40, facecolors="none",
                        edgecolors="white", linewidths=1.2, zorder=6)


    bottom = outer[1].subgridspec(2, N, hspace=GT_PRED_GAP, wspace=COL_WSPACE)

    for i in range(N):
        ax = fig.add_subplot(bottom[0, i])
        ax.imshow(_disp(gt_components[i]), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f"GT {ids_for_cols[i]}\n(Q={_fmt_q(gt_qs[i])} kg/h)", fontsize=9, pad=4)
        _apply_grid(ax, show_labels=False)

    for i in range(N):
        ax = fig.add_subplot(bottom[1, i])
        ax.imshow(_disp(pred_components[i]), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)
        ax.set_title(f"Pred {ids_for_cols[i]}\n(Q={_fmt_q(pred_qs[i])} kg/h)", fontsize=9, pad=4)
        _apply_grid(ax, show_labels=False)

    fig.subplots_adjust(left=LEFT, right=RIGHT, top=TOP, bottom=BOTTOM)
    fig.savefig(save_path, pad_inches=0.02)
    plt.close(fig)


def compute_n_error_table(df_valid, n_values=range(2, 9)):

    """Summarize emission-rate errors by source count."""
    df = df_valid.copy()


    if "Q_abs_error_kgh" not in df.columns:
        df["Q_abs_error_kgh"] = np.abs(
            df["Q_kg_h"].to_numpy(dtype=float)
            - df["Q_gt_true_kgh"].to_numpy(dtype=float)
        )

    if "Q_abs_pct_error" not in df.columns:
        df["Q_abs_pct_error"] = (
            df["Q_abs_error_kgh"].to_numpy(dtype=float)
            / (df["Q_gt_true_kgh"].to_numpy(dtype=float) + 1e-8)
            * 100.0
        )

    rows = []

    for N in n_values:
        sub = df[df["N"] == N].copy()

        if len(sub) == 0:
            rows.append({
                "N": int(N),
                "n": 0,
                "GT_mean_kgh": np.nan,
                "Pred_mean_kgh": np.nan,
                "MAE_kgh": np.nan,
                "MAPE_pct": np.nan,
                "Median_AE_kgh": np.nan,
                "Median_APE_pct": np.nan,
                "RMSE_kgh": np.nan,
                "Bias_kgh": np.nan,
            })
            continue

        gt = sub["Q_gt_true_kgh"].to_numpy(dtype=float)
        pred = sub["Q_kg_h"].to_numpy(dtype=float)

        abs_err = np.abs(pred - gt)
        ape = abs_err / (gt + 1e-8) * 100.0

        rows.append({
            "N": int(N),
            "n": int(len(sub)),
            "GT_mean_kgh": float(np.mean(gt)),
            "Pred_mean_kgh": float(np.mean(pred)),
            "MAE_kgh": float(np.mean(abs_err)),
            "MAPE_pct": float(np.mean(ape)),
            "Median_AE_kgh": float(np.median(abs_err)),
            "Median_APE_pct": float(np.median(ape)),
            "RMSE_kgh": float(np.sqrt(np.mean((pred - gt) ** 2))),
            "Bias_kgh": float(np.mean(pred - gt)),
        })

    return pd.DataFrame(rows)


def save_pred_mask_preview_grid(save_path,
                                combined,
                                gt_components,
                                gt_masks,
                                pred_components,
                                pred_masks,
                                ids_for_cols,
                                source_xy,
                                gt_true_qs,
                                gt_ime_qs,
                                gt_ime_imes,
                                pred_ime_qs,
                                pred_ime_imes,
                                title_main=""):
    """Save plume masks with IME and emission-rate annotations."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    H, W = combined.shape
    N = len(pred_components)

    fig_w = 2.8 * max(N, 3) + 1.0
    fig_h = 10.2

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
        ticks = list(range(0, L, int(step)))
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
        ax.grid(True, which="minor", linestyle=":", linewidth=0.20, alpha=0.22, color="white")

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
    outer = fig.add_gridspec(3, 1, height_ratios=[1.5, 1.5, 1.5], hspace=0.38)


    ax0 = fig.add_subplot(outer[0])
    im0 = ax0.imshow(_disp(combined), cmap=CMAP, origin="lower",
                     interpolation="bilinear", vmin=vmin, vmax=vmax)
    ax0.set_title(title_main, fontsize=12, pad=11)
    ax0.set_xlabel("X (pixel)", labelpad=4)
    ax0.set_ylabel("Y (pixel)", labelpad=4)
    _apply_grid(ax0, show_labels=True)

    from mpl_toolkits.axes_grid1 import make_axes_locatable
    divider = make_axes_locatable(ax0)
    cax = divider.append_axes("right", size="3.5%", pad=0.12)
    cbar = fig.colorbar(im0, cax=cax)
    cbar.ax.tick_params(labelsize=8, pad=2)
    cbar.ax.yaxis.tick_right()
    cbar.ax.yaxis.set_label_position("right")
    cbar.set_label("$\\Delta XCH_4$ (kg/m²)", rotation=270, labelpad=12, fontsize=9)

    for i in range(N):
        xp, yp = _marker_xy_for_plot(source_xy[i, 0], source_xy[i, 1])
        ax0.scatter([xp], [yp],
                    s=MASK_MARKER_SIZE,
                    facecolors="none",
                    edgecolors="white",
                    linewidths=1.2,
                    zorder=6)


    gt_grid = outer[1].subgridspec(1, N, wspace=0.05)

    for i in range(N):
        ax = fig.add_subplot(gt_grid[0, i])

        gt_img = gt_components[i]
        gt_mask_img = gt_masks[i]

        ax.imshow(_disp(gt_img), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)

        gt_mask_disp = _disp(gt_mask_img > 0)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 0] = 0.35
        overlay[..., 1] = 0.95
        overlay[..., 2] = 1.00
        overlay[..., 3] = gt_mask_disp.astype(np.float32) * MASK_OVERLAY_ALPHA
        ax.imshow(overlay, origin="lower", interpolation="nearest")

        try:
            ax.contour(gt_mask_disp.astype(np.float32),
                       levels=[0.5],
                       colors=["#7fe7ff"],
                       linewidths=MASK_CONTOUR_LINEWIDTH,
                       origin="lower")
        except Exception:
            pass

        xp, yp = _marker_xy_for_plot(source_xy[i, 0], source_xy[i, 1])
        ax.scatter([xp], [yp],
                   s=MASK_MARKER_SIZE,
                   facecolors="none",
                   edgecolors="white",
                   linewidths=1.0,
                   zorder=7)

        ax.set_title(
            f"GT {ids_for_cols[i]}\n"
            f"Q_true={_fmt_q(gt_true_qs[i])} kg/h\n"
            f"Q_ime={_fmt_q(gt_ime_qs[i])} kg/h | IME={gt_ime_imes[i]:.3f} kg\n",
            fontsize=8, pad=4
        )
        _apply_grid(ax, show_labels=False)


    pred_grid = outer[2].subgridspec(1, N, wspace=0.05)

    for i in range(N):
        ax = fig.add_subplot(pred_grid[0, i])

        pred_img = pred_components[i]
        mask_img = pred_masks[i]

        ax.imshow(_disp(pred_img), cmap=CMAP, origin="lower",
                  interpolation="bilinear", vmin=vmin, vmax=vmax)

        mask_disp = _disp(mask_img > 0)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[..., 0] = 0.35
        overlay[..., 1] = 0.95
        overlay[..., 2] = 1.00
        overlay[..., 3] = mask_disp.astype(np.float32) * MASK_OVERLAY_ALPHA
        ax.imshow(overlay, origin="lower", interpolation="nearest")

        try:
            ax.contour(mask_disp.astype(np.float32),
                       levels=[0.5],
                       colors=["#7fe7ff"],
                       linewidths=MASK_CONTOUR_LINEWIDTH,
                       origin="lower")
        except Exception:
            pass

        xp, yp = _marker_xy_for_plot(source_xy[i, 0], source_xy[i, 1])
        ax.scatter([xp], [yp],
                   s=MASK_MARKER_SIZE,
                   facecolors="none",
                   edgecolors="white",
                   linewidths=1.0,
                   zorder=7)

        ax.set_title(
            f"Pred {ids_for_cols[i]}\n"
            f"Q_ime={_fmt_q(pred_ime_qs[i])} kg/h | IME={pred_ime_imes[i]:.3f} kg\n",
            fontsize=8, pad=4
        )
        _apply_grid(ax, show_labels=False)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.97, bottom=0.03)
    fig.savefig(save_path, pad_inches=0.02)
    plt.close(fig)


def main():
    """Process result bundles and export per-sample figures and statistics."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    bundle_paths = list_bundle_npz(INPUT_BUNDLE_DIR)
    print(f"Bundle files found: {len(bundle_paths)}")

    per_sample_dir = os.path.join(OUTPUT_DIR, PER_SAMPLE_FIG_DIRNAME)

    all_rows = []

    pbar = tqdm(bundle_paths, desc="Processing bundles", ncols=120)

    for bundle_path in pbar:
        try:
            pbar.set_postfix_str(os.path.basename(bundle_path))

            bundle_info = load_bundle_for_q(bundle_path)

            pbar.set_postfix_str(
                f"N={bundle_info['N']}, sample={bundle_info['sample_id']}"
            )


            pred_masks, _ = build_all_pred_masks_from_one_bundle(
                bundle_info=bundle_info,
                source_coord_system=SOURCE_COORD_SYSTEM,
                abs_thr=ABS_THR,
                rel_thr=REL_THR,
                fill_holes=MASK_FILL_HOLES,
                dilate_iters=MASK_DILATE_ITERS
            )


            gt_masks, _ = build_all_gt_masks_from_one_bundle(
                bundle_info=bundle_info,
                source_coord_system=SOURCE_COORD_SYSTEM,
                abs_thr=ABS_THR,
                rel_thr=REL_THR,
                fill_holes=MASK_FILL_HOLES,
                dilate_iters=MASK_DILATE_ITERS
            )


            _, gt_ime_qs, gt_ime_imes = estimate_all_sources_from_components_and_masks(
                components=bundle_info["gt_components"],
                masks=gt_masks,
                source_xy=bundle_info["source_xy"],
                source_ids=bundle_info["source_ids"],
                ueff_list=bundle_info["ueff_list"],
                sample_id=bundle_info["sample_id"],
                bundle_path=bundle_info["bundle_path"],
                N=bundle_info["N"],
                scale_list=bundle_info["scale_list"],
                q_gt_true_kgh=bundle_info["Q_gt_true_kgh"],
                pixel_size_m=PIXEL_SIZE_M,
                pixel_area_m2=PIXEL_AREA_M2
            )


            _, pred_ime_qs, pred_ime_imes = estimate_all_sources_from_components_and_masks(
                components=bundle_info["pred_components"],
                masks=pred_masks,
                source_xy=bundle_info["source_xy"],
                source_ids=bundle_info["source_ids"],
                ueff_list=bundle_info["ueff_list"],
                sample_id=bundle_info["sample_id"],
                bundle_path=bundle_info["bundle_path"],
                N=bundle_info["N"],
                scale_list=bundle_info["scale_list"],
                q_gt_true_kgh=bundle_info["Q_gt_true_kgh"],
                pixel_size_m=PIXEL_SIZE_M,
                pixel_area_m2=PIXEL_AREA_M2
            )


            preview_dir = os.path.join(OUTPUT_DIR, MASK_PREVIEW_DIRNAME)
            preview_path = os.path.join(
                preview_dir,
                f"N_{bundle_info['N']}",
                f"{bundle_info['sample_id']}__pred_mask_preview.png"
            )

            save_pred_mask_preview_grid(
                save_path=preview_path,
                combined=bundle_info["combined"],
                gt_components=bundle_info["gt_components"],
                gt_masks=gt_masks,
                pred_components=bundle_info["pred_components"],
                pred_masks=pred_masks,
                ids_for_cols=bundle_info["source_ids"],
                source_xy=bundle_info["source_xy"],
                gt_true_qs=bundle_info["Q_gt_true_kgh"],
                gt_ime_qs=gt_ime_qs,
                gt_ime_imes=gt_ime_imes,
                pred_ime_qs=pred_ime_qs,
                pred_ime_imes=pred_ime_imes,
                title_main=f"{bundle_info['sample_id']} | N={bundle_info['N']} | GT vs Pred"
            )


            df_one, pred_q_kgh = estimate_all_sources_from_one_bundle(
                bundle_info=bundle_info,
                pixel_size_m=PIXEL_SIZE_M,
                pixel_area_m2=PIXEL_AREA_M2,
                source_coord_system=SOURCE_COORD_SYSTEM,
                abs_thr=ABS_THR,
                rel_thr=REL_THR,
                fill_holes=MASK_FILL_HOLES,
                dilate_iters=MASK_DILATE_ITERS
            )

            all_rows.append(df_one)


            source_ids = bundle_info["source_ids"]
            source_xy = bundle_info["source_xy"]
            markers = [
                (float(source_xy[i, 0]), float(source_xy[i, 1]), source_ids[i])
                for i in range(len(source_ids))
            ]

            save_path = os.path.join(
                per_sample_dir,
                f"N_{bundle_info['N']}",
                f"{bundle_info['sample_id']}__Q_grid.png"
            )

            save_comparison_grid_with_q(
                save_path=save_path,
                combined=bundle_info["combined"],
                gt_components=bundle_info["gt_components"],
                pred_components=bundle_info["pred_components"],
                ids_for_cols=source_ids,
                gt_qs=bundle_info["Q_gt_true_kgh"],
                pred_qs=pred_q_kgh,
                markers=markers,
                title_main=f"{bundle_info['sample_id']} | N={bundle_info['N']}"
            )

        except Exception as e:
            print("\n" + "=" * 80)
            print(f"[ERROR] Failed bundle: {bundle_path}")
            if 'bundle_info' in locals():
                print(f"[ERROR] sample_id: {bundle_info.get('sample_id', 'N/A')}")
                print(f"[ERROR] N: {bundle_info.get('N', 'N/A')}")
            print(f"[ERROR] Exception type: {type(e).__name__}")
            print(f"[ERROR] Exception message: {e}")
            print("=" * 80)
            raise


    if len(all_rows) == 0:
        raise RuntimeError("No bundles were processed successfully.")

    df_all = pd.concat(all_rows, axis=0, ignore_index=True)


    csv_path = os.path.join(OUTPUT_DIR, "IME_Q_all_components.csv")
    df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")


    df_sample_mean = (
        df_all.groupby(["sample_id", "N"], as_index=False)[["Q_gt_true_kgh", "Q_kg_h"]]
        .mean()
        .rename(columns={
            "Q_gt_true_kgh": "sample_mean_Q_gt_kgh",
            "Q_kg_h": "sample_mean_Q_pred_kgh",
        })
    )
    sample_mean_csv = os.path.join(OUTPUT_DIR, "IME_Q_sample_mean.csv")
    df_sample_mean.to_csv(sample_mean_csv, index=False, encoding="utf-8-sig")


    txt_path = os.path.join(OUTPUT_DIR, "IME_Q_report.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=== IME Q summary by N ===\n")
        for N in range(2, 9):
            sub = df_all[df_all["N"] == N].copy()
            if len(sub) == 0:
                continue

            gt = sub["Q_gt_true_kgh"].to_numpy(dtype=float)
            pred = sub["Q_kg_h"].to_numpy(dtype=float)

            mae = float(np.mean(np.abs(pred - gt)))
            mape = float(np.mean(np.abs(pred - gt) / (gt + 1e-8)) * 100.0)
            bias = float(np.mean(pred - gt))

            f.write(
                f"N={N} | n={len(sub)} | "
                f"GT_mean={np.mean(gt):.3f} | Pred_mean={np.mean(pred):.3f} | "
                f"MAE={mae:.3f} | MAPE={mape:.3f}% | Bias={bias:.3f}\n"
            )

    n_error_table = compute_n_error_table(df_all, n_values=range(2, 9))
    n_error_csv_path = os.path.join(OUTPUT_DIR, "IME_Q_error_by_N.csv")
    n_error_table.to_csv(n_error_csv_path, index=False, encoding="utf-8-sig")
    print("Error statistics by source count:", n_error_csv_path)


    print("\nProcessing complete.")
    print("Detailed CSV:", csv_path)
    print("Per-sample mean CSV:", sample_mean_csv)
    print("Text report:", txt_path)
    print("Per-sample figures:", per_sample_dir)


if __name__ == "__main__":
    main()
