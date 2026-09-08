# Deep Learning-Based Separation and Quantification of Overlapping Methane Plumes

Main implementation for the paper **"Deep Learning-Based Separation and Quantification of Overlapping Methane Plumes from Satellite Observations"**.

The workflow consists of three standalone Python scripts: train a spatial-attention U-Net 3+ model, evaluate plume separation, and estimate individual source emission rates using integrated mass enhancement (IME).

## 1. Model training

**Script:** [Spatial-attention U-Net 3+ for methane plume separation(train).py](Spatial-attention%20U-Net%203%2B%20for%20methane%20plume%20separation%28train%29.py)

Builds and trains a U-Net 3+ model with spatial attention. The model predicts per-source contribution weights and mask logits. Hungarian matching aligns predicted slots with labeled sources; training uses contribution and mask losses.

- **Input:** Training and validation NPZ files containing combined methane enhancement maps and per-source contribution labels.
- **Output:** `best.pt`, per-epoch checkpoints (`epoch_XXX.pt`), and `exp_config.json` in `OUT_CKPT`.
- **Configure:** Set `DATA_ROOT` and `OUT_ROOT`, and replace the `your_output_path` subdirectory in `OUT_CKPT`. Leave `PRETRAIN_CKPT = None` for training from scratch, or provide a compatible checkpoint to initialize model weights. Optimizer state is not restored.

## 2. Plume separation and evaluation

**Script:** [overlap_plumes_separation_spatial_attention(test).py](overlap_plumes_separation_spatial_attention%28test%29.py)

Loads the best trained model and evaluates labeled test samples with 2-8 point sources. It matches predicted and reference components, computes mask-based separation metrics, and exports the information needed for subsequent IME estimation.

- **Input:** The labeled test set and the trained `best.pt` checkpoint.
- **Output:**
  - `ALL_GRIDS/N_*/`: Combined-plume, ground-truth, and predicted separation figures.
  - `RESULT_BUNDLES_FOR_IME/N_*/`: NPZ bundles containing separated components, source metadata, matching results, and original sample fields.
  - `metrics_all.csv`, `evaluation_report.txt`, and `evaluation_report.xlsx`: Detailed metrics and summaries by source count, including precision, recall, F1, IoU, and Dice.
- **Configure:** Set `DATA_ROOT`, `CKPT_PATH`, and `OUT_DIR`.

## 3. IME emission-rate estimation

**Script:** [Q_estimation_of_separated_plumes_IME.py](Q_estimation_of_separated_plumes_IME.py)

Extracts source-specific plume masks and estimates emission rates from the separated components using IME, effective wind speed, and plume length. It also compares estimated rates with reference rates derived from `scale_list`.

- **Input:** `RESULT_BUNDLES_FOR_IME` from the test script and matching test-set labels containing `u10_list` wind speeds.
- **Output:**
  - `PER_SAMPLE_MASK_PREVIEW/N_*/`: Plume-mask figures with IME and emission-rate annotations.
  - `PER_SAMPLE_Q_FIGS/N_*/`: Per-sample separation and quantification figures.
  - `IME_Q_all_components.csv`, `IME_Q_sample_mean.csv`, `IME_Q_error_by_N.csv`, and `IME_Q_report.txt`: Detailed estimates and statistical summaries.
- **Configure:** Set `INPUT_BUNDLE_DIR`, `TEST_LABEL_DIR`, and `OUTPUT_DIR`. This script does not generate `SUMMARY_PLOTS`.

The supplied configuration uses 60 m pixels, `Ueff = 0.4535 * U10 + 0.6541`, and `Q_GT = 3600 * scale_list` in kg/h. Plume components must be in kg/m², and wind speeds in m/s. Check that these settings match the dataset being evaluated.

## Data organization

```text
<DATA_ROOT>/
  training_dataset/
    combined_plumes/<sample_id>.npz
    labels/<sample_id>.npz
  validation_dataset/
    combined_plumes/<sample_id>.npz
    labels/<sample_id>.npz
  test_dataset/
    combined_plumes/<sample_id>.npz
    labels/<sample_id>.npz
```

Input and label files are paired by filename. Input files contain `combined` (or `combined_clean`) with shape `(H, W)`. Labels contain `alpha_list` with shape `(N, H, W)` and the source count `N`; spatial dimensions must match.

Test labels must also provide source coordinates and IDs: `src_xy_big_plot`, `src_xy_small_plot`, or `src_xy` with `src_id`, or `markers_xy` with `markers_id`. The supplied scripts use `plot_xy` coordinates. IME estimation additionally requires `scale_list` in the exported bundle and `u10_list` in the matching test label. Source coordinates, IDs, scaling factors, and wind speeds must follow the same source order.

## Installation and execution

Install the required packages in your Python environment:

```bash
pip install torch numpy scipy pandas matplotlib tqdm openpyxl
```

Use a PyTorch installation compatible with your hardware for GPU training. Replace the path placeholders in each script, then run:

```bash
python "Spatial-attention U-Net 3+ for methane plume separation(train).py"
python "overlap_plumes_separation_spatial_attention(test).py"
python "Q_estimation_of_separated_plumes_IME.py"
```

The test script's `CKPT_PATH` should point to the training output `best.pt`. The IME script's `INPUT_BUNDLE_DIR` should point to the test output `RESULT_BUNDLES_FOR_IME`.

Datasets and trained checkpoints are not included in this code upload. Dependency versions are not pinned, and a complete training-to-quantification run has not been validated as part of this upload.

The existing `overlap_plumes_emitbg_with_switches-V1.12_Lratio_Lmask-(train).py` is retained as an earlier experimental training script. Use the three scripts described above for this workflow.
