# Test 5 — Controlled EXP10RERUN vs Mask-Correct Comparison

## Design and isolation

The canonical 430-video split was retained unchanged (300 train / 65 validation / 65 test; 41 classes). Baseline training logs and selected checkpoints were copied from the four already-completed EXP10RERUN runs whose configuration matched the pre-registered Test 5 settings; baseline runs were not repeated. The original experiment files were not modified. Four fresh mask-correct runs used the Test 4 implementation and the same seeds, batch order policy, optimizer, scheduler, epoch cap, early stopping, selection rule, and preprocessing. No augmentation was used. Checkpoints were selected by validation loss only.

The configuration and isolation preflight passed before the fresh runs: all 430 feature paths existed; train/validation split contents were finite with 100 valid frames per example. Afterward, a separate initialization check confirmed all 207 mapped parameter/buffer tensors were identical for each of the four seeds. Test feature contents were not opened during preflight. The canonical test split was loaded only after all four mask-correct checkpoints were fixed. New40 remained sealed. Both baseline and corrected evaluation used the same H100 PCIe MIG 2g.20gb device class; exact MIG instance identity for historical baseline runs was not recorded.

## Results

Cross-seed mean ± sample SD (four seeds) is in `cross_seed_summary.csv`; seed-level scores and paired changes are in `per_seed_test_metrics.csv`, `paired_metric_deltas.csv`, and `paired_seed_comparisons.csv`. Per-class scores are in `per_class_cross_seed_summary.csv`; per-seed and aggregate confusion matrices are provided as CSV files. Learning curves and test-score plots are included.

| Metric | Baseline mean ± SD | Mask-correct mean ± SD | Paired change mean ± SD |
|---|---:|---:|---:|
| test_loss | 0.0234 ± 0.0148 | 0.0260 ± 0.0209 | +0.0026 ± 0.0091 |
| accuracy | 0.9923 ± 0.0089 | 0.9923 ± 0.0089 | +0.0000 ± 0.0000 |
| macro_f1 | 0.9837 ± 0.0188 | 0.9837 ± 0.0188 | +0.0000 ± 0.0000 |
| weighted_f1 | 0.9897 ± 0.0118 | 0.9897 ± 0.0118 | +0.0000 ± 0.0000 |

Validation-selected checkpoint summary:

| Variant | Best validation loss, mean ± SD | Best validation accuracy, mean ± SD | Best epoch, mean ± SD |
|---|---:|---:|---:|
| baseline | 0.0728 ± 0.0210 | 0.9654 ± 0.0147 | 54.7500 ± 12.5532 |
| mask_correct | 0.0765 ± 0.0253 | 0.9769 ± 0.0089 | 58.0000 ± 16.8721 |

## Interpretation and limits

Interpret differences descriptively. Four seeds and 65 test clips provide limited uncertainty resolution; test examples are repeated across seeds and the same split is shared. Exact paired McNemar binomial p-values are supplied per seed but have low power and are not a substitute for independent recordings or group-held-out evaluation. Per-class metrics are particularly variable where test support is small.

The 65-video test set covers 41 classes; per-class support ranges 1–2 (median 2.0); 17 classes have one example and 0 have none. See `test_class_support.csv`. Across the four seed-paired canonical test prediction files, the models disagreed on 0 of 260 seed-by-clip predictions; see `paired_seed_comparisons.csv` for per-seed counts. Equal labels and aggregate scores on this split do not establish equal confidence calibration or cross-session robustness.

All canonical clips had 100 valid frames. Therefore the mask correction is not expected to materially change canonical computations: there are no invalid temporal positions for it to exclude. The paired predictions were identical for every seed, while test loss varied slightly; validation curves also closely overlap. Differences in loss and selected epoch may reflect floating-point/reduction order or training dynamics of the custom valid-only BatchNorm implementation, and should be interpreted against seed variation. This experiment does not establish behavior on variable-length external clips or improve source/session independence.

The first run attempt failed on the default MIG instance with an NVML/CUDA caching-allocator internal assertion during the first training forward, before any checkpoint was written. That incomplete log is preserved in `failed_attempts/`. The run then completed on a less occupied MIG instance of the same device class with all hyperparameters unchanged.

The mask-correct model has a distinct state-dict layout for normalization; old checkpoints are not directly interchangeable. No New40 evaluation or per-video inspection was done.

## Reproducibility

Configuration: `comparison_config.json`; preflight record: `preflight.json`; initialization check: `initialization_parity.json`; commands: `commands_and_paths.md`; per-run logs/checkpoints: `baseline/` and `mask_correct/`; evaluation artifacts: `evaluation/`. The Test 4 model source is copied byte-for-byte into `model_mask_correct.py` here; baseline InfoGCN and the shared dataset loader are imported read-only.
