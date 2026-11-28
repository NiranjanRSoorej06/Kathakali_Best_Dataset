# Test 6 — Mask-Correct InfoGCN External New40 Evaluation

## Protocol and integrity

The four fixed Test 5 mask-correct checkpoints (seeds 21, 42, 7, 13) were evaluated on the same 40 manifest-indexed New40 videos as the corresponding existing EXP10RERUN seed baselines. Stage 1 strictly loaded all checkpoint state dictionaries and validated the class order, feature shapes, valid-frame masks, labels, baseline rows, and video identities before any model inference. The preflight passed. The frozen evaluation reused the existing New40 NPZs, class mapping, feature loader, and 100-frame mask handling. No model was retrained or selected using New40.

The baseline and mask-correct losses are comparable cross entropy (`use_ib=False`, no label smoothing). The paired prediction disagreement was **20 of 160 seed/video pairs**. Individual mask-correct video predictions were not saved.

## Mean ± standard deviation across seeds

Standard deviations below are sample standard deviations (`ddof=1`). Paired deltas are mask-correct minus baseline within each matching seed, then summarized across the four deltas.

| Metric | EXP10RERUN baseline | Mask-correct | Paired delta mean ± SD |
|---|---:|---:|---:|
| Accuracy | 0.6688 ± 0.0554 | 0.6563 ± 0.0427 | -0.0125 ± 0.0323 |
| Macro Precision (41) | 0.5452 ± 0.0485 | 0.5208 ± 0.0410 | -0.0244 ± 0.0156 |
| Macro Recall (41) | 0.6524 ± 0.0541 | 0.6402 ± 0.0417 | -0.0122 ± 0.0315 |
| Macro F1 (41) | 0.5787 ± 0.0531 | 0.5583 ± 0.0415 | -0.0203 ± 0.0228 |
| Represented-class Macro F1 (40) | 0.5931 ± 0.0544 | 0.5723 ± 0.0425 | -0.0208 ± 0.0233 |
| Weighted F1 | 0.5931 ± 0.0544 | 0.5723 ± 0.0425 | -0.0208 ± 0.0233 |
| Cross-entropy loss | 1.3090 ± 0.2015 | 1.3417 ± 0.1947 | +0.0328 ± 0.1028 |

The pooled overall metrics below combine all four seeds (160 video/seed predictions per variant); their macro scores are calculated from the pooled confusion counts and therefore are not the average of the four per-seed macro scores.

| Pooled metric | EXP10RERUN baseline | Mask-correct | Mask-correct minus baseline |
|---|---:|---:|---:|
| Accuracy | 0.6688 | 0.6563 | -0.0125 |
| Macro Precision (41) | 0.6220 | 0.5669 | -0.0551 |
| Macro Recall (41) | 0.6524 | 0.6402 | -0.0122 |
| Macro F1 (41) | 0.6015 | 0.5770 | -0.0245 |
| Represented-class Macro F1 (40) | 0.6165 | 0.5914 | -0.0251 |
| Weighted F1 | 0.6165 | 0.5914 | -0.0251 |
| Cross-entropy loss | 1.3090 | 1.3417 | +0.0328 |

## Per-seed results

| Seed | Baseline accuracy | Mask-correct accuracy | Δ accuracy | Baseline macro F1 (41) | Mask-correct macro F1 (41) | Δ macro F1 | Disagreements / 40 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 0.7000 | 0.6500 | -0.0500 | 0.6016 | 0.5528 | -0.0488 | 6 |
| 42 | 0.6000 | 0.6000 | 0.0000 | 0.5098 | 0.5057 | -0.0041 | 9 |
| 7 | 0.6500 | 0.6750 | +0.0250 | 0.5691 | 0.5691 | 0.0000 | 4 |
| 13 | 0.7250 | 0.7000 | -0.0250 | 0.6341 | 0.6057 | -0.0285 | 1 |

## Class coverage and interpretation

All 41 canonical classes are included in the 41-class metrics. New40 represents 40 classes; **Urakkam has zero support** and is explicitly flagged in the per-class CSVs. The pooled per-class metrics and confusion matrices aggregate all 160 seed/video predictions for each variant. With only 40 external videos and one sample per represented class per seed, these results have limited precision. The observed four-seed averages are lower for mask-correct on the listed F1 and accuracy metrics, but the small paired sample does not establish a general performance difference.

## Artifacts

- `preflight.json`: Stage 1 checks, software environment, and checkpoint compatibility.
- `integrity_report.json`: manifest/checkpoint hashes, paired alignment status, aggregate disagreement count.
- `per_seed_metrics.csv`, `per_seed/seed_*_mask_correct_metrics.json`: seed-level metrics.
- `aggregate_summary.csv`: means and standard deviations for each variant and paired deltas, plus pooled overall rows.
- `overall_metrics_pooled_160.csv`: pooled metrics over all 160 seed/video pairs for each variant.
- `paired_comparison.csv`: per-seed paired metrics and disagreement counts.
- `per_class_metrics_*_pooled.csv`, `confusion_matrix_*_pooled.csv`: pooled class metrics and confusion matrices.
- `per_seed/seed_*_{baseline,mask_correct}_confusion_matrix.csv`: individual seed confusion matrices.
- `configuration.json`, `commands_and_paths.md`, `preflight.py`, `evaluate_new40.py`: protocol, exact commands, and scripts.
- `environment.json`: Python, PyTorch, CUDA, and GPU versions used.
