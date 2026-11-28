# Test 2B: Conventional overfitting audit — EXP10RERUN

## Scope

Read-only analysis of the four existing training logs (seeds 21, 42, 7, 13) and their `best_validation_metrics.json` summaries. No checkpoints, per-video predictions, confusion matrices, or New40 files were inspected. Test metrics were not used in this audit. No model was trained or tuned, and no existing file was modified.

## Per-seed training and validation behavior

| Seed | Epochs | Best val-loss epoch | Best val loss | Val acc at best-loss epoch | Final train loss | Final val loss | Final minus best val loss | Final train acc | Final val acc | Train-val acc gap at selected epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 21 | 100 | 71 | 0.0769 | 95.38% | 0.0454 | 0.0837 | +0.0067 | 99.67% | 96.92% | +3.95 pp |
| 42 | 88 | 58 | 0.0581 | 96.92% | 0.0411 | 0.0768 | +0.0188 | 99.33% | 96.92% | +3.08 pp |
| 7 | 73 | 43 | 0.1008 | 95.38% | 0.0563 | 0.1858 | +0.0850 | 98.33% | 95.38% | +2.95 pp |
| 13 | 77 | 47 | 0.0552 | 98.46% | 0.0586 | 0.1105 | +0.0553 | 99.00% | 96.92% | +0.21 pp |

All four logs contain 73–100 epochs. The saved best epochs agree with the minimum validation-loss epoch in their logs: **True**. Checkpoint selection used `val_loss`; the reported validation accuracy at that epoch need not be the maximum validation accuracy epoch.

Across seeds, best validation loss averaged 0.0728 ± 0.0210 (sample SD); peak per-run validation accuracy averaged 98.46% ± 0.00% (sample SD). Best-loss epochs ranged from 43 to 71 (mean 54.75, sample SD 12.55).

## Trends and interpretation

- All runs make substantial early progress: training and validation losses fall from their high initial values, while both accuracies rise. Final training loss is below initial training loss in 4/4 runs. The curves show that the model fits the training split strongly and reaches high validation accuracy.
- Validation loss reaches its minimum between epochs 43 and 71. It is higher at the final epoch than at its minimum in 4/4 runs; mean final-minus-minimum validation loss is +0.0415. The final validation accuracy differs from accuracy at the selected checkpoint by -0.00 percentage points on average.
- Validation loss rises after its selected minimum in all four seeds. From the selected epoch to the last logged epoch, training loss falls for seeds 42, 7, and 13, but rises for seed 21. Validation accuracy remains within one clip of its selected-epoch value in every run. This pattern is **consistent with late-stage overfitting or increasing confidence on mistakes** in some runs, especially seeds 7 and 13, but it does not prove harmful overfitting: cross-entropy can rise when confidence worsens while classification accuracy stays stable, and validation estimates are noisy at 65 clips (one clip is about 1.54 percentage points).
- The logged train-versus-validation gap is descriptive, not a diagnosis. Training and validation metrics are measured on different sample sizes and may reflect different model modes or aggregation details. A gap alone does not establish overfitting. The more relevant evidence is the epoch-wise divergence, validation-loss rebound, validation-accuracy trend, and selection behavior together.
- No clear underfitting signal appears on the canonical split: final train accuracy is high and training loss falls substantially. This does not establish generalization to independent sessions or recording setups.
- The minimum-validation-loss checkpoint is selected before the final logged epoch in every seed (see `per_seed_overfitting_summary.csv`). The saved summaries show 1/4 runs reached their configured epoch count without early stop. The validation-loss selection rule therefore chooses the best observed validation-loss checkpoint rather than the final epoch; it does not remove uncertainty from a small validation set.

## Plots

- `loss_curves_by_seed.png`: train/validation loss trajectories, with log-scaled y-axis and selected epoch markers.
- `accuracy_curves_by_seed.png`: train/validation accuracy trajectories, with selected epoch markers.
- `four_seed_mean_std.png`: per-epoch mean and between-seed ±1 SD over the common epoch range (1–73); run-specific curves show each full logged duration.

## Cautious conclusion

EXP10RERUN shows stable early convergence across all four seeds and strong fit on the canonical training/validation splits. There is evidence of late validation-loss rebound after the selected epoch in the runs summarized above, which is compatible with mild late overfitting or confidence calibration drift. The observed train-validation gaps and this rebound do not alone prove conventional overfitting; the validation set is small, classification accuracy is coarse, and the audit is limited to the existing clip-level split. The most defensible conclusion is **good in-split convergence with some late-stage validation-loss deterioration, but inconclusive evidence of harmful overfitting**.

## Artifacts and provenance

This report and plots were generated from `logs/exp10rerun/<run_name>/training_log.csv` and `best_validation_metrics.json` for seeds 21, 42, 7, and 13. Supporting files are `all_seed_epoch_metrics.csv`, `aggregate_epoch_metrics.csv` (including `n_seeds` by epoch), `per_seed_overfitting_summary.csv`, and `summary.json`. The script `analyze_overfitting.py` is a reproducible read-only analysis tool.
