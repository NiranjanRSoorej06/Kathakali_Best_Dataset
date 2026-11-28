# EXP10RERUN Multi-Seed Canonical Summary

- **Completed Seeds Included**: `4` (21, 42, 7, 13)

## Per-Seed Results (Canonical 300/65/65 Split)
| Run Name | Seed | Best Epoch | Val Loss | Val Acc | Val Macro F1 | Test Loss | Test Acc | Test Macro F1 | Test Weighted F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| exp10rerun_seed21 | 21 | 71 | 0.0769 | 0.9538 | 0.9626 | 0.0088 | 1.0000 | 1.0000 | 1.0000 |
| exp10_seed42 | 42 | 58 | 0.0581 | 0.9692 | 0.9577 | 0.0127 | 1.0000 | 1.0000 | 1.0000 |
| exp10_seed7 | 7 | 43 | 0.1008 | 0.9538 | 0.9415 | 0.0374 | 0.9846 | 0.9675 | 0.9795 |
| exp10_seed13 | 13 | 47 | 0.0552 | 0.9846 | 0.9707 | 0.0349 | 0.9846 | 0.9675 | 0.9795 |

## Aggregate Mean ± Std (Sample `ddof=1`)
| Metric | Mean | Std (ddof=1) | Std (ddof=0) | Min | Max |
|---|---:|---:|---:|---:|---:|
| `best_epoch` | 54.7500 | 12.5532 | 10.8714 | 43.0000 | 71.0000 |
| `best_validation_loss` | 0.0728 | 0.0210 | 0.0182 | 0.0552 | 0.1008 |
| `best_validation_accuracy` | 0.9654 | 0.0147 | 0.0128 | 0.9538 | 0.9846 |
| `best_validation_macro_f1` | 0.9581 | 0.0123 | 0.0107 | 0.9415 | 0.9707 |
| `test_loss` | 0.0234 | 0.0148 | 0.0128 | 0.0088 | 0.0374 |
| `test_accuracy` | 0.9923 | 0.0089 | 0.0077 | 0.9846 | 1.0000 |
| `test_macro_precision` | 0.9817 | 0.0211 | 0.0183 | 0.9634 | 1.0000 |
| `test_macro_recall` | 0.9878 | 0.0141 | 0.0122 | 0.9756 | 1.0000 |
| `test_macro_f1` | 0.9837 | 0.0188 | 0.0163 | 0.9675 | 1.0000 |
| `test_weighted_f1` | 0.9897 | 0.0118 | 0.0103 | 0.9795 | 1.0000 |
