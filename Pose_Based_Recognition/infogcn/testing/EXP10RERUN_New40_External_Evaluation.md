# EXP10RERUN Seed 21 — New40 External Evaluation Report

## 1. Experiment Details

| Field | Value |
|---|---|
| **Experiment** | `exp10rerun` (isolated InfoGCN-inspired baseline rerun) |
| **Run Name** | `exp10rerun_seed21` |
| **Random Seed** | `21` |
| **Model Architecture** | `InfoGCN` (`infogcn/experiments/exp10rerun/model_exp10rerun.py`, `324,080` parameters) |
| **Checkpoint** | `infogcn/runs/exp10rerun/exp10rerun_seed21/best_model.pt` |
| **Checkpoint Epoch** | `71` (selected by lowest canonical validation loss) |
| **External Dataset** | New40 (`40` external Kathakali gesture videos) |
| **Evaluation Workspace** | `infogcn/testing/test2/` |
| **Execution Device** | `cuda:0` — `NVIDIA H100 PCIe MIG 2g.20gb` (CUDA `12.1`, GPU-enforced) |

---

## 2. Objective

Evaluate the generalization performance of the trained **EXP10RERUN Seed 21** `InfoGCN` checkpoint on the completely separate **40-video New40 external dataset**, without retraining, fine-tuning, threshold tuning, or leaking external data into training or model selection.

---

## 3. Model, Checkpoint, and GPU Execution Details

- **Model (`experiments/exp10rerun/model_exp10rerun.py`)**: Preserves the EXP10 InfoGCN-inspired architecture (`BatchNorm1d(150)` input normalization, `1×1` Conv stem + learnable `(1, 48, 1, 50)` joint identity embeddings, 5 `InfoGCNBlock`s with 3-head clip-pooled `SelfAttentionGraphConv` + 4-branch `MultiScaleTemporalConv`, mask-aware global average pooling, `Dropout(0.3)`, and a 41-class linear head; Information Bottleneck `use_ib=False`).
- **Training & Checkpoint Selection (`logs/exp10rerun/exp10rerun_seed21/best_validation_metrics.json`)**:
  - Trained on the canonical `300`-video training split for `100` epochs (`batch_size=32`, `lr=0.001`, `weight_decay=0.0001`, `ReduceLROnPlateau` scheduler, `seed=21`).
  - Checkpoint selected strictly by lowest canonical **validation loss** (`0.076947` at **Epoch 71**; canonical validation accuracy `95.38%`, validation macro F1 `0.9626`). Canonical test data (`65` videos) was never used for model selection.
- **GPU Execution Enforcement (`testing/test2/scripts/evaluate_exp10rerun_test2.py`)**:
  - Runtime verified `torch.cuda.is_available() == True` with `cuda:0` (`NVIDIA H100 PCIe MIG 2g.20gb`), explicitly forbidding CPU fallback.
  - Verified model parameters and evaluation tensors `(x, y, mask)` resided on CUDA during `model.eval()` and `torch.inference_mode()`.

---

## 4. Evaluation Protocol and External Dataset Description

- **Source Manifest**: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/stgcn/testing/new40/processed/manifest.csv` (consumed read-only, reusing the exact same preprocessed New40 `.npz` feature files used in prior ST-GCN and original EXP10 external evaluations).
- **Dataset & Preprocessing Format**:
  - **Samples**: `40` external videos (`1` video per represented class across `40` of the `41` canonical Kathakali mudra classes; `0` skipped or invalid).
  - **Representation**: `100` uniformly sampled frames × `50` DWPose joints (`[0..7] + [92..133]`) × `3` channels (neck-centered, per-video median-shoulder-width scaled `X`, `Y`, and raw DWPose SimCC confidence).
  - **Frame Masks**: `frame_valid` boolean masks (`(100,)`) were loaded from each `.npz` file and passed directly as `frame_mask` to `InfoGCN(x, mask)`, matching canonical EXP10/EXP10RERUN evaluation (`15/40` New40 videos have `90–99` valid frames; `25/40` have all `100` frames valid).
  - **Class Mapping**: Verified against canonical `metadata_split.csv` (`41` ordered classes, indices `0–40`).

---

## 5. External Evaluation Results Summary (`testing/test2`)

| Metric | Value |
|---|---:|
| **Evaluated Videos** | `40` |
| **Skipped / Invalid Videos** | `0` |
| **Correct Predictions** | `28` |
| **Incorrect Predictions** | `12` |
| **Overall Accuracy** | **70.00%** (`0.7000`) |
| **Macro F1 (all 41 canonical classes)** | **0.6016** (`0.601626`) |
| **Macro F1 (40 represented classes)** | **0.6167** (`0.616667`) |
| **Weighted F1** | **0.6167** (`0.616667`) |
| **Macro Precision (41 canonical / 40 represented)** | `0.5610` / `0.5750` |
| **Macro Recall (41 canonical / 40 represented)** | `0.6829` / `0.7000` |
| **Mean Cross-Entropy Loss** | `1.3684` |
| **Represented Classes** | `40` |
| **Missing Canonical Class** | `Urakkam` (`label_index = 37`) |

*(Observed from `testing/test2/results/predictions.csv`: `18/25` [`72.00%`] correct on clips with `100/100` valid frames and `10/15` [`66.67%`] correct on clips with `90–99` valid frames; per-class predictions never collapsed onto a single dominant attractor class, with at most `2` predictions assigned to any class.)*

---

## 6. 41-Class vs. Represented-Class Macro F1 (and Missing Class `Urakkam`)

- **Why `Urakkam` is absent**: The external New40 dataset contains `40` videos (`1` per gesture) out of the `41` canonical classes; no video of class `Urakkam` (`index 37`) is present in `new_test_videos` / `new40/processed/manifest.csv`. Additionally, `Urakkam` was never predicted for any of the `40` videos (`support = 0`, `TP = FP = FN = 0`).
- **41-Class Macro F1 (`0.6016`)**: Averages the per-class F1 across all `41` canonical classes (`labels = [0..40]` with `zero_division=0`), assigning `Urakkam` an F1 of `0.0000` in the denominator (`24.6667 / 41 = 0.6016`).
- **Represented-Class Macro F1 (`0.6167`)**: Averages only across the `40` classes actually present in the external test set (`24.6667 / 40 = 0.6167`). Because every represented class has `support = 1`, `weighted_f1` also equals `0.6167`.
- **Reporting convention**: Both values are reported for transparency; **41-class macro F1 (`0.6016`)** matches the canonical `classification_report` macro average across the model's full `41`-class output space.

---

## 7. Scientific Interpretation of Results

1. **Completed training substantially recovered EXP10 performance**: Whereas the original `EXP10` run (`best_model.pt` at Epoch `28`, terminated prematurely at Epoch `30`) achieved `57.50%` (`23/40`) on New40, completing the full `100`-epoch schedule with `seed=21` (`Epoch 71`, `val_loss=0.0769` vs. `0.1482`) improved external accuracy to `70.00%` (`28/40`) and reduced overconfident single-class attractor errors (e.g., original EXP10 predicted `Pushpam` `5` times, whereas EXP10RERUN Seed 21 predicted no class more than twice).
2. **Improved robustness on partial-valid-frame clips**: On the `15` New40 videos with `90–99` valid frames, accuracy rose from `33.33%` (`5/15`) in original EXP10 to `66.67%` (`10/15`) in EXP10RERUN Seed 21, while accuracy on the `25` fully valid clips stayed steady at `72.00%` (`18/25`).
3. **Persistent in-distribution vs. external generalization gap**: Despite achieving `95.38%` canonical validation accuracy (and `100.00%` on the `65`-video canonical test split), accuracy drops by `~25–30` percentage points on New40 (`70.00%`), showing that substantial sensitivity to unseen recording conditions and domain shift remains.

---

## 8. Comparison with Previously Reported New40 Results

| Model / Run | Checkpoint Epoch | Evaluated Videos | Correct / Incorrect | Accuracy | Macro F1 (41 Canonical) | Macro F1 (40 Represented) |
|---|---:|---:|---:|---:|---:|---:|
| **Original EXP10 (`exp10_infogcn_baseline`)** | 28 | 40 | 23 / 17 | 57.50% | 0.5122 | 0.5250 |
| **ST-GCN Exp3 (`three_partition_xy_conf_seed21`)** | 44 | 40 | 26 / 14 | 65.00% | 0.5650* | 0.5792 |
| **EXP10RERUN Seed 21 (`exp10rerun_seed21`)** | 71 | 40 | 28 / 12 | **70.00%** | **0.6016** | **0.6167** |

*\*Note: For ST-GCN Exp3 (`stgcn/testing/new40/results/`), `summary.json` recorded `0.5792` (the 40-represented-class macro F1), while its `classification_report.csv` recorded `0.5650` across all `41` canonical classes. Both columns are shown above for an exact apples-to-apples comparison.*

### Percentage-Point Improvements (EXP10RERUN Seed 21 vs. Baselines)

| Comparison | Accuracy Gain (pp) | Correct Videos Gain | Macro F1 (41-Class) Gain (pp) | Macro F1 (40-Represented) Gain (pp) |
|---|---:|---:|---:|---:|
| **EXP10RERUN Seed 21 vs. Original EXP10** | **+12.50 pp** | +5 (`28` vs. `23`) | **+8.94 pp** (`+0.0894`) | **+9.17 pp** (`+0.0917`) |
| **EXP10RERUN Seed 21 vs. ST-GCN Exp3 (Seed 21)** | **+5.00 pp** | +2 (`28` vs. `26`) | **+3.66 pp** (`+0.0366`) | **+3.75 pp** (`+0.0375`) |

---

## 9. Limitations

1. **Small external sample size (`n = 40`)**: Each percentage point corresponds to `0.4` videos (`1` video = `2.50` percentage points), so the `+5.00 pp` (`+2` videos) margin over ST-GCN Exp3 is within small-sample statistical variance.
2. **Single training seed (`seed=21`)**: Only one seed of `exp10rerun` was evaluated on New40; multi-seed mean and standard deviation are not yet established.
3. **One missing canonical class (`Urakkam`)**: Only `40/41` canonical classes are represented in New40 (`1` sample per represented class).
4. **Cross-domain shift**: Canonical `train`/`val`/`test` clips originate from stage-cropped recordings (`4` fixed aspect ratios), whereas New40 clips are uncropped `16:9` (`1920×1080`, `3840×2160`) with different subject scale (`mean reference_shoulder_width = 224.9 px` vs. `316.4 px` in canonical train) and `15/40` clips containing trailing invalid frames (`90–99` valid frames).
5. **No New40 tuning**: No architecture changes, thresholds, augmentations, or preprocessing adjustments were tuned on New40.

---

## 10. Output Artifact Paths (`testing/test2/results`)

- Metrics JSON: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/external_eval_metrics.json`
- Per-video predictions: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/predictions.csv`
- Classification report (CSV): `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/classification_report.csv`
- Classification report (text): `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/classification_report.txt`
- Confusion matrix (`41×41`): `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/confusion_matrix.csv`
- Skipped/invalid audit: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/skipped_invalid_videos.json`
- Auto-generated run report: `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn/testing/test2/results/external_eval_report.md`

---

## 11. Conclusion

On the strict, unmodified **40-video New40** external generalization test executed on GPU, **EXP10RERUN Seed 21** achieved **70.00% accuracy (`28/40`)** and **0.6016 macro F1 (41 canonical classes)** (`0.6167` across the `40` represented classes), outperforming both the prematurely stopped original EXP10 run (`+12.50 pp` accuracy, `+8.94 pp` macro F1) and the ST-GCN Exp3 Seed 21 baseline (`+5.00 pp` accuracy, `+3.66 pp` macro F1). While encouraging, these results come from a single seed on `40` external videos (`2.50 pp` per clip) and still show a `~25–30 pp` gap relative to canonical validation/test accuracy; multi-seed evaluation and domain-robustness improvements are needed before drawing broader generalization claims.
