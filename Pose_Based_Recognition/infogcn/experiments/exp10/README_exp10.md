# Experiment 10 — InfoGCN Baseline for Kathakali

## Location & Isolation

All EXP10 code, documentation, checkpoints (`runs/exp10/`), and logs (`logs/exp10/`) reside exclusively inside `Pose_Based_Recognition/infogcn/`. Existing `stgcn/` and `dwpose/` directories (datasets, preprocessing, splits, graph definitions, and experiments 1–9) are consumed **read-only** via `importlib` with `sys.dont_write_bytecode = True` so no files or bytecode caches are written outside `infogcn/`.

## Fixed Experimental Setup

- **Dataset**: 430 videos, 41 classes (`../stgcn/datasets/processed/normalized_50J_100F`)
- **Split**: `../stgcn/datasets/processed/metadata_split.csv` — `300 train / 65 validation / 65 test`
- **Input tensor**: `(N, 3, 100, 50)` (neck-centered shoulder-scaled X/Y + raw DWPose SimCC confidence)
- **Graph**: 50 joints, 3 anatomical spatial partitions (`self`, `centripetal`, `centrifugal` around neck `1`) loaded read-only from `../stgcn/shared/graph/graph.py`
- **Augmentation**: None (initial baseline; no horizontal flipping)

## Architecture & Adaptations from Original InfoGCN

Implemented in `infogcn/experiments/exp10/model_exp10.py` using pure PyTorch (no `einops` or `timm`):

1. **Input Normalization (`BatchNorm1d(150)`)**: Standardizes `(joint, channel)` statistics across `(N, T)` to reconcile neck-centered XY magnitudes (`[-2.98, 2.68]`) with raw SimCC confidence (`[0.09, 1.08]`), matching `stgcn/experiments/exp3/model_exp3.py`.
2. **Stem + Learnable Joint Embeddings**: `1×1` Conv stem (`3 → 48`) plus a learnable per-joint identity embedding `(1, 48, 1, 50)` so self-attention distinguishes left/right finger joints after neck-centering.
3. **Self-Attention Graph Convolution (`SelfAttentionGraphConv`)**:
   - Uses 3 heads aligned with the 3 anatomical partitions (`A[0..2]`).
   - Computes **clip-pooled** (`(N, H, 50, 50)`) scaled dot-product attention with a learnable `(3, 50, 50)` relative spatial bias and learnable residual attention scales `attn_scale` (initialized to `0.5`) rather than per-frame `(N*T, H, 50, 50)` attention, lowering variance on 300 training clips.
4. **Multi-Scale Temporal Convolution (`MultiScaleTemporalConv`)**:
   - Four parallel branches (`1×1 + 3×1 d=1`, `1×1 + 3×1 d=2`, `MaxPool 3×1 + 1×1`, and `1×1`) concatenated along channels, replacing the single `(9, 1)` ST-GCN kernel.
5. **Compact Capacity & Temporal Downsampling**:
   - Five `InfoGCNBlock`s (`48 → 48 → 96 (stride 2) → 96 → 192 (stride 2)`), reducing `T=100 → 50 → 25` with residual shortcuts and frame-mask propagation.
6. **Global Pooling & Classification Head**:
   - Mask-aware global average pooling over `(T', V)` → `Dropout(0.3)` → `Linear(192, 41)`.
   - Information Bottleneck (`use_ib`, reparameterization + KL + RBF-MMD in `InfoGCNLoss`) is implemented and **disabled by default** (`--use-ib` enables it for future ablations).

## Running EXP10

Run from `~/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn`:

```bash
python -m experiments.exp10.train_exp10
python -m experiments.exp10.evaluate_exp10
```

Optional flags (defaults shown):

```bash
python -m experiments.exp10.train_exp10 \
  --run-name exp10_infogcn_baseline \
  --seed 42 \
  --epochs 100 \
  --batch-size 16 \
  --lr 0.001 \
  --weight-decay 0.0005 \
  --patience 20 \
  --base-channels 48 \
  --scheduler plateau \
  --selection-metric val_loss
```

Outputs are saved strictly under `infogcn/`:
- Checkpoints: `infogcn/runs/exp10/<run_name>/best_model.pt`
- Logs & Reports: `infogcn/logs/exp10/<run_name>/`
