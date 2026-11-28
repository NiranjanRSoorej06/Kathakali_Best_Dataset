# Experiment 10 Rerun (`exp10rerun`) — Isolated InfoGCN Baseline

## Isolation Structure

`exp10rerun` is a completely isolated rerun copy of `exp10`. **The original `experiments/exp10/`, `runs/exp10/`, and `logs/exp10/` files were not modified.**

```text
infogcn/
├── experiments/
│   ├── exp10/                     # Original EXP10 (untouched)
│   └── exp10rerun/                # Isolated EXP10RERUN implementation
│       ├── __init__.py
│       ├── model_exp10rerun.py
│       ├── train_exp10rerun.py
│       ├── evaluate_exp10rerun.py
│       └── README_exp10rerun.md
├── runs/
│   ├── exp10/                     # Original EXP10 checkpoints (untouched)
│   └── exp10rerun/<run_name>/     # EXP10RERUN checkpoints
└── logs/
    ├── exp10/                     # Original EXP10 logs (untouched)
    └── exp10rerun/<run_name>/     # EXP10RERUN logs & evaluation outputs
```

Shared read-only dependencies (identical to EXP10, consumed via `importlib` with `sys.dont_write_bytecode = True`):
- Graph topology: `../stgcn/shared/graph/graph.py` (`NUM_JOINTS = 50`, `build_adjacency`)
- Dataset loader & split: `../stgcn/experiments/common/dataset.py` and `../stgcn/datasets/processed/metadata_split.csv` (`300 train / 65 val / 65 test`, 41 classes)

## Architecture & Training Setup (Preserved from EXP10)

Implemented in `infogcn/experiments/exp10rerun/model_exp10rerun.py`:
- **Input normalization**: `BatchNorm1d(150)` over `(joint, channel)` for `(N, 3, 100, 50)` inputs.
- **Stem + Joint Embedding**: `1×1` Conv (`3 → 48`) + learnable `(1, 48, 1, 50)` joint identity embedding.
- **Spatial Attention (`SelfAttentionGraphConv`)**: 3-head clip-pooled attention combined with the 3 anatomical partitions (`self`, `centripetal`, `centrifugal`), learnable `(3, 50, 50)` `spatial_bias`, and `attn_scale` (init `0.5`).
- **Temporal Convolution (`MultiScaleTemporalConv`)**: 4 parallel branches (`1×1 + 3×1 d=1`, `1×1 + 3×1 d=2`, `MaxPool 3×1 + 1×1`, `1×1`).
- **Backbone & Head**: 5 `InfoGCNBlock`s (`48 → 48 → 96 (s=2) → 96 → 192 (s=2)`), mask-aware global average pooling, `Dropout(0.3)`, `Linear(192, 41)`. Information Bottleneck (`--use-ib`) is preserved and disabled by default.
- **Training Defaults**: `seed=42`, `epochs=100`, `batch_size=16`, `lr=0.001` (`AdamW`, `weight_decay=0.0005`), `ReduceLROnPlateau(factor=0.5, patience=6)`, `patience=20`, `selection_metric=val_loss`.

## Training CLI

Run from `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn`:

```bash
python -B -m experiments.exp10rerun.train_exp10rerun
```

All original CLI flags are supported (`--run-name`, `--seed`, `--epochs`, `--batch-size`, `--lr`, `--weight-decay`, `--patience`, `--num-workers`, `--base-channels`, `--dropout`, `--head-dropout`, `--label-smoothing`, `--selection-metric`, `--scheduler`, `--use-ib`, `--latent-dim`, `--lambda-kl`, `--lambda-mmd`, `--kl-warmup-epochs`).

## Output Locations

- **Checkpoints**: `infogcn/runs/exp10rerun/<run_name>/best_model.pt`
- **Training Logs & Val Metrics**: `infogcn/logs/exp10rerun/<run_name>/training_log.csv` and `best_validation_metrics.json`
- **Test Evaluation Outputs**: `infogcn/logs/exp10rerun/<run_name>/` (`test_metrics.json`, `classification_report.csv`, `test_classification_report.txt`, `confusion_matrix.csv`, `predictions.csv`)

## Evaluation CLI

```bash
python -B -m experiments.exp10rerun.evaluate_exp10rerun
```

Defaults to `--run-name exp10rerun_infogcn_baseline`, loading `runs/exp10rerun/<run_name>/best_model.pt` and writing outputs to `logs/exp10rerun/<run_name>/`.

## Verification Performed (No Full Training / External Eval Executed)

- Verified `experiments/exp10/`, `runs/exp10/`, and `logs/exp10/` SHA-256 checksums are identical before and after creation.
- Verified `experiments.exp10rerun.{model_exp10rerun, train_exp10rerun, evaluate_exp10rerun}` imports cleanly without touching `exp10`.
- Verified `--help` on both `train_exp10rerun` and `evaluate_exp10rerun`, path resolution (`runs/exp10rerun/`, `logs/exp10rerun/`), and `--run-name` path-traversal rejection.
- Verified minimal forward pass (`InfoGCN` on synthetic `(2, 3, 100, 50)` + `frame_mask`) and `KathakaliDataset` train/val/test split counts (`300 / 65 / 65`, `41` classes).

## Known Limitations

- **Harmless `sklearn` warning on `val`/`test` splits**: With `41` classes across `65` validation/test samples (`41 > 50%` of `65`), `sklearn` emits `UserWarning: The number of unique classes is greater than 50% of the number of samples...` during `accuracy_score`/`f1_score`/`classification_report`. Metric values are unaffected (`zero_division=0` is set), and warnings are intentionally not suppressed.
- **Architectural parity preserved**: Known EXP10 limitations (no data augmentation, `MultiScaleTemporalConv` `BatchNorm2d` bias before masking on invalid frames, `use_ib=False` default) are intentionally preserved in `exp10rerun` for an exact isolated baseline rerun.

## Recorded Run & Read-Only Audit Summary (`exp10rerun_seed21`)

- **Training Configuration**: `seed=21`, `epochs=100`, `batch_size=32`, `lr=0.001`, `weight_decay=0.0001`, `scheduler=plateau`, `selection_metric=val_loss`, `use_ib=false`, GPU `NVIDIA H100 PCIe MIG 2g.20gb`.
- **Selected Checkpoint**: `runs/exp10rerun/exp10rerun_seed21/best_model.pt` at **Epoch 71** (lowest canonical `val_loss=0.0769`, `val_accuracy=0.9538` [`95.38%`], `val_macro_f1=0.9626`).
- **Canonical 65-Video Test Results** (`logs/exp10rerun/exp10rerun_seed21/test_metrics.json`):
  - **Test Samples**: `65` (`41` classes)
  - **Accuracy**: `1.0000` (`100.00%`, `65/65`)
  - **Macro Precision / Recall / F1 / Weighted F1**: `1.0000` / `1.0000` / `1.0000` / `1.0000`
  - **Test Loss**: `0.0088` (`0.008805`)
- **Read-Only Integrity Audit Outcome**: **PASS (Credible on canonical in-distribution test split)** — Verified zero `train`/`val`/`test` overlap (`430/430` unique feature hashes, min test-to-train L2 distance `4.9567`), per-video canonical normalization, strict validation-loss checkpoint selection, `model.eval()` / `torch.no_grad()` inference, row-for-row prediction order match (`65/65`), exact confusion matrix / classification report consistency, and harmless `sklearn` `>50%` unique-class warning. Caveats: small canonical test size (`65` clips from the same 4 stage-cropped recording setups as train/val, so in-distribution `100%` does not imply equivalent cross-domain generalization), non-default run hyperparameters (`seed=21`, `batch_size=32`, `weight_decay=1e-4`), and `predictions.csv` omits `video_name`.

## Multi-Seed Reproducibility Setup (`Seeds 21, 42, 7, 13`)

To evaluate multi-seed stability under the exact `exp10rerun_seed21` protocol, `multiseed_config.json` defines four isolated runs under `experiments/exp10rerun/`:

| Run Name | Seed | Status | Checkpoints & Logs |
|---|---:|---|---|
| `exp10rerun_seed21` | `21` | Completed (reference — untouched) | `runs/exp10rerun/exp10rerun_seed21/`, `logs/exp10rerun/exp10rerun_seed21/` |
| `exp10_seed42` | `42` | Prepared (pending manual launch) | `runs/exp10rerun/exp10_seed42/`, `logs/exp10rerun/exp10_seed42/` |
| `exp10_seed7` | `7` | Prepared (pending manual launch) | `runs/exp10rerun/exp10_seed7/`, `logs/exp10rerun/exp10_seed7/` |
| `exp10_seed13` | `13` | Prepared (pending manual launch) | `runs/exp10rerun/exp10_seed13/`, `logs/exp10rerun/exp10_seed13/` |

### Shared Hyperparameters (Matching `exp10rerun_seed21`)
- `epochs=100`, `batch_size=32`, `lr=0.001`, `weight_decay=0.0001`, `patience=20`, `scheduler=plateau`, `selection_metric=val_loss`, `base_channels=48`, `dropout=0.15`, `head_dropout=0.3`, `label_smoothing=0.0`, `use_ib=False`, `latent_dim=128`.
- `train_exp10rerun.py` enforces `PYTHONHASHSEED`, `random`, `numpy`, `torch` CPU/CUDA seeds, `cudnn.deterministic=True`, `cudnn.benchmark=False`, a seeded `DataLoader` `torch.Generator`, and `worker_init_fn` (`seed_worker`) when `num_workers > 0`.

### Manual Launch Commands (Seeds 42, 7, 13)

Run from `/home/niranjanr/Kathakali_Best_Dataset/Pose_Based_Recognition/infogcn`:

```bash
# Seed 42
python -B -m experiments.exp10rerun.train_exp10rerun --run-name exp10_seed42 --seed 42 --epochs 100 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 20 --scheduler plateau --selection-metric val_loss
python -B -m experiments.exp10rerun.evaluate_exp10rerun --run-name exp10_seed42 --batch-size 32

# Seed 7
python -B -m experiments.exp10rerun.train_exp10rerun --run-name exp10_seed7 --seed 7 --epochs 100 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 20 --scheduler plateau --selection-metric val_loss
python -B -m experiments.exp10rerun.evaluate_exp10rerun --run-name exp10_seed7 --batch-size 32

# Seed 13
python -B -m experiments.exp10rerun.train_exp10rerun --run-name exp10_seed13 --seed 13 --epochs 100 --batch-size 32 --lr 0.001 --weight-decay 0.0001 --patience 20 --scheduler plateau --selection-metric val_loss
python -B -m experiments.exp10rerun.evaluate_exp10rerun --run-name exp10_seed13 --batch-size 32
```

### Consolidated Multi-Seed Summary Report

After all four seeds have completed training and canonical test evaluation, generate the consolidated mean ± std summary under `logs/exp10rerun/multiseed_summary/` (`multiseed_per_run.csv`, `multiseed_summary.json`, `multiseed_summary.md`):

```bash
python -B -m experiments.exp10rerun.summarize_multiseed_exp10rerun
```
