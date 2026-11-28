# Test 4 — Mask-Correct InfoGCN Variant

## Purpose and isolation

This directory contains a separate mask-correct InfoGCN implementation and forward-only synthetic verification. The source at `experiments/exp10rerun/model_exp10rerun.py`, its training/evaluation code, checkpoints, results, and dataset/splits were not changed. The new implementation reuses only EXP10RERUN's spatial graph-convolution class and read-only graph adjacency builder; all mask-aware normalization and temporal logic live in this directory.

No training, tuning, canonical test evaluation, or New40 access was performed. Synthetic comparisons are mechanism checks, not a performance evaluation.

## Implementation changes

`model_mask_correct.py` implements `MaskedInfoGCN`, `MaskedInfoGCNBlock`, `MaskedTemporalConv`, and `MaskedBatchNorm`.

1. **Safe invalid-value handling:** invalid input elements are replaced with `torch.where(mask, x, 0)` before normalization. This safely removes masked NaN and positive/negative infinity; multiplication by zero would not.
2. **Valid-only normalization:** masked BatchNorm calculates channel statistics over valid temporal locations only (and over joints for 2D feature maps). Invalid positions are zeroed after affine normalization. Running mean/variance update only when there are valid observations. A fully invalid sample is rejected at model input.
3. **Masking before temporal mixing:** each d1/d2 pointwise BN/ReLU output is remasked immediately before its temporal convolution. Temporal convolution therefore receives exact zeros for invalid positions. Branch-concatenation normalization is mask-aware, and invalid branch outputs are cleared afterward.
4. **Masked pooling branch:** invalid neighbors are treated as absent (`-inf` for max selection), then empty windows are replaced by zero. This avoids invalid values winning a max operation.
5. **Residual and downsampling propagation:** residual inputs are sanitized; strided output masks use the convolution center positions (`mask[:, ::stride]`) and are shape-checked. Residual normalization uses that output mask. The block result is zeroed at invalid positions.
6. **Attention and global pooling:** spatial attention uses the inherited valid-frame pooled query/key representation. Its value path is per-time; the result is mask-normalized and re-masked. Final pooling remains mask-weighted.
7. **Topology and head:** five blocks, channel progression, stride schedule, graph attention, multi-scale temporal branches, dropout, classifier and optional information-bottleneck head are retained. Masked normalization has its own state names and is not checkpoint-state-dict compatible without an explicit, reviewed conversion.

### Mask semantics

The mask may be a prefix or contain internal gaps. A downsampled position is valid when its center (the input position selected by the stride) is valid. Temporal convolutions can use other valid neighbors in their local receptive field; invalid values are always replaced with zero and cannot contribute. A gap is therefore a missing observation, not a hard barrier preventing information from valid frames on either side from being mixed. If applications require a hard barrier or only outputs with wholly valid receptive fields, the mask propagation policy must be changed explicitly. Prefix masks are the expected variable-length use case.

## Verification

Run from the repository root:

```bash
python research_audit/Test4_mask_correct_implementation/verify_mask_correct.py
```

The script uses small synthetic 41-class models, dropout disabled for deterministic comparisons, and no optimizer. It copies **207 corresponding parameter/running-stat tensors** from the baseline model before paired comparisons. Its output is in `verification_results.json`.

| Check | Result |
|---|---:|
| Corrected finite-padding change, eval mode | max logit delta `0.0` |
| Corrected finite-padding change, train mode | max logit delta `0.0` |
| Baseline finite-padding change, train mode (same synthetic case) | max logit delta `0.26098` |
| Masked NaN, +Inf, and -Inf inputs | finite corrected logits for all three |
| Same 9-frame prefix padded to total lengths 13 and 21 | eval max logit delta `0.0` |
| Four other lengths/patterns (T=11,16,19,23), train and eval | every finite-padding delta `0.0`; all output shapes correct |
| Internal gap case | finite-padding invariant in train and eval |
| Strided temporal outputs for T=19 | lengths `19,19,10,10,5`; mask lengths match; invalid output max `0.0` |
| Paired baseline vs corrected, all-valid mask | max logit delta `1.49e-8` |
| Paired baseline vs corrected, mixed validity mask | max logit delta `0.00149` |

For all-valid input, the paired output agreement confirms that the topology and mapped parameters reproduce baseline behavior up to floating-point tolerance. The small mixed-mask difference is expected from valid-only normalization and invalid-branch suppression. The baseline train-mode sensitivity reproduces the mechanism identified in Test 3.

## Limitations and follow-up considerations

- No training was performed. The corrected normalization changes training statistics and running-stat semantics for padded batches; any future training run must be a separately authorized experiment with its own checkpoint and logs.
- Existing EXP10RERUN checkpoints are not directly compatible as state dicts because masked normalization modules use different names/types. A conversion could initialize affine parameters and running statistics, but it cannot retroactively correct their training statistics; this variant should be trained from scratch if adopted.
- The synthetic checks use `base_channels=8`, 50 joints, three input channels and 41 output classes. They verify mechanics, not full-size speed, GPU numerics, gradient behavior, or predictive quality.
- The implementation's optional information-bottleneck head mirrors EXP10RERUN's sampling behavior, but this verification exercised the default deterministic non-IB head only.
- Training-mode BatchNorm still depends on which **valid** examples share a batch, as conventional batch normalization does; the guarantee is independence from invalid/padded values and locations.
- Center-sampled downsampling is appropriate for prefix validity and was tested on internal gaps. If an input has valid values only at positions skipped by every strided center, its final representation can be empty/zero. Production callers should use prefix masks or add an explicit policy for sparse masks (for example receptive-field-any propagation).
- FP16/bfloat16, distributed synchronization, gradient checks, empty post-downsample support, and large-batch performance were not tested.

## Artifacts

- `model_mask_correct.py` — isolated implementation.
- `verify_mask_correct.py` — synthetic forward-only tests and paired baseline comparison.
- `verification_results.json` — quantitative results.
- `README_test4.md` — implementation notes, test outcomes, and limitations.

The only generated artifact is the JSON result in this isolated directory. No existing experiment artifact was replaced.
