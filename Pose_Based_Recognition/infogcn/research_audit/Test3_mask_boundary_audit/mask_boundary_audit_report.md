# Test 3: Mask-Boundary and Padding Sensitivity Audit

## Scope and method

This read-only audit traced mask propagation through the isolated EXP10RERUN model and ran controlled forward passes using the seed 21 best checkpoint. The checkpoint was loaded read-only. No optimizer, parameter update, training run, or dataset inference was performed. Synthetic input had two samples with 100 temporal positions and valid-prefix lengths 83 and 97. A second forward-only check captured the outputs of temporal-branch BatchNorm/ReLU stages and then intervened in memory to zero those invalid positions before the temporal convolutions. This estimates whether those activations can affect valid pooled outputs; it is not a retraining or a modified model artifact.

The canonical split metadata was read only for valid-frame counts. No New40 files or per-video predictions were accessed.

## Findings

### Mask path through the model

| Stage | Current behavior | Padding sensitivity |
|---|---|---|
| Input and `data_bn` | `(N,C,T,V)` is rearranged to `(N,V*C,T)` and passed through `BatchNorm1d`; the frame mask is applied only after this normalization. | In training mode, invalid raw values enter the batch/time statistics and change valid normalized frames. In eval mode, fixed running statistics transform each value independently, after which finite invalid values are masked. Multiplying by zero does not remove NaNs or infinities. |
| Stem | The masked tensor goes through 1×1 convolution, `BatchNorm2d`, ReLU and joint embedding; the mask is applied again after the stem. | The stem BN sees invalid zero positions during training, so its statistics include padding. The second mask prevents finite stem activations at invalid frames from entering the next stage. |
| Spatial graph attention | Query/key attention pools only masked valid frames. Value projections and graph aggregation are per-frame in time. Spatial BN/ReLU follows and is masked before the temporal module. | Masked temporal pooling is appropriate. The spatial BN sees invalid positions during training; at evaluation its affine/running-stat transform can produce nonzero invalid activations, but the explicit mask immediately before the temporal module removes those values. |
| Temporal branches | Dilation-1 and dilation-2 branches each apply pointwise convolution, BN and ReLU, then temporal convolution. The pool and 1×1 branches run in parallel. There is no mask between the d1/d2 BN/ReLU and temporal convolution. Branch concatenation is followed by BN and dropout. | This is the main demonstrated evaluation-time boundary path: BN/ReLU can create nonzero invalid hidden activations from zero masked inputs, and d1/d2 temporal convolutions can mix those values into valid outputs within their receptive fields. During training, branch and concatenation BNs also include invalid positions in their statistics. |
| Residual and block output | The block uses `output_mask = mask[:, ::stride]`, masks the residual path, combines spatial-temporal output and residual, applies ReLU, then masks the result. | Explicit output masking prevents invalid block outputs from persisting to later blocks or final pooling. It cannot undo contamination already mixed into valid boundary outputs by temporal convolutions. For prefix masks, the stride slicing selects the expected sampled valid prefix. |
| Final pooling | The final representation is a mask-weighted temporal sum divided by the valid count. | Invalid output positions are excluded. This does not remove contamination already present in valid positions. |

Relevant implementation locations: `model_exp10rerun.py` lines 35–99 (masked spatial pooling and graph projection), 102–168 (temporal branches), 171–236 (block and stride-mask propagation), and 360–399 (input normalization, masking, and final pooling). The dataset loader at `../stgcn/experiments/common/dataset.py` reads `frame_valid` and expects a 100-position mask.

### Analytical boundary extent

The model has five temporal blocks with strides `[1, 1, 2, 1, 2]`. The widest temporal branch uses kernel size 3 with dilation 2, giving an effective width of five positions in each block. Propagating that support through the strides gives a theoretical maximum input receptive-field width of 29 frames for a final temporal output. Thus a valid output near a padding boundary can depend on hidden positions generated at invalid temporal locations, within the model's accumulated temporal support. This is a structural upper bound; the impact depends on learned BN values, convolution weights, sequence length, and boundary position.

### Controlled synthetic results

The forward-only checks are recorded in [`controlled_mask_results.json`](controlled_mask_results.json), and plotted in [`mask_sensitivity_checks.png`](mask_sensitivity_checks.png). The reproducible audit script is [`test3_mask_sensitivity.py`](test3_mask_sensitivity.py).

- **Finite raw padded values in eval mode:** replacing invalid values with `+25` or `-25` changed neither logits nor pooled representations in this check (maximum absolute difference `0.0`). The initial eval normalization is pointwise with fixed statistics and is followed by masking.
- **NaN padded values:** invalid NaNs produced non-finite logits. The implementation uses multiplication by the mask, and IEEE NaN multiplied by zero remains NaN. Invalid inputs therefore need to be finite, or be sanitized with selection rather than multiplication.
- **Finite raw padded values in train mode:** changing only invalid positions from zero to `+25` changed valid `data_bn` outputs by as much as `4.7193`, logits by `1.8765`, and changed at least one predicted class. This is a forward-only BatchNorm-statistics effect; no optimizer step was made. It applies when the model is in training mode and variable-length padded examples are present.
- **Internal invalid branch activations in eval mode:** the captured d1/d2 BN+ReLU values at masked positions were nonzero across all five blocks, with maximum magnitude `1.4643`. Zeroing these values immediately before their temporal convolutions changed logits by up to `0.02634` and pooled representations by up to `0.03244`. This controlled counterfactual demonstrates that internal invalid branch activations can influence valid pooled outputs for the sampled sequence/checkpoint. It does not establish a material accuracy effect or quantify performance on any dataset.
- **Canonical lengths:** metadata showed all 430 canonical clips have exactly 100 valid frames (minimum and maximum both 100). There is no padding boundary in those examples under the current 100-frame representation, so the specific variable-length boundary pathway is dormant for canonical inputs. Training BatchNorm still sees no invalid temporal positions in that dataset. The NaN caveat remains a general input robustness issue.

The experiment used one checkpoint and synthetic sequences, so these demonstrations are mechanistic evidence, not a dataset-level estimate. The raw finite-padding eval invariance result does not imply full invariance to a change in sequence length/mask: the masked pooled attention and final pooled representation can change when the valid sequence itself changes.

## Vulnerabilities versus measured effects

**Experimentally demonstrated in the controlled check:** training-mode input BatchNorm depends on invalid raw values; eval-mode d1/d2 branch activations at invalid locations can be nonzero and their counterfactual removal changes valid outputs; masked NaNs can propagate; finite raw padding values were invariant in eval mode for the tested input.

**Structural risks not quantified on real variable-length clips:** padding positions alter training-mode statistics in every BatchNorm whose temporal/spatial extent includes them (input, stem, spatial stage, temporal branches and branch-concatenation stage). The d1/d2 branches have cross-boundary temporal support. Exact affected frame counts and predictive significance depend on each mask and learned parameters. The pool branch uses a local max pool and has no subsequent temporal convolution within that branch; its output at invalid centers is later masked at the block boundary, though its values still participate in branch BN statistics.

## Relevance and limits

For the canonical Kathakali 430-video dataset, all clips are represented by 100 valid frames, so there are no masked temporal positions to create these padding-boundary effects. The current implementation's mask still protects pooling and supports other masks, but variable-length external clips padded to a common length can encounter both training-statistics dependence and evaluation-time boundary contamination. If external clips are passed only in eval mode, changing the *values* of finite padded frames is suppressed at the input; nevertheless, learned branch BN outputs can create invalid hidden values that reach adjacent valid frames through temporal convolutions. How often this is consequential depends on the external clipping/padding procedure.

The synthetic audit does not evaluate actual external clips, determine an acceptable valid-output boundary policy, or test all seeds/checkpoints. No claims about recognition accuracy or generalization follow from these checks.

## Proposed mask-correct design (not implemented)

1. Sanitize invalid input positions with `where(mask, x, 0)` before any normalization, so masked NaN/Inf values cannot propagate.
2. Replace temporal `BatchNorm1d/2d` behavior with mask-aware statistics over valid temporal positions (and the intended joint/channel axes), or use a normalization that does not aggregate across padded time. Apply masks consistently while computing running statistics.
3. In each temporal branch, reapply the aligned mask after pointwise convolution/normalization/activation and immediately before every temporal convolution. Mask branch outputs before concatenation and after normalization/dropout; mask again after residual addition and activation.
4. Define temporal support semantics explicitly: either only retain outputs whose full receptive field is valid, or accept boundary outputs with a documented padding rule. Derive output masks from actual convolution geometry, including stride, padding and dilation, rather than relying only on `mask[:, ::stride]` if masks can contain internal gaps or non-prefix validity.
5. Keep graph-attention pooling and final temporal pooling mask-weighted, with denominators clamped for empty masks and explicit rejection of all-invalid examples.
6. Validate with finite-value invariance tests in train and eval modes, masked NaN tests, prefix lengths covering short and odd/even lengths, internal mask gaps if supported, and stride-boundary cases. Compare predictions only after implementation is separately approved; this report makes no code changes.

## Commands and artifacts

Inspected model and loader source, read the seed 21 best checkpoint without writing it, and read canonical split metadata for frame-valid counts. Ran:

```text
python research_audit/Test3_mask_boundary_audit/test3_mask_sensitivity.py
```

The script completed and generated:

- `controlled_mask_results.json` — numerical findings and scope notes.
- `mask_sensitivity_checks.png` — summary plot.
- `test3_mask_sensitivity.py` — controlled forward-only check.
- `mask_boundary_audit_report.md` — this report.

The first script invocation failed because the repository root was absent from the Python import path (`ModuleNotFoundError: experiments`); the script was adjusted to add its own repository root and rerun successfully. No existing model, checkpoint, dataset, split, or experiment file was edited. No training or New40 inspection was performed.
