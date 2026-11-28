"""Controlled forward-only verification; no optimizer or dataset evaluation."""
import json
import re
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = Path(__file__).resolve().parent

from experiments.exp10rerun.model_exp10rerun import InfoGCN as BaselineInfoGCN
from research_audit.Test4_mask_correct_implementation.model_mask_correct import MaskedInfoGCN


def max_delta(a, b):
    return float((a - b).abs().max().item())


def compare_padded(model, x, mask, mode):
    model.train(mode == "train")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.eval()
    x0 = torch.where(mask[:, None, :, None], x, torch.zeros_like(x))
    x1 = torch.where(mask[:, None, :, None], x, torch.full_like(x, 31.0))
    with torch.no_grad():
        y0, y1 = model(x0, mask), model(x1, mask)
    return max_delta(y0, y1)


def copy_baseline_weights(baseline, corrected):
    """Copy matching EXP10RERUN topology weights and BN buffers into new names."""
    aliases = [
        (r"^stem\.0\.", "stem_conv."), (r"^stem\.1\.", "stem_bn."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.0\.", r"blocks.\1.temporal.pre_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.1\.", r"blocks.\1.temporal.bn_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d1\.3\.", r"blocks.\1.temporal.conv_d1."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.0\.", r"blocks.\1.temporal.pre_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.1\.", r"blocks.\1.temporal.bn_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_d2\.3\.", r"blocks.\1.temporal.conv_d2."),
        (r"^blocks\.(\d+)\.temporal\.branch_pool\.1\.", r"blocks.\1.temporal.pool_conv."),
        (r"^blocks\.(\d+)\.temporal\.branch_1x1\.", r"blocks.\1.temporal.point."),
        (r"^blocks\.(\d+)\.temporal\.bn\.", r"blocks.\1.temporal.out_bn."),
        (r"^blocks\.(\d+)\.residual\.0\.", r"blocks.\1.residual_conv."),
        (r"^blocks\.(\d+)\.residual\.1\.", r"blocks.\1.residual_bn."),
    ]
    source, target = baseline.state_dict(), corrected.state_dict()
    copied = []
    with torch.no_grad():
        for key, value in source.items():
            mapped = key
            for pattern, replacement in aliases:
                candidate, n = re.subn(pattern, replacement, key)
                if n:
                    mapped = candidate
                    break
            if mapped in target and target[mapped].shape == value.shape:
                target[mapped].copy_(value)
                copied.append((key, mapped))
        corrected.load_state_dict(target)
    return copied


def main():
    torch.manual_seed(20261127)
    torch.set_num_threads(1)
    kwargs = dict(in_channels=3, num_joints=50, num_classes=41, base_channels=8,
                  dropout=0.0, head_dropout=0.0)
    corrected = MaskedInfoGCN(**kwargs)
    baseline = BaselineInfoGCN(**kwargs)
    copied = copy_baseline_weights(baseline, corrected)
    results = {
        "scope": "synthetic forward-only checks; no training, tuning, canonical evaluation, or New40 access",
        "input_shape": [3, 3, 19, 50],
        "model_kwargs": kwargs,
        "cases": {},
    }
    x = torch.randn(3, 3, 19, 50)
    # Mixed prefix lengths plus an internal-gap pattern; all samples retain valid support.
    mask = torch.zeros(3, 19, dtype=torch.bool)
    mask[0, :19] = True
    mask[1, :12] = True
    mask[2, :8] = True
    mask[2, 10:17] = True

    for mode in ("eval", "train"):
        results["cases"][f"{mode}_finite_padding_invariance_corrected"] = compare_padded(corrected, x, mask, mode)
        results["cases"][f"{mode}_finite_padding_invariance_baseline"] = compare_padded(baseline, x, mask, mode)

    # Restore the same initial mapped running statistics after train-mode checks.
    copied = copy_baseline_weights(baseline, corrected)
    baseline.eval(); corrected.eval()
    with torch.no_grad():
        variable_base, variable_corrected = baseline(x, mask), corrected(x, mask)
        full_mask = torch.ones_like(mask)
        full_base, full_corrected = baseline(x, full_mask), corrected(x, full_mask)
    results["cases"]["paired_baseline_comparison"] = {
        "copied_state_tensor_pairs": len(copied),
        "variable_mask_logits_max_abs_delta": max_delta(variable_base, variable_corrected),
        "all_valid_logits_max_abs_delta": max_delta(full_base, full_corrected),
        "note": "Corresponding parameters and running-stat tensors were copied; variable-mask differences reflect changed mask handling.",
    }

    # Verify NaN and both signs of infinity are safely removed by where masking.
    corrected.eval()
    nonfinite_results = {}
    for value, key in ((float("nan"), "nan"), (float("inf"), "positive_inf"), (float("-inf"), "negative_inf")):
        special = torch.where(mask[:, None, :, None], x, torch.full_like(x, value))
        with torch.no_grad():
            special_logits = corrected(special, mask)
        nonfinite_results[key] = bool(torch.isfinite(special_logits).all())
    results["cases"]["masked_nonfinite_corrected_logits_finite"] = nonfinite_results

    # Directly verify temporal mask/shape propagation through stride 1,2,1,2,1 blocks.
    observed = []
    handles = []
    for i, block in enumerate(corrected.blocks):
        def hook(_module, _inputs, output, i=i):
            y, m = output
            observed.append({"block": i + 1, "output_t": int(y.shape[2]), "mask_t": int(m.shape[1]),
                             "mask_valid": int(m.sum().item()), "invalid_output_max": float(y.masked_select(~m[:, None, :, None]).abs().max().item())
                             if (~m).any() else 0.0})
        handles.append(block.register_forward_hook(hook))
    with torch.no_grad():
        logits = corrected(x, mask)
    for h in handles:
        h.remove()
    results["cases"]["temporal_propagation"] = observed
    results["cases"]["output_shape"] = list(logits.shape)

    # Multiple padded lengths and gap patterns in both modes.
    length_cases = {}
    for t, valid_len, gap in ((11, 6, False), (16, 9, False), (19, 13, True), (23, 18, False)):
        xx = torch.randn(2, 3, t, 50)
        mm = torch.zeros(2, t, dtype=torch.bool)
        mm[0, :valid_len] = True
        mm[1, :max(1, valid_len - 3)] = True
        if gap and t > 10:
            mm[1, 4:6] = False
        length_cases[f"T{t}_L{valid_len}_gap{gap}"] = {
            "eval_delta": compare_padded(corrected, xx, mm, "eval"),
            "train_delta": compare_padded(corrected, xx, mm, "train"),
            "expected_logits_shape": [2, 41],
            "actual_logits_shape": list(corrected(xx, mm).shape),
        }
    results["cases"]["length_and_pattern_cases"] = length_cases

    seq = torch.randn(1, 3, 9, 50)
    short = torch.zeros(1, 3, 13, 50); short[:, :, :9] = seq
    long = torch.full((1, 3, 21, 50), -17.0); long[:, :, :9] = seq
    short_mask = torch.zeros(1, 13, dtype=torch.bool); short_mask[:, :9] = True
    long_mask = torch.zeros(1, 21, dtype=torch.bool); long_mask[:, :9] = True
    corrected.eval()
    with torch.no_grad():
        results["cases"]["same_prefix_different_total_padding_eval_delta"] = max_delta(
            corrected(short, short_mask), corrected(long, long_mask))
    results["cases"]["state_mapping_pair_count"] = len(copied)

    (OUT / "verification_results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
