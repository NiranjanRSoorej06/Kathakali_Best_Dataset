"""Verify same-seed mapped initialization for all comparison seeds; no data access."""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from experiments.exp10rerun.model_exp10rerun import InfoGCN
from research_audit.Test5_mask_correction_comparison.model_mask_correct import MaskedInfoGCN
from research_audit.Test5_mask_correction_comparison.validate_setup import aliases

cfg = dict(num_classes=41, in_channels=3, num_joints=50, base_channels=48,
           dropout=0.15, head_dropout=0.3, use_ib=False, latent_dim=128)
results = {}
for seed in (21, 42, 7, 13):
    torch.manual_seed(seed); base = InfoGCN(**cfg)
    torch.manual_seed(seed); corrected = MaskedInfoGCN(**cfg)
    a, b = base.state_dict(), corrected.state_dict()
    matched, mismatched = 0, []
    for key, value in a.items():
        mapped = aliases(key)
        if mapped in b and b[mapped].shape == value.shape:
            matched += 1
            if not torch.equal(value, b[mapped]): mismatched.append([key, mapped])
        else:
            mismatched.append([key, mapped, "unmapped"])
    results[str(seed)] = {"mapped_tensors": matched, "identical": not mismatched, "mismatches": mismatched[:8]}
assert all(v["identical"] and v["mapped_tensors"] == 207 for v in results.values()), results
(OUT / "initialization_parity.json").write_text(json.dumps(results, indent=2) + "\n")
print(json.dumps(results, indent=2))
