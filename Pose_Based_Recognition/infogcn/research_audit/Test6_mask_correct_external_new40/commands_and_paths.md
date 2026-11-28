# Test 6 commands and paths

Run from `Pose_Based_Recognition/infogcn/`:

```bash
python research_audit/Test6_mask_correct_external_new40/preflight.py
python research_audit/Test6_mask_correct_external_new40/evaluate_new40.py
```

The first command completed with `PASS` and `inference_performed: false`. The second ran the frozen evaluation after that pass. No checkpoint selection, tuning, or training occurred.

## Inputs

- New40 manifest and preprocessed NPZs: `../stgcn/testing/new40/processed/manifest.csv` and referenced files.
- Ordered class mapping: `../stgcn/datasets/processed/metadata_split.csv`.
- Mask-correct checkpoints: `research_audit/Test5_mask_correction_comparison/mask_correct/seed_{21,42,7,13}/best_model.pt`.
- EXP10RERUN baseline outputs: `testing/test2/results/` for seed 21 and `testing/test2_multiseed/results/exp10_seed{42,7,13}/` for the other seeds.
- Existing shared loader: `testing/test2/scripts/evaluate_exp10rerun_test2.py`.
- Mask-correct model definition: `research_audit/Test5_mask_correction_comparison/model_mask_correct.py`.

## Outputs

All new outputs are inside `research_audit/Test6_mask_correct_external_new40/`. Mask-correct per-video predictions were used in memory for metrics and paired disagreement checks, then discarded. Existing baseline prediction files were read-only inputs; no earlier result was overwritten.
