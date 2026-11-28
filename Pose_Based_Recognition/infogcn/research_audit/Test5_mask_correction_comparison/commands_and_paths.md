# Test 5 commands and isolated paths

## Preflight already completed

```bash
python research_audit/Test5_mask_correction_comparison/validate_setup.py
python research_audit/Test5_mask_correction_comparison/verify_seed_initialization.py
```

It passed before training. See `preflight.json` for the checked configuration hash, split counts, feature path presence, train/validation validity checks, baseline run identities/hashes, and same-seed initialization result.
The separate initialization check confirmed exact mapped initialization equality for all four seeds (207 tensors per seed).

## Planned execution

```bash
CUDA_VISIBLE_DEVICES=MIG-be0d4dc8-2244-5ed0-89ec-b674eacb6a9b python research_audit/Test5_mask_correction_comparison/run_comparison.py --all-seeds
python research_audit/Test5_mask_correction_comparison/summarize_comparison.py
```

The first command stages baseline logs/checkpoints by copy from the completed EXP10RERUN runs, trains the four Test 4 mask-correct runs (seeds 21, 42, 7, 13), and only after all four validation-selected checkpoints are fixed evaluates both variants on the canonical test split. The second command produces cross-seed summaries, plots, paired comparisons, and the report.

## Paths

- Experiment workspace: `research_audit/Test5_mask_correction_comparison/`
- Baseline copied logs/checkpoints: `baseline/seed_{21,42,7,13}/`
- Fresh mask-correct logs/checkpoints: `mask_correct/seed_{21,42,7,13}/`
- Canonical test evaluation: `evaluation/seed_{21,42,7,13}/`
- Cross-seed artifacts and report: workspace root (`*.csv`, `*.png`, `Test5_comparison_report.md`)
- Archived correction implementation: `model_mask_correct.py` (copied byte-for-byte from Test 4)
- Read-only shared dataset/split: `../stgcn/datasets/processed/metadata_split.csv` and referenced pose-feature files

The existing `experiments/exp10rerun/`, `runs/exp10rerun/`, `logs/exp10rerun/`, and dataset files are never write targets.
