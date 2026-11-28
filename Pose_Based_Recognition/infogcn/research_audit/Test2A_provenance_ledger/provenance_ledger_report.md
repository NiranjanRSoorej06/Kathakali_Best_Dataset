# Test 2A: Provenance ledger and grouping protocol

## Scope

This read-only assessment structures the canonical 430-video dataset metadata and carries forward Test 1 candidate links. It does not modify the existing split, datasets, checkpoints, experiment artifacts, or results. No model was trained or tuned, and no New40 per-video prediction or error file was inspected.

## Ledger coverage

`provenance_ledger.csv` contains one row for each of the 430 canonical clips across 41 classes. It records stable clip ID, class/label, current split, absolute source path and directory, source-bundle tag, source dimensions, decoded FPS/frame count/duration, SHA-256, feature path and availability, and valid-frame counts. It also records candidate relationship IDs, candidate counterpart clips, Test 1 screening component IDs, confidence, and evidence notes.

Verified original source-video ID, recording/session ID, and dancer ID are `UNKNOWN` for every clip. The `source_bundle` field is organizational metadata only. It is not promoted to a session identifier. All 430 source and feature files matched the Test 1 manifest; no missing-file errors were recorded.

## Relationship evidence carried forward

| Status | Number of direct candidate links | Assessment |
|---|---:|---|
| Confirmed | 0 | No byte-identical SHA-256 files and no verified source/session IDs or crop-interval overlap records. |
| Strong candidate | 1 | Kashtam_8 (val) vs Kashtam_11 (train): same class directory, exactly matching frame/FPS/duration metadata, similar sampled progression. Hashes differ; related session/take is plausible but unconfirmed. |
| Weak candidate | 107 | The other Test 1 perceptual/pose candidate edges plus sparse visual candidates. Similarity may reflect repeated choreography or a common setup. |
| Unknown provenance | 430 clips | Source/session/dancer identity remains unknown for all clips; no candidate list exhausts possible cross-class session links. |

The 108 direct candidate edges consist of the 105 automated Test 1 possible pairs and three additional manually reviewed possible pairs. Two of the five visual spot-check pairs already appeared in the automated set. Test 1 also reports 37 transitive candidate components spanning splits with 167 clip members. Those component memberships are not group assignments: transitive similarity can over-join unrelated takes. `candidate_relationships.csv` preserves direct edges; `candidate_related_groups.csv` from Test 1 preserves its separate screening components.

The sparse visual review is copied here as `test1_manual_visual_review.csv` and `test1_selected_pairs_visual_spotcheck.png`. It includes five pairs: Chundu_1/3, Chundu_3/5, Kashtam_8/11, Albutham_4/5, and Thaamara_3/9. These remain candidates, not confirmed source overlaps. The Test 1 transitive component map is copied as `test1_candidate_related_groups.csv`.

## Split and group feasibility

The existing split is 300 train / 65 validation / 65 test. All 41 classes occur in all splits, but per-class support is 7–8 train clips and only 1–2 clips in validation and test. Each class has 10–11 clips overall. These are clip counts; verified independent group counts are currently unknown.

Therefore, a defensible group-aware 41-class evaluation is **not currently ready to run**. At least three independent groups per class are necessary (not sufficient) for a group-separated train/validation/test arrangement. Five outer group folds require at least five independent group contributions per class for every class to appear in every outer test fold; nested validation and cross-class group composition add further constraints. No fold manifest was created, and Test 1 candidate components were not treated as ground truth.

After curator/source metadata is available, consider (1) up to five nested stratified group folds if the class-by-group incidence table supports them, (2) whole-session held-out train/validation/test if the number of sessions is smaller, or (3) leave-one-session-out descriptive evaluation with absent-class support reported. The detailed conditions and sensitivity alternatives are in `grouped_evaluation_feasibility.md`.

## Grouping protocol

Use confirmed source lineage/session membership as the primary must-link rule, including links across gesture classes. Do not infer a session from directory, resolution, filename numbering, performer appearance, or similar pose trajectory/background. Keep the strong and weak candidates in explicit evidence tables. Quantify their impact only in labeled sensitivity scenarios; do not use them to finalize a split without source review. `grouping_rules.md` defines the categories and evidence requirements.

Suggested sensitivity analyses are:

1. Verified lineage links only (strict reference).
2. Verified links plus the one strong Kashtam candidate (candidate sensitivity only).
3. Verified links plus all weak candidates (conservative over-grouping stress test; not a ground-truth split).
4. Directory containers or dimension groups as separate exploratory source-organization/setup analyses, never as session IDs.

The dataset has seven broad source-directory containers, and all seven span train/validation/test; they contain all 430 clips collectively. This shows the current split is mixed at the directory-container level, but does not establish source-session leakage because those directories may be storage batches. Four dimensions occur (1030×1080, 1900×1710, 1900×2060, 2300×2160); 21 classes span multiple dimensions and 20 occur in one only. Resolution-held-out results would cover at most the eligible 21-class subset and would be a setup-shift evaluation, not a session-held-out one.

## Artifacts

- `provenance_ledger.csv` — 430-row ledger with provenance fields and Test 1 candidate references.
- `candidate_relationships.csv` — 108 direct candidate edges, splits, categories, confidence, machine/manual evidence, and limitations.
- `class_support_feasibility.csv` — per-class clip counts and necessary group-count conditions.
- `grouping_rules.md` — evidence categories and defensible must-link rules.
- `grouped_evaluation_feasibility.md` — conditional fold plans and sensitivity analyses.
- `summary.json` — machine-readable coverage, evidence, and feasibility totals.
- `build_ledger.py` — reproducible builder using only Test 1 artifacts in the workspace.
- `test1_manual_visual_review.csv`, `test1_selected_pairs_visual_spotcheck.png`, and `test1_candidate_related_groups.csv` — copied Test 1 evidence used for the candidate classification.

No arbitrary group assignments or new split files were produced.
