# Kathakali 430-video provenance audit and EXP11 proposal

**Scope:** Read-only investigation of the canonical 430-video, 41-class dataset and EXP10RERUN. No model was trained or fine-tuned. No data, checkpoint, experiment, or result artifact was changed. New40 was not used for per-video error analysis or model/augmentation selection; see the audit boundary below.

## 1. Visual provenance spot-check

The contact sheet shows four frames per clip sampled at 10%, 35%, 65%, and 90% of clip duration: [visual_spotcheck.png](visual_spotcheck.png). All nine source files decoded. The canonical metadata confirms the requested clips are split across train, validation, and test as follows.

| Comparison | Canonical split(s) | Frame-level evidence and cautious interpretation | Confidence |
|---|---|---|---|
| Chundu_3 vs Chundu_1 and Chundu_5 | test vs train / train | All show the same plain-wall setting, similar camera framing and a visually similar male dancer in a white dhoti performing the same gesture sequence. The sampled progression is similar, especially Chundu_1 and Chundu_3. Frame counts differ (426, 473, 425) and rates are close but not identical (~56.89–59.98 fps). This supports a common capture setup and possibly a related session or repeated take. The samples do not establish shared source frames or temporal overlap. | Medium for common setup/related session; low for shared footage. |
| Kashtam_8 vs Kashtam_11 | val vs train | Both are under Videos_Cropped5/Kashtam, have identical metadata (334 frames, 59.964 fps, 5.57 s), and show an unusually similar pose progression at all four sampled relative times. This is the strongest candidate for a shared take, a duplicate/re-encode, or near-identical takes from one session. Their file SHA-256 hashes differ, so they are not byte-identical. No sampled-frame identity or temporal alignment has been established. | Medium-high for common take/session; unresolved whether duplicate or separate takes. |
| Albutham_4 vs Albutham_5 | train vs test | Similar performer appearance, white-wall scene, framing, and hand sequence. Metadata is nearly identical (428 vs 427 frames; both 59.972 fps), but files are in Videos_Cropped4 and Videos_Cropped3. The evidence fits related takes in one setup; it does not demonstrate a recrop or temporal overlap. | Medium for common setup/session; low for shared footage. |
| Thaamara_9 vs Thaamara_3 | test vs train | Similar performer appearance, setup, and progression. Both are in Video_Collection_first_person/Thaamara; 798 vs 954 frames at about 59.97 fps. This is consistent with distinct takes in a shared collection/session. The sampled frames do not prove overlap. | Medium for common collection/setup; low for shared footage. |

Across all pairs, the white background and similar pose trajectories are clues to a shared recording setup, not proof of source overlap. Filenames and suffix numbers are not known to encode chronology. SHA-256 values differ for all nine files; this rules out exact byte copies only, not re-encoding, trimming, or recropping. The review was a sparse visual spot-check, not full-length synchronization or forensic source matching.

## 2. Provenance findings and available metadata

The canonical split file contains 430 unique video rows: 300 train, 65 validation, and 65 test, with all 41 classes in every split. Class totals are 10 or 11 videos; per-class support is 7–8 in train and only 1–2 in validation and test. The requested pairs are confirmed cross-split by `stgcn/datasets/processed/metadata_split.csv`.

Available metadata includes clip filename/class, feature path, original width/height, FPS, frame count, valid-frame counts, reference shoulder width, and split. `video_validation_inventory.csv` adds absolute source paths, duration, readability, frame count, FPS, and dimensions. Source locations include the `Videos_Cropped1`–`Videos_Cropped6` directory bundles and `GitHub_Fill_In/Video_Collection_first_person/<class>`. Every numbered bundle and the remaining first-person collection contain clips in all three splits:

| Source directory bundle | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Videos_Cropped1 | 63 | 15 | 11 | 89 |
| Videos_Cropped2 | 40 | 8 | 7 | 55 |
| Videos_Cropped3 | 39 | 7 | 7 | 53 |
| Videos_Cropped4 | 38 | 7 | 8 | 53 |
| Videos_Cropped5 | 34 | 7 | 4 | 45 |
| Videos_Cropped6 | 13 | 4 | 8 | 25 |
| First-person collection | 73 | 17 | 20 | 110 |

These are useful candidate source-bundle tags, but their names do not establish that each bundle is one recording session or dancer. The first-person total aggregates its class subdirectories.

There are four observed dimension groups: 1030×1080 (44 clips), 1900×1710 (84), 1900×2060 (82), and 2300×2160 (220). Twenty-one classes occur in more than one dimension group; twenty occur in one only. Dimensions likely combine camera, crop, framing, and encoding choices, so they are setup proxies rather than pure resolution interventions.

The reviewed manifests do not provide verified dancer IDs, recording/session IDs, original uncut-video IDs, crop start/end times, capture dates/locations, camera IDs, or a source-to-derived-clip map. No audio, EXIF, or original-source lineage was available in the inspected metadata. Visual similarity cannot fill those gaps reliably. Until an annotated provenance ledger exists, cross-split session leakage is plausible but unquantified.

## 3. Recommended trustworthy evaluation

### Group-aware primary protocol

1. Create a separate provenance ledger before training. Include immutable source-file SHA-256, full source path, original-video ID, clip start/end frame or timestamp, session/event ID, dancer ID where known, camera/setup ID, dimensions/FPS, and confidence plus evidence for each assignment. Link clips from the same original video or overlapping time intervals. Group all clips from the same session together, including different gesture classes.
2. Do not treat `Videos_CroppedN` as a confirmed session ID. Use it as a provisional sensitivity grouping only. Resolve likely duplicate/near-duplicate candidates by full sequence matching and source review; conservatively merge uncertain links in the strict primary grouping and report a less conservative sensitivity result.
3. Once groups are annotated, use constrained stratified group-held-out evaluation: outer 5-fold grouped cross-validation (or leave-one-session/group-out if fewer independent groups exist), with a separate group-held-out validation fold inside each outer training partition for checkpoint selection. No source/session group may cross train, validation, and outer test. Optimize fold assignment for class support and approximate fold size without splitting groups. Publish fold manifests before training. If 5 folds cannot retain adequate class coverage, reduce the fold count and report the constraint rather than splitting groups.
4. Keep the existing clip-level 300/65/65 split as a historical in-distribution comparator only. Compare the same architecture and fixed training recipe on that split and on the grouped protocol; do not interpret the current clip-level test score as session-independent generalization.

### Resolution/setup evaluation

For the 21 multi-dimension classes—Aakasham, Aana, Acchan, Bhoomi, Bhrahmaavu, Chandran, Irikkuka, Kaadu, Nadakkuka, Nee_Roudram, Nee_Shringaaram, Paadam, Samudram, Satyam, Shivan, Simham, Sooryan, Svargam, Veedu, Vishnu, and Vriksham—run a separate four-way leave-one-dimension-group-out analysis. Train on the other dimension groups and test on the held-out group, with session groups also kept intact wherever source IDs allow. Report class-level support and aggregate results for only the eligible 21-class subset. The four folds are not interchangeable with session-held-out folds and should be described as setup-shift evaluation, since dimension also reflects cropping/framing/camera choices.

The other 20 classes occur in only one dimension group, so the dataset cannot measure cross-dimension generalization for them. Keep them in the all-41-class session-held-out analysis when supported; report their within-setup performance separately. Do not imply a 41-class resolution-held-out result.

### Class handling and reporting

The current class distribution (10–11 clips/class) is nearly balanced, so use group-constrained split assignment rather than strong class reweighting as the first remedy. Keep the classifier at all 41 outputs. Report accuracy, macro-F1 over all 41 classes, weighted-F1, balanced accuracy, per-class precision/recall/F1 and support, and the confusion matrix. If a fold lacks a class, report its support and zero-support convention explicitly; also give represented-class macro-F1, clearly labeled, alongside the fixed 41-class macro-F1. Aggregate out-of-fold predictions and show fold/seed variation; use group-level bootstrap intervals rather than treating clips from one session as independent. The current validation/test support of one or two clips per class makes per-class estimates highly uncertain.

## 4. EXP11 proposal (design only; do not train yet)

**Objective:** measure session-independent performance and reduce sensitivity to masking and plausible capture/setup shifts, with EXP10RERUN as the frozen architectural baseline. Lock the split ledger and protocol before any EXP11 model selection. Preserve New40 as an untouched external evaluation set until model, preprocessing, and all decisions are frozen.

### Mask-boundary correction

EXP10RERUN loads `frame_valid`, masks the input after `BatchNorm1d`, and reapplies masks around blocks. However, that input normalization sees invalid time positions during training, and temporal-branch `BatchNorm2d` sees masked/padded positions; temporal convolutions also compute around mask boundaries before the block output is remasked. EXP11 should compare this baseline against a mask-aware version that excludes invalid positions from normalization statistics, masks after each bias/normalization/activation and each temporal branch, and propagates masks exactly through both stride-2 blocks and residuals. Keep masked temporal padding explicit and verify that changing padded values cannot change valid-frame outputs. Report the correction as a single controlled ablation before adding augmentation.

### Temporal receptive field

The current five temporal blocks use two stride-2 reductions and branches with effective temporal kernels up to 5 (3 taps at dilation 2). The theoretical maximum receptive field is about 29 sampled frames at the output, from a 100-frame sequence. This is a fraction of the clip, not a fixed number of seconds, because frames are uniformly sampled per video. Compare shorter, current, and longer effective maximum fields (approximately 15, 29, and 57 input samples) while holding other components fixed. Select among them only on grouped validation folds; document that clips of different duration map the same sampled-frame field to different wall-clock durations.

### Regularization and train-only augmentation

Keep the existing no-augmentation setup as the control. Prespecify a small ablation: (a) mild coordinate translation/scale/rotation; (b) confidence and joint dropout plus temporally coherent frame dropout; (c) a restrained temporal resampling/warp; and (d) weight-decay/dropout/label-smoothing alternatives. Apply perturbations only to training data and only when they preserve gesture meaning. Avoid choosing strengths from New40 outcomes. Given the 10–11 examples/class, avoid an unplanned large search; use fixed grouped validation folds and a small, pre-registered grid. Start with no class weights, then add a class-balanced sampler or loss weighting only as a prespecified training ablation if grouped folds reveal meaningful imbalance.

### Evaluation, ablations, and reproducibility

- **Primary evaluation:** session/source-group outer folds; inner group validation for checkpoint selection. Keep the current clip-stratified split as a secondary legacy comparator and dimension-held-out evaluation as a separate 21-class analysis.
- **Ablation order:** (1) EXP10RERUN baseline on fixed folds; (2) mask-aware normalization/boundaries; (3) shorter/current/longer temporal fields; (4) one augmentation family at a time; (5) selected regularization alternatives. Use the same fold assignments, class mapping, optimizer budget, and seeds for paired comparisons.
- **Metrics:** all-41 macro-F1, accuracy, balanced accuracy, weighted-F1, per-class support/recall/F1, confusion matrix, fold/seed spread, and group-bootstrap uncertainty. Report both pooled out-of-fold and per-fold values.
- **Seeds and model selection:** at least four fixed seeds per final configuration; choose checkpoints only by inner grouped validation loss (with validation macro-F1 reported, not silently substituted). No outer test or New40 feedback into selection.
- **Reproducibility ledger:** dataset version and file hashes; source/session/group and fold manifests; DWPose/preprocessing versions; feature normalization and mask rules; architecture/receptive-field calculation; augmentation ranges; optimizer/scheduler/early-stopping settings; seeds, software/CUDA/device versions; checkpoint hashes; exact commands; per-video predictions with stable IDs; and a record that New40 was opened only once after all choices were locked.

## 5. Prioritized next steps

1. **High:** build and review the source/session ledger, starting with the Kashtam_8/Kashtam_11 candidate and all same-source or overlapping-interval candidates. Preserve evidence and confidence; do not infer identities from filenames alone.
2. **High:** obtain original source-video IDs and crop intervals, and record dancer/session/camera information from a curator or capture log where available.
3. **High:** publish a separate group-fold manifest with class-support feasibility checks; retain the existing split unchanged as the clip-level comparator.
4. **Medium:** implement and validate mask-aware normalization/convolution behavior in a new EXP11 directory only after the protocol is frozen.
5. **Medium:** run the pre-registered temporal-field and train-only augmentation/regularization ablations on grouped validation folds; keep New40 sealed until selection is complete.
6. **Later:** conduct the 21-class dimension-held-out study and report the 20 single-dimension classes as not evaluable for cross-dimension generalization.

## 6. Commands, files, outputs, and limitations

**Key commands executed:** read-only `ls`, `find`/`find -L`, `rg`, `sed`, and `cat` inspection; Python/OpenCV/Pillow dependency check; Python queries of the split/inventory metadata and per-class/resolution counts; SHA-256 hashing of the nine source videos; OpenCV extraction of four frames per video to a contact sheet. One metadata one-liner initially had a syntax error and was corrected. An attempted numeric pairwise frame-comparison command returned no output, so no numerical temporal-overlap score is claimed. The first source search did not follow the Videos_Cropped symlink; `find -L` resolved it. No training, fine-tuning, evaluation, or New40 prediction-file analysis was run.

**Inspected files/directories:** `../README.md`; `../stgcn/datasets/processed/metadata.csv` and `metadata_split.csv`; `../../metadata/video_validation_inventory.csv`; the nine named source videos under `../../Videos_Cropped/` and `../../GitHub_Fill_In/Video_Collection_first_person/`; `experiments/exp10rerun/README_exp10rerun.md`, `multiseed_config.json`, `model_exp10rerun.py`, `train_exp10rerun.py`; `../stgcn/experiments/common/dataset.py`; and `testing/EXP10RERUN_New40_External_Evaluation.md`. The New40 summary was read as background only; no per-video New40 predictions/errors were inspected or used to select EXP11 changes. New40 should remain sealed for all EXP11 decisions.

**Output:** this report and [visual_spotcheck.png](visual_spotcheck.png), both under `research_audit/EXP11_provenance_audit/`. The source videos, datasets, existing experiments, checkpoints, and result files were not edited. Visual conclusions remain limited by sparse frame sampling, absent dancer/session/source IDs, absent crop intervals, and no original uncropped-video lineage. Similar appearance and background support candidate groupings, not proof of common footage.
