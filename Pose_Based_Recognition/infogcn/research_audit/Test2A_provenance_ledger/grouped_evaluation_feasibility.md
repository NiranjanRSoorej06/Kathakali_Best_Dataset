# Grouped evaluation feasibility analysis

## Finding

**A defensible group-aware evaluation cannot yet be instantiated.** The 430 clip rows have source paths and SHA-256 hashes, but verified original-video IDs, crop intervals, session/event IDs, and dancer IDs are absent. Thus, the number of independent groups and their per-class coverage are unknown. The candidate relationships from Test 1 are evidence leads, not a substitute for verified groups. No fold manifest is finalized here.

## Observed sample support

The canonical split contains 300 train, 65 validation, and 65 test clips across 41 classes. Every class appears in every existing split. Total class support is 10 or 11; train support is 7 or 8, while validation and test support is only 1 or 2 per class. The per-class counts are in `class_support_feasibility.csv`.

These are **clip counts**, not independent recording counts. For any class, three independent sessions are a necessary (not sufficient) condition to place at least one session in train, validation, and test. A five-fold outer grouped evaluation requires at least five independent groups containing each class to place each class in every outer test fold. Nested group validation imposes further coverage constraints; group composition across classes can make the assignment infeasible even when a class-level count appears adequate.

If the goal is stable per-class test estimates with at least two independent clips/groups per class in each outer test fold, about ten or more independent group contributions per class would be needed. The dataset has only 10–11 clips per class in total, so that target is unlikely unless session groups are fine-grained and mostly single-clip. Grouped folds will likely have high variance and should report per-class support, absent classes, pooled out-of-fold metrics, and fold-level spread.

## Conditional primary strategies after provenance is collected

### Strategy A: nested stratified group cross-validation

If session/source groups are verified and class coverage allows it, use up to five outer folds with groups intact and constrained class balancing. Inside each outer training partition, reserve a separate group-held-out validation subset for model selection. Never split one source/session group across outer train, inner validation, or outer test. Reduce the fold count if group integrity and class coverage conflict; disclose missing classes rather than splitting a group.

Five folds are a target, not a commitment. Before generating folds, produce a class-by-verified-group incidence table, number of unique groups per class, group sizes, and a feasibility report for every proposed assignment. The outer test folds and inner validation groups must be frozen before any training.

### Strategy B: grouped holdout

If the number of verified sessions is too small for cross-validation, reserve one or more whole sessions for a final test and separate whole sessions for validation. Approximate a 70/15/15 clip proportion only after honoring group integrity and class coverage. If a class is absent from a holdout group, report its support as zero; do not move clips across group boundaries to force stratification. A single grouped holdout will be statistically noisy but remains preferable for the stated session-generalization question to a knowingly mixed-session split.

### Strategy C: leave-one-session-out sensitivity

With a small number of verified sessions, leave one session out at a time and aggregate out-of-fold predictions. Report session-level and class-level support, folds with missing classes, and the variation across held-out sessions. This is descriptive if sessions are few or imbalanced; it does not guarantee every class has adequate independent support.

## Sensitivity scenarios (not primary group assignments)

1. **Confirmed-lineage only:** merge only exact hashes and curator/source-verified source/session links. Currently no cross-split confirmed links are recorded, so this scenario alone cannot establish session independence.
2. **Confirmed + strong candidate:** additionally treat the Kashtam_8/Kashtam_11 edge as a must-link solely to quantify its impact. It is not proven; compare with the confirmed-only result.
3. **Expanded candidate union:** include all 107 weak candidates and three manual additions as candidate must-links. Test 1 gives 37 possible connected components involving 167 clips after its candidate overlay. This may over-group repeated choreography and must be labeled an upper-bound stress test, not a verified grouping.
4. **Directory-container stress test:** use the six `Videos_CroppedN` directories and the first-person collection as broad holdout tags only. All seven containers span the current splits and collectively contain all 430 clips. Because they are not session IDs and aggregate many clips, this scenario can be uninformative or infeasible for 41-class coverage; it cannot be presented as a provenance-valid evaluation.
5. **Setup/dimension shift:** keep dimension groups separate from session groups. The four dimensions are 1030×1080 (44 clips), 1900×1710 (84), 1900×2060 (82), and 2300×2160 (220). Twenty-one classes span multiple dimension groups; the other twenty occur in only one. A dimension-held-out result can therefore cover at most the eligible 21-class subset for cross-setup performance, and it does not measure recording-session independence.

## Class imbalance and metric implications

Class totals (10–11) are nearly balanced at the clip level. The limiting factor is sparse independent support, not current class-frequency skew. Avoid aggressive class weighting as a substitute for missing sessions. If grouped folds create imbalance, constrain the fold assignment at the group level and report the residual support. Keep all 41 output classes for the primary task; give fixed 41-class macro-F1 and per-class support, and separately mark represented-class metrics when a fold lacks a class. The existing 65-clip test has only one or two clips per class and is not precise for per-class generalization.

## Readiness checklist for future fold creation

- Source file hash verified for every row and immutable source path recorded.
- Original-video ID and clip start/end source frame/time recorded where applicable.
- Session/event and dancer IDs recorded from a curator or capture log, with missingness and confidence explicit.
- Related clips across classes joined by verified session/source, not appearance or labels.
- Class-by-group incidence table passes the chosen fold feasibility conditions.
- Group manifest, protocol, seeds, and metric conventions frozen before training.
- Existing clip-level split retained as a historical comparator; no current split file overwritten.

No training or tuning was conducted for this feasibility assessment.
