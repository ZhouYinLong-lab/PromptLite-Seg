# PromptLite-Seg Research Artifact Upgrade Plan

Baseline commit: `e464cf5359e7325ca4af3401d089c73a966de7dc`

## Objective

Move PromptLite-Seg from a reproducible course project to a reviewable research
artifact. Completion requires a clean-environment execution path, representative
evaluation, pre-specified statistical claims, an anonymous paper, and a public
artifact that does not redistribute third-party data without permission.

## Frozen research contract

- The final evaluation set must never be used to tune algorithm constants,
  prompt-noise parameters, thresholds, or model-selection rules.
- Development/tuning data and confirmatory evaluation data must be disjoint and
  recorded in a machine-readable manifest before confirmatory results are run.
- Primary hypotheses, metrics, comparison directions, and multiplicity handling
  must be declared before confirmatory results are inspected.
- Oracle selection is an upper bound only. It must not be described as a
  deployable method.
- Existing 30-sample results are exploratory and cannot be relabeled as the
  confirmatory benchmark.

## Acceptance checklist

- [x] Fix the SAM robustness call contract and add mock/contract coverage for
  every SAM execution branch.
- [x] Freeze algorithms, hyperparameters, hypotheses, and evaluation protocol.
- [ ] Create a disjoint tuning split and either evaluate all 1,449 VOC 2012
  validation images or a precommitted class-stratified sample of at least 300.
- [x] Calibrate point and box perturbations using measured point-hit rate and box
  IoU (or a cited human-prompt error distribution).
- [ ] Add GrabCut and at least one standard interactive-segmentation baseline;
  add component ablations, latency, and memory measurements.
- [ ] Pre-specify primary comparisons and apply multiplicity correction.
- [ ] Update related work, results, limitations, and an anonymous submission.
- [ ] Remove redistributable-risk VOC assets from public history; add
  `THIRD_PARTY_NOTICES.md`; pin official SAM source and checkpoint SHA-256.
- [ ] Pass clean-environment tests/CI and regenerate a reviewable report.

## Current evidence and blockers

- CPU path: 19 tests pass; Python 3.10--3.12 CI is green; committed 30-sample
  CPU summaries can be regenerated exactly.
- SAM path: `scripts/run_robustness_experiment.py` defines
  `sam_predict(predictor, prompt)` but one branch calls it with three positional
  arguments. This branch is not covered by current CI.
- Evaluation: the committed subset is validation rows 0--29, covering 12 of 20
  classes (30/1,449 samples). It is exploratory and non-representative.
- Publication: the current report identifies the author/repository and is not a
  double-blind artifact.
- Public release: the Git history contains VOC images/masks and visualizations;
  Apache-2.0 applies only to original code and does not grant dataset rights.
- Supply chain: the SAM requirement currently resolves a PyPI package rather
  than a pinned official Meta commit; the checkpoint has no recorded digest.

## Work log

### 2026-07-19

- Re-evaluated baseline `e464cf5` on course, paper, and OSS rubrics.
- Confirmed the SAM robustness call-contract defect by inspecting and binding
  the function signature.
- Started this persistent plan before implementation changes.
- Centralized the SAM predictor contract in `src/promptseg/sam.py`, fixed the
  repaired-prompt call, and isolated ensemble selection into testable helpers.
- Added CPU-only fake-runtime tests that execute the basic SAM, robustness, and
  prompt-uncertainty CLIs, including point-only, box-only, point+box, repaired,
  score-selected, consistency, vote, and oracle branches. Full suite: 29 passed.
- Built immutable data-free manifests: a disjoint VOC train tuning set with five
  targets per class (100 total) and all 1,449 VOC validation rows for
  confirmatory evaluation. Source parquet files and manifests have SHA-256
  digests recorded in `protocol/manifests/manifest_summary.json`.
- Froze three primary IoU hypotheses, Holm family correction, reporting rules,
  algorithm-source hashes, and the rule that oracle results are descriptive
  upper bounds only in `protocol/research_protocol.json`.
- Calibrated prompt noise only on the tuning split. Mild point hit rate / box IoU
  are 0.8480 / 0.8538; moderate values are 0.6495 / 0.6509. The resulting scales
  are frozen in `protocol/noise_calibration.json` and runtime constants.
- Added a five-iteration point+box GrabCut baseline and registered SAM point,
  box, and point+box modes as standard interactive baselines. Added ablations
  for the color seed, spatial prior, and multi-box consensus.
- Added `run_confirmatory_cpu.py`, which emits metrics only and records failures,
  per-sample latency, total runtime, and sampled peak RSS. SLIC labels/features
  are now reused across box-consensus members; the refactor reproduces all
  original 30-sample CPU summary values exactly.
