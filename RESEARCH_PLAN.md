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
- [x] Create a disjoint tuning split and either evaluate all 1,449 VOC 2012
  validation images or a precommitted class-stratified sample of at least 300.
- [x] Calibrate point and box perturbations using measured point-hit rate and box
  IoU (or a cited human-prompt error distribution).
- [x] Add GrabCut and at least one standard interactive-segmentation baseline;
  add component ablations, latency, and memory measurements.
- [x] Pre-specify primary comparisons and apply multiplicity correction.
- [x] Update related work, results, limitations, and an anonymous submission.
- [x] Ensure no public artifact or history contains redistributable-risk VOC
  assets; keep the historical development repository private; add
  `THIRD_PARTY_NOTICES.md`; pin official SAM source and checkpoint SHA-256.
- [x] Pass clean-environment tests/CI and regenerate a reviewable report.

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
- Completed the full 1,449-sample CPU run at commit `737f87d`: all proposed and
  ablation methods succeeded on every sample; GrabCut reported eight explicit
  initialization failures. No image-bearing outputs were generated.
- Pinned official Meta SAM commit `dca509fe` through its codeload archive and
  recorded archive/checkpoint SHA-256 digests. A real RTX 5070 Ti CUDA smoke run
  completed all modality, calibrated-noise, ensemble, and oracle branches.
- Completed the frozen full-validation runs: 1,449 CPU samples and 55,062 SAM
  metric rows. Robust Superpixel reaches 0.6049 IoU, GrabCut 0.6878 when its
  eight failures are scored as zero, and SAM point+box 0.8476.
- Evaluated H1--H3 with 20,000 paired-bootstrap replicates, 50,000 sign-flip
  permutations, and Holm correction. All three pre-specified effects remain
  significant; Oracle is reported only as a descriptive upper bound.
- Rewrote the README and paper around the confirmatory evidence, added current
  related work and explicit negative results, and generated both named and
  five-page anonymous PDFs. PDF text audit found no configured identity tokens.
- Added a deterministic anonymous supplementary exporter. Its 60-file ZIP is
  allowlisted and rejects identity-bearing text, Git history, VOC-style image
  assets, and model weights. All 11 committed result artifacts pass SHA-256
  verification.
- Removed VOC samples and image-bearing exploratory outputs from the current
  Git index, added third-party notices, and documented that the existing private
  repository history must not itself be made public. A future public repository
  must start from the audited clean export, not from this historical `main`.
- Found that the previous Linux CI failure came from CRLF-sensitive frozen
  source hashes. Defined source hashes canonically over LF-normalized UTF-8;
  local compile and test validation now reports 45 passed.
- Fixed an anonymous-export leak caused by generated `.egg-info` metadata and
  removed ambiguity between success-only and failure-zero GrabCut means. Final
  validation reports 46 local tests passed and 11/11 result hashes matched.
- Committed the reviewable artifact as `0944c6c`, pushed `origin/main`, and
  observed green CPU CI on Python 3.10, 3.11, and 3.12. The repository remains
  private and no release was created; future public OSS history must start from
  the audited clean export rather than inherit this private development history.
