# Anonymous Prompted-Segmentation Artifact

This supplementary artifact accompanies the anonymous manuscript **Calibrated
Prompt Uncertainty in Zero-Shot Segmentation**.

It contains source code, frozen data-free manifests, per-sample metrics,
confirmatory statistics, tests, and the anonymous PDF. It contains no dataset
images, masks, checkpoints, qualitative visualizations, repository history, or
author identity.

## Submission compliance

- The manuscript uses the NeurIPS template and is written in English.
- Pages 1--7 contain the complete main text, including the abstract.
- References start on page 8; appendices start on page 9 and do not count
  toward the seven-page main-text limit.
- The manuscript contains no acknowledgement section or author identity.
- `reports/report_anonymous.pdf` is the submission PDF. The named manuscript is
  intentionally excluded from this artifact.

## Main evidence

- Complete PASCAL VOC 2012 validation protocol: 1,449 images, 20 classes.
- Disjoint tuning protocol: 100 VOC train images, five per class.
- Three pre-specified primary hypotheses with paired bootstrap intervals,
  sign-flip tests, and Holm correction.
- 1,449-sample CPU results and 55,062 SAM metric rows.
- SHA-256 checksums for all committed confirmatory result files.
- A separately labelled, post-confirmatory adaptive-superpixel summary with
  paired full-validation statistics and provenance hashes.
- A separately frozen five-target sensitivity analysis with 289,800 real SAM
  metric rows; it is secondary and does not relabel H3.
- A deterministic 200-sample ADE20K transfer observation drawn after scanning
  all 2,000 validation rows, with 1,400 data-free metric rows and provenance.
- A privacy-minimizing human-prompt pilot toolkit with collection locked until
  instructor/ethics approval and protocol freezing; no human result is claimed.

## Reproduction

```bash
python -m pip install -e ".[test]"
python -m pytest
python scripts/fetch_protocol_assets.py
python scripts/prepare_voc_from_manifest.py \
  --manifest protocol/manifests/confirmatory_validation.jsonl \
  --parquet data/cache/pascal_voc_2012_val.parquet \
  --output-dir data/voc_validation
python scripts/run_confirmatory_cpu.py --methods \
  center_color grabcut_point_box robust_superpixel \
  robust_no_color_seed robust_no_spatial_prior robust_single_box
```

For the verified CUDA 12.8 SAM environment, run:

```bash
python -m pip install -r requirements-sam-cu128.txt
python scripts/fetch_protocol_assets.py --include-sam
python scripts/run_confirmatory_sam.py --device cuda
python scripts/analyze_confirmatory.py
```

For another platform, install its compatible PyTorch build first, then use `requirements-sam.txt`.

See `THIRD_PARTY_NOTICES.md` before obtaining VOC or SAM assets.
