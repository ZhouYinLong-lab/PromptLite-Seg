# Anonymous Prompted-Segmentation Artifact

This supplementary artifact accompanies the anonymous manuscript **Calibrated
Prompt Uncertainty in Zero-Shot Segmentation**.

It contains source code, frozen data-free manifests, per-sample metrics,
confirmatory statistics, tests, and the anonymous PDF. It contains no dataset
images, masks, checkpoints, qualitative visualizations, repository history, or
author identity.

## Main evidence

- Complete PASCAL VOC 2012 validation protocol: 1,449 images, 20 classes.
- Disjoint tuning protocol: 100 VOC train images, five per class.
- Three pre-specified primary hypotheses with paired bootstrap intervals,
  sign-flip tests, and Holm correction.
- 1,449-sample CPU results and 55,062 SAM metric rows.
- SHA-256 checksums for all committed confirmatory result files.

## Reproduction

```bash
python -m pip install -e ".[test]"
python -m pytest
python scripts/fetch_protocol_assets.py
python scripts/prepare_voc_from_manifest.py \
  --manifest protocol/manifests/confirmatory_validation.jsonl \
  --parquet data/cache/pascal_voc_2012_val.parquet \
  --output-dir data/voc_validation
python scripts/run_confirmatory_cpu.py
```

For SAM, install a GPU-compatible PyTorch build, then run:

```bash
python -m pip install -r requirements-sam.txt
python scripts/fetch_protocol_assets.py --include-sam
python scripts/run_confirmatory_sam.py --device cuda
python scripts/analyze_confirmatory.py
```

See `THIRD_PARTY_NOTICES.md` before obtaining VOC or SAM assets.
