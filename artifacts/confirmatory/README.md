# Confirmatory Artifact

This directory contains metrics and statistical summaries only. It contains no
VOC image, semantic mask, target mask, model checkpoint, or image-bearing
visualization.

## Frozen runs

- CPU baselines and ablations: commit `e7cbe54a1eedff6965e500d1775ecd3e65f2ecc0`
- SAM confirmatory benchmark: commit `e7cbe54a1eedff6965e500d1775ecd3e65f2ecc0`
- Dataset: all 1,449 PASCAL VOC 2012 validation rows listed in the frozen
  manifest.
- SAM checkpoint SHA-256:
  `ec2df62732614e57411cdcf32a23ffdf28910380d03139ee0f4fcbe91eb8c912`

The CPU metrics retain eight explicit GrabCut initialization failures. The
confirmatory statistical summary assigns those failures IoU/Dice zero rather
than silently dropping them. SAM completed all 1,449 samples and produced
55,062 metric rows.

Both runs use dataset fingerprint
`9cc5140ce4283809f668ff87dc3d7efc13b76b693da2f43d7c42a1d7b3f90c9b`.
The materializer preserves each source JPEG byte-for-byte; the SAM run
configuration and resume fingerprint are retained in `sam/run_config.json`.

The raw CPU summary names both the success-only mean and the failure-zero mean
explicitly; the latter is the confirmatory value used in the paper and
statistical tables.

## Reproduction

```bash
python scripts/prepare_voc_from_manifest.py \
  --manifest protocol/manifests/confirmatory_validation.jsonl \
  --parquet data/cache/pascal_voc_2012_val.parquet \
  --output-dir data/voc_validation

python scripts/run_confirmatory_cpu.py
python scripts/run_confirmatory_sam.py
python scripts/analyze_confirmatory.py
```

Read `THIRD_PARTY_NOTICES.md` before obtaining or using VOC or SAM assets.
