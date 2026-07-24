# ADE20K Secondary Cross-Dataset Experiment

**Status**: secondary (not a confirmatory H1–H3 result)

- Rows scanned: 100
- Eligible rows in scanned stream: 100
- Selected/evaluated samples: 10
- Excluded: 0
- Methods: center_color, grabcut_point_box, robust_superpixel, robust_no_color_seed, robust_no_spatial_prior, robust_single_box
- Mode: pilot
- Stream complete: False

## Regenerate

```bash
python scripts/run_ade20k_cpu.py --mode pilot --max-rows 100 --max-eligible 10
```
