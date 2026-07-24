# ADE20K Bounded Transfer Observation

**Status**: secondary, deterministic 10% sample; not a full cross-dataset
confirmation and not part of H1--H3.

- Rows scanned: 2,000
- Eligible rows in scanned stream: 2,000
- Deterministically selected/evaluated samples: 200 (seed 20260720)
- Excluded: 0
- Methods: Center Color, GrabCut point+box, Robust Superpixel, three component
  ablations, and Adaptive Superpixel
- Metric rows: 1,400
- Method failures: 0
- Stream fully scanned: yes
- Stream fully evaluated: no (200/2,000)

The selected sample identifiers are fixed in `run_config.json`. The complete
source-row audit is stored in `source_manifest.jsonl`; it contains identifiers
and target geometry but no images or masks. `metrics.csv` contains one row per
sample and method. Recorded SHA-256 values are in `checksums.json`.

## Regenerate

```bash
python -m pip install -e ".[ade]"
python scripts/run_ade20k_cpu.py --mode pilot --max-rows 2000 \
  --max-eligible 200 --output-dir outputs_ade20k_200 \
  --cache-dir data/cache/ade20k --methods center_color \
  grabcut_point_box robust_superpixel robust_no_color_seed \
  robust_no_spatial_prior robust_single_box adaptive_superpixel
```
