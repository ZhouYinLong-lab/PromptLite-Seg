# Multi-Quality Point-vs-Box Sensitivity Curve

**Status**: secondary (not a confirmatory H1–H3 result)

This is a robustness/sensitivity analysis. Numeric observable matching (point hit rate vs box IoU) does NOT imply human-perceptual equivalence.

- Samples: 1449
- Quality targets: ['0.9', '0.8', '0.7', '0.6', '0.5']
- Conditions: point_noise, box_noise
- Synthetic: False
- Trials per sample/target/condition: 20
- Evaluation complete: True

## Results

Point-noise minus box-noise SAM IoU is:

| Target | Mean delta | 95% paired-bootstrap CI |
| ---: | ---: | ---: |
| 0.9 | -0.0016 | [-0.0041, 0.0009] |
| 0.8 | 0.0238 | [0.0201, 0.0275] |
| 0.7 | 0.0756 | [0.0703, 0.0809] |
| 0.6 | 0.1497 | [0.1430, 0.1563] |
| 0.5 | 0.2324 | [0.2246, 0.2402] |

The direction is stable from 0.8 to 0.5 but indistinguishable near 0.9.
Observable matching does not imply human-perceptual equivalence.

## Regenerate

```bash
python scripts/calibrate_sensitivity.py
python scripts/run_sensitivity_sam.py
python scripts/analyze_sensitivity.py
```
