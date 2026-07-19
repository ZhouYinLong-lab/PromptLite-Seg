# Confirmatory Results

All primary effects use paired sample-level deltas. H1--H3 are corrected together with Holm's method.
Oracle selections are descriptive upper bounds and are not primary hypotheses.

| Hypothesis | Mean IoU delta | 95% CI | raw p | Holm p | Reject |
| --- | ---: | ---: | ---: | ---: | --- |
| H1: robust_superpixel - center_color | 0.0632 | [0.0578, 0.0686] | 0.000020 | 0.000060 | True |
| H2: sam_score_select - sam_single_noisy | 0.0204 | [0.0145, 0.0261] | 0.000020 | 0.000060 | True |
| H3: point_noise IoU - box_noise IoU | 0.1067 | [0.0996, 0.1138] | 0.000020 | 0.000060 | True |

## Prompt-quality check

- Moderate point hit rate: 0.6391.
- Moderate box mean IoU: 0.6514.
