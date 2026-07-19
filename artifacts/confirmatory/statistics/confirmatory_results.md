# Confirmatory Results

All primary effects use paired sample-level deltas. H1--H3 are corrected together with Holm's method.
Oracle selections are descriptive upper bounds and are not primary hypotheses.

| Hypothesis | Mean IoU delta | 95% CI | raw p | Holm p | Reject |
| --- | ---: | ---: | ---: | ---: | --- |
| H1: robust_superpixel - center_color | 0.0640 | [0.0585, 0.0696] | 0.000020 | 0.000060 | True |
| H2: sam_score_select - sam_single_noisy | 0.0193 | [0.0135, 0.0251] | 0.000020 | 0.000060 | True |
| H3: point_noise IoU - box_noise IoU | 0.1069 | [0.0997, 0.1138] | 0.000020 | 0.000060 | True |

## Prompt-quality check

- Moderate point hit rate: 0.6391.
- Moderate box mean IoU: 0.6514.
