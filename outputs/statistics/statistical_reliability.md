# Statistical Reliability Analysis

All intervals are paired bootstrap 95% confidence intervals over sample-level deltas.
The p-values use a paired sign-flip permutation test against zero mean delta.

| Comparison | Pairs | Baseline | Candidate | Delta | 95% CI | p-value |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| robust_superpixel_vs_center_color | 30 | 0.546839 | 0.603078 | 0.056239 | [0.030149, 0.089117] | 0.000020 |
| sam_box_only_vs_point_box | 30 | 0.832472 | 0.856521 | 0.024049 | [0.007961, 0.042053] | 0.009220 |
| sam_point_box_vs_point_only | 30 | 0.442972 | 0.832472 | 0.389500 | [0.258545, 0.522134] | 0.000020 |
| sam_box_only_vs_point_only | 30 | 0.442972 | 0.856521 | 0.413549 | [0.283994, 0.545368] | 0.000020 |
| moderate_point_noise_vs_clean | 30 | 0.832472 | 0.850666 | 0.018194 | [0.006352, 0.031866] | 0.006580 |
| moderate_box_noise_vs_clean | 30 | 0.832472 | 0.646218 | -0.186254 | [-0.244565, -0.132198] | 0.000020 |
| moderate_point_box_noise_vs_clean | 30 | 0.832472 | 0.638450 | -0.194022 | [-0.265003, -0.130795] | 0.000020 |
| moderate_box_noise_vs_point_noise | 30 | 0.850666 | 0.646218 | -0.204448 | [-0.265470, -0.147370] | 0.000020 |
| moderate_point_box_noise_vs_point_noise | 30 | 0.850666 | 0.638450 | -0.212216 | [-0.287036, -0.144136] | 0.000020 |
| moderate_score_select_vs_single_noisy | 30 | 0.638450 | 0.679578 | 0.041128 | [-0.004396, 0.102884] | 0.154177 |
| moderate_consistency_medoid_vs_single_noisy | 30 | 0.638450 | 0.658063 | 0.019613 | [-0.004035, 0.049071] | 0.190636 |
| moderate_vote_consensus_vs_single_noisy | 30 | 0.638450 | 0.658514 | 0.020063 | [0.000320, 0.042696] | 0.076898 |
| moderate_oracle_vs_single_noisy | 30 | 0.638450 | 0.740134 | 0.101684 | [0.059266, 0.158870] | 0.000020 |

## Interpretation

- The robust superpixel baseline improves over center-color by 0.056239 IoU with CI [0.030149, 0.089117].
- Box-only SAM is not just visually better than point+box here: the paired delta is 0.024049 IoU with CI [0.007961, 0.042053].
- Moderate box noise is substantially worse than point noise: delta -0.204448 with CI [-0.265470, -0.147370].
- Score selection has a positive mean delta of 0.041128 IoU, but its CI [-0.004396, 0.102884] crosses zero, so the current 30-sample subset should treat this as promising but not statistically confirmed.
