# Prompt Robustness Analysis

This benchmark perturbs point and box prompts, then evaluates how each method degrades.

| Severity | Method | Mean IoU | Mean Dice | Cases |
| --- | --- | ---: | ---: | ---: |
| clean | center_color | 0.546839 | 0.675412 | 30 |
| clean | repaired_superpixel | 0.578146 | 0.699792 | 30 |
| clean | robust_superpixel | 0.603078 | 0.727713 | 30 |
| clean | sam_noisy_prompt | 0.832472 | 0.900172 | 30 |
| clean | sam_oracle_best_prompt | 0.848330 | 0.909928 | 30 |
| clean | sam_repaired_prompt | 0.746610 | 0.821963 | 30 |
| clean | sam_score_selected_prompt | 0.813641 | 0.886086 | 30 |
| mild | center_color | 0.483795 | 0.615338 | 60 |
| mild | repaired_superpixel | 0.527635 | 0.660372 | 60 |
| mild | robust_superpixel | 0.557939 | 0.692667 | 60 |
| mild | sam_noisy_prompt | 0.773091 | 0.853682 | 60 |
| mild | sam_oracle_best_prompt | 0.782379 | 0.861064 | 60 |
| mild | sam_repaired_prompt | 0.674873 | 0.762576 | 60 |
| mild | sam_score_selected_prompt | 0.740906 | 0.825155 | 60 |
| moderate | center_color | 0.387397 | 0.517629 | 60 |
| moderate | repaired_superpixel | 0.442463 | 0.582101 | 60 |
| moderate | robust_superpixel | 0.456847 | 0.596730 | 60 |
| moderate | sam_noisy_prompt | 0.675623 | 0.779531 | 60 |
| moderate | sam_oracle_best_prompt | 0.692081 | 0.793427 | 60 |
| moderate | sam_repaired_prompt | 0.574450 | 0.671547 | 60 |
| moderate | sam_score_selected_prompt | 0.611794 | 0.713413 | 60 |

## Interpretation

The repaired variants first use the lightweight superpixel mask to infer a new point and box, then rerun the downstream method. This tests whether a transparent baseline can act as a prompt repair module rather than only a weak competitor.
