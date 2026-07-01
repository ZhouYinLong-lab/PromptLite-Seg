# Prompt Uncertainty Research Experiment

This experiment uses SAM ViT-B with one image embedding per sample and multiple prompt queries.
It separates prompt modality, point-vs-box noise, and multi-prompt uncertainty selection.

## Summary

| Experiment | Severity | Condition | Method | Mean IoU | Mean Dice | Cases |
| --- | --- | --- | --- | ---: | ---: | ---: |
| modality | clean | box_only | sam_vit_b | 0.856521 | 0.914963 | 30 |
| modality | clean | point_box | sam_vit_b | 0.832472 | 0.900172 | 30 |
| modality | clean | point_only | sam_vit_b | 0.442972 | 0.530278 | 30 |
| noise_decomposition | clean | clean | sam_single_prompt | 0.832472 | 0.900172 | 30 |
| noise_decomposition | mild | box_noise | sam_single_prompt | 0.786362 | 0.868021 | 60 |
| noise_decomposition | mild | point_box_noise | sam_single_prompt | 0.746356 | 0.831002 | 60 |
| noise_decomposition | mild | point_noise | sam_single_prompt | 0.838599 | 0.903875 | 60 |
| noise_decomposition | moderate | box_noise | sam_single_prompt | 0.646218 | 0.751980 | 60 |
| noise_decomposition | moderate | point_box_noise | sam_single_prompt | 0.638450 | 0.741633 | 60 |
| noise_decomposition | moderate | point_noise | sam_single_prompt | 0.850666 | 0.911611 | 60 |
| uncertainty_ensemble | mild | point_box_noise | sam_consistency_medoid | 0.758167 | 0.845138 | 60 |
| uncertainty_ensemble | mild | point_box_noise | sam_oracle_best | 0.825519 | 0.896276 | 60 |
| uncertainty_ensemble | mild | point_box_noise | sam_score_select | 0.772233 | 0.857698 | 60 |
| uncertainty_ensemble | mild | point_box_noise | sam_single_noisy | 0.746356 | 0.831002 | 60 |
| uncertainty_ensemble | mild | point_box_noise | sam_vote_consensus | 0.766401 | 0.851870 | 60 |
| uncertainty_ensemble | moderate | point_box_noise | sam_consistency_medoid | 0.658063 | 0.759454 | 60 |
| uncertainty_ensemble | moderate | point_box_noise | sam_oracle_best | 0.740134 | 0.828385 | 60 |
| uncertainty_ensemble | moderate | point_box_noise | sam_score_select | 0.679578 | 0.780121 | 60 |
| uncertainty_ensemble | moderate | point_box_noise | sam_single_noisy | 0.638450 | 0.741633 | 60 |
| uncertainty_ensemble | moderate | point_box_noise | sam_vote_consensus | 0.658514 | 0.759236 | 60 |

## Findings

- Clean point+box SAM reaches 0.8325 mean IoU, while box-only reaches 0.8565. This quantifies how much the extra point contributes beyond localization.
- Under moderate point+box noise, consistency-medoid changes IoU by +0.0196 over a single noisy prompt. The oracle gap is +0.1017, estimating the recoverable headroom from prompt uncertainty.
- The key research question is no longer whether SAM is strong under clean prompts, but which prompt channel fails under realistic annotation noise and whether agreement among prompt variants is a usable reliability signal.
