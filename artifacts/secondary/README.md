# Secondary Analyses

This directory contains post-confirmatory analyses that are intentionally
separate from the pre-specified H1--H3 tests.

## Adaptive-superpixel resolution

This directory records a post-confirmatory comparison between the fixed
280-segment Robust Superpixel method and an image-size-aware segment schedule.
It is **not** one of the pre-specified H1--H3 confirmatory tests.

The comparison covers all 1,449 PASCAL VOC 2012 validation samples. The paired
mean IoU gain is 0.00739 with a 95% bootstrap interval of [0.00527, 0.00957].
Target-area quartiles are reported as a secondary diagnostic. Because the
schedule uses image dimensions rather than target area, the decreasing
quartile gains are supporting evidence, not a causal test of target size.

`adaptive_summary.json` contains the aggregate results, quartile diagnostics,
and SHA-256 provenance. The data-free per-sample CSV can be regenerated with:

```bash
python scripts/run_adaptive_comparison.py
```

The recorded CSV hash identifies the local result used to produce this
summary. Dataset images, masks, and checkpoints are not included.

## Multi-quality prompt sensitivity

`prompt_quality_sensitivity/` contains a separately frozen five-target
robustness analysis on all 1,449 VOC validation samples. Each sample, target,
and perturbation channel has 20 trials (289,800 rows total). It is secondary:
it refines the operating range of H3 but does not relabel or replace H3.

## ADE20K transfer observation and human-prompt pilot

`ade20k/` records a deterministic 200-sample evaluation drawn from all 2,000
ADE20K validation rows after a complete stream scan. It is a bounded secondary
transfer observation, not a full 2,000-sample cross-dataset confirmation.
The human-prompt protocol and tooling contain no participant result; collection
is locked until instructor/ethics approval and protocol freezing are recorded.
