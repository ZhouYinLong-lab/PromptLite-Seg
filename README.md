# PromptLite-Seg

**PromptLite-Seg: Lightweight Zero-Shot Prompted Segmentation on PASCAL VOC 2012**

Repository: https://github.com/ZhouYinLong-lab/PromptLite-Seg

This repository contains the course-design project for Direction 7: zero-shot prompt segmentation. It evaluates prompt-driven segmentation algorithms on a small reproducible subset of PASCAL VOC 2012 and produces metrics, qualitative visualizations, and an English technical report.

## Project Idea

The project studies whether a segmentation mask can be obtained from only a weak prompt, such as a center point and a bounding box, without training on the target dataset. The implemented methods are:

- `center_color`: a simple point-and-box color threshold baseline.
- `robust_superpixel`: a training-free prompt segmentation method using superpixels, foreground/background prototypes, box constraints, and perturbation consensus.

The code also keeps the project structure ready for adding a SAM/SAM2 backend, while the default experiment is intentionally lightweight and CPU-friendly.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run

Download a compact PASCAL VOC 2012 validation subset:

```powershell
python scripts/download_voc_subset.py --count 30
```

Run the experiment:

```powershell
python scripts/run_experiment.py --data-dir data/voc_subset --output-dir outputs --max-samples 30
```

Generate per-class and success/failure analysis:

```powershell
python scripts/analyze_results.py --metrics outputs/metrics.csv --output-dir outputs/analysis
```

The main outputs are:

- `outputs/metrics.csv`: per-sample IoU and Dice.
- `outputs/summary.json`: aggregate metrics.
- `outputs/figures/*.png`: qualitative predictions.
- `outputs/figures/metric_summary.png`: metric comparison plot.
- `outputs/analysis/per_class_summary.csv`: per-class metrics.
- `outputs/analysis/success_failure.md`: success and failure case analysis.

## Report

The English technical report is in `reports/report.md`.
