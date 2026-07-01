# PromptLite-Seg

**PromptLite-Seg: Lightweight Zero-Shot Prompted Segmentation on PASCAL VOC 2012**

Repository: https://github.com/ZhouYinLong-lab/PromptLite-Seg

This repository contains the course-design project for Direction 7: zero-shot prompt segmentation. It evaluates prompt-driven segmentation algorithms on a small reproducible subset of PASCAL VOC 2012 and produces metrics, qualitative visualizations, and an English technical report.

## Project Idea

The project studies whether a segmentation mask can be obtained from only a weak prompt, such as a center point and a bounding box, without training on the target dataset. The implemented methods are:

- `center_color`: a simple point-and-box color threshold baseline.
- `robust_superpixel`: a training-free prompt segmentation method using superpixels, foreground/background prototypes, box constraints, and perturbation consensus.

The default experiment is intentionally lightweight and CPU-friendly. With a CUDA GPU, the repository also runs SAM ViT-B experiments that turn the project into a small research study of prompt uncertainty: prompt modality, point-vs-box noise, and multi-prompt selection.

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

Optional GPU SAM comparison:

```powershell
python -m venv .venv-sam
.\.venv-sam\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv-sam\Scripts\python.exe -m pip install -r requirements-sam.txt
Invoke-WebRequest -Uri "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" -OutFile "checkpoints/sam_vit_b_01ec64.pth"
.\.venv-sam\Scripts\python.exe scripts/run_sam_experiment.py --max-samples 30
```

Run the deeper prompt-robustness benchmark:

```powershell
.\.venv-sam\Scripts\python.exe scripts/run_robustness_experiment.py --max-samples 30 --trials 2 --include-sam --device cuda
```

Run the research-style prompt uncertainty benchmark:

```powershell
.\.venv-sam\Scripts\python.exe scripts/run_prompt_uncertainty_experiment.py --max-samples 30 --trials 2 --ensemble-size 5 --device cuda
```

The main outputs are:

- `outputs/metrics.csv`: per-sample IoU and Dice.
- `outputs/summary.json`: aggregate metrics.
- `outputs/figures/*.png`: qualitative predictions.
- `outputs/figures/metric_summary.png`: metric comparison plot.
- `outputs/analysis/per_class_summary.csv`: per-class metrics.
- `outputs/analysis/success_failure.md`: success and failure case analysis.
- `outputs/sam/summary.json`: optional SAM point+box prompt comparison.
- `outputs/sam/method_comparison.png`: three-method comparison chart.
- `outputs/robustness/summary.csv`: prompt perturbation robustness metrics.
- `outputs/robustness/robustness_curve.png`: robustness curve under clean, mild, and moderate prompt noise.
- `outputs/prompt_uncertainty/summary.csv`: prompt modality, noise decomposition, and multi-prompt uncertainty metrics.
- `outputs/prompt_uncertainty/prompt_modality.png`: point-only, box-only, and point+box SAM comparison.
- `outputs/prompt_uncertainty/noise_decomposition.png`: point-noise versus box-noise sensitivity.
- `outputs/prompt_uncertainty/uncertainty_ensemble.png`: multi-prompt selection under moderate prompt noise.

## Key Finding

The strongest new result is that SAM ViT-B is more box-dominated than point-dominated on this benchmark. Clean box-only prompts reach 0.8565 mean IoU, while clean point+box prompts reach 0.8325 and point-only prompts reach 0.4430. Under moderate noise, point noise remains near clean performance at 0.8507 IoU, but box noise drops to 0.6462. A multi-prompt score-selection strategy recovers moderate point+box noise from 0.6385 to 0.6796 IoU, with an oracle upper bound of 0.7401.

## Report

The English technical report is in `reports/report.md`. A formal LaTeX/PDF version is available at `reports/report.tex` and `reports/report.pdf`.
