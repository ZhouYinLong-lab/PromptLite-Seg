# PromptLite-Seg: Lightweight Zero-Shot Prompted Segmentation on PASCAL VOC 2012

Repository: https://github.com/ZhouYinLong-lab/PromptLite-Seg

## Abstract

PromptLite-Seg studies zero-shot prompted segmentation on PASCAL VOC 2012. Given a weak prompt, consisting of a center point and a bounding box, the goal is to segment the target object without training on the benchmark. I implement a simple color-threshold baseline, a stronger training-free robust superpixel method, and an optional GPU SAM ViT-B comparison. The robust method uses foreground/background prototypes induced by the prompt, a box constraint, and perturbation consensus. The project produces quantitative IoU/Dice results and qualitative visualizations on a reproducible VOC 2012 validation subset.

The final version adds a prompt-robustness benchmark. Instead of evaluating only perfect center-point and bounding-box prompts, it perturbs both prompts and measures degradation under mild and moderate prompt noise. This turns the project from a single-score comparison into an analysis of prompt sensitivity.

## 1. Introduction

Prompted segmentation has become a practical interface for general-purpose vision systems. Models such as the Segment Anything Model (SAM) show that segmentation can be controlled by prompts such as points and boxes. However, large pretrained models can be expensive to run in a course project environment. This project therefore asks a narrower but useful question: how far can a lightweight, training-free prompted segmentation algorithm go on real benchmark images?

The task follows the course-design direction of zero-shot prompt segmentation. The benchmark is PASCAL VOC 2012, and the algorithm is evaluated from ground-truth-derived prompts. For each sample, I select the largest semantic object component, convert it into a binary target mask, and derive a tight bounding box and an interior center point.

## 2. Method

### 2.1 Prompt Generation

For each VOC validation image, the semantic mask provides class labels from 1 to 20 and background label 0. I select the largest connected foreground component as the target object. The bounding-box prompt is the tight box around this component. The point prompt is chosen as the pixel with maximum distance-transform value inside the component, which approximates a stable central click.

### 2.2 Center-Color Baseline

The baseline uses only the prompt point and bounding box. It estimates the target color by taking the median RGB value in a small disk around the point. Pixels inside the bounding box are segmented if their color distance to this prototype is below an adaptive Otsu threshold. Finally, small components are removed and the component nearest to the point is kept.

### 2.3 Robust Superpixel Prompt Segmentation

The proposed method first runs the center-color baseline to obtain a conservative foreground seed. It then oversegments the image into SLIC superpixels. Each superpixel is represented by mean Lab color and normalized spatial coordinates. Foreground prototypes are estimated from superpixels intersecting the center-point disk, while background prototypes are estimated from superpixels outside the prompt box and along the box border. Each superpixel receives a foreground score based on relative distance to these prototypes, combined with a weak spatial prior centered at the prompt point. The final mask is the union of the conservative color seed and the superpixel consensus across slightly perturbed boxes, followed by morphology and point-connected component selection.

This method is zero-shot because it uses no training examples, no VOC labels except for prompt construction during evaluation, and no learned parameters.

## 3. Experimental Setup

The experiment uses PASCAL VOC 2012 validation samples from the Hugging Face mirror `nateraw/pascal-voc-2012`. A compact subset is produced by `scripts/download_voc_subset.py`, which stores the RGB image, semantic mask, target binary mask, bounding box, and center point for each sample.

Metrics:

- Intersection over Union (IoU)
- Dice score

Commands:

```bash
python scripts/download_voc_subset.py --count 30
python scripts/run_experiment.py --data-dir data/voc_subset --output-dir outputs --max-samples 30
```

## 4. Results

The experiment was run on 30 VOC 2012 validation samples. Aggregate results are stored in `outputs/summary.json`, and per-sample results are stored in `outputs/metrics.csv`.

| Method | Mean IoU | Std IoU | Mean Dice | Std Dice |
| --- | ---: | ---: | ---: | ---: |
| Center-color baseline | 0.5468 | 0.2323 | 0.6754 | 0.2122 |
| Robust superpixel | 0.6031 | 0.2111 | 0.7277 | 0.1895 |
| SAM ViT-B point+box | 0.8325 | 0.1484 | 0.9002 | 0.1044 |

The robust superpixel method improves mean IoU by 0.0562 absolute points over the center-color baseline and improves mean Dice by 0.0523. The lower standard deviation also suggests that the method is less sensitive to difficult images where the center color is not representative of the whole object.

The optional GPU SAM ViT-B point+box comparison reaches 0.8325 mean IoU and 0.9002 mean Dice, confirming that the same prompts are strong enough for a modern foundation segmentation model. The lightweight method remains useful as a transparent SAM-free baseline and as a low-cost diagnostic method.

## 4.1 Prompt Robustness

To evaluate prompt sensitivity, `scripts/run_robustness_experiment.py` perturbs both the center point and the bounding box. Mild perturbations shift the point and box by roughly 5-6% of object size; moderate perturbations use roughly 10-12%. Each non-clean severity is evaluated with two deterministic trials per sample.

| Severity | Method | Mean IoU | Mean Dice |
| --- | --- | ---: | ---: |
| clean | robust_superpixel | 0.6031 | 0.7277 |
| mild | robust_superpixel | 0.5579 | 0.6927 |
| moderate | robust_superpixel | 0.4568 | 0.5967 |
| clean | SAM ViT-B noisy prompt | 0.8325 | 0.9002 |
| mild | SAM ViT-B noisy prompt | 0.7731 | 0.8537 |
| moderate | SAM ViT-B noisy prompt | 0.6756 | 0.7795 |

The robustness benchmark reveals the main limitation of prompt segmentation: SAM has much higher absolute accuracy, but it still loses about 0.157 IoU under moderate prompt noise. A naive lightweight prompt repair strategy does not improve SAM; it reduces moderate-noise SAM IoU from 0.6756 to 0.5745. SAM's own score-based selection also underperforms the original noisy prompt at 0.6118. This is a useful negative result: transparent repair masks are helpful for analysis, but SAM's native point+box interface already preserves more information than the repaired prompt.

![Prompt robustness](../outputs/robustness/robustness_curve.png)

The qualitative figures in `outputs/figures/` compare the original prompt, the ground-truth target, and the predictions of both methods. The summary bar chart is saved as `outputs/figures/metric_summary.png`.

![Metric summary](../outputs/figures/metric_summary.png)

![Qualitative example](../outputs/figures/sample_001.png)

Additional analysis is generated by `scripts/analyze_results.py`. On the 30-sample subset, the robust method wins on 23 samples, ties on 5 samples, and loses on only 2 samples relative to the center-color baseline. The largest gains appear on horse, tvmonitor, and aeroplane examples, where pure center-color thresholding tends to under-cover the object.

![Per-class IoU](../outputs/analysis/per_class_iou.png)

![IoU improvement histogram](../outputs/analysis/iou_delta_histogram.png)

![SAM comparison](../outputs/sam/method_comparison.png)

## 5. Analysis

The center-color baseline works when the object has a compact appearance and contrasts clearly with the local background. It fails when the object has multiple colors, when the center click lands on a non-representative part, or when the surrounding background has similar color.

The robust superpixel method improves the prompt interface in three ways. First, superpixels respect local image boundaries better than raw pixel color thresholding. Second, the method uses background evidence from the box boundary and outside-box region, which helps suppress leakage to nearby clutter. Third, perturbing the box and taking consensus reduces sensitivity to one exact prompt box.

The method still struggles with thin structures, complex object interiors, and cases where the bounding box contains multiple visually similar objects. It can also over-segment when the target object and background share large superpixels or similar Lab color statistics.

The hardest cases in the current subset include train, bicycle, diningtable, and boat examples. These cases expose different failure modes: train and diningtable instances often contain repetitive interior structures, bicycle masks contain thin parts that are easy to miss, and boat images may have strong background colors inside the prompt box.

Compared with SAM-style models, this method is much weaker in semantic understanding, but it is transparent, CPU-friendly, and easy to analyze. The implemented SAM ViT-B comparison shows the performance ceiling available from a foundation segmentation model, while the proposed method provides a reproducible low-resource baseline.

The robustness experiment adds a stronger research conclusion. The project does not merely show that SAM is better; it quantifies how much prompt noise hurts different methods and shows that naive prompt repair is not automatically beneficial for foundation models. This gives a clearer direction for future work: repair should preserve prompt uncertainty rather than collapse it to a single mask-derived point and box.

## 6. Conclusion

This project implements and evaluates a zero-shot prompted segmentation pipeline on PASCAL VOC 2012. The main contribution is a reproducible CPU-friendly baseline and an enhanced robust superpixel method that uses prompt-derived foreground/background cues. The project satisfies the course requirement of running at least one algorithm on a benchmark and provides a foundation for further experiments with SAM-family models.

## References

1. Mark Everingham, Luc Van Gool, Christopher K. I. Williams, John Winn, and Andrew Zisserman. The PASCAL Visual Object Classes Challenge 2012. https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/
2. Alexander Kirillov et al. Segment Anything. ICCV 2023. https://arxiv.org/abs/2304.02643
3. Radhakrishna Achanta et al. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. IEEE TPAMI 2012. https://www.epfl.ch/labs/ivrl/research/slic-superpixels/
