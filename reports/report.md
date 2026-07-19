# Calibrated Prompt Uncertainty in Zero-Shot Segmentation

## Full-Validation Evidence from Lightweight Methods and SAM

**Anonymous manuscript**

## Abstract

Promptable segmentation systems are commonly evaluated with clean points and boxes, although real prompts are uncertain and different prompt channels do not admit directly comparable perturbation scales. We present a pre-specified evaluation of zero-shot point-and-box segmentation on all 1,449 PASCAL VOC 2012 validation images. A disjoint, class-balanced 100-image subset of VOC train is used only to calibrate point displacement and box perturbation to matched observable quality targets: aggregate point hit rate and mean box IoU. We compare a center-color baseline, GrabCut, a transparent robust-superpixel method, three component ablations, and SAM ViT-B under point-only, box-only, and point-plus-box prompting. Three primary hypotheses are fixed before confirmatory evaluation and tested with paired bootstrap intervals, sign-flip permutation tests, and Holm correction. The robust-superpixel method improves over center-color by 0.0640 mean IoU (95% CI [0.0585, 0.0696]); score-based SAM candidate selection improves over a single noisy prompt by 0.0193 [0.0135, 0.0251]; and, at approximately matched prompt quality, point-noise IoU exceeds box-noise IoU by 0.1069 [0.0997, 0.1138]. All three effects survive Holm correction. Stronger baselines and ablations qualify the contribution: GrabCut outperforms the proposed lightweight method, the color seed and spatial prior are useful, and multi-box consensus has essentially zero average benefit. The result is less a claim of a new state of the art than a reproducible account of which prompt channel fails, which lightweight components matter, and which apparent gains survive representative evaluation.

## 1. Introduction

Interactive and prompted segmentation methods convert sparse user input into an object mask. Extreme points, iterative clicks, boxes, and foundation-model prompts have progressively reduced the amount of supervision required at inference time [2–7]. Segment Anything (SAM) further established a general point-and-box interface that transfers across image domains without target-dataset fine-tuning [7].

Prompt quality remains a structured source of uncertainty. A displaced foreground point may still lie inside the object, while a shifted or resized box changes both localization and spatial support. Applying the same numeric noise scale to the two channels does not make them equally difficult. Prior studies have compared box, centroid, and random point prompts for SAM [8], proposed perturbed-prompt adaptation [9], and modeled SAM uncertainty [10]. Human-interaction studies also show that synthetic interactions do not automatically reproduce real annotation behavior [11]. These observations motivate an evaluation that measures the realized quality of each prompt channel rather than assuming equal perturbation strength.

This study asks three confirmatory questions. First, does a transparent robust-superpixel pipeline reliably improve over a minimal center-color method? Second, can SAM's own predicted mask score select better candidates than a single noisy prompt? Third, after point and box perturbations are calibrated to comparable aggregate quality, is SAM still more sensitive to box noise? We answer these questions on the complete VOC 2012 validation split, with a disjoint tuning split, strong baselines, component ablations, resource measurements, and family-wise error control.

The contributions are:

1. A data-free, hash-pinned protocol that separates tuning on VOC train from confirmation on all 1,449 VOC validation images.
2. Observable-quality calibration that targets the same aggregate quality level with point hit rate and box IoU.
3. Three pre-specified paired hypotheses with Holm correction across the primary family.
4. A transparent CPU method evaluated against Center Color, GrabCut, and SAM, with ablations for its color seed, spatial prior, and box consensus.
5. A reproducible metric artifact that records every sample, every SAM trial, algorithm failures, runtime, memory, and Oracle upper bounds without redistributing VOC imagery.

## 2. Related Work

### Interactive segmentation

GrabCut combines graph cuts with iterative foreground/background appearance models and remains a strong classical box-driven baseline [2]. DEXTR demonstrated that extreme-point input could condition deep object segmentation [4]. RITM and SimpleClick later improved iterative click-based segmentation and standardized evaluation across multiple datasets and interaction rounds [5,6]. Large-scale human studies have shown a measurable gap between simulated and actual interactions, making prompt-generation assumptions part of the evaluation protocol rather than an implementation detail [11].

### Promptable foundation segmentation

SAM introduced a prompt encoder and mask decoder supporting points, boxes, and masks under zero-shot transfer [7]. Direct prompting evaluations report that boxes can be substantially stronger than centroid or random point prompts [8]. PP-SAM studies robustness through perturbed prompts during adaptation [9], while UncertainSAM formalizes uncertainty estimation and tests it across datasets [10]. More recent work optimizes point prompts with search or learned agents [12]. These studies establish that prompt robustness and prompt selection are active topics; our narrower contribution is a controlled full-validation comparison with matched observable prompt quality and pre-specified paired inference.

### Superpixels and transparent baselines

SLIC provides efficient, spatially regular image primitives in Lab space [3]. Our lightweight method uses SLIC as a transparent representation, not as a learned contribution. The method is deliberately compared with GrabCut and ablated so that engineering complexity is not mistaken for evidence of superiority.

## 3. Methods

### 3.1 Task and target construction

For every VOC image, we enumerate semantic foreground labels 1–20 and select the largest connected foreground component. This defines one binary target per image. The clean box is the tight target rectangle. The clean point is the target pixel with maximum Euclidean distance from the component boundary. These prompts use ground-truth geometry and therefore define a controlled benchmark rather than a model of unaided human annotation.

No method is trained or fine-tuned on VOC validation. The proposed CPU methods use no learned weights. SAM uses its published ViT-B checkpoint.

### 3.2 Center Color

Center Color estimates an RGB foreground prototype from a disk around the point prompt. Pixel distances inside the box are thresholded using Otsu's method with a conservative upper percentile, followed by small-object removal, hole filling, closing, and point-connected component selection.

### 3.3 Robust Superpixel

The proposed method computes SLIC superpixels and represents each with mean Lab color and normalized spatial coordinates. Foreground prototypes come from point-intersecting superpixels; background prototypes come from outside the prompt box and its inner border. Relative prototype distances form a foreground score, which is combined with a weak spatial prior. The default method unions the superpixel prediction with the Center Color mask and applies morphological and point-connected cleanup.

The original exploratory implementation also voted across a slightly contracted, original, and expanded box. Confirmatory ablation shows this consensus has effectively no average benefit, so it is retained as a documented negative result rather than a central contribution.

### 3.4 GrabCut

GrabCut receives only the point and box prompts. Pixels outside the box initialize background, pixels inside initialize probable foreground, and a small point-centered disk initializes sure foreground. Five GrabCut iterations are used. Eight images cause OpenCV's GMM initialization to fail; these failures are recorded and scored as zero in aggregate results.

### 3.5 SAM and multi-prompt selection

SAM ViT-B is evaluated with point-only, box-only, and point-plus-box prompts. For noisy point-plus-box prompts, five additional candidates are sampled locally around the observed prompt without access to the clean prompt. We compare:

- the original single noisy prompt;
- the candidate with maximum SAM score;
- the consistency medoid under pairwise mask IoU;
- pixelwise majority vote;
- Oracle best-of-six, selected using target IoU.

The Oracle is an upper bound only. It is not a deployable selector and is excluded from all primary hypotheses.

## 4. Frozen Experimental Protocol

### 4.1 Data separation

The tuning split contains 100 VOC 2012 train images, exactly five targets for each of the 20 largest-component classes, sampled with seed 20260719. The confirmatory split contains all 1,449 VOC 2012 validation rows. The old 30-image subset is exploratory and is not included in confirmatory inference.

Manifests contain row IDs, class labels, target geometry, and hashes but no image or mask payloads. Source parquet files, manifests, algorithm sources, the official SAM archive, and the ViT-B checkpoint are pinned by SHA-256.

### 4.2 Prompt-noise calibration

On the tuning split only, deterministic grid search selects separate point and box scales for two quality targets. Point quality is aggregate target hit rate; box quality is mean IoU with the clean tight box. Twenty trials per tuning target are used.

| Severity | Target quality | Point scale | Tuning hit rate | Box scale | Tuning box IoU |
| --- | ---: | ---: | ---: | ---: | ---: |
| Mild | 0.85 | 0.1675 | 0.8480 | 0.0400 | 0.8538 |
| Moderate | 0.65 | 0.2725 | 0.6495 | 0.1150 | 0.6509 |

On the confirmatory split, moderate point hit rate is 0.6391 and moderate box IoU is 0.6514. The remaining 0.0123 difference is reported rather than corrected after observing validation results.

### 4.3 Hypotheses and statistics

The primary metric is sample-macro IoU. Dice, latency, memory, per-class results, and Oracle effects are secondary or descriptive. Five SAM perturbation trials are averaged within each sample before testing.

- H1: Robust Superpixel has higher mean IoU than Center Color under clean prompts.
- H2: SAM score selection has higher mean IoU than a single noisy prompt under moderate point-plus-box noise.
- H3: At matched moderate prompt quality, box noise causes a larger IoU loss than point noise; equivalently, point-noise IoU minus box-noise IoU is positive.

Each effect uses a paired 95% bootstrap interval with 20,000 replicates and a paired sign-flip test with 50,000 permutations. Raw p-values for H1–H3 are corrected together using Holm's step-down method at family-wise alpha 0.05.

## 5. Results

### 5.1 CPU methods and ablations

| Method | Mean IoU | Mean Dice | Median latency (all attempts) | Failures |
| --- | ---: | ---: | ---: | ---: |
| Center Color | 0.5404 | 0.6753 | 13.1 ms | 0 |
| GrabCut point+box | **0.6859** | **0.7751** | 416.8 ms | 8 |
| Robust Superpixel | 0.6044 | 0.7345 | 199.8 ms | 0 |
| without color seed | 0.4633 | 0.5977 | 187.0 ms | 0 |
| without spatial prior | 0.5827 | 0.7152 | 197.6 ms | 0 |
| single box | 0.6043 | 0.7344 | 188.5 ms | 0 |

H1 is supported: Robust Superpixel improves over Center Color by 0.0640 IoU, with 95% CI [0.0585, 0.0696] and Holm-adjusted p=0.000060. However, GrabCut is substantially stronger. The color seed contributes approximately 0.1411 IoU, and the spatial prior contributes 0.0217. Removing multi-box consensus changes mean IoU by only -0.00004, contradicting the exploratory intuition that box voting was an important source of robustness.

### 5.2 Clean SAM prompt modality

| Prompt | Mean IoU | Mean Dice |
| --- | ---: | ---: |
| Point only | 0.5389 | 0.6235 |
| Box only | 0.8479 | 0.9058 |
| Point + box | **0.8491** | **0.9087** |

Point-plus-box is only 0.0012 IoU above box-only on average. The full-validation result therefore refines the exploratory claim: SAM is strongly box-dominated, but box-only is not meaningfully stronger than point-plus-box in the aggregate.

### 5.3 Calibrated prompt noise

Moderate point noise yields 0.8194 mean IoU, whereas matched-quality box noise yields 0.7125. H3's paired difference is +0.1069, with 95% CI [0.0997, 0.1138] and Holm-adjusted p=0.000060. The result supports channel-specific sensitivity even after prompt quality is approximately matched.

### 5.4 Multi-prompt uncertainty

| Selector under moderate point+box noise | Mean IoU | Mean Dice |
| --- | ---: | ---: |
| Single noisy prompt | 0.6529 | 0.7410 |
| SAM score selection | **0.6723** | 0.7532 |
| Consistency medoid | 0.6758 | **0.7634** |
| Vote consensus | 0.6710 | 0.7620 |
| Oracle best-of-six | 0.7821 | 0.8601 |

H2 is supported: score selection improves over the single prompt by 0.0193 IoU, with 95% CI [0.0135, 0.0251] and Holm-adjusted p=0.000060. The Oracle gap is much larger (+0.1292 IoU), but it is descriptive and cannot be interpreted as deployable performance. Consistency medoid has a slightly higher uncorrected mean than score selection; it was not a pre-specified primary comparison and is reported as exploratory.

## 6. Discussion

The representative benchmark changes several conclusions drawn from the original 30-image experiment. First, the proposed CPU method reliably improves over its minimal color baseline, but it does not beat GrabCut. The appropriate contribution is therefore transparent component analysis and a speed–accuracy operating point, not state-of-the-art segmentation.

Second, the box channel remains the principal localization signal for SAM. This is consistent with prior variational prompting evaluations [8], so the observation itself is not novel. The stronger contribution is that the difference persists after point and box perturbations are separately calibrated and their realized quality is reported.

Third, SAM's native score becomes useful with adequate sample size: the 30-image exploratory confidence interval crossed zero, while the 1,449-image confirmatory interval is positive. The effect remains modest and incurs additional mask-decoder calls. Future work should compare this gain against latency-aware policies and independently calibrated confidence models.

Finally, negative results matter. Multi-box consensus adds complexity without aggregate gain, and the strongest CPU baseline is classical GrabCut rather than the proposed method. Reporting these findings prevents an engineering artifact from being presented as a stronger scientific contribution than the evidence supports.

## 7. Threats to Validity

The prompts are derived from target masks and are more controlled than real user input. Numeric matching between point hit rate and box IoU creates comparable aggregate quality indices, but the two metrics are not psychologically equivalent and are not fitted to a measured human-error distribution. Real-user studies remain necessary.

The target definition selects one largest component per image rather than evaluating every VOC object instance. The validation set is complete for this target-construction rule, but conclusions should not be generalized to instance-level evaluation without further work.

Only VOC 2012 and SAM ViT-B are tested. Cross-dataset replication, SAM 2, other checkpoint scales, and standard learned interactive systems such as RITM or SimpleClick would strengthen external validity. GrabCut has eight explicit runtime failures; scoring them as zero is conservative but makes its aggregate dependent on implementation robustness.

The tuning split is class-balanced by the largest-component label, while validation retains the natural class distribution. Algorithm constants predate the confirmatory run and source hashes prevent silent changes, but the broader research questions were motivated by exploratory results. Confirmation on a second dataset would provide stronger independence.

## 8. Conclusion

This work evaluates zero-shot point-and-box segmentation under a frozen, full-validation protocol. Robust Superpixel reliably improves over Center Color, but GrabCut is stronger. SAM is highly dependent on box localization: after point and box prompt quality is approximately matched, box noise still causes a substantially larger loss. SAM score selection yields a small but statistically confirmed improvement over a single noisy prompt, while Oracle selection remains only an upper bound. The study's main value is a reproducible separation of engineering quality, prompt-channel sensitivity, and evidence strength rather than a state-of-the-art claim.

## References

1. M. Everingham et al. The PASCAL Visual Object Classes Challenge. *IJCV*, 2010. https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/
2. C. Rother, V. Kolmogorov, and A. Blake. GrabCut: Interactive Foreground Extraction Using Iterated Graph Cuts. *ACM TOG*, 2004. https://www.microsoft.com/en-us/research/publication/grabcut-interactive-foreground-extraction-using-iterated-graph-cuts/
3. R. Achanta et al. SLIC Superpixels Compared to State-of-the-Art Superpixel Methods. *TPAMI*, 2012. https://doi.org/10.1109/TPAMI.2012.120
4. K.-K. Maninis et al. Deep Extreme Cut: From Extreme Points to Object Segmentation. *CVPR*, 2018. https://openaccess.thecvf.com/content_cvpr_2018/html/Maninis_Deep_Extreme_Cut_CVPR_2018_paper.html
5. K. Sofiiuk, I. Petrov, and A. Konushin. Reviving Iterative Training with Mask Guidance for Interactive Segmentation. *ICIP*, 2022. https://arxiv.org/abs/2102.06583
6. Q. Liu et al. SimpleClick: Interactive Image Segmentation with Simple Vision Transformers. *ICCV*, 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Liu_SimpleClick_Interactive_Image_Segmentation_with_Simple_Vision_Transformers_ICCV_2023_paper.html
7. A. Kirillov et al. Segment Anything. *ICCV*, 2023. https://openaccess.thecvf.com/content/ICCV2023/html/Kirillov_Segment_Anything_ICCV_2023_paper.html
8. Y. Gaus et al. Performance Evaluation of Segment Anything Model with Variational Prompting for Application to Non-Visible Spectrum. *CVPRW*, 2024. https://openaccess.thecvf.com/content/CVPR2024W/PBVS/html/Gaus_Performance_Evaluation_of_Segment_Anything_Model_with_Variational_Prompting_for_CVPRW_2024_paper.html
9. A. Rahman et al. PP-SAM: Perturbed Prompts for Robust Adaptation of Segment Anything Model for Domain-Generalized Medical Image Segmentation. *CVPRW*, 2024. https://openaccess.thecvf.com/content/CVPR2024W/DEF-AI-MIA/html/Rahman_PP-SAM_Perturbed_Prompts_for_Robust_Adaption_of_Segment_Anything_Model_CVPRW_2024_paper.html
10. T. Kaiser et al. UncertainSAM: Segment Anything with Uncertainty Estimation. *ICML*, 2025. https://proceedings.mlr.press/v267/kaiser25a.html
11. R. Benenson, S. Popov, and V. Ferrari. Large-Scale Interactive Object Segmentation with Human Annotators. *CVPR*, 2019. https://openaccess.thecvf.com/content_CVPR_2019/html/Benenson_Large-Scale_Interactive_Object_Segmentation_With_Human_Annotators_CVPR_2019_paper.html
12. Liu et al. Attack for Defense: Adversarial Agents for Point Prompt Optimization. *CVPR*, 2026. https://openaccess.thecvf.com/content/CVPR2026/html/Liu_Attack_for_Defense_Adversarial_Agents_for_Point_Prompt_Optimization_Empowering_CVPR_2026_paper.html
