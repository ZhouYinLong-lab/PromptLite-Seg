# PromptLite-Seg

> 面向 PASCAL VOC 2012 的轻量级零样本提示分割与提示不确定性研究
> Lightweight Zero-Shot Prompted Segmentation and Prompt Uncertainty on PASCAL VOC 2012

[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![CPU CI](https://github.com/ZhouYinLong-lab/PromptLite-Seg/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ZhouYinLong-lab/PromptLite-Seg/actions/workflows/ci.yml)
[![Dataset](https://img.shields.io/badge/Evaluation-VOC%202012%20val%201449-009688)](https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2012/)
[![Protocol](https://img.shields.io/badge/Protocol-Pre--specified-success)](protocol/research_protocol.json)
[![License: Apache-2.0](https://img.shields.io/badge/Code%20License-Apache%202.0-blue.svg)](LICENSE)

PromptLite-Seg 是南京大学 2026 春《人工智能导论》方向 7“零样本提示分割”的课程项目，现已扩展为一个具有冻结协议、完整验证集评测、强基线、组件消融和确认性统计的研究制品。项目研究如何只利用一个前景点和一个边界框获得目标掩码，并分析点提示、框提示和多候选选择在提示误差下的行为。

项目包含两条互补路径：完全免训练的 CPU 方法，以及使用官方 SAM ViT-B 的 CUDA 基准。所有最终结论基于 PASCAL VOC 2012 validation 的 **1,449 张图像和 20 个类别**；原有 30 样本实验仅保留为探索性历史，不再用于确认性主张。

## 创新点与研究贡献

1. **代表性、不可事后调参的评测协议。** VOC train 中固定抽取每类 5 个目标作为 100 样本调参集，VOC validation 的全部 1,449 行作为确认集；清单、数据源、算法文件与随机协议均由 SHA-256 固定。
2. **按可观测提示质量校准点噪声与框噪声。** 点扰动以目标命中率衡量，框扰动以相对紧框 IoU 衡量；两者先在调参集上独立校准到相同的 0.85/0.65 质量目标，再在确认集上检验，从而避免比较任意且不等强的扰动尺度。
3. **预设假设与多重检验控制。** 在查看完整确认结果前固定 H1–H3、主要指标和方向，先按样本聚合重复试验，再使用配对 Bootstrap（20,000 次）、sign-flip permutation test（50,000 次）和 Holm 家族错误率校正。
4. **自适应超像素分辨率方法。** 提出图像尺寸感知的 SLIC 段数自适应策略（80–500 段，与 √(HW) 成正比），在固定 280 段基础上提升 +0.0074 IoU（95% CI [0.0053, 0.0096]），小目标四分位的二次诊断增益为 +0.0145。该变体不增加模型拟合或推理调用；它是 H1–H3 之后的二次分析，不冒充预设主检验。
5. **多质量档稳健性曲线。** 在独立冻结的二级协议下，将点命中率和框 IoU 校准到 0.9–0.5 五档，并在全部 1,449 个验证样本上每档、每通道运行 20 次，共 289,800 条真实 SAM 结果。结果把 H3 的适用边界定位为：0.8–0.5 档框噪声更有害，0.9 档差异不可区分。
6. **全面分层分析与实用决策。** 提供 per-class、per-area-quartile、per-aspect-ratio 分解，识别 bicycle、small targets 和全图框 GrabCut 失败，并据此给出方法选择表。

这些贡献主要体现在实验设计、可复核证据链、分层诊断和负结果披露；项目不宣称 SLIC、GrabCut、SAM 或各组成技术本身首次提出。

## 确认性结果

H1–H3 的冻结协议、逐样本指标和统计摘要位于 [`artifacts/confirmatory/`](artifacts/confirmatory/README.md)；Adaptive Superpixel 的后确认性摘要及来源哈希单独位于 [`artifacts/secondary/`](artifacts/secondary/README.md)。所有制品均为 CSV/JSON/Markdown，不包含 VOC 图像、掩码或模型权重。

### CPU 方法与消融

| 方法 | Mean IoU | Mean Dice | 全部尝试的 Median latency | 失败数 |
| --- | ---: | ---: | ---: | ---: |
| Center Color | 0.5404 | 0.6753 | 13.1 ms | 0 |
| GrabCut（点 + 框） | **0.6859** | **0.7751** | 416.8 ms | 8/1449 |
| Robust Superpixel | 0.6044 | 0.7345 | 199.8 ms | 0 |
| Adaptive Superpixel（后确认性） | 0.6117 | 0.7410 | 199.3 ms | 0 |
| └ 无颜色种子 | 0.4633 | 0.5977 | 187.0 ms | 0 |
| └ 无空间先验 | 0.5827 | 0.7152 | 197.6 ms | 0 |
| └ 单框、无多框共识 | 0.6043 | 0.7344 | 188.5 ms | 0 |

GrabCut 的 8 次初始化失败按 IoU/Dice=0 计入总体均值，没有从结果中删除；根因是紧框覆盖全图后缺少确定背景样本，而不是从目标纹理推测出的 GMM 不稳定。消融表明颜色种子贡献最大，空间先验有稳定正贡献；多框共识的平均增益接近零，因此不再把它表述为已证实的主要优势。

### SAM 提示模态

| SAM ViT-B 提示 | Mean IoU | Mean Dice |
| --- | ---: | ---: |
| 仅点 | 0.5389 | 0.6235 |
| 仅框 | 0.8479 | 0.9058 |
| 点 + 框 | **0.8491** | **0.9087** |

### 三个预设主假设

| 假设 | 配对 Mean IoU 差值 | 95% CI | Holm-adjusted p | 结论 |
| --- | ---: | ---: | ---: | --- |
| H1：Robust Superpixel > Center Color | +0.0640 | [0.0585, 0.0696] | 0.000060 | 支持 |
| H2：SAM 分数选择 > 单次含噪提示 | +0.0193 | [0.0135, 0.0251] | 0.000060 | 支持 |
| H3：匹配质量后框噪声损失 > 点噪声损失 | +0.1069 | [0.0997, 0.1138] | 0.000060 | 支持 |

确认集上的中等点噪声命中率为 **0.6391**，中等框噪声平均 IoU 为 **0.6514**，与预设 0.65 目标接近。H3 的正差值表示点噪声下 IoU 高于框噪声下 IoU，即在近似匹配提示质量后，SAM 仍对框定位误差更敏感。

Oracle 多候选结果只作为候选集合的描述性上界，不参与可部署方法主张或 H1–H3 检验。

### 二级多质量档敏感性分析

该分析在 H1–H3 之后独立冻结，不追溯性改写主假设。100 个 tuning 样本用于五档校准，全部 1,449 个 validation 样本各运行 20 次点扰动和 20 次框扰动。

| 质量目标 | 实测点命中率 | 实测框 IoU | 点噪声 SAM IoU | 框噪声 SAM IoU | 配对差值（点−框）95% CI |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.9 | 0.8909 | 0.8957 | 0.8395 | 0.8411 | −0.0016 [−0.0041, 0.0009] |
| 0.8 | 0.7859 | 0.8005 | 0.8306 | 0.8068 | 0.0238 [0.0201, 0.0275] |
| 0.7 | 0.6826 | 0.6989 | 0.8217 | 0.7462 | 0.0756 [0.0703, 0.0809] |
| 0.6 | 0.5748 | 0.5981 | 0.8133 | 0.6636 | 0.1497 [0.1430, 0.1563] |
| 0.5 | 0.4697 | 0.4982 | 0.8024 | 0.5700 | 0.2324 [0.2246, 0.2402] |

0.9 档区间跨零；0.8–0.5 档方向一致且差距随质量下降增大。点命中率与框 IoU 只是可观测校准量，不代表人类感知信息量等价。机器可读协议、289,800 行指标和统计摘要见 [`artifacts/secondary/prompt_quality_sensitivity/`](artifacts/secondary/prompt_quality_sensitivity/README.md)。

### ADE20K 有界迁移观察

完整扫描 ADE20K validation 的 2,000 行后，以固定种子 20260720 无放回抽取 200 行（10%）运行七种 CPU 方法，共得到 1,400 行指标且无方法失败。GrabCut、Adaptive Superpixel、固定 Robust Superpixel 和 Center Color 的 Mean IoU 分别为 0.7383、0.7288、0.7150 和 0.6350；Adaptive 相对固定分辨率描述性提高 0.0138 IoU。该结果扩大了跨数据集覆盖，但仍是二级有界样本，不等同于完整 2,000 样本确认。清单、选择 ID、逐样本指标和哈希见 [`artifacts/secondary/ade20k/`](artifacts/secondary/ade20k/README.md)。

### 真人提示试点状态

仓库提供不采集 PII 的本地标注界面、验证器、参与者聚类分析脚本和 [`protocol/human_pilot_protocol.json`](protocol/human_pilot_protocol.json)。在导师/伦理审查状态记为批准且协议冻结之前，真实采集会硬性拒绝启动。当前没有招募参与者，因而**没有真人提示结果，也不以合成演示数据代替真人证据**。

## 方法

### Center Color

在提示点附近估计 RGB 前景原型，在框内自适应阈值分割并进行连通域与形态学清理。它速度快，但容易受物体多色外观和相似背景影响。

### Robust Superpixel

使用 SLIC 超像素、Lab 颜色与空间特征建立前景/背景原型，将颜色种子和超像素预测融合，再保留与提示点关联的连通域。实现提供显式组件开关，使颜色种子、空间先验和框共识可以独立消融。

### Adaptive Superpixel

SLIC 段数不再固定为 280，而是根据图像尺寸自适应：n = max(80, min(500, ⌊√(HW) · 2.0⌋))。大图获得更细的超像素粒度，小图使用更紧凑的段数。相对固定 280 段，配对平均增益为 +0.0074 IoU；small-target 四分位的二次诊断增益为 +0.0145。由于公式读取的是图像尺寸而非目标尺寸，四分位趋势只作为支持性证据，不解释为目标尺寸的因果效应。

### GrabCut

使用边界框初始化 probable foreground、框外初始化 background，并以点邻域提供 sure foreground。该经典强基线显著超过自研轻量方法，但耗时更长；当紧框覆盖全图、框外没有确定背景像素时会初始化失败（共 8 例）。

### SAM ViT-B

使用官方 Segment Anything ViT-B，比较 point-only、box-only 和 point+box。提示不确定性实验对点、框分别施加校准噪声，并比较单提示、模型分数选择、一致性 medoid、投票共识和 Oracle 上界。

## 冻结实验协议

- 起始基线：`e464cf5359e7325ca4af3401d089c73a966de7dc`
- 调参集：VOC 2012 train，20 类 × 5 个目标，共 100 个
- 确认集：VOC 2012 validation 全部 1,449 行
- 主要指标：样本级宏平均 IoU；Dice、耗时和内存为次要指标
- 主检验：20,000 次配对 Bootstrap；50,000 次 sign-flip permutation
- 多重校正：H1–H3 使用 Holm step-down
- SAM：ViT-B、每样本 5 次噪声 trial、5 个额外候选
- 失败策略：明确记录；算法运行失败按 0 分计入总体指标
- Oracle 策略：仅作描述性上界

完整机器可读协议见 [`protocol/research_protocol.json`](protocol/research_protocol.json)，数据清单及哈希见 [`protocol/manifests/`](protocol/manifests/)。确认运行还会核对物化数据的整体指纹、运行时代码清单和运行前后 Git 状态；任一项漂移都会使结果失去 confirmatory 标记。

## 从干净环境复现

### 1. CPU 环境

要求 Python 3.10、3.11 或 3.12。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m compileall -q src scripts tests
python -m pytest
```

### 2. 下载并验证数据源

下载脚本会拒绝任何与冻结 SHA-256 不一致的数据文件。物化器只允许写入仓库的 `data/` 子目录；`--replace` 还要求目标目录带有本项目创建的所有权标记，避免误删任意路径。

```bash
python scripts/fetch_protocol_assets.py
python scripts/prepare_voc_from_manifest.py \
  --manifest protocol/manifests/tuning_train.jsonl \
  --parquet data/cache/pascal_voc_2012_train.parquet \
  --output-dir data/voc_tuning
python scripts/prepare_voc_from_manifest.py \
  --manifest protocol/manifests/confirmatory_validation.jsonl \
  --parquet data/cache/pascal_voc_2012_val.parquet \
  --output-dir data/voc_validation
```

### 3. CPU 确认性实验

```bash
python scripts/calibrate_prompt_noise.py
python scripts/run_confirmatory_cpu.py --workers 8 --methods \
  center_color grabcut_point_box robust_superpixel \
  robust_no_color_seed robust_no_spatial_prior robust_single_box
```

### 4. SAM CUDA 实验

复现论文所用的 CUDA 12.8 环境可直接安装已固定版本的 PyTorch、TorchVision 和官方 SAM 源码：

```bash
python -m pip install -r requirements-sam-cu128.txt
python scripts/fetch_protocol_assets.py --include-sam
python scripts/run_confirmatory_sam.py --device cuda
python scripts/analyze_confirmatory.py
```

其他 CUDA 或 CPU 平台请先按照 [PyTorch 官方安装器](https://pytorch.org/get-started/locally/) 选择兼容 wheel，再运行 `python -m pip install -r requirements-sam.txt`。不要安装 PyPI 上来源不明确的同名包。

本次验证环境为 Python 3.10、PyTorch 2.11.0+cu128、RTX 5070 Ti；在保留源 JPEG 字节的数据物化修复后，完整 SAM 运行耗时 744 秒，峰值 CUDA allocated memory 约 2.77 GB。不同硬件的耗时与显存可能不同，但指标应由相同清单、提示和检查点确定。

依赖扫描对该 PyTorch 版本报告一项低危 `torch.jit.script` 相关告警。本项目的执行路径不调用 TorchScript/JIT，也只接受固定 SHA-256 的官方 SAM 权重；为了不在确认性结果后改变 CUDA 运行栈，当前复现环境保持冻结。部署到接收不可信模型或启用 JIT 的环境时，应改用 PyTorch 已修复版本并重新验证结果。

## 项目结构

```text
PromptLite-Seg/
├── artifacts/confirmatory/       # H1–H3 无图像逐样本指标、统计与校验和
├── artifacts/secondary/          # 后确认性 Adaptive 摘要与来源哈希
├── protocol/                     # 冻结协议、调参/确认清单、噪声校准
├── reports/                      # 技术报告与匿名投稿稿件
├── scripts/                      # 资产验证、实验、统计、可视化和制品入口
├── src/promptseg/
│   ├── algorithms.py             # CPU 基线、GrabCut、自适应方法与消融
│   ├── ade.py                    # ADE20K 数据集加载与目标构造
│   ├── dataset.py                # 本地数据读取
│   ├── prompts.py                # 校准扰动、质量指标和候选选择
│   ├── sam.py                    # 可 mock 的 SAM predictor contract
│   └── voc.py                    # VOC 解码、目标构造与哈希工具
├── tests/                        # CPU、SAM mock、ADE20K、协议与 CLI 测试
├── RESEARCH_PLAN.md              # 持久化研究升级计划
├── THIRD_PARTY_NOTICES.md        # 数据、模型与依赖许可边界
└── LICENSE                       # 仅覆盖原创代码与文档
```

## 数据、隐私与许可证

仓库不再跟踪 VOC 原图、语义掩码、目标掩码、含原图可视化或 SAM checkpoint。用户必须自行取得第三方资产并遵守原始条款；详细边界和固定哈希见 [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md)。

原创代码和文档采用 [Apache License 2.0](LICENSE)。该许可证不覆盖 PASCAL VOC、Flickr 图片、Meta SAM 权重或第三方依赖。

历史提交曾包含 30 张 VOC 样本，因此当前私有开发分支不能直接切换为公开仓库。公开发布应使用不继承旧历史的 clean-history 分支或独立仓库，并在发布前执行资产扫描。

## 局限性

- 点和框仍由真值掩码自动构造；校准使两种噪声在可观测质量上接近，但不等同于真实人类交互分布。
- 只有 VOC 2012 和 SAM ViT-B 获得全规模验证；ADE20K 完整扫描 2,000 行后仅评测固定抽取的 200 行，仍不构成全量跨数据集确认，SAM 2、ViT-L/H 和真人提示尚未验证。
- GrabCut 的较高总体性能说明自研轻量方法不是当前最强方法；其价值主要在透明组件分析和较低延迟。
- 多框共识在完整验证集上没有平均收益，是应保留的负结果。
- SAM 分数选择的平均提升虽经确认，但幅度为 +0.0193 IoU，实际价值仍需结合额外推理成本评估。

## 报告、匿名投稿与引用

**课程提交只能使用 [`reports/report_anonymous.pdf`](reports/report_anonymous.pdf)。** 该文件使用 NeurIPS 模板，正文（含摘要）严格限定为第 1--7 页，第 8 页开始为参考文献，第 9 页起为附录，并通过身份与 PDF 元数据审计。[`reports/report.pdf`](reports/report.pdf) 是内部留存的具名版本，**不得上传到双盲评审系统**。面向双盲 OpenReview 会场时，也不得直接提交当前具名仓库链接；请生成经过身份、资产与路径审计的补充包：

```bash
python scripts/export_anonymous_artifact.py
```

默认产物为 `dist/promptlite-seg-anonymous.zip`。OpenReview 是投稿平台，具体匿名和代码政策仍以目标 venue 当年规则为准。

项目引用信息将在正式公开版本和论文题目冻结后给出。当前研究制品如需内部引用，可使用仓库提交与确认性制品校验和定位。

## 贡献与问题反馈

贡献前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，安全问题请按 [`SECURITY.md`](SECURITY.md) 私下报告。普通复现问题可提交 GitHub Issue，并附上命令、Python/PyTorch 版本、设备信息和完整错误输出。
