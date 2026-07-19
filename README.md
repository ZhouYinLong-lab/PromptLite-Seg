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
3. **预设假设与多重检验控制。** 在查看完整确认结果前固定 H1–H3、主要指标和方向，先按样本聚合重复试验，再使用配对 Bootstrap、sign-flip permutation test 和 Holm 家族错误率校正。
4. **透明轻量方法的强基线与逐组件审计。** 除 Center Color 外加入 GrabCut 和 SAM 三种提示模态，并分别移除颜色种子、空间先验和多框共识。消融显示哪些组件真正贡献性能，也诚实暴露多框共识在完整验证集上几乎没有平均收益。

这些贡献主要体现在实验设计、可复核证据链和负结果披露；项目不宣称 SLIC、GrabCut、SAM 或各组成技术本身首次提出。

## 确认性结果

冻结协议、逐样本指标和统计摘要位于 [`artifacts/confirmatory/`](artifacts/confirmatory/README.md)。所有制品均为 CSV/JSON/Markdown，不包含 VOC 图像、掩码或模型权重。

### CPU 方法与消融

| 方法 | Mean IoU | Mean Dice | Median latency | 失败数 |
| --- | ---: | ---: | ---: | ---: |
| Center Color | 0.5404 | 0.6753 | 13.1 ms | 0 |
| GrabCut（点 + 框） | **0.6859** | **0.7751** | 416.8 ms | 8/1449 |
| Robust Superpixel | 0.6044 | 0.7345 | 199.8 ms | 0 |
| └ 无颜色种子 | 0.4633 | 0.5977 | 187.0 ms | 0 |
| └ 无空间先验 | 0.5827 | 0.7152 | 197.6 ms | 0 |
| └ 单框、无多框共识 | 0.6043 | 0.7344 | 188.5 ms | 0 |

GrabCut 的 8 次初始化失败按 IoU/Dice=0 计入总体均值，没有从结果中删除。消融表明颜色种子贡献最大，空间先验有稳定正贡献；多框共识的平均增益接近零，因此不再把它表述为已证实的主要优势。

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

## 方法

### Center Color

在提示点附近估计 RGB 前景原型，在框内自适应阈值分割并进行连通域与形态学清理。它速度快，但容易受物体多色外观和相似背景影响。

### Robust Superpixel

使用 SLIC 超像素、Lab 颜色与空间特征建立前景/背景原型，将颜色种子和超像素预测融合，再保留与提示点关联的连通域。实现提供显式组件开关，使颜色种子、空间先验和框共识可以独立消融。

### GrabCut

使用边界框初始化 probable foreground、框外初始化 background，并以点邻域提供 sure foreground。该经典强基线显著超过自研轻量方法，但耗时更长，并在 8 个极端初始化案例上失败。

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

完整机器可读协议见 [`protocol/research_protocol.json`](protocol/research_protocol.json)，数据清单及哈希见 [`protocol/manifests/`](protocol/manifests/)。

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

下载脚本会拒绝任何与冻结 SHA-256 不一致的数据文件。

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
python scripts/run_confirmatory_cpu.py --workers 8
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

## 项目结构

```text
PromptLite-Seg/
├── artifacts/confirmatory/       # 无图像的逐样本指标、统计与校验和
├── protocol/                     # 冻结协议、调参/确认清单、噪声校准
├── reports/                      # 技术报告与匿名投稿稿件
├── scripts/                      # 资产验证、实验、统计和制品入口
├── src/promptseg/
│   ├── algorithms.py             # CPU 基线、GrabCut、主方法与消融
│   ├── dataset.py                # 本地数据读取
│   ├── prompts.py                # 校准扰动、质量指标和候选选择
│   ├── sam.py                    # 可 mock 的 SAM predictor contract
│   └── voc.py                    # VOC 解码、目标构造与哈希工具
├── tests/                        # CPU、SAM mock、协议与 CLI 测试
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
- 研究只覆盖 VOC 2012 和 SAM ViT-B；跨数据集、SAM 2、ViT-L/H 或真实用户提示尚未验证。
- GrabCut 的较高总体性能说明自研轻量方法不是当前最强方法；其价值主要在透明组件分析和较低延迟。
- 多框共识在完整验证集上没有平均收益，是应保留的负结果。
- SAM 分数选择的平均提升虽经确认，但幅度为 +0.0193 IoU，实际价值仍需结合额外推理成本评估。

## 报告、匿名投稿与引用

课程报告位于 [`reports/report.pdf`](reports/report.pdf)，双盲稿为 [`reports/report_anonymous.pdf`](reports/report_anonymous.pdf)。面向双盲 OpenReview 会场时，不得直接提交当前具名仓库链接；请生成经过身份、资产与路径审计的补充包：

```bash
python scripts/export_anonymous_artifact.py
```

默认产物为 `dist/promptlite-seg-anonymous.zip`。OpenReview 是投稿平台，具体匿名和代码政策仍以目标 venue 当年规则为准。

项目引用信息将在正式公开版本和论文题目冻结后给出。当前研究制品如需内部引用，可使用仓库提交与确认性制品校验和定位。

## 贡献与问题反馈

贡献前请阅读 [`CONTRIBUTING.md`](CONTRIBUTING.md)，安全问题请按 [`SECURITY.md`](SECURITY.md) 私下报告。普通复现问题可提交 GitHub Issue，并附上命令、Python/PyTorch 版本、设备信息和完整错误输出。
