---
title: "Addressing-Blind-Guessing-Calibration-of-Selection-Bias-in-M"
source: https://aclanthology.org/2025.acl-long.162.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 01:39:58"
field: "多模态模型公平性与鲁棒性"
keywords: ["selection bias", "video language models", "multiple-choice question answering", "bias calibration", "post-processing debiasing", "VLM evaluation"]
innovations: ["首次系统分析VLM在MCQA中的选择偏差并提出分解式后处理校准方法BOLD", "将公平性偏差指标适配到视频MCQA并引入JS_std/F1_std/Recall_std多维评估", "Weighted_BOLD通过COBYLA优化器在五折CV中学习各分解维度的自适应权重"]
benchmarks: ["NExT-QA", "STAR", "Perception Test", "Video-MME"]
---

# 论文速读：Addressing-Blind-Guessing-Calibration-of-Selection-Bias-in-M

## 一句话总结
本文首次系统研究了视频语言模型（VLM）在多选题问答（MCQA）中的选择偏差（selection bias）问题，即模型倾向于根据选项位置而非内容作答。作者提出了一种后处理校准方法 BOLD（Bias Optimisation Leveraging Decomposition），通过分解任务组件来估计并消除偏差，在减少选择偏差的同时还能提升模型的整体准确率与 F1 分数。

## 研究问题与动机
- 现有的 MCQA 基准测试依赖准确率评估 VLM，但无法捕捉模型的真实推理能力，因为模型存在严重的**选择偏差**——偏好特定位置的选项，而不顾其内容。
- 选择偏差在纯语言模型（LLM）中已有研究，但在**视频-语言模型**领域几乎空白，缺乏针对视频推理特有的动态事件理解、时空追踪等维度的偏差分析。
- 现有去偏差方法（如数据增强、选项重排、随机化）成本高且不够鲁棒，部分方法过度依赖先验shuffle假设，偏差与模型置信度之间并无稳定相关性。
- 不同 VLM 架构（Video-LLaMA、Video-LLaVA、SeViLA）在不同数据集上呈现差异化偏置模式，亟需统一的评估与校准框架。

## 核心贡献（创新点）
1. **首次系统分析 VLM-based MCQA 中的选择偏差**，通过 11 种数据集修改方式定位各模型在哪些选项上存在偏差，揭示偏差与视频、问题、答案选项的依赖关系。
2. **将公平性研究中的偏差指标适配到视频 MCQA 场景**，引入 F1_std、Recall_std、JS_std 三个标准差指标来量化各选项间的不公平性。
3. **提出 BOLD 后处理校准方法**，通过分别移除视频、问题、答案选项三种"残缺任务"投影来估计全局先验偏置向量，以减法形式从预测中剥离偏差，无需重新训练模型。
4. **扩展为 Weighted_BOLD**，利用 COBYLA 优化器在五折交叉验证中自动学习各分解维度的权重，更精细地控制去偏方向。

## 方法详解

**核心思想**：将 MCQA 任务 $T$ 视为由视频（$v$）、问题（$q$）、选项（$o$）三个独立组件构成，当任一组件被移除时任务变为"无定义"，此时无偏模型应均匀随机选择任意选项。

**步骤一：分解（Decomposition）**
对每种任务定义三种攻击操作：
- $A_{v=0}$：将视频替换为全黑帧（空帧）
- $A_{q=0}$：将问题替换为空字符串
- $A_{o=0}$：将答案选项替换为空字符串

在无信息场景下，观测到的预测分布 $P_o(d_i|A(T))$ 直接反映先验偏置 $P_p$。

**步骤二：估计先验偏置**
$$\tilde{P}_p(d_i|T) = \text{softmax}\left(\sum_j P_p(d_i|A_j(T))\right)$$
即对三个残缺场景下的偏置分布求和后做 softmax 归一化，得到样本级先验估计。

**步骤三：全局先验聚合**
从数据集 $D$ 中采样 $K = k \cdot |D|$ 个子集进行上述估计，然后对所有样本的先验取平均，得到全局先验 $\tilde{P}_p(d_i)$。

**步骤四：去偏计算**
$$P_d(d_i|T) = \text{softmax}\left(\log P_o(d_i|T) - \log \tilde{P}_p(d_i)\right)$$
即从原始对数概率中减去先验偏置的对数值，再归一化。

**Weighted_BOLD** 进一步引入权重 $w_j$，通过 COBYLA 优化器在 5 折 CV 中最小化 Recall_std，约束条件为 $0 \leq w_i \leq 1$（凸组合）或 $|w_i| \leq 1$（全线性子空间）。

## 实验与结果

**数据集**：NExT-QA（8564 QA，5选项）、NExT-GQA（4962 QA）、STAR（7098 QA，4选项）、Perception Test（7656 QA，3选项）、Video-MME（2700 QA）。

**模型**：Video-LLaMA、Video-LLaVA、SeViLA（均为 zero-shot 推理）。

**关键结果**（$k=0.5$，Weighted_BOLD，正权重）：

| 模型 | 数据集 | Accuracy 提升 | F1_mean 提升 | Recall_std 降低 |
|------|--------|-------------|------------|---------------|
| Video-LLaVA | STAR | +7.94%（37.53%） | +13.69% | -30.15% |
| Video-LLaVA | NExT-QA | +4.39%（52.15%） | +4.49% | -22.02% |
| Video-LLaMA | STAR | +2.24%（37.34%） | +5.16% | -12.3% |
| SeViLA | NExT-QA | +0.04%（63.93%） | +0.04% | -7.64% |

- **最强提升**：Video-LLaVA 在 STAR 上 Accuracy 提升最高达 **7.94%**，同时 Recall_std 下降 **30.15%**，JS_std 下降 **26.24%**。
- **一致性发现**：Weighted_BOLD 普遍优于 BOLD；所有模型的去偏后偏差指标均有改善；SeViLA 本身偏差较小，性能几乎不受影响。
- 最优超参为 $k=0.5$ 且使用正权重（$w_i \geq 0$）。

## 相关工作脉络
1. Zheng et al. (2024a) 提出 PriDe，通过 shuffle 分离 LLM 先验偏差与无偏结果——本文不使用 shuffle，而是将偏差视为三维向量投影。
2. Zhang et al. (2024) 针对图-文 MCQA 通过切断视觉输入去偏——本文扩展到视频-文本，并将偏差分解为多个投影而非单一模态。
3. Wang et al. (2024a) 和 Liu et al. (2023) 通过数据增强（重排+增加选项）来缓解偏差——本文将以增广为目的转为分析工具，仅用3次分解而非大量数据。
4. Pezeshkpour & Hruschka (2023) 基于模型不确定性重新分配概率——本文发现模型置信度与偏差无稳定相关性，该方法不够鲁棒。
5. Zong et al. (2024) 证明简单排列即可"愚弄"多模态模型——本文在此基础上提出可量化、可校准的去偏框架。

## 局限性与未来方向
- 分解方案仅考虑视频/问题/选项三个组件，可能存在更优的潜空间方向未被覆盖。
- 方法假设无信息场景下应为均匀分布（JS散度基准），若实际数据中选项分布本身不均，可能导致低估某些偏差的重要性。
- 目前仅对原始数据集进行了后处理校准评估，未在所有修改配置上系统验证性能提升，需要更细粒度的场景测试。
- 未来可探索不依赖均匀分布假设的去偏方法，以及结合监督信号的进一步优化路径。

## 研究启发与可借鉴点
1. **分解式偏差分析方法**：通过逐个移除任务组件构造"无定义"变体来放大并分离偏差来源，这一思路可迁移至其他多模态任务（如图文、音频-文本）的偏差诊断。
2. **后处理校准的零成本优势**：BOLD 仅需对少量样本（$k=0.5$）做3次推理即可估计全局先验，无需重新训练模型，对工业部署极具吸引力。
3. **多维去偏指标体系**：F1_std、Recall_std、JS_std 可从不同角度量化公平性，比单一准确率更能反映模型的真实性能，值得推广到其他 benchmark 评估。
4. **交叉验证优化权重的设计**：Weighted_BOLD 中用 COBYLA 在五折 CV 上学习分解权重，这种"先分解、后加权聚合"的思路可用于组合多种去偏信号。
5. 与团队现有工作结合机会：可将 BOLD 框架与团队的视频理解/推理任务结合，或在多模态大模型的评测pipeline中集成 BOLD 作为标准校准步骤。

## 关键术语表
- **Selection Bias（选择偏差）**：模型在多选题中因选项位置/形式而非内容偏好而系统性偏向某些选项的现象。
- **BOLD（Bias Optimisation Leveraging Decomposition）**：通过分解任务组件、估计全局先验偏置向量并对预测进行对数减法校准的去偏方法。
- **Weighted_BOLD**：BOLD 的扩展，引入可学习的权重系数，利用 COBYLA 优化器在五折交叉验证中自动学习各分解维度的贡献权重。
- **Ill-defined Task（无定义任务）**：缺少视频、问题或答案选项中任一组件的 MCQA 变体，此时任何答案均无意义，用于暴露模型先验偏差。
- **JS_std（Jensen-Shannon 标准差）**：衡量模型预测分布与真实均匀分布之间不一致性的指标，值越小表示去偏效果越好。
- **Recall_std**：各选项类别上召回率的方差，反映模型对某些选项是否存在系统性偏好。

## 可复现要素
- **数据集**：NExT-QA、NExT-GQA、STAR、Perception Test、Video-MME 均为公开数据集；实验使用各数据集的 test/validation split。
- **代码/权重**：论文使用了 Video-LLaMA-7B、Video-LLaVA-7B、SeViLA 开源模型；算法描述见 Appendix D（Algorithm 1 和 Algorithm 2）。
- **关键超参**：采样比例 $k=0.5$ 为最优；权重约束 $w_i \geq 0$（正权重配置）；COBYLA 优化器；5折交叉验证；选项标签使用 a0/a1/a2 而非 A/B/C 以降低 token bias。
- 论文未提及具体的开源代码仓库链接。
