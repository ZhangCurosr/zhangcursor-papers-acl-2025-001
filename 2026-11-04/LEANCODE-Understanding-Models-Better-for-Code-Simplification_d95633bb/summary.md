---
title: "LEANCODE-Understanding-Models-Better-for-Code-Simplification"
source: https://aclanthology.org/2025.acl-long.78.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-11-05 17:02:58"
field: "代码理解与软件工程AI"
keywords: ["代码简化", "注意力机制", "预训练语言模型", "代码搜索", "代码摘要"]
innovations: ["提出context-aware类别局部注意力平均替代全局平均，降低token重要性评估方差", "首次系统区分并验证CLS注意力和Encoder-Decoder注意力分别适用于分类和序列生成任务的token重要性评估"]
benchmarks: ["CodeSearchNet"]
---

# 论文速读：LEANCODE-Understanding-Models-Better-for-Code-Simplification

## 一句话总结
论文提出 LEANCODE，一种基于模型注意力分数的代码 token 级简化方法，通过区分下游任务专用的 CLS 注意力（分类任务）和 Encoder-Decoder 注意力（序列生成任务），并结合语句类别的上下文感知平均分数，实现更精准的 token 重要性评估与移除，从而降低预训练代码模型的训练与推理成本。

## 研究问题与动机
- **PLLMs 的计算开销问题**：CodeBERT 等模型存在 512 token 长度限制，长输入代码序列带来显著的算力与成本负担，需要在不损失性能的前提下精简输入。
- **DietCode 的局限**：其采用全局平均自注意力分数（cross-input average）衡量 token 重要性，但同一 token 在不同上下文中获得的注意力权重差异很大，全局平均无法反映具体实例中的真实重要性。
- **DietCode 忽视下游任务注意力机制**：分类任务中 CLS token 的自注意力直接决定分类向量，但 DietCode 仅用 encoder 自注意力；序列到序列任务中 encoder-decoder 注意力更能反映输入 token 对目标生成的贡献，同样被 DietCode 丢弃。
- **SLIMCODE 的人为规则局限**：仅 8 个优先级层级导致大量 token 并列，且"人类认为重要"的 token 未必是模型认为重要的，造成模型认知与人工判断的错位。

## 核心贡献（创新点）
- **提出 context-aware 的 token 重要性度量**：根据 token 所在的语句类型（statement category）计算类别内局部注意力平均，而非跨所有输入的全局平均，显著降低注意力方差。
- **首次系统区分并验证 CLS 与 Encoder-Decoder 注意力对下游任务的重要性**：证明对于分类任务应使用 CLS token 的自注意力，对于摘要任务应使用 encoder-decoder 注意力，二者均显著优于单纯使用 encoder 自注意力（DietCode 的方式）。
- **设计高效的 token 级移除算法（Algorithm 1）**：摒弃 DietCode 先删语句再删 token 的两步策略，直接迭代移除当前 snippet 中注意力分数最低的 token，避免误删包含关键信息的高价值语句。
- **全面验证跨模型与跨任务有效性**：在 CodeBERT、CodeT5 及 GPT-4o 上完成代码搜索与摘要任务的评估，并证明 LEANCODE 简化的代码可迁移至其他模型（如用 CodeT5 简化后喂给 GPT-4o）。

## 方法详解
- **问题形式化**：给定代码片段 $d_j = \{t_1, \ldots, t_{n_j}\}$，需移除 $\mathcal{X} = \text{SimplifiedRatio} \times n_j$ 个 token，目标是最小化被移除 token 的加权重要性之和，满足 $\sum x_i = \mathcal{X}$ 的组合优化问题。
- **动态 vs 全局 vs 类别局部注意力**：动态注意力（每个输入单独计算）成本高且不实用；全局平均（DietCode 方式）方差过大；论文提出 category-local attention average：$\mu_t^c = \frac{\sum \sum s_t}{n_t^c}$，即对属于同一语句类别 $c$ 的所有出现位置取平均。
- **分类任务（代码搜索）使用 CLS 注意力**：CLS token 的向量由 encoder 最后一层的自注意力计算得到，其对各输入 token 的注意力分数直接反映该 token 对分类决策的贡献，计算公式为 $s_i = \frac{q_{cls} \cdot k_i}{\sqrt{d}}$。
- **序列到序列任务（代码摘要）使用 Encoder-Decoder 注意力**：对每个生成 token，decoder 会对每个输入 token 计算一次注意力，论文取所有生成步中的最大注意力分数作为该输入 token 的代表性重要性分数，即 $s_i = \frac{q_t \cdot k_i}{\sqrt{d}}$（取 max over t）。
- **移除算法**：初始化数据集副本，对每个 snippet 记录最低分数 token（含 index 与 token 本身），按 SimplifiedRatio 决定的数量逐步移除，直至达到目标移除数。

## 实验与结果
- **数据集**：CodeSearchNet Corpus（Java），训练集约 0.9M（搜索）/ 0.16M（摘要），测试集约 26.9K（搜索）/ 10.9K（摘要）。
- **模型与基线**：CodeBERT、CodeT5、GPT-4o；基线为 DietCode 和 SLIMCODE。
- **评估指标**：代码搜索用 MRR（GPT-4o 用 Precision），代码摘要用 BLEU-4，效率用推理时间。
- **代码搜索**：LEANCODE 相比 DietCode 最高提升 **60.37%**（CodeBERT），相比 SLIMCODE 最高提升 **15.82%**（CodeBERT）；CodeT5 上相对 DietCode 提升 **25.84%**，相对 SLIMCODE 提升 **10.14%**。50% 移除率下 MRR 下降仅 5.48%。
- **代码摘要**：LEANCODE 相比 DietCode 最高提升 **29.36%**（CodeT5），相比 SLIMCODE 最高提升 **27.04%**（CodeT5）；CodeBERT 上相对 DietCode 提升 **14.12%**，相对 SLIMCODE 提升 **6.28%**。50% 移除率下 BLEU 下降仅 11.01%。
- **推理时间**：50% 移除率下 CodeT5 摘要推理时间降低 **40.9%**，CodeBERT 搜索降低约 **36.6%**，呈近线性下降。
- **训练开销**：仅最后一轮 epoch 收集注意力分数，总训练时间增加约 **5%**。
- **剪枝时间**：LEANCODE 剪枝时间为 SlimCode 的 2–4 倍，但仍远低于 DietCode（5–9 小时 vs 分钟级）。
- **跨模型迁移（GPT-4o）**：LEANCODE 简化的代码在 GPT-4o 上输入 token 减少约 28.81%，总 token 减少约 23.68%，代码搜索 Precision 几乎持平（30% 移除率下反而提升 0.49%），摘要 BLEU 表现最优。

## 相关工作脉络
- **DietCode（Zhang et al., 2022）**：基于 encoder 自注意力分数的全局平均进行代码简化；本文指出其未区分下游任务注意力且忽视上下文方差，LEANCODE 通过 CLS/Encoder-Decoder 注意力与类别局部平均克服这些缺陷。
- **SLIMCode（Wang et al., 2024）**：基于人工规则的 8 级优先级移除策略；本文认为模型自身注意力更能反映 token 对下游任务的实际重要性，且 8 级粗粒度在移除率较高时区分度不足。
- **Autofocus（Bui et al., 2019）**：利用 GGNN 的注意力权重评估语句相关性以定位关键代码；本文思路类似但聚焦于 Transformer 架构下的 attention score 利用，且面向 PLLM 的端到端简化。
- **SIVAND / P2IM**：基于 delta debugging 的程序简化方法，通过二分片段搜索最小功能等价子集；本文方法与之上不同，不追求语义等价而追求任务性能保持下的计算效率提升。
- **CodeBERT / CodeT5**：预训练代码模型的基础架构；本文在其之上做简化后处理，不涉及模型结构修改。

## 局限性与未来方向
- **编程语言局限**：仅在 Java 上实验，未验证 Python、Go、JavaScript 等其他语言的效果。
- **任务与模型覆盖有限**：仅评估代码搜索和摘要两个任务，以及 CodeBERT、CodeT5、GPT-4o 三种模型，对更多范式（如 code generation、defect detection）和外层模型的泛化性有待验证。
- **时间测量的外部因素**：推理时间受 GPU 型号、操作系统延迟等外部因素影响，虽在受控环境测试但仍需更多硬件平台的验证。
- **剪枝时间相对较长**：相比 SLIMCODE 的秒级剪枝，LEANCODE 需数分钟至数十分钟，对大规模数据集的实时简化构成一定瓶颈。

## 研究启发与可借鉴点
- **Context-aware 注意力平均策略**：将 token 按所属语句类别分组再计算注意力平均，可有效降低方差并提升重要性评估精度；该方法可迁移至其他基于 attention 的输入重要性分析场景（如文本分类、NLU 任务）。
- **区分预训练注意力与下游任务注意力**：自注意力服务于预训练目标（MLM/RTD），而 CLS/Encoder-Decoder 注意力直接面向下游任务；这一区分对任何依赖 attention 进行特征筛选或可解释性分析的工作均有参考价值。
- **跨模型迁移验证设计**：用模型 A 生成简化代码后喂给模型 B 评估，可有效分离"简化质量"与"模型适配性"两个变量，此实验设计值得借鉴。
- **组合优化视角的形式化**：将代码简化表述为带约束的加权最小化问题，使算法设计具有明确的理论支撑，便于后续改进与理论分析。

## 关键术语表
- **LEANCODE**：论文提出的代码简化方法，基于上下文感知的注意力分数选择性移除 token。
- **CLS Attention**：分类任务中特殊 `[CLS]` token 对所有输入 token 的自注意力分数，反映各 token 对分类决策的贡献。
- **Encoder-Decoder Attention**：序列到序列任务中 decoder 每个生成位置对 encoder 所有输入 token 的注意力分数。
- **Category-Local Attention Average**：按语句类别分组后计算的 token 注意力平均分，用于降低全局平均的高方差问题。
- **SimplifiedRatio**：代码简化比率，表示需要从原始代码中移除的 token 比例。
- **MRR（Mean Reciprocal Rank）**：代码搜索任务的评价指标，衡量正确代码片段在排序列表中的平均倒数排名。
- **BLEU-4**：代码摘要任务的评价指标，基于 4-gram n-gram 精确率衡量生成描述与参考描述的匹配程度。
- **PLLMs（Pre-trained Large Language Models）**：预训练大语言模型，指在大规模代码/文本数据上预训练的 Transformer 架构模型。

## 可复现要素
- **数据集**：CodeSearchNet Corpus，已公开，作者仅使用 Java 子集；数据集划分详见 Appendix Table 9。
- **代码/权重**：论文未提及代码开源；使用 CodeBERT、CodeT5 的官方预训练权重。
- **关键超参**：Adam 优化器，学习率 $1 \times 10^{-5}$（CodeBERT）和 $5 \times 10^{-5}$（CodeT5）；简化率测试 10%–50%；训练 8 epochs。
- **硬件**：2× Intel Xeon Gold 2.40GHz CPU，2× NVIDIA A100 GPU。
